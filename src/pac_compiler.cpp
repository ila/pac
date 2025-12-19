//
// Created by ila on 12/12/25.
// Clean PAC compiler implementation
//

#include "include/pac_compiler.hpp"
#include "include/pac_privacy_unit.hpp"
#include "include/pac_helpers.hpp"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_dummy_scan.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/common/constants.hpp"
#include <duckdb/planner/planner.hpp>
#include <duckdb/planner/operator/logical_order.hpp>
#include "duckdb/common/string_util.hpp"
#include <fstream>
#include "duckdb/optimizer/optimizer.hpp"

#include <duckdb/parser/parser.hpp>
#include <include/logical_plan_to_sql.hpp>
#include "include/pac_optimizer.hpp"

namespace duckdb {

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

// Helper to ensure rowid is present in the output columns of a LogicalGet
static void AddRowIDColumn(LogicalGet &get) {
	if (get.virtual_columns.find(COLUMN_IDENTIFIER_ROW_ID) != get.virtual_columns.end()) {
		get.virtual_columns[COLUMN_IDENTIFIER_ROW_ID] = TableColumn("rowid", LogicalTypeId::BIGINT);
	}
	get.AddColumnId(COLUMN_IDENTIFIER_ROW_ID);
	get.projection_ids.push_back(get.GetColumnIds().size() - 1);
	get.returned_types.push_back(LogicalTypeId::BIGINT);
	// We also need to add a column binding for rowid
	get.GenerateColumnBindings(get.table_index, get.GetColumnIds().size());
}

static LogicalAggregate *FindTopAggregate(unique_ptr<LogicalOperator> &op) {
    if (!op) {
        return nullptr;
    }
    if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
        return &op->Cast<LogicalAggregate>();
    }
    for (auto &child : op->children) {
        if (auto *agg = FindTopAggregate(child)) {
            return agg;
        }
    }
    return nullptr;
}

static LogicalProjection *FindTopProjection(unique_ptr<LogicalOperator> &op) {
    if (!op) {
        return nullptr;
    }
    if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
        return &op->Cast<LogicalProjection>();
    }
    for (auto &child : op->children) {
        if (auto *proj = FindTopProjection(child)) {
            return proj;
        }
    }
    return nullptr;
}

// Helper to find the parent LogicalProjection of a given child node
static LogicalProjection *FindParentProjection(unique_ptr<LogicalOperator> &root, LogicalOperator *target_child) {
    if (!root) return nullptr;
    for (auto &child : root->children) {
        if (child.get() == target_child && root->type == LogicalOperatorType::LOGICAL_PROJECTION) {
            return &root->Cast<LogicalProjection>();
        }
        if (auto *proj = FindParentProjection(child, target_child)) {
            return proj;
        }
    }
    return nullptr;
}

// -----------------------------------------------------------------------------
// Sample CTE emission
// -----------------------------------------------------------------------------

void CreateSampleCTE(ClientContext &context,
                     const std::string &privacy_unit,
                     const std::string &filename,
                     const std::string &query_normalized) {
    int64_t m_cfg = 128;
    Value m_val;
    if (context.TryGetCurrentSetting("pac_m", m_val) && !m_val.IsNull()) {
        m_cfg = MaxValue<int64_t>(1, m_val.GetValue<int64_t>());
    }

	// Truncate existing file
	std::ofstream ofs(filename, std::ofstream::trunc);
    if (!ofs) {
        throw ParserException("PAC: failed to write " + filename);
    }

    ofs << "-- PAC compiled sample CTE\n";
    ofs << "-- privacy unit: " << privacy_unit << "\n";
    ofs << "-- normalized query: " << query_normalized << "\n\n";

    ofs << "WITH pac_sample AS (\n";
    ofs << "  SELECT src.rowid, s.sample_id\n";
    ofs << "  FROM " << privacy_unit << " AS src\n";
    ofs << "  CROSS JOIN generate_series(1," << m_cfg << ") AS s(sample_id)\n";
    ofs << "),\n";

    ofs.close();
}

void CreateQueryJoiningSampleCTE(const std::string &lpts,
							  const std::string &output_filename) {
	std::ofstream ofs(output_filename, std::ofstream::app);
	if (!ofs) {
		throw ParserException("PAC: failed to write " + output_filename);
	}

	ofs << "per_sample AS (\n";
	// Strip the last semicolon
	ofs << "\t" << lpts.substr(0, lpts.size() - 1) << "\n";
	ofs << ")\n";
	ofs.close();
}

void CreatePacAggregateQuery(ClientContext &context,
							 const std::string &privacy_unit,
							 const std::string &lpts,
							 const std::string &output_filename) {
	std::ofstream ofs(output_filename, std::ofstream::app);
	if (!ofs) {
		throw ParserException("PAC: failed to write " + output_filename);
	}

	ofs << "SELECT pac_aggregate(array_agg(cnt_sample ORDER BY sample_id), array_agg(cnt_sample_sq ORDER BY sample_id)";
	// todo - mi, k
	ofs << ", 1/128, 3)\n"; // hardcoded m=128, k=3 for now
	ofs << "FROM per sample;\n";
	ofs.close();
}

// -----------------------------------------------------------------------------
// pac_sample LogicalGet
// -----------------------------------------------------------------------------
unique_ptr<LogicalGet>
CreatePacSampleLogicalGet(ClientContext &context,
                          idx_t table_index,
                          const string &privacy_unit) {

    string name = "_pac_internal_sample_" + Sanitize(privacy_unit);

    Catalog &catalog = Catalog::GetCatalog(context, DatabaseManager::GetDefaultDatabase(context));
    CatalogSearchPath path(context);

    for (auto &schema : path.Get()) {
        auto entry = catalog.GetEntry(context, CatalogType::TABLE_ENTRY,
                                      schema.schema, name,
                                      OnEntryNotFound::RETURN_NULL);
        if (!entry) {
            continue;
        }

    	auto &table_entry = entry->Cast<TableCatalogEntry>();
        vector<LogicalType> types = {LogicalType::BIGINT, LogicalType::BIGINT};
    	unique_ptr<FunctionData> bind_data;
    	auto scan_function = table_entry.GetScanFunction(context, bind_data);
    	vector<LogicalType> return_types = {};
    	vector<string> return_names = {};
    	vector<ColumnIndex> column_ids = {};
    	for (auto &col : table_entry.GetColumns().Logical()) {
    		return_types.push_back(col.Type());
    		return_names.push_back(col.Name());
    		column_ids.push_back(ColumnIndex(col.Oid()));
    	}

		unique_ptr<LogicalGet> sample_get = make_uniq<LogicalGet>(table_index, scan_function, std::move(bind_data),
		                                                          std::move(return_types), std::move(return_names));
    	sample_get->SetColumnIds(std::move(column_ids));
    	sample_get->projection_ids = {0, 1}; // project all columns: rowid, sample_id
    	sample_get->ResolveOperatorTypes();
    	sample_get->Verify(context);

        return sample_get;
    }

    throw ParserException("PAC: missing internal sample table " + name);
}

// -----------------------------------------------------------------------------
// Join construction
// -----------------------------------------------------------------------------

unique_ptr<LogicalOperator>
CreatePacSampleJoinNode(ClientContext &context,
                        unique_ptr<LogicalOperator> base,
                        unique_ptr<LogicalGet> pac_get,
                        idx_t base_idx,
                        idx_t pac_idx) {
    // Get output bindings for left and right child
    auto left_bindings = base->GetColumnBindings();
    auto right_bindings = pac_get->GetColumnBindings();
    // The rowid is the last column added for the base, and the first column for the pac_get
	idx_t left_rowid_idx = left_bindings.size() - 1;
	idx_t right_rowid_idx = 0;
    JoinCondition cond;
    cond.comparison = ExpressionType::COMPARE_EQUAL;
    cond.left = make_uniq<BoundColumnRefExpression>(
        LogicalType::BIGINT, left_bindings[left_rowid_idx]);
    cond.right = make_uniq<BoundColumnRefExpression>(
        LogicalType::BIGINT, right_bindings[right_rowid_idx]);
    vector<JoinCondition> conditions;
    conditions.push_back(std::move(cond));
    vector<unique_ptr<Expression>> extra;
    return LogicalComparisonJoin::CreateJoin(
        context, JoinType::INNER, JoinRefType::REGULAR,
        std::move(base), std::move(pac_get),
        std::move(conditions), std::move(extra));
}

// -----------------------------------------------------------------------------
// Plan rewrite
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Aggregate update
// -----------------------------------------------------------------------------

static void UpdateAggregateGroups(LogicalAggregate *agg, idx_t pac_idx) {
    if (!agg) return;
    agg->groups.push_back(
        make_uniq<BoundColumnRefExpression>(
            LogicalType::BIGINT,
            ColumnBinding(5, 1))); // The bottom projection will have the new column at the end
	// We also need to update group_stats
	// create a unique_ptr<BaseStatistics> and push it into agg->group_stats
	auto sample_id_stats_ptr = BaseStatistics::CreateUnknown(LogicalType::BIGINT).ToUnique();
	agg->group_stats.push_back(std::move(sample_id_stats_ptr));
	agg->types.push_back(LogicalType::BIGINT);
}

// Add the sample_id column to all the projections above the aggregate
// For each node, find the parent projection
// If the current node is an aggregate, the index of the new column binding is the agg.group_index and col_idx is agg->groups.size() - 1
// If the current node is a projection, the table index is the current proj->table_index and col_idx is proj->expressions.size()
// After adding the new column binding, go one level up and update the indexes accordingly
static void UpdateNodesAboveAggregate(unique_ptr<LogicalOperator> &root, LogicalOperator *start_node) {
    if (!start_node) {
        return;
    }
    LogicalOperator *search_node = start_node;
    idx_t table_idx, col_idx;

    // Determine initial table_idx and col_idx based on node type
    if (search_node->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
        auto &agg = search_node->Cast<LogicalAggregate>();
        table_idx = agg.group_index;
        col_idx = agg.groups.size() - 1;
    } else if (search_node->type == LogicalOperatorType::LOGICAL_PROJECTION) {
        auto &proj = search_node->Cast<LogicalProjection>();
        table_idx = proj.table_index;
        col_idx = proj.expressions.size();
    } else {
        throw ParserException("UpdateProjectionsAboveAggregate: unsupported node type");
    }

    while (true) {
        // Check for parent projection
        LogicalProjection *parent_proj = FindParentProjection(root, search_node);
        LogicalOperator *next_node = nullptr;
        if (parent_proj) {
            next_node = parent_proj;
        } else {
            // Find parent node in tree
            LogicalOperator *found_parent = nullptr;
            std::function<void(LogicalOperator*, LogicalOperator*)> find_parent = [&](LogicalOperator *node, LogicalOperator *child) {
                for (auto &c : node->children) {
                    if (c.get() == child) {
                        found_parent = node;
                        return;
                    }
                    find_parent(c.get(), child);
                    if (found_parent) return;
                }
            };
            find_parent(root.get(), search_node);
            if (!found_parent) break; // reached the root
            next_node = found_parent;
        }
        search_node = next_node;
        // Check for projection or ORDER BY at every step
        if (search_node->type == LogicalOperatorType::LOGICAL_PROJECTION) {
            auto &proj = search_node->Cast<LogicalProjection>();
            proj.expressions.push_back(
                make_uniq<BoundColumnRefExpression>(LogicalType::BIGINT, ColumnBinding(table_idx, col_idx))
            );
        	proj.types.push_back(LogicalType::BIGINT);
            table_idx = proj.table_index;
            col_idx = proj.expressions.size() - 1;
        } else if (search_node->type == LogicalOperatorType::LOGICAL_ORDER_BY) {
            auto &order_by = search_node->Cast<LogicalOrder>();
        	order_by.projection_map.push_back(col_idx);
        	order_by.types.push_back(LogicalType::BIGINT);
            // Do NOT update table_idx/col_idx after this step
        }
    }
 }


// -----------------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------------

// Refactored CompilePACQuery: does all steps in one place
void CompilePACQuery(OptimizerExtensionInput &input,
                     unique_ptr<LogicalOperator> &plan,
                     const std::string &privacy_unit) {

    if (privacy_unit.empty()) return;
    // 1. Create sample CTE file (unchanged)
    string normalized = NormalizeQueryForHash(input.context.GetCurrentQuery());
    string hash = HashStringToHex(normalized);
    string path = ".";
    Value v;
    if (input.context.TryGetCurrentSetting("pac_compiled_path", v) && !v.IsNull()) {
        path = v.ToString();
    }
    if (!path.empty() && path.back() != '/') path.push_back('/');
    string filename = path + privacy_unit + "_" + hash + ".sql";
	CreateSampleCTE(input.context, privacy_unit, filename, normalized);

	// Replan the plan without compressed materialization.
	// We change the "disabled_optimizers" setting temporarily and re-run the optimizer. Use RAII
	// to ensure the original setting is always restored.
	{
		auto &dbconf = DBConfig::GetConfig(input.context);
		// If the optimizer extension provided a PACOptimizerInfo, use it to prevent re-entrant replans
		PACOptimizerInfo *pac_info = nullptr;
		if (input.info) {
			pac_info = dynamic_cast<PACOptimizerInfo *>(input.info.get());
		}
		// If a replan is already in progress by this extension, skip re-optimizing to avoid recursion
		if (pac_info && pac_info->replan_in_progress.load(std::memory_order_acquire)) {
			// skip replan to avoid recursion
		} else {
			// set the in-progress flag in RAII manner if pac_info exists
			struct ReplanGuard {
				PACOptimizerInfo *info;
				ReplanGuard(PACOptimizerInfo *i) : info(i) {
					if (info) info->replan_in_progress.store(true, std::memory_order_release);
				}
				~ReplanGuard() {
					if (info) info->replan_in_progress.store(false, std::memory_order_release);
				}
			};
			ReplanGuard rg(pac_info);
			Connection con(*input.context.db);

			con.BeginTransaction();
			// todo: maybe we want to disable more optimizers (internal_optimizer_types)
			// If column_lifetime is enabled, then the existence of duplicate table indices etc etc is verified.
			// However, it massively complicates the query tree which is not nice for further usage in OpenIVM.
			// Therefore, it should be run for verification purposes once in a while (otherwise, turn it off).

			con.Query("SET disabled_optimizers='compressed_materialization, column_lifetime, statistics_propagation, "
						  "expression_rewriter, filter_pushdown';");

			con.Commit();

			Parser parser;
			Planner planner(input.context);

			parser.ParseQuery(normalized);
			auto statement = parser.statements[0].get();

			planner.CreatePlan(statement->Copy());

			Optimizer optimizer(*planner.binder, input.context);
			plan = optimizer.Optimize(std::move(planner.plan));
			plan->Print();
		}
	}

    // 2. Create LogicalGet for sample table
    idx_t sample_idx = GetNextTableIndex(plan);
    auto sample_get = CreatePacSampleLogicalGet(input.context, sample_idx, privacy_unit);
    // 3. Find the scan (LogicalGet) of the PAC table in the plan
    unique_ptr<LogicalOperator> *pac_scan_ptr = nullptr;
    idx_t pac_table_idx = DConstants::INVALID_INDEX;
    {
        vector<unique_ptr<LogicalOperator> *> stack;
        stack.push_back(&plan);
        while (!stack.empty()) {
            auto cur_ptr = stack.back();
            stack.pop_back();
            auto &cur = *cur_ptr;
            if (!cur) continue;
            if (cur->type == LogicalOperatorType::LOGICAL_GET) {
                pac_scan_ptr = cur_ptr;
                pac_table_idx = cur->Cast<LogicalGet>().table_index;
                break;
            }
            for (auto &c : cur->children) {
                stack.push_back(&c);
            }
        }
    }
    if (!pac_scan_ptr || pac_table_idx == DConstants::INVALID_INDEX) {
        // Could not find PAC scan
        return;
    }
    // Ensure rowid is present in both scans before join construction
    AddRowIDColumn(pac_scan_ptr->get()->Cast<LogicalGet>()); // PAC scan
    // The sample get already has rowid added in its creation function

    // 4. Create the join node using a copy of the PAC scan and the sample table scan
    auto pac_scan = (*pac_scan_ptr)->Copy(input.context);
    auto join = CreatePacSampleJoinNode(input.context, std::move(pac_scan), std::move(sample_get), pac_table_idx, sample_idx);
    LogicalOperator *join_ptr = join.get();
    ReplaceNode(plan, *pac_scan_ptr, join);

    // Find the parent projection of the join we just inserted
    LogicalProjection *parent_proj = FindParentProjection(plan, join_ptr);
	auto parent_proj_idx = DConstants::INVALID_INDEX;
    // Now we add the sample_id column to the parent projection if it exists
	if (parent_proj) {
		parent_proj->expressions.push_back(
			make_uniq<BoundColumnRefExpression>(
				LogicalType::BIGINT,
				ColumnBinding(sample_idx, 1)));
		parent_proj_idx = parent_proj->table_index;
	}

    // 5. Update the aggregate node to include the sample column
    auto *agg = FindTopAggregate(plan);
    UpdateAggregateGroups(agg, parent_proj_idx);

    // 6. Update the projection node to include the sample column
	UpdateNodesAboveAggregate(plan, agg);

	plan->ResolveOperatorTypes();
    plan->Verify(input.context);
	plan->Print();
    Printer::Print("LPTS output plan after PAC compilation:\n");
	auto lp_to_sql = LogicalPlanToSql(input.context, plan);
	auto ir = lp_to_sql.LogicalPlanToIR();
	Printer::Print(ir->ToQuery(true));
	CreateQueryJoiningSampleCTE(ir->ToQuery(true), filename);
	CreatePacAggregateQuery(input.context, privacy_unit, ir->ToQuery(true), filename);

	// replace the plan with a dummy plan for now
	plan = make_uniq_base<LogicalOperator, LogicalDummyScan>(0);

	// todo:
	// clean up this code
	// test with/without compressed materialization
	// feed the query to LPTS
	// implement the last query
	// implement filter pushdown
	// implement pac_sum final step
	// test everything
 }

} // namespace duckdb
