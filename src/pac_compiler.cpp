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
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/common/constants.hpp"

#include <fstream>
#include <cctype>

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

    std::ofstream ofs(filename);
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
    ofs << ")\n";

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
            ColumnBinding(pac_idx, agg->groups.size() + agg->expressions.size()))); // The bottom projection will have the new column at the end
}

static void UpdateProjection(LogicalProjection *proj, idx_t pac_idx) {
    if (!proj) return;
    proj->expressions.push_back(
        make_uniq<BoundColumnRefExpression>(
            LogicalType::BIGINT,
            ColumnBinding(pac_idx, 1)));
}

// Add the sample_id column to all the projections above the aggregate
// We start from the aggregate and go up to the root. The iterative
// `UpdateProjectionsAboveAggregate` below walks up parent projections
// from a given starting child and updates table/column indices at each step.
// (The old recursive implementation was removed because it couldn't update
// the table index properly while climbing.)
static void UpdateProjectionsAboveAggregate(unique_ptr<LogicalOperator> &root, LogicalOperator *start_child, idx_t sample_idx) {
     // Iteratively walk from the start_child up to the root, updating each parent projection.
     // At each step we add a new projection expression that references the sample column from the child
     // using the child's table index and column index. We then update the child_table/col to reflect
     // the parent's output so the next level up will reference the correct indices.
     if (!start_child) return;

     idx_t child_table_idx = sample_idx; // table index where the sample column originates
     idx_t child_col_idx = 1; // column index of sample_id in its originating node (sample_id is 1)
     LogicalOperator *current_child = start_child;

     while (true) {
         // Find the parent projection of the current child
         LogicalProjection *parent_proj = FindParentProjection(root, current_child);
         if (!parent_proj) break;

         // If this parent already exposes the sample column (by binding), skip adding it
         bool found = false;
         for (auto &expr : parent_proj->expressions) {
             if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
                 auto &colref = expr->Cast<BoundColumnRefExpression>();
                 if (colref.binding.table_index == child_table_idx && colref.binding.column_index == child_col_idx) {
                     found = true;
                     break;
                 }
             }
         }
         if (!found) {
             // Add sample_id as a new output expression of the parent projection
             parent_proj->expressions.push_back(
                 make_uniq<BoundColumnRefExpression>(LogicalType::BIGINT, ColumnBinding(child_table_idx, child_col_idx))
             );
         }

         // Update child_table_idx/child_col_idx to refer to the parent's output so that
         // the next parent up will reference the correct indices.
         child_table_idx = parent_proj->table_index;
         child_col_idx = parent_proj->expressions.size() - 1; // index of the newly-added expression

         // Move up one level
         current_child = parent_proj;
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
	parent_proj = FindParentProjection(plan, agg);
	parent_proj_idx = DConstants::INVALID_INDEX;
	// Now we add the sample_id column to the parent projection if it exists
	if (parent_proj) {
		parent_proj_idx = parent_proj->table_index;
		parent_proj->expressions.push_back(
			make_uniq<BoundColumnRefExpression>(
				LogicalType::BIGINT,
				ColumnBinding(agg->group_index, agg->groups.size() - 1)));}

	// 7. Update the topmost projection if it exists
	parent_proj = FindParentProjection(plan, parent_proj);
	// Now we add the sample_id column to the parent projection if it exists
	if (parent_proj) {
		parent_proj->expressions.push_back(
			make_uniq<BoundColumnRefExpression>(
				LogicalType::BIGINT,
				ColumnBinding(parent_proj_idx, parent_proj->expressions.size())));
				parent_proj_idx = parent_proj->table_index;

	}

    plan->Verify(input.context);
    Printer::Print("done");
}

} // namespace duckdb
