//
// Created by ila on 12/23/25.
//

#include "include/pac_compiler_helpers.hpp"

#include "duckdb/main/connection.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/common/constants.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/enums/optimizer_type.hpp"
#include "include/pac_compatibility_check.hpp"

#include <vector>
#include <duckdb/planner/planner.hpp>

namespace duckdb {

void ReplanWithoutOptimizers(ClientContext &context, const string &query, unique_ptr<LogicalOperator> &plan) {
	auto &config = DBConfig::GetConfig(context);

	// Save the original disabled optimizers
	auto original_disabled = config.options.disabled_optimizers;

	// Add optimizers to disable
	config.options.disabled_optimizers.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
	config.options.disabled_optimizers.insert(OptimizerType::COLUMN_LIFETIME);
	config.options.disabled_optimizers.insert(OptimizerType::STATISTICS_PROPAGATION);
	config.options.disabled_optimizers.insert(OptimizerType::EXPRESSION_REWRITER);
	config.options.disabled_optimizers.insert(OptimizerType::FILTER_PUSHDOWN);

	Parser parser;
	Planner planner(context);

	parser.ParseQuery(query);
	if (parser.statements.empty()) {
		// Restore original disabled optimizers before returning
		config.options.disabled_optimizers = original_disabled;
		return;
	}
	auto statement = parser.statements[0].get();
	planner.CreatePlan(statement->Copy());

	Optimizer optimizer(*planner.binder, context);
	plan = optimizer.Optimize(std::move(planner.plan));

	// Restore original disabled optimizers
	config.options.disabled_optimizers = original_disabled;
}

// Build join conditions from FK columns to PK columns
void BuildJoinConditions(LogicalGet *left_get, LogicalGet *right_get, const vector<string> &left_cols,
                         const vector<string> &right_cols, const string &left_table_name,
                         const string &right_table_name, vector<JoinCondition> &conditions) {
	if (left_cols.size() != right_cols.size() || left_cols.empty()) {
		throw InvalidInputException("PAC compiler: FK/PK column count mismatch for " + left_table_name + " -> " +
		                            right_table_name);
	}

	for (size_t i = 0; i < left_cols.size(); ++i) {
		idx_t lproj = EnsureProjectedColumn(*left_get, left_cols[i]);
		idx_t rproj = EnsureProjectedColumn(*right_get, right_cols[i]);
		if (lproj == DConstants::INVALID_INDEX || rproj == DConstants::INVALID_INDEX) {
			throw InternalException("PAC compiler: failed to project FK/PK columns for join");
		}
		JoinCondition cond;
		cond.comparison = ExpressionType::COMPARE_EQUAL;
		auto left_col_index = left_get->GetColumnIds()[lproj];
		auto right_col_index = right_get->GetColumnIds()[rproj];
		auto &left_type = left_get->GetColumnType(left_col_index);
		auto &right_type = right_get->GetColumnType(right_col_index);
		cond.left = make_uniq<BoundColumnRefExpression>(left_type, ColumnBinding(left_get->table_index, lproj));
		cond.right = make_uniq<BoundColumnRefExpression>(right_type, ColumnBinding(right_get->table_index, rproj));
		conditions.push_back(std::move(cond));
	}
}

// Create a logical join operator based on FK relationships in the compatibility check metadata
unique_ptr<LogicalOperator> CreateLogicalJoin(const PACCompatibilityResult &check, ClientContext &context,
                                              unique_ptr<LogicalOperator> left_operator, unique_ptr<LogicalGet> right) {
	// Simpler join builder: use precomputed metadata only. Find FK(s) on one side that reference
	// the other's table; pair FK columns to PK columns and produce equality JoinCondition(s).

	// The left logical operator can be a GET or another JOIN
	LogicalGet *left = nullptr;
	if (left_operator->type == LogicalOperatorType::LOGICAL_GET) {
		left = &left_operator->Cast<LogicalGet>();
	} else if (left_operator->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		// We extract the table from the right side of the join
		if (left_operator->children.size() != 2 ||
		    left_operator->children[1]->type != LogicalOperatorType::LOGICAL_GET) {
			throw InternalException("PAC compiler: expected right child of left join to be LogicalGet");
		}
		left = &left_operator->children[1]->Cast<LogicalGet>();
	} else {
		throw InternalException("PAC compiler: expected left node to be LogicalGet or LogicalComparisonJoin");
	}

	auto left_table_ptr = left->GetTable();
	auto right_table_ptr = right->GetTable();
	if (!left_table_ptr || !right_table_ptr) {
		throw InternalException("PAC compiler: expected both LogicalGet nodes to be bound to tables for join creation");
	}
	string left_table_name = left_table_ptr->name;
	string right_table_name = right_table_ptr->name;

	// Require metadata to be present; compatibility check is responsible for populating it.
	auto lit = check.table_metadata.find(left_table_name);
	auto rit = check.table_metadata.find(right_table_name);
	if (lit == check.table_metadata.end() || rit == check.table_metadata.end()) {
		throw InternalException("PAC compiler: missing table metadata for join: " + left_table_name + " <-> " +
		                        right_table_name);
	}
	const auto &left_meta = lit->second;
	const auto &right_meta = rit->second;

	vector<JoinCondition> conditions;

	// Try: left has FK referencing right
	for (auto &fk : left_meta.fks) {
		if (fk.first == right_table_name) {
			const auto &left_fk_cols = fk.second;
			const auto &right_pks = right_meta.pks;
			BuildJoinConditions(left, right.get(), left_fk_cols, right_pks, left_table_name, right_table_name,
			                    conditions);
			break;
		}
	}

	// If no condition yet, try: right has FK referencing left
	if (conditions.empty()) {
		for (auto &fk : right_meta.fks) {
			if (fk.first == left_table_name) {
				const auto &right_fk_cols = fk.second;
				const auto &left_pks = left_meta.pks;
				BuildJoinConditions(left, right.get(), left_pks, right_fk_cols, left_table_name, right_table_name,
				                    conditions);
				break;
			}
		}
	}

	if (conditions.empty()) {
		throw InvalidInputException("PAC compiler: expected FK link between " + left_table_name + " and " +
		                            right_table_name);
	}

	vector<unique_ptr<Expression>> extra;
	return LogicalComparisonJoin::CreateJoin(context, JoinType::INNER, JoinRefType::REGULAR, std::move(left_operator),
	                                         std::move(right), std::move(conditions), std::move(extra));
}

// Create a LogicalGet operator for a table by name
unique_ptr<LogicalGet> CreateLogicalGet(ClientContext &context, unique_ptr<LogicalOperator> &plan, const string &table,
                                        idx_t idx) {
	Catalog &catalog = Catalog::GetCatalog(context, DatabaseManager::GetDefaultDatabase(context));
	CatalogSearchPath path(context);

	for (auto &schema : path.Get()) {
		auto entry =
		    catalog.GetEntry(context, CatalogType::TABLE_ENTRY, schema.schema, table, OnEntryNotFound::RETURN_NULL);
		if (!entry) {
			continue;
		}

		auto &table_entry = entry->Cast<TableCatalogEntry>();
		vector<LogicalType> types = table_entry.GetTypes();
		unique_ptr<FunctionData> bind_data;
		auto scan_function = table_entry.GetScanFunction(context, bind_data);
		vector<LogicalType> return_types = {};
		vector<string> return_names = {};
		vector<ColumnIndex> column_ids = {};
		vector<idx_t> projection_ids = {};
		for (auto &col : table_entry.GetColumns().Logical()) {
			return_types.push_back(col.Type());
			return_names.push_back(col.Name());
			column_ids.push_back(ColumnIndex(col.Oid()));
			projection_ids.push_back(column_ids.size() - 1);
		}

		unique_ptr<LogicalGet> get = make_uniq<LogicalGet>(idx, scan_function, std::move(bind_data),
		                                                   std::move(return_types), std::move(return_names));
		get->SetColumnIds(std::move(column_ids));
		get->projection_ids = projection_ids; // we project everything
		get->ResolveOperatorTypes();
		get->Verify(context);

		return get;
	}

	throw ParserException("PAC: missing internal sample table " + table);
}

// Examine PACCompatibilityResult.fk_paths and populate gets_present / gets_missing
void PopulateGetsFromFKPath(const PACCompatibilityResult &check, vector<string> &gets_present,
                            vector<string> &gets_missing, string &start_table_out, vector<string> &target_pus_out) {
	// Expect at least one FK path when this is called
	if (check.fk_paths.empty()) {
		throw InternalException("PAC compiler: no fk_paths available");
	}

	// Collect all target PUs from all FK paths
	std::unordered_set<string> unique_target_pus;
	std::unordered_set<string> all_tables_in_paths;

	// Use the first FK path's start table as the start_table_out
	auto first_it = check.fk_paths.begin();
	start_table_out = first_it->first;

	// Iterate through all FK paths to collect all tables and target PUs
	for (auto &kv : check.fk_paths) {
		auto &fk_path = kv.second;
		if (fk_path.empty()) {
			continue;
		}

		// Last element in each path is a target PU
		unique_target_pus.insert(fk_path.back());

		// Collect all tables in the path
		for (auto &table_in_path : fk_path) {
			all_tables_in_paths.insert(table_in_path);
		}
	}

	// Convert target PUs set to vector
	target_pus_out.assign(unique_target_pus.begin(), unique_target_pus.end());

	// For each table in all paths, check whether a GET is already present
	for (auto &table_in_path : all_tables_in_paths) {
		bool found_get = false;
		for (auto &t : check.scanned_pu_tables) {
			if (t == table_in_path) {
				gets_present.push_back(t);
				found_get = true;
				break;
			}
		}
		if (!found_get) {
			for (auto &t : check.scanned_non_pu_tables) {
				if (t == table_in_path) {
					gets_present.push_back(t);
					found_get = true;
					break;
				}
			}
		}
		if (!found_get) {
			gets_missing.push_back(table_in_path);
		}
	}
}

// Add a column to a DELIM_JOIN's duplicate_eliminated_columns and update corresponding DELIM_GETs
// Returns the ColumnBinding and LogicalType for accessing the column via DELIM_GET
DelimColumnResult AddColumnToDelimJoin(unique_ptr<LogicalOperator> &plan, LogicalGet &source_get,
                                       const string &column_name, LogicalAggregate *target_agg) {
	// Find the DELIM_JOIN that is an ancestor of the target aggregate
	// The DELIM_JOIN connects the outer query (containing source_get) to the subquery (containing target_agg)

	// First, find the DELIM_JOIN by walking up from the root
	LogicalComparisonJoin *delim_join = nullptr;
	std::function<bool(LogicalOperator *)> find_delim_join = [&](LogicalOperator *op) -> bool {
		if (!op)
			return false;

		if (op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			// Check if this DELIM_JOIN has the target aggregate in its subtree
			// and the source_get in the other subtree
			auto &join = op->Cast<LogicalComparisonJoin>();

			// Check if target_agg is in children[1] (the subquery side)
			bool agg_in_right = false;
			std::function<bool(LogicalOperator *)> find_agg = [&](LogicalOperator *child) -> bool {
				if (child == target_agg)
					return true;
				for (auto &c : child->children) {
					if (find_agg(c.get()))
						return true;
				}
				return false;
			};

			if (join.children.size() >= 2) {
				agg_in_right = find_agg(join.children[1].get());
			}

			// Check if source_get is in children[0] (the outer query side)
			bool source_in_left = false;
			std::function<bool(LogicalOperator *)> find_source = [&](LogicalOperator *child) -> bool {
				if (child->type == LogicalOperatorType::LOGICAL_GET) {
					auto &get = child->Cast<LogicalGet>();
					if (get.table_index == source_get.table_index)
						return true;
				}
				for (auto &c : child->children) {
					if (find_source(c.get()))
						return true;
				}
				return false;
			};

			if (!join.children.empty()) {
				source_in_left = find_source(join.children[0].get());
			}

			if (agg_in_right && source_in_left) {
				delim_join = &join;
				return true;
			}
		}

		for (auto &child : op->children) {
			if (find_delim_join(child.get()))
				return true;
		}
		return false;
	};

	find_delim_join(plan.get());

	DelimColumnResult invalid_result;
	invalid_result.binding = ColumnBinding(DConstants::INVALID_INDEX, DConstants::INVALID_INDEX);
	invalid_result.type = LogicalType::INVALID;

	if (!delim_join) {
		// No DELIM_JOIN found - return invalid result
		return invalid_result;
	}

	// Ensure the column is projected in source_get
	idx_t col_proj_idx = EnsureProjectedColumn(source_get, column_name);
	if (col_proj_idx == DConstants::INVALID_INDEX) {
		return invalid_result;
	}

	// Get the column type
	auto col_index = source_get.GetColumnIds()[col_proj_idx];
	auto col_type = source_get.GetColumnType(col_index);

	// Create a column reference expression for the source column
	auto source_binding = ColumnBinding(source_get.table_index, col_proj_idx);
	auto col_ref = make_uniq<BoundColumnRefExpression>(col_type, source_binding);

	// Add to DELIM_JOIN's duplicate_eliminated_columns
	idx_t new_col_idx = delim_join->duplicate_eliminated_columns.size();
	delim_join->duplicate_eliminated_columns.push_back(std::move(col_ref));

	// Find and update all DELIM_GETs in the subquery that reference this DELIM_JOIN
	// We need to add the new column type to their chunk_types
	std::function<void(LogicalOperator *)> update_delim_gets = [&](LogicalOperator *op) {
		if (!op)
			return;

		if (op->type == LogicalOperatorType::LOGICAL_DELIM_GET) {
			auto &delim_get = op->Cast<LogicalDelimGet>();
			// Add the new column type
			delim_get.chunk_types.push_back(col_type);
		}

		for (auto &child : op->children) {
			update_delim_gets(child.get());
		}
	};

	// Only update DELIM_GETs in the subquery side (children[1])
	if (delim_join->children.size() >= 2) {
		update_delim_gets(delim_join->children[1].get());
	}

	// Find the DELIM_GET that the aggregate can access and return the binding for the new column
	// Walk from aggregate to find the closest DELIM_GET
	std::function<LogicalDelimGet *(LogicalOperator *)> find_delim_get = [&](LogicalOperator *op) -> LogicalDelimGet * {
		if (!op)
			return nullptr;

		if (op->type == LogicalOperatorType::LOGICAL_DELIM_GET) {
			return &op->Cast<LogicalDelimGet>();
		}

		for (auto &child : op->children) {
			auto result = find_delim_get(child.get());
			if (result)
				return result;
		}
		return nullptr;
	};

	auto *delim_get = find_delim_get(target_agg);
	if (!delim_get) {
		return invalid_result;
	}

	// Return binding and type for the new column in the DELIM_GET
	DelimColumnResult result;
	result.binding = ColumnBinding(delim_get->table_index, new_col_idx);
	result.type = col_type;
	return result;
}

} // namespace duckdb
