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
                            vector<string> &gets_missing, string &start_table_out, string &target_pu_out) {
	// Expect at least one FK path when this is called
	if (check.fk_paths.empty()) {
		throw InternalException("PAC compiler: no fk_paths available");
	}
	auto it = check.fk_paths.begin();
	start_table_out = it->first; // scanned table name
	auto fk_path = it->second;   // vector from start -> ... -> privacy_unit
	if (fk_path.empty()) {
		throw InternalException("PAC compiler: FK path is empty");
	}
	// canonical target privacy unit (last element)
	target_pu_out = fk_path.back();
	// For each table in the FK path, check whether a GET is already present
	for (auto &table_in_path : fk_path) {
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

} // namespace duckdb
