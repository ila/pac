//
// Created by ila on 12/23/25.
//

#include "compiler/pac_compiler_helpers.hpp"
#include "pac_debug.hpp"
#include "core/pac_optimizer.hpp"

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
#include "metadata/pac_compatibility_check.hpp"
#include "parser/pac_parser.hpp"

#include <vector>
#include <duckdb/planner/planner.hpp>

namespace duckdb {

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

	// Get PAC metadata manager for looking up PAC LINKs with referenced_columns
	auto &metadata_mgr = PACMetadataManager::Get();

	vector<JoinCondition> conditions;

	// Try: left has FK referencing right
	for (auto &fk : left_meta.fks) {
		if (fk.first == right_table_name) {
			const auto &left_fk_cols = fk.second;
			vector<string> right_cols = right_meta.pks;

			// If right table has no PKs defined, try to get referenced_columns from PAC LINK
			if (right_cols.empty()) {
				auto *left_pac_metadata = metadata_mgr.GetTableMetadata(left_table_name);
				if (left_pac_metadata) {
					for (auto &link : left_pac_metadata->links) {
						if (StringUtil::Lower(link.referenced_table) == StringUtil::Lower(right_table_name) &&
						    link.local_columns == left_fk_cols && !link.referenced_columns.empty()) {
							right_cols = link.referenced_columns;
							break;
						}
					}
				}
			}

			BuildJoinConditions(left, right.get(), left_fk_cols, right_cols, left_table_name, right_table_name,
			                    conditions);
			break;
		}
	}

	// If no condition yet, try: right has FK referencing left
	if (conditions.empty()) {
		for (auto &fk : right_meta.fks) {
			if (fk.first == left_table_name) {
				const auto &right_fk_cols = fk.second;
				vector<string> left_cols = left_meta.pks;

				// If left table has no PKs defined, try to get referenced_columns from PAC LINK
				if (left_cols.empty()) {
					auto *right_pac_metadata = metadata_mgr.GetTableMetadata(right_table_name);
					if (right_pac_metadata) {
						for (auto &link : right_pac_metadata->links) {
							if (StringUtil::Lower(link.referenced_table) == StringUtil::Lower(left_table_name) &&
							    link.local_columns == right_fk_cols && !link.referenced_columns.empty()) {
								left_cols = link.referenced_columns;
								break;
							}
						}
					}
				}

				BuildJoinConditions(left, right.get(), left_cols, right_fk_cols, left_table_name, right_table_name,
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

// Create a LogicalGet operator for a table by name, projecting only the specified columns
// If required_columns is empty, projects all columns (backward compatible)
unique_ptr<LogicalGet> CreateLogicalGet(ClientContext &context, unique_ptr<LogicalOperator> &plan, const string &table,
                                        idx_t idx, const vector<string> &required_columns) {
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

		// Build a set of required column names for fast lookup (case-insensitive)
		std::unordered_set<string> required_set;
		for (auto &col_name : required_columns) {
			required_set.insert(StringUtil::Lower(col_name));
		}

		// Track which required columns we found
		std::unordered_set<string> found_columns;

		for (auto &col : table_entry.GetColumns().Logical()) {
			string col_name_lower = StringUtil::Lower(col.Name());

			// If required_columns is empty, include all columns (backward compatible)
			// Otherwise, only include columns that are in the required set
			if (required_columns.empty() || required_set.count(col_name_lower) > 0) {
				return_types.push_back(col.Type());
				return_names.push_back(col.Name());
				column_ids.push_back(ColumnIndex(col.Oid()));
				projection_ids.push_back(column_ids.size() - 1);

				if (!required_columns.empty()) {
					found_columns.insert(col_name_lower);
				}
			}
		}

		// If we couldn't find some required columns, fall back to projecting all columns
		// This is a safety net for edge cases like missing schema PKs or PAC LINK metadata
		if (!required_columns.empty() && found_columns.size() < required_set.size()) {
#ifdef DEBUG
			string missing;
			for (auto &col_name : required_columns) {
				if (found_columns.count(StringUtil::Lower(col_name)) == 0) {
					if (!missing.empty()) {
						missing += ", ";
					}
					missing += col_name;
				}
			}
			PAC_DEBUG_PRINT("CreateLogicalGet WARNING: Could not find columns [" + missing + "] in table " + table +
			               ", projecting all columns instead");
#endif

			// Fall back to projecting all columns
			return_types.clear();
			return_names.clear();
			column_ids.clear();
			projection_ids.clear();
			for (auto &col : table_entry.GetColumns().Logical()) {
				return_types.push_back(col.Type());
				return_names.push_back(col.Name());
				column_ids.push_back(ColumnIndex(col.Oid()));
				projection_ids.push_back(column_ids.size() - 1);
			}
		}

		unique_ptr<LogicalGet> get = make_uniq<LogicalGet>(idx, scan_function, std::move(bind_data),
		                                                   std::move(return_types), std::move(return_names));
		get->SetColumnIds(std::move(column_ids));
		get->projection_ids = projection_ids;
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

	// Sort fk_paths keys for deterministic behavior across platforms
	// (unordered_map iteration order is not guaranteed)
	vector<string> sorted_fk_path_keys;
	for (auto &kv : check.fk_paths) {
		sorted_fk_path_keys.push_back(kv.first);
	}
	std::sort(sorted_fk_path_keys.begin(), sorted_fk_path_keys.end());

	// Use the first (sorted) FK path's start table as the start_table_out
	start_table_out = sorted_fk_path_keys[0];

	// Iterate through all FK paths to collect all tables and target PUs
	for (auto &key : sorted_fk_path_keys) {
		auto &fk_path = check.fk_paths.at(key);
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
	// Sort for deterministic behavior across platforms (unordered_set iteration order is not guaranteed)
	std::sort(target_pus_out.begin(), target_pus_out.end());

	// Convert all_tables_in_paths to a sorted vector for deterministic iteration
	vector<string> sorted_tables_in_paths(all_tables_in_paths.begin(), all_tables_in_paths.end());
	std::sort(sorted_tables_in_paths.begin(), sorted_tables_in_paths.end());

	// For each table in all paths, check whether a GET is already present
	for (auto &table_in_path : sorted_tables_in_paths) {
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
