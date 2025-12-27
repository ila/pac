//
// Created by ila on 12/21/25.
//

#include "include/pac_bitslice_compiler.hpp"
#include "include/pac_helpers.hpp"
#include "include/pac_compatibility_check.hpp"
#include "include/pac_compiler_helpers.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/parser/expression/function_expression.hpp>
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog.hpp"

namespace duckdb {

// Helper: ensure PK columns are present in a LogicalGet's column_ids and projection_ids.
// If a PK column is not present in the scan, this function adds its column id to the LogicalGet
// and ensures it is projected and that column bindings are regenerated.
void AddPKColumns(LogicalGet &get, const std::vector<std::string> &pks) {
	// We'll look up the table to map column names to column indexes
	auto table_entry_ptr = get.GetTable();
	if (!table_entry_ptr) {
		throw InternalException("PAC compiler: expected LogicalGet to be bound to a table when PKs are present");
	}
	auto &table_entry = *table_entry_ptr;

	// For each PK name, ensure it is present in the LogicalGet's column_ids and projection
	for (auto &pk : pks) {
		bool found = false;
		// Check if any of the returned names equals PK (case-sensitive as stored)
		for (idx_t i = 0; i < get.names.size(); i++) {
			if (get.names[i] == pk) {
				// Ensure this column is part of projection_ids (so it's produced by the scan)
				// Find the corresponding ColumnIndex in get.GetColumnIds() - match by primary index
				for (idx_t cid = 0; cid < get.GetColumnIds().size(); cid++) {
					auto col_index = get.GetColumnIds()[cid];
					if (!col_index.IsVirtualColumn() && col_index.GetPrimaryIndex() == i) {
						// mark projection if not already
						if (std::find(get.projection_ids.begin(), get.projection_ids.end(), cid) ==
						    get.projection_ids.end()) {
							get.projection_ids.push_back(cid);
						}
						found = true;
						break;
					}
				}
				if (found) {
					break;
				}
			}
		}
		if (!found) {
			// Need to add the column id for this PK: find its logical index in the table
			idx_t logical_idx = DConstants::INVALID_INDEX;
			{
				idx_t ti = 0;
				for (auto &col : table_entry.GetColumns().Logical()) {
					if (col.Name() == pk) {
						logical_idx = ti;
						break;
					}
					ti++;
				}
			}
			if (logical_idx == DConstants::INVALID_INDEX) {
				throw InternalException("PAC compiler: could not find PK column " + pk + " in table");
			}
			// Add the column id and project it
			get.AddColumnId(logical_idx);
			get.projection_ids.push_back(get.GetColumnIds().size() - 1);
			// Ensure returned_types/names are OK - LogicalGet stores full returned_types/names already
			// Generate new column bindings for the added column count
			get.GenerateColumnBindings(get.table_index, get.GetColumnIds().size());
		}
	}
}

unique_ptr<LogicalOperator> CreateLogicalJoin(const PACCompatibilityResult &check, ClientContext &context,
                                              unique_ptr<LogicalGet> left, unique_ptr<LogicalGet> right) {
	// Simpler join builder: use precomputed metadata only. Find FK(s) on one side that reference
	// the other's table; pair FK columns to PK columns and produce equality JoinCondition(s).
	auto left_table_ptr = left->GetTable();
	auto right_table_ptr = right->GetTable();
	if (!left_table_ptr || !right_table_ptr) {
		throw InternalException("PAC compiler: expected both LogicalGet nodes to be bound to tables for join creation");
	}
	std::string left_table_name = left_table_ptr->name;
	std::string right_table_name = right_table_ptr->name;

	// Require metadata to be present; compatibility check is responsible for populating it.
	auto lit = check.table_metadata.find(left_table_name);
	auto rit = check.table_metadata.find(right_table_name);
	if (lit == check.table_metadata.end() || rit == check.table_metadata.end()) {
		throw InternalException("PAC compiler: missing table metadata for join: " + left_table_name + " <-> " +
		                        right_table_name);
	}
	const auto &left_meta = lit->second;
	const auto &right_meta = rit->second;

	// Helper: ensure a column is projected in a LogicalGet and return its projection index
	auto ensure_proj_idx = [&](LogicalGet *g, const std::string &col_name) -> idx_t {
		// try existing projected columns by matching returned names via ColumnIndex primary
		for (idx_t cid = 0; cid < g->GetColumnIds().size(); ++cid) {
			auto col_idx = g->GetColumnIds()[cid];
			if (!col_idx.IsVirtualColumn()) {
				idx_t primary = col_idx.GetPrimaryIndex();
				if (primary < g->names.size() && g->names[primary] == col_name) {
					return cid;
				}
			}
		}
		// otherwise add column from table schema
		auto table_entry = g->GetTable();
		if (!table_entry) {
			return DConstants::INVALID_INDEX;
		}
		idx_t logical_idx = DConstants::INVALID_INDEX;
		idx_t ti = 0;
		for (auto &col : table_entry->GetColumns().Logical()) {
			if (col.Name() == col_name) {
				logical_idx = ti;
				break;
			}
			ti++;
		}
		if (logical_idx == DConstants::INVALID_INDEX) {
			return DConstants::INVALID_INDEX;
		}
		g->AddColumnId(logical_idx);
		g->projection_ids.push_back(g->GetColumnIds().size() - 1);
		g->GenerateColumnBindings(g->table_index, g->GetColumnIds().size());
		return g->GetColumnIds().size() - 1;
	};

	vector<JoinCondition> conditions;

	// Try: left has FK referencing right
	for (auto &fk : left_meta.fks) {
		if (fk.first == right_table_name) {
			// fk.second: FK column names on left
			const auto &left_fk_cols = fk.second;
			const auto &right_pks = right_meta.pks;
			if (right_pks.size() != left_fk_cols.size() || right_pks.empty()) {
				throw InvalidInputException("PAC compiler: FK/PK column count mismatch for " + left_table_name +
				                            " -> " + right_table_name);
			}
			for (size_t i = 0; i < left_fk_cols.size(); ++i) {
				idx_t lproj = ensure_proj_idx(left.get(), left_fk_cols[i]);
				idx_t rproj = ensure_proj_idx(right.get(), right_pks[i]);
				if (lproj == DConstants::INVALID_INDEX || rproj == DConstants::INVALID_INDEX) {
					throw InternalException("PAC compiler: failed to project FK/PK columns for join");
				}
				JoinCondition cond;
				cond.comparison = ExpressionType::COMPARE_EQUAL;
				auto left_col_index = left->GetColumnIds()[lproj];
				auto right_col_index = right->GetColumnIds()[rproj];
				auto &left_type = left->GetColumnType(left_col_index);
				auto &right_type = right->GetColumnType(right_col_index);
				cond.left = make_uniq<BoundColumnRefExpression>(left_type, ColumnBinding(left->table_index, lproj));
				cond.right = make_uniq<BoundColumnRefExpression>(right_type, ColumnBinding(right->table_index, rproj));
				conditions.push_back(std::move(cond));
			}
			break;
		}
	}

	// If no condition yet, try: right has FK referencing left
	if (conditions.empty()) {
		for (auto &fk : right_meta.fks) {
			if (fk.first == left_table_name) {
				const auto &right_fk_cols = fk.second;
				const auto &left_pks = left_meta.pks;
				if (left_pks.size() != right_fk_cols.size() || left_pks.empty()) {
					throw InvalidInputException("PAC compiler: FK/PK column count mismatch for " + right_table_name +
					                            " -> " + left_table_name);
				}
				for (size_t i = 0; i < right_fk_cols.size(); ++i) {
					// pair left PK with right FK
					idx_t lproj = ensure_proj_idx(left.get(), left_pks[i]);
					idx_t rproj = ensure_proj_idx(right.get(), right_fk_cols[i]);
					if (lproj == DConstants::INVALID_INDEX || rproj == DConstants::INVALID_INDEX) {
						throw InternalException("PAC compiler: failed to project FK/PK columns for join");
					}
					JoinCondition cond;
					cond.comparison = ExpressionType::COMPARE_EQUAL;
					auto left_col_index = left->GetColumnIds()[lproj];
					auto right_col_index = right->GetColumnIds()[rproj];
					auto &left_type = left->GetColumnType(left_col_index);
					auto &right_type = right->GetColumnType(right_col_index);
					cond.left = make_uniq<BoundColumnRefExpression>(left_type, ColumnBinding(left->table_index, lproj));
					cond.right =
					    make_uniq<BoundColumnRefExpression>(right_type, ColumnBinding(right->table_index, rproj));
					conditions.push_back(std::move(cond));
				}
				break;
			}
		}
	}

	if (conditions.empty()) {
		throw InvalidInputException("PAC compiler: expected FK link between " + left_table_name + " and " +
		                            right_table_name);
	}

	vector<unique_ptr<Expression>> extra;
	return LogicalComparisonJoin::CreateJoin(context, JoinType::INNER, JoinRefType::REGULAR, std::move(left),
	                                         std::move(right), std::move(conditions), std::move(extra));
}

unique_ptr<LogicalGet> CreateLogicalGet(ClientContext &context, unique_ptr<LogicalOperator> &plan, const string &table,
                                        std::vector<string> &pks) {

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

		auto table_index = GetNextTableIndex(plan);
		unique_ptr<LogicalGet> get = make_uniq<LogicalGet>(table_index, scan_function, std::move(bind_data),
		                                                   std::move(return_types), std::move(return_names));
		get->SetColumnIds(std::move(column_ids));
		get->projection_ids = projection_ids; // we project everything
		get->ResolveOperatorTypes();
		get->Verify(context);

		return get;
	}

	throw ParserException("PAC: missing internal sample table " + table);
}

// Helper: examine PACCompatibilityResult.fk_paths and populate gets_present / gets_missing.
// Also returns the start table and canonical target privacy unit via reference output parameters.
void PopulateGetsFromFKPath(const PACCompatibilityResult &check, std::vector<std::string> &gets_present,
                            std::vector<std::string> &gets_missing, std::string &start_table_out,
                            std::string &target_pu_out) {
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

void ModifyPlanWithoutPU(const PACCompatibilityResult &check, OptimizerExtensionInput &input,
                         unique_ptr<LogicalOperator> &plan, const std::vector<std::string> &gets_missing,
                         const std::vector<std::string> &gets_present, const std::vector<std::string> &fk_path,
                         const std::string &privacy_unit) {
	// Note: we assume we don't use rowid

	// Create the necessary LogicalGets for missing tables
	std::unordered_map<std::string, unique_ptr<LogicalGet>> get_map;
	for (auto &table : gets_missing) {
		auto it = check.table_metadata.find(table);
		if (it == check.table_metadata.end()) {
			throw InternalException("PAC compiler: missing table metadata for missing GET: " + table);
		}
		std::vector<std::string> pks = it->second.pks;
		auto get = CreateLogicalGet(input.context, plan, table, pks);
		get_map[table] = std::move(get);
	}

	// If there are no missing gets, nothing to do
	if (gets_missing.empty()) {
		return;
	}

	// For simplicity (common case) we only support joining a single missing GET to an existing GET
	if (gets_missing.size() > 1) {
		throw NotImplementedException(
		    "PAC compiler: ModifyPlanWithoutPU currently supports only a single missing GET (found " +
		    to_string(gets_missing.size()) + ")");
	}

	// Find the unique_ptr reference to the existing LogicalGet for gets_present[0]
	std::function<unique_ptr<LogicalOperator> *(unique_ptr<LogicalOperator> *, const std::string &)> find_node_ref;
	find_node_ref = [&](unique_ptr<LogicalOperator> *node_ref,
	                    const std::string &tbl) -> unique_ptr<LogicalOperator> * {
		if (!node_ref || !node_ref->get()) {
			return nullptr;
		}
		LogicalOperator *node_ptr = node_ref->get();
		if (node_ptr->type == LogicalOperatorType::LOGICAL_GET) {
			auto &g = node_ptr->Cast<LogicalGet>();
			auto tblptr = g.GetTable();
			if (tblptr && tblptr->name == tbl) {
				return node_ref;
			}
		}
		for (auto &c : node_ptr->children) {
			// search child unique_ptr references recursively
			for (auto &c_ref : node_ptr->children) {
				if (c_ref.get() == c.get()) {
					auto res = find_node_ref(&c_ref, tbl);
					if (res) {
						return res;
					}
					break;
				}
			}
		}
		return nullptr;
	};

	unique_ptr<LogicalOperator> *target_ref =
	    find_node_ref(&plan, gets_present.empty() ? std::string() : gets_present[0]);
	if (!target_ref) {
		throw InternalException("PAC compiler: could not find existing LogicalGet for table " +
		                        (gets_present.empty() ? std::string("<none>") : gets_present[0]));
	}

	// Move out the existing node and create a join with the missing get
	unique_ptr<LogicalOperator> existing_node = (*target_ref)->Copy(input.context);
	if (!existing_node) {
		throw InternalException("PAC compiler: unexpected null node while extracting existing GET");
	}
	if (existing_node->type != LogicalOperatorType::LOGICAL_GET) {
		throw InternalException("PAC compiler: expected a LogicalGet when extracting existing GET");
	}
	unique_ptr<LogicalGet> left_get(static_cast<LogicalGet *>(existing_node.release()));

	auto it = get_map.find(gets_missing[0]);
	if (it == get_map.end()) {
		throw InternalException("PAC compiler: created GET for missing table not found in get_map");
	}
	unique_ptr<LogicalGet> right_get = std::move(it->second);
	auto pu_op = right_get->Copy(input.context);
	auto pu_get = dynamic_cast<LogicalGet *>(pu_op.get());

	vector<string> pu_pks;
	for (auto &pk : check.table_metadata.at(privacy_unit).pks) {
		pu_pks.push_back(pk);
	}

	auto join_node = CreateLogicalJoin(check, input.context, std::move(left_get), std::move(right_get));

	// Use ReplaceNode to insert the join and handle column binding remapping
	ReplaceNode(plan, *target_ref, join_node);
#if DEBUG
	plan->Print();
#endif

	// Now we need to edit the aggregate node to use pac functions
	auto *agg = FindTopAggregate(plan);
	if (agg->expressions.size() > 1) {
		throw NotImplementedException("PacBitsliceQuery does not support multiple aggregations!");
	}
	// We create a hash expression over the XOR of PK columns from the privacy unit table
	// Return type is double
	unique_ptr<Expression> hash_input_expr;

		// Build XOR(pk1, pk2, ...) as a scalar expression then hash(...)
		vector<unique_ptr<Expression>> pk_cols;
		for (auto &pk : pu_pks) {
			// find the binding for this pk in the LogicalGet projection (search GetColumnIds and names)
			idx_t proj_idx = DConstants::INVALID_INDEX;
			for (idx_t cid = 0; cid < pu_get->GetColumnIds().size(); cid++) {
				auto col_index = pu_get->GetColumnIds()[cid];
				idx_t primary = col_index.GetPrimaryIndex();
				if (!col_index.IsVirtualColumn() && primary < pu_get->names.size() && pu_get->names[primary] == pk) {
					proj_idx = cid;
					break;
				}
			}
			if (proj_idx == DConstants::INVALID_INDEX) {
				throw InternalException("PAC compiler: failed to find PK column " + pk);
			}
			// create BoundColumnRefExpression referencing table_index and proj_idx
			auto col_binding = ColumnBinding(pu_get->table_index, proj_idx);
			// determine the column's logical type from the LogicalGet's column index
			auto col_index_obj = pu_get->GetColumnIds()[proj_idx];
			auto &col_type = pu_get->GetColumnType(col_index_obj);
			pk_cols.push_back(make_uniq<BoundColumnRefExpression>(col_type, col_binding));

		// If there is only one PK, no XOR needed
		unique_ptr<Expression> xor_expr;
		if (pk_cols.size() == 1) {
			xor_expr = std::move(pk_cols[0]);
		} else {
			// Build chain using optimizer.BindScalarFunction with operator "^" (bitwise XOR)
			// Start with first two, then iteratively XOR with the next
			auto left = std::move(pk_cols[0]);
			for (size_t i = 1; i < pk_cols.size(); ++i) {
				auto right = std::move(pk_cols[i]);
				// use the public two-arg BindScalarFunction
				left = input.optimizer.BindScalarFunction("^", std::move(left), std::move(right));
			}
			xor_expr = std::move(left);
		}

		// Finally bind the hash over the xor_expr
		auto bound_hash = input.optimizer.BindScalarFunction("hash", std::move(xor_expr));
		hash_input_expr = std::move(bound_hash);
	}

	// Extract the original aggregate's value child expression (e.g., the `val` in SUM(val))
	// Hardcoded for now! Assumes only one aggregate expression (fixme)
	unique_ptr<Expression> value_child;
	string pac_function_name;
	string function_name;
	if (agg->expressions[0]->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE) {
		auto &old_aggr = agg->expressions[0]->Cast<BoundAggregateExpression>();
		function_name = old_aggr.function.name;
		if (old_aggr.children.empty()) {
			throw InternalException("PAC compiler: expected aggregate to have a child expression");
		}
		value_child = old_aggr.children[0]->Copy();
	} else {
		throw NotImplementedException("Not found expected aggregate expression in PAC compiler");
	}

	FunctionBinder function_binder(input.context);
	ErrorData error;
	vector<LogicalType> arg_types;
	arg_types.push_back(hash_input_expr->return_type);
	arg_types.push_back(value_child->return_type);

	if (function_name == "sum" || function_name == "sum_no_overflow") {
		pac_function_name = "pac_sum";
	} else if (function_name == "count" || function_name == "count_star") {
		pac_function_name = "pac_count";
	} else {
		throw NotImplementedException("PAC compiler: unsupported aggregate function " + function_name);
	}

	auto &entry = Catalog::GetSystemCatalog(input.context)
	                  .GetEntry<AggregateFunctionCatalogEntry>(input.context, DEFAULT_SCHEMA, pac_function_name);
	auto &aggr_catalog = entry.Cast<AggregateFunctionCatalogEntry>();

	auto best = function_binder.BindFunction(aggr_catalog.name, aggr_catalog.functions, arg_types, error);
	if (!best.IsValid()) {
		throw InternalException("PAC compiler: failed to bind pac aggregate for given argument types");
	}
	auto bound_aggr_func = aggr_catalog.functions.GetFunctionByOffset(best.GetIndex());

	vector<unique_ptr<Expression>> aggr_children;
	aggr_children.push_back(std::move(hash_input_expr));
	aggr_children.push_back(std::move(value_child));

	auto new_aggr = function_binder.BindAggregateFunction(bound_aggr_func, std::move(aggr_children), nullptr,
	                                                      AggregateType::NON_DISTINCT);

	agg->expressions[0] = std::move(new_aggr);
	agg->ResolveOperatorTypes();



}

void ModifyPlanWithPU(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                      const std::vector<std::string> &pks, bool use_rowid) {

	auto pu_scan_ptr = FindPrivacyUnitGetNode(plan);
	auto &get = pu_scan_ptr->get()->Cast<LogicalGet>();
	if (use_rowid) {
		AddRowIDColumn(get);
	} else {
		// Ensure primary key columns are present in the LogicalGet (add them if necessary)
		AddPKColumns(get, pks);
	}

	// Now we need to edit the aggregate node to use pac functions
	auto *agg = FindTopAggregate(plan);
	if (agg->expressions.size() > 1) {
		throw NotImplementedException("PacBitsliceQuery does not support multiple aggregations!");
	}

	// We create a hash expression over either the row id or the XOR of PK columns
	// Return type is double
	unique_ptr<Expression> hash_input_expr;
	if (use_rowid) {
		// rowid is the last column added
		auto rowid_binding = ColumnBinding(get.table_index, get.GetColumnIds().size() - 1);
		auto rowid_col = make_uniq<BoundColumnRefExpression>(LogicalType::BIGINT, rowid_binding);
		auto bound_hash = input.optimizer.BindScalarFunction("hash", std::move(rowid_col));
		hash_input_expr = std::move(bound_hash);
	} else {
		// Build XOR(pk1, pk2, ...) as a scalar expression then hash(...)
		vector<unique_ptr<Expression>> pk_cols;
		for (auto &pk : pks) {
			// find the binding for this pk in the LogicalGet projection (search GetColumnIds and names)
			idx_t proj_idx = DConstants::INVALID_INDEX;
			for (idx_t cid = 0; cid < get.GetColumnIds().size(); cid++) {
				auto col_index = get.GetColumnIds()[cid];
				idx_t primary = col_index.GetPrimaryIndex();
				if (!col_index.IsVirtualColumn() && primary < get.names.size() && get.names[primary] == pk) {
					proj_idx = cid;
					break;
				}
			}
			if (proj_idx == DConstants::INVALID_INDEX) {
				throw InternalException("PAC compiler: failed to find PK column " + pk);
			}
			// create BoundColumnRefExpression referencing table_index and proj_idx
			auto col_binding = ColumnBinding(get.table_index, proj_idx);
			// determine the column's logical type from the LogicalGet's column index
			auto col_index_obj = get.GetColumnIds()[proj_idx];
			auto &col_type = get.GetColumnType(col_index_obj);
			pk_cols.push_back(make_uniq<BoundColumnRefExpression>(col_type, col_binding));
		}

		// If there is only one PK, no XOR needed
		unique_ptr<Expression> xor_expr;
		if (pk_cols.size() == 1) {
			xor_expr = std::move(pk_cols[0]);
		} else {
			// Build chain using optimizer.BindScalarFunction with operator "^" (bitwise XOR)
			// Start with first two, then iteratively XOR with the next
			auto left = std::move(pk_cols[0]);
			for (size_t i = 1; i < pk_cols.size(); ++i) {
				auto right = std::move(pk_cols[i]);
				// use the public two-arg BindScalarFunction
				left = input.optimizer.BindScalarFunction("^", std::move(left), std::move(right));
			}
			xor_expr = std::move(left);
		}

		// Finally bind the hash over the xor_expr
		auto bound_hash = input.optimizer.BindScalarFunction("hash", std::move(xor_expr));
		hash_input_expr = std::move(bound_hash);
	}

	// Extract the original aggregate's value child expression (e.g., the `val` in SUM(val))
	// Hardcoded for now! Assumes only one aggregate expression (fixme)
	unique_ptr<Expression> value_child;
	string pac_function_name;
	string function_name;
	if (agg->expressions[0]->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE) {
		auto &old_aggr = agg->expressions[0]->Cast<BoundAggregateExpression>();
		function_name = old_aggr.function.name;
		if (old_aggr.children.empty()) {
			throw InternalException("PAC compiler: expected aggregate to have a child expression");
		}
		value_child = old_aggr.children[0]->Copy();
	} else {
		throw NotImplementedException("Not found expected aggregate expression in PAC compiler");
	}

	FunctionBinder function_binder(input.context);
	ErrorData error;
	vector<LogicalType> arg_types;
	arg_types.push_back(hash_input_expr->return_type);
	arg_types.push_back(value_child->return_type);

	if (function_name == "sum" || function_name == "sum_no_overflow") {
		pac_function_name = "pac_sum";
	} else if (function_name == "count" || function_name == "count_star") {
		pac_function_name = "pac_count";
	} else {
		throw NotImplementedException("PAC compiler: unsupported aggregate function " + function_name);
	}

	auto &entry = Catalog::GetSystemCatalog(input.context)
	                  .GetEntry<AggregateFunctionCatalogEntry>(input.context, DEFAULT_SCHEMA, pac_function_name);
	auto &aggr_catalog = entry.Cast<AggregateFunctionCatalogEntry>();

	auto best = function_binder.BindFunction(aggr_catalog.name, aggr_catalog.functions, arg_types, error);
	if (!best.IsValid()) {
		throw InternalException("PAC compiler: failed to bind pac aggregate for given argument types");
	}
	auto bound_aggr_func = aggr_catalog.functions.GetFunctionByOffset(best.GetIndex());

	vector<unique_ptr<Expression>> aggr_children;
	aggr_children.push_back(std::move(hash_input_expr));
	aggr_children.push_back(std::move(value_child));

	auto new_aggr = function_binder.BindAggregateFunction(bound_aggr_func, std::move(aggr_children), nullptr,
	                                                      AggregateType::NON_DISTINCT);

	agg->expressions[0] = std::move(new_aggr);
	agg->ResolveOperatorTypes();
}

void CompilePacBitsliceQuery(const PACCompatibilityResult &check, OptimizerExtensionInput &input,
                             unique_ptr<LogicalOperator> &plan, const std::string &privacy_unit,
                             const std::string &query, const std::string &query_hash) {

#ifdef DEBUG
	Printer::Print("CompilePacBitsliceQuery called for PU=" + privacy_unit + " hash=" + query_hash);
#endif

	string path = GetPacCompiledPath(input.context, ".");
	if (!path.empty() && path.back() != '/') {
		path.push_back('/');
	}
	string filename = path + privacy_unit + "_" + query_hash + "_bitslice.sql";

	// The bitslice compiler works in the following way:
	// a) the query scans PU table:
	// a.1) the PU table has 1 PK: we hash it
	// a.2) the PU table has multiple PKs: we XOR them and hash the result
	// a.3) the PU table has no PK: we hash rowid
	// b) the query does not scan PU table:
	// b.1) we follow the FK path to find the PK(s) of the PU table
	// b.2) we join the chain of tables from the scanned table to the PU table
	// b.3) we hash the PK(s) as in a)

	// Example: SELECT group_key, SUM(val) AS sum_val FROM t_single GROUP BY group_key;
	// Becomes: SELECT group_key, pac_sum(HASH(rowid), val) AS sum_val FROM t_single GROUP BY group_key;
	// todo- what is MI? what is k?

	bool pu_present_in_tree = false;
	bool use_rowid = false;

	if (!check.scanned_pu_tables.empty()) {
		pu_present_in_tree = true;
	}

	// Case a) query scans PU table (we assume only 1 PAC table for now)
	std::vector<string> pks;
	// Build two vectors: present (GETs already in the plan) and missing (GETs to create)
	std::vector<std::string> gets_present;
	std::vector<std::string> gets_missing;
	std::vector<LogicalGet *> new_gets;

	// fk_path_to_use will be populated when we detect an FK path (used for multi-hop joins)
	std::vector<std::string> fk_path_to_use;

	if (pu_present_in_tree) {
		// Look up PKs for the scanned PAC table via table_metadata (filled by the compatibility check)
		if (!check.scanned_pu_tables.empty()) {
			auto tbl = check.scanned_pu_tables[0];
			auto it = check.table_metadata.find(tbl);
			if (it != check.table_metadata.end() && !it->second.pks.empty()) {
				pks = it->second.pks;
			} else {
				// no PKs found -> use rowid
				use_rowid = true;
			}
		} else {
			use_rowid = true;
		}
	} else if (!check.fk_paths.empty()) {
		// The query does not scan the PU table: we need to follow the FK path
		string start_table;
		string target_pu;
		PopulateGetsFromFKPath(check, gets_present, gets_missing, start_table, target_pu);

		// extract the ordered fk_path from the compatibility result
		auto it = check.fk_paths.find(start_table);
		if (it == check.fk_paths.end() || it->second.empty()) {
			throw InternalException("PAC compiler: expected fk_path for start table " + start_table);
		}
		fk_path_to_use = it->second;

#ifdef DEBUG
		Printer::Print("PAC bitslice: FK path detection");
		Printer::Print("start_table: " + start_table);
		Printer::Print("target_pu: " + target_pu);
		Printer::Print("fk_path:");
		for (auto &p : fk_path_to_use) {
			Printer::Print("  " + p);
		}
		Printer::Print("gets_present:");
		for (auto &g : gets_present) {
			Printer::Print("  " + g);
		}
		Printer::Print("gets_missing:");
		for (auto &g : gets_missing) {
			Printer::Print("  " + g);
		}
#endif
		// fk_path_to_use is populated and will be passed into ModifyPlanWithoutPU below
	}

	// Now we need to find the PAC scan node
	// Replan the plan without compressed materialization
	ReplanWithoutOptimizers(input.context, query, plan);

	if (pu_present_in_tree) {
		ModifyPlanWithPU(input, plan, pks, use_rowid);
	} else {
		ModifyPlanWithoutPU(check, input, plan, gets_missing, gets_present, fk_path_to_use, privacy_unit);
	}

	plan->ResolveOperatorTypes();
	plan->Verify(input.context);
#ifdef DEBUG
	plan->Print();
#endif
}

} // namespace duckdb
