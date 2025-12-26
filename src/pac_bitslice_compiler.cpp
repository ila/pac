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

unique_ptr<LogicalGet> CreateLogicalGet(ClientContext &context, unique_ptr<LogicalOperator> &plan,
                                        const string &table) {

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

// Helper: examine PACCompatibilityResult.fk_paths and populate gets_present / gets_missing.
// Also returns the start table and canonical target privacy unit via reference output parameters.
void PopulateGetsFromFKPath(const PACCompatibilityResult &check, const std::string &path,
                            std::vector<std::string> &gets_present, std::vector<std::string> &gets_missing,
                            std::string &start_table_out, std::string &target_pu_out) {
	// Expect at least one FK path when this is called
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
		for (auto &t : check.scanned_pac_tables) {
			if (t == table_in_path) {
				gets_present.push_back(t);
				found_get = true;
				break;
			}
		}
		for (auto &t : check.scanned_non_pac_tables) {
			if (t == table_in_path) {
				gets_present.push_back(t);
				found_get = true;
				break;
			}
		}
		if (!found_get) {
			gets_missing.push_back(table_in_path);
		}
	}
}

void ModifyPlanWithoutPU(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                         const std::vector<std::string> &gets_missing, const std::vector<std::string> &gets_present,
                         const std::string &privacy_unit, bool use_rowid) {
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

	// Now bind the pac_sum aggregate function with the arguments (hash, value)
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

	// Lookup the pac_sum aggregate in the system catalog
	auto &entry = Catalog::GetSystemCatalog(input.context)
	                  .GetEntry<AggregateFunctionCatalogEntry>(input.context, DEFAULT_SCHEMA, pac_function_name);
	auto &aggr_catalog = entry.Cast<AggregateFunctionCatalogEntry>();

	auto best = function_binder.BindFunction(aggr_catalog.name, aggr_catalog.functions, arg_types, error);
	if (!best.IsValid()) {
		throw InternalException("PAC compiler: failed to bind pac_sum for given argument types");
	}
	auto bound_aggr_func = aggr_catalog.functions.GetFunctionByOffset(best.GetIndex());

	vector<unique_ptr<Expression>> aggr_children;
	aggr_children.push_back(std::move(hash_input_expr));
	aggr_children.push_back(std::move(value_child));

	auto new_aggr = function_binder.BindAggregateFunction(bound_aggr_func, std::move(aggr_children), nullptr,
	                                                      AggregateType::NON_DISTINCT);

	// Replace the aggregate expression with the newly bound pac_sum aggregate
	agg->expressions[0] = std::move(new_aggr);
	// Hardcoded for now! Assumes only one aggregate expression (fixme)
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

	if (!check.scanned_pac_tables.empty()) {
		pu_present_in_tree = true;
	}

	// Case a) query scans PU table (we assume only 1 PAC table for now)
	std::vector<string> pks;
	// Build two vectors: present (GETs already in the plan) and missing (GETs to create)
	std::vector<std::string> gets_present;
	std::vector<std::string> gets_missing;
	std::vector<LogicalGet *> new_gets;

	if (pu_present_in_tree) {
		if (!check.privacy_unit_pks.empty()) {
			// Has PKs
			pks = check.privacy_unit_pks.at(check.scanned_pac_tables[0]);
		} else {
			use_rowid = true;
		}
	} else if (!check.fk_paths.empty()) {
		// The query does not scan the PU table: we need to follow the FK path
		string start_table;
		string target_pu;
		PopulateGetsFromFKPath(check, path, gets_present, gets_missing, start_table, target_pu);

#ifdef DEBUG
		Printer::Print("PAC bitslice: FK path detection");
		Printer::Print("start_table: " + start_table);
		Printer::Print("target_pu: " + target_pu);
		Printer::Print("gets_present:");
		for (auto &g : gets_present)
			Printer::Print("  " + g);
		Printer::Print("gets_missing:");
		for (auto &g : gets_missing)
			Printer::Print("  " + g);
#endif
	}

	// Now we need to find the PAC scan node
	// Replan the plan without compressed materialization
	ReplanWithoutOptimizers(input.context, query, plan);

	if (pu_present_in_tree) {
		ModifyPlanWithPU(input, plan, pks, use_rowid);
	} else {
		ModifyPlanWithoutPU(input, plan, gets_missing, gets_present, privacy_unit, use_rowid);
	}

	plan->ResolveOperatorTypes();
	plan->Verify(input.context);
#ifdef DEBUG
	plan->Print();
#endif
}

} // namespace duckdb
