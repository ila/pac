//
// Created by ila on 12/21/25.
//

#include "include/pac_bitslice_compiler.hpp"
#include "include/pac_helpers.hpp"
#include <iostream>
#include "include/pac_compiler_helpers.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/parser/expression/function_expression.hpp>
#include <duckdb/planner/planner.hpp>
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog.hpp"

namespace duckdb {

void CompilePacBitsliceQuery(const PACCompatibilityResult &check, OptimizerExtensionInput &input,
                             unique_ptr<LogicalOperator> &plan, const std::string &privacy_unit,
                             const std::string &query, const std::string &query_hash) {
	// Bitslice compilation is intentionally left as a stub for now.
	// Implement algorithm here when ready. For now, just emit a diagnostic.
	Printer::Print("CompilePacBitsliceQuery called for PU=" + privacy_unit + " hash=" + query_hash);

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

	// Case a) query scans PU table (we assume only 1 PAC table for now)
	std::vector<string> pks;
	bool use_rowid = false;
	if (!check.scanned_pac_tables.empty()) {
		if (!check.privacy_unit_pks.empty()) {
			pks = check.privacy_unit_pks.at(check.scanned_pac_tables[0]);
		} else {
			use_rowid = true;
		}
	}

	// Now we need to find the PAC scan node
	// Replan the plan without compressed materialization
	ReplanWithoutOptimizers(input.context, query, plan);
	auto pu_scan_ptr = FindPrivacyUnitGetNode(plan);
	auto &get = pu_scan_ptr->get()->Cast<LogicalGet>();
	if (use_rowid) {
		AddRowIDColumn(get);
	}
	// todo
	// else {
	// 	// There are PK columns: check whether they are present in the scan; if not, add them
	// 	for (auto &pk : pks) {
	// 		bool found = false;
	// 		for (idx_t i = 0; i < get.names.size(); i++) {
	// 			if (get.names[i] == pk) {
	// 				found = true;
	// 			}
	// 		}
	// 		if (!found) {
	// 			get.AddColumnId(0);
	// 			get.projection_ids.push_back(get.GetColumnIds().size() - 1);
	// 			get.returned_types.push_back(LogicalTypeId::BIGINT); // todo
	// 			// We also need to add a column binding
	// 			get.GenerateColumnBindings(get.table_index, get.GetColumnIds().size());
	// 		}
	// 	}
	// }
	// TODO - case B

	// Now we need to edit the aggregate node to use pac_sum
	auto *agg = FindTopAggregate(plan);
	if (agg->expressions.size() > 1) {
		throw NotImplementedException("PacBitsliceQuery does not support multiple aggregations!");
	}

	// We create a hash expression over the row id
	// Return type is double
	if (use_rowid) {
		// rowid is the last column added
		auto rowid_binding = ColumnBinding(get.table_index, get.GetColumnIds().size() - 1);
		auto rowid_col = make_uniq<BoundColumnRefExpression>(LogicalType::BIGINT, rowid_binding);

		// Bind the scalar hash(...) function on the rowid column using the optimizer helper
		// This returns a bound scalar expression (BoundFunctionExpression)
		auto bound_hash = input.optimizer.BindScalarFunction("hash", std::move(rowid_col));

		// Extract the original aggregate's value child expression (e.g., the `val` in SUM(val))
		unique_ptr<Expression> value_child;
		if (agg->expressions[0]->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE) {
			auto &old_aggr = agg->expressions[0]->Cast<BoundAggregateExpression>();
			if (old_aggr.children.empty()) {
				throw InternalException("PAC compiler: expected aggregate to have a child expression");
			}
			value_child = old_aggr.children[0]->Copy();
		} else {
			// Fallback: copy whatever expression is present (best-effort)
			value_child = agg->expressions[0]->Copy();
		}

		// Now bind the pac_sum aggregate function with the arguments (hash, value)
		FunctionBinder function_binder(input.context);
		ErrorData error;
		vector<LogicalType> arg_types;
		arg_types.push_back(bound_hash->return_type);
		arg_types.push_back(value_child->return_type);

		// Lookup the pac_sum aggregate in the system catalog
		auto &entry = Catalog::GetSystemCatalog(input.context).GetEntry<AggregateFunctionCatalogEntry>(input.context, DEFAULT_SCHEMA, "pac_sum");
		auto &aggr_catalog = entry.Cast<AggregateFunctionCatalogEntry>();

		auto best = function_binder.BindFunction(aggr_catalog.name, aggr_catalog.functions, arg_types, error);
		if (!best.IsValid()) {
			throw InternalException("PAC compiler: failed to bind pac_sum for given argument types");
		}
		auto bound_aggr_func = aggr_catalog.functions.GetFunctionByOffset(best.GetIndex());

		vector<unique_ptr<Expression>> aggr_children;
		aggr_children.push_back(std::move(bound_hash));
		aggr_children.push_back(std::move(value_child));

		auto new_aggr = function_binder.BindAggregateFunction(bound_aggr_func, std::move(aggr_children), nullptr,
		                                                     AggregateType::NON_DISTINCT);

		// Replace the aggregate expression with the newly bound pac_sum aggregate
		agg->expressions[0] = std::move(new_aggr);
		agg->ResolveOperatorTypes();
	} else {
		// TODO - multiple PKs
		throw NotImplementedException("PacBitsliceQuery does not support multiple PKs yet!");
	}




	plan->ResolveOperatorTypes();
	plan->Verify(input.context);
	//plan->Print();
	//Printer::Print("ok");
}

} // namespace duckdb
