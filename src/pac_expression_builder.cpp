//
// Created by ila on 1/6/26.
//

#include "include/pac_expression_builder.hpp"

#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/common/constants.hpp"
#include "duckdb/optimizer/optimizer.hpp"

namespace duckdb {

// Ensure a column is projected in a LogicalGet and return its projection index
idx_t EnsureProjectedColumn(LogicalGet &g, const string &col_name) {
	// try existing projected columns by matching returned names via ColumnIndex primary
	for (idx_t cid = 0; cid < g.GetColumnIds().size(); ++cid) {
		auto col_idx = g.GetColumnIds()[cid];
		if (!col_idx.IsVirtualColumn()) {
			idx_t primary = col_idx.GetPrimaryIndex();
			if (primary < g.names.size() && g.names[primary] == col_name) {
				return cid;
			}
		}
	}
	// otherwise add column from table schema
	auto table_entry = g.GetTable();
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
	g.AddColumnId(logical_idx);
	g.projection_ids.push_back(g.GetColumnIds().size() - 1);
	g.GenerateColumnBindings(g.table_index, g.GetColumnIds().size());
	return g.GetColumnIds().size() - 1;
}

// Ensure PK columns are present in a LogicalGet's column_ids and projection_ids
void AddPKColumns(LogicalGet &get, const vector<string> &pks) {
	auto table_entry_ptr = get.GetTable();
	if (!table_entry_ptr) {
		throw InternalException("PAC compiler: expected LogicalGet to be bound to a table when PKs are present");
	}
	for (auto &pk : pks) {
		idx_t proj_idx = EnsureProjectedColumn(get, pk);
		if (proj_idx == DConstants::INVALID_INDEX) {
			throw InternalException("PAC compiler: could not find PK column " + pk + " in table");
		}
	}
}

// Helper to ensure rowid is present in the output columns of a LogicalGet
void AddRowIDColumn(LogicalGet &get) {
	if (get.virtual_columns.find(COLUMN_IDENTIFIER_ROW_ID) != get.virtual_columns.end()) {
		get.virtual_columns[COLUMN_IDENTIFIER_ROW_ID] = TableColumn("rowid", LogicalTypeId::BIGINT);
	}
	get.AddColumnId(COLUMN_IDENTIFIER_ROW_ID);
	get.projection_ids.push_back(get.GetColumnIds().size() - 1);
	get.returned_types.push_back(LogicalTypeId::BIGINT);
	// We also need to add a column binding for rowid
	get.GenerateColumnBindings(get.table_index, get.GetColumnIds().size());
}

// Build XOR(pk1, pk2, ...) then hash(...) bound expression for the given LogicalGet's PKs
unique_ptr<Expression> BuildXorHashFromPKs(OptimizerExtensionInput &input, LogicalGet &get, const vector<string> &pks) {
	vector<unique_ptr<Expression>> pk_cols;
	for (auto &pk : pks) {
		idx_t proj_idx = EnsureProjectedColumn(get, pk);
		if (proj_idx == DConstants::INVALID_INDEX) {
			throw InternalException("PAC compiler: failed to find PK column " + pk);
		}
		auto col_binding = ColumnBinding(get.table_index, proj_idx);
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
		auto left = std::move(pk_cols[0]);
		for (size_t i = 1; i < pk_cols.size(); ++i) {
			auto right = std::move(pk_cols[i]);
			left = input.optimizer.BindScalarFunction("^", std::move(left), std::move(right));
		}
		xor_expr = std::move(left);
	}

	// Finally bind the hash over the xor_expr
	return input.optimizer.BindScalarFunction("hash", std::move(xor_expr));
}

// Map aggregate function name to PAC function name
static string GetPacAggregateFunctionName(const string &function_name) {
	string pac_function_name;
	if (function_name == "sum" || function_name == "sum_no_overflow") {
		pac_function_name = "pac_sum";
	} else if (function_name == "count" || function_name == "count_star") {
		pac_function_name = "pac_count";
	} else if (function_name == "avg") {
		pac_function_name = "pac_avg";
	} else if (function_name == "min") {
		pac_function_name = "pac_min";
	} else if (function_name == "max") {
		pac_function_name = "pac_max";
	} else {
		throw NotImplementedException("PAC compiler: unsupported aggregate function " + function_name);
	}
	return pac_function_name;
}

// Modify aggregate expressions to use PAC functions (extracted from duplicate code)
void ModifyAggregatesWithPacFunctions(OptimizerExtensionInput &input, LogicalAggregate *agg,
                                      unique_ptr<Expression> &hash_input_expr) {
	FunctionBinder function_binder(input.context);

	// Process each aggregate expression
	for (idx_t i = 0; i < agg->expressions.size(); i++) {
		if (agg->expressions[i]->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
			throw NotImplementedException("Not found expected aggregate expression in PAC compiler");
		}

		auto &old_aggr = agg->expressions[i]->Cast<BoundAggregateExpression>();
		string function_name = old_aggr.function.name;

		// Extract the original aggregate's value child expression (e.g., the `val` in SUM(val))
		// COUNT(*) has no children, so we create a constant 1 expression when there's no child
		unique_ptr<Expression> value_child;
		if (old_aggr.children.empty()) {
			// COUNT(*) case - create a constant 1
			if (function_name == "count_star" || function_name == "count") {
				value_child = make_uniq_base<Expression, BoundConstantExpression>(Value::BIGINT(1));
			} else {
				throw InternalException("PAC compiler: expected aggregate to have a child expression");
			}
		} else {
			value_child = old_aggr.children[0]->Copy();
		}

		// Get PAC function name
		string pac_function_name = GetPacAggregateFunctionName(function_name);

		// Bind the PAC aggregate function
		ErrorData error;
		vector<LogicalType> arg_types;
		arg_types.push_back(hash_input_expr->return_type);
		arg_types.push_back(value_child->return_type);

		auto &entry = Catalog::GetSystemCatalog(input.context)
		                  .GetEntry<AggregateFunctionCatalogEntry>(input.context, DEFAULT_SCHEMA, pac_function_name);
		auto &aggr_catalog = entry.Cast<AggregateFunctionCatalogEntry>();

		auto best = function_binder.BindFunction(aggr_catalog.name, aggr_catalog.functions, arg_types, error);
		if (!best.IsValid()) {
			throw InternalException("PAC compiler: failed to bind pac aggregate for given argument types");
		}
		auto bound_aggr_func = aggr_catalog.functions.GetFunctionByOffset(best.GetIndex());

		// Build arguments for this aggregate (copy hash expression for each aggregate)
		vector<unique_ptr<Expression>> aggr_children;
		aggr_children.push_back(hash_input_expr->Copy());
		aggr_children.push_back(std::move(value_child));

		// Pass through the DISTINCT flag from the original aggregate
		AggregateType agg_type = old_aggr.IsDistinct() ? AggregateType::DISTINCT : AggregateType::NON_DISTINCT;

		auto new_aggr =
		    function_binder.BindAggregateFunction(bound_aggr_func, std::move(aggr_children), nullptr, agg_type);

		agg->expressions[i] = std::move(new_aggr);
	}

	agg->ResolveOperatorTypes();
}

} // namespace duckdb
