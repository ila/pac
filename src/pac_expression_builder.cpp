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

// Helper function to find a LogicalGet by table_index in the operator tree
static LogicalGet *FindLogicalGetByTableIndex(LogicalOperator &op, idx_t table_index) {
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op.Cast<LogicalGet>();
		if (get.table_index == table_index) {
			return &get;
		}
	}
	for (auto &child : op.children) {
		auto result = FindLogicalGetByTableIndex(*child, table_index);
		if (result) {
			return result;
		}
	}
	return nullptr;
}

// Helper function to resolve a column name from a binding by finding the corresponding LogicalGet
static string ResolveColumnNameFromBinding(LogicalOperator &root, const ColumnBinding &binding) {
	auto get = FindLogicalGetByTableIndex(root, binding.table_index);
	if (!get) {
		return "[" + std::to_string(binding.table_index) + "." + std::to_string(binding.column_index) + "]";
	}

	const auto &column_ids = get->GetColumnIds();
	if (binding.column_index >= column_ids.size()) {
		return "[" + std::to_string(binding.table_index) + "." + std::to_string(binding.column_index) + "]";
	}

	auto col_name = get->GetColumnName(column_ids[binding.column_index]);
	if (col_name.empty()) {
		return "[" + std::to_string(binding.table_index) + "." + std::to_string(binding.column_index) + "]";
	}

	return col_name;
}

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

	// IMPORTANT: If projection_ids is empty, DuckDB generates bindings from column_ids.size().
	// Before we add a new column, we need to populate projection_ids with all existing indices
	// to maintain the correct binding generation.
	if (g.projection_ids.empty()) {
		for (idx_t i = 0; i < g.GetColumnIds().size(); i++) {
			g.projection_ids.push_back(i);
		}
	}

	// Add the column to the LogicalGet
	g.AddColumnId(logical_idx);

	// The projection index is the position in the output, which equals the new size - 1
	idx_t new_proj_idx = g.GetColumnIds().size() - 1;
	g.projection_ids.push_back(new_proj_idx);

	// ResolveOperatorTypes() calls the protected ResolveTypes() which rebuilds
	// the types vector from column_ids/projection_ids
	g.ResolveOperatorTypes();

	return new_proj_idx;
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

	// IMPORTANT: If projection_ids is empty, DuckDB generates bindings from column_ids.size().
	// Before we add a new column, we need to populate projection_ids with all existing indices
	// to maintain the correct binding generation.
	if (get.projection_ids.empty()) {
		for (idx_t i = 0; i < get.GetColumnIds().size(); i++) {
			get.projection_ids.push_back(i);
		}
	}

	get.AddColumnId(COLUMN_IDENTIFIER_ROW_ID);
	get.projection_ids.push_back(get.GetColumnIds().size() - 1);

	// ResolveOperatorTypes() calls the protected ResolveTypes() which rebuilds
	// the types vector from column_ids/projection_ids
	get.ResolveOperatorTypes();
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

// Build AND expression from multiple hash expressions (for multiple PUs)
unique_ptr<Expression> BuildAndFromHashes(OptimizerExtensionInput &input, vector<unique_ptr<Expression>> &hash_exprs) {
	if (hash_exprs.empty()) {
		throw InternalException("PAC compiler: cannot build AND expression from empty hash list");
	}

	// If there is only one hash, return it directly
	if (hash_exprs.size() == 1) {
		return std::move(hash_exprs[0]);
	}

	// Build chain using optimizer.BindScalarFunction with operator "&" (bitwise AND)
	auto left = std::move(hash_exprs[0]);
	for (size_t i = 1; i < hash_exprs.size(); ++i) {
		auto right = std::move(hash_exprs[i]);
		left = input.optimizer.BindScalarFunction("&", std::move(left), std::move(right));
	}

	return left;
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

/**
 * ModifyAggregatesWithPacFunctions: Transforms regular aggregate expressions to PAC aggregate expressions
 *
 * Purpose: This is the core transformation function that replaces standard aggregates (SUM, AVG, COUNT, MIN, MAX)
 * with their PAC equivalents (pac_sum, pac_avg, pac_count, pac_min, pac_max) by adding the hash expression
 * as the first argument.
 *
 * Arguments:
 * @param input - Optimizer extension input containing context and function binder
 * @param agg - The LogicalAggregate node whose expressions will be transformed
 * @param hash_input_expr - The hash expression identifying privacy units (e.g., hash(c_custkey) or hash(xor(pk1, pk2)))
 *
 * Logic:
 * 1. For each aggregate expression in the node:
 *    - Extract the original aggregate function name (sum, avg, count, etc.)
 *    - Extract the value expression (the data being aggregated)
 *      * For COUNT(*), create a constant 1 expression
 *      * For others (SUM(val), AVG(val)), use the existing child expression
 *    - Map to the PAC function name (sum -> pac_sum, avg -> pac_avg, etc.)
 *    - Bind the PAC aggregate function with two arguments:
 *      * First argument: hash expression (identifies which PU each row belongs to)
 *      * Second argument: value expression (the data to aggregate per PU)
 *    - Preserve the DISTINCT flag from the original aggregate
 *    - Replace the old aggregate expression with the new PAC aggregate
 * 2. Resolve operator types to ensure the plan is valid
 *
 * Transformation Examples:
 * - SUM(l_extendedprice) -> pac_sum(hash(c_custkey), l_extendedprice)
 * - AVG(l_quantity) -> pac_avg(hash(c_custkey), l_quantity)
 * - COUNT(*) -> pac_count(hash(c_custkey), 1)
 * - COUNT(DISTINCT l_orderkey) -> pac_count(hash(c_custkey), l_orderkey) with DISTINCT flag
 *
 * Nested Aggregate Handling (IMPORTANT):
 * This function is only called on aggregates that were filtered by the calling code to have
 * PU or FK-linked tables in their subtree. The filtering logic determines which aggregates get transformed:
 *
 * - If we have 2 aggregates stacked on top of each other:
 *   * Inner aggregate WITH PU/FK-linked tables -> Gets transformed (this function is called)
 *   * Outer aggregate WITHOUT PU/FK-linked tables -> NOT transformed (this function not called)
 *   * Outer aggregate WITH PU/FK-linked tables -> Also gets transformed (this function called separately)
 *
 * Example 1 - Both aggregates have PU tables (user's TPC-H Q17 example):
 *   SELECT sum(l_extendedprice) / 7.0 FROM lineitem WHERE l_quantity < (SELECT avg(l_quantity) FROM lineitem WHERE ...)
 *   - Inner: Has lineitem -> pac_avg(hash(rowid), l_quantity)
 *   - Outer: Also has lineitem -> pac_sum(hash(rowid), l_extendedprice)
 *
 * Example 2 - Only inner has PU tables:
 *   SELECT sum(inner_result) FROM (SELECT sum(customer_col) FROM customer) AS subq
 *   - Inner: Has customer -> pac_sum(hash(c_custkey), customer_col)
 *   - Outer: No PU tables -> Regular sum(inner_result), NOT transformed
 */
void ModifyAggregatesWithPacFunctions(OptimizerExtensionInput &input, LogicalAggregate *agg,
                                      unique_ptr<Expression> &hash_input_expr) {
	FunctionBinder function_binder(input.context);

#ifdef DEBUG
	Printer::Print("ModifyAggregatesWithPacFunctions: Processing aggregate with " +
	               std::to_string(agg->expressions.size()) + " expressions");

	// Debug: Print hash input expression details
	Printer::Print("ModifyAggregatesWithPacFunctions: Hash input expression: " + hash_input_expr->ToString());
	Printer::Print("ModifyAggregatesWithPacFunctions: Hash input type: " + hash_input_expr->return_type.ToString());

	// Debug: Print all column references in the hash expression
	ExpressionIterator::EnumerateExpression(hash_input_expr, [&](Expression &expr) {
		if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
			auto &col_ref = expr.Cast<BoundColumnRefExpression>();
			Printer::Print("ModifyAggregatesWithPacFunctions: Hash expr references column [" +
			               std::to_string(col_ref.binding.table_index) + "." +
			               std::to_string(col_ref.binding.column_index) + "] type=" + col_ref.return_type.ToString());
		}
	});

	// Debug: Print aggregate's groups
	Printer::Print("ModifyAggregatesWithPacFunctions: Aggregate has " + std::to_string(agg->groups.size()) +
	               " groups:");
	for (idx_t i = 0; i < agg->groups.size(); i++) {
		Printer::Print("  Group " + std::to_string(i) + ": " + agg->groups[i]->ToString());
	}

	// Debug: Print aggregate's group_index and aggregate_index
	Printer::Print("ModifyAggregatesWithPacFunctions: group_index=" + std::to_string(agg->group_index) +
	               ", aggregate_index=" + std::to_string(agg->aggregate_index));

	// Debug: Print child operator info
	if (!agg->children.empty()) {
		Printer::Print("ModifyAggregatesWithPacFunctions: Child operator type=" +
		               std::to_string(static_cast<int>(agg->children[0]->type)));
		if (agg->children[0]->type == LogicalOperatorType::LOGICAL_GET) {
			auto &child_get = agg->children[0]->Cast<LogicalGet>();
			Printer::Print("ModifyAggregatesWithPacFunctions: Child is GET with table_index=" +
			               std::to_string(child_get.table_index) +
			               ", columns=" + std::to_string(child_get.GetColumnIds().size()));
		}
	}
#endif

	// Process each aggregate expression
	for (idx_t i = 0; i < agg->expressions.size(); i++) {
		if (agg->expressions[i]->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
			throw NotImplementedException("Not found expected aggregate expression in PAC compiler");
		}

		auto &old_aggr = agg->expressions[i]->Cast<BoundAggregateExpression>();
		string function_name = old_aggr.function.name;

#ifdef DEBUG
		Printer::Print("ModifyAggregatesWithPacFunctions: Transforming " + function_name + " to PAC function");
		Printer::Print("ModifyAggregatesWithPacFunctions: Old aggregate expression: " + old_aggr.ToString());
		if (!old_aggr.children.empty()) {
			Printer::Print("ModifyAggregatesWithPacFunctions: Old aggregate child: " +
			               old_aggr.children[0]->ToString());
			if (old_aggr.children[0]->type == ExpressionType::BOUND_COLUMN_REF) {
				auto &child_ref = old_aggr.children[0]->Cast<BoundColumnRefExpression>();
				Printer::Print("ModifyAggregatesWithPacFunctions: Old aggregate child binding: [" +
				               std::to_string(child_ref.binding.table_index) + "." +
				               std::to_string(child_ref.binding.column_index) + "]");
			}
		}
#endif

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

#ifdef DEBUG
		Printer::Print("ModifyAggregatesWithPacFunctions: New PAC aggregate expression: " + new_aggr->ToString());

		// Print column names for the PAC aggregate arguments
		string hash_col_name = "unknown";
		string value_col_name = "unknown";

		// Extract hash column name from hash_input_expr by resolving bindings
		ExpressionIterator::EnumerateExpression(hash_input_expr, [&](Expression &expr) {
			if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
				auto &col_ref = expr.Cast<BoundColumnRefExpression>();
				hash_col_name = ResolveColumnNameFromBinding(*agg, col_ref.binding);
			}
		});

		// Extract value column name from the new aggregate's children
		if (!new_aggr->Cast<BoundAggregateExpression>().children.empty() &&
		    new_aggr->Cast<BoundAggregateExpression>().children.size() > 1) {
			auto &value_expr = new_aggr->Cast<BoundAggregateExpression>().children[1];
			if (value_expr->type == ExpressionType::BOUND_COLUMN_REF) {
				auto &val_ref = value_expr->Cast<BoundColumnRefExpression>();
				value_col_name = ResolveColumnNameFromBinding(*agg, val_ref.binding);
			} else if (value_expr->type == ExpressionType::VALUE_CONSTANT) {
				value_col_name = "1"; // COUNT(*) case
			} else {
				// For complex expressions, try to extract column names from any column refs
				vector<string> col_names;
				ExpressionIterator::EnumerateExpression(value_expr, [&](Expression &e) {
					if (e.type == ExpressionType::BOUND_COLUMN_REF) {
						auto &cr = e.Cast<BoundColumnRefExpression>();
						col_names.push_back(ResolveColumnNameFromBinding(*agg, cr.binding));
					}
				});
				if (!col_names.empty()) {
					value_col_name = "expression(" + StringUtil::Join(col_names, ", ") + ")";
				} else {
					value_col_name = value_expr->ToString();
				}
			}
		}

		Printer::Print("ModifyAggregatesWithPacFunctions: Constructing " + pac_function_name + " with hash of " +
		               hash_col_name + ", value of " + value_col_name);
#endif

		agg->expressions[i] = std::move(new_aggr);
	}

#ifdef DEBUG
	Printer::Print("ModifyAggregatesWithPacFunctions: Calling ResolveOperatorTypes on aggregate");
#endif
	agg->ResolveOperatorTypes();
#ifdef DEBUG
	Printer::Print("ModifyAggregatesWithPacFunctions: After ResolveOperatorTypes, types.size()=" +
	               std::to_string(agg->types.size()));
	for (idx_t i = 0; i < agg->types.size(); i++) {
		Printer::Print("  Type " + std::to_string(i) + ": " + agg->types[i].ToString());
	}
#endif
}

} // namespace duckdb
