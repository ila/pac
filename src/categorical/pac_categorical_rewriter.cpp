//
// PAC Categorical Query Rewriter - Implementation
//
// See pac_categorical_rewriter.hpp for design documentation.
//
// Created by ila on 1/23/26.
//

#include "categorical/pac_categorical_rewriter.hpp"
#include "pac_debug.hpp"
#include "query_processing/pac_plan_traversal.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_subquery_expression.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_lambda_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/lambda_functions.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/function/scalar/struct_functions.hpp"

namespace duckdb {

// List of PAC aggregate function names
static const vector<string> PAC_AGGREGATE_NAMES = {"pac_sum", "pac_count", "pac_avg", "pac_min", "pac_max"};

// Check if a function name is a PAC aggregate
static bool IsPacAggregate(const string &name) {
	for (auto &pac_name : PAC_AGGREGATE_NAMES) {
		if (name == pac_name) {
			return true;
		}
	}
	return false;
}

// Check if a function name is already a PAC counters variant
static bool IsPacCountersAggregate(const string &name) {
	return name.find("_counters") != string::npos;
}

// Check if a function name is a PAC list aggregate (pac_*_list)
static bool IsPacListAggregate(const string &name) {
	return name.find("pac_") == 0 && name.find("_list") != string::npos;
}

string GetCountersVariant(const string &aggregate_name) {
	// Already a counters variant
	if (IsPacCountersAggregate(aggregate_name)) {
		return aggregate_name;
	}
	return aggregate_name + "_counters";
}

// Hash a column binding to a unique 64-bit value for use in maps/sets
static inline uint64_t HashBinding(const ColumnBinding &binding) {
	return (uint64_t(binding.table_index) << 32) | binding.column_index;
}

// Check if a type is numerical (can be used with pac_noised)
static bool IsNumericalType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UHUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return true;
	default:
		return false;
	}
}

// Forward declarations
static string FindPacAggregateInExpressionWithPlan(Expression *expr, LogicalOperator *plan_root);
static string TracePacAggregateFromBinding(const ColumnBinding &binding, LogicalOperator *plan_root);
static LogicalAggregate *FindAggregateForBinding(const ColumnBinding &binding, LogicalOperator *plan_root);

// Information about a single PAC aggregate binding found in an expression
struct PacBindingInfo {
	ColumnBinding binding;
	string aggregate_name;     // e.g., "pac_sum", "pac_count"
	LogicalType original_type; // The type before conversion to LIST<DOUBLE>
	idx_t index;               // Position in the list (0-based, for list_zip field access)
};

// Forward declaration for FindAllPacBindingsInExpression
static vector<PacBindingInfo> FindAllPacBindingsInExpression(Expression *expr, LogicalOperator *plan_root);

// Forward declaration for IsAlreadyWrappedInPacNoised
static bool IsAlreadyWrappedInPacNoised(Expression *expr);

// Helper: Find the first aggregate operator in a subtree
static LogicalAggregate *FindFirstAggregateInSubtree(LogicalOperator *op) {
	if (!op) {
		return nullptr;
	}
	if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return &op->Cast<LogicalAggregate>();
	}
	for (auto &child : op->children) {
		auto *agg = FindFirstAggregateInSubtree(child.get());
		if (agg) {
			return agg;
		}
	}
	return nullptr;
}

// Helper: Find the operator in the plan that produces a given table_index
static LogicalOperator *FindOperatorByTableIndex(LogicalOperator *op, idx_t table_index) {
	if (!op) {
		return nullptr;
	}

	// Check if this operator produces the table_index
	if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = op->Cast<LogicalProjection>();
		if (proj.table_index == table_index) {
			return op;
		}
	} else if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &aggr = op->Cast<LogicalAggregate>();
		if (aggr.group_index == table_index || aggr.aggregate_index == table_index) {
			return op;
		}
	}
	// Note: Joins don't have their own table_index, they pass through from children
	// So we just recurse into children (handled below)

	// Recurse into children
	for (auto &child : op->children) {
		auto *result = FindOperatorByTableIndex(child.get(), table_index);
		if (result) {
			return result;
		}
	}
	return nullptr;
}

// Check if a function name is a PAC aggregate (original, counters, or list variant)
static bool IsPacAggregateOrCounters(const string &name) {
	return IsPacAggregate(name) || IsPacCountersAggregate(name) || IsPacListAggregate(name);
}

// Get the base PAC aggregate name (strip _counters suffix if present)
static string GetBasePacAggregateName(const string &name) {
	if (IsPacCountersAggregate(name)) {
		// Remove "_counters" suffix
		return name.substr(0, name.size() - 9);
	}
	return name;
}

// Recognize DuckDB's scalar subquery wrapper patterns:
//
// Pattern 1 (uncorrelated): Projection -> Aggregate(first) -> Projection
//   Projection (CASE error if count > 1, else first(value))
//   └── Aggregate (first(#0), count_star())
//       └── Projection (#0)
//           └── [actual scalar subquery result]
//
// Pattern 2 (correlated): Projection with CASE and error() check
//   Projection (CASE error if count > 1, else value from join)
//   └── JOIN (DELIM_JOIN, COMPARISON_JOIN, etc.)
//       └── ... [contains the actual aggregate]
//
// Returns the operator to search for PAC aggregates (skips the wrapper).
// Returns nullptr if pattern not recognized.
static LogicalOperator *RecognizeDuckDBScalarWrapper(LogicalOperator *op) {
	if (!op || op->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return nullptr;
	}

	auto &outer_proj = op->Cast<LogicalProjection>();
	if (outer_proj.children.empty()) {
		return nullptr;
	}

	auto *child = outer_proj.children[0].get();
	PAC_DEBUG_PRINT("      RecognizeDuckDBScalarWrapper: child type=" + LogicalOperatorToString(child->type));

	// Pattern 1: Uncorrelated - Projection -> Aggregate(first) -> Projection
	if (child->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &agg = child->Cast<LogicalAggregate>();
		bool has_first = false;
		for (auto &agg_expr : agg.expressions) {
			if (agg_expr->type == ExpressionType::BOUND_AGGREGATE) {
				auto &bound_agg = agg_expr->Cast<BoundAggregateExpression>();
				PAC_DEBUG_PRINT("      RecognizeDuckDBScalarWrapper: agg func=" + bound_agg.function.name);
				if (bound_agg.function.name == "first") {
					has_first = true;
					break;
				}
			}
		}
		if (has_first && !agg.children.empty()) {
			auto *inner_proj_op = agg.children[0].get();
			PAC_DEBUG_PRINT("      RecognizeDuckDBScalarWrapper: has_first=true, inner_proj type=" +
			                LogicalOperatorToString(inner_proj_op->type));
			if (inner_proj_op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto &inner_proj = inner_proj_op->Cast<LogicalProjection>();
				if (!inner_proj.children.empty()) {
					PAC_DEBUG_PRINT("      RecognizeDuckDBScalarWrapper: MATCHED! Returning inner's child");
					return inner_proj.children[0].get();
				}
			}
		}
	}

	// Pattern 2: Correlated - Projection with CASE/error check, child is a JOIN
	// Check if outer projection has a CASE with error() function (scalar subquery error check)
	bool has_error_case = false;
	for (auto &expr : outer_proj.expressions) {
		if (expr->type == ExpressionType::CASE_EXPR) {
			auto &case_expr = expr->Cast<BoundCaseExpression>();
			for (auto &case_check : case_expr.case_checks) {
				if (case_check.then_expr && case_check.then_expr->type == ExpressionType::BOUND_FUNCTION) {
					auto &func = case_check.then_expr->Cast<BoundFunctionExpression>();
					if (func.function.name == "error") {
						has_error_case = true;
						break;
					}
				}
			}
			if (has_error_case) {
				break;
			}
		}
	}

	// If it has the error() CASE pattern and child is a join, return the child
	// so we can search it for PAC aggregates
	if (has_error_case) {
		if (child->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
		    child->type == LogicalOperatorType::LOGICAL_DELIM_JOIN ||
		    child->type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
			return child;
		}
	}

	return nullptr;
}

// Forward declarations
static string FindPacAggregateInOperator(LogicalOperator *op);
static LogicalOperator *FindScalarWrapperForBinding(const ColumnBinding &binding, LogicalOperator *plan_root);

// Trace a column binding through the plan to find if it comes from a PAC aggregate
// Returns the PAC aggregate name if found (base name without _counters), empty string otherwise
static string TracePacAggregateFromBinding(const ColumnBinding &binding, LogicalOperator *plan_root) {
	if (!plan_root) {
		return "";
	}

	// Find the operator that produces this binding
	auto *source_op = FindOperatorByTableIndex(plan_root, binding.table_index);
	if (!source_op) {
		return "";
	}

	if (source_op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		// Check if this is DuckDB's scalar subquery wrapper pattern
		auto *unwrapped = RecognizeDuckDBScalarWrapper(source_op);
		if (unwrapped) {
			// Skip the wrapper and search the actual scalar subquery source
			return FindPacAggregateInOperator(unwrapped);
		}

		auto &proj = source_op->Cast<LogicalProjection>();
		if (binding.column_index < proj.expressions.size()) {
			// Recursively search this expression for PAC aggregates
			return FindPacAggregateInExpressionWithPlan(proj.expressions[binding.column_index].get(), plan_root);
		}
	} else if (source_op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &aggr = source_op->Cast<LogicalAggregate>();
		PAC_DEBUG_PRINT("      TracePacAggregate: binding.table_index=" + std::to_string(binding.table_index) +
		                " aggregate_index=" + std::to_string(aggr.aggregate_index) +
		                " group_index=" + std::to_string(aggr.group_index));
		if (binding.table_index == aggr.aggregate_index) {
			// This binding comes from an aggregate expression
			if (binding.column_index < aggr.expressions.size()) {
				auto &agg_expr = aggr.expressions[binding.column_index];
				PAC_DEBUG_PRINT("      TracePacAggregate: agg_expr type=" + ExpressionTypeToString(agg_expr->type));
				if (agg_expr->type == ExpressionType::BOUND_AGGREGATE) {
					auto &bound_agg = agg_expr->Cast<BoundAggregateExpression>();
					PAC_DEBUG_PRINT("      TracePacAggregate: function=" + bound_agg.function.name +
					                " is_pac=" + (IsPacAggregateOrCounters(bound_agg.function.name) ? "yes" : "no"));
					// Check for both original and counters variants
					if (IsPacAggregateOrCounters(bound_agg.function.name)) {
						return GetBasePacAggregateName(bound_agg.function.name);
					}
				}
			}
		}
		// If it's from group_index, it's a GROUP BY column, not an aggregate result
	}

	return "";
}

// Search an operator subtree for PAC aggregates
// Returns the first PAC aggregate name found, empty string otherwise
static string FindPacAggregateInOperator(LogicalOperator *op) {
	if (!op) {
		return "";
	}

	// Check if this is an aggregate operator with PAC aggregates
	if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &agg = op->Cast<LogicalAggregate>();
		for (auto &agg_expr : agg.expressions) {
			if (agg_expr->type == ExpressionType::BOUND_AGGREGATE) {
				auto &bound_agg = agg_expr->Cast<BoundAggregateExpression>();
				if (IsPacAggregateOrCounters(bound_agg.function.name)) {
					return GetBasePacAggregateName(bound_agg.function.name);
				}
			}
		}
	}

	// Recurse into children
	for (auto &child : op->children) {
		string result = FindPacAggregateInOperator(child.get());
		if (!result.empty()) {
			return result;
		}
	}

	return "";
}

// Recursively search for PAC aggregate in an expression tree, with plan context for tracing column refs
// Returns the base aggregate name (without _counters suffix)
static string FindPacAggregateInExpressionWithPlan(Expression *expr, LogicalOperator *plan_root) {
	if (!expr) {
		return "";
	}

	// Check if this is a bound aggregate expression with a PAC function
	if (expr->type == ExpressionType::BOUND_AGGREGATE) {
		auto &agg_expr = expr->Cast<BoundAggregateExpression>();
		if (IsPacAggregateOrCounters(agg_expr.function.name)) {
			return GetBasePacAggregateName(agg_expr.function.name);
		}
		// Check children of aggregate
		for (auto &child : agg_expr.children) {
			string result = FindPacAggregateInExpressionWithPlan(child.get(), plan_root);
			if (!result.empty()) {
				return result;
			}
		}
		return "";
	}

	// Check if this is a function expression (could be wrapping a PAC aggregate result)
	if (expr->type == ExpressionType::BOUND_FUNCTION) {
		auto &func_expr = expr->Cast<BoundFunctionExpression>();
		// Check children
		for (auto &child : func_expr.children) {
			string result = FindPacAggregateInExpressionWithPlan(child.get(), plan_root);
			if (!result.empty()) {
				return result;
			}
		}
		return "";
	}

	// Check column references - trace through the plan to find the source
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();
		return TracePacAggregateFromBinding(col_ref.binding, plan_root);
	}

	// Check if this is a subquery expression - search inside for PAC aggregates
	if (expr->type == ExpressionType::SUBQUERY) {
		auto &subquery_expr = expr->Cast<BoundSubqueryExpression>();
		// Search the subquery's children (for IN, ANY, ALL operators)
		for (auto &child : subquery_expr.children) {
			string result = FindPacAggregateInExpressionWithPlan(child.get(), plan_root);
			if (!result.empty()) {
				return result;
			}
		}
		// For scalar subqueries, we need to search the subquery plan
		// The subquery has already been planned, and the result may reference PAC aggregates
		// Search the plan_root for aggregates that match this subquery's return type context
		// This is a heuristic: if the subquery is a scalar subquery and the plan has PAC aggregates,
		// assume this subquery contains PAC aggregates
		if (subquery_expr.subquery_type == SubqueryType::SCALAR) {
			// Search all children of plan_root for any PAC aggregates
			// This catches cases where the subquery plan was flattened into the main plan
			std::function<string(LogicalOperator *)> searchForPacAggregate = [&](LogicalOperator *op) -> string {
				if (!op) {
					return "";
				}
				if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
					auto &agg = op->Cast<LogicalAggregate>();
					for (auto &agg_expr : agg.expressions) {
						if (agg_expr->type == ExpressionType::BOUND_AGGREGATE) {
							auto &bound_agg = agg_expr->Cast<BoundAggregateExpression>();
							if (IsPacAggregateOrCounters(bound_agg.function.name)) {
								return GetBasePacAggregateName(bound_agg.function.name);
							}
						}
					}
				}
				for (auto &child : op->children) {
					string result = searchForPacAggregate(child.get());
					if (!result.empty()) {
						return result;
					}
				}
				return "";
			};
			string result = searchForPacAggregate(plan_root);
			if (!result.empty()) {
				return result;
			}
		}
		return "";
	}

	// Check comparison expressions
	if (expr->type == ExpressionType::COMPARE_GREATERTHAN || expr->type == ExpressionType::COMPARE_LESSTHAN ||
	    expr->type == ExpressionType::COMPARE_GREATERTHANOREQUALTO ||
	    expr->type == ExpressionType::COMPARE_LESSTHANOREQUALTO || expr->type == ExpressionType::COMPARE_EQUAL ||
	    expr->type == ExpressionType::COMPARE_NOTEQUAL) {
		auto &comp_expr = expr->Cast<BoundComparisonExpression>();
		string left_result = FindPacAggregateInExpressionWithPlan(comp_expr.left.get(), plan_root);
		if (!left_result.empty()) {
			return left_result;
		}
		string right_result = FindPacAggregateInExpressionWithPlan(comp_expr.right.get(), plan_root);
		if (!right_result.empty()) {
			return right_result;
		}
		return "";
	}

	// Check operator expressions (e.g., arithmetic, cast, coalesce)
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_OPERATOR) {
		auto &op_expr = expr->Cast<BoundOperatorExpression>();
		for (auto &child : op_expr.children) {
			string result = FindPacAggregateInExpressionWithPlan(child.get(), plan_root);
			if (!result.empty()) {
				return result;
			}
		}
		return "";
	}

	// Check constant expressions - no children, just return empty
	if (expr->type == ExpressionType::VALUE_CONSTANT) {
		return "";
	}

	// Check cast expressions
	if (expr->type == ExpressionType::OPERATOR_CAST) {
		auto &cast_expr = expr->Cast<BoundCastExpression>();
		return FindPacAggregateInExpressionWithPlan(cast_expr.child.get(), plan_root);
	}

	// For any other expression types, return empty to be safe
	return "";
}

// Original version without plan context (for backward compatibility)
string FindPacAggregateInExpression(Expression *expr) {
	return FindPacAggregateInExpressionWithPlan(expr, nullptr);
}

// Plan-aware version of IsComparisonWithPacAggregate
static bool IsComparisonWithPacAggregateWithPlan(Expression *expr, CategoricalPatternInfo &info,
                                                 LogicalOperator *plan_root) {
	if (!expr) {
		return false;
	}

	// Must be a comparison expression
	if (expr->type != ExpressionType::COMPARE_GREATERTHAN && expr->type != ExpressionType::COMPARE_LESSTHAN &&
	    expr->type != ExpressionType::COMPARE_GREATERTHANOREQUALTO &&
	    expr->type != ExpressionType::COMPARE_LESSTHANOREQUALTO && expr->type != ExpressionType::COMPARE_EQUAL &&
	    expr->type != ExpressionType::COMPARE_NOTEQUAL) {
		return false;
	}

	auto &comp_expr = expr->Cast<BoundComparisonExpression>();

	// Check left side for PAC aggregate (use plan-aware version)
	string left_pac = FindPacAggregateInExpressionWithPlan(comp_expr.left.get(), plan_root);
	if (!left_pac.empty()) {
		info.comparison_expr = expr;
		info.subquery_expr = comp_expr.left.get();
		info.subquery_side = 0;
		info.aggregate_name = left_pac;
		return true;
	}

	// Check right side for PAC aggregate (use plan-aware version)
	string right_pac = FindPacAggregateInExpressionWithPlan(comp_expr.right.get(), plan_root);
	if (!right_pac.empty()) {
		info.comparison_expr = expr;
		info.subquery_expr = comp_expr.right.get();
		info.subquery_side = 1;
		info.aggregate_name = right_pac;
		return true;
	}

	return false;
}

bool IsComparisonWithPacAggregate(Expression *expr, CategoricalPatternInfo &info) {
	return IsComparisonWithPacAggregateWithPlan(expr, info, nullptr);
}

// Helper to check if a filter's child is an aggregate that produces the given binding
// This detects HAVING clauses where the comparison references the immediate child aggregate
static bool IsHavingClausePattern(LogicalOperator *filter_op, const ColumnBinding &binding,
                                  LogicalOperator *plan_root) {
	if (!filter_op || filter_op->children.empty()) {
		return false;
	}

	// Check if the immediate child (or through projections) is an aggregate that produces this binding
	LogicalOperator *child = filter_op->children[0].get();

	// Skip projections to find the aggregate
	while (child && child->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		if (child->children.empty()) {
			break;
		}
		child = child->children[0].get();
	}

	if (child && child->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &agg = child->Cast<LogicalAggregate>();
		// Check if this aggregate produces the binding we're comparing against
		if (binding.table_index == agg.aggregate_index || binding.table_index == agg.group_index) {
			return true;
		}
	}

	return false;
}

// Helper to trace a binding through projections to find the original aggregate binding
static ColumnBinding TraceBindingThroughProjections(const ColumnBinding &binding, LogicalOperator *plan_root) {
	auto *source_op = FindOperatorByTableIndex(plan_root, binding.table_index);
	if (!source_op) {
		return binding;
	}

	if (source_op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = source_op->Cast<LogicalProjection>();
		if (binding.column_index < proj.expressions.size()) {
			auto &expr = proj.expressions[binding.column_index];
			if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
				auto &col_ref = expr->Cast<BoundColumnRefExpression>();
				return TraceBindingThroughProjections(col_ref.binding, plan_root);
			}
			// For functions like 0.5 * agg_result, trace through the function's children
			if (expr->type == ExpressionType::BOUND_FUNCTION) {
				auto &func_expr = expr->Cast<BoundFunctionExpression>();
				for (auto &child : func_expr.children) {
					if (child->type == ExpressionType::BOUND_COLUMN_REF) {
						auto &col_ref = child->Cast<BoundColumnRefExpression>();
						auto traced = TraceBindingThroughProjections(col_ref.binding, plan_root);
						// If we found a different binding, return it
						if (traced.table_index != col_ref.binding.table_index) {
							return traced;
						}
						return col_ref.binding;
					}
				}
			}
		}
	}

	return binding;
}

// Recursively search the plan for categorical patterns (plan-aware version)
// Now detects ANY filter expression containing a PAC aggregate, not just comparisons
static void FindCategoricalPatternsInOperatorWithPlan(LogicalOperator *op, LogicalOperator *plan_root,
                                                      vector<CategoricalPatternInfo> &patterns, bool inside_aggregate) {
	if (!op) {
		return;
	}

	// Track if we're entering an aggregate
	bool now_inside_aggregate = inside_aggregate || (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY);

	// Check filter expressions - detect ANY boolean expression containing a PAC aggregate
	if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
		auto &filter = op->Cast<LogicalFilter>();
		for (idx_t i = 0; i < filter.expressions.size(); i++) {
			auto &filter_expr = filter.expressions[i];

			// Find ALL PAC aggregate bindings in this expression (not just single)
			auto pac_bindings = FindAllPacBindingsInExpression(filter_expr.get(), plan_root);
			if (pac_bindings.empty()) {
				continue;
			}

			// NOTE: We previously had a check "if (now_inside_aggregate) continue;" here,
			// but that was incorrect. It blocked categorical patterns when the outer query
			// has an aggregate (e.g., Q17: sum(l_extendedprice) with a subquery avg comparison).
			// The IsHavingClausePattern check below correctly distinguishes:
			// - HAVING: comparison uses the immediate parent aggregate
			// - Categorical: comparison uses a subquery aggregate

			// Check if ANY of the bindings is NOT a HAVING clause aggregate
			// If at least one binding is from a subquery (not HAVING), this is categorical
			bool has_non_having_binding = false;
			ColumnBinding first_non_having_binding;
			string first_aggregate_name;

			for (auto &binding_info : pac_bindings) {
				ColumnBinding traced_binding = TraceBindingThroughProjections(binding_info.binding, plan_root);
				bool is_having = IsHavingClausePattern(op, traced_binding, plan_root);
				if (!is_having) {
					has_non_having_binding = true;
					if (first_aggregate_name.empty()) {
						first_non_having_binding = binding_info.binding;
						first_aggregate_name = binding_info.aggregate_name;
					}
				}
			}

			if (has_non_having_binding) {
				CategoricalPatternInfo info;
				info.parent_op = op;
				info.expr_index = i;
				info.pac_binding = first_non_having_binding;
				info.has_pac_binding = true;
				info.aggregate_name = first_aggregate_name;
				// For backward compatibility, also try to identify comparison structure
				IsComparisonWithPacAggregateWithPlan(filter_expr.get(), info, plan_root);

				// Check if this binding goes through a scalar subquery wrapper
				info.scalar_wrapper_op = FindScalarWrapperForBinding(first_non_having_binding, plan_root);

				// Capture original return type BEFORE conversion to LIST<DOUBLE>
				// This is needed for the double-lambda rewrite to cast elements back
				if (info.subquery_expr) {
					info.original_return_type = info.subquery_expr->return_type;
				} else {
					// Try to get the type from the aggregate directly
					ColumnBinding traced = TraceBindingThroughProjections(first_non_having_binding, plan_root);
					auto *agg_op = FindAggregateForBinding(traced, plan_root);
					if (agg_op && traced.column_index < agg_op->expressions.size()) {
						info.original_return_type = agg_op->expressions[traced.column_index]->return_type;
					}
				}
#if PAC_DEBUG
				PAC_DEBUG_PRINT("Captured original_return_type: " + info.original_return_type.ToString());
				PAC_DEBUG_PRINT("Found categorical pattern with " + std::to_string(pac_bindings.size()) +
				                " PAC binding(s)" + (info.scalar_wrapper_op ? " (has scalar wrapper)" : ""));
#endif
				patterns.push_back(info);
			}
		}
	}

	// Check join conditions (for semi/anti/mark joins with subqueries)
	if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	    op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		auto &join = op->Cast<LogicalComparisonJoin>();
		PAC_DEBUG_PRINT("DEBUG: Checking " + LogicalOperatorToString(op->type) + " with " +
		                std::to_string(join.conditions.size()) +
		                " conditions, join_type=" + JoinTypeToString(join.join_type));
		for (idx_t i = 0; i < join.conditions.size(); i++) {
			auto &cond = join.conditions[i];
			PAC_DEBUG_PRINT("  Cond " + std::to_string(i) + ": " + ExpressionTypeToString(cond.comparison));
			PAC_DEBUG_PRINT("    Left type: " + ExpressionTypeToString(cond.left->type) + " = " +
			                cond.left->ToString());
			PAC_DEBUG_PRINT("    Right type: " + ExpressionTypeToString(cond.right->type) + " = " +
			                cond.right->ToString());
			// Check if comparison involves PAC aggregate (use plan-aware version)
			CategoricalPatternInfo info;
			string left_pac = FindPacAggregateInExpressionWithPlan(cond.left.get(), plan_root);
			string right_pac = FindPacAggregateInExpressionWithPlan(cond.right.get(), plan_root);
			PAC_DEBUG_PRINT("    Left PAC: '" + left_pac + "', Right PAC: '" + right_pac + "'");

			// NOTE: Unlike FILTER expressions, COMPARISON_JOIN conditions cannot be HAVING clauses.
			// HAVING filters are always FILTER operators, not join conditions.
			// So we don't need the now_inside_aggregate check here - any PAC aggregate
			// in a join condition is a categorical pattern (correlated subquery).
			if (!left_pac.empty()) {
				info.comparison_expr = nullptr; // Join conditions are handled differently
				info.parent_op = op;
				info.expr_index = i;
				info.subquery_expr = cond.left.get();
				info.subquery_side = 0;
				info.aggregate_name = left_pac;
				// Check for scalar wrapper if left is a column ref
				if (cond.left->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = cond.left->Cast<BoundColumnRefExpression>();
					info.scalar_wrapper_op = FindScalarWrapperForBinding(col_ref.binding, plan_root);
				}
				patterns.push_back(info);
			} else if (!right_pac.empty()) {
				info.comparison_expr = nullptr;
				info.parent_op = op;
				info.expr_index = i;
				info.subquery_expr = cond.right.get();
				info.subquery_side = 1;
				info.aggregate_name = right_pac;
				// Check for scalar wrapper if right is a column ref
				if (cond.right->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = cond.right->Cast<BoundColumnRefExpression>();
					info.scalar_wrapper_op = FindScalarWrapperForBinding(col_ref.binding, plan_root);
				}
				patterns.push_back(info);
			}
		}
	}

	// Check projection expressions for arithmetic involving multiple PAC aggregates
	// This handles cases like Q08: sum(CASE...)/sum(volume) in SELECT list
	// NOTE: Only check projections if we haven't already found filter/join patterns,
	// because those patterns will handle the projections via RewriteProjectionsWithCounters.
	// We only want standalone projection patterns (no filter/join categorical patterns).
	if (!inside_aggregate && patterns.empty() && op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = op->Cast<LogicalProjection>();
		for (idx_t i = 0; i < proj.expressions.size(); i++) {
			auto &expr = proj.expressions[i];

			// Skip simple column references - they don't need rewriting
			if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
				continue;
			}

			// Find ALL PAC aggregate bindings in this expression
			auto pac_bindings = FindAllPacBindingsInExpression(expr.get(), plan_root);
			if (pac_bindings.empty()) {
				continue;
			}

			// Check if this expression has arithmetic with PAC aggregates
			// - Multiple PAC bindings (e.g., pac_sum(...) / pac_sum(...))
			// - Or single PAC binding with arithmetic (e.g., pac_sum(...) * 0.5)
			bool is_arithmetic_with_pac = false;
			if (pac_bindings.size() >= 2) {
				// Multiple PAC aggregates - definitely needs lambda rewrite
				is_arithmetic_with_pac = true;
			} else if (pac_bindings.size() == 1) {
				// Single PAC aggregate - check if it's in an arithmetic expression
				// (not just a column ref or simple cast)
				if (expr->type != ExpressionType::BOUND_COLUMN_REF && expr->type != ExpressionType::OPERATOR_CAST) {
					is_arithmetic_with_pac = true;
				} else if (expr->type == ExpressionType::OPERATOR_CAST) {
					// Check if cast contains arithmetic
					auto &cast_expr = expr->Cast<BoundCastExpression>();
					if (cast_expr.child->type != ExpressionType::BOUND_COLUMN_REF) {
						is_arithmetic_with_pac = true;
					}
				}
			}

			if (is_arithmetic_with_pac) {
				// Only create pattern if the expression result is numerical
				// (we can't apply pac_noised to non-numerical types)
				if (!IsNumericalType(expr->return_type)) {
					continue;
				}

				CategoricalPatternInfo info;
				info.parent_op = op;
				info.expr_index = i;
				info.pac_binding = pac_bindings[0].binding;
				info.has_pac_binding = true;
				info.aggregate_name = pac_bindings[0].aggregate_name;
				info.original_return_type = expr->return_type;
				info.comparison_expr = nullptr;
				info.subquery_expr = nullptr;
				info.scalar_wrapper_op = nullptr;
#if PAC_DEBUG
				PAC_DEBUG_PRINT("Found projection-based categorical pattern in expr " + std::to_string(i) + " with " +
				                std::to_string(pac_bindings.size()) + " PAC binding(s)");
				PAC_DEBUG_PRINT("  Expression: " + expr->ToString().substr(0, 80));
#endif
				patterns.push_back(info);
			}
		}
	}

	// Recurse into children
	for (auto &child : op->children) {
		FindCategoricalPatternsInOperatorWithPlan(child.get(), plan_root, patterns, now_inside_aggregate);
	}
}

bool IsCategoricalQuery(unique_ptr<LogicalOperator> &plan, vector<CategoricalPatternInfo> &patterns) {
	patterns.clear();
	// Use plan-aware version, passing plan root for column binding tracing
	FindCategoricalPatternsInOperatorWithPlan(plan.get(), plan.get(), patterns, false);
	return !patterns.empty();
}

// Helper to find the aggregate operator that produces a given column binding
static LogicalAggregate *FindAggregateForBinding(const ColumnBinding &binding, LogicalOperator *plan_root) {
	auto *source_op = FindOperatorByTableIndex(plan_root, binding.table_index);
#if PAC_DEBUG
	PAC_DEBUG_PRINT("FindAggregateForBinding: binding=[" + std::to_string(binding.table_index) + "." +
	                std::to_string(binding.column_index) +
	                "], source_op=" + (source_op ? LogicalOperatorToString(source_op->type) : "null"));
#endif
	if (!source_op) {
		return nullptr;
	}

	if (source_op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
#if PAC_DEBUG
		PAC_DEBUG_PRINT("  Found aggregate directly!");
#endif
		return &source_op->Cast<LogicalAggregate>();
	}

	// If it's a projection, trace through to find the aggregate
	if (source_op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = source_op->Cast<LogicalProjection>();
		if (binding.column_index < proj.expressions.size()) {
			auto &expr = proj.expressions[binding.column_index];
#if PAC_DEBUG
			PAC_DEBUG_PRINT("  Projection expr type: " + ExpressionTypeToString(expr->type));
#endif
			// Check if this expression is a column reference to an aggregate
			if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
				auto &col_ref = expr->Cast<BoundColumnRefExpression>();
				return FindAggregateForBinding(col_ref.binding, plan_root);
			}
			// Check if it's a cast expression (e.g., CAST(agg_result AS DOUBLE) or CAST(coalesce(agg, 0) AS INTEGER))
			if (expr->type == ExpressionType::OPERATOR_CAST) {
				auto &cast_expr = expr->Cast<BoundCastExpression>();
#if PAC_DEBUG
				PAC_DEBUG_PRINT("  Cast child type: " + ExpressionTypeToString(cast_expr.child->type));
#endif
				if (cast_expr.child->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = cast_expr.child->Cast<BoundColumnRefExpression>();
					return FindAggregateForBinding(col_ref.binding, plan_root);
				}
				// Also check for CAST(function(...)) or CAST(coalesce(...)) where it contains aggregate reference
				// Handle both BOUND_FUNCTION and OPERATOR_COALESCE
				if (cast_expr.child->type == ExpressionType::BOUND_FUNCTION ||
				    cast_expr.child->type == ExpressionType::OPERATOR_COALESCE) {
					// Get children based on expression type
					vector<Expression *> children;
					if (cast_expr.child->type == ExpressionType::BOUND_FUNCTION) {
						auto &func_expr = cast_expr.child->Cast<BoundFunctionExpression>();
						for (auto &c : func_expr.children) {
							children.push_back(c.get());
						}
					} else {
						auto &op_expr = cast_expr.child->Cast<BoundOperatorExpression>();
						for (auto &c : op_expr.children) {
							children.push_back(c.get());
						}
					}
#if PAC_DEBUG
					PAC_DEBUG_PRINT("  Checking " + std::to_string(children.size()) + " children for aggregate ref");
#endif
					for (auto *child : children) {
#if PAC_DEBUG
						PAC_DEBUG_PRINT("    Child type: " + ExpressionTypeToString(child->type));
#endif
						if (child->type == ExpressionType::BOUND_COLUMN_REF) {
							auto &col_ref = child->Cast<BoundColumnRefExpression>();
							auto *agg = FindAggregateForBinding(col_ref.binding, plan_root);
							if (agg) {
								return agg;
							}
						}
						// Also check for aggregate expressions directly (e.g., COALESCE(sum(...), 0))
						if (child->type == ExpressionType::BOUND_AGGREGATE) {
							auto &agg_expr = child->Cast<BoundAggregateExpression>();
#if PAC_DEBUG
							PAC_DEBUG_PRINT("    Found BOUND_AGGREGATE: " + agg_expr.function.name);
#endif
							if (IsPacAggregateOrCounters(agg_expr.function.name)) {
								// The aggregate is embedded in the projection expression
								// Search the children of the source operator (the projection) for the aggregate
								for (auto &source_child : source_op->children) {
									auto *agg = FindFirstAggregateInSubtree(source_child.get());
									if (agg) {
										return agg;
									}
								}
							}
						}
					}
				}
			}
			// Check if it's a function that references an aggregate (e.g., 0.5 * agg_result)
			if (expr->type == ExpressionType::BOUND_FUNCTION) {
				auto &func_expr = expr->Cast<BoundFunctionExpression>();
				for (auto &child : func_expr.children) {
					if (child->type == ExpressionType::BOUND_COLUMN_REF) {
						auto &col_ref = child->Cast<BoundColumnRefExpression>();
						auto *agg = FindAggregateForBinding(col_ref.binding, plan_root);
						if (agg) {
							return agg;
						}
					}
					// Also check for CAST(col_ref) inside functions (e.g., CAST(agg_result AS DOUBLE) / 5.0)
					if (child->type == ExpressionType::OPERATOR_CAST) {
						auto &cast_expr = child->Cast<BoundCastExpression>();
						if (cast_expr.child->type == ExpressionType::BOUND_COLUMN_REF) {
							auto &col_ref = cast_expr.child->Cast<BoundColumnRefExpression>();
							auto *agg = FindAggregateForBinding(col_ref.binding, plan_root);
							if (agg) {
								return agg;
							}
						}
					}
				}
			}
		}
	}

	return nullptr;
}

// Helper to find aggregate operators that are part of categorical patterns
static void FindCategoricalAggregates(LogicalOperator *plan_root, const vector<CategoricalPatternInfo> &patterns,
                                      unordered_set<LogicalAggregate *> &categorical_aggregates) {
	for (auto &pattern : patterns) {
		if (!pattern.parent_op) {
			continue;
		}

		// For filter patterns, find ALL PAC aggregates in the filter expression
		// This ensures that when comparing two PAC aggregates, BOTH get converted to counters
		if (pattern.parent_op->type == LogicalOperatorType::LOGICAL_FILTER) {
			auto &filter = pattern.parent_op->Cast<LogicalFilter>();
			if (pattern.expr_index < filter.expressions.size()) {
				auto &filter_expr = filter.expressions[pattern.expr_index];
				auto all_bindings = FindAllPacBindingsInExpression(filter_expr.get(), plan_root);
				for (auto &binding_info : all_bindings) {
					auto *agg = FindAggregateForBinding(binding_info.binding, plan_root);
					if (agg) {
						categorical_aggregates.insert(agg);
					}
				}
				continue;
			}
		}

		// For COMPARISON_JOIN patterns, find ALL PAC aggregates in BOTH sides of the join condition
		// This ensures that both sides of the comparison get converted to counters
		if (pattern.parent_op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
		    pattern.parent_op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			auto &join = pattern.parent_op->Cast<LogicalComparisonJoin>();
			if (pattern.expr_index < join.conditions.size()) {
				auto &cond = join.conditions[pattern.expr_index];
				// Find all PAC bindings in left side
				auto left_bindings = FindAllPacBindingsInExpression(cond.left.get(), plan_root);
				for (auto &binding_info : left_bindings) {
					auto *agg = FindAggregateForBinding(binding_info.binding, plan_root);
					if (agg) {
						categorical_aggregates.insert(agg);
					}
				}
				// Find all PAC bindings in right side
				auto right_bindings = FindAllPacBindingsInExpression(cond.right.get(), plan_root);
				for (auto &binding_info : right_bindings) {
					auto *agg = FindAggregateForBinding(binding_info.binding, plan_root);
					if (agg) {
						categorical_aggregates.insert(agg);
					}
				}
				continue;
			}
		}

		// Try subquery_expr first (for backward compatibility with comparison patterns)
		if (pattern.subquery_expr && pattern.subquery_expr->type == ExpressionType::BOUND_COLUMN_REF) {
			auto &col_ref = pattern.subquery_expr->Cast<BoundColumnRefExpression>();
			auto *agg = FindAggregateForBinding(col_ref.binding, plan_root);
			if (agg) {
				categorical_aggregates.insert(agg);
				continue;
			}
		}

		// Fall back to pac_binding (for general expressions like `sum * 0.5 > 40`)
		if (pattern.has_pac_binding) {
			auto *agg = FindAggregateForBinding(pattern.pac_binding, plan_root);
			if (agg) {
				categorical_aggregates.insert(agg);
			}
		}
	}
}

// Helper to replace PAC aggregate with counters variant in a logical plan subtree
// Only replaces aggregates that are in the categorical_aggregates set (if provided)
// If categorical_aggregates is null, replaces all PAC aggregates (legacy behavior)
static void ReplacePacAggregatesWithCounters(LogicalOperator *op, ClientContext &context,
                                             const unordered_set<LogicalAggregate *> *categorical_aggregates) {
	if (!op) {
		return;
	}

	// If this is an aggregate operator, check for PAC aggregates
	if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &agg = op->Cast<LogicalAggregate>();

		// If we have a set of target aggregates, only process if this aggregate is in the set
		if (categorical_aggregates && categorical_aggregates->find(&agg) == categorical_aggregates->end()) {
			// This aggregate is not part of a categorical pattern, skip it
			// But still recurse into children
			for (auto &child : op->children) {
				ReplacePacAggregatesWithCounters(child.get(), context, categorical_aggregates);
			}
			return;
		}

		for (idx_t i = 0; i < agg.expressions.size(); i++) {
			auto &agg_expr = agg.expressions[i];
			if (agg_expr->type == ExpressionType::BOUND_AGGREGATE) {
				auto &bound_agg = agg_expr->Cast<BoundAggregateExpression>();
				if (IsPacAggregate(bound_agg.function.name)) {
					// Get the counters variant name
					string counters_name = GetCountersVariant(bound_agg.function.name);

					// Look up the counters function from the catalog
					auto &catalog = Catalog::GetSystemCatalog(context);
					auto &func_entry =
					    catalog.GetEntry<AggregateFunctionCatalogEntry>(context, DEFAULT_SCHEMA, counters_name);

					// Get the argument types from the existing children
					vector<LogicalType> arg_types;
					for (auto &child : bound_agg.children) {
						arg_types.push_back(child->return_type);
					}

					// Find the best matching function
					ErrorData error;
					FunctionBinder function_binder(context);
					auto best_function =
					    function_binder.BindFunction(counters_name, func_entry.functions, arg_types, error);

					if (!best_function.IsValid()) {
#if PAC_DEBUG
						PAC_DEBUG_PRINT("Warning: Could not bind " + counters_name + ": " + error.Message());
#endif
						// Fall back to just changing the name and return type
						bound_agg.function.name = counters_name;
						bound_agg.function.return_type = LogicalType::LIST(LogicalType::DOUBLE);
						agg_expr->return_type = LogicalType::LIST(LogicalType::DOUBLE);
					} else {
						// Get the bound function
						AggregateFunction counters_func =
						    func_entry.functions.GetFunctionByOffset(best_function.GetIndex());

						// Move children out of the old aggregate
						vector<unique_ptr<Expression>> children;
						for (auto &child : bound_agg.children) {
							children.push_back(child->Copy());
						}

						// Bind the new aggregate expression
						auto new_aggr = function_binder.BindAggregateFunction(
						    counters_func, std::move(children), nullptr,
						    bound_agg.IsDistinct() ? AggregateType::DISTINCT : AggregateType::NON_DISTINCT);

#if PAC_DEBUG
						PAC_DEBUG_PRINT("Rebound aggregate to " + counters_name + " with return type " +
						                new_aggr->return_type.ToString());
#endif

						// Replace the expression
						agg.expressions[i] = std::move(new_aggr);
					}

					// Update the aggregate operator's types vector
					idx_t types_index = agg.groups.size() + i;
					if (types_index < agg.types.size()) {
						agg.types[types_index] = LogicalType::LIST(LogicalType::DOUBLE);
#if PAC_DEBUG
						PAC_DEBUG_PRINT("Updated aggregate types[" + std::to_string(types_index) +
						                "] to LIST<DOUBLE> for " + counters_name);
#endif
					}
				}
			}
		}
	}

	// Recurse into children
	for (auto &child : op->children) {
		ReplacePacAggregatesWithCounters(child.get(), context, categorical_aggregates);
	}
}

// Legacy function (kept for backward compatibility in expression-based replacement)
static void ReplacePacAggregatesInPlan(LogicalOperator *op, OptimizerExtensionInput *input) {
	ReplacePacAggregatesWithCounters(op, input->context, nullptr);
}

// Map standard aggregate names to their pac_*_list equivalents
static string GetListAggregateVariant(const string &name) {
	if (name == "sum") {
		return "pac_sum_list";
	}
	if (name == "avg") {
		return "pac_avg_list";
	}
	if (name == "count") {
		return "pac_count_list";
	}
	if (name == "min") {
		return "pac_min_list";
	}
	if (name == "max") {
		return "pac_max_list";
	}
	// Note: first/any_value are used by scalar subquery handlers.
	// We don't convert them - instead we strip the scalar subquery wrapper entirely.
	return "";
}

// Check if an expression traces back to a PAC _counters aggregate
static bool TracesPacCountersAggregate(Expression *expr, LogicalOperator *plan_root) {
	if (!expr || !plan_root) {
		return false;
	}

	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();
		string pac_name = TracePacAggregateFromBinding(col_ref.binding, plan_root);
		// TracePacAggregateFromBinding uses IsPacAggregateOrCounters, so it will find _counters too
		return !pac_name.empty();
	}

	// Check children recursively
	bool found = false;
	ExpressionIterator::EnumerateChildren(*expr, [&](Expression &child) {
		if (TracesPacCountersAggregate(&child, plan_root)) {
			found = true;
		}
	});
	return found;
}

// Replace standard aggregates that operate on PAC counter results
// with pac_*_list variants that aggregate element-wise
static void ReplaceAggregatesOverCounters(LogicalOperator *op, ClientContext &context, LogicalOperator *plan_root) {
	if (!op) {
		return;
	}

	if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &agg = op->Cast<LogicalAggregate>();
		for (idx_t i = 0; i < agg.expressions.size(); i++) {
			auto &agg_expr = agg.expressions[i];
			if (agg_expr->type != ExpressionType::BOUND_AGGREGATE) {
				continue;
			}

			auto &bound_agg = agg_expr->Cast<BoundAggregateExpression>();
			if (bound_agg.children.empty()) {
				continue;
			}

			// Check if input traces to a PAC _counters aggregate
			bool traces_counters = TracesPacCountersAggregate(bound_agg.children[0].get(), plan_root);
			PAC_DEBUG_PRINT("  Checking aggregate " + bound_agg.function.name +
			                " traces_counters=" + (traces_counters ? "true" : "false"));
			if (!traces_counters) {
				continue;
			}

			string list_variant = GetListAggregateVariant(bound_agg.function.name);
			if (list_variant.empty()) {
				continue;
			}

			PAC_DEBUG_PRINT("Converting " + bound_agg.function.name + " over LIST<DOUBLE> to " + list_variant);

			// Rebind to the list variant
			auto &catalog = Catalog::GetSystemCatalog(context);
			auto &func_entry = catalog.GetEntry<AggregateFunctionCatalogEntry>(context, DEFAULT_SCHEMA, list_variant);

			// pac_*_list functions expect LIST<DOUBLE>
			vector<LogicalType> arg_types = {LogicalType::LIST(LogicalType::DOUBLE)};

			ErrorData error;
			FunctionBinder function_binder(context);
			auto best_function = function_binder.BindFunction(list_variant, func_entry.functions, arg_types, error);

			if (!best_function.IsValid()) {
				PAC_DEBUG_PRINT("Warning: Could not bind " + list_variant + ": " + error.Message());
				continue;
			}

			AggregateFunction list_func = func_entry.functions.GetFunctionByOffset(best_function.GetIndex());

			vector<unique_ptr<Expression>> children;
			for (auto &child : bound_agg.children) {
				auto child_copy = child->Copy();
				// Update the child's type to LIST<DOUBLE> since it now comes from _counters
				if (child_copy->type == ExpressionType::BOUND_COLUMN_REF) {
					child_copy->return_type = LogicalType::LIST(LogicalType::DOUBLE);
				}
				children.push_back(std::move(child_copy));
			}

			auto new_aggr = function_binder.BindAggregateFunction(list_func, std::move(children), nullptr,
			                                                      bound_agg.IsDistinct() ? AggregateType::DISTINCT
			                                                                             : AggregateType::NON_DISTINCT);

			agg.expressions[i] = std::move(new_aggr);

			// Update types vector
			idx_t types_index = agg.groups.size() + i;
			if (types_index < agg.types.size()) {
				agg.types[types_index] = LogicalType::LIST(LogicalType::DOUBLE);
			}
		}
	}
}

// Wrapper that processes bottom-up (children first, then current operator)
// This ensures that child aggregates are converted before checking parent aggregates
static void ReplaceAggregatesOverCountersBottomUp(LogicalOperator *op, ClientContext &context,
                                                  LogicalOperator *plan_root) {
	if (!op) {
		return;
	}

	// First, process all children (bottom-up)
	for (auto &child : op->children) {
		ReplaceAggregatesOverCountersBottomUp(child.get(), context, plan_root);
	}

	// Then process this operator
	ReplaceAggregatesOverCounters(op, context, plan_root);
}

// Helper to replace PAC aggregate with counters variant in an expression tree
static void ReplacePacAggregateWithCounters(Expression *expr, OptimizerExtensionInput &input,
                                            unique_ptr<LogicalOperator> &plan) {
	if (!expr) {
		return;
	}

	// If this is a bound aggregate expression directly, replace it
	if (expr->type == ExpressionType::BOUND_AGGREGATE) {
		auto &bound_agg = expr->Cast<BoundAggregateExpression>();
		if (IsPacAggregate(bound_agg.function.name)) {
			string counters_name = GetCountersVariant(bound_agg.function.name);
			bound_agg.function.name = counters_name;
			bound_agg.return_type = LogicalType::LIST(LogicalType::DOUBLE);
		}
		return;
	}

	// If this is a subquery expression, we need to search the plan tree for the corresponding
	// logical operators and replace PAC aggregates there
	if (expr->type == ExpressionType::SUBQUERY) {
		// The subquery's logical plan is part of the main plan tree (as DELIM_JOIN children, etc.)
		// We search the entire plan for PAC aggregates and replace them
		// This is safe because we only replace pac_* with pac_*_counters
		ReplacePacAggregatesInPlan(plan.get(), &input);
		return;
	}

	// Recursively process children
	ExpressionIterator::EnumerateChildren(
	    *expr, [&](Expression &child) { ReplacePacAggregateWithCounters(&child, input, plan); });
}

// Helper to collect ALL distinct PAC aggregate bindings in an expression
// Returns the bindings in the order they were discovered
static void CollectPacBindingsInExpression(Expression *expr, LogicalOperator *plan_root,
                                           vector<PacBindingInfo> &bindings,
                                           unordered_map<uint64_t, idx_t> &binding_hash_to_index) {
	if (!expr || !plan_root) {
		return;
	}

	// Check if this is a column reference that traces back to a PAC aggregate
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();
		string pac_name = TracePacAggregateFromBinding(col_ref.binding, plan_root);
		if (!pac_name.empty()) {
			// Hash the binding for uniqueness check
			uint64_t binding_hash = HashBinding(col_ref.binding);
			if (binding_hash_to_index.find(binding_hash) == binding_hash_to_index.end()) {
				PacBindingInfo info;
				info.binding = col_ref.binding;
				info.aggregate_name = pac_name;
				info.original_type = col_ref.return_type; // Capture before counters conversion
				info.index = bindings.size();
				binding_hash_to_index[binding_hash] = info.index;
				bindings.push_back(info);
			}
		}
	}

	// Recursively check children
	ExpressionIterator::EnumerateChildren(*expr, [&](Expression &child) {
		CollectPacBindingsInExpression(&child, plan_root, bindings, binding_hash_to_index);
	});
}

// Helper to count distinct PAC aggregate bindings in an expression
// Returns the count of unique PAC aggregate bindings found
// Optionally returns the binding and aggregate name if exactly one is found
static idx_t CountPacAggregateBindingsInExpression(Expression *expr, LogicalOperator *plan_root,
                                                   unordered_set<uint64_t> &pac_binding_hashes,
                                                   ColumnBinding *out_binding = nullptr,
                                                   string *out_aggregate_name = nullptr) {
	if (!expr || !plan_root) {
		return 0;
	}

	idx_t count = 0;

	// Check if this is a column reference that traces back to a PAC aggregate
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();
		string pac_name = TracePacAggregateFromBinding(col_ref.binding, plan_root);
		if (!pac_name.empty()) {
			// Hash the binding for uniqueness check
			uint64_t binding_hash = HashBinding(col_ref.binding);
			if (pac_binding_hashes.find(binding_hash) == pac_binding_hashes.end()) {
				pac_binding_hashes.insert(binding_hash);
				count = 1;
				// Store the first binding found
				if (out_binding && pac_binding_hashes.size() == 1) {
					*out_binding = col_ref.binding;
				}
				if (out_aggregate_name && pac_binding_hashes.size() == 1) {
					*out_aggregate_name = pac_name;
				}
			}
		}
	}

	// Recursively check children
	ExpressionIterator::EnumerateChildren(*expr, [&](Expression &child) {
		count += CountPacAggregateBindingsInExpression(&child, plan_root, pac_binding_hashes, out_binding,
		                                               out_aggregate_name);
	});

	return count;
}

// Find all PAC aggregate bindings in an expression
static vector<PacBindingInfo> FindAllPacBindingsInExpression(Expression *expr, LogicalOperator *plan_root) {
	vector<PacBindingInfo> bindings;
	unordered_map<uint64_t, idx_t> binding_hash_to_index;
	CollectPacBindingsInExpression(expr, plan_root, bindings, binding_hash_to_index);
	return bindings;
}

//==============================================================================
// Lambda-based Categorical Rewrite Helpers
//==============================================================================

// Build an operator expression from cloned children, handling COALESCE type coercion
// For COALESCE, all children must have compatible types. If the first child changed to DOUBLE
// (typical when PAC binding becomes a lambda element), cast other children to match.
static unique_ptr<Expression> BuildClonedOperatorExpression(Expression *original_expr,
                                                            vector<unique_ptr<Expression>> new_children) {
	auto &op = original_expr->Cast<BoundOperatorExpression>();
	LogicalType result_type = op.return_type;

	// For COALESCE with mismatched child types, cast all to the first child's type
	if (original_expr->type == ExpressionType::OPERATOR_COALESCE && new_children.size() > 1) {
		LogicalType first_type = new_children[0]->return_type;
		bool types_mismatch = false;
		for (idx_t i = 1; i < new_children.size(); i++) {
			if (new_children[i]->return_type != first_type) {
				types_mismatch = true;
				break;
			}
		}
		if (types_mismatch) {
			for (auto &child : new_children) {
				if (child->return_type != first_type) {
					child = BoundCastExpression::AddDefaultCastToType(std::move(child), first_type);
				}
			}
			result_type = first_type;
		}
	}

	auto result = make_uniq<BoundOperatorExpression>(original_expr->type, result_type);
	for (auto &child : new_children) {
		result->children.push_back(std::move(child));
	}
	return result;
}

// Check if a CASE expression is DuckDB's scalar subquery wrapper pattern:
// CASE WHEN ... THEN error(...) ELSE value END
static bool IsScalarSubqueryWrapper(const BoundCaseExpression &case_expr) {
	if (case_expr.case_checks.size() != 1 || !case_expr.else_expr) {
		return false;
	}
	auto &check = case_expr.case_checks[0];
	if (check.then_expr && check.then_expr->type == ExpressionType::BOUND_FUNCTION) {
		auto &func = check.then_expr->Cast<BoundFunctionExpression>();
		return func.function.name == "error";
	}
	return false;
}

// Capture a non-PAC column reference for use in a lambda
// Returns the BoundReferenceExpression index (1 + capture_idx since index 0 is the element)
static unique_ptr<Expression> CaptureColumnRef(const BoundColumnRefExpression &col_ref,
                                               vector<unique_ptr<Expression>> &captures,
                                               unordered_map<uint64_t, idx_t> &capture_map) {
	uint64_t hash = HashBinding(col_ref.binding);
	idx_t capture_idx;

	auto it = capture_map.find(hash);
	if (it != capture_map.end()) {
		capture_idx = it->second;
	} else {
		capture_idx = captures.size();
		capture_map[hash] = capture_idx;
		captures.push_back(col_ref.Copy());
	}

	return make_uniq<BoundReferenceExpression>(col_ref.alias, col_ref.return_type, 1 + capture_idx);
}

// Build a struct_extract_at expression to extract a field from a struct element
// field_idx is 0-based internally, but struct_extract_at needs 1-based argument
static unique_ptr<Expression> BuildStructFieldExtract(const LogicalType &struct_type, idx_t field_idx,
                                                      const string &field_name) {
	auto elem_ref = make_uniq<BoundReferenceExpression>("elem", struct_type, 0);

	// Get the field type from the struct
	auto child_types = StructType::GetChildTypes(struct_type);
	LogicalType extract_return_type = LogicalType::DOUBLE;
	for (idx_t j = 0; j < child_types.size(); j++) {
		if (child_types[j].first == field_name) {
			extract_return_type = child_types[j].second;
			break;
		}
	}

	auto extract_func = StructExtractAtFun::GetFunction();
	auto bind_data = StructExtractAtFun::GetBindData(field_idx);

	vector<unique_ptr<Expression>> extract_children;
	extract_children.push_back(std::move(elem_ref));
	extract_children.push_back(make_uniq<BoundConstantExpression>(Value::BIGINT(static_cast<int64_t>(field_idx + 1))));

	return make_uniq<BoundFunctionExpression>(extract_return_type, extract_func, std::move(extract_children),
	                                          std::move(bind_data));
}

// Clone an expression tree for use as a lambda body
// PAC aggregate column ref becomes BoundReferenceExpression(0) - the list element
// Other column refs are captured and become BoundReferenceExpression(1+i)
// element_type: the type of the lambda parameter (index 0). With double-lambda approach,
//               the inner lambda has already cast to the correct type, so no additional casting needed.
static unique_ptr<Expression> CloneForLambdaBody(Expression *expr, const ColumnBinding &pac_binding,
                                                 vector<unique_ptr<Expression>> &captures,
                                                 unordered_map<uint64_t, idx_t> &capture_map,
                                                 LogicalOperator *plan_root, const LogicalType &element_type) {
	if (!expr) {
		return nullptr;
	}

	// Handle column references
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();

		PAC_DEBUG_PRINT("    CloneForLambdaBody: col_ref [" + std::to_string(col_ref.binding.table_index) + "." +
		                std::to_string(col_ref.binding.column_index) + "] '" + col_ref.alias + "' vs pac_binding [" +
		                std::to_string(pac_binding.table_index) + "." + std::to_string(pac_binding.column_index) + "]");

		// Check if this traces back to the PAC aggregate binding
		// We need to trace through projections to find the original source
		ColumnBinding traced = col_ref.binding;
		auto *source_op = FindOperatorByTableIndex(plan_root, traced.table_index);

		// Trace through projections to find the actual PAC aggregate binding
		while (source_op && source_op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			auto &proj = source_op->Cast<LogicalProjection>();
			if (traced.column_index < proj.expressions.size()) {
				auto &proj_expr = proj.expressions[traced.column_index];
				if (proj_expr->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &inner_ref = proj_expr->Cast<BoundColumnRefExpression>();
					traced = inner_ref.binding;
					source_op = FindOperatorByTableIndex(plan_root, traced.table_index);
				} else if (proj_expr->type == ExpressionType::BOUND_FUNCTION) {
					// Check if this function (like pac_scale_counters) references the PAC binding
					auto &func_expr = proj_expr->Cast<BoundFunctionExpression>();
					bool found_pac = false;
					for (auto &child : func_expr.children) {
						if (child->type == ExpressionType::BOUND_COLUMN_REF) {
							auto &child_ref = child->Cast<BoundColumnRefExpression>();
							if (child_ref.binding == pac_binding) {
								found_pac = true;
								break;
							}
						}
					}
					if (found_pac) {
						// This column ref leads to the PAC aggregate through a function
						// Replace with BoundReferenceExpression(0) - the list element
						// With double-lambda, element_type is already correct
						return make_uniq<BoundReferenceExpression>(col_ref.alias, element_type, 0);
					}
					break;
				} else {
					break;
				}
			} else {
				break;
			}
		}

		// Check if this is the PAC aggregate binding (direct or traced)
		if (col_ref.binding == pac_binding || traced == pac_binding) {
			PAC_DEBUG_PRINT(
			    "    CloneForLambdaBody: MATCHED! Replacing [" + std::to_string(col_ref.binding.table_index) + "." +
			    std::to_string(col_ref.binding.column_index) + "] with elem ref, type=" + element_type.ToString());
			// Replace with BoundReferenceExpression(0) - the list element
			// Use generic alias to avoid confusion with original column name
			return make_uniq<BoundReferenceExpression>("elem", element_type, 0);
		}

#if PAC_DEBUG
		PAC_DEBUG_PRINT("    CloneForLambdaBody: NO MATCH - traced to [" + std::to_string(traced.table_index) + "." +
		                std::to_string(traced.column_index) + "] - capturing");
#endif
		// Other column ref - needs to be captured
		return CaptureColumnRef(col_ref, captures, capture_map);
	}

	// Handle constant expressions
	if (expr->type == ExpressionType::VALUE_CONSTANT) {
		return expr->Copy();
	}

	// Handle cast expressions
	if (expr->type == ExpressionType::OPERATOR_CAST) {
		auto &cast = expr->Cast<BoundCastExpression>();
		auto child_clone =
		    CloneForLambdaBody(cast.child.get(), pac_binding, captures, capture_map, plan_root, element_type);

		// If the child's type already matches the target type, skip the cast
		// This is important because the PAC element has already been cast to the correct type
		// by the inner lambda, and the copied cast function expects the original input type
		if (child_clone->return_type == cast.return_type) {
			return child_clone;
		}

		// Otherwise, create a new cast with the correct function for the new child type
		return BoundCastExpression::AddDefaultCastToType(std::move(child_clone), cast.return_type);
	}

	// Handle comparison expressions
	// With double-lambda, the element type is already correct, so no need to cast comparison operands
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON) {
		auto &comp = expr->Cast<BoundComparisonExpression>();
		auto left_clone =
		    CloneForLambdaBody(comp.left.get(), pac_binding, captures, capture_map, plan_root, element_type);
		auto right_clone =
		    CloneForLambdaBody(comp.right.get(), pac_binding, captures, capture_map, plan_root, element_type);
		return make_uniq<BoundComparisonExpression>(expr->type, std::move(left_clone), std::move(right_clone));
	}

	// Handle function expressions
	if (expr->type == ExpressionType::BOUND_FUNCTION) {
		auto &func = expr->Cast<BoundFunctionExpression>();
		vector<unique_ptr<Expression>> new_children;
		for (auto &child : func.children) {
			new_children.push_back(
			    CloneForLambdaBody(child.get(), pac_binding, captures, capture_map, plan_root, element_type));
		}
		auto result = make_uniq<BoundFunctionExpression>(func.return_type, func.function, std::move(new_children),
		                                                 func.bind_info ? func.bind_info->Copy() : nullptr);
		return result;
	}

	// Handle operator expressions (e.g., AND, OR, NOT, arithmetic, COALESCE)
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_OPERATOR) {
		auto &op = expr->Cast<BoundOperatorExpression>();
		vector<unique_ptr<Expression>> new_children;
		for (auto &child : op.children) {
			new_children.push_back(
			    CloneForLambdaBody(child.get(), pac_binding, captures, capture_map, plan_root, element_type));
		}
		return BuildClonedOperatorExpression(expr, std::move(new_children));
	}

	// Handle CASE expressions
	if (expr->type == ExpressionType::CASE_EXPR) {
		auto &case_expr = expr->Cast<BoundCaseExpression>();

		if (IsScalarSubqueryWrapper(case_expr)) {
			PAC_DEBUG_PRINT("    CloneForLambdaBody: Stripping DuckDB scalar subquery wrapper, using ELSE branch");
			return CloneForLambdaBody(case_expr.else_expr.get(), pac_binding, captures, capture_map, plan_root,
			                          element_type);
		}

		// Regular CASE - recurse into all branches
		// Start with original return type, will update based on cloned branches
		auto result = make_uniq<BoundCaseExpression>(case_expr.return_type);

		for (auto &check : case_expr.case_checks) {
			BoundCaseCheck new_check;
			new_check.when_expr =
			    CloneForLambdaBody(check.when_expr.get(), pac_binding, captures, capture_map, plan_root, element_type);
			new_check.then_expr =
			    CloneForLambdaBody(check.then_expr.get(), pac_binding, captures, capture_map, plan_root, element_type);
			result->case_checks.push_back(std::move(new_check));
		}

		if (case_expr.else_expr) {
			result->else_expr = CloneForLambdaBody(case_expr.else_expr.get(), pac_binding, captures, capture_map,
			                                       plan_root, element_type);
			// Update return type to match ELSE branch (the PAC element type)
			result->return_type = result->else_expr->return_type;
		}

		// Cast THEN branches to match the return type if needed
		for (auto &check : result->case_checks) {
			if (check.then_expr && check.then_expr->return_type != result->return_type) {
				check.then_expr =
				    BoundCastExpression::AddDefaultCastToType(std::move(check.then_expr), result->return_type);
			}
		}

		return result;
	}

	// For any other expression type, try to copy it directly
	// This handles constants and other leaf nodes
	return expr->Copy();
}

// Build a BoundLambdaExpression from a lambda body and captures
static unique_ptr<Expression> BuildPacLambda(unique_ptr<Expression> lambda_body,
                                             vector<unique_ptr<Expression>> captures) {
	// BoundLambdaExpression requirements:
	// - type: ExpressionType::LAMBDA
	// - return_type: LogicalType::LAMBDA (for function binding)
	// - lambda_expr: the transformed body
	// - parameter_count: 1 (just the element from the list)
	auto lambda =
	    make_uniq<BoundLambdaExpression>(ExpressionType::LAMBDA, LogicalType::LAMBDA, std::move(lambda_body), 1);
	lambda->captures = std::move(captures);
	return lambda;
}

// Build a list_transform function call with proper binding
// element_return_type: the type each element maps to (e.g., BOOLEAN for predicates, some other type for casts)
static unique_ptr<Expression> BuildListTransformCall(OptimizerExtensionInput &input,
                                                     unique_ptr<Expression> counters_list,
                                                     unique_ptr<Expression> lambda_expr,
                                                     const LogicalType &element_return_type = LogicalType::BOOLEAN) {
	// Get the lambda body for ListLambdaBindData
	auto &bound_lambda = lambda_expr->Cast<BoundLambdaExpression>();

	// Get list_transform function from catalog
	auto &catalog = Catalog::GetSystemCatalog(input.context);
	auto &func_entry = catalog.GetEntry<ScalarFunctionCatalogEntry>(input.context, DEFAULT_SCHEMA, "list_transform");

	// Get the first function overload (list_transform has only one signature pattern)
	auto &scalar_func = func_entry.functions.functions[0];

	// Create the ListLambdaBindData with the lambda body
	auto list_return_type = LogicalType::LIST(element_return_type);
	PAC_DEBUG_PRINT("  BuildListTransformCall: lambda body = " + bound_lambda.lambda_expr->ToString());
	PAC_DEBUG_PRINT("  BuildListTransformCall: lambda body type = " +
	                ExpressionTypeToString(bound_lambda.lambda_expr->type));
	auto bind_data = make_uniq<ListLambdaBindData>(list_return_type, std::move(bound_lambda.lambda_expr), false, false);

	// Build children: [list, captures...]
	// Note: The lambda itself is NOT a child after binding - only its captures are
	vector<unique_ptr<Expression>> children;
	children.push_back(std::move(counters_list));

	// Add captures as children
	for (auto &capture : bound_lambda.captures) {
		children.push_back(std::move(capture));
	}

	// Create the bound function expression
	auto result =
	    make_uniq<BoundFunctionExpression>(list_return_type, scalar_func, std::move(children), std::move(bind_data));

	return result;
}

// Get struct field name for index (a, b, c, ..., z, aa, ab, ...)
static string GetStructFieldName(idx_t index) {
	if (index < 26) {
		return string(1, static_cast<char>('a' + static_cast<unsigned char>(index)));
	}
	// For indices >= 26, use aa, ab, etc.
	return string(1, static_cast<char>('a' + static_cast<unsigned char>(index / 26 - 1))) +
	       string(1, static_cast<char>('a' + static_cast<unsigned char>(index % 26)));
}

// Clone an expression tree for use as a lambda body with MULTIPLE PAC bindings
// Each PAC binding i is replaced with struct_extract(elem, field_name_i) where elem is the list_zip element
// Other column refs are captured as before
// struct_type: the type of the struct element from list_zip
// binding_to_index: maps binding hash -> index in the struct (for field name lookup)
// binding_to_type: maps binding hash -> the cast type for that binding
static unique_ptr<Expression> CloneForLambdaBodyMulti(Expression *expr,
                                                      const unordered_map<uint64_t, idx_t> &binding_to_index,
                                                      const unordered_map<uint64_t, LogicalType> &binding_to_type,
                                                      vector<unique_ptr<Expression>> &captures,
                                                      unordered_map<uint64_t, idx_t> &capture_map,
                                                      LogicalOperator *plan_root, const LogicalType &struct_type) {
	if (!expr) {
		return nullptr;
	}

	// Handle column references
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();

		// Hash the binding
		uint64_t binding_hash = HashBinding(col_ref.binding);

		// Check if this is one of the PAC bindings
		auto it = binding_to_index.find(binding_hash);
		if (it != binding_to_index.end()) {
			idx_t field_index = it->second;
			string field_name = GetStructFieldName(field_index);

			// Get the type for this binding
			LogicalType field_type = LogicalType::DOUBLE;
			auto type_it = binding_to_type.find(binding_hash);
			if (type_it != binding_to_type.end()) {
				field_type = type_it->second;
			}

#if PAC_DEBUG
			PAC_DEBUG_PRINT("    CloneForLambdaBodyMulti: PAC binding [" + std::to_string(col_ref.binding.table_index) +
			                "." + std::to_string(col_ref.binding.column_index) + "] -> struct field '" + field_name +
			                "' with type " + field_type.ToString());
#endif
			return BuildStructFieldExtract(struct_type, field_index, field_name);
		}

		// Not a PAC binding - check if it traces to one through projections
		ColumnBinding traced = col_ref.binding;
		auto *source_op = FindOperatorByTableIndex(plan_root, traced.table_index);

		while (source_op && source_op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			auto &proj = source_op->Cast<LogicalProjection>();
			if (traced.column_index < proj.expressions.size()) {
				auto &proj_expr = proj.expressions[traced.column_index];
				if (proj_expr->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &inner_ref = proj_expr->Cast<BoundColumnRefExpression>();
					traced = inner_ref.binding;
					uint64_t traced_hash = HashBinding(traced);
					auto traced_it = binding_to_index.find(traced_hash);
					if (traced_it != binding_to_index.end()) {
						// Found it through tracing
						idx_t field_index_traced = traced_it->second;
						string field_name_traced = GetStructFieldName(field_index_traced);
						return BuildStructFieldExtract(struct_type, field_index_traced, field_name_traced);
					}
					source_op = FindOperatorByTableIndex(plan_root, traced.table_index);
				} else {
					break;
				}
			} else {
				break;
			}
		}

		// Other column ref - needs to be captured
		return CaptureColumnRef(col_ref, captures, capture_map);
	}

	// Handle constant expressions
	if (expr->type == ExpressionType::VALUE_CONSTANT) {
		return expr->Copy();
	}

	// Handle cast expressions
	if (expr->type == ExpressionType::OPERATOR_CAST) {
		auto &cast = expr->Cast<BoundCastExpression>();
		auto child_clone = CloneForLambdaBodyMulti(cast.child.get(), binding_to_index, binding_to_type, captures,
		                                           capture_map, plan_root, struct_type);
		// If the child's type changed (e.g., from DECIMAL to DOUBLE due to PAC counter conversion),
		// we need to create a new cast with proper cast info for the new source type.
		// Using AddDefaultCastToType creates a proper cast from child's actual return type to target.
		if (child_clone->return_type != cast.child->return_type) {
			return BoundCastExpression::AddDefaultCastToType(std::move(child_clone), cast.return_type, cast.try_cast);
		}
		// Types match - safe to copy the original cast
		auto cast_copy = cast.Copy();
		auto &new_cast = cast_copy->Cast<BoundCastExpression>();
		new_cast.child = std::move(child_clone);
		return cast_copy;
	}

	// Handle comparison expressions
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON) {
		auto &comp = expr->Cast<BoundComparisonExpression>();
		auto left_clone = CloneForLambdaBodyMulti(comp.left.get(), binding_to_index, binding_to_type, captures,
		                                          capture_map, plan_root, struct_type);
		auto right_clone = CloneForLambdaBodyMulti(comp.right.get(), binding_to_index, binding_to_type, captures,
		                                           capture_map, plan_root, struct_type);
		return make_uniq<BoundComparisonExpression>(expr->type, std::move(left_clone), std::move(right_clone));
	}

	// Handle function expressions
	if (expr->type == ExpressionType::BOUND_FUNCTION) {
		auto &func = expr->Cast<BoundFunctionExpression>();
		vector<unique_ptr<Expression>> new_children;
		for (auto &child : func.children) {
			new_children.push_back(CloneForLambdaBodyMulti(child.get(), binding_to_index, binding_to_type, captures,
			                                               capture_map, plan_root, struct_type));
		}
		auto result = make_uniq<BoundFunctionExpression>(func.return_type, func.function, std::move(new_children),
		                                                 func.bind_info ? func.bind_info->Copy() : nullptr);
		return result;
	}

	// Handle operator expressions (e.g., AND, OR, NOT, arithmetic, COALESCE)
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_OPERATOR) {
		vector<unique_ptr<Expression>> new_children;
		for (auto &child : expr->Cast<BoundOperatorExpression>().children) {
			new_children.push_back(CloneForLambdaBodyMulti(child.get(), binding_to_index, binding_to_type, captures,
			                                               capture_map, plan_root, struct_type));
		}
		return BuildClonedOperatorExpression(expr, std::move(new_children));
	}

	// Handle CASE expressions
	if (expr->type == ExpressionType::CASE_EXPR) {
		auto &case_expr = expr->Cast<BoundCaseExpression>();

		if (IsScalarSubqueryWrapper(case_expr)) {
			PAC_DEBUG_PRINT("    CloneForLambdaBodyMulti: Stripping DuckDB scalar subquery wrapper, using ELSE branch");
			return CloneForLambdaBodyMulti(case_expr.else_expr.get(), binding_to_index, binding_to_type, captures,
			                               capture_map, plan_root, struct_type);
		}

		// Regular CASE - recurse into all branches
		auto result = make_uniq<BoundCaseExpression>(case_expr.return_type);

		for (auto &check : case_expr.case_checks) {
			BoundCaseCheck new_check;
			new_check.when_expr = CloneForLambdaBodyMulti(check.when_expr.get(), binding_to_index, binding_to_type,
			                                              captures, capture_map, plan_root, struct_type);
			new_check.then_expr = CloneForLambdaBodyMulti(check.then_expr.get(), binding_to_index, binding_to_type,
			                                              captures, capture_map, plan_root, struct_type);
			result->case_checks.push_back(std::move(new_check));
		}

		if (case_expr.else_expr) {
			result->else_expr = CloneForLambdaBodyMulti(case_expr.else_expr.get(), binding_to_index, binding_to_type,
			                                            captures, capture_map, plan_root, struct_type);
		}

		return result;
	}

	// For any other expression type, try to copy it directly
	return expr->Copy();
}

// Build a list_zip function call combining multiple counter lists
// Returns LIST<STRUCT<a T1, b T2, ...>> where each field corresponds to one PAC binding
static unique_ptr<Expression> BuildListZipCall(OptimizerExtensionInput &input,
                                               vector<unique_ptr<Expression>> counter_lists,
                                               LogicalType &out_struct_type) {
	if (counter_lists.empty()) {
		return nullptr;
	}

	// Build the struct type for list_zip result
	// list_zip returns LIST<STRUCT<a T1, b T2, ...>>
	child_list_t<LogicalType> struct_children;
	for (idx_t i = 0; i < counter_lists.size(); i++) {
		string field_name = GetStructFieldName(i);
		// All counter lists are LIST<DOUBLE>
		struct_children.push_back(make_pair(field_name, LogicalType::DOUBLE));
	}
	out_struct_type = LogicalType::STRUCT(struct_children);
	auto list_struct_type = LogicalType::LIST(out_struct_type);

	// Get list_zip function from catalog
	auto &catalog = Catalog::GetSystemCatalog(input.context);
	auto &func_entry = catalog.GetEntry<ScalarFunctionCatalogEntry>(input.context, DEFAULT_SCHEMA, "list_zip");

	// Find the appropriate overload (list_zip is variadic)
	vector<LogicalType> arg_types;
	for (auto &list : counter_lists) {
		arg_types.push_back(list->return_type);
	}

	ErrorData error;
	FunctionBinder function_binder(input.context);
	auto best_function = function_binder.BindFunction(func_entry.name, func_entry.functions, arg_types, error);

	if (!best_function.IsValid()) {
#if PAC_DEBUG
		PAC_DEBUG_PRINT("Warning: Could not bind list_zip: " + error.Message());
#endif
		return nullptr;
	}

	auto scalar_func = func_entry.functions.GetFunctionByOffset(best_function.GetIndex());

	// Build the function expression
	auto result = make_uniq<BoundFunctionExpression>(list_struct_type, scalar_func, std::move(counter_lists), nullptr);
	return result;
}

// Build a pac_noised function call to reduce LIST<DOUBLE> (64 counters) to a single noised DOUBLE
static unique_ptr<Expression> BuildPacNoisedCall(OptimizerExtensionInput &input, unique_ptr<Expression> counters_list) {
	// counters_list is LIST<DOUBLE> (64 values)
	// Returns: pac_noised(counters_list) -> DOUBLE
	return input.optimizer.BindScalarFunction("pac_noised", std::move(counters_list));
}

// Forward declaration
static void PropagateCountersType(const ColumnBinding &binding, LogicalOperator *plan_root);

// Forward declaration
static void SyncColumnRefTypes(LogicalOperator *op, LogicalOperator *plan_root);

// Helper to get the actual return type for a binding from the plan
static LogicalType GetBindingReturnType(const ColumnBinding &binding, LogicalOperator *plan_root) {
	auto *source = FindOperatorByTableIndex(plan_root, binding.table_index);
	if (!source) {
		return LogicalType::INVALID;
	}

	if (source->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &agg = source->Cast<LogicalAggregate>();
		if (binding.table_index == agg.aggregate_index && binding.column_index < agg.expressions.size()) {
			return agg.expressions[binding.column_index]->return_type;
		}
		if (binding.table_index == agg.group_index && binding.column_index < agg.groups.size()) {
			return agg.groups[binding.column_index]->return_type;
		}
	} else if (source->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = source->Cast<LogicalProjection>();
		if (binding.column_index < proj.expressions.size()) {
			return proj.expressions[binding.column_index]->return_type;
		}
	}

	// Fallback: use the operator's types vector if available
	if (binding.column_index < source->types.size()) {
		return source->types[binding.column_index];
	}

	return LogicalType::INVALID;
}

// Recursively update return types in an expression to match what the source operators produce
// This is needed because after converting aggregates to _counters, the expressions
// referencing them still have the old types
static void UpdateExpressionTypesToList(Expression *expr, LogicalOperator *plan_root) {
	if (!expr) {
		return;
	}

	// Check if this expression is a column ref and update its type from the source
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();
		LogicalType source_type = GetBindingReturnType(col_ref.binding, plan_root);
		PAC_DEBUG_PRINT("    UpdateExpTypesToList: col_ref [" + std::to_string(col_ref.binding.table_index) + "." +
		                std::to_string(col_ref.binding.column_index) + "] '" + col_ref.alias +
		                "' current=" + col_ref.return_type.ToString() + " source=" + source_type.ToString());
		if (source_type.id() != LogicalTypeId::INVALID && source_type != col_ref.return_type) {
			col_ref.return_type = source_type;
			PAC_DEBUG_PRINT("      -> Updated to " + source_type.ToString());
		}
	}

	// Handle subquery expressions - update type if it traces to PAC counters
	if (expr->type == ExpressionType::SUBQUERY) {
		if (TracesPacCountersAggregate(expr, plan_root)) {
			expr->return_type = LogicalType::LIST(LogicalType::DOUBLE);
		}
	}

	// Handle cast expressions
	if (expr->type == ExpressionType::OPERATOR_CAST) {
		auto &cast = expr->Cast<BoundCastExpression>();
		UpdateExpressionTypesToList(cast.child.get(), plan_root);
		// If child is now LIST<DOUBLE>, the cast itself should also be LIST<DOUBLE>
		if (cast.child->return_type.id() == LogicalTypeId::LIST) {
			cast.return_type = cast.child->return_type;
		}
	}

	// Recurse into function children
	if (expr->type == ExpressionType::BOUND_FUNCTION) {
		auto &func = expr->Cast<BoundFunctionExpression>();
		for (auto &child : func.children) {
			UpdateExpressionTypesToList(child.get(), plan_root);
		}
	}

	// Recurse into operator children
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_OPERATOR) {
		auto &op = expr->Cast<BoundOperatorExpression>();
		for (auto &child : op.children) {
			UpdateExpressionTypesToList(child.get(), plan_root);
		}
	}

	// Recurse into aggregate children
	if (expr->type == ExpressionType::BOUND_AGGREGATE) {
		auto &agg = expr->Cast<BoundAggregateExpression>();
		for (auto &child : agg.children) {
			UpdateExpressionTypesToList(child.get(), plan_root);
		}
	}
}

// Sync column ref types in a single expression tree (bottom-up within expression)
// Also propagates LIST<DOUBLE> types up through parent expressions like CASE, CAST, etc.
static void SyncColumnRefTypesInExpression(Expression *expr, LogicalOperator *plan_root) {
	if (!expr) {
		return;
	}

	// First recurse into all children (bottom-up: children before parent)
	ExpressionIterator::EnumerateChildren(
	    *expr, [&](Expression &child) { SyncColumnRefTypesInExpression(&child, plan_root); });

	// Then update this expression
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();
		LogicalType source_type = GetBindingReturnType(col_ref.binding, plan_root);
		PAC_DEBUG_PRINT("  SyncColRef [" + std::to_string(col_ref.binding.table_index) + "." +
		                std::to_string(col_ref.binding.column_index) + "] '" + col_ref.alias +
		                "' current=" + col_ref.return_type.ToString() + " source=" + source_type.ToString());
		if (source_type.id() != LogicalTypeId::INVALID && source_type != col_ref.return_type) {
			col_ref.return_type = source_type;
			PAC_DEBUG_PRINT("    -> Updated");
		}
	}

	// Propagate LIST<DOUBLE> type up through parent expressions
	// CASE expression: if THEN/ELSE branch is LIST<DOUBLE>, update CASE return type
	if (expr->type == ExpressionType::CASE_EXPR) {
		auto &case_expr = expr->Cast<BoundCaseExpression>();
		bool any_list = false;
		// Check ELSE clause
		if (case_expr.else_expr && case_expr.else_expr->return_type.id() == LogicalTypeId::LIST) {
			any_list = true;
		}
		// Check WHEN clauses
		for (auto &when_clause : case_expr.case_checks) {
			if (when_clause.then_expr && when_clause.then_expr->return_type.id() == LogicalTypeId::LIST) {
				any_list = true;
				break;
			}
		}
		if (any_list && expr->return_type.id() != LogicalTypeId::LIST) {
			expr->return_type = LogicalType::LIST(LogicalType::DOUBLE);
			PAC_DEBUG_PRINT("  Updated CASE return_type to LIST<DOUBLE>");
		}
	}

	// CAST expression: if child is LIST<DOUBLE>, update CAST return type
	if (expr->type == ExpressionType::OPERATOR_CAST) {
		auto &cast = expr->Cast<BoundCastExpression>();
		if (cast.child && cast.child->return_type.id() == LogicalTypeId::LIST &&
		    expr->return_type.id() != LogicalTypeId::LIST) {
			expr->return_type = cast.child->return_type;
			PAC_DEBUG_PRINT("  Updated CAST return_type to " + expr->return_type.ToString());
		}
	}

	// Function expression: if any argument is LIST<DOUBLE> and function is arithmetic,
	// the result should also be LIST<DOUBLE> (for list_* functions)
	if (expr->type == ExpressionType::BOUND_FUNCTION) {
		auto &func = expr->Cast<BoundFunctionExpression>();
		// Only update if it's a list operation that should return LIST
		if (func.function.name.find("list_") == 0 || func.function.name == "*" || func.function.name == "/" ||
		    func.function.name == "+" || func.function.name == "-") {
			for (auto &child : func.children) {
				if (child && child->return_type.id() == LogicalTypeId::LIST) {
					// The function operates on lists - check if we need to update return type
					// Don't change aggregate function return types
					break;
				}
			}
		}
	}
}

// Comprehensive function to sync all column ref types throughout the plan
// Call this after all aggregate conversions to ensure types are consistent
// Uses BOTTOM-UP processing so that child operator types are updated before we query them
static void SyncColumnRefTypes(LogicalOperator *op, LogicalOperator *plan_root) {
	if (!op) {
		return;
	}

	// First recurse into children (BOTTOM-UP: children before current operator)
	// This ensures that when we query GetBindingReturnType for a projection,
	// the projection's column refs have already been updated from their source aggregates
	for (auto &child : op->children) {
		SyncColumnRefTypes(child.get(), plan_root);
	}

	// Update types in all expressions of this operator
	for (auto &expr : op->expressions) {
		SyncColumnRefTypesInExpression(expr.get(), plan_root);
	}

	// Handle operator-specific expressions
	if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
		auto &filter = op->Cast<LogicalFilter>();
		for (auto &expr : filter.expressions) {
			SyncColumnRefTypesInExpression(expr.get(), plan_root);
		}
	} else if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = op->Cast<LogicalProjection>();
		for (auto &expr : proj.expressions) {
			SyncColumnRefTypesInExpression(expr.get(), plan_root);
		}
	} else if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &agg = op->Cast<LogicalAggregate>();
		for (auto &expr : agg.expressions) {
			SyncColumnRefTypesInExpression(expr.get(), plan_root);
		}
		for (auto &expr : agg.groups) {
			SyncColumnRefTypesInExpression(expr.get(), plan_root);
		}
	} else if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	           op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		auto &join = op->Cast<LogicalComparisonJoin>();
		for (auto &cond : join.conditions) {
			SyncColumnRefTypesInExpression(cond.left.get(), plan_root);
			SyncColumnRefTypesInExpression(cond.right.get(), plan_root);
		}
	}
}

// Strip invalid CAST expressions from the plan where child is LIST but CAST expects scalar
// Returns true if the expression was modified
static bool StripInvalidCastsInExpression(unique_ptr<Expression> &expr) {
	if (!expr) {
		return false;
	}
	bool modified = false;

	// First recurse into children
	ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) {
		if (StripInvalidCastsInExpression(child)) {
			modified = true;
		}
	});

	// Check if this is a CAST where child is LIST but return type is also LIST
	// This indicates we updated the return_type but the underlying cast is invalid
	if (expr->type == ExpressionType::OPERATOR_CAST) {
		auto &cast = expr->Cast<BoundCastExpression>();
		if (cast.child && cast.child->return_type.id() == LogicalTypeId::LIST) {
			// Child is already LIST, strip the CAST and use child directly
			PAC_DEBUG_PRINT("  Stripping invalid CAST: " + expr->ToString() + " -> " + cast.child->ToString());
			expr = std::move(cast.child);
			modified = true;
		}
	}

	return modified;
}

// Strip invalid CASTs throughout the plan
static void StripInvalidCastsInPlan(LogicalOperator *op) {
	if (!op) {
		return;
	}

	// First recurse into children
	for (auto &child : op->children) {
		StripInvalidCastsInPlan(child.get());
	}

	// Strip invalid CASTs in expressions
	for (auto &expr : op->expressions) {
		StripInvalidCastsInExpression(expr);
	}

	// Handle operator-specific expressions
	if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
		auto &filter = op->Cast<LogicalFilter>();
		for (auto &expr : filter.expressions) {
			StripInvalidCastsInExpression(expr);
		}
	} else if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = op->Cast<LogicalProjection>();
		for (auto &expr : proj.expressions) {
			StripInvalidCastsInExpression(expr);
		}
	} else if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	           op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		auto &join = op->Cast<LogicalComparisonJoin>();
		for (auto &cond : join.conditions) {
			StripInvalidCastsInExpression(cond.left);
			StripInvalidCastsInExpression(cond.right);
		}
	}
}

// Helper to extract scalar constant and PAC column ref from a multiplication expression
// Returns true if the expression matches pattern: scalar * col_ref or col_ref * scalar
static bool ExtractMultiplicationPattern(Expression *expr, LogicalOperator *plan_root,
                                         unique_ptr<Expression> &out_scalar, ColumnBinding &out_pac_binding) {
	if (!expr || expr->type != ExpressionType::BOUND_FUNCTION) {
		return false;
	}

	auto &func = expr->Cast<BoundFunctionExpression>();
	if (func.function.name != "*" || func.children.size() != 2) {
		return false;
	}

	// Check which child is the PAC aggregate and which is scalar
	for (idx_t i = 0; i < 2; i++) {
		auto &child = func.children[i];
		auto &other = func.children[1 - i];

		// Check if child is a column ref to PAC aggregate
		if (child->type == ExpressionType::BOUND_COLUMN_REF) {
			auto &col_ref = child->Cast<BoundColumnRefExpression>();
			string pac_name = TracePacAggregateFromBinding(col_ref.binding, plan_root);
			if (!pac_name.empty()) {
				// Found PAC binding, check if other is a constant or cast of constant
				if (other->type == ExpressionType::VALUE_CONSTANT || other->type == ExpressionType::OPERATOR_CAST) {
					out_scalar = other->Copy();
					out_pac_binding = col_ref.binding;
					return true;
				}
			}
		}
	}

	return false;
}

// Check if an expression is already wrapped in a categorical rewrite terminal function
// This includes pac_noised, pac_filter, list_transform, and list_zip
// These functions indicate that the expression has already been processed by categorical rewriting
static bool IsAlreadyWrappedInPacNoised(Expression *expr) {
	if (!expr) {
		return false;
	}
	if (expr->type == ExpressionType::BOUND_FUNCTION) {
		auto &func = expr->Cast<BoundFunctionExpression>();
		// Check for all categorical rewrite terminal functions
		if (func.function.name == "pac_noised" || func.function.name == "pac_filter" ||
		    func.function.name == "list_transform" || func.function.name == "list_zip") {
			return true;
		}
	}
	// Check for CAST(terminal_function(...))
	if (expr->type == ExpressionType::OPERATOR_CAST) {
		auto &cast = expr->Cast<BoundCastExpression>();
		return IsAlreadyWrappedInPacNoised(cast.child.get());
	}
	return false;
}

// Check if this projection expression is part of a filter rewrite (not a final output)
// Filter rewrite expressions will have boolean return type ultimately
static bool IsFilterPatternProjection(Expression *expr, LogicalOperator *op) {
	// If the expression returns LIST<BOOLEAN>, it's likely a filter pattern
	if (expr->return_type.id() == LogicalTypeId::LIST) {
		auto &list_type = ListType::GetChildType(expr->return_type);
		if (list_type.id() == LogicalTypeId::BOOLEAN) {
			return true;
		}
	}
	return false;
}

// Find scalar wrapper for a binding (if any)
// Returns the outer Projection of the wrapper pattern, or nullptr
static LogicalOperator *FindScalarWrapperForBinding(const ColumnBinding &binding, LogicalOperator *plan_root) {
	auto *source_op = FindOperatorByTableIndex(plan_root, binding.table_index);
	PAC_DEBUG_PRINT("    FindScalarWrapperForBinding: binding [" + std::to_string(binding.table_index) + "." +
	                std::to_string(binding.column_index) +
	                "] source_op=" + (source_op ? LogicalOperatorToString(source_op->type) : "nullptr"));
	if (!source_op || source_op->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return nullptr;
	}
	// Check if this projection is the outer part of a scalar wrapper
	auto *unwrapped = RecognizeDuckDBScalarWrapper(source_op);
	PAC_DEBUG_PRINT("    FindScalarWrapperForBinding: RecognizeDuckDBScalarWrapper returned " +
	                (unwrapped ? LogicalOperatorToString(unwrapped->type) : "nullptr"));
	if (unwrapped) {
		return source_op;
	}
	return nullptr;
}

// Strip a scalar subquery wrapper operator in place
// Pattern: Project(CASE) #X -> Aggregate(first, count*) -> Project #Z -> [inner]
// We delete outer Project and Aggregate, keep inner Project with outer's table_index
// This preserves column references like [X.0] that point to the outer
static void StripScalarWrapperInPlace(unique_ptr<LogicalOperator> &wrapper_ptr) {
	if (!wrapper_ptr || wrapper_ptr->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return;
	}

	auto &outer_proj = wrapper_ptr->Cast<LogicalProjection>();
	if (outer_proj.children.empty()) {
		return;
	}

	auto *agg_op = outer_proj.children[0].get();
	if (agg_op->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return;
	}

	auto &agg = agg_op->Cast<LogicalAggregate>();
	if (agg.children.empty()) {
		return;
	}

	auto *inner_proj_op = agg.children[0].get();
	if (inner_proj_op->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return;
	}

	auto &inner_proj = inner_proj_op->Cast<LogicalProjection>();

	PAC_DEBUG_PRINT("Stripping scalar subquery wrapper: outer_proj #" + std::to_string(outer_proj.table_index) +
	                " -> inner_proj #" + std::to_string(inner_proj.table_index));

	// Change inner projection's table_index to match outer's
	// This preserves any column references like [outer.0] that existed
	inner_proj.table_index = outer_proj.table_index;

	// Update inner projection's types to match its expressions
	inner_proj.types.clear();
	for (auto &expr : inner_proj.expressions) {
		inner_proj.types.push_back(expr->return_type);
	}

	// Replace the wrapper_ptr with the inner projection
	// This removes outer projection and aggregate from the plan entirely
	wrapper_ptr = std::move(agg.children[0]);
}

// Helper to check if an expression contains a column ref with a specific table_index
static bool ExpressionContainsColumnRefToTable(Expression *expr, idx_t table_index) {
	if (!expr) {
		return false;
	}
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();
		if (col_ref.binding.table_index == table_index) {
			return true;
		}
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) {
		if (ExpressionContainsColumnRefToTable(child.get(), table_index)) {
			found = true;
		}
	});
	return found;
}

// Check if this projection's output is referenced by a categorical filter pattern
// If so, we should NOT wrap it with pac_noised - the filter will handle the rewrite
static bool IsProjectionReferencedByFilterPattern(LogicalProjection &proj,
                                                  const vector<CategoricalPatternInfo> &patterns,
                                                  LogicalOperator *plan_root) {
	// For each pattern, check if the filter expression references this projection's output
	PAC_DEBUG_PRINT("  IsProjectionReferencedByFilterPattern: proj #" + std::to_string(proj.table_index) +
	                " checking against " + std::to_string(patterns.size()) + " patterns");

	for (auto &pattern : patterns) {
		if (!pattern.parent_op) {
			continue;
		}

		PAC_DEBUG_PRINT("    Pattern: parent_op=" + LogicalOperatorToString(pattern.parent_op->type) +
		                ", expr_index=" + std::to_string(pattern.expr_index));

		// Handle LOGICAL_FILTER patterns
		if (pattern.parent_op->type == LogicalOperatorType::LOGICAL_FILTER) {
			auto &filter = pattern.parent_op->Cast<LogicalFilter>();
			if (pattern.expr_index >= filter.expressions.size()) {
				continue;
			}

			auto &filter_expr = filter.expressions[pattern.expr_index];

			// Check for direct column refs to this projection
			if (ExpressionContainsColumnRefToTable(filter_expr.get(), proj.table_index)) {
				return true;
			}

			// Also check traced PAC bindings
			auto pac_bindings = FindAllPacBindingsInExpression(filter_expr.get(), plan_root);
			for (auto &binding_info : pac_bindings) {
				if (binding_info.binding.table_index == proj.table_index) {
					return true;
				}
			}
		}
		// Handle COMPARISON_JOIN and DELIM_JOIN patterns
		else if (pattern.parent_op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
		         pattern.parent_op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			auto &join = pattern.parent_op->Cast<LogicalComparisonJoin>();
			if (pattern.expr_index >= join.conditions.size()) {
				continue;
			}

			auto &cond = join.conditions[pattern.expr_index];

			PAC_DEBUG_PRINT("      Join cond left: " + cond.left->ToString() +
			                " type=" + ExpressionTypeToString(cond.left->type));
			PAC_DEBUG_PRINT("      Join cond right: " + cond.right->ToString() +
			                " type=" + ExpressionTypeToString(cond.right->type));

			// Check for direct column refs to this projection on either side
			bool left_has = ExpressionContainsColumnRefToTable(cond.left.get(), proj.table_index);
			bool right_has = ExpressionContainsColumnRefToTable(cond.right.get(), proj.table_index);
			PAC_DEBUG_PRINT("      left_has_ref=" + std::to_string(left_has) +
			                ", right_has_ref=" + std::to_string(right_has));

			if (left_has || right_has) {
				PAC_DEBUG_PRINT("      -> FOUND! Projection #" + std::to_string(proj.table_index) +
				                " is referenced by join condition");
				return true;
			}

			// Also check traced PAC bindings
			auto left_bindings = FindAllPacBindingsInExpression(cond.left.get(), plan_root);
			auto right_bindings = FindAllPacBindingsInExpression(cond.right.get(), plan_root);

			for (auto &binding_info : left_bindings) {
				if (binding_info.binding.table_index == proj.table_index) {
					return true;
				}
			}
			for (auto &binding_info : right_bindings) {
				if (binding_info.binding.table_index == proj.table_index) {
					return true;
				}
			}
		}
	}
	return false;
}

// Rewrite projection expressions that contain arithmetic with PAC aggregates
// Handles three cases:
// 1. Simple column reference: Just update type to LIST<DOUBLE>
// 2. Single PAC aggregate with arithmetic: list_transform + pac_noised for scalar result
// 3. Multiple PAC aggregates: list_zip + list_transform + pac_noised
// This must be called AFTER ReplacePacAggregatesWithCounters
// IMPORTANT: Projections that are referenced by filter patterns should NOT be rewritten here
//            because the filter pattern rewrite will handle them with list_zip + list_transform + pac_filter
static void RewriteProjectionsWithCounters(LogicalOperator *op, OptimizerExtensionInput &input,
                                           LogicalOperator *plan_root,
                                           const unordered_set<LogicalAggregate *> &categorical_aggregates,
                                           const vector<CategoricalPatternInfo> &patterns) {
	if (!op) {
		return;
	}

	// Process projections that have arithmetic with PAC counters
	if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = op->Cast<LogicalProjection>();
		PAC_DEBUG_PRINT("RewriteProjectionsWithCounters: Processing projection #" + std::to_string(proj.table_index) +
		                " with " + std::to_string(proj.expressions.size()) + " expressions");

		// Check if this projection's output is referenced by a categorical filter pattern
		// If so, we need to rewrite expressions to use list_transform but NOT add pac_noised
		// The filter pattern will handle the final pac_filter wrapping
		bool is_filter_pattern_projection = IsProjectionReferencedByFilterPattern(proj, patterns, plan_root);
#if PAC_DEBUG
		if (is_filter_pattern_projection) {
			PAC_DEBUG_PRINT("Projection #" + std::to_string(proj.table_index) +
			                " is referenced by filter pattern - will rewrite without pac_noised");
		}
#endif

		for (idx_t i = 0; i < proj.expressions.size(); i++) {
			auto &expr = proj.expressions[i];
			PAC_DEBUG_PRINT("  Expr " + std::to_string(i) + ": " + expr->ToString().substr(0, 60) +
			                " type=" + ExpressionTypeToString(expr->type));

			// Skip if already wrapped in pac_noised
			if (IsAlreadyWrappedInPacNoised(expr.get())) {
				PAC_DEBUG_PRINT("    -> Skipping: already wrapped in pac_noised");
				continue;
			}

			// Find ALL PAC bindings in this expression
			auto pac_bindings = FindAllPacBindingsInExpression(expr.get(), plan_root);
			if (pac_bindings.empty()) {
				PAC_DEBUG_PRINT("    -> Skipping: no PAC bindings");
				continue;
			}
			PAC_DEBUG_PRINT("    -> Found " + std::to_string(pac_bindings.size()) + " PAC binding(s)");

			// Check if this expression is just a column reference (no arithmetic)
			if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
				// Just a column ref, update its type and continue
				expr->return_type = LogicalType::LIST(LogicalType::DOUBLE);
				PAC_DEBUG_PRINT("    -> Just column ref, updated type to LIST<DOUBLE>");
				continue;
			}

			// Skip filter pattern projections (they're handled separately)
			if (IsFilterPatternProjection(expr.get(), op)) {
				continue;
			}

			// Capture the original expression's target type for final casting
			LogicalType target_type = expr->return_type;

			// Only rewrite expressions with numerical result types
			// pac_noised expects numerical input and produces DOUBLE
			if (!IsNumericalType(target_type)) {
#if PAC_DEBUG
				PAC_DEBUG_PRINT("RewriteProjectionsWithCounters: Skipping non-numerical type " +
				                target_type.ToString());
#endif
				continue;
			}

#if PAC_DEBUG
			PAC_DEBUG_PRINT("RewriteProjectionsWithCounters: Found projection with " +
			                std::to_string(pac_bindings.size()) + " PAC binding(s)");
			PAC_DEBUG_PRINT("  Original expr: " + expr->ToString());
			PAC_DEBUG_PRINT("  Target type: " + target_type.ToString());
#endif

			if (pac_bindings.size() == 1) {
				// ============================================================
				// SINGLE AGGREGATE: list_transform + pac_noised
				// ============================================================
				auto &binding_info = pac_bindings[0];
				ColumnBinding pac_binding = binding_info.binding;

				// Propagate type to the binding source
				PropagateCountersType(pac_binding, plan_root);

				// Try to extract simple multiplication pattern: scalar * col_ref
				unique_ptr<Expression> scalar_expr;
				ColumnBinding mult_pac_binding;
				if (ExtractMultiplicationPattern(expr.get(), plan_root, scalar_expr, mult_pac_binding)) {
#if PAC_DEBUG
					PAC_DEBUG_PRINT("  Single-aggregate multiplication pattern");
#endif
					// Build fresh lambda body using BindScalarFunction to get correct types
					// Lambda body: elem -> scalar * elem (where elem is DOUBLE)
					auto elem_ref = make_uniq<BoundReferenceExpression>("elem", LogicalType::DOUBLE, 0);

					// Cast scalar to DOUBLE if needed
					if (scalar_expr->return_type != LogicalType::DOUBLE) {
						scalar_expr =
						    BoundCastExpression::AddDefaultCastToType(std::move(scalar_expr), LogicalType::DOUBLE);
					}

					// Bind multiplication with DOUBLE types
					auto mult_body =
					    input.optimizer.BindScalarFunction("*", std::move(scalar_expr), std::move(elem_ref));

					// Build the lambda
					auto lambda = BuildPacLambda(std::move(mult_body), {});

					// Get the counters column reference
					auto counters_ref = make_uniq<BoundColumnRefExpression>(
					    "pac_counters", LogicalType::LIST(LogicalType::DOUBLE), mult_pac_binding);

					// Build: list_transform(counters, elem -> scalar * elem) -> LIST<DOUBLE>
					auto list_transform =
					    BuildListTransformCall(input, std::move(counters_ref), std::move(lambda), LogicalType::DOUBLE);

					unique_ptr<Expression> final_expr;
					LogicalType final_type;

					if (is_filter_pattern_projection) {
						// For filter pattern projections, keep as LIST<DOUBLE> - no pac_noised
						// The filter rewrite's double-lambda will handle the rest
						final_expr = std::move(list_transform);
						final_type = LogicalType::LIST(LogicalType::DOUBLE);
					} else {
						// Wrap with pac_noised to get scalar result
						auto noised = BuildPacNoisedCall(input, std::move(list_transform));
						// Cast to target type if needed
						if (target_type != LogicalType::DOUBLE) {
							noised = BoundCastExpression::AddDefaultCastToType(std::move(noised), target_type);
						}
						final_expr = std::move(noised);
						final_type = target_type;
					}

#if PAC_DEBUG
					PAC_DEBUG_PRINT("  Rewritten to: " + final_expr->ToString());
#endif

					// Replace the expression
					proj.expressions[i] = std::move(final_expr);
					proj.types[i] = final_type;
					continue;
				}

				// For single-aggregate expressions, use list_transform approach
				// Filter pattern projections need LIST<DOUBLE> output; non-filter need scalar via pac_noised

				if (is_filter_pattern_projection) {
					// Check if expression is just CAST(col_ref) - can output raw counters
					bool is_simple_cast =
					    expr->type == ExpressionType::OPERATOR_CAST &&
					    expr->Cast<BoundCastExpression>().child->type == ExpressionType::BOUND_COLUMN_REF;
					if (is_simple_cast) {
						// Simple cast on column ref - output raw counters, filter rewrite handles the cast
						proj.expressions[i] = make_uniq<BoundColumnRefExpression>(
						    "pac_counters", LogicalType::LIST(LogicalType::DOUBLE), pac_binding);
						proj.types[i] = LogicalType::LIST(LogicalType::DOUBLE);
						continue;
					}

					// Complex expression - strip outer CAST if present, then transform
					Expression *expr_to_clone = expr.get();
					if (expr->type == ExpressionType::OPERATOR_CAST &&
					    expr->Cast<BoundCastExpression>().return_type != LogicalType::DOUBLE) {
						expr_to_clone = expr->Cast<BoundCastExpression>().child.get();
					}

					vector<unique_ptr<Expression>> captures;
					unordered_map<uint64_t, idx_t> capture_map;
					auto lambda_body = CloneForLambdaBody(expr_to_clone, pac_binding, captures, capture_map, plan_root,
					                                      LogicalType::DOUBLE);
					auto lambda = BuildPacLambda(std::move(lambda_body), std::move(captures));
					auto counters_ref = make_uniq<BoundColumnRefExpression>(
					    "pac_counters", LogicalType::LIST(LogicalType::DOUBLE), pac_binding);
					proj.expressions[i] =
					    BuildListTransformCall(input, std::move(counters_ref), std::move(lambda), LogicalType::DOUBLE);
					proj.types[i] = LogicalType::LIST(LogicalType::DOUBLE);
				} else {
					// Non-filter: full transformation with pac_noised for scalar output
					// pac_noised expects LIST<DOUBLE>, so we must ensure the lambda body produces DOUBLE
					// Strip outer cast if present (we'll re-apply after pac_noised)
					Expression *expr_to_clone = expr.get();
					if (expr->type == ExpressionType::OPERATOR_CAST &&
					    expr->Cast<BoundCastExpression>().return_type != LogicalType::DOUBLE) {
						expr_to_clone = expr->Cast<BoundCastExpression>().child.get();
					}
					PAC_DEBUG_PRINT("  [SINGLE-AGG NON-FILTER] PAC binding: [" +
					                std::to_string(pac_binding.table_index) + "." +
					                std::to_string(pac_binding.column_index) + "]");
					PAC_DEBUG_PRINT("  [SINGLE-AGG NON-FILTER] Cloning expression: " + expr_to_clone->ToString());
					PAC_DEBUG_PRINT("  [SINGLE-AGG NON-FILTER] Expression type: " +
					                ExpressionTypeToString(expr_to_clone->type));

					vector<unique_ptr<Expression>> captures;
					unordered_map<uint64_t, idx_t> capture_map;
					auto lambda_body = CloneForLambdaBody(expr_to_clone, pac_binding, captures, capture_map, plan_root,
					                                      LogicalType::DOUBLE);

					// Ensure lambda body produces DOUBLE for pac_noised
					if (lambda_body->return_type != LogicalType::DOUBLE) {
						lambda_body =
						    BoundCastExpression::AddDefaultCastToType(std::move(lambda_body), LogicalType::DOUBLE);
					}

					auto lambda = BuildPacLambda(std::move(lambda_body), std::move(captures));
					auto counters_ref = make_uniq<BoundColumnRefExpression>(
					    "pac_counters", LogicalType::LIST(LogicalType::DOUBLE), pac_binding);
					auto list_transform =
					    BuildListTransformCall(input, std::move(counters_ref), std::move(lambda), LogicalType::DOUBLE);
					auto noised = BuildPacNoisedCall(input, std::move(list_transform));
					if (target_type != LogicalType::DOUBLE) {
						noised = BoundCastExpression::AddDefaultCastToType(std::move(noised), target_type);
					}
					proj.expressions[i] = std::move(noised);
					proj.types[i] = target_type;
				}

			} else {
				// ============================================================
				// MULTIPLE AGGREGATES: list_zip + list_transform + pac_noised
				// ============================================================
#if PAC_DEBUG
				PAC_DEBUG_PRINT("  Multi-aggregate list_zip pattern with " + std::to_string(pac_bindings.size()) +
				                " bindings");
#endif

				// Propagate types to all binding sources
				for (auto &binding_info : pac_bindings) {
					PropagateCountersType(binding_info.binding, plan_root);
				}

				// Build counter expressions for each PAC binding
				vector<unique_ptr<Expression>> counter_lists;
				unordered_map<uint64_t, idx_t> binding_to_index;
				unordered_map<uint64_t, LogicalType> binding_to_type;

				for (auto &binding_info : pac_bindings) {
					auto counters_expr = make_uniq<BoundColumnRefExpression>(
					    "pac_counters", LogicalType::LIST(LogicalType::DOUBLE), binding_info.binding);
					counter_lists.push_back(std::move(counters_expr));

					uint64_t hash = HashBinding(binding_info.binding);
					binding_to_index[hash] = binding_info.index;
					binding_to_type[hash] = binding_info.original_type;
				}

				// Build list_zip(A1, A2, ...) -> LIST<STRUCT<a DOUBLE, b DOUBLE, ...>>
				LogicalType struct_type;
				auto zipped_list = BuildListZipCall(input, std::move(counter_lists), struct_type);

				if (!zipped_list) {
#if PAC_DEBUG
					PAC_DEBUG_PRINT("  Failed to build list_zip call");
#endif
					continue;
				}

#if PAC_DEBUG
				PAC_DEBUG_PRINT("  Built list_zip with struct type: " + struct_type.ToString());
#endif

				// Build lambda body using CloneForLambdaBodyMulti
				// This clones the expression, replacing each PAC binding with struct field extraction
				vector<unique_ptr<Expression>> captures;
				unordered_map<uint64_t, idx_t> capture_map;

				// For non-filter projections going to pac_noised, strip outer cast and ensure DOUBLE output
				// pac_noised expects LIST<DOUBLE>
				Expression *expr_to_clone = expr.get();
				if (!is_filter_pattern_projection && expr->type == ExpressionType::OPERATOR_CAST &&
				    expr->Cast<BoundCastExpression>().return_type != LogicalType::DOUBLE) {
					expr_to_clone = expr->Cast<BoundCastExpression>().child.get();
				}

				auto lambda_body = CloneForLambdaBodyMulti(expr_to_clone, binding_to_index, binding_to_type, captures,
				                                           capture_map, plan_root, struct_type);

				// For non-filter projections, ensure lambda body produces DOUBLE for pac_noised
				if (!is_filter_pattern_projection && lambda_body->return_type != LogicalType::DOUBLE) {
					lambda_body =
					    BoundCastExpression::AddDefaultCastToType(std::move(lambda_body), LogicalType::DOUBLE);
				}

#if PAC_DEBUG
				PAC_DEBUG_PRINT("  Lambda body created with " + std::to_string(captures.size()) + " captures");
				PAC_DEBUG_PRINT("  Lambda body: " + lambda_body->ToString());
#endif

				// Build lambda: elem -> expr(elem.a, elem.b, ...)
				auto lambda = BuildPacLambda(std::move(lambda_body), std::move(captures));

				unique_ptr<Expression> final_expr;
				LogicalType final_type;

				if (is_filter_pattern_projection) {
					// For filter pattern projections, keep as LIST<target_type> - no pac_noised
					// The filter rewrite will combine these with list_zip + list_transform + pac_filter
					auto list_transform =
					    BuildListTransformCall(input, std::move(zipped_list), std::move(lambda), target_type);
					final_expr = std::move(list_transform);
					final_type = LogicalType::LIST(target_type);
				} else {
					// Build: list_transform(zipped_list, lambda) -> LIST<DOUBLE> for pac_noised
					auto list_transform =
					    BuildListTransformCall(input, std::move(zipped_list), std::move(lambda), LogicalType::DOUBLE);
					// Wrap with pac_noised to get scalar result
					auto noised = BuildPacNoisedCall(input, std::move(list_transform));
					// Cast to target type if needed
					if (target_type != LogicalType::DOUBLE) {
						noised = BoundCastExpression::AddDefaultCastToType(std::move(noised), target_type);
					}
					final_expr = std::move(noised);
					final_type = target_type;
				}

#if PAC_DEBUG
				PAC_DEBUG_PRINT("  Rewritten to: " + final_expr->ToString());
#endif

				// Replace the expression
				proj.expressions[i] = std::move(final_expr);
				proj.types[i] = final_type;
			}
		}
	}

	// Recurse into children
	for (auto &child : op->children) {
		RewriteProjectionsWithCounters(child.get(), input, plan_root, categorical_aggregates, patterns);
	}
}

// Propagate LIST<DOUBLE> type through the plan for a given binding
static void PropagateCountersType(const ColumnBinding &binding, LogicalOperator *plan_root) {
	auto *source_op = FindOperatorByTableIndex(plan_root, binding.table_index);
	if (!source_op) {
		return;
	}

	if (source_op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = source_op->Cast<LogicalProjection>();
		if (binding.column_index < proj.expressions.size()) {
			proj.expressions[binding.column_index]->return_type = LogicalType::LIST(LogicalType::DOUBLE);
			if (binding.column_index < proj.types.size()) {
				proj.types[binding.column_index] = LogicalType::LIST(LogicalType::DOUBLE);
			}
		}
	} else if (source_op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &agg = source_op->Cast<LogicalAggregate>();
		if (binding.table_index == agg.aggregate_index && binding.column_index < agg.expressions.size()) {
			agg.expressions[binding.column_index]->return_type = LogicalType::LIST(LogicalType::DOUBLE);
			idx_t types_idx = agg.groups.size() + binding.column_index;
			if (types_idx < agg.types.size()) {
				agg.types[types_idx] = LogicalType::LIST(LogicalType::DOUBLE);
			}
		}
	}
}

void RewriteCategoricalQuery(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                             vector<CategoricalPatternInfo> &patterns) {
	// Save the original output types before transformation
	// We'll need these to cast back at the end to match what the result collector expects
	vector<LogicalType> original_output_types = plan->types;

	// Step 1: Find which aggregates are part of categorical patterns
	// Only these aggregates should be converted to _counters variants
	unordered_set<LogicalAggregate *> categorical_aggregates;
	FindCategoricalAggregates(plan.get(), patterns, categorical_aggregates);

#if PAC_DEBUG
	PAC_DEBUG_PRINT("Found " + std::to_string(categorical_aggregates.size()) + " categorical aggregate(s) to convert");
#endif

	// Step 2: Replace PAC aggregates with counters variants
	// When a categorical pattern is detected, we convert ALL PAC aggregates in the plan
	// because the comparison may involve nested subqueries with PAC aggregates
	// that need to return counters for proper element-wise comparison
	ReplacePacAggregatesWithCounters(plan.get(), input.context, nullptr); // nullptr = convert ALL

	// Step 2b: Resolve operator types after counters replacement
	plan->ResolveOperatorTypes();

	// Step 2c: Replace standard aggregates (sum, avg, count, min, max) that now operate on
	// PAC counter results with pac_*_list variants
	// This must happen BEFORE projection rewriting so that projections see the correct aggregate types
	// Process bottom-up so child aggregates are converted before checking parent aggregates
	PAC_DEBUG_PRINT("=== Checking for aggregates over counters ===");
	ReplaceAggregatesOverCountersBottomUp(plan.get(), input.context, plan.get());
	// Re-resolve types after this replacement
	plan->ResolveOperatorTypes();

	// Step 2d: Strip scalar subquery wrappers that contain PAC aggregates
	// These wrappers (CASE->first->count*) are just checks that DuckDB adds for scalar subqueries,
	// but we know PAC aggregates always return exactly one row, so we strip them
	// Search the entire plan for wrapper patterns containing PAC aggregates
	PAC_DEBUG_PRINT("=== Stripping scalar subquery wrappers ===");
	std::function<void(LogicalOperator *)> stripWrappersRecursive = [&](LogicalOperator *op) {
		if (!op) {
			return;
		}

		// Check each child - if it's a wrapper, replace it in place
		for (auto &child : op->children) {
			if (child && child->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto *unwrapped = RecognizeDuckDBScalarWrapper(child.get());
				if (unwrapped) {
					string pac_name = FindPacAggregateInOperator(unwrapped);
					if (!pac_name.empty()) {
						PAC_DEBUG_PRINT("Found scalar wrapper containing PAC aggregate: " + pac_name);
						StripScalarWrapperInPlace(child);
					}
				}
			}
		}

		// Recurse into children (after potential stripping)
		for (auto &child : op->children) {
			stripWrappersRecursive(child.get());
		}
	};
	stripWrappersRecursive(plan.get());
	plan->ResolveOperatorTypes();

	// Step 3: Rewrite projections that do arithmetic with PAC aggregates
	// e.g., 0.5 * agg_result becomes list_transform(agg_result, elem -> 0.5 * elem)
	// Note: Skip projections that are referenced by filter patterns - those are handled in Step 5
	RewriteProjectionsWithCounters(plan.get(), input, plan.get(), categorical_aggregates, patterns);

	// Step 4: Resolve operator types after projection replacement
	plan->ResolveOperatorTypes();

	// Step 4c: Sync all column ref types to match their source operators
	// This ensures that after aggregate conversions, all references have correct types
	PAC_DEBUG_PRINT("=== Syncing column ref types ===");
	SyncColumnRefTypes(plan.get(), plan.get());
	plan->ResolveOperatorTypes();

	// Step 4d: Strip invalid CASTs that have LIST children but scalar cast operations
	PAC_DEBUG_PRINT("=== Stripping invalid CASTs ===");
	StripInvalidCastsInPlan(plan.get());
	plan->ResolveOperatorTypes();

#if PAC_DEBUG
	PAC_DEBUG_PRINT("=== PLAN AFTER COUNTERS REPLACEMENT ===");
	plan->Print();
#endif

	// Step 5: Lambda-based filter rewriting with support for multiple PAC aggregates
	// For single PAC aggregate: double-lambda approach (cast then predicate)
	// For multiple PAC aggregates: list_zip + single lambda approach
	//
	// Single aggregate (n=1):
	//   WHERE bool_expr(PAC_AGG)
	//   -> WHERE pac_filter(
	//        list_transform(
	//          list_transform(PAC_AGG_COUNTERS, elem -> CAST(elem AS original_type)),
	//          elem -> bool_expr(elem)
	//        )
	//      )
	//
	// Multiple aggregates (n>1):
	//   WHERE bool_expr(PAC_AGG1, PAC_AGG2, ...)
	//   -> WHERE pac_filter(
	//        list_transform(
	//          list_zip(A1_counters, A2_counters, ...),  -- LIST<STRUCT<a DOUBLE, b DOUBLE, ...>>
	//          elem -> bool_expr(CAST(elem.a, type1), CAST(elem.b, type2), ...)
	//        )
	//      )

	// Process each filter pattern
	// Note: Multiple patterns may refer to the same filter expression
	// We need to process each unique filter expression only once
	unordered_set<Expression *> processed_filters;

	for (auto &pattern : patterns) {
		if (!pattern.parent_op) {
			continue;
		}

		PAC_DEBUG_PRINT("Processing pattern: parent_op=" + LogicalOperatorToString(pattern.parent_op->type) +
		                ", expr_index=" + std::to_string(pattern.expr_index) + ", aggregate=" + pattern.aggregate_name);

		if (pattern.parent_op->type == LogicalOperatorType::LOGICAL_FILTER) {
			auto &filter = pattern.parent_op->Cast<LogicalFilter>();
			auto &filter_expr = filter.expressions[pattern.expr_index];

			// Skip if we've already processed this filter expression
			if (processed_filters.find(filter_expr.get()) != processed_filters.end()) {
				continue;
			}
			processed_filters.insert(filter_expr.get());

			// Find ALL PAC bindings in this filter expression
			auto pac_bindings = FindAllPacBindingsInExpression(filter_expr.get(), plan.get());

#if PAC_DEBUG
			PAC_DEBUG_PRINT("Filter expression has " + std::to_string(pac_bindings.size()) + " PAC binding(s)");
			PAC_DEBUG_PRINT("  Filter expr: " + filter_expr->ToString());
#endif

			if (pac_bindings.empty()) {
				continue;
			}

			// Propagate LIST<DOUBLE> type for all PAC bindings
			for (auto &binding_info : pac_bindings) {
				PropagateCountersType(binding_info.binding, plan.get());
			}

			if (pac_bindings.size() == 1) {
				// ============================================================
				// SINGLE AGGREGATE: Use existing double-lambda approach
				// ============================================================
				auto &binding_info = pac_bindings[0];
				ColumnBinding pac_binding = binding_info.binding;

				// Use the pattern's original_return_type if available, otherwise use binding's type
				LogicalType original_type = pattern.original_return_type.id() != LogicalTypeId::INVALID
				                                ? pattern.original_return_type
				                                : binding_info.original_type;

#if PAC_DEBUG
				PAC_DEBUG_PRINT("Single-aggregate double-lambda rewrite:");
				PAC_DEBUG_PRINT("  PAC binding: [" + std::to_string(pac_binding.table_index) + "." +
				                std::to_string(pac_binding.column_index) + "]");
				PAC_DEBUG_PRINT("  Original type: " + original_type.ToString());
				PAC_DEBUG_PRINT("  Aggregate: " + binding_info.aggregate_name);
#endif

				// Create counters expression
				auto counters_expr = make_uniq<BoundColumnRefExpression>(
				    "pac_counters", LogicalType::LIST(LogicalType::DOUBLE), pac_binding);

				// Inner lambda: elem -> CAST(elem AS original_type)
				// Note: Scaling by 2.0 is now done in the _counters functions themselves
				auto inner_elem_ref = make_uniq<BoundReferenceExpression>("elem", LogicalType::DOUBLE, 0);
				unique_ptr<Expression> inner_body = std::move(inner_elem_ref);

				// Cast to original type if needed
				if (original_type != LogicalType::DOUBLE) {
					inner_body = BoundCastExpression::AddDefaultCastToType(std::move(inner_body), original_type);
				}

				auto inner_lambda = BuildPacLambda(std::move(inner_body), {});
				auto cast_list =
				    BuildListTransformCall(input, std::move(counters_expr), std::move(inner_lambda), original_type);

				// Outer lambda: elem -> bool_expr(elem)
				vector<unique_ptr<Expression>> captures;
				unordered_map<uint64_t, idx_t> capture_map;

				auto outer_body = CloneForLambdaBody(filter_expr.get(), pac_binding, captures, capture_map, plan.get(),
				                                     original_type);

				auto outer_lambda = BuildPacLambda(std::move(outer_body), std::move(captures));
				auto list_bool =
				    BuildListTransformCall(input, std::move(cast_list), std::move(outer_lambda), LogicalType::BOOLEAN);

				// Wrap with pac_filter
				auto pac_filter = input.optimizer.BindScalarFunction("pac_filter", std::move(list_bool));
				filter_expr = std::move(pac_filter);

#if PAC_DEBUG
				PAC_DEBUG_PRINT("  Single-aggregate rewrite complete");
#endif
			} else {
				// ============================================================
				// MULTIPLE AGGREGATES: Use list_zip approach
				// ============================================================
#if PAC_DEBUG
				PAC_DEBUG_PRINT("Multi-aggregate list_zip rewrite:");
				for (auto &bi : pac_bindings) {
					PAC_DEBUG_PRINT("  PAC binding " + std::to_string(bi.index) + ": [" +
					                std::to_string(bi.binding.table_index) + "." +
					                std::to_string(bi.binding.column_index) + "] " + bi.aggregate_name);
				}
#endif

				// Build counter expressions for each PAC binding
				vector<unique_ptr<Expression>> counter_lists;
				unordered_map<uint64_t, idx_t> binding_to_index;
				unordered_map<uint64_t, LogicalType> binding_to_type;

				for (auto &binding_info : pac_bindings) {
					auto counters_expr = make_uniq<BoundColumnRefExpression>(
					    "pac_counters", LogicalType::LIST(LogicalType::DOUBLE), binding_info.binding);
					counter_lists.push_back(std::move(counters_expr));

					uint64_t hash = HashBinding(binding_info.binding);
					binding_to_index[hash] = binding_info.index;
					binding_to_type[hash] = binding_info.original_type;
				}

				// Build list_zip(A1, A2, ...) -> LIST<STRUCT<a DOUBLE, b DOUBLE, ...>>
				LogicalType struct_type;
				auto zipped_list = BuildListZipCall(input, std::move(counter_lists), struct_type);

				if (!zipped_list) {
#if PAC_DEBUG
					PAC_DEBUG_PRINT("  Failed to build list_zip call");
#endif
					continue;
				}

#if PAC_DEBUG
				PAC_DEBUG_PRINT("  Built list_zip with struct type: " + struct_type.ToString());
#endif

				// Build lambda body that:
				// 1. Extracts each struct field (elem.a, elem.b, ...)
				// 2. Casts to original type
				// 3. Evaluates the filter expression
				vector<unique_ptr<Expression>> captures;
				unordered_map<uint64_t, idx_t> capture_map;

				auto lambda_body = CloneForLambdaBodyMulti(filter_expr.get(), binding_to_index, binding_to_type,
				                                           captures, capture_map, plan.get(), struct_type);

#if PAC_DEBUG
				PAC_DEBUG_PRINT("  Lambda body created with " + std::to_string(captures.size()) + " captures");
				PAC_DEBUG_PRINT("  Lambda body: " + lambda_body->ToString());
#endif

				// Build lambda: elem -> bool_expr(elem.a, elem.b, ...)
				auto lambda = BuildPacLambda(std::move(lambda_body), std::move(captures));

				// Build: list_transform(zipped_list, lambda) -> LIST<BOOL>
				auto list_bool =
				    BuildListTransformCall(input, std::move(zipped_list), std::move(lambda), LogicalType::BOOLEAN);

				// Wrap with pac_filter
				auto pac_filter = input.optimizer.BindScalarFunction("pac_filter", std::move(list_bool));
				filter_expr = std::move(pac_filter);

#if PAC_DEBUG
				PAC_DEBUG_PRINT("  Multi-aggregate rewrite complete");
#endif
			}
		}
		// Handle COMPARISON_JOIN patterns
		else if (pattern.parent_op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
		         pattern.parent_op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			auto &join = pattern.parent_op->Cast<LogicalComparisonJoin>();
			if (pattern.expr_index >= join.conditions.size()) {
				continue;
			}

			auto &cond = join.conditions[pattern.expr_index];

			// Find PAC bindings in BOTH sides of the join condition
			// After conversion, both sides may return LIST<DOUBLE>
			auto left_bindings = FindAllPacBindingsInExpression(cond.left.get(), plan.get());
			auto right_bindings = FindAllPacBindingsInExpression(cond.right.get(), plan.get());

			PAC_DEBUG_PRINT("COMPARISON_JOIN rewrite:");
			PAC_DEBUG_PRINT("  Left: " + cond.left->ToString() + " type=" + cond.left->return_type.ToString() + " (" +
			                std::to_string(left_bindings.size()) + " PAC bindings)");
			PAC_DEBUG_PRINT("  Right: " + cond.right->ToString() + " type=" + cond.right->return_type.ToString() +
			                " (" + std::to_string(right_bindings.size()) + " PAC bindings)");

			if (left_bindings.empty() && right_bindings.empty()) {
				PAC_DEBUG_PRINT("  No PAC bindings found, skipping");
				continue;
			}

			// Propagate LIST<DOUBLE> type for all PAC bindings
			for (auto &bi : left_bindings) {
				PropagateCountersType(bi.binding, plan.get());
			}
			for (auto &bi : right_bindings) {
				PropagateCountersType(bi.binding, plan.get());
			}

			// After type propagation, check if both sides are now LIST<DOUBLE>
			// Resolve types first
			plan->ResolveOperatorTypes();

			// Check based on return type, PAC bindings, or tracing
			// Also check if expression is/contains a SUBQUERY that traces to PAC counters
			bool left_is_list = cond.left->return_type.id() == LogicalTypeId::LIST || !left_bindings.empty() ||
			                    TracesPacCountersAggregate(cond.left.get(), plan.get());
			bool right_is_list = cond.right->return_type.id() == LogicalTypeId::LIST || !right_bindings.empty() ||
			                     TracesPacCountersAggregate(cond.right.get(), plan.get());

			PAC_DEBUG_PRINT("    Right expr type: " + ExpressionTypeToString(cond.right->type));

			// In categorical mode, we've converted all PAC aggregates to _counters and regular
			// aggregates over counters to _list variants. Both sides of the comparison are likely
			// lists now. If either side is a subquery or column ref to an aggregate, treat as list.
			// This is safe because we only reach here when we detected a categorical pattern.
			if (!left_is_list) {
				// If left has PAC aggregate bindings or is/contains a subquery, treat as list
				if (cond.left->type == ExpressionType::SUBQUERY ||
				    (cond.left->type == ExpressionType::OPERATOR_CAST &&
				     cond.left->Cast<BoundCastExpression>().child->type == ExpressionType::SUBQUERY)) {
					left_is_list = true;
					PAC_DEBUG_PRINT("  Treating left as list (SUBQUERY in categorical mode)");
				}
			}
			if (!right_is_list) {
				// Same for right side - trace through projections to find if it comes from an aggregate
				if (cond.right->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = cond.right->Cast<BoundColumnRefExpression>();
					// Use TracesPacCountersAggregate which already handles projection tracing
					// Note: We need to re-check after conversions since avg->pac_avg_list may have happened
					// For now, if we're in categorical mode and this is a col ref, assume it might be a list
					// if it traces to ANY aggregate (which would now be converted)
					auto *source = FindOperatorByTableIndex(plan.get(), col_ref.binding.table_index);
					if (source) {
						// Trace through projections to find actual source
						ColumnBinding traced = col_ref.binding;
						while (source && source->type == LogicalOperatorType::LOGICAL_PROJECTION) {
							auto &proj = source->Cast<LogicalProjection>();
							if (traced.column_index < proj.expressions.size()) {
								auto &expr = proj.expressions[traced.column_index];
								PAC_DEBUG_PRINT("    Proj expr type: " + ExpressionTypeToString(expr->type) + " = " +
								                expr->ToString());
								if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
									traced = expr->Cast<BoundColumnRefExpression>().binding;
									source = FindOperatorByTableIndex(plan.get(), traced.table_index);
								} else {
									// Not a simple column ref - check if it's from an aggregate child
									// In categorical mode, assume any complex expression from a projection
									// that traces here is likely LIST<DOUBLE>
									if (!source->children.empty()) {
										auto *child_source = source->children[0].get();
										if (child_source &&
										    child_source->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
											source = child_source;
											PAC_DEBUG_PRINT("    Found aggregate in projection child");
										}
									}
									break;
								}
							} else {
								break;
							}
						}
						PAC_DEBUG_PRINT(
						    "  Traced right to: " + (source ? LogicalOperatorToString(source->type) : "null") +
						    " binding=[" + std::to_string(traced.table_index) + "." +
						    std::to_string(traced.column_index) + "]");
						if (source && source->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
							right_is_list = true;
							PAC_DEBUG_PRINT("  Treating right as list (aggregate result in categorical mode)");
						}
					}
				} else if (cond.right->type == ExpressionType::SUBQUERY ||
				           (cond.right->type == ExpressionType::OPERATOR_CAST &&
				            cond.right->Cast<BoundCastExpression>().child->type == ExpressionType::SUBQUERY)) {
					right_is_list = true;
					PAC_DEBUG_PRINT("  Treating right as list (SUBQUERY in categorical mode)");
				}
			}

			PAC_DEBUG_PRINT("  After ResolveOperatorTypes for categorical pattern:");
			PAC_DEBUG_PRINT("    Left return_type: " + cond.left->return_type.ToString());
			PAC_DEBUG_PRINT("    Right return_type: " + cond.right->return_type.ToString());

			PAC_DEBUG_PRINT("  left_is_list=" + std::string(left_is_list ? "true" : "false") +
			                ", right_is_list=" + std::string(right_is_list ? "true" : "false"));

			unique_ptr<Expression> pac_filter_result;

			if (left_is_list && right_is_list) {
				// BOTH sides are lists from PAC aggregates.
				// Convert COMPARISON_JOIN to CROSS_PRODUCT + FILTER because:
				// - The pac_filter expression needs to reference bindings from BOTH sides
				// - Join conditions can only access one side's bindings at a time during resolution
				// - CROSS_PRODUCT makes all bindings available, FILTER can access them all
				PAC_DEBUG_PRINT("  Two-list comparison: converting to CROSS_PRODUCT + FILTER");

				// Update types to LIST<DOUBLE>
				UpdateExpressionTypesToList(cond.left.get(), plan.get());
				UpdateExpressionTypesToList(cond.right.get(), plan.get());

				if (cond.left->return_type.id() != LogicalTypeId::LIST) {
					cond.left->return_type = LogicalType::LIST(LogicalType::DOUBLE);
				}
				if (cond.right->return_type.id() != LogicalTypeId::LIST) {
					cond.right->return_type = LogicalType::LIST(LogicalType::DOUBLE);
				}

				// Sync types in the expressions
				SyncColumnRefTypesInExpression(cond.left.get(), plan.get());
				SyncColumnRefTypesInExpression(cond.right.get(), plan.get());

				// Make sure types are LIST before proceeding
				if (cond.left->return_type.id() != LogicalTypeId::LIST ||
				    cond.right->return_type.id() != LogicalTypeId::LIST) {
					PAC_DEBUG_PRINT("  WARNING: Types not LIST after update, skipping");
					continue;
				}

				// Unwrap unnecessary CAST expressions
				unique_ptr<Expression> left_expr;
				unique_ptr<Expression> right_expr;

				if (cond.left->type == ExpressionType::OPERATOR_CAST) {
					auto &cast = cond.left->Cast<BoundCastExpression>();
					if (cast.child->return_type.id() == LogicalTypeId::LIST) {
						left_expr = cast.child->Copy();
						PAC_DEBUG_PRINT("  Stripped CAST from left side");
					} else {
						left_expr = cond.left->Copy();
					}
				} else {
					left_expr = cond.left->Copy();
				}

				if (cond.right->type == ExpressionType::OPERATOR_CAST) {
					auto &cast = cond.right->Cast<BoundCastExpression>();
					if (cast.child->return_type.id() == LogicalTypeId::LIST) {
						right_expr = cast.child->Copy();
						PAC_DEBUG_PRINT("  Stripped CAST from right side");
					} else {
						right_expr = cond.right->Copy();
					}
				} else {
					right_expr = cond.right->Copy();
				}

				// Build list_zip(left_list, right_list) -> LIST<STRUCT(a: DOUBLE, b: DOUBLE)>
				PAC_DEBUG_PRINT("  list_zip left_expr: " + left_expr->ToString() +
				                " type: " + ExpressionTypeToString(left_expr->type));
				PAC_DEBUG_PRINT("  list_zip right_expr: " + right_expr->ToString() +
				                " type: " + ExpressionTypeToString(right_expr->type));
				vector<unique_ptr<Expression>> zip_args;
				zip_args.push_back(std::move(left_expr));
				zip_args.push_back(std::move(right_expr));

				LogicalType struct_type;
				auto zipped_list = BuildListZipCall(input, std::move(zip_args), struct_type);
				if (!zipped_list) {
					PAC_DEBUG_PRINT("  Failed to build list_zip call");
					continue;
				}
				PAC_DEBUG_PRINT("  Built list_zip: " + zipped_list->ToString());

				// Build lambda body: x -> x.a <= x.b
				auto elem_ref = make_uniq<BoundReferenceExpression>("x", struct_type, 0);

				// Use StructExtractAtFun to extract struct fields by index
				// Field "a" is at index 0, field "b" is at index 1
				auto extract_func = StructExtractAtFun::GetFunction();

				// Extract x.a (field at index 0)
				auto bind_data_a = StructExtractAtFun::GetBindData(0);
				vector<unique_ptr<Expression>> extract_a_args;
				extract_a_args.push_back(elem_ref->Copy());
				// struct_extract_at takes a 1-based index argument
				extract_a_args.push_back(make_uniq<BoundConstantExpression>(Value::BIGINT(1)));
				auto field_a = make_uniq<BoundFunctionExpression>(LogicalType::DOUBLE, extract_func,
				                                                  std::move(extract_a_args), std::move(bind_data_a));

				// Extract x.b (field at index 1)
				auto bind_data_b = StructExtractAtFun::GetBindData(1);
				vector<unique_ptr<Expression>> extract_b_args;
				extract_b_args.push_back(elem_ref->Copy());
				extract_b_args.push_back(make_uniq<BoundConstantExpression>(Value::BIGINT(2)));
				auto field_b = make_uniq<BoundFunctionExpression>(LogicalType::DOUBLE, extract_func,
				                                                  std::move(extract_b_args), std::move(bind_data_b));

				// Build comparison expression
				auto compare_expr =
				    make_uniq<BoundComparisonExpression>(cond.comparison, std::move(field_a), std::move(field_b));
				PAC_DEBUG_PRINT("  Lambda body (compare_expr): " + compare_expr->ToString());
				PAC_DEBUG_PRINT("  Lambda body type: " + ExpressionTypeToString(compare_expr->type));

				// Build lambda and list_transform
				vector<unique_ptr<Expression>> captures;
				auto lambda = BuildPacLambda(std::move(compare_expr), std::move(captures));
				PAC_DEBUG_PRINT("  Built lambda: " + lambda->ToString());

				auto list_bool =
				    BuildListTransformCall(input, std::move(zipped_list), std::move(lambda), LogicalType::BOOLEAN);
				if (!list_bool) {
					PAC_DEBUG_PRINT("  Failed to build list_transform call");
					continue;
				}

				// Wrap with pac_filter
				auto pac_filter_expr = input.optimizer.BindScalarFunction("pac_filter", std::move(list_bool));
				PAC_DEBUG_PRINT("  Built pac_filter: " + pac_filter_expr->ToString());

				// Debug: print full expression tree
				std::function<void(Expression *, int)> printExprTree = [&](Expression *e, int depth) {
					if (!e) {
						return;
					}
					string indent(static_cast<size_t>(depth) * 2, ' ');
					PAC_DEBUG_PRINT(indent + "- " + ExpressionTypeToString(e->type) + ": " +
					                e->ToString().substr(0, 80));
					ExpressionIterator::EnumerateChildren(*e,
					                                      [&](Expression &child) { printExprTree(&child, depth + 1); });
				};
				PAC_DEBUG_PRINT("  === Expression tree for pac_filter ===");
				printExprTree(pac_filter_expr.get(), 0);

				// Convert COMPARISON_JOIN to CROSS_PRODUCT + FILTER
				// 1. Create CROSS_PRODUCT from the join's children
				auto &join_op = pattern.parent_op->Cast<LogicalComparisonJoin>();
				auto cross_product =
				    LogicalCrossProduct::Create(std::move(join_op.children[0]), std::move(join_op.children[1]));

				// 2. Create FILTER with pac_filter expression
				auto filter = make_uniq<LogicalFilter>();
				filter->expressions.push_back(std::move(pac_filter_expr));
				filter->children.push_back(std::move(cross_product));

				// 3. Replace the join with the filter in the plan
				// Find the join's parent and replace the join with our filter
				bool replaced = false;
				std::function<void(unique_ptr<LogicalOperator> &)> replace_join;
				replace_join = [&](unique_ptr<LogicalOperator> &op) {
					if (!op || replaced) {
						return;
					}
					for (auto &child : op->children) {
						if (child.get() == pattern.parent_op) {
							child = std::move(filter);
							replaced = true;
							return;
						}
						replace_join(child);
					}
				};

				// Special case: if the join IS the plan root
				if (plan.get() == pattern.parent_op) {
					plan = std::move(filter);
					replaced = true;
				} else {
					replace_join(plan);
				}

				if (!replaced) {
					PAC_DEBUG_PRINT("  WARNING: Failed to replace join with filter");
					// Restore the join's children since we moved them
					continue;
				}

				PAC_DEBUG_PRINT("  Converted COMPARISON_JOIN to CROSS_PRODUCT + FILTER");

				// Skip to next pattern since we've completely replaced this join
				// Don't modify cond since we've replaced the entire operator
				break;
			} else {
				// Only one side is a list (original single-aggregate case)
				PAC_DEBUG_PRINT("  Using single-list comparison");

				auto &pac_bindings = left_is_list ? left_bindings : right_bindings;
				unique_ptr<Expression> &scalar_side = left_is_list ? cond.right : cond.left;

				if (pac_bindings.empty()) {
					PAC_DEBUG_PRINT("  No PAC bindings for list side, skipping");
					continue;
				}

				auto &binding_info = pac_bindings[0];
				LogicalType original_type = binding_info.original_type;

				auto counters_expr = make_uniq<BoundColumnRefExpression>(
				    "pac_counters", LogicalType::LIST(LogicalType::DOUBLE), binding_info.binding);

				// Inner lambda: elem -> CAST(elem AS original_type)
				auto inner_elem_ref = make_uniq<BoundReferenceExpression>("elem", LogicalType::DOUBLE, 0);
				unique_ptr<Expression> inner_body = std::move(inner_elem_ref);
				if (original_type != LogicalType::DOUBLE) {
					inner_body = BoundCastExpression::AddDefaultCastToType(std::move(inner_body), original_type);
				}

				auto inner_lambda = BuildPacLambda(std::move(inner_body), {});
				auto cast_list =
				    BuildListTransformCall(input, std::move(counters_expr), std::move(inner_lambda), original_type);

				// Outer lambda with capture of scalar side
				vector<unique_ptr<Expression>> captures;
				captures.push_back(scalar_side->Copy());

				auto elem_ref = make_uniq<BoundReferenceExpression>("elem", original_type, 0);
				auto captured_ref = make_uniq<BoundReferenceExpression>("captured", scalar_side->return_type, 1);

				unique_ptr<Expression> cmp_left, cmp_right;
				if (left_is_list) {
					cmp_left = std::move(elem_ref);
					cmp_right = std::move(captured_ref);
				} else {
					cmp_left = std::move(captured_ref);
					cmp_right = std::move(elem_ref);
				}

				auto comparison_expr =
				    make_uniq<BoundComparisonExpression>(cond.comparison, std::move(cmp_left), std::move(cmp_right));

				auto outer_lambda = BuildPacLambda(std::move(comparison_expr), std::move(captures));
				auto list_bool =
				    BuildListTransformCall(input, std::move(cast_list), std::move(outer_lambda), LogicalType::BOOLEAN);

				pac_filter_result = input.optimizer.BindScalarFunction("pac_filter", std::move(list_bool));
			}

			PAC_DEBUG_PRINT("  Replacing join condition...");
			// Replace the join condition with the pac_filter result
			// The condition becomes: pac_filter(...) = TRUE
			// Or we can put pac_filter directly as left side with TRUE on right for EQUAL comparison
			cond.left = std::move(pac_filter_result);
			cond.right = make_uniq<BoundConstantExpression>(Value::BOOLEAN(true));
			cond.comparison = ExpressionType::COMPARE_EQUAL;

			PAC_DEBUG_PRINT("  COMPARISON_JOIN rewrite complete");
			PAC_DEBUG_PRINT("  New condition: " + cond.left->ToString() + " = " + cond.right->ToString());
		}
	}

	// Final resolve after all modifications
	plan->ResolveOperatorTypes();

	// Step 6: Ensure plan output types match original expected types
	// After categorical rewriting, some outputs may be LIST<DOUBLE> or DOUBLE instead of
	// the original types (e.g., DECIMAL). We need to wrap with pac_noised and cast as needed.
	bool types_changed = false;
	for (idx_t i = 0; i < plan->types.size() && i < original_output_types.size(); i++) {
		if (plan->types[i] != original_output_types[i]) {
			types_changed = true;
			break;
		}
	}

	if (types_changed) {
		PAC_DEBUG_PRINT("Plan output types changed - adding pac_noised and casts to restore original types");

		// Build expressions for a new projection that converts outputs to original types
		vector<unique_ptr<Expression>> proj_expressions;
		auto bindings = plan->GetColumnBindings();
		for (idx_t i = 0; i < plan->types.size(); i++) {
			auto col_ref = make_uniq<BoundColumnRefExpression>(plan->types[i], bindings[i]);
			unique_ptr<Expression> final_expr;

			if (plan->types[i].id() == LogicalTypeId::LIST) {
				// Wrap with pac_noised to convert LIST<DOUBLE> to DOUBLE
				final_expr = input.optimizer.BindScalarFunction("pac_noised", std::move(col_ref));
			} else {
				final_expr = std::move(col_ref);
			}

			// Cast to original type if different
			LogicalType target_type = (i < original_output_types.size()) ? original_output_types[i] : plan->types[i];
			if (final_expr->return_type != target_type) {
				final_expr = BoundCastExpression::AddDefaultCastToType(std::move(final_expr), target_type);
			}

			proj_expressions.push_back(std::move(final_expr));
		}

		// Create a projection to restore original types
		auto proj =
		    make_uniq<LogicalProjection>(input.optimizer.binder.GenerateTableIndex(), std::move(proj_expressions));

		// Move the current plan as child of the projection
		proj->children.push_back(std::move(plan));
		proj->ResolveOperatorTypes();
		plan = std::move(proj);
	}

#if PAC_DEBUG
	// Debug: Print types of all projections to see if our changes persisted
	PAC_DEBUG_PRINT("=== FINAL TYPE CHECK ===");
	std::function<void(LogicalOperator *, int)> printTypes = [&](LogicalOperator *op, int depth) {
		if (!op) {
			return;
		}
		string indent(static_cast<size_t>(depth) * 2, ' ');
		if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			auto &proj = op->Cast<LogicalProjection>();
			PAC_DEBUG_PRINT(indent + "PROJECTION #" + std::to_string(proj.table_index) + " types:");
			for (idx_t i = 0; i < proj.types.size(); i++) {
				PAC_DEBUG_PRINT(indent + "  [" + std::to_string(i) + "] = " + proj.types[i].ToString());
			}
		} else if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
			auto &agg = op->Cast<LogicalAggregate>();
			PAC_DEBUG_PRINT(indent + "AGGREGATE #" + std::to_string(agg.aggregate_index) + " types:");
			for (idx_t i = 0; i < agg.types.size(); i++) {
				PAC_DEBUG_PRINT(indent + "  [" + std::to_string(i) + "] = " + agg.types[i].ToString());
			}
		} else if (op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			auto &join = op->Cast<LogicalComparisonJoin>();
			PAC_DEBUG_PRINT(indent + "DELIM_JOIN types:");
			for (idx_t i = 0; i < op->types.size(); i++) {
				PAC_DEBUG_PRINT(indent + "  [" + std::to_string(i) + "] = " + op->types[i].ToString());
			}
			PAC_DEBUG_PRINT(indent + "DELIM_JOIN conditions:");
			for (idx_t i = 0; i < join.conditions.size(); i++) {
				PAC_DEBUG_PRINT(indent + "  cond[" + std::to_string(i) +
				                "].left type = " + join.conditions[i].left->return_type.ToString());
				PAC_DEBUG_PRINT(indent + "  cond[" + std::to_string(i) +
				                "].right type = " + join.conditions[i].right->return_type.ToString());
			}
		}
		for (auto &child : op->children) {
			printTypes(child.get(), depth + 1);
		}
	};
	printTypes(plan.get(), 0);
#endif
}

} // namespace duckdb
