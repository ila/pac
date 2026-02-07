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
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"

namespace duckdb {

// Forward declarations
static string FindPacAggregateInExpression(Expression *expr, LogicalOperator *plan_root);
static string TracePacAggregateFromBinding(const ColumnBinding &binding, LogicalOperator *plan_root);
static LogicalAggregate *FindAggregateForBinding(const ColumnBinding &binding, LogicalOperator *plan_root);
static vector<PacBindingInfo> FindAllPacBindingsInExpression(Expression *expr, LogicalOperator *plan_root);
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
	for (auto &child : op->children) {
		auto *result = FindOperatorByTableIndex(child.get(), table_index);
		if (result) {
			return result;
		}
	}
	return nullptr;
}

// Recognize DuckDB's scalar subquery wrapper pattern
//
// Pattern  (uncorrelated): Projection -> Aggregate(first) -> Projection
//   Projection (CASE error if count > 1, else first(value))
//   └── Aggregate (first(#0), count_star())
//       └── Projection (#0)
//           └── [actual scalar subquery result]
static LogicalOperator *RecognizeDuckDBScalarWrapper(LogicalOperator *op) {
	auto &outer_proj = op->Cast<LogicalProjection>();
	if (outer_proj.expressions.empty()) {
		return nullptr;
	}
	// expressions[0] is the CASE wrapper: CASE WHEN count>1 THEN error(...) ELSE first(...) END
	auto &expr = outer_proj.expressions[0];
	if (expr->type != ExpressionType::CASE_EXPR) {
		return nullptr;
	}
	auto &case_expr = expr->Cast<BoundCaseExpression>();
	if (case_expr.case_checks.empty()) {
		return nullptr;
	}
	// case_checks[0].then_expr is the error() call
	auto &case_check = case_expr.case_checks[0];
	if (!case_check.then_expr || case_check.then_expr->type != ExpressionType::BOUND_FUNCTION) {
		return nullptr;
	}
	auto &func = case_check.then_expr->Cast<BoundFunctionExpression>();
	if (StringUtil::Lower(func.function.name) != "error") {
		return nullptr;
	}
	// Child is the aggregate: first(value), count_star()
	if (outer_proj.children.empty()) {
		return nullptr;
	}
	auto *child = outer_proj.children[0].get();
	if (child->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return nullptr;
	}
	auto &agg = child->Cast<LogicalAggregate>();
	if (agg.expressions.empty()) {
		return nullptr;
	}
	// expressions[0] is the first() aggregate
	auto &agg_expr = agg.expressions[0];
	if (agg_expr->type != ExpressionType::BOUND_AGGREGATE) {
		return nullptr;
	}
	auto &bound_agg = agg_expr->Cast<BoundAggregateExpression>();
	if (StringUtil::Lower(bound_agg.function.name) != "first") {
		return nullptr;
	}
	// Inner projection below the aggregate
	if (agg.children.empty()) {
		return nullptr;
	}
	auto *inner_proj_op = agg.children[0].get();
	if (inner_proj_op->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return nullptr;
	}
	auto &inner_proj = inner_proj_op->Cast<LogicalProjection>();
	return inner_proj.children.empty() ? nullptr : inner_proj.children[0].get();
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
			return FindPacAggregateInExpression(proj.expressions[binding.column_index].get(), plan_root);
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
					                " is_pac=" + (IsAnyPacAggregate(bound_agg.function.name) ? "yes" : "no"));
					// Check for both original and counters variants
					if (IsAnyPacAggregate(bound_agg.function.name)) {
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
				if (IsAnyPacAggregate(bound_agg.function.name)) {
					return GetBasePacAggregateName(bound_agg.function.name);
				}
			}
		}
	}
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
static string FindPacAggregateInExpression(Expression *expr, LogicalOperator *plan_root) {
	if (!expr) {
		return "";
	}

	// Base case: direct PAC aggregate
	if (expr->type == ExpressionType::BOUND_AGGREGATE) {
		auto &agg_expr = expr->Cast<BoundAggregateExpression>();
		if (IsAnyPacAggregate(agg_expr.function.name)) {
			return GetBasePacAggregateName(agg_expr.function.name);
		}
		// Fall through to check children below
	}

	// Column references need plan-aware tracing
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();
		return TracePacAggregateFromBinding(col_ref.binding, plan_root);
	}

	// Subqueries need special plan-root heuristic search
	if (expr->type == ExpressionType::SUBQUERY) {
		auto &subquery_expr = expr->Cast<BoundSubqueryExpression>();
		// Search the subquery's children (for IN, ANY, ALL operators)
		for (auto &child : subquery_expr.children) {
			string result = FindPacAggregateInExpression(child.get(), plan_root);
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
							if (IsAnyPacAggregate(bound_agg.function.name)) {
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

	// Generic traversal for all other expression types (comparisons, operators, casts,
	// functions, constants, CASE, BETWEEN, conjunctions, window functions, etc.)
	string result;
	ExpressionIterator::EnumerateChildren(*expr, [&](Expression &child) {
		if (result.empty()) {
			result = FindPacAggregateInExpression(&child, plan_root);
		}
	});
	return result;
}

// Plan-aware version: find PAC aggregate in expression and populate pattern info
// For comparisons, identifies which side has the PAC aggregate (left=0, right=1)
// For other expression types, searches the entire expression
static bool IsComparisonWithPacAggregate(Expression *expr, CategoricalPatternInfo &info, LogicalOperator *plan_root) {
	if (!expr) {
		return false;
	}

	// For comparison expressions, identify which side has the PAC aggregate
	if (expr->type == ExpressionType::COMPARE_GREATERTHAN || expr->type == ExpressionType::COMPARE_LESSTHAN ||
	    expr->type == ExpressionType::COMPARE_GREATERTHANOREQUALTO ||
	    expr->type == ExpressionType::COMPARE_LESSTHANOREQUALTO || expr->type == ExpressionType::COMPARE_EQUAL ||
	    expr->type == ExpressionType::COMPARE_NOTEQUAL) {
		auto &comp_expr = expr->Cast<BoundComparisonExpression>();

		// Check left side for PAC aggregate
		string left_pac = FindPacAggregateInExpression(comp_expr.left.get(), plan_root);
		if (!left_pac.empty()) {
			info.comparison_expr = expr;
			info.subquery_expr = comp_expr.left.get();
			info.subquery_side = 0;
			info.aggregate_name = left_pac;
			return true;
		}

		// Check right side for PAC aggregate
		string right_pac = FindPacAggregateInExpression(comp_expr.right.get(), plan_root);
		if (!right_pac.empty()) {
			info.comparison_expr = expr;
			info.subquery_expr = comp_expr.right.get();
			info.subquery_side = 1;
			info.aggregate_name = right_pac;
			return true;
		}

		return false;
	}

	// For any other expression type, search the whole expression
	string pac_name = FindPacAggregateInExpression(expr, plan_root);
	if (!pac_name.empty()) {
		info.comparison_expr = expr;
		info.subquery_expr = expr;
		info.subquery_side = 0;
		info.aggregate_name = pac_name;
		return true;
	}

	return false;
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
static void FindCategoricalPatternsInOperator(LogicalOperator *op, LogicalOperator *plan_root,
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
				IsComparisonWithPacAggregate(filter_expr.get(), info, plan_root);

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
			string left_pac = FindPacAggregateInExpression(cond.left.get(), plan_root);
			string right_pac = FindPacAggregateInExpression(cond.right.get(), plan_root);
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

				PAC_DEBUG_PRINT("Found projection-based categorical pattern in expr " + std::to_string(i) + " with " +
				                std::to_string(pac_bindings.size()) + " PAC binding(s)");
				PAC_DEBUG_PRINT("  Expression: " + expr->ToString().substr(0, 80));

				patterns.push_back(info);
			}
		}
	}

	// Recurse into children
	for (auto &child : op->children) {
		FindCategoricalPatternsInOperator(child.get(), plan_root, patterns, now_inside_aggregate);
	}
}

bool IsCategoricalQuery(unique_ptr<LogicalOperator> &plan, vector<CategoricalPatternInfo> &patterns) {
	patterns.clear();
	// Use plan-aware version, passing plan root for column binding tracing
	FindCategoricalPatternsInOperator(plan.get(), plan.get(), patterns, false);
	return !patterns.empty();
}

// Helper to find the aggregate operator that produces a given column binding
static LogicalAggregate *FindAggregateForBinding(const ColumnBinding &binding, LogicalOperator *plan_root) {
	auto *source_op = FindOperatorByTableIndex(plan_root, binding.table_index);
	PAC_DEBUG_PRINT("FindAggregateForBinding: binding=[" + std::to_string(binding.table_index) + "." +
	                std::to_string(binding.column_index) +
	                "], source_op=" + (source_op ? LogicalOperatorToString(source_op->type) : "null"));
	if (!source_op) {
		return nullptr;
	}

	if (source_op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		PAC_DEBUG_PRINT("  Found aggregate directly!");
		return &source_op->Cast<LogicalAggregate>();
	}

	// If it's a projection, trace through to find the aggregate
	if (source_op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = source_op->Cast<LogicalProjection>();
		if (binding.column_index < proj.expressions.size()) {
			auto &expr = proj.expressions[binding.column_index];
			PAC_DEBUG_PRINT("  Projection expr type: " + ExpressionTypeToString(expr->type));
			// Check if this expression is a column reference to an aggregate
			if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
				auto &col_ref = expr->Cast<BoundColumnRefExpression>();
				return FindAggregateForBinding(col_ref.binding, plan_root);
			}
			// Check if it's a cast expression (e.g., CAST(agg_result AS DOUBLE) or CAST(coalesce(agg, 0) AS INTEGER))
			if (expr->type == ExpressionType::OPERATOR_CAST) {
				auto &cast_expr = expr->Cast<BoundCastExpression>();
				PAC_DEBUG_PRINT("  Cast child type: " + ExpressionTypeToString(cast_expr.child->type));
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
					PAC_DEBUG_PRINT("  Checking " + std::to_string(children.size()) + " children for aggregate ref");
					for (auto *child : children) {
						PAC_DEBUG_PRINT("    Child type: " + ExpressionTypeToString(child->type));
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
							PAC_DEBUG_PRINT("    Found BOUND_AGGREGATE: " + agg_expr.function.name);
							if (IsAnyPacAggregate(agg_expr.function.name)) {
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
						PAC_DEBUG_PRINT("Warning: Could not bind " + counters_name + ": " + error.Message());
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
						PAC_DEBUG_PRINT("Rebound aggregate to " + counters_name + " with return type " +
						                new_aggr->return_type.ToString());
						// Replace the expression
						agg.expressions[i] = std::move(new_aggr);
					}

					// Update the aggregate operator's types vector
					idx_t types_index = agg.groups.size() + i;
					if (types_index < agg.types.size()) {
						agg.types[types_index] = LogicalType::LIST(LogicalType::DOUBLE);
						PAC_DEBUG_PRINT("Updated aggregate types[" + std::to_string(types_index) +
						                "] to LIST<DOUBLE> for " + counters_name);
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

// Check if an expression traces back to a PAC _counters aggregate
static bool TracesPacCountersAggregate(Expression *expr, LogicalOperator *plan_root) {
	if (!expr || !plan_root) {
		return false;
	}

	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();
		string pac_name = TracePacAggregateFromBinding(col_ref.binding, plan_root);
		// TracePacAggregateFromBinding uses IsAnyPacAggregate, so it will find _counters too
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
		PAC_DEBUG_PRINT("    CloneForLambdaBody: NO MATCH - traced to [" + std::to_string(traced.table_index) + "." +
		                std::to_string(traced.column_index) + "] - capturing");
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
		bool any_cast_needed = false;
		for (idx_t i = 0; i < func.children.size(); i++) {
			auto child_clone =
			    CloneForLambdaBody(func.children[i].get(), pac_binding, captures, capture_map, plan_root, element_type);
			// If a child's type changed (e.g., DECIMAL->DOUBLE from PAC counter conversion),
			// cast it to the type the bound function expects, so the function binding stays valid.
			if (i < func.function.arguments.size() && child_clone->return_type != func.function.arguments[i]) {
				child_clone =
				    BoundCastExpression::AddDefaultCastToType(std::move(child_clone), func.function.arguments[i]);
				any_cast_needed = true;
			}
			new_children.push_back(std::move(child_clone));
		}
		unique_ptr<Expression> result =
		    make_uniq<BoundFunctionExpression>(func.return_type, func.function, std::move(new_children),
		                                       func.bind_info ? func.bind_info->Copy() : nullptr);
		// When element_type is DOUBLE (raw counters context), children were cast from DOUBLE to
		// the function's bound types (e.g., DECIMAL). Cast the result back to DOUBLE so the
		// list_transform output stays LIST<DOUBLE>. In other contexts (e.g., outer lambda with
		// element_type=DECIMAL), the function's natural return type should be preserved.
		if (any_cast_needed && element_type == LogicalType::DOUBLE && result->return_type != LogicalType::DOUBLE) {
			result = BoundCastExpression::AddDefaultCastToType(std::move(result), LogicalType::DOUBLE);
		}
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

// Wrap a list expression with pac_coalesce(list_expr):
// If list_expr is NULL, returns a LIST<DOUBLE> of 64 NULLs.
// Otherwise returns list_expr unchanged.
// This ensures list_transform gets a valid 64-element list even when the subquery
// produces no rows (LEFT_DELIM_JOIN NULL-fills the counters column).
static unique_ptr<Expression> WrapListNullSafe(unique_ptr<Expression> list_expr, OptimizerExtensionInput &input) {
	return input.optimizer.BindScalarFunction("pac_coalesce", std::move(list_expr));
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

			PAC_DEBUG_PRINT("    CloneForLambdaBodyMulti: PAC binding [" + std::to_string(col_ref.binding.table_index) +
			                "." + std::to_string(col_ref.binding.column_index) + "] -> struct field '" + field_name +
			                "' with type " + field_type.ToString());
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
		bool any_cast_needed = false;
		for (idx_t i = 0; i < func.children.size(); i++) {
			auto child_clone = CloneForLambdaBodyMulti(func.children[i].get(), binding_to_index, binding_to_type,
			                                           captures, capture_map, plan_root, struct_type);
			// If a child's type changed (e.g., DECIMAL->DOUBLE from PAC counter conversion),
			// cast it to the type the bound function expects, so the function binding stays valid.
			if (i < func.function.arguments.size() && child_clone->return_type != func.function.arguments[i]) {
				child_clone =
				    BoundCastExpression::AddDefaultCastToType(std::move(child_clone), func.function.arguments[i]);
				any_cast_needed = true;
			}
			new_children.push_back(std::move(child_clone));
		}
		unique_ptr<Expression> result =
		    make_uniq<BoundFunctionExpression>(func.return_type, func.function, std::move(new_children),
		                                       func.bind_info ? func.bind_info->Copy() : nullptr);
		// If children were cast to preserve the function binding, the return type reflects the
		// original binding (e.g., DECIMAL) rather than DOUBLE from struct field extraction.
		// Cast back to DOUBLE so downstream expressions see consistent types.
		if (any_cast_needed && result->return_type != LogicalType::DOUBLE) {
			result = BoundCastExpression::AddDefaultCastToType(std::move(result), LogicalType::DOUBLE);
		}
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
		PAC_DEBUG_PRINT("Warning: Could not bind list_zip: " + error.Message());
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

// Build a list_transform expression over PAC counter bindings.
//
// For single binding: list_transform(pac_coalesce(counters), elem -> body(elem))
//   If element_type != DOUBLE, inserts an inner cast lambda first (double-lambda approach)
//   so the outer lambda body sees elements of element_type rather than raw DOUBLE.
//
// For multiple bindings: list_transform(list_zip(c1, c2, ...), elem -> body(elem.a, elem.b, ...))
//   CloneForLambdaBodyMulti handles per-field type casting internally.
//
// Returns the list_transform expression, or nullptr on failure.
// The caller wraps with pac_noised or pac_filter as needed.
static unique_ptr<Expression> BuildCounterListTransform(OptimizerExtensionInput &input,
                                                        const vector<PacBindingInfo> &pac_bindings,
                                                        Expression *expr_to_transform, LogicalOperator *plan_root,
                                                        const LogicalType &element_type,
                                                        const LogicalType &result_element_type);

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
	// First recurse into children (BOTTOM-UP: children before current operator)
	// This ensures that when we query GetBindingReturnType for a projection,
	// the projection's column refs have already been updated from their source aggregates
	for (auto &child : op->children) {
		SyncColumnRefTypes(child.get(), plan_root);
	}
	// Update types in ALL expressions of this operator (including operator-specific ones)
	LogicalOperatorVisitor::EnumerateExpressions(*op, [&](unique_ptr<Expression> *expr_ptr) {
		SyncColumnRefTypesInExpression(expr_ptr->get(), plan_root);
	});
}

// Strip invalid CAST expressions from the plan where child is LIST but CAST expects scalar
// Returns true if the expression was modified
static bool StripInvalidCastsInExpression(unique_ptr<Expression> &expr) {
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
	// First recurse into children
	for (auto &child : op->children) {
		StripInvalidCastsInPlan(child.get());
	}
	// Strip invalid CASTs in ALL expressions (including operator-specific ones)
	LogicalOperatorVisitor::EnumerateExpressions(*op, [&](unique_ptr<Expression> *expr_ptr) {
		StripInvalidCastsInExpression(*expr_ptr);
	});
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
		    func.function.name == "pac_coalesce" || func.function.name == "list_transform" ||
		    func.function.name == "list_zip") {
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

// Recognize (and optionally strip) a scalar subquery wrapper operator in place
// Pattern: Project(CASE) #X -> Aggregate(first, count*) -> Project #Z -> [inner]
// When remove=true: deletes outer Project and Aggregate, keeps inner Project with outer's table_index
// Returns the inner operator (below inner projection) if pattern matches, nullptr otherwise
static LogicalOperator *StripScalarWrapperInPlace(unique_ptr<LogicalOperator> &wrapper_ptr, bool remove = true) {
	if (!wrapper_ptr || wrapper_ptr->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return nullptr;
	}

	auto *unwrapped = RecognizeDuckDBScalarWrapper(wrapper_ptr.get());
	if (!unwrapped) {
		return nullptr;
	}

	if (remove) {
		auto &outer_proj = wrapper_ptr->Cast<LogicalProjection>();
		auto &agg = outer_proj.children[0]->Cast<LogicalAggregate>();
		auto &inner_proj = agg.children[0]->Cast<LogicalProjection>();

		PAC_DEBUG_PRINT("Stripping scalar subquery wrapper: outer_proj #" +
		                std::to_string(outer_proj.table_index) +
		                " -> inner_proj #" + std::to_string(inner_proj.table_index));

		// Change inner projection's table_index to match outer's
		inner_proj.table_index = outer_proj.table_index;

		// Update inner projection's types to match its expressions
		inner_proj.types.clear();
		for (auto &expr : inner_proj.expressions) {
			inner_proj.types.push_back(expr->return_type);
		}

		// Replace the wrapper_ptr with the inner projection
		wrapper_ptr = std::move(agg.children[0]);
	}

	return unwrapped;
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
				PAC_DEBUG_PRINT("RewriteProjectionsWithCounters: Skipping non-numerical type " +
				                target_type.ToString());
				continue;
			}
			PAC_DEBUG_PRINT("RewriteProjectionsWithCounters: Found projection with " +
			                std::to_string(pac_bindings.size()) + " PAC binding(s)");
			PAC_DEBUG_PRINT("  Original expr: " + expr->ToString());
			PAC_DEBUG_PRINT("  Target type: " + target_type.ToString());
			// Handle early-out for filter pattern simple cast (single aggregate only)
			if (is_filter_pattern_projection && pac_bindings.size() == 1) {
				bool is_simple_cast =
				    expr->type == ExpressionType::OPERATOR_CAST &&
				    expr->Cast<BoundCastExpression>().child->type == ExpressionType::BOUND_COLUMN_REF;
				if (is_simple_cast) {
					PropagateCountersType(pac_bindings[0].binding, plan_root);
					proj.expressions[i] = make_uniq<BoundColumnRefExpression>(
					    "pac_counters", LogicalType::LIST(LogicalType::DOUBLE), pac_bindings[0].binding);
					proj.types[i] = LogicalType::LIST(LogicalType::DOUBLE);
					continue;
				}
			}

			// Determine expression to clone (strip outer CAST if needed)
			Expression *expr_to_clone = expr.get();
			if (expr->type == ExpressionType::OPERATOR_CAST &&
			    expr->Cast<BoundCastExpression>().return_type != LogicalType::DOUBLE) {
				expr_to_clone = expr->Cast<BoundCastExpression>().child.get();
			}

			// Build list_transform over counter bindings
			auto list_expr = BuildCounterListTransform(input, pac_bindings, expr_to_clone, plan_root,
			                                           LogicalType::DOUBLE, LogicalType::DOUBLE);
			if (!list_expr) {
				continue;
			}

			if (is_filter_pattern_projection) {
				proj.expressions[i] = std::move(list_expr);
				proj.types[i] = LogicalType::LIST(LogicalType::DOUBLE);
			} else {
				auto noised = BuildPacNoisedCall(input, std::move(list_expr));
				if (target_type != LogicalType::DOUBLE) {
					noised = BoundCastExpression::AddDefaultCastToType(std::move(noised), target_type);
				}
				proj.expressions[i] = std::move(noised);
				proj.types[i] = target_type;
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

// Implementation of BuildCounterListTransform
static unique_ptr<Expression> BuildCounterListTransform(OptimizerExtensionInput &input,
                                                        const vector<PacBindingInfo> &pac_bindings,
                                                        Expression *expr_to_transform, LogicalOperator *plan_root,
                                                        const LogicalType &element_type,
                                                        const LogicalType &result_element_type) {
	// Propagate LIST<DOUBLE> type for all bindings
	for (auto &bi : pac_bindings) {
		PropagateCountersType(bi.binding, plan_root);
	}

	if (pac_bindings.size() == 1) {
		// --- SINGLE AGGREGATE ---
		auto &binding_info = pac_bindings[0];
		ColumnBinding pac_binding = binding_info.binding;

		auto counters_ref = make_uniq<BoundColumnRefExpression>("pac_counters", LogicalType::LIST(LogicalType::DOUBLE),
		                                                        pac_binding);
		auto safe_counters = WrapListNullSafe(std::move(counters_ref), input);

		unique_ptr<Expression> input_list;
		if (element_type != LogicalType::DOUBLE) {
			// Double-lambda: first transform DOUBLE -> element_type
			auto inner_elem = make_uniq<BoundReferenceExpression>("elem", LogicalType::DOUBLE, 0);
			unique_ptr<Expression> inner_body =
			    BoundCastExpression::AddDefaultCastToType(std::move(inner_elem), element_type);
			auto inner_lambda = BuildPacLambda(std::move(inner_body), {});
			input_list =
			    BuildListTransformCall(input, std::move(safe_counters), std::move(inner_lambda), element_type);
		} else {
			input_list = std::move(safe_counters);
		}

		// Outer lambda: clone the expression replacing PAC binding with lambda element
		vector<unique_ptr<Expression>> captures;
		unordered_map<uint64_t, idx_t> capture_map;
		auto lambda_body =
		    CloneForLambdaBody(expr_to_transform, pac_binding, captures, capture_map, plan_root, element_type);

		// Ensure body returns result_element_type if needed
		if (result_element_type == LogicalType::DOUBLE && lambda_body->return_type != LogicalType::DOUBLE) {
			lambda_body = BoundCastExpression::AddDefaultCastToType(std::move(lambda_body), LogicalType::DOUBLE);
		}

		auto lambda = BuildPacLambda(std::move(lambda_body), std::move(captures));
		return BuildListTransformCall(input, std::move(input_list), std::move(lambda), result_element_type);

	} else {
		// --- MULTIPLE AGGREGATES ---
		vector<unique_ptr<Expression>> counter_lists;
		unordered_map<uint64_t, idx_t> binding_to_index;
		unordered_map<uint64_t, LogicalType> binding_to_type;

		for (auto &bi : pac_bindings) {
			counter_lists.push_back(WrapListNullSafe(
			    make_uniq<BoundColumnRefExpression>("pac_counters", LogicalType::LIST(LogicalType::DOUBLE), bi.binding),
			    input));
			uint64_t hash = HashBinding(bi.binding);
			binding_to_index[hash] = bi.index;
			binding_to_type[hash] = bi.original_type;
		}

		LogicalType struct_type;
		auto zipped_list = BuildListZipCall(input, std::move(counter_lists), struct_type);
		if (!zipped_list) {
			return nullptr;
		}

		vector<unique_ptr<Expression>> captures;
		unordered_map<uint64_t, idx_t> capture_map;
		auto lambda_body = CloneForLambdaBodyMulti(expr_to_transform, binding_to_index, binding_to_type, captures,
		                                           capture_map, plan_root, struct_type);

		if (result_element_type == LogicalType::DOUBLE && lambda_body->return_type != LogicalType::DOUBLE) {
			lambda_body = BoundCastExpression::AddDefaultCastToType(std::move(lambda_body), LogicalType::DOUBLE);
		}

		auto lambda = BuildPacLambda(std::move(lambda_body), std::move(captures));
		return BuildListTransformCall(input, std::move(zipped_list), std::move(lambda), result_element_type);
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

	PAC_DEBUG_PRINT("Found " + std::to_string(categorical_aggregates.size()) + " categorical aggregate(s) to convert");

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

		// Check each child - if it's a wrapper containing a PAC aggregate, strip it
		for (auto &child : op->children) {
			auto *unwrapped = StripScalarWrapperInPlace(child, false);
			if (unwrapped) {
				string pac_name = FindPacAggregateInOperator(unwrapped);
				if (!pac_name.empty()) {
					PAC_DEBUG_PRINT("Found scalar wrapper containing PAC aggregate: " + pac_name);
					StripScalarWrapperInPlace(child);
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

	PAC_DEBUG_PRINT("=== PLAN AFTER COUNTERS REPLACEMENT ===");
#if PAC_DEBUG
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

			PAC_DEBUG_PRINT("Filter expression has " + std::to_string(pac_bindings.size()) + " PAC binding(s)");
			PAC_DEBUG_PRINT("  Filter expr: " + filter_expr->ToString());

			if (pac_bindings.empty()) {
				continue;
			}

			// Determine element_type: for single aggregate, use original_type
			// to trigger double-lambda (DOUBLE -> original_type cast);
			// for multiple aggregates, CloneForLambdaBodyMulti handles casting internally
			LogicalType original_type = pattern.original_return_type.id() != LogicalTypeId::INVALID
			                                ? pattern.original_return_type
			                                : pac_bindings[0].original_type;
			LogicalType element_type = (pac_bindings.size() == 1) ? original_type : LogicalType::DOUBLE;

			auto list_bool = BuildCounterListTransform(input, pac_bindings, filter_expr.get(), plan.get(),
			                                           element_type, LogicalType::BOOLEAN);
			if (!list_bool) {
				continue;
			}

			auto pac_filter_expr = input.optimizer.BindScalarFunction("pac_filter", std::move(list_bool));
			filter_expr = std::move(pac_filter_expr);
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
				zip_args.push_back(WrapListNullSafe(std::move(left_expr), input));
				zip_args.push_back(WrapListNullSafe(std::move(right_expr), input));

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
				auto cast_list = BuildListTransformCall(input, WrapListNullSafe(std::move(counters_expr), input),
				                                        std::move(inner_lambda), original_type);

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
