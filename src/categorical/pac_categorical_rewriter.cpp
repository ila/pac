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
static LogicalOperator *StripScalarWrapperInPlace(unique_ptr<LogicalOperator> &wrapper_ptr, bool remove);

// Trace a column binding through the plan to find if it comes from a PAC aggregate
// Returns the PAC aggregate name if found (base name without _counters), empty string otherwise
static string TracePacAggregateFromBinding(const ColumnBinding &binding, LogicalOperator *plan_root) {
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

// Resolve a column binding to its source by following through projection operators
// (including functions within projections like `0.5 * agg_result` or `pac_scale_counters(col)`)
static ColumnBinding ResolveBindingSource(const ColumnBinding &binding, LogicalOperator *plan_root) {
	auto *source_op = FindOperatorByTableIndex(plan_root, binding.table_index);
	if (!source_op || source_op->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return binding;
	}
	auto &proj = source_op->Cast<LogicalProjection>();
	if (binding.column_index < proj.expressions.size()) {
		auto &expr = proj.expressions[binding.column_index];
		if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
			auto &col_ref = expr->Cast<BoundColumnRefExpression>();
			return ResolveBindingSource(col_ref.binding, plan_root);
		}
		// For functions like 0.5 * agg_result, trace through the function's children
		if (expr->type == ExpressionType::BOUND_FUNCTION) {
			auto &func_expr = expr->Cast<BoundFunctionExpression>();
			for (auto &child : func_expr.children) {
				if (child->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = child->Cast<BoundColumnRefExpression>();
					auto traced = ResolveBindingSource(col_ref.binding, plan_root);
					// If we found a different binding, return it
					if (traced.table_index != col_ref.binding.table_index) {
						return traced;
					}
					return col_ref.binding;
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
	// Track if we're entering an aggregate
	bool now_inside_aggregate = inside_aggregate || (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY);

	// Track patterns count before checking this operator AND its children.
	// Used at the end to strip scalar wrappers when any new patterns were found.
	size_t patterns_before_all = patterns.size();

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
			// Check if ANY of the bindings is NOT a HAVING clause aggregate
			// If at least one binding is from a subquery (not HAVING), this is categorical
			bool has_non_having_binding = false;
			ColumnBinding first_non_having_binding;
			string first_aggregate_name;

			for (auto &binding_info : pac_bindings) {
				ColumnBinding traced_binding = ResolveBindingSource(binding_info.binding, plan_root);
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

				// Check if this binding goes through a scalar subquery wrapper
				info.scalar_wrapper_op = FindScalarWrapperForBinding(first_non_having_binding, plan_root);

				// Capture original return type from the first non-having PAC binding
				for (auto &bi : pac_bindings) {
					if (bi.binding == first_non_having_binding) {
						info.original_return_type = bi.original_type;
						break;
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
				info.parent_op = op;
				info.expr_index = i;
				info.aggregate_name = left_pac;
				if (cond.left->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = cond.left->Cast<BoundColumnRefExpression>();
					info.scalar_wrapper_op = FindScalarWrapperForBinding(col_ref.binding, plan_root);
				}
				patterns.push_back(info);
			} else if (!right_pac.empty()) {
				info.parent_op = op;
				info.expr_index = i;
				info.aggregate_name = right_pac;
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
	// On the way back up: if patterns were found at this operator OR in this subtree,
	// strip scalar wrappers in direct children. This ensures wrappers around scalar subqueries
	// in join children are stripped when the join itself has a categorical pattern.
	if (patterns.size() > patterns_before_all) {
		for (auto &child : op->children) {
			if (child->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto *unwrapped = RecognizeDuckDBScalarWrapper(child.get());
				if (unwrapped) {
					PAC_DEBUG_PRINT("Stripping scalar wrapper during pattern detection");
					StripScalarWrapperInPlace(child, true);
				}
			}
		}
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

// Check if an expression traces back to a PAC _counters aggregate
static bool TracesPacCountersAggregate(Expression *expr, LogicalOperator *plan_root) {
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
	if (op->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return;
	}
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
		// Update types vector
		agg.expressions[i] = std::move(new_aggr);
		idx_t types_index = agg.groups.size() + i;
		if (types_index < agg.types.size()) {
			agg.types[types_index] = LogicalType::LIST(LogicalType::DOUBLE);
		}
	}
}

// Helper to collect ALL distinct PAC aggregate bindings in an expression
// Returns the bindings in the order they were discovered
static void CollectPacBindingsInExpression(Expression *expr, LogicalOperator *plan_root,
                                           vector<PacBindingInfo> &bindings,
                                           unordered_map<uint64_t, idx_t> &binding_hash_to_index) {
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

// Get struct field name for index (a, b, c, ..., z, aa, ab, ...)
static string GetStructFieldName(idx_t index) {
	if (index < 26) {
		return string(1, static_cast<char>('a' + static_cast<unsigned char>(index)));
	}
	// For indices >= 26, use aa, ab, etc.
	return string(1, static_cast<char>('a' + static_cast<unsigned char>(index / 26 - 1))) +
	       string(1, static_cast<char>('a' + static_cast<unsigned char>(index % 26)));
}

// Clone an expression tree for use as a lambda body (unified single/multi binding version).
// PAC aggregate column refs are replaced with lambda element references.
// Other column refs are captured and become BoundReferenceExpression(1+i).
//
// pac_binding_map: maps binding hash -> struct field index (single: one entry mapping to 0)
// element_type: the type of the lambda parameter. For single binding, this is the aggregate's
//               original type (or DOUBLE for raw counters). For multi binding, always DOUBLE.
// struct_type: nullptr = single binding (elem ref), non-null = multi binding (struct field extract)
static unique_ptr<Expression>
CloneForLambdaBody(Expression *expr, const unordered_map<uint64_t, idx_t> &pac_binding_map,
                   vector<unique_ptr<Expression>> &captures, unordered_map<uint64_t, idx_t> &capture_map,
                   LogicalOperator *plan_root, const LogicalType &element_type, const LogicalType *struct_type) {
	// Helper to build the replacement expression for a matched PAC binding
	auto make_pac_replacement = [&](const unordered_map<uint64_t, idx_t>::const_iterator &it,
	                                const string &alias) -> unique_ptr<Expression> {
		if (struct_type) {
			return BuildStructFieldExtract(*struct_type, it->second, GetStructFieldName(it->second));
		} else {
			return make_uniq<BoundReferenceExpression>(alias, element_type, 0);
		}
	};

	// Handle column references
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();
		uint64_t binding_hash = HashBinding(col_ref.binding);
		PAC_DEBUG_PRINT("    CloneForLambdaBody: col_ref [" + std::to_string(col_ref.binding.table_index) + "." +
		                std::to_string(col_ref.binding.column_index) + "] '" + col_ref.alias + "'");

		// Direct match
		auto it = pac_binding_map.find(binding_hash);
		if (it != pac_binding_map.end()) {
			PAC_DEBUG_PRINT("    CloneForLambdaBody: MATCHED (direct)");
			return make_pac_replacement(it, "elem");
		}

		// Trace through projections (and functions like pac_scale_counters) to find PAC binding
		ColumnBinding traced = ResolveBindingSource(col_ref.binding, plan_root);
		if (!(traced == col_ref.binding)) {
			auto traced_it = pac_binding_map.find(HashBinding(traced));
			if (traced_it != pac_binding_map.end()) {
				PAC_DEBUG_PRINT("    CloneForLambdaBody: MATCHED (traced through projection)");
				return make_pac_replacement(traced_it, col_ref.alias);
			}
		}
		PAC_DEBUG_PRINT("    CloneForLambdaBody: NO MATCH - capturing");
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
		auto child_clone = CloneForLambdaBody(cast.child.get(), pac_binding_map, captures, capture_map, plan_root,
		                                      element_type, struct_type);
		// If the child's type already matches the target type, skip the cast
		if (child_clone->return_type == cast.return_type) {
			return child_clone;
		}
		// Otherwise, create a new cast with the correct function for the new child type
		return BoundCastExpression::AddDefaultCastToType(std::move(child_clone), cast.return_type);
	}
	// Handle comparison expressions
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON) {
		auto &comp = expr->Cast<BoundComparisonExpression>();
		auto left_clone = CloneForLambdaBody(comp.left.get(), pac_binding_map, captures, capture_map, plan_root,
		                                     element_type, struct_type);
		auto right_clone = CloneForLambdaBody(comp.right.get(), pac_binding_map, captures, capture_map, plan_root,
		                                      element_type, struct_type);
		// Reconcile types if they differ (needed for multi where struct fields are DOUBLE
		// but original CASTs may introduce DECIMAL)
		if (left_clone->return_type != right_clone->return_type) {
			if (left_clone->return_type != LogicalType::DOUBLE) {
				left_clone = BoundCastExpression::AddDefaultCastToType(std::move(left_clone), LogicalType::DOUBLE);
			}
			if (right_clone->return_type != LogicalType::DOUBLE) {
				right_clone = BoundCastExpression::AddDefaultCastToType(std::move(right_clone), LogicalType::DOUBLE);
			}
		}
		return make_uniq<BoundComparisonExpression>(expr->type, std::move(left_clone), std::move(right_clone));
	}
	// Handle function expressions
	if (expr->type == ExpressionType::BOUND_FUNCTION) {
		auto &func = expr->Cast<BoundFunctionExpression>();
		vector<unique_ptr<Expression>> new_children;
		bool any_cast_needed = false;
		for (idx_t i = 0; i < func.children.size(); i++) {
			auto child_clone = CloneForLambdaBody(func.children[i].get(), pac_binding_map, captures, capture_map,
			                                      plan_root, element_type, struct_type);
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
		// When element_type is DOUBLE (raw counters context or multi binding), children were cast
		// from DOUBLE to the function's bound types (e.g., DECIMAL). Cast the result back to DOUBLE
		// so the list_transform output stays LIST<DOUBLE>.
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
			new_children.push_back(CloneForLambdaBody(child.get(), pac_binding_map, captures, capture_map, plan_root,
			                                          element_type, struct_type));
		}
		return BuildClonedOperatorExpression(expr, std::move(new_children));
	}
	// Handle CASE expressions
	if (expr->type == ExpressionType::CASE_EXPR) {
		auto &case_expr = expr->Cast<BoundCaseExpression>();

		if (IsScalarSubqueryWrapper(case_expr)) {
			PAC_DEBUG_PRINT("    CloneForLambdaBody: Stripping DuckDB scalar subquery wrapper, using ELSE branch");
			return CloneForLambdaBody(case_expr.else_expr.get(), pac_binding_map, captures, capture_map, plan_root,
			                          element_type, struct_type);
		}
		// Regular CASE - recurse into all branches
		// Start with original return type, will update based on cloned branches
		auto result = make_uniq<BoundCaseExpression>(case_expr.return_type);
		for (auto &check : case_expr.case_checks) {
			BoundCaseCheck new_check;
			new_check.when_expr = CloneForLambdaBody(check.when_expr.get(), pac_binding_map, captures, capture_map,
			                                         plan_root, element_type, struct_type);
			new_check.then_expr = CloneForLambdaBody(check.then_expr.get(), pac_binding_map, captures, capture_map,
			                                         plan_root, element_type, struct_type);
			result->case_checks.push_back(std::move(new_check));
		}
		if (case_expr.else_expr) {
			result->else_expr = CloneForLambdaBody(case_expr.else_expr.get(), pac_binding_map, captures, capture_map,
			                                       plan_root, element_type, struct_type);
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
//   CloneForLambdaBody (multi mode) handles per-field type casting internally.
//
// Returns the list_transform expression, or nullptr on failure.
// The caller wraps with pac_noised or pac_filter as needed.
static unique_ptr<Expression> BuildCounterListTransform(OptimizerExtensionInput &input,
                                                        const vector<PacBindingInfo> &pac_bindings,
                                                        Expression *expr_to_transform, LogicalOperator *plan_root,
                                                        const LogicalType &element_type,
                                                        const LogicalType &result_element_type);

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
		PAC_DEBUG_PRINT("Stripping scalar subquery wrapper: outer_proj #" + std::to_string(outer_proj.table_index) +
		                " -> inner_proj #" + std::to_string(inner_proj.table_index));

		// Change inner projection's table_index to match outer's
		inner_proj.table_index = outer_proj.table_index;

		// Update inner projection's types to match its expressions
		inner_proj.types.clear();
		for (auto &expr : inner_proj.expressions) {
			inner_proj.types.push_back(expr->return_type);
		}
		wrapper_ptr = std::move(agg.children[0]); // Replace the wrapper_ptr with the inner projection
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
		} else if (pattern.parent_op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
		           pattern.parent_op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			// Handle COMPARISON_JOIN and DELIM_JOIN patterns
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

// Implementation of BuildCounterListTransform
static unique_ptr<Expression> BuildCounterListTransform(OptimizerExtensionInput &input,
                                                        const vector<PacBindingInfo> &pac_bindings,
                                                        Expression *expr_to_transform, LogicalOperator *plan_root,
                                                        const LogicalType &element_type,
                                                        const LogicalType &result_element_type) {
	// Note: In the bottom-up rewrite pass, aggregate types are already updated to LIST<DOUBLE>
	// and intermediate projection col_refs are updated. No PropagateCountersType needed.
	if (pac_bindings.size() == 1) { // --- SINGLE AGGREGATE ---
		auto &binding_info = pac_bindings[0];
		ColumnBinding pac_binding = binding_info.binding;

		auto counters_ref =
		    make_uniq<BoundColumnRefExpression>("pac_counters", LogicalType::LIST(LogicalType::DOUBLE), pac_binding);
		auto safe_counters = WrapListNullSafe(std::move(counters_ref), input);

		unique_ptr<Expression> input_list;
		if (element_type != LogicalType::DOUBLE) {
			// Double-lambda: first transform DOUBLE -> element_type
			auto inner_elem = make_uniq<BoundReferenceExpression>("elem", LogicalType::DOUBLE, 0);
			unique_ptr<Expression> inner_body =
			    BoundCastExpression::AddDefaultCastToType(std::move(inner_elem), element_type);
			auto inner_lambda = BuildPacLambda(std::move(inner_body), {});
			input_list = BuildListTransformCall(input, std::move(safe_counters), std::move(inner_lambda), element_type);
		} else {
			input_list = std::move(safe_counters);
		}
		// Outer lambda: clone the expression replacing PAC binding with lambda element
		unordered_map<uint64_t, idx_t> pac_binding_map;
		pac_binding_map[HashBinding(pac_binding)] = 0;
		vector<unique_ptr<Expression>> captures;
		unordered_map<uint64_t, idx_t> capture_map;
		auto lambda_body = CloneForLambdaBody(expr_to_transform, pac_binding_map, captures, capture_map, plan_root,
		                                      element_type, nullptr);

		// Ensure body returns result_element_type if needed
		if (result_element_type == LogicalType::DOUBLE && lambda_body->return_type != LogicalType::DOUBLE) {
			lambda_body = BoundCastExpression::AddDefaultCastToType(std::move(lambda_body), LogicalType::DOUBLE);
		}
		auto lambda = BuildPacLambda(std::move(lambda_body), std::move(captures));
		return BuildListTransformCall(input, std::move(input_list), std::move(lambda), result_element_type);

	} else { // --- MULTIPLE AGGREGATES ---
		vector<unique_ptr<Expression>> counter_lists;
		unordered_map<uint64_t, idx_t> binding_to_index;

		for (auto &bi : pac_bindings) {
			counter_lists.push_back(WrapListNullSafe(
			    make_uniq<BoundColumnRefExpression>("pac_counters", LogicalType::LIST(LogicalType::DOUBLE), bi.binding),
			    input));
			binding_to_index[HashBinding(bi.binding)] = bi.index;
		}
		LogicalType struct_type;
		auto zipped_list = BuildListZipCall(input, std::move(counter_lists), struct_type);
		if (!zipped_list) {
			return nullptr;
		}
		vector<unique_ptr<Expression>> captures;
		unordered_map<uint64_t, idx_t> capture_map;
		auto lambda_body = CloneForLambdaBody(expr_to_transform, binding_to_index, captures, capture_map, plan_root,
		                                      LogicalType::DOUBLE, &struct_type);
		if (result_element_type == LogicalType::DOUBLE && lambda_body->return_type != LogicalType::DOUBLE) {
			lambda_body = BoundCastExpression::AddDefaultCastToType(std::move(lambda_body), LogicalType::DOUBLE);
		}
		auto lambda = BuildPacLambda(std::move(lambda_body), std::move(captures));
		return BuildListTransformCall(input, std::move(zipped_list), std::move(lambda), result_element_type);
	}
}

// The kind of wrapping to apply after BuildCounterListTransform
enum class PacWrapKind {
	PAC_NOISED, // Projection: list_transform -> pac_noised -> optional cast
	PAC_FILTER  // Filter/Join: list_transform -> pac_filter
};

// Unified expression rewrite: build list_transform over PAC counters and wrap.
// pac_bindings: all PAC aggregate bindings in the expression
// expr: the expression to transform (boolean for filter, numeric for projection)
// wrap_kind: determines terminal function and type parameters
// target_type: for PAC_NOISED, cast result to this type (ignored for PAC_FILTER)
static unique_ptr<Expression> RewriteExpressionWithCounters(OptimizerExtensionInput &input,
                                                            const vector<PacBindingInfo> &pac_bindings,
                                                            Expression *expr, LogicalOperator *plan_root,
                                                            PacWrapKind wrap_kind,
                                                            const LogicalType &target_type = LogicalType::DOUBLE) {
	if (pac_bindings.empty()) {
		return nullptr;
	}

	LogicalType element_type;
	LogicalType result_element_type;

	if (wrap_kind == PacWrapKind::PAC_NOISED) {
		element_type = LogicalType::DOUBLE;
		result_element_type = LogicalType::DOUBLE;
	} else {
		// Filter/Join: use original type for single binding (enables double-lambda),
		// DOUBLE for multi-binding (list_zip always produces DOUBLE)
		element_type = (pac_bindings.size() == 1) ? pac_bindings[0].original_type : LogicalType::DOUBLE;
		result_element_type = LogicalType::BOOLEAN;
	}

	auto list_expr = BuildCounterListTransform(input, pac_bindings, expr, plan_root, element_type, result_element_type);
	if (!list_expr) {
		return nullptr;
	}

	if (wrap_kind == PacWrapKind::PAC_NOISED) {
		auto noised = BuildPacNoisedCall(input, std::move(list_expr));
		if (target_type != LogicalType::DOUBLE) {
			noised = BoundCastExpression::AddDefaultCastToType(std::move(noised), target_type);
		}
		return noised;
	} else {
		return input.optimizer.BindScalarFunction("pac_filter", std::move(list_expr));
	}
}

// Build a map from parent operators to their ExpressionRewriteInfo entries.
// Groups patterns by (parent_op, expr_index) and pre-collects PAC bindings per expression.
static unordered_map<LogicalOperator *, vector<ExpressionRewriteInfo>>
BuildRewriteMap(const vector<CategoricalPatternInfo> &patterns, LogicalOperator *plan_root) {
	unordered_map<LogicalOperator *, vector<ExpressionRewriteInfo>> result;
	unordered_set<uint64_t> seen;

	for (auto &pattern : patterns) {
		if (!pattern.parent_op) {
			continue;
		}
		// Key: combine pointer and expr_index for uniqueness
		uint64_t key = reinterpret_cast<uint64_t>(pattern.parent_op) ^ (uint64_t(pattern.expr_index) << 48);
		if (seen.count(key)) {
			continue;
		}
		seen.insert(key);

		ExpressionRewriteInfo info;
		info.parent_op = pattern.parent_op;
		info.expr_index = pattern.expr_index;
		info.original_return_type = pattern.original_return_type;

		// Pre-collect PAC bindings for this expression
		if (pattern.parent_op->type == LogicalOperatorType::LOGICAL_FILTER) {
			auto &filter = pattern.parent_op->Cast<LogicalFilter>();
			if (pattern.expr_index < filter.expressions.size()) {
				info.pac_bindings =
				    FindAllPacBindingsInExpression(filter.expressions[pattern.expr_index].get(), plan_root);
			}
		} else if (pattern.parent_op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			auto &proj = pattern.parent_op->Cast<LogicalProjection>();
			if (pattern.expr_index < proj.expressions.size()) {
				info.pac_bindings =
				    FindAllPacBindingsInExpression(proj.expressions[pattern.expr_index].get(), plan_root);
			}
		} else if (pattern.parent_op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
		           pattern.parent_op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			auto &join = pattern.parent_op->Cast<LogicalComparisonJoin>();
			if (pattern.expr_index < join.conditions.size()) {
				auto left_bindings =
				    FindAllPacBindingsInExpression(join.conditions[pattern.expr_index].left.get(), plan_root);
				auto right_bindings =
				    FindAllPacBindingsInExpression(join.conditions[pattern.expr_index].right.get(), plan_root);
				for (auto &b : left_bindings) {
					info.pac_bindings.push_back(std::move(b));
				}
				for (auto &b : right_bindings) {
					info.pac_bindings.push_back(std::move(b));
				}
			}
		}
		result[pattern.parent_op].push_back(std::move(info));
	}
	return result;
}

// Single bottom-up rewrite pass.
// Processes children first, then current operator. Handles:
// - AGGREGATE: convert pac_sum → pac_sum_counters, then aggregate-over-counters → _list
// - PROJECTION: update simple col_ref types, build list_transform + pac_noised for arithmetic
// - FILTER (in rewrite_map): build list_transform + pac_filter
// - JOIN (in rewrite_map): rewrite conditions (two-list → CROSS_PRODUCT+FILTER, single-list → double-lambda)
static void RewriteBottomUp(unique_ptr<LogicalOperator> &op_ptr, OptimizerExtensionInput &input,
                            unique_ptr<LogicalOperator> &plan,
                            const unordered_map<LogicalOperator *, vector<ExpressionRewriteInfo>> &rewrite_map,
                            vector<CategoricalPatternInfo> &patterns) {
	auto *op = op_ptr.get();

	// 1. Recurse into children first (bottom-up)
	for (auto &child : op->children) {
		RewriteBottomUp(child, input, plan, rewrite_map, patterns);
	}

	LogicalOperator *plan_root = plan.get();

	// === AGGREGATE: convert PAC aggregates to _counters, then check aggregates-over-counters ===
	if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &agg = op->Cast<LogicalAggregate>();

		// Convert PAC aggregates to _counters variants
		for (idx_t i = 0; i < agg.expressions.size(); i++) {
			auto &agg_expr = agg.expressions[i];
			if (agg_expr->type != ExpressionType::BOUND_AGGREGATE) {
				continue;
			}
			auto &bound_agg = agg_expr->Cast<BoundAggregateExpression>();
			if (!IsPacAggregate(bound_agg.function.name)) {
				continue;
			}
			string counters_name = GetCountersVariant(bound_agg.function.name);
			auto &catalog = Catalog::GetSystemCatalog(input.context);
			auto &func_entry =
			    catalog.GetEntry<AggregateFunctionCatalogEntry>(input.context, DEFAULT_SCHEMA, counters_name);

			vector<LogicalType> arg_types;
			for (auto &child_expr : bound_agg.children) {
				arg_types.push_back(child_expr->return_type);
			}
			ErrorData error;
			FunctionBinder function_binder(input.context);
			auto best_function = function_binder.BindFunction(counters_name, func_entry.functions, arg_types, error);

			if (!best_function.IsValid()) {
				PAC_DEBUG_PRINT("Warning: Could not bind " + counters_name + ": " + error.Message());
				bound_agg.function.name = counters_name;
				bound_agg.function.return_type = LogicalType::LIST(LogicalType::DOUBLE);
				agg_expr->return_type = LogicalType::LIST(LogicalType::DOUBLE);
			} else {
				AggregateFunction counters_func = func_entry.functions.GetFunctionByOffset(best_function.GetIndex());
				vector<unique_ptr<Expression>> children;
				for (auto &child_expr : bound_agg.children) {
					children.push_back(child_expr->Copy());
				}
				auto new_aggr = function_binder.BindAggregateFunction(
				    counters_func, std::move(children), nullptr,
				    bound_agg.IsDistinct() ? AggregateType::DISTINCT : AggregateType::NON_DISTINCT);
				agg.expressions[i] = std::move(new_aggr);
			}
			idx_t types_index = agg.groups.size() + i;
			if (types_index < agg.types.size()) {
				agg.types[types_index] = LogicalType::LIST(LogicalType::DOUBLE);
			}
		}
		// Check for standard aggregates over counters (e.g., sum(LIST<DOUBLE>) → pac_sum_list)
		// Children already converted (bottom-up), so their types are LIST<DOUBLE>
		ReplaceAggregatesOverCounters(op, input.context, plan_root);
	}

	// === PROJECTION: rewrite expressions containing PAC aggregate bindings ===
	if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = op->Cast<LogicalProjection>();
		bool is_filter_pattern_projection = IsProjectionReferencedByFilterPattern(proj, patterns, plan_root);

		PAC_DEBUG_PRINT("RewriteBottomUp PROJECTION #" + std::to_string(proj.table_index) +
		                (is_filter_pattern_projection ? " (filter-pattern)" : ""));

		for (idx_t i = 0; i < proj.expressions.size(); i++) {
			auto &expr = proj.expressions[i];
			if (IsAlreadyWrappedInPacNoised(expr.get())) {
				continue;
			}
			auto pac_bindings = FindAllPacBindingsInExpression(expr.get(), plan_root);
			if (pac_bindings.empty()) {
				continue;
			}
			// Simple column reference: just update type to LIST<DOUBLE>
			if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
				expr->return_type = LogicalType::LIST(LogicalType::DOUBLE);
				if (i < proj.types.size()) {
					proj.types[i] = LogicalType::LIST(LogicalType::DOUBLE);
				}
				PAC_DEBUG_PRINT("  Updated col_ref type to LIST<DOUBLE> at index " + std::to_string(i));
				continue;
			}
			if (IsFilterPatternProjection(expr.get(), op)) {
				continue;
			}
			LogicalType target_type = expr->return_type;
			if (!IsNumericalType(target_type)) {
				continue;
			}
			// Filter pattern simple cast (single aggregate): replace with direct counters ref
			if (is_filter_pattern_projection && pac_bindings.size() == 1) {
				bool is_simple_cast = expr->type == ExpressionType::OPERATOR_CAST &&
				                      expr->Cast<BoundCastExpression>().child->type == ExpressionType::BOUND_COLUMN_REF;
				if (is_simple_cast) {
					proj.expressions[i] = make_uniq<BoundColumnRefExpression>(
					    "pac_counters", LogicalType::LIST(LogicalType::DOUBLE), pac_bindings[0].binding);
					proj.types[i] = LogicalType::LIST(LogicalType::DOUBLE);
					PAC_DEBUG_PRINT("  Replaced simple cast with direct counters ref at index " + std::to_string(i));
					continue;
				}
			}
			// Determine expression to clone (strip outer CAST if needed)
			Expression *expr_to_clone = expr.get();
			if (expr->type == ExpressionType::OPERATOR_CAST &&
			    expr->Cast<BoundCastExpression>().return_type != LogicalType::DOUBLE) {
				expr_to_clone = expr->Cast<BoundCastExpression>().child.get();
			}
			if (is_filter_pattern_projection) {
				// Intermediate: produce LIST<DOUBLE> for downstream filter (no terminal wrapping)
				auto list_expr = BuildCounterListTransform(input, pac_bindings, expr_to_clone, plan_root,
				                                           LogicalType::DOUBLE, LogicalType::DOUBLE);
				if (!list_expr) {
					continue;
				}
				proj.expressions[i] = std::move(list_expr);
				proj.types[i] = LogicalType::LIST(LogicalType::DOUBLE);
			} else {
				auto result = RewriteExpressionWithCounters(input, pac_bindings, expr_to_clone, plan_root,
				                                            PacWrapKind::PAC_NOISED, target_type);
				if (!result) {
					continue;
				}
				proj.expressions[i] = std::move(result);
				proj.types[i] = target_type;
			}
			PAC_DEBUG_PRINT("  Rewrote projection expr at index " + std::to_string(i));
		}
	}
	// === FILTER: rewrite filter expressions with pac_filter ===
	if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
		auto it = rewrite_map.find(op);
		if (it != rewrite_map.end()) {
			auto &filter = op->Cast<LogicalFilter>();
			unordered_set<idx_t> processed;

			for (auto &info : it->second) {
				if (processed.count(info.expr_index)) {
					continue;
				}
				processed.insert(info.expr_index);
				if (info.expr_index >= filter.expressions.size()) {
					continue;
				}
				auto &filter_expr = filter.expressions[info.expr_index];
				auto pac_bindings = FindAllPacBindingsInExpression(filter_expr.get(), plan_root);
				if (pac_bindings.empty()) {
					continue;
				}
				PAC_DEBUG_PRINT("RewriteBottomUp FILTER: " + std::to_string(pac_bindings.size()) + " PAC binding(s)");

				auto result = RewriteExpressionWithCounters(input, pac_bindings, filter_expr.get(), plan_root,
				                                            PacWrapKind::PAC_FILTER);
				if (!result) {
					continue;
				}
				filter.expressions[info.expr_index] = std::move(result);
				PAC_DEBUG_PRINT("  Rewrote filter expr at index " + std::to_string(info.expr_index));
			}
		}
	}
	// === JOIN: rewrite comparison join conditions ===
	if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	    op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		auto it = rewrite_map.find(op);
		if (it != rewrite_map.end()) {
			auto &join = op->Cast<LogicalComparisonJoin>();

			for (auto &info : it->second) {
				if (info.expr_index >= join.conditions.size()) {
					continue;
				}
				auto &cond = join.conditions[info.expr_index];
				auto left_bindings = FindAllPacBindingsInExpression(cond.left.get(), plan_root);
				auto right_bindings = FindAllPacBindingsInExpression(cond.right.get(), plan_root);
				bool left_is_list = !left_bindings.empty();
				bool right_is_list = !right_bindings.empty();

				PAC_DEBUG_PRINT("RewriteBottomUp JOIN: left_is_list=" + std::string(left_is_list ? "true" : "false") +
				                ", right_is_list=" + std::string(right_is_list ? "true" : "false"));

				if (!left_is_list && !right_is_list) {
					continue;
				}
				if (left_is_list && right_is_list) {
					// Both sides are lists: CROSS_PRODUCT + FILTER with list_zip
					// Combine all PAC bindings, deduplicating by hash
					vector<PacBindingInfo> all_bindings;
					unordered_set<uint64_t> seen_hashes;
					for (auto &b : left_bindings) {
						uint64_t h = HashBinding(b.binding);
						if (seen_hashes.insert(h).second) {
							all_bindings.push_back(b);
						}
					}
					for (auto &b : right_bindings) {
						uint64_t h = HashBinding(b.binding);
						if (seen_hashes.insert(h).second) {
							all_bindings.push_back(b);
						}
					}
					for (idx_t j = 0; j < all_bindings.size(); j++) {
						all_bindings[j].index = j;
					}
					// Build comparison expression from the join condition
					auto comparison =
					    make_uniq<BoundComparisonExpression>(cond.comparison, cond.left->Copy(), cond.right->Copy());
					auto pac_filter_expr = RewriteExpressionWithCounters(input, all_bindings, comparison.get(),
					                                                     plan_root, PacWrapKind::PAC_FILTER);
					if (!pac_filter_expr) {
						continue;
					}

					// Convert COMPARISON_JOIN to CROSS_PRODUCT + FILTER
					auto cross_product =
					    LogicalCrossProduct::Create(std::move(join.children[0]), std::move(join.children[1]));
					auto filter_op = make_uniq<LogicalFilter>();
					filter_op->expressions.push_back(std::move(pac_filter_expr));
					filter_op->children.push_back(std::move(cross_product));

					// Invalidate all patterns that pointed to the old join (now destroyed)
					for (auto &p : patterns) {
						if (p.parent_op == op) {
							p.parent_op = nullptr;
						}
					}
					op_ptr = std::move(filter_op);
					PAC_DEBUG_PRINT("  Converted two-list JOIN to CROSS_PRODUCT + FILTER");
					break; // Replaced operator, can't process more conditions
				} else {
					// One side is list: rewrite the comparison with pac_filter
					auto &pac_bindings = left_is_list ? left_bindings : right_bindings;
					if (pac_bindings.empty()) {
						continue;
					}
					auto comparison =
					    make_uniq<BoundComparisonExpression>(cond.comparison, cond.left->Copy(), cond.right->Copy());
					auto result = RewriteExpressionWithCounters(input, pac_bindings, comparison.get(), plan_root,
					                                            PacWrapKind::PAC_FILTER);
					if (!result) {
						continue;
					}
					cond.left = std::move(result);
					cond.right = make_uniq<BoundConstantExpression>(Value::BOOLEAN(true));
					cond.comparison = ExpressionType::COMPARE_EQUAL;
					PAC_DEBUG_PRINT("  Rewrote single-list JOIN condition");
				}
			}
		}
	}
}

void RewriteCategoricalQuery(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                             vector<CategoricalPatternInfo> &patterns) {
	// Save the original output types before transformation
	// We'll need these to cast back at the end to match what the result collector expects
	vector<LogicalType> original_output_types = plan->types;

	// Phase A: Build rewrite map from patterns (grouped by expression)
	auto rewrite_map = BuildRewriteMap(patterns, plan.get());
	PAC_DEBUG_PRINT("Built rewrite map with " + std::to_string(rewrite_map.size()) + " operator(s)");

	// Phase B: Single bottom-up pass doing ALL rewrites
	// - Aggregates: pac_sum → pac_sum_counters, then aggregate-over-counters → _list
	// - Projections: update col_ref types, build list_transform + pac_noised
	// - Filters: build list_transform + pac_filter
	// - Joins: rewrite conditions (two-list → CROSS_PRODUCT+FILTER, single-list → double-lambda)
	// Scalar wrappers were already stripped during pattern detection (Phase 0).
	RewriteBottomUp(plan, input, plan, rewrite_map, patterns);
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
