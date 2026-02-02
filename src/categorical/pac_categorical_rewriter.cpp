//
// PAC Categorical Query Rewriter - Implementation
//
// See pac_categorical_rewriter.hpp for design documentation.
//
// Created by ila on 1/23/26.
//

#include "categorical/pac_categorical_rewriter.hpp"
#include "query_processing/pac_plan_traversal.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_subquery_expression.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_lambda_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/lambda_functions.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/common/types.hpp"

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

string GetCountersVariant(const string &aggregate_name) {
	// Already a counters variant
	if (IsPacCountersAggregate(aggregate_name)) {
		return aggregate_name;
	}
	return aggregate_name + "_counters";
}

// Forward declarations
static string FindPacAggregateInExpressionWithPlan(Expression *expr, LogicalOperator *plan_root);
static string TracePacAggregateFromBinding(const ColumnBinding &binding, LogicalOperator *plan_root);
static bool FindSinglePacBindingInExpression(Expression *expr, LogicalOperator *plan_root, ColumnBinding &out_binding,
                                             string &out_aggregate_name);
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

	// Recurse into children
	for (auto &child : op->children) {
		auto *result = FindOperatorByTableIndex(child.get(), table_index);
		if (result) {
			return result;
		}
	}
	return nullptr;
}

// Check if a function name is a PAC aggregate (either original or counters variant)
static bool IsPacAggregateOrCounters(const string &name) {
	return IsPacAggregate(name) || IsPacCountersAggregate(name);
}

// Get the base PAC aggregate name (strip _counters suffix if present)
static string GetBasePacAggregateName(const string &name) {
	if (IsPacCountersAggregate(name)) {
		// Remove "_counters" suffix
		return name.substr(0, name.size() - 9);
	}
	return name;
}

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
		auto &proj = source_op->Cast<LogicalProjection>();
		if (binding.column_index < proj.expressions.size()) {
			// Recursively search this expression for PAC aggregates
			return FindPacAggregateInExpressionWithPlan(proj.expressions[binding.column_index].get(), plan_root);
		}
	} else if (source_op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &aggr = source_op->Cast<LogicalAggregate>();
		if (binding.table_index == aggr.aggregate_index) {
			// This binding comes from an aggregate expression
			if (binding.column_index < aggr.expressions.size()) {
				auto &agg_expr = aggr.expressions[binding.column_index];
				if (agg_expr->type == ExpressionType::BOUND_AGGREGATE) {
					auto &bound_agg = agg_expr->Cast<BoundAggregateExpression>();
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

	// Check if this is a subquery expression
	if (expr->type == ExpressionType::SUBQUERY) {
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
		if (child->children.empty())
			break;
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

			// Only add if NOT inside an aggregate (categorical = outer query has no aggregate)
			if (now_inside_aggregate) {
				continue;
			}

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
#ifdef DEBUG
				Printer::Print("Captured original_return_type: " + info.original_return_type.ToString());
				Printer::Print("Found categorical pattern with " + std::to_string(pac_bindings.size()) +
				               " PAC binding(s)");
#endif
				patterns.push_back(info);
			}
		}
	}

	// Check join conditions (for semi/anti/mark joins with subqueries)
	if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	    op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		auto &join = op->Cast<LogicalComparisonJoin>();
		for (idx_t i = 0; i < join.conditions.size(); i++) {
			auto &cond = join.conditions[i];
			// Check if comparison involves PAC aggregate (use plan-aware version)
			CategoricalPatternInfo info;
			string left_pac = FindPacAggregateInExpressionWithPlan(cond.left.get(), plan_root);
			string right_pac = FindPacAggregateInExpressionWithPlan(cond.right.get(), plan_root);

			if (!left_pac.empty() && !now_inside_aggregate) {
				info.comparison_expr = nullptr; // Join conditions are handled differently
				info.parent_op = op;
				info.expr_index = i;
				info.subquery_expr = cond.left.get();
				info.subquery_side = 0;
				info.aggregate_name = left_pac;
				patterns.push_back(info);
			} else if (!right_pac.empty() && !now_inside_aggregate) {
				info.comparison_expr = nullptr;
				info.parent_op = op;
				info.expr_index = i;
				info.subquery_expr = cond.right.get();
				info.subquery_side = 1;
				info.aggregate_name = right_pac;
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
	if (!source_op) {
		return nullptr;
	}

	if (source_op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return &source_op->Cast<LogicalAggregate>();
	}

	// If it's a projection, trace through to find the aggregate
	if (source_op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = source_op->Cast<LogicalProjection>();
		if (binding.column_index < proj.expressions.size()) {
			auto &expr = proj.expressions[binding.column_index];
			// Check if this expression is a column reference to an aggregate
			if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
				auto &col_ref = expr->Cast<BoundColumnRefExpression>();
				return FindAggregateForBinding(col_ref.binding, plan_root);
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
#ifdef DEBUG
						Printer::Print("Warning: Could not bind " + counters_name + ": " + error.Message());
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

#ifdef DEBUG
						Printer::Print("Rebound aggregate to " + counters_name + " with return type " +
						               new_aggr->return_type.ToString());
#endif

						// Replace the expression
						agg.expressions[i] = std::move(new_aggr);
					}

					// Update the aggregate operator's types vector
					idx_t types_index = agg.groups.size() + i;
					if (types_index < agg.types.size()) {
						agg.types[types_index] = LogicalType::LIST(LogicalType::DOUBLE);
#ifdef DEBUG
						Printer::Print("Updated aggregate types[" + std::to_string(types_index) +
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
			uint64_t binding_hash = (uint64_t(col_ref.binding.table_index) << 32) | col_ref.binding.column_index;
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
			uint64_t binding_hash = (uint64_t(col_ref.binding.table_index) << 32) | col_ref.binding.column_index;
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

// Find a single PAC aggregate binding in an expression
// Returns true if exactly one PAC aggregate binding is found
static bool FindSinglePacBindingInExpression(Expression *expr, LogicalOperator *plan_root, ColumnBinding &out_binding,
                                             string &out_aggregate_name) {
	unordered_set<uint64_t> pac_binding_hashes;
	idx_t count =
	    CountPacAggregateBindingsInExpression(expr, plan_root, pac_binding_hashes, &out_binding, &out_aggregate_name);
	return count == 1;
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

#ifdef DEBUG
		Printer::Print("    CloneForLambdaBody: checking col_ref [" + std::to_string(col_ref.binding.table_index) +
		               "." + std::to_string(col_ref.binding.column_index) + "] vs pac_binding [" +
		               std::to_string(pac_binding.table_index) + "." + std::to_string(pac_binding.column_index) + "]");
#endif

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
#ifdef DEBUG
			Printer::Print("    CloneForLambdaBody: MATCHED PAC binding! Replacing with element ref, type=" +
			               element_type.ToString());
#endif
			// Replace with BoundReferenceExpression(0) - the list element
			// With double-lambda, element_type is already the correct original type
			return make_uniq<BoundReferenceExpression>(col_ref.alias, element_type, 0);
		}

#ifdef DEBUG
		Printer::Print("    CloneForLambdaBody: NO MATCH - traced to [" + std::to_string(traced.table_index) + "." +
		               std::to_string(traced.column_index) + "] - capturing");
#endif
		// Other column ref - needs to be captured
		uint64_t hash = (uint64_t(col_ref.binding.table_index) << 32) | col_ref.binding.column_index;
		idx_t capture_idx;

		auto it = capture_map.find(hash);
		if (it != capture_map.end()) {
			capture_idx = it->second;
		} else {
			capture_idx = captures.size();
			capture_map[hash] = capture_idx;
			captures.push_back(col_ref.Copy());
		}

		// Replace with BoundReferenceExpression(1 + capture_idx)
		// Index 0 is the element, indices 1+ are captures
		return make_uniq<BoundReferenceExpression>(col_ref.alias, col_ref.return_type, 1 + capture_idx);
	}

	// Handle constant expressions
	if (expr->type == ExpressionType::VALUE_CONSTANT) {
		return expr->Copy();
	}

	// Handle cast expressions - use Copy() since BoundCastInfo is not easily copyable
	if (expr->type == ExpressionType::OPERATOR_CAST) {
		auto &cast = expr->Cast<BoundCastExpression>();
		auto child_clone =
		    CloneForLambdaBody(cast.child.get(), pac_binding, captures, capture_map, plan_root, element_type);
		// Copy the entire cast expression and replace its child
		auto cast_copy = cast.Copy();
		auto &new_cast = cast_copy->Cast<BoundCastExpression>();
		new_cast.child = std::move(child_clone);
		return cast_copy;
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

	// Handle operator expressions (e.g., AND, OR, NOT, arithmetic)
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_OPERATOR) {
		auto &op = expr->Cast<BoundOperatorExpression>();
		auto result = make_uniq<BoundOperatorExpression>(expr->type, op.return_type);
		for (auto &child : op.children) {
			result->children.push_back(
			    CloneForLambdaBody(child.get(), pac_binding, captures, capture_map, plan_root, element_type));
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
		return string(1, 'a' + index);
	}
	// For indices >= 26, use aa, ab, etc.
	return string(1, 'a' + (index / 26 - 1)) + string(1, 'a' + (index % 26));
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
		uint64_t binding_hash = (uint64_t(col_ref.binding.table_index) << 32) | col_ref.binding.column_index;

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

#ifdef DEBUG
			Printer::Print("    CloneForLambdaBodyMulti: PAC binding [" + std::to_string(col_ref.binding.table_index) +
			               "." + std::to_string(col_ref.binding.column_index) + "] -> struct field '" + field_name +
			               "' with type " + field_type.ToString());
#endif

			// Create: elem.field_name (where elem is BoundReferenceExpression(0))
			// The struct_extract will be done by accessing the struct field
			// In DuckDB, for a struct, we use struct_extract(struct, 'field_name')
			// But inside a lambda, we have BoundReferenceExpression(0) as the element
			// We need to create a struct_extract function call

			// Create the struct element reference
			auto elem_ref = make_uniq<BoundReferenceExpression>("elem", struct_type, 0);

			// Create struct_extract(elem, 'field_name')
			auto field_const = make_uniq<BoundConstantExpression>(Value(field_name));

			// Build the struct_extract manually (it's a special function)
			// The return type is the field type from the struct
			auto child_types = StructType::GetChildTypes(struct_type);
			LogicalType extract_return_type = LogicalType::DOUBLE;
			for (auto &child : child_types) {
				if (child.first == field_name) {
					extract_return_type = child.second;
					break;
				}
			}

			// Use struct_extract function
			vector<unique_ptr<Expression>> extract_children;
			extract_children.push_back(std::move(elem_ref));
			extract_children.push_back(std::move(field_const));

			// For struct_extract, we create a bound operator expression
			auto result = make_uniq<BoundOperatorExpression>(ExpressionType::STRUCT_EXTRACT, extract_return_type);
			result->children = std::move(extract_children);

			return result;
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
					uint64_t traced_hash = (uint64_t(traced.table_index) << 32) | traced.column_index;
					auto traced_it = binding_to_index.find(traced_hash);
					if (traced_it != binding_to_index.end()) {
						// Found it through tracing - same logic as above
						idx_t field_index = traced_it->second;
						string field_name = GetStructFieldName(field_index);
						LogicalType field_type = LogicalType::DOUBLE;
						auto type_it = binding_to_type.find(traced_hash);
						if (type_it != binding_to_type.end()) {
							field_type = type_it->second;
						}

						auto elem_ref = make_uniq<BoundReferenceExpression>("elem", struct_type, 0);
						auto field_const = make_uniq<BoundConstantExpression>(Value(field_name));
						auto child_types = StructType::GetChildTypes(struct_type);
						LogicalType extract_return_type = LogicalType::DOUBLE;
						for (auto &child : child_types) {
							if (child.first == field_name) {
								extract_return_type = child.second;
								break;
							}
						}
						vector<unique_ptr<Expression>> extract_children;
						extract_children.push_back(std::move(elem_ref));
						extract_children.push_back(std::move(field_const));
						auto result =
						    make_uniq<BoundOperatorExpression>(ExpressionType::STRUCT_EXTRACT, extract_return_type);
						result->children = std::move(extract_children);
						return result;
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
		idx_t capture_idx;
		auto cap_it = capture_map.find(binding_hash);
		if (cap_it != capture_map.end()) {
			capture_idx = cap_it->second;
		} else {
			capture_idx = captures.size();
			capture_map[binding_hash] = capture_idx;
			captures.push_back(col_ref.Copy());
		}

		// Replace with BoundReferenceExpression(1 + capture_idx)
		return make_uniq<BoundReferenceExpression>(col_ref.alias, col_ref.return_type, 1 + capture_idx);
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

	// Handle operator expressions (e.g., AND, OR, NOT, arithmetic)
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_OPERATOR) {
		auto &op = expr->Cast<BoundOperatorExpression>();
		auto result = make_uniq<BoundOperatorExpression>(expr->type, op.return_type);
		for (auto &child : op.children) {
			result->children.push_back(CloneForLambdaBodyMulti(child.get(), binding_to_index, binding_to_type, captures,
			                                                   capture_map, plan_root, struct_type));
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
#ifdef DEBUG
		Printer::Print("Warning: Could not bind list_zip: " + error.Message());
#endif
		return nullptr;
	}

	auto scalar_func = func_entry.functions.GetFunctionByOffset(best_function.GetIndex());

	// Build the function expression
	auto result = make_uniq<BoundFunctionExpression>(list_struct_type, scalar_func, std::move(counter_lists), nullptr);
	return result;
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

// Rewrite projection expressions that contain arithmetic with PAC aggregates
// e.g., 0.5 * agg_result becomes list_transform(agg_result, elem -> 0.5 * elem)
// This must be called AFTER ReplacePacAggregatesWithCounters
static void RewriteProjectionsWithCounters(LogicalOperator *op, OptimizerExtensionInput &input,
                                           LogicalOperator *plan_root,
                                           const unordered_set<LogicalAggregate *> &categorical_aggregates) {
	if (!op) {
		return;
	}

	// Process projections that have arithmetic with PAC counters
	if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = op->Cast<LogicalProjection>();
		for (idx_t i = 0; i < proj.expressions.size(); i++) {
			auto &expr = proj.expressions[i];

			// Check if this expression contains a PAC aggregate binding
			ColumnBinding pac_binding;
			string aggregate_name;
			if (!FindSinglePacBindingInExpression(expr.get(), plan_root, pac_binding, aggregate_name)) {
				continue;
			}

			// Check if this expression is just a column reference (no arithmetic)
			if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
				// Just a column ref, update its type and continue
				expr->return_type = LogicalType::LIST(LogicalType::DOUBLE);
				continue;
			}

			// Try to extract simple multiplication pattern: scalar * col_ref
			unique_ptr<Expression> scalar_expr;
			ColumnBinding mult_pac_binding;
			if (ExtractMultiplicationPattern(expr.get(), plan_root, scalar_expr, mult_pac_binding)) {
#ifdef DEBUG
				Printer::Print("Rewriting multiplication projection with PAC aggregate:");
				Printer::Print("  Original: " + expr->ToString());
				Printer::Print("  PAC binding: [" + std::to_string(mult_pac_binding.table_index) + "." +
				               std::to_string(mult_pac_binding.column_index) + "]");
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
				auto mult_body = input.optimizer.BindScalarFunction("*", std::move(scalar_expr), std::move(elem_ref));

				// Build the lambda
				auto lambda = BuildPacLambda(std::move(mult_body), {});

				// Get the counters column reference
				auto counters_ref = make_uniq<BoundColumnRefExpression>(
				    "pac_counters", LogicalType::LIST(LogicalType::DOUBLE), mult_pac_binding);

				// Build: list_transform(counters, elem -> scalar * elem) -> LIST<DOUBLE>
				auto list_transform =
				    BuildListTransformCall(input, std::move(counters_ref), std::move(lambda), LogicalType::DOUBLE);

#ifdef DEBUG
				Printer::Print("  Rewritten: " + list_transform->ToString());
#endif

				// Replace the expression
				proj.expressions[i] = std::move(list_transform);
				proj.types[i] = LogicalType::LIST(LogicalType::DOUBLE);
				continue;
			}

			// For more complex expressions, fall back to generic cloning approach
			// This may not work correctly for all cases due to type mismatches
#ifdef DEBUG
			Printer::Print("Warning: Complex projection expression with PAC aggregate not fully supported:");
			Printer::Print("  Expression: " + expr->ToString());
#endif
			// Just update the expression's return type and hope for the best
			// This is a fallback and may cause issues
		}
	}

	// Recurse into children
	for (auto &child : op->children) {
		RewriteProjectionsWithCounters(child.get(), input, plan_root, categorical_aggregates);
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
	// Step 1: Find which aggregates are part of categorical patterns
	// Only these aggregates should be converted to _counters variants
	unordered_set<LogicalAggregate *> categorical_aggregates;
	FindCategoricalAggregates(plan.get(), patterns, categorical_aggregates);

#ifdef DEBUG
	Printer::Print("Found " + std::to_string(categorical_aggregates.size()) + " categorical aggregate(s) to convert");
#endif

	// Step 2: Replace only the PAC aggregates that are part of categorical patterns
	ReplacePacAggregatesWithCounters(plan.get(), input.context, &categorical_aggregates);

	// Step 3: Rewrite projections that do arithmetic with PAC aggregates
	// e.g., 0.5 * agg_result becomes list_transform(agg_result, elem -> 0.5 * elem)
	RewriteProjectionsWithCounters(plan.get(), input, plan.get(), categorical_aggregates);

	// Step 4: Resolve operator types after aggregate and projection replacement
	// This ensures column references get the correct LIST<DOUBLE> type
	plan->ResolveOperatorTypes();

#ifdef DEBUG
	Printer::Print("=== PLAN AFTER COUNTERS REPLACEMENT ===");
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

#ifdef DEBUG
			Printer::Print("Filter expression has " + std::to_string(pac_bindings.size()) + " PAC binding(s)");
			Printer::Print("  Filter expr: " + filter_expr->ToString());
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

#ifdef DEBUG
				Printer::Print("Single-aggregate double-lambda rewrite:");
				Printer::Print("  PAC binding: [" + std::to_string(pac_binding.table_index) + "." +
				               std::to_string(pac_binding.column_index) + "]");
				Printer::Print("  Original type: " + original_type.ToString());
				Printer::Print("  Aggregate: " + binding_info.aggregate_name);
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

#ifdef DEBUG
				Printer::Print("  Single-aggregate rewrite complete");
#endif
			} else {
				// ============================================================
				// MULTIPLE AGGREGATES: Use list_zip approach
				// ============================================================
#ifdef DEBUG
				Printer::Print("Multi-aggregate list_zip rewrite:");
				for (auto &bi : pac_bindings) {
					Printer::Print("  PAC binding " + std::to_string(bi.index) + ": [" +
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

					uint64_t hash =
					    (uint64_t(binding_info.binding.table_index) << 32) | binding_info.binding.column_index;
					binding_to_index[hash] = binding_info.index;
					binding_to_type[hash] = binding_info.original_type;
				}

				// Build list_zip(A1, A2, ...) -> LIST<STRUCT<a DOUBLE, b DOUBLE, ...>>
				LogicalType struct_type;
				auto zipped_list = BuildListZipCall(input, std::move(counter_lists), struct_type);

				if (!zipped_list) {
#ifdef DEBUG
					Printer::Print("  Failed to build list_zip call");
#endif
					continue;
				}

#ifdef DEBUG
				Printer::Print("  Built list_zip with struct type: " + struct_type.ToString());
#endif

				// Build lambda body that:
				// 1. Extracts each struct field (elem.a, elem.b, ...)
				// 2. Casts to original type
				// 3. Evaluates the filter expression
				vector<unique_ptr<Expression>> captures;
				unordered_map<uint64_t, idx_t> capture_map;

				auto lambda_body = CloneForLambdaBodyMulti(filter_expr.get(), binding_to_index, binding_to_type,
				                                           captures, capture_map, plan.get(), struct_type);

#ifdef DEBUG
				Printer::Print("  Lambda body created with " + std::to_string(captures.size()) + " captures");
				Printer::Print("  Lambda body: " + lambda_body->ToString());
#endif

				// Build lambda: elem -> bool_expr(elem.a, elem.b, ...)
				auto lambda = BuildPacLambda(std::move(lambda_body), std::move(captures));

				// Build: list_transform(zipped_list, lambda) -> LIST<BOOL>
				auto list_bool =
				    BuildListTransformCall(input, std::move(zipped_list), std::move(lambda), LogicalType::BOOLEAN);

				// Wrap with pac_filter
				auto pac_filter = input.optimizer.BindScalarFunction("pac_filter", std::move(list_bool));
				filter_expr = std::move(pac_filter);

#ifdef DEBUG
				Printer::Print("  Multi-aggregate rewrite complete");
#endif
			}
		}
	}

	// Final resolve after all modifications
	plan->ResolveOperatorTypes();

#ifdef DEBUG
	// Debug: Print types of all projections to see if our changes persisted
	Printer::Print("=== FINAL TYPE CHECK ===");
	std::function<void(LogicalOperator *, int)> printTypes = [&](LogicalOperator *op, int depth) {
		if (!op)
			return;
		string indent(depth * 2, ' ');
		if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			auto &proj = op->Cast<LogicalProjection>();
			Printer::Print(indent + "PROJECTION #" + std::to_string(proj.table_index) + " types:");
			for (idx_t i = 0; i < proj.types.size(); i++) {
				Printer::Print(indent + "  [" + std::to_string(i) + "] = " + proj.types[i].ToString());
			}
		} else if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
			auto &agg = op->Cast<LogicalAggregate>();
			Printer::Print(indent + "AGGREGATE #" + std::to_string(agg.aggregate_index) + " types:");
			for (idx_t i = 0; i < agg.types.size(); i++) {
				Printer::Print(indent + "  [" + std::to_string(i) + "] = " + agg.types[i].ToString());
			}
		} else if (op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			auto &join = op->Cast<LogicalComparisonJoin>();
			Printer::Print(indent + "DELIM_JOIN types:");
			for (idx_t i = 0; i < op->types.size(); i++) {
				Printer::Print(indent + "  [" + std::to_string(i) + "] = " + op->types[i].ToString());
			}
			Printer::Print(indent + "DELIM_JOIN conditions:");
			for (idx_t i = 0; i < join.conditions.size(); i++) {
				Printer::Print(indent + "  cond[" + std::to_string(i) +
				               "].left type = " + join.conditions[i].left->return_type.ToString());
				Printer::Print(indent + "  cond[" + std::to_string(i) +
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
