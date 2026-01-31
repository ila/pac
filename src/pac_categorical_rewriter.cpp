//
// PAC Categorical Query Rewriter - Implementation
//
// See pac_categorical_rewriter.hpp for design documentation.
//
// Created by ila on 1/23/26.
//

#include "include/pac_categorical_rewriter.hpp"
#include "include/pac_plan_traversal.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_subquery_expression.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"

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

string GetPacComparisonFunction(ExpressionType comparison_type) {
	switch (comparison_type) {
	case ExpressionType::COMPARE_GREATERTHAN:
		return "pac_gt";
	case ExpressionType::COMPARE_LESSTHAN:
		return "pac_lt";
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return "pac_gte";
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return "pac_lte";
	case ExpressionType::COMPARE_EQUAL:
		return "pac_eq";
	case ExpressionType::COMPARE_NOTEQUAL:
		return "pac_neq";
	default:
		return "";
	}
}

// Forward declaration for plan-aware version
static string FindPacAggregateInExpressionWithPlan(Expression *expr, LogicalOperator *plan_root);

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

// Trace a column binding through the plan to find if it comes from a PAC aggregate
// Returns the PAC aggregate name if found, empty string otherwise
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
					if (IsPacAggregate(bound_agg.function.name)) {
						return bound_agg.function.name;
					}
				}
			}
		}
		// If it's from group_index, it's a GROUP BY column, not an aggregate result
	}

	return "";
}

// Recursively search for PAC aggregate in an expression tree, with plan context for tracing column refs
static string FindPacAggregateInExpressionWithPlan(Expression *expr, LogicalOperator *plan_root) {
	if (!expr) {
		return "";
	}

	// Check if this is a bound aggregate expression with a PAC function
	if (expr->type == ExpressionType::BOUND_AGGREGATE) {
		auto &agg_expr = expr->Cast<BoundAggregateExpression>();
		if (IsPacAggregate(agg_expr.function.name)) {
			return agg_expr.function.name;
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
static void FindCategoricalPatternsInOperatorWithPlan(LogicalOperator *op, LogicalOperator *plan_root,
                                                      vector<CategoricalPatternInfo> &patterns, bool inside_aggregate) {
	if (!op) {
		return;
	}

	// Track if we're entering an aggregate
	bool now_inside_aggregate = inside_aggregate || (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY);

	// Check filter expressions
	if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
		auto &filter = op->Cast<LogicalFilter>();
		for (idx_t i = 0; i < filter.expressions.size(); i++) {
			CategoricalPatternInfo info;
			if (IsComparisonWithPacAggregateWithPlan(filter.expressions[i].get(), info, plan_root)) {
				// Only add if NOT inside an aggregate (categorical = outer query has no aggregate)
				if (!now_inside_aggregate) {
					// IMPORTANT: Check if this is a HAVING clause pattern
					// A HAVING clause references an aggregate from the immediate child aggregate operator
					// A categorical query references an aggregate from a subquery (different branch)
					bool is_having_clause = false;

					if (info.subquery_expr && info.subquery_expr->type == ExpressionType::BOUND_COLUMN_REF) {
						auto &col_ref = info.subquery_expr->Cast<BoundColumnRefExpression>();
						// Trace the binding through projections to find the original source
						ColumnBinding traced_binding = TraceBindingThroughProjections(col_ref.binding, plan_root);
						is_having_clause = IsHavingClausePattern(op, traced_binding, plan_root);
					}

					if (!is_having_clause) {
						info.parent_op = op;
						info.expr_index = i;
						patterns.push_back(info);
					}
				}
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
		if (!pattern.subquery_expr) {
			continue;
		}

		// Trace from the subquery expression to find the aggregate operator
		if (pattern.subquery_expr->type == ExpressionType::BOUND_COLUMN_REF) {
			auto &col_ref = pattern.subquery_expr->Cast<BoundColumnRefExpression>();
			auto *agg = FindAggregateForBinding(col_ref.binding, plan_root);
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

// Helper to check if an expression references a PAC counters aggregate result
// This traces column references back to their source
static bool ReferencesPacCountersAggregate(Expression *expr, LogicalOperator *plan_root) {
	if (!expr || !plan_root) {
		return false;
	}

	// Direct check for LIST type (if ResolveOperatorTypes worked)
	if (expr->return_type.id() == LogicalTypeId::LIST) {
		return true;
	}

	// If it's a column reference, trace it back to the source
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr->Cast<BoundColumnRefExpression>();
		auto *source_op = FindOperatorByTableIndex(plan_root, col_ref.binding.table_index);
		if (source_op && source_op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
			auto &aggr = source_op->Cast<LogicalAggregate>();
			if (col_ref.binding.table_index == aggr.aggregate_index &&
			    col_ref.binding.column_index < aggr.expressions.size()) {
				auto &agg_expr = aggr.expressions[col_ref.binding.column_index];
				if (agg_expr->type == ExpressionType::BOUND_AGGREGATE) {
					auto &bound_agg = agg_expr->Cast<BoundAggregateExpression>();
					if (IsPacCountersAggregate(bound_agg.function.name)) {
						return true;
					}
				}
			}
		}
	}

	return false;
}

// Helper to rewrite multiplication expressions involving LIST<DOUBLE> counters to pac_scale_counters
// Must be called AFTER ResolveOperatorTypes() so column references have updated types
static void RewriteArithmeticWithCounters(LogicalOperator *op, OptimizerExtensionInput *input,
                                          LogicalOperator *plan_root) {
	if (!op || !input) {
		return;
	}

	// If this is a projection, check for arithmetic expressions that wrap aggregate results
	// e.g., 0.5 * #[26.0] where #[26.0] is now LIST<DOUBLE> needs to become pac_scale_counters(0.5, #[26.0])
	if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = op->Cast<LogicalProjection>();
		for (idx_t i = 0; i < proj.expressions.size(); i++) {
			auto &expr = proj.expressions[i];
			// Check for multiplication: constant * column_ref or column_ref * constant
			// In DuckDB, multiplication is a BOUND_FUNCTION with name "*"
			if (expr->type == ExpressionType::BOUND_FUNCTION) {
				auto &func_expr = expr->Cast<BoundFunctionExpression>();
				if (func_expr.function.name == "*" && func_expr.children.size() == 2) {
					Expression *scalar_child = nullptr;
					Expression *list_child = nullptr;
					idx_t scalar_idx = 0;
					idx_t list_idx = 0;

					// Find which child is the scalar and which might be a list (PAC counters)
					for (idx_t j = 0; j < 2; j++) {
						auto &child = func_expr.children[j];
						// Check if this child references a PAC counters aggregate
						if (ReferencesPacCountersAggregate(child.get(), plan_root)) {
							list_child = child.get();
							list_idx = j;
						} else if (child->type == ExpressionType::VALUE_CONSTANT ||
						           child->type == ExpressionType::OPERATOR_CAST) {
							scalar_child = child.get();
							scalar_idx = j;
						}
					}

					// If we found a scalar * list pattern, replace with pac_scale_counters
					if (scalar_child && list_child) {
#ifdef DEBUG
						Printer::Print("Rewriting multiplication to pac_scale_counters");
						Printer::Print("  scalar type: " + func_expr.children[scalar_idx]->return_type.ToString());
						Printer::Print("  list type: " + func_expr.children[list_idx]->return_type.ToString());
#endif
						// Update the column reference's return type to LIST<DOUBLE> before binding
						func_expr.children[list_idx]->return_type = LogicalType::LIST(LogicalType::DOUBLE);

						auto scale_func = input->optimizer.BindScalarFunction("pac_scale_counters",
						                                                      std::move(func_expr.children[scalar_idx]),
						                                                      std::move(func_expr.children[list_idx]));
						proj.expressions[i] = std::move(scale_func);
					}
				}
			}
		}
	}

	// Recurse into children
	for (auto &child : op->children) {
		RewriteArithmeticWithCounters(child.get(), input, plan_root);
	}
}

// Legacy function that combines both passes (kept for backward compatibility in expression-based replacement)
static void ReplacePacAggregatesInPlan(LogicalOperator *op, OptimizerExtensionInput *input) {
	ReplacePacAggregatesWithCounters(op, input->context, nullptr);
	// Note: For proper arithmetic rewriting, caller should call ResolveOperatorTypes() then
	// RewriteArithmeticWithCounters()
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

	// Step 3: Resolve operator types after aggregate replacement
	// This ensures column references get the correct LIST<DOUBLE> type
	plan->ResolveOperatorTypes();

	// Step 4: Rewrite arithmetic expressions that wrap counters (e.g., 0.5 * pac_sum -> pac_scale_counters)
	// This must happen AFTER ResolveOperatorTypes() so column refs have LIST<DOUBLE> type
	RewriteArithmeticWithCounters(plan.get(), &input, plan.get());

	// Step 5: Resolve types again after arithmetic rewriting
	plan->ResolveOperatorTypes();

#ifdef DEBUG
	Printer::Print("=== PLAN AFTER COUNTERS REPLACEMENT ===");
	plan->Print();
#endif

	// Step 6: Now rewrite the comparison expressions
	for (auto &pattern : patterns) {
		if (!pattern.parent_op) {
			continue;
		}

		// Wrap the comparison with pac_gt/pac_lt/etc. and pac_select
		if (pattern.parent_op->type == LogicalOperatorType::LOGICAL_FILTER) {
			auto &filter = pattern.parent_op->Cast<LogicalFilter>();
			auto &comp_expr_ptr = filter.expressions[pattern.expr_index];

			if (comp_expr_ptr->type == ExpressionType::COMPARE_GREATERTHAN ||
			    comp_expr_ptr->type == ExpressionType::COMPARE_LESSTHAN ||
			    comp_expr_ptr->type == ExpressionType::COMPARE_GREATERTHANOREQUALTO ||
			    comp_expr_ptr->type == ExpressionType::COMPARE_LESSTHANOREQUALTO ||
			    comp_expr_ptr->type == ExpressionType::COMPARE_EQUAL ||
			    comp_expr_ptr->type == ExpressionType::COMPARE_NOTEQUAL) {

				auto &comp = comp_expr_ptr->Cast<BoundComparisonExpression>();
				string pac_func = GetPacComparisonFunction(comp_expr_ptr->type);

				if (!pac_func.empty()) {
					// Determine which operand is the scalar and which is the counters list
					// The subquery side has the counters, the other side is the scalar value
					unique_ptr<Expression> scalar_expr;
					unique_ptr<Expression> counters_expr;

					if (pattern.subquery_side == 0) {
						// Left side has subquery (counters), right side is scalar
						counters_expr = std::move(comp.left);
						scalar_expr = std::move(comp.right);
					} else {
						// Right side has subquery (counters), left side is scalar
						scalar_expr = std::move(comp.left);
						counters_expr = std::move(comp.right);
					}

#ifdef DEBUG
					Printer::Print("Categorical rewrite: counters_expr type = " +
					               std::to_string((int)counters_expr->type));
					Printer::Print("Categorical rewrite: counters_expr class = " +
					               std::to_string((int)counters_expr->GetExpressionClass()));
					if (counters_expr->type == ExpressionType::BOUND_COLUMN_REF) {
						auto &col_ref = counters_expr->Cast<BoundColumnRefExpression>();
						Printer::Print("Categorical rewrite: column binding = [" +
						               std::to_string(col_ref.binding.table_index) + "." +
						               std::to_string(col_ref.binding.column_index) + "]");
						Printer::Print("Categorical rewrite: column type before = " + col_ref.return_type.ToString());
					}
#endif

					// If counters_expr is a column reference, we need to update the source operator's type too
					if (counters_expr->type == ExpressionType::BOUND_COLUMN_REF) {
						auto &col_ref = counters_expr->Cast<BoundColumnRefExpression>();
						auto *source_op = FindOperatorByTableIndex(plan.get(), col_ref.binding.table_index);

						if (source_op) {
							// Update the source projection or aggregate expression's return type
							if (source_op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
								auto &source_proj = source_op->Cast<LogicalProjection>();
								if (col_ref.binding.column_index < source_proj.expressions.size()) {
									auto &source_expr = source_proj.expressions[col_ref.binding.column_index];
									// Check if this is a pac_scale_counters or similar function
									if (source_expr->type == ExpressionType::BOUND_FUNCTION) {
										auto &func = source_expr->Cast<BoundFunctionExpression>();
										if (func.function.name == "pac_scale_counters" ||
										    func.function.name.find("pac_") == 0) {
											// Update all the type fields
											func.return_type = LogicalType::LIST(LogicalType::DOUBLE);
											source_expr->return_type = LogicalType::LIST(LogicalType::DOUBLE);

											// CRITICAL: Also update the projection's types vector directly
											// This is what DuckDB uses during execution
											if (col_ref.binding.column_index < source_proj.types.size()) {
												source_proj.types[col_ref.binding.column_index] =
												    LogicalType::LIST(LogicalType::DOUBLE);
#ifdef DEBUG
												Printer::Print("Updated source projection #" +
												               std::to_string(col_ref.binding.table_index) + " types[" +
												               std::to_string(col_ref.binding.column_index) +
												               "] to LIST<DOUBLE>");
												Printer::Print("  source_proj.types.size() = " +
												               std::to_string(source_proj.types.size()));
												for (idx_t ti = 0; ti < source_proj.types.size(); ti++) {
													Printer::Print("    types[" + std::to_string(ti) +
													               "] = " + source_proj.types[ti].ToString());
												}
#endif
											}
										}
									} else if (source_expr->type == ExpressionType::BOUND_COLUMN_REF) {
										// The projection contains a column reference - check if it points to a PAC
										// counters aggregate
										auto &inner_col_ref = source_expr->Cast<BoundColumnRefExpression>();
										auto *inner_source_op =
										    FindOperatorByTableIndex(plan.get(), inner_col_ref.binding.table_index);
										if (inner_source_op &&
										    inner_source_op->type ==
										        LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
											auto &inner_agg = inner_source_op->Cast<LogicalAggregate>();
											// Check if this binding refers to an aggregate expression (not a group
											// column)
											if (inner_col_ref.binding.table_index == inner_agg.aggregate_index &&
											    inner_col_ref.binding.column_index < inner_agg.expressions.size()) {
												auto &inner_agg_expr =
												    inner_agg.expressions[inner_col_ref.binding.column_index];
												if (inner_agg_expr->type == ExpressionType::BOUND_AGGREGATE) {
													auto &inner_bound_agg =
													    inner_agg_expr->Cast<BoundAggregateExpression>();
													if (IsPacCountersAggregate(inner_bound_agg.function.name)) {
														// Update the column reference's return type
														inner_col_ref.return_type =
														    LogicalType::LIST(LogicalType::DOUBLE);
														source_expr->return_type =
														    LogicalType::LIST(LogicalType::DOUBLE);

														// Update the projection's types vector
														if (col_ref.binding.column_index < source_proj.types.size()) {
															source_proj.types[col_ref.binding.column_index] =
															    LogicalType::LIST(LogicalType::DOUBLE);
#ifdef DEBUG
															Printer::Print(
															    "Updated source projection #" +
															    std::to_string(col_ref.binding.table_index) +
															    " types[" +
															    std::to_string(col_ref.binding.column_index) +
															    "] to LIST<DOUBLE> (via column ref to aggregate)");
#endif
														}
													}
												}
											}
										}
									}
								}
							} else if (source_op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
								auto &source_agg = source_op->Cast<LogicalAggregate>();
								if (col_ref.binding.table_index == source_agg.aggregate_index &&
								    col_ref.binding.column_index < source_agg.expressions.size()) {
									auto &source_expr = source_agg.expressions[col_ref.binding.column_index];
									if (source_expr->type == ExpressionType::BOUND_AGGREGATE) {
										auto &agg_func = source_expr->Cast<BoundAggregateExpression>();
										if (IsPacCountersAggregate(agg_func.function.name)) {
											// Update all the type fields
											agg_func.function.return_type = LogicalType::LIST(LogicalType::DOUBLE);
											agg_func.return_type = LogicalType::LIST(LogicalType::DOUBLE);
											source_expr->return_type = LogicalType::LIST(LogicalType::DOUBLE);

											// CRITICAL: Also update the aggregate's types vector directly
											if (col_ref.binding.column_index < source_agg.types.size()) {
												source_agg.types[col_ref.binding.column_index] =
												    LogicalType::LIST(LogicalType::DOUBLE);
#ifdef DEBUG
												Printer::Print("Updated source aggregate #" +
												               std::to_string(col_ref.binding.table_index) + " types[" +
												               std::to_string(col_ref.binding.column_index) +
												               "] to LIST<DOUBLE>");
#endif
											}
										}
									}
								}
							}
						}
					}

					// The counters expression may still have the old type (e.g., DECIMAL) if
					// ResolveOperatorTypes didn't propagate the LIST<DOUBLE> type through all references.
					// Force update the return type to LIST<DOUBLE> for proper function binding.
					counters_expr->return_type = LogicalType::LIST(LogicalType::DOUBLE);

#ifdef DEBUG
					Printer::Print("Categorical rewrite: scalar type = " + scalar_expr->return_type.ToString());
					Printer::Print("Categorical rewrite: counters type = " + counters_expr->return_type.ToString());
#endif

					// Build pac_gt/pac_lt call: pac_gt(scalar, counters) -> UBIGINT mask
					auto pac_comparison =
					    input.optimizer.BindScalarFunction(pac_func, std::move(scalar_expr), std::move(counters_expr));

					// Build pac_select call: pac_select(mask) -> BOOLEAN
					auto pac_select = input.optimizer.BindScalarFunction("pac_select", std::move(pac_comparison));

					// Replace the original comparison with pac_select(pac_gt(...))
					comp_expr_ptr = std::move(pac_select);
				}
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
