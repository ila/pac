//
// PAC Categorical Query Rewriter
//
// This file implements automatic detection and rewriting of categorical queries.
// A categorical query is one where:
// - An inner subquery contains a PAC aggregate (e.g., pac_sum, pac_count)
// - The outer query compares the aggregate result without its own aggregate
// - Example: WHERE ps_availqty > (SELECT 0.5 * pac_sum(...) FROM lineitem ...)
//
// Detection (IsCategoricalQuery):
// - Finds comparisons (>, <, >=, <=, =) in the plan
// - Checks if one operand comes from a subquery with a PAC aggregate
// - Checks if the comparison is NOT inside another aggregate
//
// Rewriting:
// - Replaces pac_* aggregates with pac_*_counters variants (return LIST<DOUBLE>)
// - Wraps comparisons with pac_gt/pac_lt/etc. functions (return UBIGINT mask)
// - Adds pac_select at the outermost filter to make final probabilistic decision
//
// Example transformation for TPC-H Q20:
//   BEFORE: ps_availqty > (SELECT 0.5 * pac_sum(hash, l_quantity) FROM ...)
//   AFTER:  pac_select(pac_gt(ps_availqty, (SELECT 0.5 * pac_sum_counters(hash, l_quantity) FROM ...)))
//
// Created by ila on 1/23/26.
//

#ifndef PAC_CATEGORICAL_REWRITER_HPP
#define PAC_CATEGORICAL_REWRITER_HPP

#include "duckdb.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"

namespace duckdb {

// Information about a detected categorical pattern
struct CategoricalPatternInfo {
	// The comparison expression that needs rewriting
	Expression *comparison_expr;
	// The parent operator containing the comparison (usually a Filter)
	LogicalOperator *parent_op;
	// Index of the expression in the parent's expressions list
	idx_t expr_index;
	// The subquery expression containing the PAC aggregate
	Expression *subquery_expr;
	// Which side of the comparison has the subquery (0 = left, 1 = right)
	idx_t subquery_side;
	// The aggregate function name (e.g., "pac_sum", "pac_count")
	string aggregate_name;

	CategoricalPatternInfo()
	    : comparison_expr(nullptr), parent_op(nullptr), expr_index(0), subquery_expr(nullptr), subquery_side(0) {
	}
};

// Check if the plan contains a categorical query pattern
// Returns true if found, and populates pattern_info with details
bool IsCategoricalQuery(unique_ptr<LogicalOperator> &plan, vector<CategoricalPatternInfo> &patterns);

// Check if an expression contains a PAC aggregate (directly or in subquery)
// Returns the name of the PAC aggregate if found, empty string otherwise
string FindPacAggregateInExpression(Expression *expr);

// Check if an expression is a comparison that involves a PAC aggregate result
bool IsComparisonWithPacAggregate(Expression *expr, CategoricalPatternInfo &info);

// Rewrite a categorical query to use counters and mask-based selection
// This modifies the plan in-place
void RewriteCategoricalQuery(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                             vector<CategoricalPatternInfo> &patterns);

// Convert a PAC aggregate name to its counters variant
// e.g., "pac_sum" -> "pac_sum_counters", "pac_count" -> "pac_count_counters"
string GetCountersVariant(const string &aggregate_name);

// Get the appropriate comparison function for a given comparison type
// e.g., ExpressionType::COMPARE_GREATERTHAN -> "pac_gt"
string GetPacComparisonFunction(ExpressionType comparison_type);

} // namespace duckdb

#endif // PAC_CATEGORICAL_REWRITER_HPP
