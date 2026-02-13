//
// PAC Categorical Query Rewriter
//
// This file implements automatic detection and rewriting of categorical queries.
// A categorical query is one where:
// - An inner subquery contains a PAC aggregate (e.g., pac_sum, pac_count)
// - The outer query compares the aggregate result without its own aggregate
// - Example: WHERE ps_availqty > (SELECT 0.5 * pac_sum(...) FROM lineitem ...)
//
// Detection (FindCategoricalPatternsInOperator):
// - Finds comparisons (>, <, >=, <=, =) in the plan
// - Checks if one operand comes from a subquery with a PAC aggregate
// - Checks if the comparison is NOT inside another aggregate
//
// Rewriting:
// - Replaces pac_* aggregates with pac_*_counters variants (return LIST<DOUBLE>)
// - Wraps comparisons with pac_gt/pac_lt/etc. functions (return UBIGINT mask)
// - Adds pac_filter at the outermost filter to make final probabilistic decision
//
// Example transformation for TPC-H Q20:
//   BEFORE: ps_availqty > (SELECT 0.5 * pac_sum(hash, l_quantity) FROM ...)
//   AFTER:  pac_filter(pac_gt(ps_availqty, (SELECT 0.5 * pac_sum_counters(hash, l_quantity) FROM ...)))
//
// Created by ila on 1/23/26.
//

#ifndef PAC_CATEGORICAL_REWRITER_HPP
#define PAC_CATEGORICAL_REWRITER_HPP

#include "duckdb.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "query_processing/pac_plan_traversal.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_subquery_expression.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
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

// Information about a single PAC aggregate binding found in an expression
struct PacBindingInfo {
	ColumnBinding binding;
	string aggregate_name;     // e.g., "pac_sum", "pac_count"
	LogicalType original_type; // The type before conversion to LIST<DOUBLE>
	idx_t index;               // Position in the list (0-based, for list_zip field access)
};

// Information about a detected categorical pattern
struct CategoricalPatternInfo {
	// The parent operator containing the expression (Filter, Join, or Projection)
	LogicalOperator *parent_op;
	// Index of the expression in the parent's expressions list (or conditions list for joins)
	idx_t expr_index;
	// The aggregate function name (e.g., "pac_sum", "pac_count")
	string aggregate_name;
	// The column binding that references the PAC aggregate result
	ColumnBinding pac_binding;
	// Whether we have a valid pac_binding
	bool has_pac_binding;
	// The original return type of the PAC aggregate expression (before conversion to LIST<DOUBLE>)
	// Used by double-lambda rewrite to cast list elements back to the expected type
	LogicalType original_return_type;
	// Scalar subquery wrapper that was skipped during pattern detection (if any)
	// Points to the outer Projection of the pattern: Project(CASE) -> Aggregate(first) -> Project
	// When set, these three operators should be stripped during rewrite
	LogicalOperator *scalar_wrapper_op;
	// Pre-collected PAC bindings from detection
	vector<PacBindingInfo> pac_bindings;

	CategoricalPatternInfo()
	    : parent_op(nullptr), expr_index(0), has_pac_binding(false), original_return_type(LogicalType::DOUBLE),
	      scalar_wrapper_op(nullptr) {
	}
};

// The kind of wrapping to apply after BuildCounterListTransform
enum class PacWrapKind {
	PAC_NOISED, // Projection: list_transform -> pac_noised -> optional cast
	PAC_FILTER  // Filter/Join: list_transform -> pac_filter
};

// Detect and rewrite categorical query patterns to use counters and mask-based selection.
// This modifies the plan in-place. No-op if no categorical patterns are found.
void RewriteCategoricalQuery(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan);

// ==========================================================================================
// simple inlined methods
// ==========================================================================================
static inline bool IsPacAggregate(const string &pattern, const string &suffix = "", const string &prefix = "pac_") {
	const string name = StringUtil::Lower(pattern);
	for (auto &aggr_name : {"sum", "count", "avg", "min", "max"}) {
		if (name == prefix + aggr_name + suffix) {
			return true; // function name is some PAC aggregate
		}
	}
	return false;
}

static inline bool IsPacCountersAggregate(const string &name) {
	return IsPacAggregate(name, "_counters"); // Check if a function name is already a PAC counters variant
}

static inline bool IsPacListAggregate(const string &name) {
	return IsPacAggregate(name, "_list"); // Check if a function name is a PAC list aggregate (pac_*_list)
}

static inline bool IsAnyPacAggregate(const string &name) {
	return IsPacAggregate(name) || IsPacCountersAggregate(name) || IsPacListAggregate(name);
}

string inline GetCountersVariant(const string &aggregate_name) {
	if (IsPacCountersAggregate(aggregate_name)) {
		return aggregate_name;
	}
	D_ASSERT(IsPacAggregate(aggregate_name));
	return aggregate_name + "_counters";
}

static inline string GetBasePacAggregateName(const string &name) {
	if (IsPacCountersAggregate(name)) {
		return name.substr(0, name.size() - 9); // Remove "_counters" suffix
	}
	return name;
}

static inline string GetListAggregateVariant(const string &name) {
	if (IsPacAggregate(name, "", "")) {
		return "pac_" + name + "_list";
	}
	return "";
}

// Check if a type is numerical (can be used with pac_noised)
inline bool IsNumericalType(const LogicalType &type) {
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

// Hash a column binding to a unique 64-bit value for use in maps/sets
static inline uint64_t HashBinding(const ColumnBinding &binding) {
	return (uint64_t(binding.table_index) << 32) | binding.column_index;
}

// ==========================================================================================
// inlined  helper methods for the categorical rewrites
// ==========================================================================================

// Get struct field name for index (a, b, c, ..., z, aa, ab, ...)
static inline string GetStructFieldName(idx_t index) {
	if (index < 26) {
		return string(1, static_cast<char>('a' + static_cast<unsigned char>(index)));
	}
	// For indices >= 26, use aa, ab, etc.
	return string(1, static_cast<char>('a' + static_cast<unsigned char>(index / 26 - 1))) +
	       string(1, static_cast<char>('a' + static_cast<unsigned char>(index % 26)));
}

// Check if an expression is already wrapped in a categorical rewrite terminal function
// This includes pac_noised, pac_filter, list_transform, and list_zip
// These functions indicate that the expression has already been processed by categorical rewriting
static inline bool IsAlreadyWrappedInPacNoised(Expression *expr) {
	if (!expr) {
		return false;
	}
	if (expr->type == ExpressionType::BOUND_FUNCTION) {
		auto &func = expr->Cast<BoundFunctionExpression>();
		// Check for all categorical rewrite terminal functions
		if (func.function.name == "pac_noised" || func.function.name == "pac_filter" ||
		    func.function.name == "pac_coalesce" || func.function.name == "list_transform" ||
		    func.function.name == "list_zip" || StringUtil::StartsWith(func.function.name, "pac_filter_")) {
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

// Search an operator subtree for PAC aggregates
// Returns the first PAC aggregate name found, empty string otherwise
static inline string FindPacAggregateInOperator(LogicalOperator *op) {
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

// Helper: Find the operator in the plan that produces a given table_index
static inline LogicalOperator *FindOperatorByTableIndex(LogicalOperator *op, idx_t table_index) {
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

// DuckDB wraps scalar subqueries in a runtime check that the query returns 1 row. For our rewrites, we know this is
// fine, and the wrapper operators are in the way, so we find and remove them, when inside a rewritten subtree.
//
// Recognize (and optionally strip) a scalar subquery wrapper operator in place
// Pattern: Project(CASE) #X -> Aggregate(first, count*) -> Project #Z -> [inner]
// When remove=true: deletes outer Project and Aggregate, keeps inner Project with outer's table_index
// Returns the inner operator (below inner projection) if pattern matches, nullptr otherwise
static inline bool IsScalarSubqueryWrapper(const BoundCaseExpression &case_expr) {
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

// Recognize DuckDB's scalar subquery wrapper pattern
//
// Pattern  (uncorrelated): Projection -> Aggregate(first) -> Projection
//   Projection (CASE error if count > 1, else first(value))
//   └── Aggregate (first(#0), count_star())
//       └── Projection (#0)
//           └── [actual scalar subquery result]
// Check if a CASE expression is DuckDB's scalar subquery wrapper pattern:
// CASE WHEN count>1 THEN error(...) ELSE value END
static inline LogicalOperator *RecognizeDuckDBScalarWrapper(LogicalOperator *op) {
	auto &outer_proj = op->Cast<LogicalProjection>();
	if (outer_proj.expressions.empty()) {
		return nullptr;
	}
	auto &expr = outer_proj.expressions[0];
	if (expr->type != ExpressionType::CASE_EXPR) {
		return nullptr;
	}
	if (!IsScalarSubqueryWrapper(expr->Cast<BoundCaseExpression>())) {
		return nullptr;
	}
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
	auto &agg_expr = agg.expressions[0];
	if (agg_expr->type != ExpressionType::BOUND_AGGREGATE) {
		return nullptr;
	}
	auto &bound_agg = agg_expr->Cast<BoundAggregateExpression>();
	if (StringUtil::Lower(bound_agg.function.name) != "first") {
		return nullptr;
	}
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

static inline LogicalOperator *StripScalarWrapperInPlace(unique_ptr<LogicalOperator> &wrapper_ptr, bool remove = true) {
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
		inner_proj.table_index = outer_proj.table_index; // Change inner projection's table_index to match outer's
		inner_proj.types.clear();                        // Update inner projection's types to match its expressions
		for (auto &expr : inner_proj.expressions) {
			inner_proj.types.push_back(expr->return_type);
		}
		wrapper_ptr = std::move(agg.children[0]); // Replace the wrapper_ptr with the inner projection
	}
	return unwrapped;
}

// ============================================================================
// Algebraic simplification: try to emit pac_filter_<cmp> instead of lambda
// ============================================================================
// Given a comparison expression with one PAC binding, try to algebraically
// move arithmetic from the list side to the scalar side, then emit a single
// pac_filter_<cmp>(scalar, counters) call instead of list_transform + pac_filter.

// Map ExpressionType comparison to pac_filter_<cmp> function name
static inline const char *GetPacFilterCmpName(ExpressionType cmp_type) {
	switch (cmp_type) {
	case ExpressionType::COMPARE_GREATERTHAN:
		return "pac_filter_gt";
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return "pac_filter_gte";
	case ExpressionType::COMPARE_LESSTHAN:
		return "pac_filter_lt";
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return "pac_filter_lte";
	case ExpressionType::COMPARE_EQUAL:
		return "pac_filter_eq";
	case ExpressionType::COMPARE_NOTEQUAL:
		return "pac_filter_neq";
	default:
		return nullptr;
	}
}

// Flip a comparison when swapping sides (PAC was on left, move to right)
static inline ExpressionType FlipComparison(ExpressionType cmp_type) {
	switch (cmp_type) {
	case ExpressionType::COMPARE_GREATERTHAN:
		return ExpressionType::COMPARE_LESSTHAN;
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return ExpressionType::COMPARE_LESSTHANOREQUALTO;
	case ExpressionType::COMPARE_LESSTHAN:
		return ExpressionType::COMPARE_GREATERTHAN;
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return ExpressionType::COMPARE_GREATERTHANOREQUALTO;
	case ExpressionType::COMPARE_EQUAL:
		return ExpressionType::COMPARE_EQUAL;
	case ExpressionType::COMPARE_NOTEQUAL:
		return ExpressionType::COMPARE_NOTEQUAL;
	default:
		return cmp_type;
	}
}

// Try to extract a double value from an expression, looking through casts.
// Returns true and sets out_val if the expression is a constant that can be cast to DOUBLE.
static inline bool TryGetConstantDouble(Expression *expr, double &out_val) {
	while (expr->type == ExpressionType::OPERATOR_CAST) {
		expr = expr->Cast<BoundCastExpression>().child.get();
	}
	if (expr->type != ExpressionType::VALUE_CONSTANT) {
		return false;
	}
	auto &const_expr = expr->Cast<BoundConstantExpression>();
	if (const_expr.value.IsNull()) {
		return false;
	}
	Value double_val;
	string error_msg;
	if (const_expr.value.DefaultTryCastAs(LogicalType::DOUBLE, double_val, &error_msg)) {
		out_val = double_val.GetValue<double>();
		return true;
	}
	return false;
}

// Check if an expression is a positive constant (value > 0), looking through casts
static inline bool IsPositiveConstant(Expression *expr) {
	double val;
	return TryGetConstantDouble(expr, val) && val > 0.0;
}

// Given a binary arithmetic op and which child holds the PAC counter, return the inverse op
// to move the scalar operand to the other side of a comparison.
// Commutative ops (+, *) work regardless of pac_child position.
// Non-commutative ops (-, /) only work when counter is on the left (pac_child == 0).
// Sets needs_positive_check when the scalar operand must be > 0 to preserve inequality direction.
static inline const char *GetInverseArithmeticOp(const string &fname, int pac_child, bool &needs_positive_check) {
	needs_positive_check = false;
	if (fname == "+") {
		return "-";
	} else if (fname == "-") {
		return pac_child == 0 ? "+" : nullptr;
	} else if (fname == "*") {
		needs_positive_check = true;
		return "/";
	} else if (fname == "/" && pac_child == 0) {
		needs_positive_check = true;
		return "*";
	}
	return nullptr;
}

} // namespace duckdb

#endif // PAC_CATEGORICAL_REWRITER_HPP
