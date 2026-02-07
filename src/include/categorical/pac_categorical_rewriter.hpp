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

namespace duckdb {

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

	CategoricalPatternInfo()
	    : parent_op(nullptr), expr_index(0), has_pac_binding(false), original_return_type(LogicalType::DOUBLE),
	      scalar_wrapper_op(nullptr) {
	}
};

// Check if the plan contains a categorical query pattern
// Returns true if found, and populates pattern_info with details
bool IsCategoricalQuery(unique_ptr<LogicalOperator> &plan, vector<CategoricalPatternInfo> &patterns);

// Rewrite a categorical query to use counters and mask-based selection
// This modifies the plan in-place
void RewriteCategoricalQuery(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                             vector<CategoricalPatternInfo> &patterns);

// Check if a function name is a PAC aggregate
static inline bool IsPacAggregate(const string &pattern, const string &suffix = "", const string &prefix = "pac_") {
	const string name = StringUtil::Lower(pattern);
	for (auto &aggr_name : {"sum", "count", "avg", "min", "max"}) {
		if (name == prefix + aggr_name + suffix) {
			return true;
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

// Information about a single PAC aggregate binding found in an expression
struct PacBindingInfo {
	ColumnBinding binding;
	string aggregate_name;     // e.g., "pac_sum", "pac_count"
	LogicalType original_type; // The type before conversion to LIST<DOUBLE>
	idx_t index;               // Position in the list (0-based, for list_zip field access)
};

// Pre-collected rewrite info for a single expression (filter, join cond, or projection expr).
// Groups all PAC bindings found in that expression so we know upfront if we need list_zip.
struct ExpressionRewriteInfo {
	LogicalOperator *parent_op;          // Filter, Join, or Projection operator
	idx_t expr_index;                    // Index into parent's expressions/conditions
	vector<PacBindingInfo> pac_bindings; // ALL PAC bindings in this expression
	LogicalType original_return_type;    // Original type before counters conversion
};

} // namespace duckdb

#endif // PAC_CATEGORICAL_REWRITER_HPP
