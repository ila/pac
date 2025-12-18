#include "include/pac_compatibility_check.hpp"
#include "include/pac_privacy_unit.hpp"

#include "duckdb.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

using namespace duckdb;

namespace duckdb {

static bool IsAllowedAggregate(const std::string &func) {
    static const std::unordered_set<std::string> allowed = {"sum", "sum_no_overflow", "count", "count_star", "avg"};
    std::string lower_func = func;
    std::transform(lower_func.begin(), lower_func.end(), lower_func.begin(), ::tolower);
    return allowed.count(lower_func) > 0;
}

static bool ContainsDisallowedJoin(const LogicalOperator &op) {
    // Handle different logical join operator types that derive from LogicalJoin
    if (op.type == LogicalOperatorType::LOGICAL_JOIN ||
        op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
        op.type == LogicalOperatorType::LOGICAL_DEPENDENT_JOIN ||
        op.type == LogicalOperatorType::LOGICAL_ASOF_JOIN ||
        op.type == LogicalOperatorType::LOGICAL_POSITIONAL_JOIN) {
        auto &join = op.Cast<LogicalJoin>();
        if (join.join_type != JoinType::INNER) {
            // Non-inner join detected: signal disallowed join via boolean return to let caller throw
            return true;
        }
    } else if (
        op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN ||
        op.type == LogicalOperatorType::LOGICAL_ANY_JOIN ||
        op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT ||
        op.type == LogicalOperatorType::LOGICAL_UNION ||
        op.type == LogicalOperatorType::LOGICAL_EXCEPT ||
        op.type == LogicalOperatorType::LOGICAL_INTERSECT) {
        // These operator types are disallowed for PAC compilation
        return true;
    }
    for (auto &child : op.children) {
        if (ContainsDisallowedJoin(*child)) return true;
    }
    return false;
}

static bool ContainsWindowFunction(const LogicalOperator &op) {
    if (op.type == LogicalOperatorType::LOGICAL_WINDOW) return true;
    for (auto &child : op.children) {
        if (ContainsWindowFunction(*child)) return true;
    }
    return false;
}

static bool ContainsDistinct(const LogicalOperator &op) {
    // If there's an explicit DISTINCT operator in the logical plan, detect it
    if (op.type == LogicalOperatorType::LOGICAL_DISTINCT) return true;

    // Inspect expressions attached to this operator: COUNT(DISTINCT ...), etc.
    for (auto &expr : op.expressions) {
        if (!expr) continue;
        if (expr->IsAggregate()) {
            auto &aggr = expr->Cast<BoundAggregateExpression>();
            if (aggr.IsDistinct()) return true;
        }
    }

    // Also check LogicalAggregate nodes' expressions (GROUP BY select list)
    if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
        auto &aggr = op.Cast<LogicalAggregate>();
        for (auto &expr : aggr.expressions) {
            if (!expr) continue;
            if (expr->IsAggregate()) {
                auto &a = expr->Cast<BoundAggregateExpression>();
                if (a.IsDistinct()) return true;
            }
        }
    }

    for (auto &child : op.children) {
        if (ContainsDistinct(*child)) return true;
    }
    return false;
}

static bool ContainsAggregation(const LogicalOperator &op) {
    if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
        auto &aggr = op.Cast<LogicalAggregate>();
        for (auto &expr : aggr.expressions) {
            if (expr && expr->IsAggregate()) {
                auto &ag = expr->Cast<BoundAggregateExpression>();
                if (IsAllowedAggregate(ag.function.name)) return true;
            }
        }
    }
    for (auto &child : op.children) {
        if (ContainsAggregation(*child)) return true;
    }
    return false;
}

// helper: traverse the plan and count how many times each table/CTE name is scanned
static void CountScans(const LogicalOperator &op, std::unordered_map<std::string, idx_t> &counts) {
    if (op.type == LogicalOperatorType::LOGICAL_GET) {
        auto &scan = op.Cast<LogicalGet>();
        auto table_entry = scan.GetTable();
        if (table_entry) {
            counts[table_entry->name]++;
        } else {
            for (auto &n : scan.names) {
                counts[n]++;
            }
        }
    }
    for (auto &child : op.children) {
        CountScans(*child, counts);
    }
}

bool PACRewriteQueryCheck(LogicalOperator &plan, ClientContext &context) {
    // Determine PAC tables filename from context setting (default pac_tables.csv)
    std::string pac_privacy_file = "pac_tables.csv";
    Value pac_privacy_file_value;
    if (context.TryGetCurrentSetting("pac_privacy_file", pac_privacy_file_value) && !pac_privacy_file_value.IsNull()) {
        pac_privacy_file = pac_privacy_file_value.ToString();
    }
    auto pac_tables = ReadPacTablesFile(pac_privacy_file);

	// Case 1: contains PAC tables?
	// Then, we check further conditions (aggregation, joins, etc.)
	// If there are as many sample-CTE scans as PAC table scans, then nothing to do (return false)
	// If there are only PAC table scans, check the other conditions and return true/false accordingly
	// If some conditions fail, throw an error in the caller
	// Case 2: no PAC tables? (return false, nothing to do)

    // count all scanned tables/CTEs in the plan
    std::unordered_map<std::string, idx_t> scan_counts;
    CountScans(plan, scan_counts);

    // require that at least one PAC table is scanned
    bool any_pac = false;
    for (auto &t : pac_tables) {
        if (scan_counts[t] > 0) {
            any_pac = true;
            break;
        }
    }
    if (!any_pac) return false;

    // For each scanned PAC table, require that the corresponding sample-CTE
    // "_pac_internal_sample_<table_name>" is scanned the same number of times.
    // If all PAC tables have matching sample scans, then nothing to do (return false).
    // If some tables have PAC scans but zero sample scans, we proceed to check eligibility.
    // If there is a mismatch where both counts are non-zero but unequal, that's an error.
    bool all_matched = true;
    for (auto &t : pac_tables) {
        idx_t pac_count = scan_counts[t];
        if (pac_count == 0) continue;
        std::string sample_name = std::string("_pac_internal_sample_") + t;
        idx_t sample_count = 0;
        auto it = scan_counts.find(sample_name);
        if (it != scan_counts.end()) sample_count = it->second;
        if (pac_count != sample_count) {
            all_matched = false;
            if (sample_count == 0) {
                // only PAC table scanned for this table, proceed to eligibility checks
            } else {
                // mismatch where both are non-zero -> ambiguous/invalid plan for PAC rewriting
                throw InvalidInputException("PAC rewrite: mismatch between PAC table scans (%s=%llu) and internal sample scans (%s=%llu)",
                                            t.c_str(), (unsigned long long)pac_count, sample_name.c_str(), (unsigned long long)sample_count);
            }
        }
    }

    if (all_matched) {
        // All PAC tables already have corresponding sample CTE scans the same number of times.
        // Nothing for the PAC rewriter to do.
        return false;
    }

    // If we reach here, there is at least one PAC table that has PAC scans without matching sample-CTE scans.
    // Now validate plan structure: ensure there is an aggregation using allowed aggregates; disallow window, distinct, non-inner joins.

    // Require there to be an aggregation (sum/count/avg) somewhere in the plan; otherwise nothing to do
    // If there's no allowed aggregation but we have PAC tables without matching samples,
    // this is an invalid query for PAC compilation and should be rejected.
	if (ContainsWindowFunction(plan)) {
		throw InvalidInputException("PAC rewrite: window functions are not supported for PAC compilation");
	}
    if (!ContainsAggregation(plan)) {
        throw InvalidInputException("Query does not contain any allowed aggregation (sum, count, avg)!");
    }
    if (ContainsDistinct(plan)) {
        throw InvalidInputException("PAC rewrite: DISTINCT is not supported for PAC compilation");
    }
    if (ContainsDisallowedJoin(plan)) {
        throw InvalidInputException("PAC rewrite: only INNER JOINs are supported for PAC compilation");
    }

    // All checks passed: the plan is eligible for PAC rewriting/compilation
    return true;
}

} // namespace duckdb
