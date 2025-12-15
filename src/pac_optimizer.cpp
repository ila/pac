//
// Created by ila on 12/12/25.
//

#include "include/pac_optimizer.hpp"
#include <fstream>
#include <unordered_set>
#include <string>
#include <algorithm>

// Include public helper to access the configured PAC tables filename and read helper
#include "include/pac_privacy_unit.hpp"

// Include concrete logical operator headers and bound aggregate expression
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"

using namespace duckdb;

namespace duckdb {

static bool IsAllowedAggregate(const std::string &func) {
    static const std::unordered_set<std::string> allowed = {"sum_no_overflow", "sum", "count_star", "count", "avg"};
    std::string lower_func = func;
    std::transform(lower_func.begin(), lower_func.end(), lower_func.begin(), ::tolower);
    return allowed.count(lower_func) > 0;
}

// Merged aggregate analyzer: throws on disallowed aggregates (including nested), and sets
// `has_allowed` to true if an allowed aggregate (sum/count/avg) is encountered.
static void AnalyzeAggregates(const LogicalOperator &op, bool in_aggregate, bool &has_allowed) {
    if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
        if (in_aggregate) {
            // Nested aggregate -> disallowed
            throw ParserException("Query contains disallowed aggregates (only sum, count, avg allowed; no nested aggregates)!");
        }
        auto &aggr = op.Cast<LogicalAggregate>();
        for (auto &expr : aggr.expressions) {
            if (expr && expr->IsAggregate() && expr->type == ExpressionType::BOUND_AGGREGATE) {
                auto &ag = expr->Cast<BoundAggregateExpression>();
                // Disallow DISTINCT aggregates (e.g., COUNT(DISTINCT ...))
                if (ag.aggr_type == AggregateType::DISTINCT) {
                    throw ParserException("Query contains DISTINCT, which is not allowed in PAC-compatible queries!");
                }
                if (!IsAllowedAggregate(ag.function.name)) {
                    throw ParserException("Query contains disallowed aggregates (only sum, count, avg allowed; no nested aggregates)!");
                }
                has_allowed = true;
            }
        }
        // Recurse into children but mark that we are inside an aggregate context
        for (auto &child : op.children) {
            AnalyzeAggregates(*child, true, has_allowed);
        }
        return;
    }
    for (auto &child : op.children) {
        AnalyzeAggregates(*child, in_aggregate, has_allowed);
    }
}

static bool ContainsDisallowedJoin(const LogicalOperator &op) {
    // Regular join: only INNER allowed
    if (op.type == LogicalOperatorType::LOGICAL_JOIN || op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
        auto &join = op.Cast<LogicalJoin>();
        if (join.join_type != JoinType::INNER) {
            return true;
        }
    }

    // Disallow other join-like or set-op logical operator types
    if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN ||
        op.type == LogicalOperatorType::LOGICAL_ANY_JOIN ||
        op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT ||
        op.type == LogicalOperatorType::LOGICAL_POSITIONAL_JOIN ||
        op.type == LogicalOperatorType::LOGICAL_ASOF_JOIN ||
        op.type == LogicalOperatorType::LOGICAL_DEPENDENT_JOIN ||
        op.type == LogicalOperatorType::LOGICAL_UNION ||
        op.type == LogicalOperatorType::LOGICAL_EXCEPT ||
        op.type == LogicalOperatorType::LOGICAL_INTERSECT) {
        return true;
    }

    for (auto &child : op.children) {
        if (ContainsDisallowedJoin(*child)) {
        	return true;
		}
    }
    return false;
}

static bool ContainsWindowFunction(const LogicalOperator &op) {
    if (op.type == LogicalOperatorType::LOGICAL_WINDOW) {
        return true;
    }
    for (auto &child : op.children) {
        if (ContainsWindowFunction(*child)) return true;
    }
    return false;
}

static bool ContainsDistinct(const LogicalOperator &op) {
    if (op.type == LogicalOperatorType::LOGICAL_DISTINCT) {
        return true;
    }
    for (auto &child : op.children) {
        if (ContainsDistinct(*child)) return true;
    }
    return false;
}

static bool ScanOnPACTable(const LogicalOperator &op, const std::unordered_set<std::string> &pac_tables) {
    if (op.type == LogicalOperatorType::LOGICAL_GET) {
        auto &scan = op.Cast<LogicalGet>();
        // Try to obtain the underlying table catalog entry and check its name(s)
        auto table_entry = scan.GetTable();
        if (table_entry) {
            auto &tentry = *table_entry;
            if (pac_tables.count(tentry.name) > 0) {
                return true;
            }
        } else {
            // fallback: check names vector if present
            for (auto &n : scan.names) {
                if (pac_tables.count(n) > 0) return true;
            }
        }
    }
    for (auto &child : op.children) {
        if (ScanOnPACTable(*child, pac_tables)) return true;
    }
    return false;
}

void PACRewriteRule::IsPACCompatible(LogicalOperator &plan, ClientContext &context) {
	string pac_privacy_file = "pac_tables.csv";
	Value pac_privacy_file_value;
	context.TryGetCurrentSetting("pac_privacy_file", pac_privacy_file_value);
	if (!pac_privacy_file_value.IsNull()) {
		// by default, the ivm files path is the database path
		// however this can be overridden by a setting
		pac_privacy_file = pac_privacy_file_value.ToString();
	}
    // 1. Read PAC tables using the configured filename (default) and also merge test file if present
    auto pac_tables = ReadPacTablesFile(pac_privacy_file);
    // 2. Must scan a PAC table
    if (!ScanOnPACTable(plan, pac_tables)) {
	    throw ParserException("Query does not scan any PAC table!");
    }
	// 3. Must not contain window functions (a WINDOW node can contain aggregates inside)
	if (ContainsWindowFunction(plan)) {
		throw ParserException("Query contains window functions, which are not allowed in PAC-compatible queries!");
	}
    // 4. Aggregates validation: detect disallowed aggregates first (throws), and ensure at least one allowed aggregate exists
    bool has_allowed_aggregate = false;
    AnalyzeAggregates(plan, false, has_allowed_aggregate);
    if (!has_allowed_aggregate) {
        throw ParserException("Query does not contain any allowed aggregation (sum, count, avg)!");
    }
    // 5. Must not contain DISTINCT
    if (ContainsDistinct(plan)) {
	    throw ParserException("Query contains DISTINCT, which is not allowed in PAC-compatible queries!");
	}
    // 6. Must not contain disallowed joins (only INNER JOIN allowed)
    if (ContainsDisallowedJoin(plan)) {
	    throw ParserException("Query contains disallowed joins (only INNER JOIN allowed in PAC-compatible queries)!");
    }
    // Passed all checks
}

void PACRewriteRule::PACRewriteRuleFunction(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {

	// Run the PAC compatibility checks only if the plan is a projection (i.e., a SELECT query)
	if (!plan || plan->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return;
	}
    // Will throw ParserException with a specific message if not compatible
    IsPACCompatible(*plan, input.context);
    // PAC compatible: do nothing for now
	bool apply_noise = false;
	Value pac_noise_value;
	input.context.TryGetCurrentSetting("pac_noise", pac_noise_value);
	if (!pac_noise_value.IsNull() && pac_noise_value.GetValue<bool>()) {
		apply_noise = true;
	}
	if (apply_noise) {
		// todo
	}
}

} // namespace duckdb
