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
// Include PAC compiler
#include "include/pac_compiler.hpp"

// Include concrete logical operator headers and bound aggregate expression
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"

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
        if (ContainsWindowFunction(*child)) { return true; }
    }
    return false;
}

static bool ContainsDistinct(const LogicalOperator &op) {
    if (op.type == LogicalOperatorType::LOGICAL_DISTINCT) {
        return true;
    }
    for (auto &child : op.children) {
        if (ContainsDistinct(*child)) { return true; }
    }
    return false;
}

// Scan the logical operator tree for referenced PAC tables and return the single privacy-unit name
// if found (empty string if none). Throws ParserException if more than one privacy unit is referenced.
static std::string ScanOnPACTable(const LogicalOperator &op, const std::unordered_set<std::string> &pac_tables) {
    // Recursive traversal that merges child results. If two different non-empty names are found,
    // we throw a ParserException.
    std::string result;
    // Check current node if it's a table scan
    if (op.type == LogicalOperatorType::LOGICAL_GET) {
        auto &scan = op.Cast<LogicalGet>();
        auto table_entry = scan.GetTable();
        if (table_entry) {
            auto &tentry = *table_entry;
            if (pac_tables.count(tentry.name) > 0) {
                result = tentry.name;
            }
        } else {
            for (auto &n : scan.names) {
                if (pac_tables.count(n) > 0) {
                    result = n;
                    break;
                }
            }
        }
    }

    // Merge child results
    for (auto &c : op.children) {
        std::string child_res = ScanOnPACTable(*c, pac_tables);
        if (!child_res.empty()) {
            if (result.empty()) {
                result = std::move(child_res);
            } else if (result != child_res) {
                throw ParserException("PAC compilation: queries referencing more than one privacy unit are not supported");
            }
        }
    }
    return result;
}

void PACRewriteRule::IsPACCompatible(LogicalOperator &plan, ClientContext &context) {
    // Assumes the plan references exactly one PAC table (the scan and multi-table check are done in the caller).
    // 1. Must not contain window functions (a WINDOW node can contain aggregates inside)
    if (ContainsWindowFunction(plan)) {
        throw ParserException("Query contains window functions, which are not allowed in PAC-compatible queries!");
    }
    // 2. Aggregates validation: detect disallowed aggregates first (throws), and ensure at least one allowed aggregate exists
    bool has_allowed_aggregate = false;
    AnalyzeAggregates(plan, false, has_allowed_aggregate);
    if (!has_allowed_aggregate) {
        throw ParserException("Query does not contain any allowed aggregation (sum, count, avg)!");
    }
    // 3. Must not contain DISTINCT
    if (ContainsDistinct(plan)) {
        throw ParserException("Query contains DISTINCT, which is not allowed in PAC-compatible queries!");
    }
    // 4. Must not contain disallowed joins (only INNER JOIN allowed)
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
    // Load configured PAC tables once
    string pac_privacy_file = "pac_tables.csv";
    Value pac_privacy_file_value;
    input.context.TryGetCurrentSetting("pac_privacy_file", pac_privacy_file_value);
    if (!pac_privacy_file_value.IsNull()) {
        pac_privacy_file = pac_privacy_file_value.ToString();
    }
    auto pac_tables = ReadPacTablesFile(pac_privacy_file);

    // Scan the plan for referenced PAC tables; ScanOnPACTable returns the single privacy unit name or
    // an empty string if none, and throws if more than one is referenced.
    std::string privacy_unit = ScanOnPACTable(*plan, pac_tables);
    if (privacy_unit.empty()) {
        // no PAC tables referenced -> nothing to do
        return;
    }

    // Exactly one privacy unit referenced. Continue with compatibility checks
    IsPACCompatible(*plan, input.context);
    bool apply_noise = true;
    Value pac_noise_value;
    input.context.TryGetCurrentSetting("pac_noise", pac_noise_value);
    if (!pac_noise_value.IsNull() && pac_noise_value.GetValue<bool>()) {
        apply_noise = true;
    }
    if (apply_noise) {
    	// PAC compatible: invoke compiler to produce artifacts (e.g., sample CTE)
    	CompilePACQuery(input, plan, privacy_unit);
    }
}

} // namespace duckdb
