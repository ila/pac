#include "include/pac_compatibility_check.hpp"
#include "include/pac_privacy_unit.hpp"

#include "duckdb.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"

#include <algorithm>

using namespace duckdb;

namespace duckdb {

static bool IsAllowedAggregate(const std::string &func) {
    static const std::unordered_set<std::string> allowed = {"sum", "count", "avg"};
    std::string lower_func = func;
    std::transform(lower_func.begin(), lower_func.end(), lower_func.begin(), ::tolower);
    return allowed.count(lower_func) > 0;
}

static bool ContainsDisallowedAggregate(const LogicalOperator &op, bool in_aggregate = false) {
    if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
        if (in_aggregate) return true;
        auto &aggr = op.Cast<LogicalAggregate>();
        for (auto &expr : aggr.expressions) {
            if (expr && expr->IsAggregate()) {
                auto &func = expr->Cast<BoundAggregateExpression>();
                std::string name = func.function.name;
                if (!IsAllowedAggregate(name)) return true;
            }
        }
        for (auto &child : op.children) {
            if (ContainsDisallowedAggregate(*child, true)) return true;
        }
        return false;
    }
    for (auto &child : op.children) {
        if (ContainsDisallowedAggregate(*child, in_aggregate)) return true;
    }
    return false;
}

static bool ContainsDisallowedJoin(const LogicalOperator &op) {
    if (op.type == LogicalOperatorType::LOGICAL_JOIN) {
        auto &join = op.Cast<LogicalJoin>();
        if (join.join_type != JoinType::INNER) return true;
    } else if (
        op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN ||
        op.type == LogicalOperatorType::LOGICAL_ANY_JOIN ||
        op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT ||
        op.type == LogicalOperatorType::LOGICAL_UNION ||
        op.type == LogicalOperatorType::LOGICAL_EXCEPT ||
        op.type == LogicalOperatorType::LOGICAL_INTERSECT) {
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
    if (op.type == LogicalOperatorType::LOGICAL_DISTINCT) return true;
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

static bool ScanOnPACTable(const LogicalOperator &op, const std::unordered_set<std::string> &pac_tables) {
    if (op.type == LogicalOperatorType::LOGICAL_GET) {
        auto &scan = op.Cast<LogicalGet>();
        auto table_entry = scan.GetTable();
        if (table_entry) {
            if (pac_tables.count(table_entry->name) > 0) return true;
        } else {
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

bool CheckPACCompatibility(LogicalOperator &plan, const std::string &pac_tables_filename) {
    auto pac_tables = ReadPacTablesFile(pac_tables_filename);
    if (!ScanOnPACTable(plan, pac_tables)) return false;
    if (!ContainsAggregation(plan)) return false;
    if (ContainsDisallowedAggregate(plan)) return false;
    if (ContainsWindowFunction(plan)) return false;
    if (ContainsDistinct(plan)) return false;
    if (ContainsDisallowedJoin(plan)) return false;
    return true;
}

} // namespace duckdb

