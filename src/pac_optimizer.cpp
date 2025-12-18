//
// Created by ila on 12/12/25.
//

#include "include/pac_optimizer.hpp"
#include <fstream>
#include <unordered_set>
#include <string>
#include <algorithm>
#include <functional>

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

#include <pac_compatibility_check.hpp>

namespace duckdb {

// todo- optimizer rule for DROP TABLE

// Helper: find a privacy unit (PAC table) scanned in the plan. Returns empty string if none.
static std::string FindPrivacyUnitInPlan(const LogicalOperator &plan, const std::unordered_set<std::string> &pac_tables) {
    // count scans
    std::unordered_map<std::string, idx_t> scan_counts;
    std::function<void(const LogicalOperator &)> CountScansRec;
    CountScansRec = [&](const LogicalOperator &op) {
        if (op.type == LogicalOperatorType::LOGICAL_GET) {
            auto &scan = op.Cast<LogicalGet>();
            auto table_entry = scan.GetTable();
            if (table_entry) {
                scan_counts[table_entry->name]++;
            } else {
                for (auto &n : scan.names) scan_counts[n]++;
            }
        }
        for (auto &child : op.children) CountScansRec(*child);
    };
    CountScansRec(plan);
    for (auto &t : pac_tables) {
        auto it = scan_counts.find(t);
        if (it != scan_counts.end() && it->second > 0) return t;
    }
    return std::string();
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

    // Delegate compatibility checks (including detecting PAC table presence and internal sample scans)
    // to PACRewriteQueryCheck. If it returns false, nothing to do.
    if (!PACRewriteQueryCheck(*plan, input.context)) {
        return;
    }

    // After PACRewriteQueryCheck validated the plan is eligible, find the privacy unit name to pass to the compiler.
    auto pac_tables = ReadPacTablesFile(pac_privacy_file);
    std::string privacy_unit = FindPrivacyUnitInPlan(*plan, pac_tables);
    if (privacy_unit.empty()) {
        // Shouldn't happen if PACRewriteQueryCheck passed, but be defensive.
        return;
    }

    bool apply_noise = true;
    Value pac_noise_value;
    input.context.TryGetCurrentSetting("pac_noise", pac_noise_value);
    if (!pac_noise_value.IsNull() && !pac_noise_value.GetValue<bool>()) {
        apply_noise = false;
    }
    if (apply_noise) {
        // PAC compatible: invoke compiler to produce artifacts (e.g., sample CTE)
        // Diagnostics: inform that this query will be compiled by PAC
        Printer::Print("Query requires PAC Compilation");
        CompilePACQuery(input, plan, privacy_unit);
     }
 }

 } // namespace duckdb
