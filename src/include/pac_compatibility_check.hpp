#pragma once

#include "duckdb.hpp"
#include <unordered_set>
#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb {

struct PACCompatibilityResult {
    // Map from scanned table name (start) to FK path vector of qualified table names from start to privacy unit
    std::unordered_map<std::string, std::vector<std::string>> fk_paths;
    // Map from privacy unit name to its primary key column names
    std::unordered_map<std::string, std::vector<std::string>> privacy_unit_pks;
    // Whether plan passed basic PAC-eligibility checks (aggregation/join/window/distinct checks)
    bool eligible_for_rewrite = false;
    // List of configured PAC tables that were actually scanned in the plan
    std::vector<std::string> scanned_pac_tables;
};

// Check whether a logical plan is PAC-compatible according to the project's rules.
// The caller should pass in the list of configured PAC table names (read once).
// Returns a PACCompatibilityResult with fk_paths empty when no PAC rewrite is needed.
// If `replan_in_progress` is true the function will return an empty result immediately to avoid recursion.
DUCKDB_API PACCompatibilityResult PACRewriteQueryCheck(LogicalOperator &plan, ClientContext &context, const std::vector<std::string> &pac_tables, bool replan_in_progress = false);

} // namespace duckdb
