#pragma once

#include "duckdb.hpp"
#include "core/pac_optimizer.hpp"
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace duckdb {

// Lightweight metadata about a table discovered during compatibility checking.
// - table_name: unqualified table name
// - pks: primary key column names in order
// - fks: list of foreign-key relationships declared on the table; each pair is
//        (referenced_table_name, vector<fk_column_names_on_this_table>)
struct ColumnMetadata {
	string table_name;
	vector<string> pks;
	vector<pair<string, vector<string>>> fks;
};

struct PACCompatibilityResult {
	// Map from scanned table name (start) to FK/LINK path vector of table names from start to privacy unit
	// This includes paths via both actual FK constraints AND PAC LINK metadata
	std::unordered_map<string, vector<string>> fk_paths;
	// List of tables that have protected columns (these are treated as implicit privacy units)
	vector<string> tables_with_protected_columns;
	// Lightweight per-table metadata (pk/fk) for scanned tables
	std::unordered_map<string, ColumnMetadata> table_metadata;
	// Whether plan passed basic PAC-eligibility checks (aggregation/join/window/distinct checks)
	bool eligible_for_rewrite = false;
	// List of configured PAC tables that were actually scanned in the plan
	vector<string> scanned_pu_tables;
	// List of scanned tables that are NOT configured PAC tables
	vector<string> scanned_non_pu_tables;
	// Per-table set of protected column names (lowercased).
	// Union of: PU PKs, LINK FK columns reaching a PU, metadata PROTECTED columns.
	std::unordered_map<string, std::unordered_set<string>> protected_columns;
};

// Check whether a logical plan is PAC-compatible according to the project's rules.
// The caller should pass in the list of configured PAC table names (read once).
// Returns a PACCompatibilityResult with fk_paths empty when no PAC rewrite is needed.
// If `replan_in_progress` is true the function will return an empty result immediately to avoid recursion.
PACCompatibilityResult PACRewriteQueryCheck(unique_ptr<LogicalOperator> &plan, ClientContext &context,
                                            const vector<string> &pac_tables,
                                            PACOptimizerInfo *optimizer_info = nullptr);

void CountScans(const LogicalOperator &op, std::unordered_map<string, idx_t> &counts);
} // namespace duckdb
