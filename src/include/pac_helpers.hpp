// filepath: /home/ila/Code/pac/src/pac_helpers.hpp
#pragma once

#include "duckdb.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include <utility>

namespace duckdb {

// Sanitize a string to be used as a PAC privacy unit or table name (alphanumeric + underscores only)
DUCKDB_API std::string Sanitize(const std::string &in);

// Normalize a query string by collapsing whitespace and lower-casing. Returns the normalized string.
// (Hashing is provided by HashStringToHex below.)
DUCKDB_API std::string NormalizeQueryForHash(const std::string &query);

// Compute a hex string of the std::hash of the given input string
DUCKDB_API std::string HashStringToHex(const std::string &input);

// Determine the next available table index by scanning existing logical operators in the plan
DUCKDB_API idx_t GetNextTableIndex(unique_ptr<LogicalOperator> &plan);

// Update the plan tree by inserting `new_parent` as the parent of `child_ref`.
// - `root` is the top of the logical plan (used to run a ColumnBindingReplacer so bindings
//   above the inserted node are updated).
// - `child_ref` is a reference to the unique_ptr holding the child in its parent->children vector.
//   The function will consume `new_parent` and replace `child_ref` with it, making the original
//   child a child of `new_parent`.
// - `new_parent` must not have any children prior to calling this function.
// The function will try to propagate column binding changes using ColumnBindingReplacer so that
// expressions above the insertion point keep referring to the correct ColumnBindings.
DUCKDB_API void ReplaceNode(unique_ptr<LogicalOperator> &root,
                            unique_ptr<LogicalOperator> &old_node,
                            unique_ptr<LogicalOperator> &new_node);

// Find the primary key column names for the given table (searching the client's catalog search path).
// Returns a vector with the primary key column names in order; empty vector if there is no PK.
DUCKDB_API vector<std::string> FindPrimaryKey(ClientContext &context, const std::string &table_name);

// Find foreign keys declared on the given table (searching the client's catalog search path).
// Returns a vector of pairs: (referenced_table_name, list_of_fk_column_names) for each FK constraint.
DUCKDB_API vector<std::pair<std::string, vector<std::string>>> FindForeignKeys(ClientContext &context, const std::string &table_name);

// Find foreign-key path(s) from any of `table_names` to any of `privacy_units`.
// Returns a map: start_table (as provided) -> path (vector of qualified table names from start to privacy unit, inclusive).
// If no path exists for a start table, it will not appear in the returned map.
DUCKDB_API std::unordered_map<std::string, std::vector<std::string>> FindForeignKeyBetween(
    ClientContext &context, const std::vector<std::string> &privacy_units, const std::vector<std::string> &table_names);

// -----------------------------------------------------------------------------
// PAC-specific small helpers
// -----------------------------------------------------------------------------

// RAII guard that sets PACOptimizerInfo::replan_in_progress to true for the lifetime of the guard
// and restores the previous value on destruction. Construct with nullptr to have no effect.
struct PACOptimizerInfo; // forward-declare to avoid including pac_optimizer.hpp here
DUCKDB_API class ReplanGuard {
public:
    explicit ReplanGuard(PACOptimizerInfo *info);
    ~ReplanGuard();

    ReplanGuard(const ReplanGuard &) = delete;
    ReplanGuard &operator=(const ReplanGuard &) = delete;
private:
    PACOptimizerInfo *info;
    bool prev;
};

// Configuration helpers that read PAC-related settings from the client's context and
// return a canonicalized value or a default when not configured.
DUCKDB_API std::string GetPacPrivacyFile(ClientContext &context, const std::string &default_filename = "pac_tables.csv");
DUCKDB_API std::string GetPacCompiledPath(ClientContext &context, const std::string &default_path = ".");
DUCKDB_API int64_t GetPacM(ClientContext &context, int64_t default_m = 128);
DUCKDB_API bool IsPacNoiseEnabled(ClientContext &context, bool default_value = true);

// Helper to convert ReadPacTablesFile's unordered_set into a deterministic vector (sorted)
// so callers don't need to repeat this conversion.
DUCKDB_API std::vector<std::string> PacTablesSetToVector(const std::unordered_set<std::string> &set);

} // namespace duckdb
