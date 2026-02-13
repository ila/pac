// filepath: /home/ila/Code/pac/src/pac_helpers.hpp
#pragma once

#include "duckdb.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include <utility>

namespace duckdb {

class Binder;

// Sanitize a string to be used as a PAC privacy unit or table name (alphanumeric + underscores only)
string Sanitize(const string &in);

// Normalize a query string by collapsing whitespace and lower-casing. Returns the normalized string.
// (Hashing is provided by HashStringToHex below.)
string NormalizeQueryForHash(const string &query);

// Compute a hex string of the std::hash of the given input string
string HashStringToHex(const string &input);

// Determine the next available table index by scanning existing logical operators in the plan
idx_t GetNextTableIndex(unique_ptr<LogicalOperator> &plan);

// Update the plan tree by inserting `new_parent` as the parent of `child_ref`.
// - `root` is the top of the logical plan (used to run a ColumnBindingReplacer so bindings
//   above the inserted node are updated).
// - `child_ref` is a reference to the unique_ptr holding the child in its parent->children vector.
//   The function will consume `new_parent` and replace `child_ref` with it, making the original
//   child a child of `new_parent`.
// - `new_parent` must not have any children prior to calling this function.
// The function will try to propagate column binding changes using ColumnBindingReplacer so that
// expressions above the insertion point keep referring to the correct ColumnBindings.
void ReplaceNode(unique_ptr<LogicalOperator> &root, unique_ptr<LogicalOperator> &old_node,
                 unique_ptr<LogicalOperator> &new_node, Binder *binder = nullptr);

// Find the primary key column names for the given table (searching the client's catalog search path).
// Returns a vector with the primary key column names in order; empty vector if there is no PK.
vector<string> FindPrimaryKey(ClientContext &context, const string &table_name);

// Find foreign keys declared on the given table (searching the client's catalog search path).
// Returns a vector of pairs: (referenced_table_name, list_of_fk_column_names) for each FK constraint.
vector<std::pair<string, vector<string>>> FindForeignKeys(ClientContext &context, const string &table_name);

// Find foreign-key path(s) from any of `table_names` to any of `privacy_units`.
// Returns a map: start_table (as provided) -> path (vector of qualified table names from start to privacy unit,
// inclusive). If no path exists for a start table, it will not appear in the returned map.
std::unordered_map<string, vector<string>>
FindForeignKeyBetween(ClientContext &context, const vector<string> &privacy_units, const vector<string> &table_names);

// -----------------------------------------------------------------------------
// PAC-specific small helpers
// -----------------------------------------------------------------------------

// RAII guard that sets PACOptimizerInfo::replan_in_progress to true for the lifetime of the guard
// and restores the previous value on destruction. Construct with nullptr to have no effect.
struct PACOptimizerInfo; // forward-declare to avoid including pac_optimizer.hpp here
class ReplanGuard {
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
string GetPacPrivacyFile(ClientContext &context, const string &default_filename = "pac_tables.csv");
string GetPacCompiledPath(ClientContext &context, const string &default_path = ".");
int64_t GetPacM(ClientContext &context, int64_t default_m = 128);
bool IsPacNoiseEnabled(ClientContext &context, bool default_value = true);
string GetPacCompileMethod(ClientContext &context, const string &default_method = "standard");

// Helper to safely retrieve boolean settings with defaults
bool GetBooleanSetting(ClientContext &context, const string &setting_name, bool default_value);

// Helper to convert ReadPacTablesFile's unordered_set into a deterministic vector (sorted)
// so callers don't need to repeat this conversion.
vector<string> PacTablesSetToVector(const std::unordered_set<string> &set);

// Return true if the given ColumnBinding in a logical plan ultimately originates from the
// specified base table name (i.e., from a LogicalGet of that table), false otherwise.
// This resolves bindings via table_index back to the source LogicalGet nodes.
bool ColumnBelongsToTable(LogicalOperator &plan, const string &table_name, const ColumnBinding &binding);

} // namespace duckdb
