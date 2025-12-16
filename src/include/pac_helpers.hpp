// filepath: /home/ila/Code/pac/src/pac_helpers.hpp
#pragma once

#include "duckdb.hpp"
#include "duckdb/planner/logical_operator.hpp"

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
DUCKDB_API void UpdateParent(unique_ptr<LogicalOperator> &root,
                             unique_ptr<LogicalOperator> &child_ref,
                             unique_ptr<LogicalOperator> new_parent);

} // namespace duckdb
