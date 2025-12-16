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

} // namespace duckdb
