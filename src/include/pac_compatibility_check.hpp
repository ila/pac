#pragma once

#include "duckdb.hpp"
#include <unordered_set>
#include <string>

namespace duckdb {

// Check whether a logical plan is PAC-compatible according to the project's rules
DUCKDB_API bool PACRewriteQueryCheck(LogicalOperator &plan, ClientContext &context);

} // namespace duckdb

