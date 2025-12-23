//
// Created by ila on 12/21/25.
//

#ifndef PAC_BITSLICE_COMPILER_HPP
#define PAC_BITSLICE_COMPILER_HPP

#include "duckdb.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "pac_compatibility_check.hpp"

namespace duckdb {

// Bitslice-style PAC compiler entrypoint (currently a stub)
void CompilePacBitsliceQuery(const PACCompatibilityResult &check, OptimizerExtensionInput &input,
                             unique_ptr<LogicalOperator> &plan, const std::string &privacy_unit,
                             const std::string &query, const std::string &query_hash);

} // namespace duckdb

#endif // PAC_BITSLICE_COMPILER_HPP
