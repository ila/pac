//
// Created by ila on 12/21/25.
//

#ifndef PAC_BITSLICE_COMPILER_HPP
#define PAC_BITSLICE_COMPILER_HPP

#include "duckdb.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "pac_compatibility_check.hpp"

namespace duckdb {

// forward-declare LogicalGet to avoid including heavy headers in this header
class LogicalGet;

// Helper to ensure PK columns are present in a LogicalGet's column_ids and projection_ids.
void AddPKColumns(LogicalGet &get, const std::vector<std::string> &pks);

// Helper to inspect FK paths and populate lists of GETs present/missing. Defined in pac_bitslice_compiler.cpp
void PopulateGetsFromFKPath(const PACCompatibilityResult &check, std::vector<std::string> &gets_present,
                            std::vector<std::string> &gets_missing, std::string &start_table_out,
                            std::string &target_pu_out);

// Modify plan when the privacy unit table is present in the plan (case a)
void ModifyPlanWithPU(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                      const std::vector<std::string> &pks, bool use_rowid);

// Modify plan when the privacy unit table is NOT present (case b) - partial implementation
// fk_path: ordered vector of table names from start -> ... -> privacy_unit (inclusive)
void ModifyPlanWithoutPU(const PACCompatibilityResult &check, OptimizerExtensionInput &input,
                         unique_ptr<LogicalOperator> &plan, const std::vector<std::string> &gets_missing,
                         const std::vector<std::string> &gets_present, const std::vector<std::string> &fk_path,
                         const std::string &privacy_unit);

// Bitslice-style PAC compiler entrypoint
void CompilePacBitsliceQuery(const PACCompatibilityResult &check, OptimizerExtensionInput &input,
                             unique_ptr<LogicalOperator> &plan, const std::string &privacy_unit,
                             const std::string &query, const std::string &query_hash);

} // namespace duckdb

#endif // PAC_BITSLICE_COMPILER_HPP
