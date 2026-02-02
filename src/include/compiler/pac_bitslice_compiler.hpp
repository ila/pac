//
// Created by ila on 12/21/25.
//

#ifndef PAC_BITSLICE_COMPILER_HPP
#define PAC_BITSLICE_COMPILER_HPP

#include "duckdb.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "metadata/pac_compatibility_check.hpp"

namespace duckdb {

// forward-declare LogicalGet to avoid including heavy headers in this header
class LogicalGet;

// Helper to ensure PK columns are present in a LogicalGet's column_ids and projection_ids.
void AddPKColumns(LogicalGet &get, const vector<string> &pks);

// Helper to inspect FK paths and populate lists of GETs present/missing. Defined in pac_bitslice_compiler.cpp
void PopulateGetsFromFKPath(const PACCompatibilityResult &check, vector<string> &gets_present,
                            vector<string> &gets_missing, string &start_table_out, vector<string> &target_pus_out);

// Modify plan when the privacy unit table is present in the plan (case a)
void ModifyPlanWithPU(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                      const vector<string> &pu_table_names, const PACCompatibilityResult &check);

// Modify plan when the privacy unit table is NOT present (case b) - partial implementation
// fk_path: ordered vector of table names from start -> ... -> privacy_unit (inclusive)
void ModifyPlanWithoutPU(const PACCompatibilityResult &check, OptimizerExtensionInput &input,
                         unique_ptr<LogicalOperator> &plan, const vector<string> &gets_missing,
                         const vector<string> &gets_present, const vector<string> &fk_path,
                         const vector<string> &privacy_units);

// Bitslice-style PAC compiler entrypoint
void CompilePacBitsliceQuery(const PACCompatibilityResult &check, OptimizerExtensionInput &input,
                             unique_ptr<LogicalOperator> &plan, const vector<string> &privacy_units,
                             const string &query, const string &query_hash);

} // namespace duckdb

#endif // PAC_BITSLICE_COMPILER_HPP
