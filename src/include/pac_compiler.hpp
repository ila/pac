//
// Created by ila on 12/12/25.
//

#ifndef PAC_COMPILER_HPP
#define PAC_COMPILER_HPP

#include "duckdb.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"

namespace duckdb {

// Compile PAC-compatible query into intermediate artifacts (entry point)
// privacy_unit: single privacy unit name (must be non-empty to compile)
DUCKDB_API void CompilePACQuery(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                                const std::string &privacy_unit);

// Create the sample CTE and write it to a file (void): filename provided by caller
DUCKDB_API void CreateSampleCTE(ClientContext &context, const std::string &privacy_unit,
                                const std::string &filename, const std::string &query_normalized);

// Modify the logical query plan to join in the per-sample table (pac_sample) and extend
// group-by keys / projections as needed. Implemented as an in-place modification of `plan`.
DUCKDB_API void JoinWithSampleTable(ClientContext &context, unique_ptr<LogicalOperator> &plan);

// Create a pac_sample LogicalGet node for the plan using the next available table index
DUCKDB_API unique_ptr<LogicalGet> CreatePacSampleGetNode(ClientContext &context, unique_ptr<LogicalOperator> &plan,
                                                         const std::string &privacy_table_name);

// Overload: Accept a prebuilt pac_sample LogicalGet (caller can construct it first and pass via move)
DUCKDB_API void JoinWithSampleTable(ClientContext &context, unique_ptr<LogicalOperator> &plan,
                                    unique_ptr<LogicalGet> pac_get);

// Modify aggregate operators to add sample_id as a grouping key where pac_sample participates
DUCKDB_API void ModifyAggregateForSample(ClientContext &context, unique_ptr<LogicalOperator> &plan, idx_t pac_idx);

} // namespace duckdb

#endif //PAC_COMPILER_HPP
