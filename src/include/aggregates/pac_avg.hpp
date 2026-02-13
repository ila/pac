//
// Created by ila on 1/23/26.
//

#ifndef PAC_AVG_HPP
#define PAC_AVG_HPP

#include "duckdb.hpp"
#include "pac_sum.hpp"

namespace duckdb {

// Register pac_avg functions (uses pac_sum infrastructure with DIVIDE_BY_COUNT=true)
void RegisterPacAvgFunctions(ExtensionLoader &loader);
void RegisterPacAvgCountersFunctions(ExtensionLoader &loader);

// pac_avg finalize wrappers (defined in pac_sum.cpp, used by pac_avg registration)
void PacAvgFinalizeDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset);
void PacAvgFinalizeSignedDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset);
void PacAvgFinalizeUnsignedDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset);

// ============================================================================
// Function declarations for pac_sum components used by pac_avg
// (These are defined in pac_sum.cpp but only needed by pac_avg.cpp)
// ============================================================================

// Size and initialize functions - these still use old AggregateFunction signature
idx_t PacSumIntStateSize(const AggregateFunction &);
void PacSumIntInitialize(const AggregateFunction &, data_ptr_t state_p);
idx_t PacSumDoubleStateSize(const AggregateFunction &);
void PacSumDoubleInitialize(const AggregateFunction &, data_ptr_t state_ptr);

// Update functions (simple updates)
void PacSumUpdateTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count);
void PacSumUpdateSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count);
void PacSumUpdateInteger(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count);
void PacSumUpdateBigInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count);
void PacSumUpdateHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count);
void PacSumUpdateUTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count);
void PacSumUpdateUSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count);
void PacSumUpdateUInteger(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count);
void PacSumUpdateUBigInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count);
void PacSumUpdateUHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count);
void PacSumUpdateFloat(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count);
void PacSumUpdateDouble(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count);

// Scatter update functions
void PacSumScatterUpdateTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count);
void PacSumScatterUpdateSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count);
void PacSumScatterUpdateInteger(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count);
void PacSumScatterUpdateBigInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count);
void PacSumScatterUpdateHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count);
void PacSumScatterUpdateUTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count);
void PacSumScatterUpdateUSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count);
void PacSumScatterUpdateUInteger(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count);
void PacSumScatterUpdateUBigInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count);
void PacSumScatterUpdateUHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count);
void PacSumScatterUpdateFloat(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count);
void PacSumScatterUpdateDouble(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count);

#ifndef PAC_EXACTSUM
// HugeInt variants using double state (for approx mode)
void PacSumUpdateHugeIntDouble(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count);
void PacSumScatterUpdateHugeIntDouble(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count);
#endif

// Combine functions
void PacSumCombineSigned(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count);
void PacSumCombineUnsigned(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count);
void PacSumCombineDoubleWrapper(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count);

// Bind function
unique_ptr<FunctionData> PacSumBind(ClientContext &ctx, AggregateFunction &, vector<unique_ptr<Expression>> &args);

} // namespace duckdb

#endif // PAC_AVG_HPP
