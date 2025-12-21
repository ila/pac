//
// Created by ila on 12/19/25.
//

#ifndef PAC_COUNT_HPP
#define PAC_COUNT_HPP

#include "duckdb.hpp"
#include <random>

namespace duckdb {

// State for pac_count: 64 counters and totals8 intermediate accumulators
struct PacCountState {
	uint64_t totals8[8];   // SIMD-friendly intermediate accumulators (8 x 8 bytes)
	uint64_t totals64[64]; // Final counters (64 x 8 bytes)
	uint8_t update_count;  // Counts updates, flushes when wraps to 0
	double mi;             // Privacy parameter (default 128.0)

	void Flush();
};

// Mask used by pac_count inner loops
#define PAC_COUNT_MASK  \
   ((1ULL << 0) | (1ULL << 8) | (1ULL << 16) | (1ULL << 24) | (1ULL << 32) | (1ULL << 40) | (1ULL << 48) | (1ULL << 56))

// Declarations of functions implemented in pac_count.cpp
idx_t PacCountStateSize(const AggregateFunction &);
void PacCountInitialize(const AggregateFunction &, data_ptr_t state_ptr);
void PacCountUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, data_ptr_t state_ptr, idx_t count);
void PacCountScatterUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &states, idx_t count);
void PacCountCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count);

// PacCountFinalize is implemented in pac_aggregate.cpp (uses PacNoisySampleFrom64Counters)
void PacCountFinalize(Vector &states, AggregateInputData &aggr_input, Vector &result, idx_t count, idx_t offset);

// Bind function (returns PacBindData) implemented in pac_count.cpp
unique_ptr<FunctionData> PacCountBind(ClientContext &context, AggregateFunction &function, vector<unique_ptr<Expression>> &arguments);

// Register the pac_count aggregate functions with the loader
void RegisterPacCountFunctions(ExtensionLoader &loader);

} // namespace duckdb

#endif // PAC_COUNT_HPP
