//
// Created by ila on 12/19/25.
//

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/string_type.hpp"

#include <cstdint>
#include <random>
#include <cstring>
#include <type_traits>

#ifndef PAC_SUM_HPP
#define PAC_SUM_HPP

// Enable AVX2 vectorization for update functions
#define AUTOVECTORIZE __attribute__((target("avx2")))

namespace duckdb {

// Cascaded float state with float32 and float64 levels
// Buffer: 512 (totals64) + 256 (totals32) + 64 (alignment) = 832 bytes
struct PacSumFloatCascadeState {
	char totals_buf[832];
	double *totals64;
#ifndef PAC_SUM_NONCASCADING
	float *totals32;
	uint32_t count32;
#endif
	bool seen_null;

#ifndef PAC_SUM_NONCASCADING
	// Declare Flush32 here; define it after the kFloatFlushThreshold constant is declared
	void Flush32(bool force);
#endif
};

static idx_t PacSumFloatCascadeStateSize(const AggregateFunction &) {
	return sizeof(PacSumFloatCascadeState);
}
// Define PAC_SUM_NONCASCADING to disable cascading for benchmarking purposes
// #define PAC_SUM_NONCASCADING

#ifndef PAC_SUM_NONCASCADING
// Float cascade constants: values with |value| < kFloatMaxForFloat32 use float, otherwise double
// We can safely sum kFloatFlushThreshold float values before flushing to double
static constexpr double kFloatMaxForFloat32 = 1000000.0;
static constexpr uint32_t kFloatFlushThreshold = 16;

static inline bool FloatNeedsDouble(double value) {
	double abs_v = value >= 0 ? value : -value;
	return abs_v >= kFloatMaxForFloat32;
}

// Define Flush32 implementation now that kFloatFlushThreshold is visible
inline void PacSumFloatCascadeState::Flush32(bool force) {
	if (force || count32 >= kFloatFlushThreshold) {
		for (int i = 0; i < 64; i++) {
			totals64[i] += static_cast<double>(totals32[i]);
			totals32[i] = 0.0f;
		}
		count32 = 0;
	}
}
#endif

// Helper to align a pointer to 64-byte boundary
static inline uintptr_t AlignTo64(uintptr_t ptr) {
	return (ptr + 63) & ~static_cast<uintptr_t>(63);
}

// Inner AUTOVECTORIZE functions for float cascade
#ifndef PAC_SUM_NONCASCADING
AUTOVECTORIZE
static inline void AddToTotals32Float(float *totals, float value, uint64_t key_hash) {
	for (int j = 0; j < 64; j++) {
		totals[j] += value * static_cast<float>((key_hash >> j) & 1ULL);
	}
}
#endif

// Inner AUTOVECTORIZE functions for the 64-element loops
// These are the hot paths that benefit from SIMD
#ifndef PAC_SUM_NONCASCADING
AUTOVECTORIZE
static inline void AddToTotals8Signed(int8_t *totals, int64_t value, uint64_t key_hash) {
	int8_t v = static_cast<int8_t>(value);
	for (int j = 0; j < 64; j++) {
		totals[j] += v * static_cast<int8_t>((key_hash >> j) & 1ULL);
	}
}
AUTOVECTORIZE
static inline void AddToTotals16Signed(int16_t *totals, int64_t value, uint64_t key_hash) {
	int16_t v = static_cast<int16_t>(value);
	for (int j = 0; j < 64; j++) {
		totals[j] += v * static_cast<int16_t>((key_hash >> j) & 1ULL);
	}
}
AUTOVECTORIZE
static inline void AddToTotals32Signed(int32_t *totals, int64_t value, uint64_t key_hash) {
	int32_t v = static_cast<int32_t>(value);
	for (int j = 0; j < 64; j++) {
		totals[j] += v * static_cast<int32_t>((key_hash >> j) & 1ULL);
	}
}
AUTOVECTORIZE
static inline void AddToTotals64Signed(int64_t *totals, int64_t value, uint64_t key_hash) {
	for (int j = 0; j < 64; j++) {
		totals[j] += value * static_cast<int64_t>((key_hash >> j) & 1ULL);
	}
}

AUTOVECTORIZE
static inline void AddToTotals8Unsigned(uint8_t *totals, uint64_t value, uint64_t key_hash) {
	uint8_t v = static_cast<uint8_t>(value);
	for (int j = 0; j < 64; j++) {
		totals[j] += v * static_cast<uint8_t>((key_hash >> j) & 1ULL);
	}
}
AUTOVECTORIZE
static inline void AddToTotals16Unsigned(uint16_t *totals, uint64_t value, uint64_t key_hash) {
	uint16_t v = static_cast<uint16_t>(value);
	for (int j = 0; j < 64; j++) {
		totals[j] += v * static_cast<uint16_t>((key_hash >> j) & 1ULL);
	}
}
AUTOVECTORIZE
static inline void AddToTotals32Unsigned(uint32_t *totals, uint64_t value, uint64_t key_hash) {
	uint32_t v = static_cast<uint32_t>(value);
	for (int j = 0; j < 64; j++) {
		totals[j] += v * static_cast<uint32_t>((key_hash >> j) & 1ULL);
	}
}
AUTOVECTORIZE
static inline void AddToTotals64Unsigned(uint64_t *totals, uint64_t value, uint64_t key_hash) {
	for (int j = 0; j < 64; j++) {
		totals[j] += value * static_cast<uint64_t>((key_hash >> j) & 1ULL);
	}
}
#endif

// Direct to hugeint_t - used when PAC_SUM_NONCASCADING is defined, or for values too large for int64
#ifdef PAC_SUM_NONCASCADING
static inline void AddToTotals128Signed(hugeint_t *totals, int64_t value, uint64_t key_hash) {
	hugeint_t v(value);
	for (int j = 0; j < 64; j++) {
		totals[j] += v * hugeint_t((key_hash >> j) & 1ULL);
	}
}

static inline void AddToTotals128Unsigned(hugeint_t *totals, uint64_t value, uint64_t key_hash) {
	hugeint_t v(value);
	for (int j = 0; j < 64; j++) {
		totals[j] += v * hugeint_t((key_hash >> j) & 1ULL);
	}
}
#endif

AUTOVECTORIZE
static inline void AddToTotals64Float(double *totals, double value, uint64_t key_hash) {
	for (int j = 0; j < 64; j++) {
		totals[j] += value * static_cast<double>((key_hash >> j) & 1ULL);
	}
}

// Bind function for pac_sum with optional mi parameter
static unique_ptr<FunctionData> PacSumBind(ClientContext &context, AggregateFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
	double mi = 128.0; // default
	if (arguments.size() >= 3) {
		if (!arguments[2]->IsFoldable()) {
			throw InvalidInputException("pac_sum: mi parameter must be a constant");
		}
		auto mi_val = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
		mi = mi_val.GetValue<double>();
		if (mi <= 0.0) {
			throw InvalidInputException("pac_sum: mi must be > 0");
		}
	}
	return make_uniq<PacBindData>(mi);
}


	// Register pac_sum functions (helper implemented in pac_sum.cpp)
	void RegisterPacSumFunctions(ExtensionLoader &loader);
} // namespace duckdb


#endif // PAC_SUM_HPP
