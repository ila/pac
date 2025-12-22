//
// Created by ila on 12/19/25.
//

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "pac_aggregate.hpp"

#include <cstdint>
#include <random>
#include <cstring>
#include <type_traits>

#ifndef PAC_SUM_HPP
#define PAC_SUM_HPP

// ============================================================================
// PAC_SUM(hash_key, value)  aggregate function
// ============================================================================
// State: 64 sums, one for each bit position
// Update: for each (key_hash, value), add value to sums[i] if bit i of key_hash is set
// Finalize: compute the PAC-noised sum from the 64 counters
//
// We keep sub-totals[64] in multiple data types, from small to large, and try to handle
// most summing in the smallest possible data-type. Because, we can do the 64 sums with
// auto-vectorization. (Rather than naive FOR(i=0;i<64;i++) IF (keybit[i]) totals[i]+=val
// we rewrite into SIMD-friendly multiplication FOR(i=0;i<64;i++) totals[i]+=keybit[i]*val),
//
// mimicking what DuckDB's SUM() normally does we have the following cases:
// 1) integers: PAC_SUM(key_hash, [U](BIG||SMALL|TINY)INT) -> [U]HUGEINT
//              We keep sub-totals8/16/32/64/128 and sum each value in smallest subtotal that fits.
//				We ensure "things fit" by flushing totalsX into the next wider total every 2^bX
//	 			additions, and only by allowing values to be added into totalsX if they have the
//	 			highest bX bits unset, so overflow cannot happen (b8=3, b16=5, b32=6, b64=8).
//              In combine/finalize, we flush out all totalsX<128 into totals128
//              In Finalize() the noised result is computed from totals128
// 2) floating: PAC_SUM(key_hash, (FLOAT|DOUBLE)) -> DOUBLE
//              similar, but with two levels only (float,double), and 16 additions of |val| < 1M
//              into the float-subtotals. This is a compromise based on some rather rough
//	            numerical analysis. It should be noted that (e.g. due to parallelism) the outcome
//              of even standard DuckDB SUM on floating-point numbers is unstable anyway.
// 3) huge-int: PAC_SUM(key_hash, [U]HUGEINT) -> DOUBLE
//				DuckDB produces DOUBLE outcomes for 128-bits integer sums, so we do as well.
//              This basically uses the DOUBLE methods where the updates perform a cast from hugeint

namespace duckdb {

//#define PAC_SUM_NONCASCADING 1 seems 10x slower on Apple

// This macro controls how we filter the values. Rather than IF (bit_is_set) THEN totals += value
// we rather set value to 0 if !bit_is_set and always do totals += value. This is SIMD-friendly.
//
// We have two ways to set value to 0 if !bit_is_set:
// - MULT: value *= bit_is_set
// - AND:  value &= (bit_is_set - 1)
// In micro-benchmarks on Apple, MULT is 30% faster.
// Note: AND only works for integers, not floating point, so DOUBLE always uses MULT.
#define PAC_FILTER_MULT(val, tpe, key, pos) ((val) * static_cast<tpe>(((key) >> (pos)) & 1ULL))
#define PAC_FILTER_AND(val, tpe, key, pos) ((val) & (static_cast<tpe>(((key) >> (pos)) & 1ULL) - 1ULL))

// INT filter is configurable (currently MULT, can switch to AND for benchmarking)
#define PAC_FILTER_INT PAC_FILTER_MULT
// DOUBLE filter must always be MULT (AND doesn't work for floating point)
#define PAC_FILTER_DOUBLE PAC_FILTER_MULT

// Filter kind selector for AddToTotals template
enum class PacFilter { INT, DOUBLE };

// Inner AUTOVECTORIZE function for the 64-element loops
// This is the hot path that benefits from SIMD
// Unified template for adding to totals at any accumulator width
template <PacFilter KIND, typename ACCUM_T, typename VALUE_T>
AUTOVECTORIZE static inline void AddToTotals(ACCUM_T *totals, VALUE_T value, uint64_t key_hash) {
	ACCUM_T v = static_cast<ACCUM_T>(value);
	for (int j = 0; j < 64; j++) {
		if constexpr (KIND == PacFilter::DOUBLE) {
			totals[j] += PAC_FILTER_DOUBLE(v, ACCUM_T, key_hash, j);
		} else {
			totals[j] += PAC_FILTER_INT(v, ACCUM_T, key_hash, j);
		}
	}
}

// =========================
// Integer pac_sum (cascaded multi-level accumulation for SIMD efficiency)
// =========================

// Number of top bits bX for counters of uintX reserved for overflow headroom at each level
static constexpr int kTopBits8 = 3;
static constexpr int kTopBits16 = 4;
static constexpr int kTopBits32 = 5;
static constexpr int kTopBits64 = 8;

// Flush threshold = 2^bX (signed types use half the threshold) - how many times can we add without overflow?
#define FLUSH_THRESHOLD_SIGNED(X)   (1 << (kTopBits##X - 1))
#define FLUSH_THRESHOLD_UNSIGNED(X) (1 << kTopBits##X)

// Check whether a value is small, i.e. top-bits are 0
#define HAS_TOP_BITS_SET_SIGNED(value, bits) \
	((((value) >= 0 ? static_cast<uint64_t>(value) : static_cast<uint64_t>(-(value))) >> (bits - kTopBits##bits)) != 0)
#define HAS_TOP_BITS_SET_UNSIGNED(value, bits) ((static_cast<uint64_t>(value) >> (bits - kTopBits##bits)) != 0)

// Bool parameter version - compiler optimizes away the ternary
#define HAS_TOP_BITS_SET(value, bits, is_signed) \
	((is_signed) ? HAS_TOP_BITS_SET_SIGNED(value, bits) : HAS_TOP_BITS_SET_UNSIGNED(value, bits))

// Templated integer state - SIGNED selects signed/unsigned types and thresholds
template <bool SIGNED>
struct PacSumIntState {
	using T8 = typename std::conditional<SIGNED, int8_t, uint8_t>::type;
	using T16 = typename std::conditional<SIGNED, int16_t, uint16_t>::type;
	using T32 = typename std::conditional<SIGNED, int32_t, uint32_t>::type;
	using T64 = typename std::conditional<SIGNED, int64_t, uint64_t>::type;

#ifndef PAC_SUM_NONCASCADING
	T8 totals8[64];
	T16 totals16[64];
	T32 totals32[64];
	T64 totals64[64];
#endif
	hugeint_t totals128[64]; // want this array last (smaller totals first) for sequential CPU cache access
#ifndef PAC_SUM_NONCASCADING
	uint32_t count8, count16, count32, count64;

	AUTOVECTORIZE inline void Flush64(bool force) {
		if (force || ++count64 >= (SIGNED ? FLUSH_THRESHOLD_SIGNED(64) : FLUSH_THRESHOLD_UNSIGNED(64))) {
			for (int i = 0; i < 64; i++) {
				totals128[i] += hugeint_t(totals64[i]);
				totals64[i] = 0;
			}
			count64 = 0;
		}
	}
	AUTOVECTORIZE inline void Flush32(bool force) {
		if (force || ++count32 >= (SIGNED ? FLUSH_THRESHOLD_SIGNED(32) : FLUSH_THRESHOLD_UNSIGNED(32))) {
			for (int i = 0; i < 64; i++) {
				totals64[i] += totals32[i];
				totals32[i] = 0;
			}
			count64 += count32;
			count32 = 0;
			Flush64(force);
		}
	}
	AUTOVECTORIZE inline void Flush16(bool force) {
		if (force || ++count16 >= (SIGNED ? FLUSH_THRESHOLD_SIGNED(16) : FLUSH_THRESHOLD_UNSIGNED(16))) {
			for (int i = 0; i < 64; i++) {
				totals32[i] += totals16[i];
				totals16[i] = 0;
			}
			count32 += count16;
			count16 = 0;
			Flush32(force);
		}
	}
	AUTOVECTORIZE inline void Flush8(bool force) {
		if (force || ++count8 >= (SIGNED ? FLUSH_THRESHOLD_SIGNED(8) : FLUSH_THRESHOLD_UNSIGNED(8))) {
			for (int i = 0; i < 64; i++) {
				totals16[i] += totals8[i];
				totals8[i] = 0;
			}
			count16 += count8;
			count8 = 0;
			Flush16(force);
		}
	}
#endif
	bool seen_null;
};

// Double pac_sum (cascaded float32/float64 accumulation)
//
// Cascade constants: values with |value| < kDoubleMaxForFloat32 use float, otherwise double
// We can safely sum kDoubleFlushThreshold float values before flushing to double
static constexpr double kDoubleMaxForFloat32 = 1000000.0;
static constexpr uint32_t kDoubleFlushThreshold = 16;

static inline bool AdditionFitsInFloat(double value) {
	double abs_v = value >= 0 ? value : -value;
	return abs_v >= kDoubleMaxForFloat32;
}

struct PacSumDoubleState {
#ifndef PAC_SUM_NONCASCADING
	float totals32[64];
#endif
	double totals64[64]; // want this array last (smaller totals first) for sequential CPU cache access
#ifndef PAC_SUM_NONCASCADING
	uint32_t count32;

	AUTOVECTORIZE inline void Flush32(bool force) {
		if (force || ++count32 >= kDoubleFlushThreshold) {
			for (int i = 0; i < 64; i++) {
				totals64[i] += static_cast<double>(totals32[i]);
				totals32[i] = 0.0f;
			}
			count32 = 0;
		}
	}
#endif
	bool seen_null;
};

// Register pac_sum functions (helper implemented in pac_sum.cpp)
void RegisterPacSumFunctions(ExtensionLoader &loader);

} // namespace duckdb

#endif // PAC_SUM_HPP
