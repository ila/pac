//
// Created by ila on 12/19/25.
//

#include "duckdb.hpp"
#include "pac_aggregate.hpp"

#ifndef PAC_SUM_HPP
#define PAC_SUM_HPP

namespace duckdb {

void RegisterPacSumFunctions(ExtensionLoader &loader);

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
// mimicking what DuckDB's SUM() normally does, we have the following cases:
// 1) integers: PAC_SUM(key_hash, [U](BIG||SMALL|TINY)INT) -> HUGEINT
//              We keep sub-totals8/16/32/64 and uint128_t totals and sum each value in smallest subtotal that fits.
//				We ensure "things fit" by flushing totalsX into the next wider total every 2^bX additions, and only
//				by allowing values to be added into totalsX if they have the highest bX bits unset, so overflow cannot
//				happen (b8=3, b16=5, b32=6, b64=8).
//              In combine/finalize, we flush out all subtotalsX[] into totals[]
//              In Finalize() the noised result is computed from totals[]
// 2) floating: PAC_SUM(key_hash, (FLOAT|DOUBLE)) -> DOUBLE
//              similar, but with only two levels (float,double), and 16 additions of |val| < 1M into  float-subtotals.
//              This is a compromise between speed and precision based on a rough numerical analysis. Please do note
//              that (e.g. due to parallelism) the outcome of even standard SUM on floating-point numbers is unstable.
// 3) huge-int: PAC_SUM(key_hash, [U]HUGEINT) -> DOUBLE
//				DuckDB produces DOUBLE outcomes for 128-bits integer sums, so we do as well.
//              This basically uses the DOUBLE methods where the updates perform a cast from hugeint

//#define PAC_SUM_NONCASCADING 1 // seems 10x slower on Apple

// Simple version for double/[u]int64_t/hugeint (uses multiplication for conditional add)
// This auto-vectorizes well for 64-bits data-types because key_hash is also 64-bits
template <typename ACCUM_T, typename VALUE_T>
AUTOVECTORIZE static inline void AddToTotalsSimple(ACCUM_T *totals, VALUE_T value, uint64_t key_hash) {
	ACCUM_T v = static_cast<ACCUM_T>(value);
	for (int j = 0; j < 64; j++) {
		totals[j] += v * static_cast<ACCUM_T>((key_hash >> j) & 1ULL);
	}
}
// For the 8-bits, 16-bits and 32-bits PAC_SUM() SIMD-benefits are greatest, but achieving these is harder
//
// We need SWAR (SIMD Within A Register). The idea in case of int8_t:
// - Pack 8 int8_t counters into each uint64_t (totals[8] instead of totals[64])
// - totals[i] holds counters for bit positions i, i+8, i+16, i+24, i+32, i+40, i+48, i+56
// - 8 iterations instead of 64
//
// Note that PAC_COUNT() effectively also used SWAR
//
// PAC_SUM() uses SWAR (SIMD Within A Register) for all integer bit widths
// - BITS=8:  8 values packed per uint64_t, totals[8],  8 iterations
// - BITS=16: 4 values packed per uint64_t, totals[16], 16 iterations
// - BITS=32: 2 values packed per uint64_t, totals[32], 32 iterations

// SWAR accumulation: pack multiple counters into uint64_t registers (for 8/16/32-bit elements)
// SIGNED_T/UNSIGNED_T: types for the packed elements (e.g., int8_t/uint8_t)
// MASK: broadcast mask (one bit per element, e.g., 0x0101010101010101 for 8-bit)
// VALUE_T: input value type
template <typename SIGNED_T, typename UNSIGNED_T, uint64_t MASK, typename VALUE_T>
AUTOVECTORIZE static inline void AddToTotalsSWAR(uint64_t *totals, VALUE_T value, uint64_t key_hash) {
	constexpr int BITS = sizeof(SIGNED_T) * 8;
	// Cast to unsigned type to avoid sign extension, then broadcast to all lanes
	uint64_t val_packed = static_cast<UNSIGNED_T>(static_cast<SIGNED_T>(value)) * MASK;
	for (int i = 0; i < BITS; i++) {
		uint64_t bits = (key_hash >> i) & MASK;
		uint64_t expanded = (bits << BITS) - bits; // 0x01 -> 0xFF, 0x0001 -> 0xFFFF, etc.
		totals[i] += val_packed & expanded;
	}
}

// =========================
// Integer pac_sum (cascaded multi-level accumulation for SIMD efficiency)
// =========================

// SIGNED is compile-time known, so will be compiled away (signed is then left with 2 comparisons, unsigned with 1)
#define CHECK_BOUNDS_32(val) ((val > (SIGNED ? INT32_MAX : UINT32_MAX)) || (SIGNED && (val < INT32_MIN)))
#define CHECK_BOUNDS_16(val) ((val > (SIGNED ? INT16_MAX : UINT16_MAX)) || (SIGNED && (val < INT16_MIN)))
#define CHECK_BOUNDS_8(val)  ((val > (SIGNED ? INT8_MAX : UINT8_MAX)) || (SIGNED && (val < INT8_MIN)))

// Templated integer state - SIGNED selects signed/unsigned types and thresholds
template <bool SIGNED>
struct PacSumIntState {
	// Type aliases based on signedness
	typedef typename std::conditional<SIGNED, int8_t, uint8_t>::type T8;
	typedef typename std::conditional<SIGNED, int16_t, uint16_t>::type T16;
	typedef typename std::conditional<SIGNED, int32_t, uint32_t>::type T32;
	typedef typename std::conditional<SIGNED, int64_t, uint64_t>::type T64;

#ifndef PAC_SUM_NONCASCADING
	// All levels use SWAR packed format: uint64_t arrays holding packed values
	// subtotals_X[i] holds counters for bits i, i+X, i+2*X, ... (interleaved layout)
	uint64_t subtotals8[8];   // 8 x uint64_t, each holds 8 packed T8
	uint64_t subtotals16[16]; // 16 x uint64_t, each holds 4 packed T16
	uint64_t subtotals32[32]; // 32 x uint64_t, each holds 2 packed T32
	uint64_t subtotals64[64]; // 64 x uint64_t, each holds 1 T64
#endif
	hugeint_t probabilistic_totals[64]; // final totals, want last for sequential cache access
#ifndef PAC_SUM_NONCASCADING
	// these hold the exact subtotal of each aggregation level, we flush once we see this overflow
	// Use T64 for exact_subtotal64 to handle unsigned values correctly
	int64_t exact_subtotal8, exact_subtotal16, exact_subtotal32;
	T64 exact_subtotal64;

	AUTOVECTORIZE inline void Flush64(T64 value, bool force = false) {
		T64 new_total = value + exact_subtotal64;
		bool would_overflow = ((SIGNED && value < 0) ? (new_total > exact_subtotal64) : (new_total < exact_subtotal64));
		if (would_overflow || force) {
			const T64 *src = reinterpret_cast<const T64 *>(subtotals64);
			for (int i = 0; i < 64; i++) {
				probabilistic_totals[i] += Hugeint::Convert(src[i]); // Use Convert for correct uint64_t handling
			}
			memset(subtotals64, 0, sizeof(subtotals64));
			exact_subtotal64 = value;
		} else {
			exact_subtotal64 = new_total;
		}
	}
	AUTOVECTORIZE inline void Flush32(int64_t value, bool force = false) {
		int64_t new_total = value + exact_subtotal32;
		bool would_overflow = CHECK_BOUNDS_32(new_total);
		if (would_overflow || force) {
			const T32 *src = reinterpret_cast<const T32 *>(subtotals32);
			T64 *dst = reinterpret_cast<T64 *>(subtotals64);
			for (int bit = 0; bit < 64; bit++) { // convert between SWAR orders
				dst[bit] += src[(bit % 32) * 2 + (bit / 32)];
			}
			memset(subtotals32, 0, sizeof(subtotals32));
			Flush64(exact_subtotal32, force);
			exact_subtotal32 = value;
		} else {
			exact_subtotal32 = new_total;
		}
	}
	AUTOVECTORIZE inline void Flush16(int64_t value, bool force = false) {
		int64_t new_total = value + exact_subtotal16;
		bool would_overflow = CHECK_BOUNDS_16(new_total);
		if (would_overflow || force) {
			const T16 *src = reinterpret_cast<const T16 *>(subtotals16);
			T32 *dst = reinterpret_cast<T32 *>(subtotals32);
			for (int bit = 0; bit < 64; bit++) { // convert between SWAR orders
				int dst_idx = (bit % 32) * 2 + (bit / 32);
				int src_idx = (bit % 16) * 4 + (bit / 16);
				dst[dst_idx] += src[src_idx];
			}
			memset(subtotals16, 0, sizeof(subtotals16));
			Flush32(exact_subtotal16, force);
			exact_subtotal16 = value;
		} else {
			exact_subtotal16 = new_total;
		}
	}
	AUTOVECTORIZE inline void Flush8(int64_t value, bool force = false) {
		int64_t new_total = value + exact_subtotal8;
		bool would_overflow = CHECK_BOUNDS_8(new_total);
		if (would_overflow || force) {
			const T8 *src = reinterpret_cast<const T8 *>(subtotals8);
			T16 *dst = reinterpret_cast<T16 *>(subtotals16);
			for (int bit = 0; bit < 64; bit++) { // convert between SWAR orders
				int dst_idx = (bit % 16) * 4 + (bit / 16);
				int src_idx = (bit % 8) * 8 + (bit / 8);
				dst[dst_idx] += src[src_idx];
			}
			memset(subtotals8, 0, sizeof(subtotals8));
			Flush16(exact_subtotal8, force);
			exact_subtotal8 = value;
		} else {
			exact_subtotal8 = new_total;
		}
	}
	void Flush() {
		Flush8(0ULL, true);
	}
#endif
	bool seen_null;
};

// Double pac_sum (cascaded float32/float64 accumulation)
//
// Cascade constants: values with |value| < MaxIncrementFloat32 use float, otherwise double
// We can safely sum MinIncrementsFloat32 float values before flushing to double
static constexpr double MaxIncrementFloat32 = 1000000.0;
static constexpr double MinIncrementsFloat32 = 16;

static inline bool FloatSubtotalFitsDouble(double value, double num = 1) {
	return (value > -MaxIncrementFloat32 * num) && (value < MaxIncrementFloat32 * num);
}

struct PacSumDoubleState {
#ifndef PAC_SUM_NONCASCADING
	float probabilistic_subtotals[64];
#endif
	double probabilistic_totals[64]; // want this array last for sequential CPU cache access: smaller subtotals first
#ifndef PAC_SUM_NONCASCADING
	double exact_subtotal;

	AUTOVECTORIZE inline void Flush32(double value, bool force = false) {
		double raw_subtotal = exact_subtotal + value;
		bool would_overflow = FloatSubtotalFitsDouble(raw_subtotal, MinIncrementsFloat32);
		if (would_overflow || force) {
			for (int i = 0; i < 64; i++) {
				probabilistic_totals[i] += static_cast<double>(probabilistic_subtotals[i]);
			}
			memset(probabilistic_subtotals, 0, sizeof(probabilistic_subtotals));
			exact_subtotal = value;
		} else {
			exact_subtotal = raw_subtotal;
		}
	}
	void Flush() {
		Flush32(0, true);
	}
#endif
	bool seen_null;
};

} // namespace duckdb

#endif // PAC_SUM_HPP
