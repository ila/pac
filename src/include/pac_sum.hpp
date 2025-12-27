//
// Created by ila on 12/19/25.
//

#include "duckdb.hpp"
#include "pac_aggregate.hpp"

#ifndef PAC_SUM_HPP
#define PAC_SUM_HPP

namespace duckdb {

void RegisterPacSumFunctions(ExtensionLoader &loader);
void RegisterPacAvgFunctions(ExtensionLoader &loader);

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
// 1) integers: PAC_SUM(key_hash, HUGEINT | [U](BIG||SMALL|TINY)INT) -> HUGEINT
//              We keep sub-totals8/16/32/64 and uint128_t totals and sum each value in smallest subtotal that fits.
//              We ensure "things fit" by flushing totalsX into the next wider total every 2^bX additions, and only
//              by allowing values to be added into totalsX if they have the highest bX bits unset, so overflow cannot
//              happen (b8=3, b16=5, b32=6, b64=8).
//              In combine/finalize, we flush out all subtotalsX[] into totals[]
//              In Finalize() the noised result is computed from totals[]
// 2) floating: PAC_SUM(key_hash, (FLOAT|DOUBLE)) -> DOUBLE
//              Accumulates directly into double[64] totals using AddToTotalsSimple.
// 3) hugeint:  PAC_SUM(key_hash, UHUGEINT) -> DOUBLE
//              DuckDB produces DOUBLE outcomes for unsigned 128-bits integer sums, so we do as well.
//              This basically uses the DOUBLE methods where the updates perform a cast from hugeint
//
// for DECIMAL types, we look at binding time which physical type is used and choose a relevant integer type.
//
// we also implement PAC_AVG(key_hash, value) with the same implementation functions as PAC_SUM(). To keep the
// code simple, we added an exact_counter also to PAC_SUM(). The only difference is that in the Finalize() for
// PAC_AVG() we divide the counter numbers by this exact_counter first.
//
// The cascading counter state can be quite large: ~2KB per aggregate resultr value. In aggregations with very many
// distinct GROUP BY values (and relatively modest sums, therefore), the bigger counters are often not needed.
// Therefore, we allocate counters lazily now: only when say 8-bits counters overflow we allocate the 16-bits counters
// This optimization can reduce the memory footprint by 2-8x, which can help in avoiding spilling.

// for benchmarking/reproducibility purposes, we can disable cascading counters (just sum directly to the largest tyoe)
// and in cascading mode we can still use eager memory allocation.
//#define PAC_SUM_NONCASCADING 1 // seems 10x slower on Apple
//#define PAC_SUM_NONLAZY 1  // Pre-allocate all levels at initialization

// Float cascading: accumulate in float subtotals, periodically flush to double totals
// Only beneficial on x86 which has variable-shift SIMD (vpsrlvq). ARM lacks this and
// showed no benefit from float cascading approaches (SWAR, lookup tables, etc.)
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define PAC_SUM_FLOAT_CASCADING 1
#endif

// Simple version for double/[u]int64_t/hugeint (uses multiplication for conditional add)
// This auto-vectorizes well for 64-bits data-types because key_hash is also 64-bits
template <typename ACCUM_T, typename VALUE_T>
AUTOVECTORIZE static inline void AddToTotalsSimple(ACCUM_T *__restrict__ totals, VALUE_T value, uint64_t key_hash) {
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
AUTOVECTORIZE static inline void AddToTotalsSWAR(uint64_t *__restrict__ totals, VALUE_T value, uint64_t key_hash) {
	constexpr int BITS = sizeof(SIGNED_T) * 8;
	// Cast to unsigned type to avoid sign extension, then broadcast to all lanes
	uint64_t val_packed = static_cast<UNSIGNED_T>(static_cast<SIGNED_T>(value)) * MASK;
	for (int i = 0; i < BITS; i++) {
		uint64_t bits = (key_hash >> i) & MASK;
		uint64_t expanded = (bits << BITS) - bits; // 0x01 -> 0xFF, 0x0001 -> 0xFFFF, etc.
		totals[i] += val_packed & expanded;
	}
}

#ifdef PAC_SUM_FLOAT_CASCADING
// SWAR approach for x86: extract bytes, expand bits to float masks via union, multiply-accumulate
// x86 has variable-shift SIMD (vpsrlvq) so this vectorizes well
AUTOVECTORIZE static inline void AddToTotalsFloat(float *totals, float value, uint64_t key_hash) {
	union {
		uint64_t u64;
		uint8_t u8[8];
	} key = {key_hash};

	for (int byte = 0; byte < 8; byte++) {
		uint32_t b = key.u8[byte];
		float *dst = totals + byte * 8;
		// Use union for proper type punning: bit -> 0x00000000 or 0x3F800000 (1.0f)
		union {
			uint32_t u[8];
			float f[8];
		} masks;
		for (int bit = 0; bit < 8; bit++) {
			masks.u[bit] = ((b >> bit) & 1u) * 0x3F800000u;
		}
		for (int bit = 0; bit < 8; bit++) {
			dst[bit] += value * masks.f[bit];
		}
	}
}
#endif // PAC_SUM_FLOAT_CASCADING

// =========================
// Integer pac_sum (cascaded multi-level accumulation for SIMD efficiency)
// =========================

// SIGNED is compile-time known, so will be compiled away (signed is then left with 2 comparisons, unsigned with 1)
#define CHECK_BOUNDS_32(val) ((val > (SIGNED ? INT32_MAX : UINT32_MAX)) || (SIGNED && (val < INT32_MIN)))
#define CHECK_BOUNDS_16(val) ((val > (SIGNED ? INT16_MAX : UINT16_MAX)) || (SIGNED && (val < INT16_MIN)))
#define CHECK_BOUNDS_8(val)  ((val > (SIGNED ? INT8_MAX : UINT8_MAX)) || (SIGNED && (val < INT8_MIN)))
// 64-bit: detect wrap-around (int64_t can't exceed bounds, so check if addition wrapped)
#define CHECK_BOUNDS_64(new_total, value, exact_total)                                                                 \
	((SIGNED && (value) < 0) ? ((new_total) > (exact_total)) : ((new_total) < (exact_total)))

// Templated integer state - SIGNED selects signed/unsigned types and thresholds
// Uses lazy allocation via DuckDB's ArenaAllocator for memory management.
// Arena handles cleanup automatically when aggregate operation completes.
template <bool SIGNED>
struct PacSumIntState {
	// Type aliases based on signedness
	typedef typename std::conditional<SIGNED, int8_t, uint8_t>::type T8;
	typedef typename std::conditional<SIGNED, int16_t, uint16_t>::type T16;
	typedef typename std::conditional<SIGNED, int32_t, uint32_t>::type T32;
	typedef typename std::conditional<SIGNED, int64_t, uint64_t>::type T64;

#ifndef PAC_SUM_NONCASCADING
	// Pointer to DuckDB's arena allocator (set during first update)
	ArenaAllocator *allocator;

	// All levels lazily allocated via arena allocator (nullptr if not allocated)
	uint64_t *probabilistic_totals8;    // 8 x uint64_t (64 bytes) when allocated, each holds 8 packed T8
	uint64_t *probabilistic_totals16;   // 16 x uint64_t (128 bytes) when allocated, each holds 4 packed T16
	uint64_t *probabilistic_totals32;   // 32 x uint64_t (256 bytes) when allocated, each holds 2 packed T32
	uint64_t *probabilistic_totals64;   // 64 x uint64_t (512 bytes) when allocated, each holds 1 T64
	hugeint_t *probabilistic_totals128; // 64 x hugeint_t (1024 bytes) when allocated
#else
	hugeint_t probabilistic_totals128[64]; // final totals (non-cascading mode only)
#endif
#ifndef PAC_SUM_NONCASCADING
	// these hold the exact subtotal of each aggregation level, we flush once we see this overflow
	T64 exact_total8, exact_total16, exact_total32, exact_total64;
#endif
	uint64_t exact_count; // total count of values added (for pac_avg)
	bool seen_null;

#ifdef PAC_SUM_NONCASCADING
	// NONCASCADING: dummy methods for uniform interface
	void Flush() {
	} // no-op
	void GetTotalsAsDouble(double *dst) const {
		ToDoubleArray(probabilistic_totals128, dst);
	}
#else
	// Lazily allocate a level's buffer if not yet allocated
	// BUF_T: buffer element type (uint64_t for SWAR levels, hugeint_t for level 128)
	// Returns 0 if allocated, otherwise returns exact_total unchanged
	template <typename BUF_T, typename EXACT_T = int>
	inline EXACT_T EnsureLevelAllocated(BUF_T *&buffer, idx_t count, EXACT_T exact_total = 0) {
		if (!buffer) {
			buffer = reinterpret_cast<BUF_T *>(allocator->Allocate(count * sizeof(BUF_T)));
			memset(buffer, 0, count * sizeof(BUF_T));
			return 0;
		}
		return exact_total;
	}

	// Cascade SWAR-packed counters from one level to the next with proper bit reordering
	// SRC_T/DST_T: element types, SRC_SWAR/DST_SWAR: SWAR widths (8/16/32/64)
	template <typename SRC_T, typename DST_T, int SRC_SWAR, int DST_SWAR>
	static inline void CascadeToNextLevel(const uint64_t *src_buf, uint64_t *dst_buf) {
		const SRC_T *src = reinterpret_cast<const SRC_T *>(src_buf);
		DST_T *dst = reinterpret_cast<DST_T *>(dst_buf);
		constexpr int SRC_PER_U64 = 64 / SRC_SWAR;
		constexpr int DST_PER_U64 = 64 / DST_SWAR;
		for (int bit = 0; bit < 64; bit++) {
			int src_idx = (bit % SRC_SWAR) * SRC_PER_U64 + (bit / SRC_SWAR);
			int dst_idx = (bit % DST_SWAR) * DST_PER_U64 + (bit / DST_SWAR);
			dst[dst_idx] += src[src_idx];
		}
	}

	// Helper to cascade level 64 to 128 (no SWAR reorder, just convert to hugeint)
	inline void Cascade64To128() {
		const T64 *src = reinterpret_cast<const T64 *>(probabilistic_totals64);
		for (int i = 0; i < 64; i++) {
			probabilistic_totals128[i] += Hugeint::Convert(src[i]);
		}
		memset(probabilistic_totals64, 0, 64 * sizeof(uint64_t));
	}

	// Flush64: cascade probabilistic_totals64 to probabilistic_totals128
	AUTOVECTORIZE inline void Flush64(T64 value, bool force) {
		D_ASSERT(probabilistic_totals64);
		T64 new_total = value + exact_total64;
		bool would_overflow = CHECK_BOUNDS_64(new_total, value, exact_total64);
		if (would_overflow || (force && probabilistic_totals128)) {
			if (would_overflow) {
				EnsureLevelAllocated(probabilistic_totals128, idx_t(64));
			}
			Cascade64To128();
			exact_total64 = value;
		} else {
			exact_total64 = new_total;
		}
	}

	// Flush32: cascade probabilistic_totals32 to probabilistic_totals64
	AUTOVECTORIZE inline void Flush32(int64_t value, bool force) {
		D_ASSERT(probabilistic_totals32);
		int64_t new_total = value + exact_total32;
		bool would_overflow = CHECK_BOUNDS_32(new_total);
		if (would_overflow || (force && probabilistic_totals64)) {
			if (would_overflow) {
				exact_total64 = EnsureLevelAllocated(probabilistic_totals64, 64, exact_total64);
			}
			CascadeToNextLevel<T32, T64, 32, 64>(probabilistic_totals32, probabilistic_totals64);
			memset(probabilistic_totals32, 0, 32 * sizeof(uint64_t));
			Flush64(exact_total32, force);
			exact_total32 = value;
		} else {
			exact_total32 = new_total;
		}
	}

	// Flush16: cascade probabilistic_totals16 to probabilistic_totals32
	AUTOVECTORIZE inline void Flush16(int64_t value, bool force) {
		D_ASSERT(probabilistic_totals16);
		int64_t new_total = value + exact_total16;
		bool would_overflow = CHECK_BOUNDS_16(new_total);
		if (would_overflow || (force && probabilistic_totals32)) {
			if (would_overflow) {
				exact_total32 = EnsureLevelAllocated(probabilistic_totals32, 32, exact_total32);
			}
			CascadeToNextLevel<T16, T32, 16, 32>(probabilistic_totals16, probabilistic_totals32);
			memset(probabilistic_totals16, 0, 16 * sizeof(uint64_t));
			Flush32(exact_total16, force);
			exact_total16 = value;
		} else {
			exact_total16 = new_total;
		}
	}

	// Flush8: cascade probabilistic_totals8 to probabilistic_totals16
	AUTOVECTORIZE inline void Flush8(int64_t value, bool force) {
		D_ASSERT(probabilistic_totals8);
		int64_t new_total = value + exact_total8;
		bool would_overflow = CHECK_BOUNDS_8(new_total);
		if (would_overflow || (force && probabilistic_totals16)) {
			if (would_overflow) {
				exact_total16 = EnsureLevelAllocated(probabilistic_totals16, 16, exact_total16);
			}
			CascadeToNextLevel<T8, T16, 8, 16>(probabilistic_totals8, probabilistic_totals16);
			memset(probabilistic_totals8, 0, 8 * sizeof(uint64_t));
			Flush16(exact_total8, force);
			exact_total8 = value;
		} else {
			exact_total8 = new_total;
		}
	}

	void Flush() {
		if (probabilistic_totals8) {
			Flush8(0LL, true);
		}
	}

	// Convert SWAR packed subtotals to double[64] for finalization
	// These read from the lowest allocated level and unpack to bit-position order
	template <typename SRC_T>
	static void UnpackSWARToDouble(const uint64_t *swar_data, int swar_size, double *dst) {
		const SRC_T *src = reinterpret_cast<const SRC_T *>(swar_data);
		constexpr int elements_per_u64 = sizeof(uint64_t) / sizeof(SRC_T);
		for (int bit = 0; bit < 64; bit++) {
			int src_idx = (bit % swar_size) * elements_per_u64 + (bit / swar_size);
			dst[bit] = static_cast<double>(src[src_idx]);
		}
	}

	// Get the probabilistic totals as doubles, reading from the highest allocated level
	void GetTotalsAsDouble(double *dst) const {
		if (probabilistic_totals128) {
			ToDoubleArray(probabilistic_totals128, dst);
		} else if (probabilistic_totals64) {
			const T64 *src = reinterpret_cast<const T64 *>(probabilistic_totals64);
			for (int i = 0; i < 64; i++) {
				dst[i] = static_cast<double>(src[i]);
			}
		} else if (probabilistic_totals32) {
			UnpackSWARToDouble<T32>(probabilistic_totals32, 32, dst);
		} else if (probabilistic_totals16) {
			UnpackSWARToDouble<T16>(probabilistic_totals16, 16, dst);
		} else if (probabilistic_totals8) {
			UnpackSWARToDouble<T8>(probabilistic_totals8, 8, dst);
		} else {
			// No data allocated - return zeros
			memset(dst, 0, 64 * sizeof(double));
		}
	}

	// Pre-allocate all levels (for NONLAZY mode)
	void InitializeAllLevels(ArenaAllocator &alloc) {
		allocator = &alloc;
		EnsureLevelAllocated(probabilistic_totals8, 8);
		EnsureLevelAllocated(probabilistic_totals16, 16);
		EnsureLevelAllocated(probabilistic_totals32, 32);
		EnsureLevelAllocated(probabilistic_totals64, 64);
		EnsureLevelAllocated(probabilistic_totals128, idx_t(64));
	}
#endif
};

// Double pac_sum state
struct PacSumDoubleState {
#ifdef PAC_SUM_FLOAT_CASCADING
	// Float cascading: accumulate small values in float subtotals, flush to double totals
	float probabilistic_totals_float[64];
#endif
	double probabilistic_totals[64];
#ifdef PAC_SUM_FLOAT_CASCADING
	double exact_total;
#endif
	uint64_t exact_count; // total count of values added (for pac_avg)
	bool seen_null;

	// Cascade constants for float cascading
	static constexpr double MaxIncrementFloat32 = 1000000.0;
	static constexpr double MinIncrementsFloat32 = 16;

	static inline bool FloatSubtotalFitsDouble(double value, double num = 1) {
		return (value > -MaxIncrementFloat32 * num) && (value < MaxIncrementFloat32 * num);
	}

	AUTOVECTORIZE inline void Flush32(double value, bool force = false) {
#ifdef PAC_SUM_FLOAT_CASCADING
		double raw_subtotal = exact_total + value;
		bool would_overflow = FloatSubtotalFitsDouble(raw_subtotal, MinIncrementsFloat32);
		if (would_overflow || force) {
			for (int i = 0; i < 64; i++) {
				probabilistic_totals[i] += static_cast<double>(probabilistic_totals_float[i]);
			}
			memset(probabilistic_totals_float, 0, sizeof(probabilistic_totals_float));
			exact_total = value;
		} else {
			exact_total = raw_subtotal;
		}
#endif
	}

	void Flush() {
#ifdef PAC_SUM_FLOAT_CASCADING
		Flush32(0, true);
#endif
	}

	void GetTotalsAsDouble(double *dst) const {
		for (int i = 0; i < 64; i++) {
			dst[i] = probabilistic_totals[i];
		}
	}
};

} // namespace duckdb

#endif // PAC_SUM_HPP
