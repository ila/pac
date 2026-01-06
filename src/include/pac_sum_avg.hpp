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
// We keep sub-total[64] in multiple data types, from small to large, and try to handle
// most summing in the smallest possible data-type. Because, we can do the 64 sums with
// auto-vectorization. (Rather than naive FOR(i=0;i<64;i++) IF (keybit[i]) total[i]+=val
// we rewrite into SIMD-friendly multiplication FOR(i=0;i<64;i++) total[i]+=keybit[i]*val),
//
// mimicking what DuckDB's SUM() normally does, we have the following cases:
// 1) integers: PAC_SUM(key_hash, HUGEINT | [U](BIG||SMALL|TINY)INT) -> HUGEINT
//              We keep sub-total8/16/32/64/128_t probabilistic_total and sum each value in smallest total that fits.
//              We ensure "things fit" by flushing probabilistic_totalX into the next wider total every 2^bX additions,
//              and by only allowing values to be added into probaiistic_totalX if they have the highest bX bits unset,
//              so overflow cannot happen (b8=3, b16=5, b32=6, b64=8).
//              In Combine() we combine the levels of src and dst states, either by moving them from src to dst (if dst
//              did not have that level yet), or by summing them. In Finalize() the noised result is computed from the
//              sum of all allocated levels.
// 2) floating: PAC_SUM(key_hash, (FLOAT|DOUBLE)) -> DOUBLE
//              Accumulates directly into double[64] total using AddToTotalsSimple (ARM).
//              On x86 we first sum into floats and cascade these into doubles later.
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
// The cascading counter state can be quite large: ~2KB per aggregate result value. In aggregations with very many
// distinct GROUP BY values (and relatively modest sums, therefore), the bigger counters are often not needed.
// Therefore, we allocate counters lazily now: only when say 8-bits counters overflow we allocate the 16-bits counters
// This optimization can reduce the memory footprint by 2-8x, which can help in avoiding spilling.
//
// DuckDB's parallel aggregation partitions per thread and creates small 32K hash-tables, which get abandoned
// when they fill. This means that dynamically allocated aggregate state memory leaks. Typically, it happens
// when every group-by key in the hash table just had one insert for it (or a few). With abandoning the input
// is just partitioned, and reprocessed later. If you generate group-ids with range(100M) / 100 you get group locality
// and do not suffer from this, but with range(100M) % 1M you get very different performance due to this.
//
// We therefore delay any aggregate processing (and thus allocation) until 3-4 (hash,value) pairs have been received.
// These values are buffered in the state memory (it becomes a union because of this dual use).

// for benchmarking/reproducibility purposes, we can disable cascading counters (just sum directly to the largest tyoe)
// and in cascading mode we can still use eager memory allocation.
//#define PAC_SUMAVG_NONCASCADING 1 // seems 10x slower on Apple
//#define PAC_SUMAVG_NONLAZY 1  // Pre-allocate all levels at initialization
// Define PAC_SUMAVG_NOBUFFERING to disable input buffering (aggregation state initialized immediately)
//#define PAC_SUMAVG_NOBUFFERING 1

// NULL handling: by default we ignore NULLs (safe behavior, like DuckDB's SUM/AVG).
// Define PAC_SUMAVG_UNSAFENULL to return NULL if any input value is NULL.
//#define PAC_SUMAVG_UNSAFENULL 1

// Float cascading: accumulate in float subtotal, periodically flush to double total
// Only beneficial on x86 which has variable-shift SIMD (vpsrlvq). ARM lacks this and
// showed no benefit from float cascading approaches (SWAR, lookup tables, etc.)
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define PAC_SUMAVG_FLOAT_CASCADING 1
#endif

// Simple version for double/[u]int64_t/hugeint (uses multiplication for conditional add)
// This auto-vectorizes well for 64-bits data-types because key_hash is also 64-bits
template <typename ACCUM_T, typename VALUE_T>
AUTOVECTORIZE static inline void AddToTotalsSimple(ACCUM_T *__restrict__ total, VALUE_T value, uint64_t key_hash) {
	ACCUM_T v = static_cast<ACCUM_T>(value);
	for (int j = 0; j < 64; j++) {
		total[j] += v * static_cast<ACCUM_T>((key_hash >> j) & 1ULL);
	}
}
// For the 8-bits, 16-bits and 32-bits PAC_SUM() SIMD-benefits are greatest, but achieving these is harder
//
// We need SWAR (SIMD Within A Register). The idea in case of int8_t:
// - Pack 8 int8_t counters into each uint64_t (total[8] instead of total[64])
// - total[i] holds counters for bit positions i, i+8, i+16, i+24, i+32, i+40, i+48, i+56
// - 8 iterations instead of 64
//
// Note that PAC_COUNT() effectively also used SWAR
//
// PAC_SUM() uses SWAR (SIMD Within A Register) for all integer bit widths
// - BITS=8:  8 values packed per uint64_t, total[8],  8 iterations
// - BITS=16: 4 values packed per uint64_t, total[16], 16 iterations
// - BITS=32: 2 values packed per uint64_t, total[32], 32 iterations
//
// prototyped here: https://godbolt.org/z/jr7aKocTW

// SWAR accumulation: pack multiple counters into uint64_t registers (for 8/16/32-bit elements)
// SIGNED_T/UNSIGNED_T: types for the packed elements (e.g., int8_t/uint8_t)
// MASK: broadcast mask (one bit per element, e.g., 0x0101010101010101 for 8-bit)
// VALUE_T: input value type
template <typename SIGNED_T, typename UNSIGNED_T, uint64_t MASK, typename VALUE_T>
AUTOVECTORIZE static inline void AddToTotalsSWAR(uint64_t *__restrict__ total, VALUE_T value, uint64_t key_hash) {
	constexpr int BITS = sizeof(SIGNED_T) * 8;
	// Cast to unsigned type to avoid sign extension, then broadcast to all lanes
	uint64_t val_packed = static_cast<UNSIGNED_T>(static_cast<SIGNED_T>(value)) * MASK;
	for (int i = 0; i < BITS; i++) {
		uint64_t bits = (key_hash >> i) & MASK;
		uint64_t expanded = (bits << BITS) - bits; // 0x01 -> 0xFF, 0x0001 -> 0xFFFF, etc.
		total[i] += val_packed & expanded;
	}
}

#ifdef PAC_SUMAVG_FLOAT_CASCADING
// SWAR approach for x86: extract bytes, expand bits to float masks via union, multiply-accumulate
// x86 has variable-shift SIMD (vpsrlvq) so this vectorizes well
// prototyped here: https://godbolt.org/z/jodKW3er7
AUTOVECTORIZE static inline void AddToTotalsFloat(float *total, float value, uint64_t key_hash) {
	union {
		uint64_t u64;
		uint8_t u8[8];
	} key = {key_hash};

	for (int byte = 0; byte < 8; byte++) {
		uint32_t b = key.u8[byte];
		float *dst = total + byte * 8;
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
#endif // PAC_SUMAVG_FLOAT_CASCADING

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
//
// Input buffering: To avoid memory allocation when DuckDB abandons pre-aggregation
// hash tables (which leaks arena memory), we buffer the first 3 values before
// starting actual aggregation. The buffer uses a union with the aggregation state.
template <bool SIGNED>
struct PacSumIntState {
	// Type aliases based on signedness
	typedef typename std::conditional<SIGNED, int8_t, uint8_t>::type T8;
	typedef typename std::conditional<SIGNED, int16_t, uint16_t>::type T16;
	typedef typename std::conditional<SIGNED, int32_t, uint32_t>::type T32;
	typedef typename std::conditional<SIGNED, int64_t, uint64_t>::type T64;

#ifdef PAC_SUMAVG_UNSAFENULL
	bool seen_null;
#endif
	uint64_t exact_count; // total count of values added (for pac_avg), also buffer count when buffering

#ifdef PAC_SUMAVG_NONCASCADING
	hugeint_t probabilistic_total128[64]; // final total (non-cascading mode only)
#else
	// Input buffering: buffer first 3 values before allocating arrays
	// When buffering == true: buf_hashes/buf_values hold buffered inputs
	// When buffering == false: exact_totalX and probabilistic_totalX hold aggregation state
	static constexpr uint8_t BUFFER_CAPACITY = 3;

	// buffering field is first in both union members to reduce padding and ensure identical layout
	union {
		struct { // Buffering mode (when buffering == true)
			bool buffering;
			uint64_t buf_hashes[BUFFER_CAPACITY];
			T64 buf_values[BUFFER_CAPACITY]; // Actual type T64
		};
		struct {             // Aggregating mode (when buffering == false)
			bool buffering_; // same memory location as buffering above

			// Exact subtotals for each level - sized to fit level's representable range.
			// After any Flush, exact_totalN is either:
			//   - the incoming `value` (which fits in TN since it was routed to level N), or
			//   - `new_total` which we verified fits in TN range before assigning
			// Flush functions use int64_t for intermediate calculations to detect overflow.
			T8 exact_total8;   // sum of values at level 8 since last flush (fits in T8 after flush)
			T16 exact_total16; // sum of values at level 16 since last flush (fits in T16 after flush)
			T32 exact_total32; // sum of values at level 32 since last flush (fits in T32 after flush)
			T64 exact_total64; // sum of values at level 64 since last flush

			// All levels lazily allocated via arena allocator (nullptr if not allocated)
			uint64_t *probabilistic_total8;    // 8 x uint64_t (64 bytes) when allocated, each holds 8 packed T8
			uint64_t *probabilistic_total16;   // 16 x uint64_t (128 bytes) when allocated, each holds 4 packed T16
			uint64_t *probabilistic_total32;   // 32 x uint64_t (256 bytes) when allocated, each holds 2 packed T32
			uint64_t *probabilistic_total64;   // 64 x uint64_t (512 bytes) when allocated, each holds 1 T64
			hugeint_t *probabilistic_total128; // 64 x hugeint_t (1024 bytes) when allocated
		};
	};

	bool IsBuffering() const {
#ifdef PAC_SUMAVG_NOBUFFERING
		return false; // buffering disabled at compile time
#else
		return buffering;
#endif
	}
#endif
#ifdef PAC_SUMAVG_NONCASCADING
	// NONCASCADING: dummy methods for uniform interface
	bool IsBuffering() const {
		return false;
	}
	void InitializeAggregationState() {
	} // no-op - NONCASCADING doesn't buffer
	void Flush(ArenaAllocator &) {
	} // no-op
	void GetTotalsAsDouble(double *dst) const {
		ToDoubleArray(probabilistic_total128, dst);
	}
#else
	// Lazily allocate a level's buffer if not yet allocated
	// BUF_T: buffer element type (uint64_t for SWAR levels, hugeint_t for level 128)
	// Returns 0 if allocated, otherwise returns exact_total unchanged
	template <typename BUF_T, typename EXACT_T = int>
	static inline EXACT_T EnsureLevelAllocated(ArenaAllocator &allocator, BUF_T *&buffer, idx_t count,
	                                           EXACT_T exact_total = 0) {
		if (!buffer) {
			buffer = reinterpret_cast<BUF_T *>(allocator.Allocate(count * sizeof(BUF_T)));
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
		const T64 *src = reinterpret_cast<const T64 *>(probabilistic_total64);
		for (int i = 0; i < 64; i++) {
			probabilistic_total128[i] += Hugeint::Convert(src[i]);
		}
		memset(probabilistic_total64, 0, 64 * sizeof(uint64_t));
	}

	// Flush64: cascade probabilistic_total64 to probabilistic_total128
	AUTOVECTORIZE inline void Flush64(ArenaAllocator &allocator, T64 value, bool force) {
		D_ASSERT(probabilistic_total64);
		T64 new_total = value + exact_total64;
		bool would_overflow = CHECK_BOUNDS_64(new_total, value, exact_total64);
		if (would_overflow || (force && probabilistic_total128)) {
			if (would_overflow) {
				EnsureLevelAllocated(allocator, probabilistic_total128, idx_t(64));
			}
			Cascade64To128();
			exact_total64 = value;
		} else {
			exact_total64 = new_total;
		}
	}

	// Flush32: cascade probabilistic_total32 to probabilistic_total64
	AUTOVECTORIZE inline void Flush32(ArenaAllocator &allocator, int64_t value, bool force) {
		D_ASSERT(probabilistic_total32);
		int64_t new_total = value + exact_total32;
		bool would_overflow = CHECK_BOUNDS_32(new_total);
		if (would_overflow || (force && probabilistic_total64)) {
			if (would_overflow) {
				exact_total64 = EnsureLevelAllocated(allocator, probabilistic_total64, 64, exact_total64);
			}
			// Flush level 64 first (propagates exact_total32, ensures room for cascade)
			Flush64(allocator, exact_total32, force);
			CascadeToNextLevel<T32, T64, 32, 64>(probabilistic_total32, probabilistic_total64);
			memset(probabilistic_total32, 0, 32 * sizeof(uint64_t));
			exact_total32 = value;
		} else {
			exact_total32 = new_total;
		}
	}

	// Flush16: cascade probabilistic_total16 to probabilistic_total32
	AUTOVECTORIZE inline void Flush16(ArenaAllocator &allocator, int64_t value, bool force) {
		D_ASSERT(probabilistic_total16);
		int64_t new_total = value + exact_total16;
		bool would_overflow = CHECK_BOUNDS_16(new_total);
		if (would_overflow || (force && probabilistic_total32)) {
			if (would_overflow) {
				exact_total32 = EnsureLevelAllocated(allocator, probabilistic_total32, 32, exact_total32);
			}
			// Flush level 32 first (propagates exact_total16, ensures room for cascade)
			Flush32(allocator, exact_total16, force);
			CascadeToNextLevel<T16, T32, 16, 32>(probabilistic_total16, probabilistic_total32);
			memset(probabilistic_total16, 0, 16 * sizeof(uint64_t));
			exact_total16 = value;
		} else {
			exact_total16 = new_total;
		}
	}

	// Flush8: cascade probabilistic_total8 to probabilistic_total16
	AUTOVECTORIZE inline void Flush8(ArenaAllocator &allocator, int64_t value, bool force) {
		D_ASSERT(probabilistic_total8);
		int64_t new_total = value + exact_total8;
		bool would_overflow = CHECK_BOUNDS_8(new_total);
		if (would_overflow || (force && probabilistic_total16)) {
			if (would_overflow) {
				exact_total16 = EnsureLevelAllocated(allocator, probabilistic_total16, 16, exact_total16);
			}
			// Flush level 16 first (propagates exact_total8, ensures room for cascade)
			Flush16(allocator, exact_total8, force);
			CascadeToNextLevel<T8, T16, 8, 16>(probabilistic_total8, probabilistic_total16);
			memset(probabilistic_total8, 0, 8 * sizeof(uint64_t));
			exact_total8 = value;
		} else {
			exact_total8 = new_total;
		}
	}

	// Initialize aggregation state (transition from buffering mode)
	// Called when buffer is full or at Combine/Finalize time
	// NOTE: Caller must copy buf_hashes/buf_values to stack BEFORE calling this,
	// as this method overwrites the union with aggregation state fields.
	void InitializeAggregationState() {
		// Zero out union fields first before changing buffering state
		exact_total8 = exact_total16 = exact_total32 = exact_total64 = 0;
		probabilistic_total8 = probabilistic_total16 = nullptr;
		probabilistic_total32 = probabilistic_total64 = nullptr;
		probabilistic_total128 = nullptr;
		// Now transition to aggregation mode
		buffering = false;
	}

	void Flush(ArenaAllocator &allocator) {
		// Start flushing from the lowest allocated level
		// (data may have skipped lower levels if values were always large)
		if (probabilistic_total8) {
			Flush8(allocator, 0LL, true);
		} else if (probabilistic_total16) {
			Flush16(allocator, 0LL, true);
		} else if (probabilistic_total32) {
			Flush32(allocator, 0LL, true);
		} else if (probabilistic_total64) {
			Flush64(allocator, 0LL, true);
		}
	}

	// Convert SWAR packed subtotal to double[64] for finalization
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

	// Get the probabilistic total as doubles, reading from the highest allocated level
	void GetTotalsAsDouble(double *dst) const {
		if (probabilistic_total128) {
			ToDoubleArray(probabilistic_total128, dst);
		} else if (probabilistic_total64) {
			const T64 *src = reinterpret_cast<const T64 *>(probabilistic_total64);
			for (int i = 0; i < 64; i++) {
				dst[i] = static_cast<double>(src[i]);
			}
		} else if (probabilistic_total32) {
			UnpackSWARToDouble<T32>(probabilistic_total32, 32, dst);
		} else if (probabilistic_total16) {
			UnpackSWARToDouble<T16>(probabilistic_total16, 16, dst);
		} else if (probabilistic_total8) {
			UnpackSWARToDouble<T8>(probabilistic_total8, 8, dst);
		} else {
			// No data allocated - return zeros
			memset(dst, 0, 64 * sizeof(double));
		}
	}

	// Pre-allocate all levels (for NONLAZY mode)
	void InitializeAllLevels(ArenaAllocator &allocator) {
		EnsureLevelAllocated(allocator, probabilistic_total8, 8);
		EnsureLevelAllocated(allocator, probabilistic_total16, 16);
		EnsureLevelAllocated(allocator, probabilistic_total32, 32);
		EnsureLevelAllocated(allocator, probabilistic_total64, 64);
		EnsureLevelAllocated(allocator, probabilistic_total128, idx_t(64));
	}
#endif
};

// Double pac_sum state
struct PacSumDoubleState {
	// Field ordering optimized for memory layout
#ifdef PAC_SUMAVG_UNSAFENULL
	bool seen_null;
#endif
#ifdef PAC_SUMAVG_FLOAT_CASCADING
	double exact_total;
#endif
	uint64_t exact_count; // total count of values added (for pac_avg)
#ifdef PAC_SUMAVG_FLOAT_CASCADING
	// Float cascading: accumulate small values in float subtotal, flush to double total
	float probabilistic_total_float[64];
#endif
	double probabilistic_total[64];

	// Cascade constants for float cascading
	static constexpr double MaxIncrementFloat32 = 1000000.0;
	static constexpr double MinIncrementsFloat32 = 16;

	static inline bool FloatSubtotalFitsDouble(double value, double num = 1) {
		return (value > -MaxIncrementFloat32 * num) && (value < MaxIncrementFloat32 * num);
	}

	AUTOVECTORIZE inline void Flush32(double value, bool force = false) {
#ifdef PAC_SUMAVG_FLOAT_CASCADING
		double raw_subtotal = exact_total + value;
		bool would_overflow = FloatSubtotalFitsDouble(raw_subtotal, MinIncrementsFloat32);
		if (would_overflow || force) {
			for (int i = 0; i < 64; i++) {
				probabilistic_total[i] += static_cast<double>(probabilistic_total_float[i]);
			}
			memset(probabilistic_total_float, 0, sizeof(probabilistic_total_float));
			exact_total = value;
		} else {
			exact_total = raw_subtotal;
		}
#endif
	}

	void Flush(ArenaAllocator &) {
		// Note: allocator unused - PacSumDoubleState doesn't do lazy allocation
#ifdef PAC_SUMAVG_FLOAT_CASCADING
		Flush32(0, true);
#endif
	}

	void GetTotalsAsDouble(double *dst) const {
		for (int i = 0; i < 64; i++) {
			dst[i] = probabilistic_total[i];
		}
	}

	// No buffering for double state (arrays are inline, no lazy allocation)
	bool IsBuffering() const {
		return false;
	}
};

} // namespace duckdb

#endif // PAC_SUM_HPP
