//
// Created by ila on 12/19/25.
//

#ifndef PAC_SUM_HPP
#define PAC_SUM_HPP

// the default implementation is a simd-friendly (16-bits) approximate, dual-unsigned, buffering, sum.
// benchmarking defines that disable certain optimizations (some imply the other)
// #define PAC_SIGNEDSUM 1 // to disable the use of an extra unsigned negative sum for negative values.
// #define PAC_NOBUFFERING 1 // to disable the buffering optimization.
// #define PAC_EXACTSUM 1 // to use exact cascading instead of the approximate algorithm
// #define PAC_NOCASCADING 1 // to disable multi-level cascading from small into wider types (aggregate in final type)
// #define PAC_NOSIMD 1 // to get the IF..THEN SIMD-unfriendly aggregate computation kernel
#ifdef PAC_NOSIMD
#define PAC_NOCASCADING 1
#endif
#ifdef PAC_NOCASCADING
#define PAC_EXACTSUM 1
#endif
#if defined(PAC_EXACTSUM) || defined(PAC_NOBUFFERING)
#define PAC_SIGNEDSUM 1
#endif

// PAC_GODBOLT mode: cpp -DPAC_GODBOLT -P -E -w src/include/pac_sum_avg.hpp
// Isolates the SIMD kernel for Godbolt analysis (-P removes line markers)
#ifdef PAC_GODBOLT
using uint8_t = unsigned char;
using int8_t = signed char;
using uint16_t = unsigned short;
using int16_t = signed short;
using uint32_t = unsigned int;
using int32_t = signed int;
using uint64_t = unsigned long long;
using int64_t = signed long long;
#else
#include "duckdb.hpp"
#include "pac_aggregate.hpp"
#include <random>
#include <atomic>

namespace duckdb {
void RegisterPacSumFunctions(ExtensionLoader &loader);
void RegisterPacAvgFunctions(ExtensionLoader &loader);
void RegisterPacSumCountersFunctions(ExtensionLoader &loader);

// ============================================================================
// PAC_SUM(hash_key, value)  aggregate function
// ============================================================================
// State: 64 sums, one for each bit position
// Update: for each (key_hash, value), add value to sums[i] if bit i of key_hash is set
// Finalize: compute the PAC-noised sum from the 64 counters
//
// Rather than naive FOR(i=0;i<64;i++) IF (keybit[i]) total[i]+=val
// we rewrite into SIMD-friendly multiplication FOR(i=0;i<64;i++) total[i]+=keybit[i]*val),
//
// mimicking what DuckDB's SUM() normally does, we have the following cases:
// 1) integers: PAC_SUM(key_hash, HUGEINT | [U](BIG||SMALL|TINY)INT) -> HUGEINT
// CASCADING EXACTSUM: keep sub-total8/16/32/64/128_t probabilistic_total, sum each value in smallest total that fits.
//              We ensure "things fit" by flushing probabilistic_totalX into the next wider total before it overflows,
//              and by only allowing values to be added to probabilistic_totalX if they have the highest bX bits unset,
//              so we can at least add 2^b values before we need to flush (b8=3, b16=5, b32=6, b64=8).
//              In Combine() we combine the levels of src and dst states, either by moving them from src to dst (if dst
//              did not have that level yet), or by summing them. In Finalize() the noised result is computed from the
//              sum of all allocated levels.
// APPROX (NEW!): only uses 16-bits counters in 25 lazily allocated levels, staggered by 4 bits (112 bits covered)
//              We sum into the highest level, such that the highest set bit in value is 4 positions clear from its max.
//              This means at least 16, and on average >32, such values can be summed before overflow could happen.
//              On overflow, all 16 bits are filled, the major 12 are added to the next counter and we reset to 0.
//              This ensures a max error of 1/4096 (2^12) = 0.025% in the sum totals, which is comfortable for PAC.
//              Because 25 level pointers is a largish array (200 bytes) that is often quite empty, we to use this space
//              for storing the first set of 16-bits counters (128 bytes) as long as only small (<9) levels are used.
//
// NEGATIVE VALUES: we now treat negative values differently: we negate them and put them in a second sum
//             only at Finalize, the negative sum is subtracted from the positivie sum.
//             This has two benefits: (i) it is numerically stabler (ii) unsigned sums have one more bit of precision
//             The extra negative sum may duplicate the memory usage. But in many cases, signed database columns
//             only contain positive values. And we allocate the negative sum lazily. So now we get unsigned perfomance
//             on signed data. This optimization can be disabled with PAC_SIGNEDSUM.
//
// 2) floating: PAC_SUM(key_hash, (FLOAT|DOUBLE)) -> DOUBLE
//              Accumulates directly into double[64] total using AddToTotalsSimple (ARM).
//
// 3) hugeint:  PAC_SUM(key_hash, UHUGEINT) -> DOUBLE
//              DuckDB produces DOUBLE outcomes for unsigned 128-bits integer sums, so we do as well.
//              This basically uses the DOUBLE methods where the updates perform a cast from hugeint
// APPROX (DEFAULT): now also uses the double path for unsigned 128-bits integers. It is faster and approximate.
//
// for DECIMAL types, we look at binding time which physical type is used and choose a relevant integer type.
//
// we also implement PAC_AVG(key_hash, value) with the same implementation functions as PAC_SUM(). To keep the
// code simple, we added an exact_counter also to PAC_SUM(). The only difference is that in the Finalize() for
// PAC_AVG() we divide the counter numbers by this exact_counter first.
//
// The cascading counter state could be quite large: ~2KB per aggregate result value. In aggregations with very many
// distinct GROUP BY values (and relatively modest sums, therefore), the bigger counters are often not needed.
// Therefore, we allocate counters lazily: only when values appear that need a particular counter level or when
// a level overflows, we allocate the space for the counters. This optimization reduces memory footprint, and
// helps in avoiding OOM, but bu itself was not enough. Because if there are very many unique states and the
// group-by keys appear far from each other, DuckDB starts with a new hash table, and the first phase of aggregation
// is in fact just partitioning. The real aggregation happens when the different partitions go to a Combine.
// In order to keep the initial size of the states low, it is therefore important to delay the state
// allocation until multiple values have been received (buffering). Processing a buffer
// rather than individual values reduces cache misses and increases chances for SIMD
#endif
// Simple version for double/[u]int64_t/hugeint (uses multiplication for conditional add)
// This auto-vectorizes well for 64-bits data-types because key_hash is also 64-bits
template <typename ACCUM_T, typename VALUE_T>
#ifndef PAC_GODBOLT
AUTOVECTORIZE static inline
#endif
    void
    AddToTotalsSimple(ACCUM_T *__restrict__ total, VALUE_T value, uint64_t key_hash) {
	ACCUM_T v = static_cast<ACCUM_T>(value);
	for (int j = 0; j < 64; j++) {
#ifdef PAC_NOSIMD
		if ((key_hash >> j) & 1ULL) { // IF..THEN cannot be SIMDized (and is 50%: has heavy branch misprediction cost).
			total[j] += v;
		}
#else
		total[j] += v * static_cast<ACCUM_T>((key_hash >> j) & 1ULL); // multiply value with 0 or 1; then add
#endif
	}
}

// For the 8-bits, 16-bits and 32-bits PAC_SUM() SIMD-benefits are greatest, but achieving these is harder
//
// We need SWAR (SIMD Within A Register). The idea in case of int8_t:
// - Pack 8 int8_t counters into each uint64_t (total[8] instead of total[64])
// - total[i] holds counters for bit positions i, i+8, i+16, i+24, i+32, i+40, i+48, i+56
// - 8 iterations instead of 64
//
// PAC_SUM() uses SWAR (SIMD Within A Register) for all integer bit widths (also for bigger int types if cascading)
// - BITS=8:  8 values packed per uint64_t, total[8],  8 iterations
// - BITS=16: 4 values packed per uint64_t, total[16], 16 iterations (only level used by new default approx approach)
// - BITS=32: 2 values packed per uint64_t, total[32], 32 iterations

// SWAR accumulation: pack multiple counters into uint64_t registers (for 8/16/32-bit elements)
// SIGNED_T/UNSIGNED_T: types for the packed elements (e.g., int8_t/uint8_t)
// MASK: broadcast mask (one bit per element, e.g., 0x0101010101010101 for 8-bit)
// VALUE_T: input value type
template <typename SIGNED_T, typename UNSIGNED_T, uint64_t MASK, typename VALUE_T>
#ifndef PAC_GODBOLT
AUTOVECTORIZE static inline
#endif
    void
    AddToTotalsSWAR(uint64_t *__restrict__ total, VALUE_T value, uint64_t key_hash) {
	constexpr int BITS = sizeof(SIGNED_T) * 8;
	// Cast to unsigned type to avoid sign extension, then broadcast to all lanes
	uint64_t val_packed = static_cast<UNSIGNED_T>(static_cast<SIGNED_T>(value)) * MASK;
	for (int i = 0; i < BITS; i++) {
		uint64_t bits = (key_hash >> i) & MASK;
		// Expand each 0x01 byte to 0xFF: (bits << BITS) - bits turns 0x01 into 0xFF per byte
		uint64_t expanded = (bits << BITS) - bits;
		total[i] += val_packed & expanded;
	}
}

#ifdef PAC_GODBOLT
// Explicit instantiations for Godbolt analysis
template void AddToTotalsSWAR<int8_t, uint8_t, 0x0101010101010101ULL, int64_t>(uint64_t *, int64_t, uint64_t);
template void AddToTotalsSWAR<int16_t, uint16_t, 0x0001000100010001ULL, int64_t>(uint64_t *, int64_t, uint64_t);
template void AddToTotalsSWAR<int32_t, uint32_t, 0x0000000100000001ULL, int64_t>(uint64_t *, int64_t, uint64_t);
template void AddToTotalsSimple<int64_t, int64_t>(int64_t *, int64_t, uint64_t);
template void AddToTotalsSimple<double, double>(double *, double, uint64_t);
#else
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

// ============================================================================
// Approximate sum constants (default, disabled when PAC_EXACTSUM is defined)
// ============================================================================
// 25 levels of 16-bits subtotals, staggered by 4 bits (covers 112 bits -- deemed enough for summing 64-bits ints into)
constexpr int PAC_APPROX_NUM_LEVELS = 25;
// SWAR elements per level (16 uint64_t, each holding 4 int16_t counters = 128 bytes)
constexpr int PAC_APPROX_SWAR_ELEMENTS = 16;
// Level shift: 4 bits between adjacent levels (16-bit counters overlap by 4 bits)
constexpr int PAC_APPROX_LEVEL_SHIFT = 4;
// SWAR mask for 16-bit elements (one bit per 16-bit lane)
constexpr uint64_t PAC_APPROX_SWAR_MASK = 0x0001000100010001ULL;

// Templated integer state - SIGNED selects signed/unsigned types and thresholds
// Uses lazy allocation via DuckDB's ArenaAllocator for memory management.
// Arena handles cleanup automatically when aggregate operation completes.
//
// By default uses the approximate algorithm with 13 levels of 16-bit counters.
// Define PAC_EXACTSUM to use the exact 5-level cascading implementation instead.
template <bool SIGNED>
struct PacSumIntState {
	// Type aliases based on signedness
	typedef typename std::conditional<SIGNED, int8_t, uint8_t>::type T8;
	typedef typename std::conditional<SIGNED, int16_t, uint16_t>::type T16;
	typedef typename std::conditional<SIGNED, int32_t, uint32_t>::type T32;
	typedef typename std::conditional<SIGNED, int64_t, uint64_t>::type T64;

	// Common fields for all variants
	uint64_t key_hash;    // OR of all key_hashes seen (for PacNoiseInNull)
	uint64_t exact_count; // total count of values added (for pac_avg)

#ifndef PAC_EXACTSUM
	// ========== APPROXIMATE STATE LAYOUT WITH INLINE STORAGE ==========
	// Uses 25 levels of 16-bit counters with 4-bit shifts (covers 112 bits & provides >=12-bit accuracy on the totals).
	int8_t max_level_used;   // highest level that received data (-1 if none)
	int8_t inline_level_idx; // which level uses inline storage (-1 if none/spilled)

	// Exact subtotal at each level - tracks cumulative sum for overflow detection
	int32_t exact_total[PAC_APPROX_NUM_LEVELS];

	union LevelStorage { // inline allocation optimization uses union to ensure it also works on 32-bits platforms
		uint64_t *levels[PAC_APPROX_NUM_LEVELS]; // motivation: this is quite an array: 25*8bytes = 200 bytes
		struct InlineStorage {                   // try to reuse the back 128 bytes for the first allocated level
			void *_dummy[9];                     // do this only if allocated levels < 9
			uint64_t inline_level[PAC_APPROX_SWAR_ELEMENTS]; // 128 bytes + 9*8 = 200
		} inline_storage;
	} u;

	// Get the level index for a value based on its highest set bit
	// For 16-bit counters with 4-bit shift: shifted_val = value >> (level * 4) should fit in ~12 bits
	static inline int GetLevel(int64_t abs_val) {
		int level = ((63 - (__builtin_clzll(1 | abs_val))) - 8) >> 2;
		return (abs_val < 4096) ? 0 : level;
	}

	// Allocate a single level (helper)
	inline void AllocateLevel(ArenaAllocator &allocator, int k) {
		if (k >= 9 && inline_level_idx >= 0) {
			uint64_t *ext = // Need level >= 9 and have inline level? Copy it out first
			    reinterpret_cast<uint64_t *>(allocator.Allocate(PAC_APPROX_SWAR_ELEMENTS * sizeof(uint64_t)));
			memcpy(ext, u.inline_storage.inline_level, PAC_APPROX_SWAR_ELEMENTS * sizeof(uint64_t));
			u.levels[inline_level_idx] = ext;
			inline_level_idx = -1;
			// Clear u.inline_storage.inline_level area so u.levels[9..24] read as nullptr
			memset(u.inline_storage.inline_level, 0, PAC_APPROX_SWAR_ELEMENTS * sizeof(uint64_t));
		}
		// ok, now  allocate
		if (k < 9 && inline_level_idx < 0) { // Use inline if k < 9 and not yet used
			u.levels[k] = u.inline_storage.inline_level;
			memset(u.inline_storage.inline_level, 0, PAC_APPROX_SWAR_ELEMENTS * sizeof(uint64_t));
			inline_level_idx = static_cast<int8_t>(k);
		} else {
			u.levels[k] = reinterpret_cast<uint64_t *>(allocator.Allocate(PAC_APPROX_SWAR_ELEMENTS * sizeof(uint64_t)));
			memset(u.levels[k], 0, PAC_APPROX_SWAR_ELEMENTS * sizeof(uint64_t));
		}
	}

	// Ensure level k is allocated; if k < max_level_used, also allocate all levels in between
	inline void EnsureLevelAllocated(ArenaAllocator &allocator, int k) {
		if (DUCKDB_LIKELY(k <= max_level_used)) {
			return;
		}
		// Allocate all levels from max_level_used+1 to k
		for (int i = max_level_used + 1; i <= k; i++) {
			AllocateLevel(allocator, i);
		}
		max_level_used = static_cast<int8_t>(k);
	}

	// 16-bit overflow thresholds
	static constexpr int32_t OVERFLOW_THRESHOLD = SIGNED ? 32767 : 65535;
	static constexpr int32_t UNDERFLOW_THRESHOLD = SIGNED ? -32768 : 0;

	// Add amount to exact_total[level], allocating level if needed, cascading if overflow
	inline void AddToExactTotal(int level, int32_t amount, ArenaAllocator &allocator) {
		EnsureLevelAllocated(allocator, level);
		int32_t new_total = exact_total[level] + amount;
		if (DUCKDB_UNLIKELY(new_total > OVERFLOW_THRESHOLD || new_total < UNDERFLOW_THRESHOLD)) {
			Cascade(level, allocator);
			exact_total[level] = amount;
		} else {
			exact_total[level] = new_total;
		}
	}

	// Cascade level k to level k+1 (called when level k would overflow)
	// Counter type: int16_t for signed, uint16_t for unsigned
	using CounterT = typename std::conditional<SIGNED, int16_t, uint16_t>::type;

	void Cascade(int k, ArenaAllocator &allocator) {
		if (k >= PAC_APPROX_NUM_LEVELS - 1 || !u.levels[k]) {
			return;
		}
		int32_t cascade_amount = exact_total[k] >> PAC_APPROX_LEVEL_SHIFT;
		AddToExactTotal(k + 1, cascade_amount, allocator);

		CounterT *src = reinterpret_cast<CounterT *__restrict__>(u.levels[k]);
		CounterT *dst = reinterpret_cast<CounterT *__restrict__>(u.levels[k + 1]);

		for (int j = 0; j < 64; j++) {
			dst[j] += static_cast<CounterT>(src[j] >> PAC_APPROX_LEVEL_SHIFT);
		}
		memset(u.levels[k], 0, PAC_APPROX_SWAR_ELEMENTS * sizeof(uint64_t));
		exact_total[k] = 0;
	}

	// Flush all levels by cascading up to the highest level that received data
	void Flush(ArenaAllocator &allocator) {
		if (max_level_used < 0) {
			return;
		}
		for (int k = 0; k < max_level_used; k++) {
			Cascade(k, allocator);
		}
	}

	// Get probabilistic totals as double[64] for finalization
	void GetTotalsAsDouble(double *dst) const {
		if (max_level_used < 0) {
			memset(dst, 0, 64 * sizeof(double));
			return;
		}
		double scale = static_cast<double>(1ULL << (PAC_APPROX_LEVEL_SHIFT * max_level_used));
		const CounterT *counters = reinterpret_cast<const CounterT *>(u.levels[max_level_used]);
		for (int j = 0; j < 64; j++) {
			int swar_idx = (j % 16) * 4 + (j / 16);
			dst[j] = static_cast<double>(counters[swar_idx]) * scale;
		}
	}

	// Combine another state into this one (used in Combine phase)
	void CombineFrom(PacSumIntState *src, ArenaAllocator &allocator) {
		if (!src)
			return;
		key_hash |= src->key_hash;
		exact_count += src->exact_count;
		for (int k = 0; k <= src->max_level_used; k++) {
			if (!src->u.levels[k])
				continue;
			if (k > max_level_used) { // dst does not have this yet. Try to steal from src
#ifdef PAC_NOBUFFERING
				// can only steal src's pointer if it's arena-allocated (not inline storage).
				if (k != src->inline_level_idx)         // without buffering, the SumState is not arena-allocated
#endif
					if (k < 9 || max_level_used >= 9) { // beware not to mess up dst inline storage
						u.levels[k] = src->u.levels[k];
						exact_total[k] = src->exact_total[k];
						max_level_used = k;
						continue;
					}
				EnsureLevelAllocated(allocator, k);
			}
			constexpr int32_t THRESHOLD = SIGNED ? 32767 : 65535;
			int32_t new_total = exact_total[k] + src->exact_total[k];
			if (new_total > THRESHOLD || (SIGNED && new_total < -32768)) {
				Cascade(k, allocator);
				exact_total[k] = src->exact_total[k];
			} else {
				exact_total[k] = new_total;
			}
			uint64_t *src_level = src->u.levels[k];
			uint64_t *dst_level = u.levels[k];
			for (int j = 0; j < PAC_APPROX_SWAR_ELEMENTS; j++) {
				dst_level[j] += src_level[j];
			}
		}
	}

#elif defined(PAC_NOCASCADING)
	// ========== NON-CASCADING STATE LAYOUT ==========
	hugeint_t probabilistic_total128[64]; // final total (non-cascading mode only)

	void Flush(ArenaAllocator &) {
	} // no-op
	void GetTotalsAsDouble(double *dst) const {
		ToDoubleArray(probabilistic_total128, dst);
	}

#else
// SIGNED is compile-time known, so for unsigned the negative cases (value < 0) will be compiled away
#define ACCUMULATE_BITMARGIN      2 // val must be 2 bits shorter than accumulator to allow >=4 updates without overflow
#define UPPERBOUND_BITWIDTH(bits) (1LL << ((bits - SIGNED) - ACCUMULATE_BITMARGIN))
#define LOWERBOUND_BITWIDTH(bits) -(static_cast<int64_t>(SIGNED) << ((bits - SIGNED) - ACCUMULATE_BITMARGIN))

	// ========== EXACT STATE LAYOUT (default) ==========
	// Field ordering optimized for memory layout:
	// 1. exact_totals (sized to fit level's range - value is always valid after flush)
	// 2. key_hash (used to track initialization of all counters)
	// 3. exact_count (required for AVG really)
	// 4. probabilistic pointers (lazily allocated)
	// Exact subtotals for each level - sized to fit level's representable range.
	// After any Flush, exact_totalN is either:
	//   - the incoming `value` (which fits in TN since it was routed to level N), or
	//   - `new_total` which we verified fits in TN range before assigning
	// Flush functions use int64_t for intermediate calculations to detect overflow.
	T8 exact_total8;   // sum of values at level 8 since last flush (fits in T8 after flush)
	T16 exact_total16; // sum of values at level 16 since last flush (fits in T16 after flush)
	T32 exact_total32; // sum of values at level 32 since last flush (fits in T32 after flush)
	T64 exact_total64; // sum of values at level 64 since last flush (fits in T64 after flush)

	// All levels lazily allocated via arena allocator (nullptr if not allocated)
	uint64_t *probabilistic_total8;    // 8 x uint64_t (64 bytes) when allocated, each holds 8 packed T8
	uint64_t *probabilistic_total16;   // 16 x uint64_t (128 bytes) when allocated, each holds 4 packed T16
	uint64_t *probabilistic_total32;   // 32 x uint64_t (256 bytes) when allocated, each holds 2 packed T32
	uint64_t *probabilistic_total64;   // 64 x uint64_t (512 bytes) when allocated, each holds 1 T64
	hugeint_t *probabilistic_total128; // 64 x hugeint_t (1024 bytes) when allocated

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
#endif // (if default-approx else exactsum)

	// Interface methods for wrapper compatibility
	PacSumIntState *GetState() {
		return this;
	}
	PacSumIntState *EnsureState(ArenaAllocator &) {
		return this;
	}
};

// Double pac_sum state is noncascading: directly aggregates float/double into double
struct PacSumDoubleState {
	uint64_t key_hash;    // OR of all key_hashes seen (for PacNoiseInNull)
	uint64_t exact_count; // total count of values added (for pac_avg)
#ifdef PAC_EXACTSUM
	double probabilistic_total[64];
#else
	float probabilistic_total[64]; // approximate double with float
#endif
	void GetTotalsAsDouble(double *dst) const {
		for (int i = 0; i < 64; i++) {
			dst[i] = probabilistic_total[i];
		}
	}
	// Interface methods for wrapper compatibility
	inline void Flush32(double value, bool force = false) {
	}
	void Flush(ArenaAllocator &) {
	}
	PacSumDoubleState *GetState() {
		return this;
	}
	PacSumDoubleState *EnsureState(ArenaAllocator &) {
		return this;
	}
};

#ifndef PAC_NOBUFFERING
// ============================================================================
// PacSumStateWrapper: unified buffering wrapper for both int and double states
// Buffers (hash, value) pairs before allocating the inner state from arena
// ============================================================================
template <typename InnerState, typename ValueT>
struct PacSumStateWrapper {
	using State = InnerState;
	using Value = ValueT;
	static constexpr int BUF_SIZE = 2;
	static constexpr uint64_t BUF_MASK = 3ULL;

	ValueT val_buf[BUF_SIZE];
	uint64_t hash_buf[BUF_SIZE];
	union {
		uint64_t n_buffered; // lower 2 bits: count, upper bits: state pointer
		State *state;
	};
#ifndef PAC_SIGNEDSUM
	// Have two pos and neg states that store POSITIVE values with UNSIGNED arithmetic alone
	PacSumIntState<false> *neg_state; // (this is more stable and resilient against mixed-sign value distributions)
#endif

	State *GetState() const {
		return reinterpret_cast<State *>(reinterpret_cast<uintptr_t>(state) & ~7ULL);
	}

	State *EnsureState(ArenaAllocator &a) {
		State *s = GetState();
		if (!s) {
			s = reinterpret_cast<State *>(a.Allocate(sizeof(State)));
			memset(s, 0, sizeof(State));
			InitializeState(s);
			state = s;
		}
		return s;
	}

#ifndef PAC_SIGNEDSUM
	// neg_state uses unsigned arithmetic (stores absolute values of negatives)
	PacSumIntState<false> *GetNegState() const {
		return neg_state;
	}
	PacSumIntState<false> *EnsureNegState(ArenaAllocator &a) {
		if (!neg_state) {
			neg_state = reinterpret_cast<PacSumIntState<false> *>(a.Allocate(sizeof(PacSumIntState<false>)));
			memset(neg_state, 0, sizeof(PacSumIntState<false>));
			neg_state->max_level_used = -1;
			neg_state->inline_level_idx = -1;
		}
		return neg_state;
	}
#endif

	// StateSize: for unsigned types in double-sided mode, neg_state is never used
	static idx_t StateSize() {
#ifndef PAC_SIGNEDSUM
		// Only exclude neg_state for unsigned integer types (double doesn't have neg_state issues)
		if (!std::is_same<State, PacSumDoubleState>::value && std::is_unsigned<ValueT>::value) {
			return sizeof(PacSumStateWrapper) - sizeof(PacSumIntState<false> *); // exclude neg_state
		}
#endif
		return sizeof(PacSumStateWrapper);
	}

private:
	// Helper to initialize state - only sets max_level_used for integer state in approx mode
	template <typename S = State>
	typename std::enable_if<!std::is_same<S, PacSumDoubleState>::value>::type InitializeState(S *s) {
#ifndef PAC_EXACTSUM
		s->max_level_used = -1;   // no levels have received data yet
		s->inline_level_idx = -1; // no inline level yet
#else
		(void)s;
#endif
	}
	template <typename S = State>
	typename std::enable_if<std::is_same<S, PacSumDoubleState>::value>::type InitializeState(S *) {
		// PacSumDoubleState doesn't use approx algorithm
	}
};

// Type aliases for convenience
template <bool SIGNED>
using PacSumIntStateWrapper = PacSumStateWrapper<PacSumIntState<SIGNED>, typename PacSumIntState<SIGNED>::T64>;
using PacSumDoubleStateWrapper = PacSumStateWrapper<PacSumDoubleState, double>;
#endif // PAC_NOBUFFERING

// Forward declarations for pac_sum functions exported for pac_avg

// State type selection (defined in pac_sum.cpp, needed by pac_avg.cpp)
#ifdef PAC_NOBUFFERING
template <bool SIGNED>
using ScatterIntState = PacSumIntState<SIGNED>;
using ScatterDoubleState = PacSumDoubleState;
#else
template <bool SIGNED>
using ScatterIntState = PacSumIntStateWrapper<SIGNED>;
using ScatterDoubleState = PacSumDoubleStateWrapper;

// FlushBuffer - flushes src's buffer into dst's inner state (declared here, defined in pac_sum.cpp)
template <bool SIGNED, typename WrapperT>
void PacSumFlushBuffer(WrapperT &src, WrapperT &dst, ArenaAllocator &a);
#endif

// ============================================================================
// Double-sided mode helpers (used by pac_sum and pac_avg finalization)
// ============================================================================
#ifndef PAC_SIGNEDSUM
// For int states: cast to unsigned before Flush (uses unsigned counters in Cascade)
template <class State>
inline typename std::enable_if<!std::is_same<typename State::State, PacSumDoubleState>::value>::type
FlushPosState(typename State::State *pos, ArenaAllocator &allocator) {
	reinterpret_cast<PacSumIntState<false> *>(pos)->Flush(allocator);
}
// For double states: call directly (no cast needed)
template <class State>
inline typename std::enable_if<std::is_same<typename State::State, PacSumDoubleState>::value>::type
FlushPosState(typename State::State *pos, ArenaAllocator &allocator) {
	pos->Flush(allocator);
}

// For int states: cast to unsigned for GetTotalsAsDouble
template <class State>
inline typename std::enable_if<!std::is_same<typename State::State, PacSumDoubleState>::value>::type
GetPosStateTotals(typename State::State *pos, double *buf) {
	reinterpret_cast<PacSumIntState<false> *>(pos)->GetTotalsAsDouble(buf);
}
// For double states: call directly
template <class State>
inline typename std::enable_if<std::is_same<typename State::State, PacSumDoubleState>::value>::type
GetPosStateTotals(typename State::State *pos, double *buf) {
	pos->GetTotalsAsDouble(buf);
}

// For signed int states: subtract neg_state totals from buf, return neg count
template <class State, bool SIGNED>
inline typename std::enable_if<SIGNED && !std::is_same<typename State::State, PacSumDoubleState>::value, uint64_t>::type
SubtractNegStateTotals(State *wrapper, double *buf, uint64_t &key_hash, ArenaAllocator &allocator) {
	if (auto *neg = wrapper->GetNegState()) {
		double neg_buf[64] = {0};
		neg->Flush(allocator);
		neg->GetTotalsAsDouble(neg_buf);
		key_hash |= neg->key_hash;
		for (int j = 0; j < 64; j++) {
			buf[j] -= neg_buf[j];
		}
		return neg->exact_count;
	}
	return 0;
}
// For unsigned int states or double states: no-op, return 0
template <class State, bool SIGNED>
inline typename std::enable_if<!SIGNED || std::is_same<typename State::State, PacSumDoubleState>::value, uint64_t>::type
SubtractNegStateTotals(State *, double *, uint64_t &, ArenaAllocator &) {
	return 0; // No neg_state for unsigned types or double states
}
#endif

// Generic finalize template (used by pac_sum and pac_avg)
template <class State, class ACC_TYPE, bool SIGNED, bool DIVIDE_BY_COUNT = false>
void PacSumFinalize(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset);

// Templated bind for DECIMAL pac_sum/pac_avg (IS_AVG=false for sum, IS_AVG=true for avg)
template <bool IS_AVG>
unique_ptr<FunctionData> BindDecimalPacSumAvg(ClientContext &ctx, AggregateFunction &function,
                                              vector<unique_ptr<Expression>> &args);

// Templated FinalizeCounters for pac_sum_counters/pac_avg_counters (DIVIDE_BY_COUNT=false for sum, true for avg)
template <class State, bool SIGNED, bool DIVIDE_BY_COUNT>
void PacSumAvgFinalizeCounters(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset);

} // namespace duckdb
#endif // PAC_GODBOLT

#endif // PAC_SUM_HPP
