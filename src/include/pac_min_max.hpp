//
// pac_min_max.hpp - PAC MIN/MAX aggregate functions
//

#ifndef PAC_MIN_MAX_HPP
#define PAC_MIN_MAX_HPP

#include "duckdb.hpp"
#include "pac_aggregate.hpp"

namespace duckdb {

void RegisterPacMinFunctions(ExtensionLoader &loader);
void RegisterPacMaxFunctions(ExtensionLoader &loader);

// ============================================================================
// PAC_MIN/PAC_MAX(hash_key, value) aggregate functions
// ============================================================================
// State: 64 extreme values, one for each bit position
// Update: for each (key_hash, value), update extremes[i] if bit i of key_hash is set
// Finalize: compute the PAC-noised min/max from the 64 counters
//
// Optimization: keep a global bound (min of all maxes, or max of all mins)
// to skip processing values that can't affect any extreme.
//
// Cascading: start with smallest integer type, upgrade when value doesn't fit.
//
// some #defines to demonstrate the effects of our optimizations:
// Define PAC_MINMAX_NONCASCADING to use fixed-width arrays (input value type).
// Define PAC_MINMAX_NONLAZY to pre-allocate all levels at initialization (and suffer the consequences)
// Define PAC_MINMAX_NOBOUNDOP to not compute a global bound that allows early-out often (will be slower likely)

//#define PAC_MINMAX_NONCASCADING 1
//#define PAC_MINMAX_NONLAZY 1
//#define PAC_MINMAX_NOBOUNDOPT 1

// Recompute global_bound every N updates (reduces overhead of bound computation)
static constexpr uint16_t BOUND_RECOMPUTE_INTERVAL = 2048;

// Type-agnostic comparison macros (IS_MAX must be in scope)
#define PAC_IS_BETTER(a, b) (IS_MAX ? ((a) > (b)) : ((a) < (b)))
#define PAC_BETTER(a, b)    (PAC_IS_BETTER(a, b) ? (a) : (b))
#define PAC_WORSE(a, b)     (PAC_IS_BETTER(a, b) ? (b) : (a))

// ============================================================================
// SIMD-friendly update functions for Min/Max extremes arrays
// Uses union-based bitwise ops that work for all types (int and float)
// prototyped here: https://godbolt.org/z/dYWqd3qEW
// ============================================================================

// SWAR traits for different element sizes
// BitsT: signed type for mask generation (same size as element)
// UintT: unsigned type for bitwise ops on value (same size as element)
template <int SIZE>
struct SWARTraits;

template <>
struct SWARTraits<1> { // int8
	using BitsT = int8_t;
	using UintT = uint8_t;
	static constexpr uint64_t MASK = 0x0101010101010101ULL;
	static constexpr int SHIFTS = 8;
	static constexpr int ELEM_PER_U64 = 8;
};

template <>
struct SWARTraits<2> { // int16
	using BitsT = int16_t;
	using UintT = uint16_t;
	static constexpr uint64_t MASK = 0x0001000100010001ULL;
	static constexpr int SHIFTS = 16;
	static constexpr int ELEM_PER_U64 = 4;
};

template <>
struct SWARTraits<4> { // int32, float
	using BitsT = int32_t;
	using UintT = uint32_t;
	static constexpr uint64_t MASK = 0x0000000100000001ULL;
	static constexpr int SHIFTS = 32;
	static constexpr int ELEM_PER_U64 = 2;
};

template <>
struct SWARTraits<8> { // int64, double - linear layout
	using BitsT = int64_t;
	using UintT = uint64_t;
	static constexpr uint64_t MASK = 1ULL;
	static constexpr int SHIFTS = 64;
	static constexpr int ELEM_PER_U64 = 1;
};

template <>
struct SWARTraits<16> { // hugeint - linear layout
	using BitsT = int64_t;
	using UintT = uint64_t;
	static constexpr uint64_t MASK = 1ULL;
	static constexpr int SHIFTS = 64;
	static constexpr int ELEM_PER_U64 = 1;
};

// Unified SIMD update using union for bitwise ops (works for all types)
// SWAR layout for sizeof(T) <= 4, linear for sizeof(T) == 8
template <typename T, bool IS_MAX, int SIZE = sizeof(T)>
struct UpdateExtremesKernel {
	using Traits = SWARTraits<SIZE>;
	using BitsT = typename Traits::BitsT;
	using UintT = typename Traits::UintT;

	AUTOVECTORIZE static inline void update(T *__restrict__ result, uint64_t key_hash, T value) {
		union {
			uint64_t u64[Traits::SHIFTS];
			BitsT bits[64];
		} buf;
		for (int i = 0; i < Traits::SHIFTS; i++) {
			buf.u64[i] = (key_hash >> i) & Traits::MASK;
		}
		for (int i = 0; i < 64; i++) {
			UintT mask_u = static_cast<UintT>(-buf.bits[i]);
			T extreme = PAC_BETTER(value, result[i]);
			union {
				T val;
				UintT bits;
			} extreme_u, result_u, out;
			extreme_u.val = extreme;
			result_u.val = result[i];
			out.bits = (extreme_u.bits & mask_u) | (result_u.bits & ~mask_u);
			result[i] = out.val;
		}
	}
};

// Specialization for 16-byte types (hugeint) - scalar comparison, no bitwise tricks
template <typename T, bool IS_MAX>
struct UpdateExtremesKernel<T, IS_MAX, 16> {
	AUTOVECTORIZE static inline void update(T *__restrict__ result, uint64_t key_hash, T value) {
		for (int i = 0; i < 64; i++) {
			if (((key_hash >> i) & 1ULL) && PAC_IS_BETTER(value, result[i])) {
				result[i] = value;
			}
		}
	}
};

// Convenience wrapper
template <typename T, bool IS_MAX>
AUTOVECTORIZE static inline void UpdateExtremes(T *__restrict__ result, uint64_t key_hash, T value) {
	UpdateExtremesKernel<T, IS_MAX>::update(result, key_hash, value);
}

// ============================================================================
// SWAR index helpers
// ============================================================================
// Convert linear bit index to SWAR index for given element width.
// This must match the layout produced by the union-based bit extraction in UpdateExtremes.
// ELEM_PER_U64: elements per uint64_t (8 for 8-bit, 4 for 16-bit, 2 for 32-bit)
template <int ELEM_PER_U64>
static inline int LinearToSWAR(int linear_idx) {
	constexpr int NUM_U64 = 64 / ELEM_PER_U64; // The union packs bits such that linear index L maps to SWAR index:
	return ELEM_PER_U64 * (linear_idx % NUM_U64) + (linear_idx / NUM_U64); // EPU * (L % NUM_U64) + (L / NUM_U64)
}

// Extract from SWAR layout to linear double array (where we do noising)
template <typename T, int ELEM_PER_U64>
static inline void ExtractSWAR(const T *swar_data, double *dst) {
	for (int i = 0; i < 64; i++) {
		int swar_idx = LinearToSWAR<ELEM_PER_U64>(i);
		dst[i] = static_cast<double>(swar_data[swar_idx]);
	}
}

// ============================================================================
// Helper to recompute global bound from extremes array
// NOTE: ToDouble<T> is already defined in pac_aggregate.hpp
// ============================================================================
template <typename T, typename BOUND_T, bool IS_MAX>
AUTOVECTORIZE static inline BOUND_T ComputeGlobalBound(const T *extremes) {
	BOUND_T bound = static_cast<BOUND_T>(extremes[0]);
	for (int i = 1; i < 64; i++) {
		BOUND_T ext = static_cast<BOUND_T>(extremes[i]);
		bound = PAC_WORSE(ext, bound);
	}
	return bound;
}

// ============================================================================
// Unified state for min/max (handles both integer and floating-point types)
// ============================================================================
// IS_FLOAT: true for float/double, false for integer types
// SIGNED: signed vs unsigned (ignored for float)
// IS_MAX: true for pac_max, false for pac_min
// MAXLEVEL: maximum cascading level (1-5 for int, 1-2 for float)
//
// Level abstraction:
//   Level 1: int8/uint8 or float (1-4 bytes, uses SWAR)
//   Level 2: int16/uint16 or double (2-8 bytes, SWAR for int16, linear for double)
//   Level 3: int32/uint32 (4 bytes, uses SWAR) - integers only
//   Level 4: int64/uint64 (8 bytes, linear) - integers only
//   Level 5: hugeint/uhugeint (16 bytes, linear) - integers only
//
// All pointer fields are always defined but ordered by level at the end of the struct.
// We use offsetof() to report smaller state_size to DuckDB, so it allocates only
// the memory needed for the MAXLEVEL. Regular if guards (not if constexpr) prevent
// access to unallocated fields - compiler optimizes these away since MAXLEVEL is constant.
template <bool IS_FLOAT, bool SIGNED, bool IS_MAX, int MAXLEVEL>
struct PacMinMaxState {
	// Constants
	static constexpr int MINLEVEL = 1;
	static constexpr int NUMLEVEL = IS_FLOAT ? 2 : 5;

	// Type aliases - for float: T1=float, T2=double; T3-T5 = double (never used but must compile)
	//              - for integer: T1=int8, T2=int16, T3=int32, T4=int64, T5=hugeint
	template <bool S>
	using SignedInt = typename std::conditional<S, int8_t, uint8_t>::type;
	template <bool S>
	using SignedInt16 = typename std::conditional<S, int16_t, uint16_t>::type;
	template <bool S>
	using SignedInt32 = typename std::conditional<S, int32_t, uint32_t>::type;
	template <bool S>
	using SignedInt64 = typename std::conditional<S, int64_t, uint64_t>::type;
	template <bool S>
	using SignedHuge = typename std::conditional<S, hugeint_t, uhugeint_t>::type;

	using T1 = typename std::conditional<IS_FLOAT, float, SignedInt<SIGNED>>::type;
	using T2 = typename std::conditional<IS_FLOAT, double, SignedInt16<SIGNED>>::type;
	using T3 = typename std::conditional<IS_FLOAT, double, SignedInt32<SIGNED>>::type;
	using T4 = typename std::conditional<IS_FLOAT, double, SignedInt64<SIGNED>>::type;
	using T5 = typename std::conditional<IS_FLOAT, double, SignedHuge<SIGNED>>::type;

	// TMAX: type at MAXLEVEL (helper to select T1-T5 by level)
	template <int L>
	using TAtLevel = typename std::conditional<
	    L == 5, T5,
	    typename std::conditional<
	        L == 4, T4,
	        typename std::conditional<L == 3, T3, typename std::conditional<L == 2, T2, T1>::type>::type>::type>::type;
	using TMAX = TAtLevel<MAXLEVEL>;
	using ValueType = TMAX;

	// Float range for level 1 cascading (values outside this upgrade to double)
	static constexpr double FLOAT_RANGE_MIN = -1000000.0;
	static constexpr double FLOAT_RANGE_MAX = 1000000.0;

	// Get init value (worst possible for the aggregation direction)
	// Float types use infinity, integer types use NumericLimits
	template <typename T>
	static inline T TypeInit() {
		if (std::is_floating_point<T>::value) {
			return IS_MAX ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
		}
		return IS_MAX ? NumericLimits<T>::Minimum() : NumericLimits<T>::Maximum();
	}

	// Check if value fits at level 1 (int8/uint8 or float range)
	static inline bool FitsInLevel1(TMAX val) {
		if (IS_FLOAT) {
			return val >= static_cast<TMAX>(FLOAT_RANGE_MIN) && val <= static_cast<TMAX>(FLOAT_RANGE_MAX);
		}
		return val >= static_cast<TMAX>(SIGNED ? INT8_MIN : 0) &&
		       val <= static_cast<TMAX>(SIGNED ? INT8_MAX : UINT8_MAX);
	}

	// Check if value fits at level 2 (int16/uint16 or double - double fits everything)
	static inline bool FitsInLevel2(TMAX val) {
		if (IS_FLOAT) {
			return true; // float fits all double values
		}
		return val >= static_cast<TMAX>(SIGNED ? INT16_MIN : 0) &&
		       val <= static_cast<TMAX>(SIGNED ? INT16_MAX : UINT16_MAX);
	}

	// Check if value fits at level 3 (int32/uint32) - only used for integers
	static inline bool FitsInLevel3(TMAX val) {
		return val >= static_cast<TMAX>(SIGNED ? INT32_MIN : 0) &&
		       val <= static_cast<TMAX>(SIGNED ? INT32_MAX : UINT32_MAX);
	}

	// Check if value fits at level 4 (int64/uint64) - only used for integers
	static inline bool FitsInLevel4(TMAX val) {
		return val >= static_cast<TMAX>(SIGNED ? INT64_MIN : 0) &&
		       val <= static_cast<TMAX>(SIGNED ? INT64_MAX : UINT64_MAX);
	}

	// ========== Common fields (defined once for both modes) ==========
	bool initialized;
	bool seen_null;
	uint16_t update_count;
	TMAX global_bound; // For MAX: min of all maxes; for MIN: max of all mins

#ifdef PAC_MINMAX_NONCASCADING
	// ========== Non-cascading mode: single fixed-width array ==========
	TMAX extremes[64];

	void GetTotalsAsDouble(double *dst) const {
		for (int i = 0; i < 64; i++) {
			dst[i] = ToDouble(extremes[i]);
		}
	}

	void Initialize() {
		for (int i = 0; i < 64; i++) {
			extremes[i] = TypeInit<TMAX>();
		}
		global_bound = TypeInit<TMAX>();
		update_count = 0;
		initialized = true;
	}

	// Recompute global bound - periodically called after updates
	void RecomputeBound() {
#ifndef PAC_MINMAX_NOBOUNDOPT
		if (++update_count == BOUND_RECOMPUTE_INTERVAL) {
			update_count = 0;
			global_bound = ComputeGlobalBound<TMAX, TMAX, IS_MAX>(extremes);
		}
#endif
	}
#else
	// ========== Cascading mode fields ==========
	// All pointers defined, but ordered so unused ones are at the end.
	// DuckDB allocates only up to the needed pointer based on state_size.
	uint8_t current_level; // 1, 2, 3, 4, or 5
	ArenaAllocator *allocator;

	// Pointer fields ordered by level - unused ones at the end won't be allocated
	T1 *extremes_level1;
	T2 *extremes_level2;
	T3 *extremes_level3; // Only for integers (unused for float)
	T4 *extremes_level4; // Only for integers (unused for float)
	T5 *extremes_level5; // Only for integers (unused for float)

	// Allocate a level's buffer
	template <typename T>
	inline T *AllocateLevel(T init_value) {
		T *buf = reinterpret_cast<T *>(allocator->Allocate(64 * sizeof(T)));
		for (int i = 0; i < 64; i++) {
			buf[i] = init_value;
		}
		return buf;
	}

	// Helper to get SWAR index, handling different sizes
	// Returns linear index for types > 4 bytes (no SWAR benefit)
	template <typename T>
	static inline int GetIndex(int i) {
		// Types > 4 bytes use linear indexing
		if (sizeof(T) > 4)
			return i;
		// Types <= 4 bytes use SWAR layout
		if (sizeof(T) == 4)
			return LinearToSWAR<2>(i);
		if (sizeof(T) == 2)
			return LinearToSWAR<4>(i);
		return LinearToSWAR<8>(i); // sizeof(T) == 1
	}

	// Upgrade from one level to the next, automatically handling SWAR vs linear layout
	// Types <= 4 bytes use SWAR layout, types > 4 bytes use linear layout
	// If src value equals src_init (never updated), use dst_init instead
	template <typename SRC_T, typename DST_T>
	inline DST_T *UpgradeLevel(SRC_T *src, SRC_T src_init, DST_T dst_init) {
		DST_T *dst = reinterpret_cast<DST_T *>(allocator->Allocate(64 * sizeof(DST_T)));
		if (src) {
			for (int i = 0; i < 64; i++) {
				int src_idx = GetIndex<SRC_T>(i);
				int dst_idx = GetIndex<DST_T>(i);
				dst[dst_idx] = (src[src_idx] == src_init) ? dst_init : static_cast<DST_T>(src[src_idx]);
			}
		} else {
			for (int i = 0; i < 64; i++) {
				dst[i] = dst_init;
			}
		}
		return dst;
	}

	void AllocateFirstLevel(ArenaAllocator &alloc) {
		allocator = &alloc;
#ifdef PAC_MINMAX_NONLAZY
		// Pre-allocate all levels that exist for this MAXLEVEL
		extremes_level1 = AllocateLevel<T1>(TypeInit<T1>());
		if (MAXLEVEL >= 2) {
			extremes_level2 = AllocateLevel<T2>(TypeInit<T2>());
		}
		if (MAXLEVEL >= 3) {
			extremes_level3 = AllocateLevel<T3>(TypeInit<T3>());
		}
		if (MAXLEVEL >= 4) {
			extremes_level4 = AllocateLevel<T4>(TypeInit<T4>());
		}
		if (MAXLEVEL >= 5) {
			extremes_level5 = AllocateLevel<T5>(TypeInit<T5>());
		}
		current_level = MAXLEVEL;
#else
		extremes_level1 = AllocateLevel<T1>(TypeInit<T1>());
		current_level = 1;
#endif
		update_count = 0;
		global_bound = TypeInit<TMAX>();
		initialized = true;
	}

	void UpgradeToLevel2() {
		if (MAXLEVEL >= 2) {
			extremes_level2 = UpgradeLevel<T1, T2>(extremes_level1, TypeInit<T1>(), TypeInit<T2>());
			current_level = 2;
		}
	}

	void UpgradeToLevel3() {
		if (MAXLEVEL >= 3) {
			extremes_level3 = UpgradeLevel<T2, T3>(extremes_level2, TypeInit<T2>(), TypeInit<T3>());
			current_level = 3;
		}
	}

	void UpgradeToLevel4() {
		if (MAXLEVEL >= 4) {
			extremes_level4 = UpgradeLevel<T3, T4>(extremes_level3, TypeInit<T3>(), TypeInit<T4>());
			current_level = 4;
		}
	}

	void UpgradeToLevel5() {
		if (MAXLEVEL >= 5) {
			extremes_level5 = UpgradeLevel<T4, T5>(extremes_level4, TypeInit<T4>(), TypeInit<T5>());
			current_level = 5;
		}
	}

	// Get value at index j, cast to type T, from whatever level is current
	template <typename T>
	T GetValueAs(int j) const {
		// Levels 3-5: only for integers
		if (!IS_FLOAT && MAXLEVEL >= 5 && current_level >= 5) {
			return static_cast<T>(extremes_level5[j]);
		}
		if (!IS_FLOAT && MAXLEVEL >= 4 && current_level >= 4) {
			return static_cast<T>(extremes_level4[j]);
		}
		if (!IS_FLOAT && MAXLEVEL >= 3 && current_level >= 3) {
			return static_cast<T>(extremes_level3[j]);
		}
		if (MAXLEVEL >= 2 && current_level >= 2) {
			return static_cast<T>(extremes_level2[j]);
		}
		return static_cast<T>(extremes_level1[j]);
	}

	void GetTotalsAsDouble(double *dst) const {
		if (!IS_FLOAT && MAXLEVEL >= 5 && current_level == 5) { // Level 5 (hugeint): linear layout
			for (int i = 0; i < 64; i++) {
				dst[i] = ToDouble(extremes_level5[i]);
			}
		} else if (!IS_FLOAT && MAXLEVEL >= 4 && current_level == 4) { // Level 4 (int64): linear layout
			for (int i = 0; i < 64; i++) {
				dst[i] = ToDouble(extremes_level4[i]);
			}
		} else if (!IS_FLOAT && MAXLEVEL >= 3 && current_level == 3) { // Level 3 (int32): SWAR layout
			ExtractSWAR<T3, 2>(extremes_level3, dst);
		} else if (MAXLEVEL >= 2 && current_level == 2) { // Level 2: SWAR for int16 (4 per u64), linear for double
			if (!IS_FLOAT) {                              // int16: SWAR layout, 4 per u64
				ExtractSWAR<T2, 4>(extremes_level2, dst);
			} else { // double: linear layout
				for (int i = 0; i < 64; i++) {
					dst[i] = static_cast<double>(extremes_level2[i]);
				}
			}
		} else if (current_level == 1) { // Level 1: SWAR for int8 (8 per u64), float (2 per u64)
			if (!IS_FLOAT) {             // int8: SWAR layout, 8 per u64
				ExtractSWAR<T1, 8>(extremes_level1, dst);
			} else { // float: SWAR layout, 2 per u64
				ExtractSWAR<T1, 2>(extremes_level1, dst);
			}
		} else { // Not initialized - return init values
			double init_val = ToDouble(TypeInit<TMAX>());
			for (int i = 0; i < 64; i++) {
				dst[i] = init_val;
			}
		}
	}

	// Upgrade level if value doesn't fit in current level
	void MaybeUpgrade(TMAX value) {
		if (current_level == 1 && !FitsInLevel1(value)) {
			UpgradeToLevel2();
		}
		if (!IS_FLOAT) { // Levels 3-5: only for integers
			if (MAXLEVEL >= 3 && current_level == 2 && !FitsInLevel2(value)) {
				UpgradeToLevel3();
			}
			if (MAXLEVEL >= 4 && current_level == 3 && !FitsInLevel3(value)) {
				UpgradeToLevel4();
			}
			if (MAXLEVEL >= 5 && current_level == 4 && !FitsInLevel4(value)) {
				UpgradeToLevel5();
			}
		}
	}

	// Update extremes at current level using SIMD-friendly implementation
	AUTOVECTORIZE void UpdateAtCurrentLevel(uint64_t key_hash, TMAX value) {
		if (current_level == 1) {
			UpdateExtremes<T1, IS_MAX>(extremes_level1, key_hash, static_cast<T1>(value));
		} else if (MAXLEVEL >= 2 && current_level == 2) {
			UpdateExtremes<T2, IS_MAX>(extremes_level2, key_hash, static_cast<T2>(value));
		} else if (!IS_FLOAT && MAXLEVEL >= 3 && current_level == 3) {
			UpdateExtremes<T3, IS_MAX>(extremes_level3, key_hash, static_cast<T3>(value));
		} else if (!IS_FLOAT && MAXLEVEL >= 4 && current_level == 4) {
			UpdateExtremes<T4, IS_MAX>(extremes_level4, key_hash, static_cast<T4>(value));
		} else if (!IS_FLOAT && MAXLEVEL >= 5 && current_level == 5) {
			UpdateExtremes<T5, IS_MAX>(extremes_level5, key_hash, static_cast<T5>(value));
		}
	}

	// Recompute global bound from current level's extremes - periodically called after updates
	void RecomputeBound() {
#ifndef PAC_MINMAX_NOBOUNDOPT
		if (++update_count == BOUND_RECOMPUTE_INTERVAL) {
			update_count = 0;
			if (current_level == 1) {
				global_bound = ComputeGlobalBound<T1, TMAX, IS_MAX>(extremes_level1);
			} else if (MAXLEVEL >= 2 && current_level == 2) {
				global_bound = ComputeGlobalBound<T2, TMAX, IS_MAX>(extremes_level2);
			} else if (!IS_FLOAT && MAXLEVEL >= 3 && current_level == 3) {
				global_bound = ComputeGlobalBound<T3, TMAX, IS_MAX>(extremes_level3);
			} else if (!IS_FLOAT && MAXLEVEL >= 4 && current_level == 4) {
				global_bound = ComputeGlobalBound<T4, TMAX, IS_MAX>(extremes_level4);
			} else if (!IS_FLOAT && MAXLEVEL >= 5 && current_level == 5) {
				global_bound = ComputeGlobalBound<T5, TMAX, IS_MAX>(extremes_level5);
			}
		}
#endif
	}

	// Combine with another state (merge src into this)
	void CombineWith(const PacMinMaxState &src, ArenaAllocator &alloc) {
		if (!src.initialized) {
			return;
		}
		if (!initialized) {
			allocator = &alloc;
			extremes_level1 = AllocateLevel<T1>(TypeInit<T1>());
			current_level = 1;
			update_count = 0;
			global_bound = TypeInit<TMAX>();
			initialized = true;
		}
		// Upgrade this state to match src level
		if (MAXLEVEL >= 2 && current_level == 1 && current_level < src.current_level) {
			UpgradeToLevel2();
		}
		if (!IS_FLOAT && MAXLEVEL >= 3 && current_level == 2 && current_level < src.current_level) {
			UpgradeToLevel3();
		}
		if (!IS_FLOAT && MAXLEVEL >= 4 && current_level == 3 && current_level < src.current_level) {
			UpgradeToLevel4();
		}
		if (!IS_FLOAT && MAXLEVEL >= 5 && current_level == 4 && current_level < src.current_level) {
			UpgradeToLevel5();
		}
		// Combine at current level
		if (current_level == 1) {
			for (int j = 0; j < 64; j++) {
				extremes_level1[j] = PAC_BETTER(extremes_level1[j], src.template GetValueAs<T1>(j));
			}
		} else if (MAXLEVEL >= 2 && current_level == 2) {
			for (int j = 0; j < 64; j++) {
				extremes_level2[j] = PAC_BETTER(extremes_level2[j], src.template GetValueAs<T2>(j));
			}
		} else if (!IS_FLOAT && MAXLEVEL >= 3 && current_level == 3) {
			for (int j = 0; j < 64; j++) {
				extremes_level3[j] = PAC_BETTER(extremes_level3[j], src.template GetValueAs<T3>(j));
			}
		} else if (!IS_FLOAT && MAXLEVEL >= 4 && current_level == 4) {
			for (int j = 0; j < 64; j++) {
				extremes_level4[j] = PAC_BETTER(extremes_level4[j], src.template GetValueAs<T4>(j));
			}
		} else if (!IS_FLOAT && MAXLEVEL >= 5 && current_level == 5) {
			for (int j = 0; j < 64; j++) {
				extremes_level5[j] = PAC_BETTER(extremes_level5[j], src.template GetValueAs<T5>(j));
			}
		}
		RecomputeBound();
	}
#endif

	// State size for DuckDB allocation - uses offsetof to exclude unused pointer fields
	static idx_t StateSize() {
#ifdef PAC_MINMAX_NONCASCADING
		return sizeof(PacMinMaxState);
#else
		if (MAXLEVEL >= 5) {
			return sizeof(PacMinMaxState);
		} else if (MAXLEVEL >= 4) {
			return offsetof(PacMinMaxState, extremes_level5);
		} else if (MAXLEVEL >= 3) {
			return offsetof(PacMinMaxState, extremes_level4);
		} else if (MAXLEVEL >= 2) {
			return offsetof(PacMinMaxState, extremes_level3);
		}
		return offsetof(PacMinMaxState, extremes_level2);
#endif
	}
};

} // namespace duckdb

#endif // PAC_MIN_MAX_HPP
