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
// Cascading: be prepared to keep states at different granularities:
//            int: 8,16,32,64,128 (but never bigger than the input type)
//            floating-point: float,double (but in case of float, just float)
// then, start keeping Min/Max in the smallest type, upgrade when value doesn't fit.
// we lazily allocate the various arrays of a certain width, so if we do not need
// huge aggregates, then we never allocate them (memory saving).
//
// when upgrading the size (e.g. 16->32-bits) we want to keep using the memory used for 16-bits
// therefore, memory is organized in "banks"of the size needed for 8-bits, and when upgrading
// we progressivly allocate more banks, re-using the old ones also for the bigger size;
//
// Cascading/banking is only implemented for integers.
// Floating-point directly aggregate in their own datatyue (float,double): double does not
// attempt to use float for small values. This is done to make the code less complex.
//
// some #defines to demonstrate the effects of our optimizations:
// Define PAC_MINMAX_NONBANKED to use fixed-width contiguous arrays (no bank splitting).
// Define PAC_MINMAX_NONLAZY to pre-allocate all levels at initialization (and suffer the consequences)
// Define PAC_MINMAX_NOBOUNDOP to not compute a global bound that allows early-out often (will be slower likely)

//#define PAC_MINMAX_NONBANKED 1
//#define PAC_MINMAX_NONLAZY 1
//#define PAC_MINMAX_NOBOUNDOPT 1

// NULL handling: by default we ignore NULLs (safe behavior, like DuckDB's MIN/MAX).
// Define PAC_MINMAX_UNSAFENULL to return NULL if any input value is NULL.
//#define PAC_MINMAX_UNSAFENULL 1

// Recompute global_bound every N updates (reduces overhead of bound computation)
static constexpr uint16_t BOUND_RECOMPUTE_INTERVAL = 2048;

// Type-agnostic comparison macros (IS_MAX must be in scope)
#define PAC_IS_BETTER(a, b) (IS_MAX ? ((a) > (b)) : ((a) < (b)))
#define PAC_BETTER(a, b)    (PAC_IS_BETTER(a, b) ? (a) : (b))
#define PAC_WORSE(a, b)     (PAC_IS_BETTER(a, b) ? (b) : (a))

// ============================================================================
// SIMD-friendly update functions for Min/Max extremes arrays
// Uses union-based bitwise ops that work for all types (int and float)
// prototyped here: https://godbolt.org/z/Tbq3jWWY6
// ============================================================================

// Unified SIMD Min/Max update kernel that works for all integers <=64 and double,float even
// Updates `count` elements in result[], using bits [start_bit..start_bit+count) from key_hash
template <typename T, bool IS_MAX, int SIZE = sizeof(T)>
struct UpdateExtremesKernel {
	// Type aliases (signed/unsigned variants of SIZE-byte integers)
	using BitsT = typename std::conditional<
	    SIZE == 1, int8_t,
	    typename std::conditional<SIZE == 2, int16_t,
	                              typename std::conditional<SIZE == 4, int32_t, int64_t>::type>::type>::type;
	using UintT = typename std::conditional<
	    SIZE == 1, uint8_t,
	    typename std::conditional<SIZE == 2, uint16_t,
	                              typename std::conditional<SIZE == 4, uint32_t, uint64_t>::type>::type>::type;

	// SWAR constants computed from SIZE
	static constexpr int SHIFTS = SIZE * 8; // 8, 16, 32, 64
	static constexpr uint64_t MASK = SIZE == 1   ? 0x0101010101010101ULL
	                                 : SIZE == 2 ? 0x0001000100010001ULL
	                                 : SIZE == 4 ? 0x0000000100000001ULL
	                                             : 1ULL;

	AUTOVECTORIZE static inline void update(T *__restrict__ result, uint64_t key_hash, T value, int start_bit,
	                                        int count) {
		union {
			uint64_t u64[SHIFTS]; // keyhash as uint64
			BitsT bits[64];       // signed int type as wide as T (we want 0x00 - 1 to become 0xFF)
		} buf;
		for (int i = 0; i < SHIFTS; i++) {
			buf.u64[i] = (key_hash >> i) & MASK; // process multiple T in one uint64 (SWAR)
		}
		for (int i = 0; i < count; i++) {
			int bit_idx = start_bit + i;
			UintT mask = static_cast<UintT>(-buf.bits[bit_idx]); // 1->0x00, 0->0xFF
			union { // this union is there to be able to combine integer masking operations with floating-point min/max
				T val;      // could be float or double
				UintT bits; // unsigned int type as wide as T
			} extreme_u, result_u, out;
			extreme_u.val = PAC_BETTER(value, result[i]); // SIMD min/max
			result_u.val = result[i];
			out.bits = (extreme_u.bits & mask) | (result_u.bits & ~mask);
			result[i] = out.val;
		}
	}
};

// Specialization for uint8_t MAX: turns out to be faster to mask value first (only possible for unsigned)
template <>
struct UpdateExtremesKernel<uint8_t, true, 1> {
	AUTOVECTORIZE static inline void update(uint8_t *__restrict__ result, uint64_t key_hash, uint8_t value,
	                                        int start_bit, int count) {
		uint64_t buf[8];
		for (int i = 0; i < 8; i++) {
			buf[i] = (key_hash >> i) & 0x0101010101010101ULL;
		}
		int8_t *__restrict__ bits = reinterpret_cast<int8_t *>(buf);
		for (int i = 0; i < count; i++) {
			int bit_idx = start_bit + i;
			uint8_t mask = static_cast<uint8_t>(-bits[bit_idx]);
			result[i] = std::max(static_cast<uint8_t>(value & mask), result[i]);
		}
	}
};

// Specialization for [u]hugeint - scalar comparison, no bitwise tricks, use IF..THEN to avoid slow hugeint calculations
template <typename T, bool IS_MAX>
struct UpdateExtremesKernel<T, IS_MAX, 16> {
	AUTOVECTORIZE static inline void update(T *__restrict__ result, uint64_t key_hash, T value, int start_bit,
	                                        int count) {
		for (int i = 0; i < count; i++) {
			int bit_idx = start_bit + i;
			if (((key_hash >> bit_idx) & 1ULL) && PAC_IS_BETTER(value, result[i])) {
				result[i] = value;
			}
		}
	}
};

// Unified update function - banked or non-banked depending on compile flag
#ifdef PAC_MINMAX_NONBANKED
// Non-banked: single contiguous array
template <typename T, bool IS_MAX>
AUTOVECTORIZE static inline void UpdateExtremes(T *__restrict__ result, uint64_t key_hash, T value) {
	UpdateExtremesKernel<T, IS_MAX>::update(result, key_hash, value, 0, 64);
}
#else
// Banked: iterate over banks, calling kernel once per bank
template <typename T, bool IS_MAX>
AUTOVECTORIZE static inline void UpdateExtremes(uint8_t **banks, uint64_t key_hash, T value) {
	constexpr int ELEMS_PER_BANK = 64 / sizeof(T);
	for (int b = 0; b < static_cast<int>(sizeof(T)); b++) {
		T *bank_data = reinterpret_cast<T *>(banks[b]);
		UpdateExtremesKernel<T, IS_MAX>::update(bank_data, key_hash, value, b * ELEMS_PER_BANK, ELEMS_PER_BANK);
	}
}
#endif

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
// Simple state for float/double - no cascading, no banks
// T = float (256 bytes) or double (512 bytes)
// ============================================================================
template <typename T, bool IS_MAX>
struct PacMinMaxFloatingState {
	T global_bound;
	uint16_t update_count;
	bool initialized;
#ifdef PAC_MINMAX_UNSAFENULL
	bool seen_null;
#endif
	T extremes[64]; // 256 bytes for float, 512 bytes for double

	using TMAX = T;
	using ValueType = T;

	static inline T TypeInit() {
		return IS_MAX ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
	}

	// Returns the init value as double (for detecting never-updated counters)
	static double InitAsDouble() {
		return static_cast<double>(TypeInit());
	}

	void Initialize() {
		for (int i = 0; i < 64; i++) {
			extremes[i] = TypeInit();
		}
		global_bound = TypeInit();
		update_count = 0;
		initialized = true;
	}

	void GetTotalsAsDouble(double *dst) const {
		for (int i = 0; i < 64; i++) {
			dst[i] = static_cast<double>(extremes[i]);
		}
	}

	void RecomputeBound() {
#ifndef PAC_MINMAX_NOBOUNDOPT
		if (++update_count == BOUND_RECOMPUTE_INTERVAL) {
			update_count = 0;
			global_bound = ComputeGlobalBound<T, T, IS_MAX>(extremes);
		}
#endif
	}

	// Interface methods for compatibility with PacMinMaxUpdateOne template
	void AllocateFirstLevel(ArenaAllocator &) {
		if (!initialized) {
			Initialize();
		}
	}

	void MaybeUpgrade(ArenaAllocator &, T) {
		// No cascading for floating-point state
	}

	AUTOVECTORIZE void UpdateAtCurrentLevel(uint64_t key_hash, T value) {
		UpdateExtremesKernel<T, IS_MAX>::update(extremes, key_hash, value, 0, 64);
	}

	// Combine with another state (allocator unused, present for interface compatibility)
	void CombineWith(const PacMinMaxFloatingState &src, ArenaAllocator &) {
		if (!src.initialized) {
			return;
		}
		if (!initialized) {
			Initialize();
		}
		for (int i = 0; i < 64; i++) {
			extremes[i] = PAC_BETTER(extremes[i], src.extremes[i]);
		}
		RecomputeBound();
	}

	static idx_t StateSize() {
		return sizeof(PacMinMaxFloatingState);
	}
};

// ============================================================================
// Unified state for min/max (integer types only)
// ============================================================================
// SIGNED: signed vs unsigned
// IS_MAX: true for pac_max, false for pac_min
// MAXLEVEL: maximum cascading level (8, 16, 32, 64, or 128 bits)
//
// Level abstraction (by bit width):
//   Level  8: int8/uint8       ( 1 byte,  uses SWAR)
//   Level 16: int16/uint16     ( 2 bytes, uses SWAR)
//   Level 32: int32/uint32     ( 4 bytes, uses SWAR)
//   Level 64: int64/uint64     ( 8 bytes, linear)
//   Level 128: hugeint/uhugeint (16 bytes, linear)
//
// All pointer fields are always defined but ordered by level at the end of the struct.
// We use offsetof() to report smaller state_size to DuckDB, so it allocates only
// the memory needed for the MAXLEVEL. Regular if guards (not if constexpr) prevent
// access to unallocated fields - compiler optimizes these away since MAXLEVEL is constant.
template <bool SIGNED, bool IS_MAX, int MAXLEVEL>
struct PacMinMaxState {
	// Constants
	static constexpr int MINLEVEL = 8;

	// Type aliases for integers by bit width
	template <bool S>
	using SignedInt8 = typename std::conditional<S, int8_t, uint8_t>::type;
	template <bool S>
	using SignedInt16 = typename std::conditional<S, int16_t, uint16_t>::type;
	template <bool S>
	using SignedInt32 = typename std::conditional<S, int32_t, uint32_t>::type;
	template <bool S>
	using SignedInt64 = typename std::conditional<S, int64_t, uint64_t>::type;
	template <bool S>
	using SignedInt128 = typename std::conditional<S, hugeint_t, uhugeint_t>::type;

	using T8 = SignedInt8<SIGNED>;
	using T16 = SignedInt16<SIGNED>;
	using T32 = SignedInt32<SIGNED>;
	using T64 = SignedInt64<SIGNED>;
	using T128 = SignedInt128<SIGNED>;

	// TMAX: type at MAXLEVEL (helper to select type by bit width)
	template <int L>
	using TAtLevel = typename std::conditional<
	    L == 128, T128,
	    typename std::conditional<
	        L == 64, T64,
	        typename std::conditional<L == 32, T32, typename std::conditional<L == 16, T16, T8>::type>::type>::type>::
	    type;
	using TMAX = TAtLevel<MAXLEVEL>;
	using ValueType = TMAX;

	// Get init value (worst possible for the aggregation direction)
	template <typename T>
	static inline T TypeInit() {
		return IS_MAX ? NumericLimits<T>::Minimum() : NumericLimits<T>::Maximum();
	}

	// Returns the init value as double (for detecting never-updated counters)
	// Counters start at level 8, so we use T8's init value
	static double InitAsDouble() {
		return ToDouble(TypeInit<T8>());
	}

	// Check if value fits at level 8 (int8/uint8)
	static inline bool FitsInLevel8(TMAX val) {
		return val >= static_cast<TMAX>(SIGNED ? INT8_MIN : 0) &&
		       val <= static_cast<TMAX>(SIGNED ? INT8_MAX : UINT8_MAX);
	}

	// Check if value fits at level 16 (int16/uint16)
	static inline bool FitsInLevel16(TMAX val) {
		return val >= static_cast<TMAX>(SIGNED ? INT16_MIN : 0) &&
		       val <= static_cast<TMAX>(SIGNED ? INT16_MAX : UINT16_MAX);
	}

	// Check if value fits at level 32 (int32/uint32)
	static inline bool FitsInLevel32(TMAX val) {
		return val >= static_cast<TMAX>(SIGNED ? INT32_MIN : 0) &&
		       val <= static_cast<TMAX>(SIGNED ? INT32_MAX : UINT32_MAX);
	}

	// Check if value fits at level 64 (int64/uint64)
	static inline bool FitsInLevel64(TMAX val) {
		return val >= static_cast<TMAX>(SIGNED ? INT64_MIN : 0) &&
		       val <= static_cast<TMAX>(SIGNED ? INT64_MAX : UINT64_MAX);
	}

	// ========== Common fields (defined once for both modes) ==========
	TMAX global_bound; // For MAX: min of all maxes; for MIN: max of all mins
	uint16_t update_count;
	bool initialized;
#ifdef PAC_MINMAX_UNSAFENULL
	bool seen_null;
#endif
#ifdef PAC_MINMAX_NONBANKED
	// ========== Non-banked mode: single fixed-width contiguous array ==========
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
	// ========== Cascading mode with bank-based memory layout ==========
	// Banks are 64-byte aligned chunks (one AVX-512 register each).
	// Memory is allocated incrementally to avoid waste when upgrading levels.
	//
	// Bank layout:
	//   bank0: 64 bytes inline (always present, for level 8)
	//   banks1: 1×64 bytes (allocated for level 16+)
	//   banks2: 2×64 bytes (allocated for level 32+)
	//   banks3: 4×64 bytes (allocated for level 64+)
	//   banks4: 8×64 bytes (allocated for level 128)
	//
	// Total banks per level:
	//   Level  8 (int8):         1 bank  =  64 bytes
	//   Level 16 (int16):        2 banks = 128 bytes
	//   Level 32 (int32):        4 banks = 256 bytes
	//   Level 64 (int64):        8 banks = 512 bytes
	//   Level 128 (hugeint):    16 banks = 1024 bytes

	uint8_t current_level; // 8, 16, 32, 64, or 128

	// Bank storage - inline bank0 + pointers to additional banks
	uint8_t bank0[64]; // Always present (level 8)
	uint8_t *banks1;   // 1 bank (64 bytes) for level 16+
	uint8_t *banks2;   // 2 banks (128 bytes) for level 32+
	uint8_t *banks3;   // 4 banks (256 bytes) for level 64+
	uint8_t *banks4;   // 8 banks (512 bytes) for level 128

	// Get pointer to bank n (0-15)
	inline uint8_t *GetBank(int n) const {
		if (n < 2) {
			return (n == 0) ? const_cast<uint8_t *>(bank0) : banks1;
		}
		if (n < 4) {
			return banks2 + (n - 2) * 64;
		}
		if (n < 8) {
			return banks3 + (n - 4) * 64;
		}
		return banks4 + (n - 8) * 64;
	}

	// Build array of bank pointers for use with UpdateExtremes
	template <typename T>
	inline void GetBankPointers(uint8_t *out[]) {
		constexpr int NUM_BANKS = sizeof(T);
		for (int i = 0; i < NUM_BANKS; i++) {
			out[i] = GetBank(i);
		}
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

	// Get element at linear index i from banked storage
	template <typename T>
	inline T GetElement(int i) const {
		int swar_idx = GetIndex<T>(i);
		int byte_offset = swar_idx * sizeof(T);
		int bank_idx = byte_offset / 64;
		int bank_offset = byte_offset % 64;
		return *reinterpret_cast<const T *>(GetBank(bank_idx) + bank_offset);
	}

	// Set element at linear index i in banked storage
	template <typename T>
	inline void SetElement(int i, T value) {
		int swar_idx = GetIndex<T>(i);
		int byte_offset = swar_idx * sizeof(T);
		int bank_idx = byte_offset / 64;
		int bank_offset = byte_offset % 64;
		*reinterpret_cast<T *>(GetBank(bank_idx) + bank_offset) = value;
	}

	// Initialize banks for a level with init value
	// Uses simple linear fill since all elements get the same value
	template <typename T>
	inline void InitializeBanks(T init_value) {
		constexpr int NUM_BANKS = sizeof(T);
		constexpr int ELEMS_PER_BANK = 64 / sizeof(T);
		for (int b = 0; b < NUM_BANKS; b++) {
			T *bank_data = reinterpret_cast<T *>(GetBank(b));
			for (int i = 0; i < ELEMS_PER_BANK; i++) {
				bank_data[i] = init_value;
			}
		}
	}

	// Allocate banks based on number of banks needed (only allocates new banks)
	// num_banks: 1,2,4,8,16 depending on element type size
	inline void AllocateBanksForCount(ArenaAllocator &allocator, int num_banks) {
		if (num_banks >= 2 && !banks1) {
			banks1 = reinterpret_cast<uint8_t *>(allocator.Allocate(64));
		}
		if (num_banks >= 4 && !banks2) {
			banks2 = reinterpret_cast<uint8_t *>(allocator.Allocate(128));
		}
		if (num_banks >= 8 && !banks3) {
			banks3 = reinterpret_cast<uint8_t *>(allocator.Allocate(256));
		}
		if (num_banks >= 16 && !banks4) {
			banks4 = reinterpret_cast<uint8_t *>(allocator.Allocate(512));
		}
	}

	// Allocate banks for type T
	template <typename T>
	inline void AllocateBanksForType(ArenaAllocator &allocator) {
		AllocateBanksForCount(allocator, sizeof(T));
	}

	// Upgrade from one level to the next, transforming data in-place across banks
	// SRC_T: source element type, DST_T: destination element type
	// Uses simple gather/scatter since kernel maintains its own layout
	template <typename SRC_T, typename DST_T>
	inline void UpgradeLevelImpl(ArenaAllocator &allocator, SRC_T src_init, DST_T dst_init) {
		constexpr int SRC_BANKS = sizeof(SRC_T);
		constexpr int SRC_PER_BANK = 64 / sizeof(SRC_T);

		// Gather source values (simple concatenation from banks)
		SRC_T src_values[64];
		for (int b = 0; b < SRC_BANKS; b++) {
			const SRC_T *bank_data = reinterpret_cast<const SRC_T *>(GetBank(b));
			for (int i = 0; i < SRC_PER_BANK; i++) {
				src_values[b * SRC_PER_BANK + i] = bank_data[i];
			}
		}

		// Allocate banks needed for destination type
		AllocateBanksForType<DST_T>(allocator);

		// Convert and scatter to destination banks
		constexpr int DST_BANKS = sizeof(DST_T);
		constexpr int DST_PER_BANK = 64 / sizeof(DST_T);
		for (int b = 0; b < DST_BANKS; b++) {
			DST_T *bank_data = reinterpret_cast<DST_T *>(GetBank(b));
			for (int i = 0; i < DST_PER_BANK; i++) {
				int src_idx = b * DST_PER_BANK + i;
				SRC_T src_val = (src_idx < 64) ? src_values[src_idx] : src_init;
				bank_data[i] = (src_val == src_init) ? dst_init : static_cast<DST_T>(src_val);
			}
		}
	}

	void AllocateFirstLevel(ArenaAllocator &allocator) {
#ifdef PAC_MINMAX_NONLAZY
		// Pre-allocate all levels that exist for this MAXLEVEL
		if (MAXLEVEL == 128) {
			AllocateBanksForType<T128>(allocator);
			InitializeBanks<T128>(TypeInit<T128>());
		} else if (MAXLEVEL == 64) {
			AllocateBanksForType<T64>(allocator);
			InitializeBanks<T64>(TypeInit<T64>());
		} else if (MAXLEVEL == 32) {
			AllocateBanksForType<T32>(allocator);
			InitializeBanks<T32>(TypeInit<T32>());
		} else if (MAXLEVEL == 16) {
			AllocateBanksForType<T16>(allocator);
			InitializeBanks<T16>(TypeInit<T16>());
		} else {
			AllocateBanksForType<T8>(allocator);
			InitializeBanks<T8>(TypeInit<T8>());
		}
		current_level = MAXLEVEL;
#else
		// Allocate banks needed for level 8 type (T8)
		// For int8: 1 bank (bank0 inline)
		AllocateBanksForType<T8>(allocator);
		InitializeBanks<T8>(TypeInit<T8>());
		current_level = 8;
#endif
		update_count = 0;
		global_bound = TypeInit<TMAX>();
		initialized = true;
	}

	void UpgradeToLevel16(ArenaAllocator &allocator) {
		if (MAXLEVEL >= 16) {
			UpgradeLevelImpl<T8, T16>(allocator, TypeInit<T8>(), TypeInit<T16>());
			current_level = 16;
		}
	}

	void UpgradeToLevel32(ArenaAllocator &allocator) {
		if (MAXLEVEL >= 32) {
			UpgradeLevelImpl<T16, T32>(allocator, TypeInit<T16>(), TypeInit<T32>());
			current_level = 32;
		}
	}

	void UpgradeToLevel64(ArenaAllocator &allocator) {
		if (MAXLEVEL >= 64) {
			UpgradeLevelImpl<T32, T64>(allocator, TypeInit<T32>(), TypeInit<T64>());
			current_level = 64;
		}
	}

	void UpgradeToLevel128(ArenaAllocator &allocator) {
		if (MAXLEVEL >= 128) {
			UpgradeLevelImpl<T64, T128>(allocator, TypeInit<T64>(), TypeInit<T128>());
			current_level = 128;
		}
	}

	// Helper to gather from banks to contiguous buffer (const version)
	template <typename T>
	void GatherToBuf(T *buffer) const {
		constexpr int NUM_BANKS = sizeof(T);
		constexpr int ELEMS_PER_BANK = 64 / sizeof(T);
		for (int b = 0; b < NUM_BANKS; b++) {
			const T *bank_data = reinterpret_cast<const T *>(GetBank(b));
			for (int i = 0; i < ELEMS_PER_BANK; i++) {
				buffer[b * ELEMS_PER_BANK + i] = bank_data[i];
			}
		}
	}

	// Get value at linear index j, cast to type T, from whatever level is current
	// This gathers to a temp buffer and extracts using SWAR conversion
	template <typename T>
	T GetValueAs(int j) const {
		if (MAXLEVEL >= 128 && current_level >= 128) {
			T128 buf[64];
			GatherToBuf<T128>(buf);
			return static_cast<T>(buf[j]); // hugeint is linear
		}
		if (MAXLEVEL >= 64 && current_level >= 64) {
			T64 buf[64];
			GatherToBuf<T64>(buf);
			return static_cast<T>(buf[j]); // int64 is linear
		}
		if (MAXLEVEL >= 32 && current_level >= 32) {
			T32 buf[64];
			GatherToBuf<T32>(buf);
			return static_cast<T>(buf[LinearToSWAR<2>(j)]); // int32 SWAR
		}
		if (MAXLEVEL >= 16 && current_level >= 16) {
			T16 buf[64];
			GatherToBuf<T16>(buf);
			return static_cast<T>(buf[LinearToSWAR<4>(j)]); // int16 SWAR
		}
		T8 buf[64];
		GatherToBuf<T8>(buf);
		return static_cast<T>(buf[LinearToSWAR<8>(j)]); // int8 SWAR
	}

	void GetTotalsAsDouble(double *dst) const {
		// Gather from banks and apply SWAR extraction based on type
		if (MAXLEVEL >= 128 && current_level == 128) {
			T128 buf[64];
			GatherToBuf<T128>(buf);
			for (int i = 0; i < 64; i++) {
				dst[i] = ToDouble(buf[i]); // hugeint is linear
			}
		} else if (MAXLEVEL >= 64 && current_level == 64) {
			T64 buf[64];
			GatherToBuf<T64>(buf);
			for (int i = 0; i < 64; i++) {
				dst[i] = ToDouble(buf[i]); // int64 is linear
			}
		} else if (MAXLEVEL >= 32 && current_level == 32) {
			T32 buf[64];
			GatherToBuf<T32>(buf);
			ExtractSWAR<T32, 2>(buf, dst); // int32: 2 per u64
		} else if (MAXLEVEL >= 16 && current_level == 16) {
			T16 buf[64];
			GatherToBuf<T16>(buf);
			ExtractSWAR<T16, 4>(buf, dst); // int16: 4 per u64
		} else if (current_level == 8) {
			T8 buf[64];
			GatherToBuf<T8>(buf);
			ExtractSWAR<T8, 8>(buf, dst); // int8: 8 per u64
		} else {                          // Not initialized - return init values
			double init_val = ToDouble(TypeInit<TMAX>());
			for (int i = 0; i < 64; i++) {
				dst[i] = init_val;
			}
		}
	}

	// Upgrade level if value doesn't fit in current level
	void MaybeUpgrade(ArenaAllocator &allocator, TMAX value) {
		if (current_level == 8 && !FitsInLevel8(value)) {
			UpgradeToLevel16(allocator);
		}
		if (MAXLEVEL >= 32 && current_level == 16 && !FitsInLevel16(value)) {
			UpgradeToLevel32(allocator);
		}
		if (MAXLEVEL >= 64 && current_level == 32 && !FitsInLevel32(value)) {
			UpgradeToLevel64(allocator);
		}
		if (MAXLEVEL >= 128 && current_level == 64 && !FitsInLevel64(value)) {
			UpgradeToLevel128(allocator);
		}
	}

	// Update extremes at current level using SIMD-friendly implementation
	AUTOVECTORIZE void UpdateAtCurrentLevel(uint64_t key_hash, TMAX value) {
		uint8_t *banks[16]; // Max banks needed (for hugeint)
		if (current_level == 8) {
			GetBankPointers<T8>(banks);
			UpdateExtremes<T8, IS_MAX>(banks, key_hash, static_cast<T8>(value));
		} else if (MAXLEVEL >= 16 && current_level == 16) {
			GetBankPointers<T16>(banks);
			UpdateExtremes<T16, IS_MAX>(banks, key_hash, static_cast<T16>(value));
		} else if (MAXLEVEL >= 32 && current_level == 32) {
			GetBankPointers<T32>(banks);
			UpdateExtremes<T32, IS_MAX>(banks, key_hash, static_cast<T32>(value));
		} else if (MAXLEVEL >= 64 && current_level == 64) {
			GetBankPointers<T64>(banks);
			UpdateExtremes<T64, IS_MAX>(banks, key_hash, static_cast<T64>(value));
		} else if (MAXLEVEL >= 128 && current_level == 128) {
			GetBankPointers<T128>(banks);
			UpdateExtremes<T128, IS_MAX>(banks, key_hash, static_cast<T128>(value));
		}
	}

	// Recompute global bound from current level's extremes - periodically called after updates
	void RecomputeBound() {
#ifndef PAC_MINMAX_NOBOUNDOPT
		if (++update_count == BOUND_RECOMPUTE_INTERVAL) {
			update_count = 0;
			TMAX bound = GetValueAs<TMAX>(0);
			for (int i = 1; i < 64; i++) {
				TMAX ext = GetValueAs<TMAX>(i);
				bound = PAC_WORSE(ext, bound);
			}
			global_bound = bound;
		}
#endif
	}

	// Combine with another state (merge src into this)
	void CombineWith(const PacMinMaxState &src, ArenaAllocator &allocator) {
		if (!src.initialized) {
			return;
		}
		if (!initialized) {
			AllocateBanksForType<T8>(allocator);
			InitializeBanks<T8>(TypeInit<T8>());
			current_level = 8;
			update_count = 0;
			global_bound = TypeInit<TMAX>();
			initialized = true;
		}
		// Upgrade this state to match src level
		if (MAXLEVEL >= 16 && current_level == 8 && current_level < src.current_level) {
			UpgradeToLevel16(allocator);
		}
		if (MAXLEVEL >= 32 && current_level == 16 && current_level < src.current_level) {
			UpgradeToLevel32(allocator);
		}
		if (MAXLEVEL >= 64 && current_level == 32 && current_level < src.current_level) {
			UpgradeToLevel64(allocator);
		}
		if (MAXLEVEL >= 128 && current_level == 64 && current_level < src.current_level) {
			UpgradeToLevel128(allocator);
		}
		// Combine at current level using element accessors
		if (current_level == 8) {
			for (int j = 0; j < 64; j++) {
				T8 mine = GetElement<T8>(j);
				T8 theirs = src.template GetValueAs<T8>(j);
				SetElement<T8>(j, PAC_BETTER(mine, theirs));
			}
		} else if (MAXLEVEL >= 16 && current_level == 16) {
			for (int j = 0; j < 64; j++) {
				T16 mine = GetElement<T16>(j);
				T16 theirs = src.template GetValueAs<T16>(j);
				SetElement<T16>(j, PAC_BETTER(mine, theirs));
			}
		} else if (MAXLEVEL >= 32 && current_level == 32) {
			for (int j = 0; j < 64; j++) {
				T32 mine = GetElement<T32>(j);
				T32 theirs = src.template GetValueAs<T32>(j);
				SetElement<T32>(j, PAC_BETTER(mine, theirs));
			}
		} else if (MAXLEVEL >= 64 && current_level == 64) {
			for (int j = 0; j < 64; j++) {
				T64 mine = GetElement<T64>(j);
				T64 theirs = src.template GetValueAs<T64>(j);
				SetElement<T64>(j, PAC_BETTER(mine, theirs));
			}
		} else if (MAXLEVEL >= 128 && current_level == 128) {
			for (int j = 0; j < 64; j++) {
				T128 mine = GetElement<T128>(j);
				T128 theirs = src.template GetValueAs<T128>(j);
				SetElement<T128>(j, PAC_BETTER(mine, theirs));
			}
		}
		RecomputeBound();
	}
#endif

	// State size for DuckDB allocation - uses offsetof to exclude unused pointer fields
	// State size is based on max banks needed for TMAX (the largest type at MAXLEVEL):
	//   1 bank:   only bank0 (inline)
	//   2 banks:  need banks1 pointer
	//   4 banks:  need banks2 pointer
	//   8 banks:  need banks3 pointer
	//   16 banks: need banks4 pointer
	static idx_t StateSize() {
#ifdef PAC_MINMAX_NONBANKED
		return sizeof(PacMinMaxState);
#else
		// Compute max banks based on type at MAXLEVEL
		constexpr int MAX_BANKS = sizeof(TMAX);
		if (MAX_BANKS >= 16) {
			return sizeof(PacMinMaxState);
		} else if (MAX_BANKS >= 8) {
			return offsetof(PacMinMaxState, banks4);
		} else if (MAX_BANKS >= 4) {
			return offsetof(PacMinMaxState, banks3);
		} else if (MAX_BANKS >= 2) {
			return offsetof(PacMinMaxState, banks2);
		}
		return offsetof(PacMinMaxState, banks1);
#endif
	}
};

} // namespace duckdb

#endif // PAC_MIN_MAX_HPP
