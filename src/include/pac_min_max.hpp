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
// Templated helper for updating extremes at a given level
// ============================================================================
// T: element type of the extremes array (int8_t, int16_t, ..., float, double)
// BOUND_T: type for the global bound (T64 for integers, double for floats)
// IS_MAX: true for pac_max, false for pac_min
//
// Returns the new global bound (worst of all extremes)
template <typename T, typename BOUND_T, bool IS_MAX>
AUTOVECTORIZE static inline BOUND_T UpdateExtremesAtLevel(T *extremes, uint64_t key_hash, T value) {
	BOUND_T new_bound = static_cast<BOUND_T>(extremes[0]);
	for (int j = 0; j < 64; j++) {
		if ((key_hash >> j) & 1ULL) {
			if constexpr (IS_MAX) {
				extremes[j] = (value > extremes[j]) ? value : extremes[j];
			} else {
				extremes[j] = (value < extremes[j]) ? value : extremes[j];
			}
		}
		BOUND_T ext = static_cast<BOUND_T>(extremes[j]);
		if constexpr (IS_MAX) {
			new_bound = (ext < new_bound) ? ext : new_bound; // Worse = min for MAX
		} else {
			new_bound = (ext > new_bound) ? ext : new_bound; // Worse = max for MIN
		}
	}
	return new_bound;
}

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
// Define PAC_MINMAX_NONCASCADING to use fixed-width arrays (input value type).
// Define PAC_MINMAX_NONLAZY to pre-allocate all levels at initialization.

//#define PAC_MINMAX_NONCASCADING 1
//#define PAC_MINMAX_NONLAZY 1

// Templated integer state for min/max
// SIGNED: whether using signed types
// IS_MAX: true for pac_max, false for pac_min
template <bool SIGNED, bool IS_MAX>
struct PacMinMaxIntState {
	// Type aliases based on signedness
	typedef typename std::conditional<SIGNED, int8_t, uint8_t>::type T8;
	typedef typename std::conditional<SIGNED, int16_t, uint16_t>::type T16;
	typedef typename std::conditional<SIGNED, int32_t, uint32_t>::type T32;
	typedef typename std::conditional<SIGNED, int64_t, uint64_t>::type T64;

	// Limits for each type
	static constexpr T8 T8_INIT = IS_MAX ? (SIGNED ? INT8_MIN : 0) : (SIGNED ? INT8_MAX : UINT8_MAX);
	static constexpr T16 T16_INIT = IS_MAX ? (SIGNED ? INT16_MIN : 0) : (SIGNED ? INT16_MAX : UINT16_MAX);
	static constexpr T32 T32_INIT = IS_MAX ? (SIGNED ? INT32_MIN : 0) : (SIGNED ? INT32_MAX : UINT32_MAX);
	static constexpr T64 T64_INIT = IS_MAX ? (SIGNED ? INT64_MIN : 0) : (SIGNED ? INT64_MAX : UINT64_MAX);

	// Check if value fits in a given type
	static inline bool FitsIn8(T64 val) {
		return val >= (SIGNED ? INT8_MIN : 0) && val <= (SIGNED ? INT8_MAX : UINT8_MAX);
	}
	static inline bool FitsIn16(T64 val) {
		return val >= (SIGNED ? INT16_MIN : 0) && val <= (SIGNED ? INT16_MAX : UINT16_MAX);
	}
	static inline bool FitsIn32(T64 val) {
		return val >= (SIGNED ? INT32_MIN : 0) && val <= (SIGNED ? INT32_MAX : UINT32_MAX);
	}

	// Compare for better extreme (max: greater is better, min: smaller is better)
	static inline bool IsBetter(T64 a, T64 b) {
		return IS_MAX ? (a > b) : (a < b);
	}
	static inline T64 Better(T64 a, T64 b) {
		return IS_MAX ? (a > b ? a : b) : (a < b ? a : b);
	}
	static inline T64 Worse(T64 a, T64 b) {
		return IS_MAX ? (a < b ? a : b) : (a > b ? a : b);
	}

#ifdef PAC_MINMAX_NONCASCADING
	T64 extremes[64];
	T64 global_bound; // For MAX: min of all maxes; for MIN: max of all mins
	bool initialized;
	bool seen_null;

	void Flush() {
	} // no-op

	void GetTotalsAsDouble(double *dst) const {
		for (int i = 0; i < 64; i++) {
			dst[i] = static_cast<double>(extremes[i]);
		}
	}

	void Initialize() {
		for (int i = 0; i < 64; i++) {
			extremes[i] = T64_INIT;
		}
		global_bound = T64_INIT;
		initialized = true;
	}
#else
	ArenaAllocator *allocator;

	// Lazily allocated levels (only one is active at a time for the 64 extremes)
	T8 *extremes8;
	T16 *extremes16;
	T32 *extremes32;
	T64 *extremes64;

	uint8_t current_level; // 8, 16, 32, or 64 (0 means not yet initialized)
	T64 global_bound;      // Always stored in widest type
	bool seen_null;

	// Allocate a level's buffer
	template <typename T>
	inline T *AllocateLevel(T init_value) {
		T *buf = reinterpret_cast<T *>(allocator->Allocate(64 * sizeof(T)));
		for (int i = 0; i < 64; i++) {
			buf[i] = init_value;
		}
		return buf;
	}

	// Upgrade from one level to the next, copying values
	template <typename SRC_T, typename DST_T>
	inline DST_T *UpgradeLevel(SRC_T *src, DST_T init_value) {
		DST_T *dst = reinterpret_cast<DST_T *>(allocator->Allocate(64 * sizeof(DST_T)));
		if (src) {
			for (int i = 0; i < 64; i++) {
				dst[i] = static_cast<DST_T>(src[i]);
			}
		} else {
			for (int i = 0; i < 64; i++) {
				dst[i] = init_value;
			}
		}
		return dst;
	}

	void EnsureLevel8() {
		if (current_level == 0) {
			extremes8 = AllocateLevel<T8>(T8_INIT);
			current_level = 8;
			global_bound = T64_INIT;
		}
	}

	void UpgradeTo16() {
		if (current_level < 16) {
			extremes16 = UpgradeLevel<T8, T16>(extremes8, T16_INIT);
			current_level = 16;
		}
	}

	void UpgradeTo32() {
		if (current_level < 32) {
			if (current_level == 8) {
				UpgradeTo16();
			}
			extremes32 = UpgradeLevel<T16, T32>(extremes16, T32_INIT);
			current_level = 32;
		}
	}

	void UpgradeTo64() {
		if (current_level < 64) {
			if (current_level < 32) {
				UpgradeTo32();
			}
			extremes64 = UpgradeLevel<T32, T64>(extremes32, T64_INIT);
			current_level = 64;
		}
	}

	void Flush() {
	} // no-op for min/max

	void GetTotalsAsDouble(double *dst) const {
		switch (current_level) {
		case 64:
			for (int i = 0; i < 64; i++) {
				dst[i] = static_cast<double>(extremes64[i]);
			}
			break;
		case 32:
			for (int i = 0; i < 64; i++) {
				dst[i] = static_cast<double>(extremes32[i]);
			}
			break;
		case 16:
			for (int i = 0; i < 64; i++) {
				dst[i] = static_cast<double>(extremes16[i]);
			}
			break;
		case 8:
			for (int i = 0; i < 64; i++) {
				dst[i] = static_cast<double>(extremes8[i]);
			}
			break;
		default:
			// Not initialized - return init values
			for (int i = 0; i < 64; i++) {
				dst[i] = static_cast<double>(T64_INIT);
			}
			break;
		}
	}

	// Pre-allocate all levels (for NONLAZY mode)
	void InitializeAllLevels(ArenaAllocator &alloc) {
		allocator = &alloc;
		extremes8 = AllocateLevel<T8>(T8_INIT);
		extremes16 = AllocateLevel<T16>(T16_INIT);
		extremes32 = AllocateLevel<T32>(T32_INIT);
		extremes64 = AllocateLevel<T64>(T64_INIT);
		current_level = 64; // Use widest level
		global_bound = T64_INIT;
	}
#endif
};

// Double state for min/max (floating point values)
// Cascading: start with float if values fit in [-1000000, 1000000], upgrade to double otherwise
template <bool IS_MAX>
struct PacMinMaxDoubleState {
	static constexpr float FLOAT_RANGE_MIN = -1000000.0f;
	static constexpr float FLOAT_RANGE_MAX = 1000000.0f;
	static constexpr float FLOAT_INIT =
	    IS_MAX ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
	static constexpr double DOUBLE_INIT =
	    IS_MAX ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();

	static inline bool IsBetter(double a, double b) {
		return IS_MAX ? (a > b) : (a < b);
	}
	static inline double Better(double a, double b) {
		return IS_MAX ? (a > b ? a : b) : (a < b ? a : b);
	}
	static inline double Worse(double a, double b) {
		return IS_MAX ? (a < b ? a : b) : (a > b ? a : b);
	}

	// Check if value fits in float range
	static inline bool FitsInFloat(double val) {
		return val >= FLOAT_RANGE_MIN && val <= FLOAT_RANGE_MAX;
	}

#ifdef PAC_MINMAX_NONCASCADING
	double extremes[64];
	double global_bound;
	bool initialized;
	bool seen_null;

	void Flush() {
	} // no-op

	void GetTotalsAsDouble(double *dst) const {
		for (int i = 0; i < 64; i++) {
			dst[i] = extremes[i];
		}
	}

	void Initialize() {
		for (int i = 0; i < 64; i++) {
			extremes[i] = DOUBLE_INIT;
		}
		global_bound = DOUBLE_INIT;
		initialized = true;
	}
#else
	ArenaAllocator *allocator;

	// Lazily allocated levels
	float *extremesF;  // 32-bit float level
	double *extremesD; // 64-bit double level

	uint8_t current_level; // 32 for float, 64 for double, 0 = not initialized
	double global_bound;   // Always stored as double
	bool seen_null;

	// Allocate float level
	inline float *AllocateFloatLevel() {
		float *buf = reinterpret_cast<float *>(allocator->Allocate(64 * sizeof(float)));
		for (int i = 0; i < 64; i++) {
			buf[i] = FLOAT_INIT;
		}
		return buf;
	}

	// Allocate double level
	inline double *AllocateDoubleLevel() {
		double *buf = reinterpret_cast<double *>(allocator->Allocate(64 * sizeof(double)));
		for (int i = 0; i < 64; i++) {
			buf[i] = DOUBLE_INIT;
		}
		return buf;
	}

	// Upgrade from float to double
	inline double *UpgradeToDouble() {
		double *dst = reinterpret_cast<double *>(allocator->Allocate(64 * sizeof(double)));
		if (extremesF) {
			for (int i = 0; i < 64; i++) {
				dst[i] = static_cast<double>(extremesF[i]);
			}
		} else {
			for (int i = 0; i < 64; i++) {
				dst[i] = DOUBLE_INIT;
			}
		}
		return dst;
	}

	void EnsureLevelFloat() {
		if (current_level == 0) {
			extremesF = AllocateFloatLevel();
			current_level = 32;
			global_bound = DOUBLE_INIT;
		}
	}

	void UpgradeTo64() {
		if (current_level < 64) {
			extremesD = UpgradeToDouble();
			current_level = 64;
		}
	}

	void Flush() {
	} // no-op for min/max

	void GetTotalsAsDouble(double *dst) const {
		switch (current_level) {
		case 64:
			for (int i = 0; i < 64; i++) {
				dst[i] = extremesD[i];
			}
			break;
		case 32:
			for (int i = 0; i < 64; i++) {
				dst[i] = static_cast<double>(extremesF[i]);
			}
			break;
		default:
			// Not initialized - return init values
			for (int i = 0; i < 64; i++) {
				dst[i] = DOUBLE_INIT;
			}
			break;
		}
	}

	// Pre-allocate all levels (for NONLAZY mode)
	void InitializeAllLevels(ArenaAllocator &alloc) {
		allocator = &alloc;
		extremesF = AllocateFloatLevel();
		extremesD = AllocateDoubleLevel();
		current_level = 64; // Use widest level
		global_bound = DOUBLE_INIT;
	}
#endif
};

} // namespace duckdb

#endif // PAC_MIN_MAX_HPP
