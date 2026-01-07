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
// Define PAC_MINMAX_NOBOUNDOPT to disable global bound optimization.

//#define PAC_MINMAX_NOBOUNDOPT 1

static constexpr uint16_t BOUND_RECOMPUTE_INTERVAL = 2048;

// Comparison macros (IS_MAX must be a template parameter in scope)
#define PAC_IS_BETTER(a, b) (IS_MAX ? ((a) > (b)) : ((a) < (b)))
#define PAC_BETTER(a, b)    (PAC_IS_BETTER(a, b) ? (a) : (b))
#define PAC_WORSE(a, b)     (PAC_IS_BETTER(a, b) ? (b) : (a))

// ============================================================================
// Update kernels - unified SWAR template for all type sizes
// ============================================================================

// SWAR update for signed types and floats (bit-select approach)
// Works for all sizes: 8-bit (SHIFTS=8), 16-bit (SHIFTS=16), 32-bit (SHIFTS=32), 64-bit (SHIFTS=64)
template <typename T, typename UintT, typename BitsT, int SHIFTS, uint64_t MASK, bool IS_MAX>
AUTOVECTORIZE inline void UpdateExtremesSWAR(T *extremes, uint64_t key_hash, T value) {
	union {
		uint64_t u64[SHIFTS];
		BitsT bits[64];
	} buf;
	for (int i = 0; i < SHIFTS; i++) {
		buf.u64[i] = (key_hash >> i) & MASK;
	}
	for (int i = 0; i < 64; i++) {
		UintT mask = static_cast<UintT>(-buf.bits[i]); // 1->0xFF..., 0->0x00
		union {
			T val;
			UintT bits;
		} extreme_u, result_u, out;
		extreme_u.val = PAC_BETTER(value, extremes[i]);
		result_u.val = extremes[i];
		out.bits = (extreme_u.bits & mask) | (result_u.bits & ~mask);
		extremes[i] = out.val;
	}
}

// Unsigned integer update - faster approach using mask on value directly
// MAX: value & mask -> 0 if inactive (won't beat current), value if active
// MIN: value | ~mask -> all 1s if inactive (won't beat current), value if active
template <typename T, typename UintT, typename BitsT, int SHIFTS, uint64_t MASK, bool IS_MAX>
AUTOVECTORIZE inline void UpdateExtremesUnsigned(T *extremes, uint64_t key_hash, T value) {
	union {
		uint64_t u64[SHIFTS];
		BitsT bits[64];
	} buf;
	for (int i = 0; i < SHIFTS; i++) {
		buf.u64[i] = (key_hash >> i) & MASK;
	}
	for (int i = 0; i < 64; i++) {
		UintT mask = static_cast<UintT>(-buf.bits[i]);
		if (IS_MAX) {
			extremes[i] = std::max(static_cast<T>(value & mask), extremes[i]);
		} else {
			extremes[i] = std::min(static_cast<T>(value | static_cast<T>(~mask)), extremes[i]);
		}
	}
}

// Branchless hugeint update - apply mask to both 64-bit halves
template <typename T, bool IS_MAX>
AUTOVECTORIZE inline void UpdateExtremesHugeint(T *extremes, uint64_t key_hash, T value) {
	for (int i = 0; i < 64; i++) {
		uint64_t mask = -((key_hash >> i) & 1);
		T better = PAC_BETTER(value, extremes[i]);
		extremes[i].lower = (better.lower & mask) | (extremes[i].lower & ~mask);
		extremes[i].upper = (better.upper & mask) | (extremes[i].upper & ~mask);
	}
}

// Simple branching update for PAC_MINMAX_NOSIMD benchmarking
template <typename T, bool IS_MAX>
inline void UpdateExtremesScalar(T *extremes, uint64_t key_hash, T value) {
	for (int i = 0; i < 64; i++) {
		if ((key_hash >> i) & 1) {
			extremes[i] = PAC_BETTER(value, extremes[i]);
		}
	}
}

// ============================================================================
// Simple min/max state - T[64] extremes array
// ============================================================================
template <typename T, bool IS_MAX>
struct PacMinMaxState {
	using ValueType = T;

	static inline T InitValue() {
		if (std::is_floating_point<T>::value) {
			return IS_MAX ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
		}
		return IS_MAX ? NumericLimits<T>::Minimum() : NumericLimits<T>::Maximum();
	}

	bool initialized;
	uint16_t update_count;
	T global_bound;
	T extremes[64];

	void Initialize() {
		T init = InitValue();
		for (int i = 0; i < 64; i++) {
			extremes[i] = init;
		}
		global_bound = init;
		update_count = 0;
		initialized = true;
	}

	AUTOVECTORIZE void Update(uint64_t key_hash, T value);

	void RecomputeBound() {
#ifndef PAC_MINMAX_NOBOUNDOPT
		if (++update_count >= BOUND_RECOMPUTE_INTERVAL) {
			update_count = 0;
			T bound = extremes[0];
			for (int i = 1; i < 64; i++) {
				bound = PAC_WORSE(bound, extremes[i]);
			}
			global_bound = bound;
		}
#endif
	}

	void GetTotalsAsDouble(double *dst) const {
		T init = InitValue();
		for (int i = 0; i < 64; i++) {
			dst[i] = (extremes[i] == init) ? 0.0 : ToDouble(extremes[i]);
		}
	}

	void CombineWith(const PacMinMaxState &src) {
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

	PacMinMaxState *GetState() {
		return this;
	}
	PacMinMaxState *EnsureState(ArenaAllocator &) {
		if (!initialized) {
			Initialize();
		}
		return this;
	}

	static idx_t StateSize() {
		return sizeof(PacMinMaxState);
	}
};

// ============================================================================
// Update specializations - instantiated where called
// PAC_MINMAX_NOSIMD: use simple branching scalar loop (for benchmarking)
// Default: use branchless SWAR/mask-based updates
// ============================================================================

#ifdef PAC_MINMAX_NOSIMD
// NOSIMD: all types use simple branching scalar loop
#define DEFINE_UPDATE_SCALAR(T)                                                                                        \
	template <>                                                                                                        \
	inline void PacMinMaxState<T, true>::Update(uint64_t h, T v) {                                                     \
		UpdateExtremesScalar<T, true>(extremes, h, v);                                                                 \
	}                                                                                                                  \
	template <>                                                                                                        \
	inline void PacMinMaxState<T, false>::Update(uint64_t h, T v) {                                                    \
		UpdateExtremesScalar<T, false>(extremes, h, v);                                                                \
	}

DEFINE_UPDATE_SCALAR(int8_t)
DEFINE_UPDATE_SCALAR(uint8_t)
DEFINE_UPDATE_SCALAR(int16_t)
DEFINE_UPDATE_SCALAR(uint16_t)
DEFINE_UPDATE_SCALAR(int32_t)
DEFINE_UPDATE_SCALAR(uint32_t)
DEFINE_UPDATE_SCALAR(float)
DEFINE_UPDATE_SCALAR(int64_t)
DEFINE_UPDATE_SCALAR(uint64_t)
DEFINE_UPDATE_SCALAR(double)
DEFINE_UPDATE_SCALAR(hugeint_t)
DEFINE_UPDATE_SCALAR(uhugeint_t)

#undef DEFINE_UPDATE_SCALAR

#else // Optimized branchless implementations

// int8 (SWAR 8-way, bit-select)
template <>
inline void PacMinMaxState<int8_t, true>::Update(uint64_t h, int8_t v) {
	UpdateExtremesSWAR<int8_t, uint8_t, int8_t, 8, 0x0101010101010101ULL, true>(extremes, h, v);
}
template <>
inline void PacMinMaxState<int8_t, false>::Update(uint64_t h, int8_t v) {
	UpdateExtremesSWAR<int8_t, uint8_t, int8_t, 8, 0x0101010101010101ULL, false>(extremes, h, v);
}

// uint8 (SWAR 8-way, unsigned optimization)
template <>
inline void PacMinMaxState<uint8_t, true>::Update(uint64_t h, uint8_t v) {
	UpdateExtremesUnsigned<uint8_t, uint8_t, int8_t, 8, 0x0101010101010101ULL, true>(extremes, h, v);
}
template <>
inline void PacMinMaxState<uint8_t, false>::Update(uint64_t h, uint8_t v) {
	UpdateExtremesUnsigned<uint8_t, uint8_t, int8_t, 8, 0x0101010101010101ULL, false>(extremes, h, v);
}

// int16 (SWAR 4-way, bit-select)
template <>
inline void PacMinMaxState<int16_t, true>::Update(uint64_t h, int16_t v) {
	UpdateExtremesSWAR<int16_t, uint16_t, int16_t, 16, 0x0001000100010001ULL, true>(extremes, h, v);
}
template <>
inline void PacMinMaxState<int16_t, false>::Update(uint64_t h, int16_t v) {
	UpdateExtremesSWAR<int16_t, uint16_t, int16_t, 16, 0x0001000100010001ULL, false>(extremes, h, v);
}

// uint16 (SWAR 4-way, unsigned optimization)
template <>
inline void PacMinMaxState<uint16_t, true>::Update(uint64_t h, uint16_t v) {
	UpdateExtremesUnsigned<uint16_t, uint16_t, int16_t, 16, 0x0001000100010001ULL, true>(extremes, h, v);
}
template <>
inline void PacMinMaxState<uint16_t, false>::Update(uint64_t h, uint16_t v) {
	UpdateExtremesUnsigned<uint16_t, uint16_t, int16_t, 16, 0x0001000100010001ULL, false>(extremes, h, v);
}

// int32 (SWAR 2-way, bit-select)
template <>
inline void PacMinMaxState<int32_t, true>::Update(uint64_t h, int32_t v) {
	UpdateExtremesSWAR<int32_t, uint32_t, int32_t, 32, 0x0000000100000001ULL, true>(extremes, h, v);
}
template <>
inline void PacMinMaxState<int32_t, false>::Update(uint64_t h, int32_t v) {
	UpdateExtremesSWAR<int32_t, uint32_t, int32_t, 32, 0x0000000100000001ULL, false>(extremes, h, v);
}

// uint32 (SWAR 2-way, unsigned optimization)
template <>
inline void PacMinMaxState<uint32_t, true>::Update(uint64_t h, uint32_t v) {
	UpdateExtremesUnsigned<uint32_t, uint32_t, int32_t, 32, 0x0000000100000001ULL, true>(extremes, h, v);
}
template <>
inline void PacMinMaxState<uint32_t, false>::Update(uint64_t h, uint32_t v) {
	UpdateExtremesUnsigned<uint32_t, uint32_t, int32_t, 32, 0x0000000100000001ULL, false>(extremes, h, v);
}

// float (SWAR 2-way, bit-select)
template <>
inline void PacMinMaxState<float, true>::Update(uint64_t h, float v) {
	UpdateExtremesSWAR<float, uint32_t, int32_t, 32, 0x0000000100000001ULL, true>(extremes, h, v);
}
template <>
inline void PacMinMaxState<float, false>::Update(uint64_t h, float v) {
	UpdateExtremesSWAR<float, uint32_t, int32_t, 32, 0x0000000100000001ULL, false>(extremes, h, v);
}

// int64 (SWAR 1-way, bit-select)
template <>
inline void PacMinMaxState<int64_t, true>::Update(uint64_t h, int64_t v) {
	UpdateExtremesSWAR<int64_t, uint64_t, int64_t, 64, 1ULL, true>(extremes, h, v);
}
template <>
inline void PacMinMaxState<int64_t, false>::Update(uint64_t h, int64_t v) {
	UpdateExtremesSWAR<int64_t, uint64_t, int64_t, 64, 1ULL, false>(extremes, h, v);
}

// uint64 (SWAR 1-way, unsigned optimization)
template <>
inline void PacMinMaxState<uint64_t, true>::Update(uint64_t h, uint64_t v) {
	UpdateExtremesUnsigned<uint64_t, uint64_t, int64_t, 64, 1ULL, true>(extremes, h, v);
}
template <>
inline void PacMinMaxState<uint64_t, false>::Update(uint64_t h, uint64_t v) {
	UpdateExtremesUnsigned<uint64_t, uint64_t, int64_t, 64, 1ULL, false>(extremes, h, v);
}

// double (SWAR 1-way, bit-select)
template <>
inline void PacMinMaxState<double, true>::Update(uint64_t h, double v) {
	UpdateExtremesSWAR<double, uint64_t, int64_t, 64, 1ULL, true>(extremes, h, v);
}
template <>
inline void PacMinMaxState<double, false>::Update(uint64_t h, double v) {
	UpdateExtremesSWAR<double, uint64_t, int64_t, 64, 1ULL, false>(extremes, h, v);
}

// hugeint (branchless 128-bit)
template <>
inline void PacMinMaxState<hugeint_t, true>::Update(uint64_t h, hugeint_t v) {
	UpdateExtremesHugeint<hugeint_t, true>(extremes, h, v);
}
template <>
inline void PacMinMaxState<hugeint_t, false>::Update(uint64_t h, hugeint_t v) {
	UpdateExtremesHugeint<hugeint_t, false>(extremes, h, v);
}

// uhugeint (branchless 128-bit)
template <>
inline void PacMinMaxState<uhugeint_t, true>::Update(uint64_t h, uhugeint_t v) {
	UpdateExtremesHugeint<uhugeint_t, true>(extremes, h, v);
}
template <>
inline void PacMinMaxState<uhugeint_t, false>::Update(uint64_t h, uhugeint_t v) {
	UpdateExtremesHugeint<uhugeint_t, false>(extremes, h, v);
}

#endif // PAC_MINMAX_NOSIMD

// ============================================================================
// PacMinMaxUpdateOne: direct state update (always available)
// ============================================================================
template <typename T, bool IS_MAX>
inline void PacMinMaxUpdateOne(PacMinMaxState<T, IS_MAX> &state, uint64_t key_hash, T value, ArenaAllocator &a) {
	state.EnsureState(a);
#ifndef PAC_MINMAX_NOBOUNDOPT
	if (!PAC_IS_BETTER(value, state.global_bound)) {
		return;
	}
#endif
	state.Update(key_hash, value);
	state.RecomputeBound();
}

#ifndef PAC_MINMAX_NOBUFFERING
// ============================================================================
// PacMinMaxStateWrapper: buffers (hash, value) pairs before allocating state
// Buffer size: 2 for types <= 8 bytes, 1 for hugeint (16 bytes)
// ============================================================================
template <typename T, bool IS_MAX>
struct PacMinMaxStateWrapper {
	using State = PacMinMaxState<T, IS_MAX>;
	static constexpr int BUF_SIZE = sizeof(T) <= 8 ? 2 : 1;
	static constexpr uint64_t BUF_MASK = sizeof(T) <= 8 ? 3ULL : 1ULL;

	T val_buf[BUF_SIZE];
	uint64_t hash_buf[BUF_SIZE];
	union {
		uint64_t n_buffered;
		State *state;
	};

	State *GetState() const {
		return reinterpret_cast<State *>(reinterpret_cast<uintptr_t>(state) & ~7ULL);
	}

	State *EnsureState(ArenaAllocator &a) {
		State *s = GetState();
		if (!s) {
			s = reinterpret_cast<State *>(a.Allocate(sizeof(State)));
			memset(s, 0, sizeof(State));
			state = s;
		}
		if (!s->initialized) {
			s->Initialize();
		}
		return s;
	}

	AUTOVECTORIZE static inline void FlushBufferInternal(State &dst, const T *__restrict__ vals,
	                                                     const uint64_t *__restrict__ hashes, uint64_t cnt) {
		for (uint64_t i = 0; i < cnt; i++) {
			dst.Update(hashes[i], vals[i]);
		}
		dst.RecomputeBound();
	}

	inline void FlushBuffer(PacMinMaxStateWrapper &agg, ArenaAllocator &a) {
		uint64_t cnt = agg.n_buffered & BUF_MASK;
		if (cnt > 0) {
			State &dst = *agg.EnsureState(a);
			FlushBufferInternal(dst, agg.val_buf, agg.hash_buf, cnt);
			agg.n_buffered &= ~BUF_MASK;
		}
	}
};

// Wrapper update with buffering (for scatter/grouped updates)
template <typename T, bool IS_MAX>
AUTOVECTORIZE inline void PacMinMaxBufferOrUpdateOne(PacMinMaxStateWrapper<T, IS_MAX> &agg, uint64_t key_hash, T value,
                                                     ArenaAllocator &a) {
	using State = PacMinMaxState<T, IS_MAX>;
	using Wrapper = PacMinMaxStateWrapper<T, IS_MAX>;

#ifndef PAC_MINMAX_NOBOUNDOPT
	// Bounds check if we already have a state with a valid bound
	State *s = agg.GetState();
	if (s && s->initialized && !PAC_IS_BETTER(value, s->global_bound)) {
		return;
	}
#endif

	uint64_t cnt = agg.n_buffered & Wrapper::BUF_MASK;
	if (DUCKDB_UNLIKELY(cnt == Wrapper::BUF_SIZE)) {
		State &dst = *agg.EnsureState(a);
		T vals[Wrapper::BUF_SIZE + 1];
		uint64_t hashes[Wrapper::BUF_SIZE + 1];
		for (int i = 0; i < Wrapper::BUF_SIZE; i++) {
			vals[i] = agg.val_buf[i];
			hashes[i] = agg.hash_buf[i];
		}
		vals[Wrapper::BUF_SIZE] = value;
		hashes[Wrapper::BUF_SIZE] = key_hash;
		Wrapper::FlushBufferInternal(dst, vals, hashes, Wrapper::BUF_SIZE + 1);
		agg.n_buffered &= ~Wrapper::BUF_MASK;
	} else {
		agg.val_buf[cnt] = value;
		agg.hash_buf[cnt] = key_hash;
		agg.n_buffered++;
	}
}

// Wrapper update without buffering (for non-grouped updates)
template <typename T, bool IS_MAX>
inline void PacMinMaxUpdateOne(PacMinMaxStateWrapper<T, IS_MAX> &agg, uint64_t key_hash, T value, ArenaAllocator &a) {
	PacMinMaxUpdateOne(*agg.EnsureState(a), key_hash, value, a);
}
#endif // PAC_MINMAX_NOBUFFERING

} // namespace duckdb

#endif // PAC_MIN_MAX_HPP
