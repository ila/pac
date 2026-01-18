//
// Created by ila on 12/19/25.
//

#ifndef PAC_MIN_MAX_HPP
#define PAC_MIN_MAX_HPP

// benchmarking defines that disable certain optimizations
//#define PAC_NOBUFFERING 1
//#define PAC_NOBOUNDOPT 1
//#define PAC_NOSIMD 1

// PAC_GODBOLT mode: cpp -DPAC_GODBOLT -P -E -w src/include/pac_min_max.hpp
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
// we compute the minimum of all maximums for MAX (and vice versa for MIN)
// we do this every many value updates, in order to keep costs down
// if a new value is worse than the worst of all bounds, it can be skipped
// this is called BOUNDOPT
//
// the state directly keeps TYPE probabilistic totals[64] (no cascading with smaller types)
// we tried cascading, but it does not help much, and usually BOUNDSOP means that most
// aggregations actually do not need to do an actual MIN/MAX and do not touch the array,
// hence optimizing its size does not reduce the footprint much, and for code
// simplicity it was removed.
//
// In order to keep the size of the states low, it is more important to delay the state
// allocation until multiple values have been received (buffering). Processing a buffer
// rather than individual values reduces cache misses and increases chances for SIMD.
//
// While we predicate the counting and SWAR-optimize it, we provide a naive IF..THEN baseline that is SIMD-unfriendly.
//
// Define PAC_NOBUFFERING to disable the buffering optimization.
// Define PAC_NOBOUNDOPT to disable global bound optimization.
// Define PAC_NOSIMD to get the IF..THEN SIMD-unfriendly aggergate computation kernel
#endif
#ifndef PAC_GODBOLT
static constexpr uint16_t BOUND_RECOMPUTE_INTERVAL = 2048;
#endif

// Comparison macros (IS_MAX must be a template parameter in scope)
#define PAC_IS_BETTER(a, b) (IS_MAX ? ((a) > (b)) : ((a) < (b)))
#define PAC_BETTER(a, b)    (PAC_IS_BETTER(a, b) ? (a) : (b))
#define PAC_WORSE(a, b)     (PAC_IS_BETTER(a, b) ? (b) : (a))

// SIMD-friendly branchless update (bit-select approach)
// Works for all sizes: 8-bit (SHIFTS=8), 16-bit (SHIFTS=16), 32-bit (SHIFTS=32), 64-bit (SHIFTS=64)
// FLOAT=true: for float/double, uses union to convert between T and UintT
// FLOAT=false: for integers, uses neutral value approach (SIGNED distinguishes signed vs unsigned)
template <typename T, typename UintT, typename BitsT, int SHIFTS, uint64_t MASK, bool IS_MAX, bool FLOAT, bool SIGNED>
#ifndef PAC_GODBOLT
AUTOVECTORIZE inline
#endif
    void
    UpdateExtremesSIMD(T *extremes, uint64_t key_hash, T value) {
	union {                   // union to deal with bit extraction of uint_t key_hash and differently sized T
		uint64_t u64[SHIFTS]; // an array of 64-bit uints that is equally big as 64 values of T
		BitsT bits[64];       // signed type with width equal to T (the type being aggregated)
	} buf;
	for (int i = 0; i < SHIFTS; i++) {       // fewer iterations than 64 here (SWAR)
		buf.u64[i] = (key_hash >> i) & MASK; // this operates on the uint64_t key_hash granularity
	}
	for (int i = 0; i < 64; i++) {                     // 64 iterations on 64 T values => hopefully will autovectorize
		UintT mask = static_cast<UintT>(-buf.bits[i]); // 1->0xFF..., 0->0x00
		if (FLOAT) {
			union {    // union to mitigate between type T and its bitwise representation (for 0x00/0xFF minmax masking)
				T val; // float or double, i.e. non-integer
				UintT bits; // because float is 32-bits, if T=float => UintT = uint32_t
			} extreme_u, result_u;
			extreme_u.val = extremes[i];
			result_u.val = PAC_BETTER(value, extreme_u.val);
			result_u.bits = (result_u.bits & mask) | (extreme_u.bits & ~mask); // masking on bits
			extremes[i] = result_u.val; // same value but now viewed as (multiple SWAR) values of type T
		} else {                        // integer path that directly mask on value
			UintT UNSIGNED_NOOP = (UintT(SIGNED) & ~mask) << (sizeof(T) * 8 - 1);
			UintT val_bits = static_cast<UintT>(value);
			UintT res_bits = IS_MAX ? ((val_bits & mask) | UNSIGNED_NOOP) : ((val_bits | ~mask) & ~UNSIGNED_NOOP);
			extremes[i] = PAC_BETTER(static_cast<T>(res_bits), extremes[i]); // operation on values
		}
	}
}

#ifdef PAC_GODBOLT
// Explicit instantiations for Godbolt analysis (omit hugeint)
template void UpdateExtremesSIMD<uint8_t, uint8_t, uint8_t, 8, 0x0000000100000001ULL, true, false, false>(uint8_t *,
                                                                                                          uint64_t,
                                                                                                          uint8_t);
template void
UpdateExtremesSIMD<int8_t, uint8_t, int8_t, 8, 0x0000000100000001ULL, false, false, true>(int8_t *, uint64_t, int8_t);
template void
UpdateExtremesSIMD<uint16_t, uuint16_t, uint16_t, 16, 0x0000000100000001ULL, true, false, true>(uint16_t *, uint64_t,
                                                                                                uint16_t);
template void UpdateExtremesSIMD<int16_t, uint16_t, int16_t, 16, 0x0000000100000001ULL, false, false, true>(int16_t *,
                                                                                                            uint64_t,
                                                                                                            int16_t);
template void UpdateExtremesSIMD<uint32_t, uint32_t, uint32_t, 32, 0x0000000100000001ULL, true, false, true>(uint32_t *,
                                                                                                             uint64_t,
                                                                                                             uint32_t);
template void UpdateExtremesSIMD<int32_t, uint32_t, int32_t, 32, 0x0000000100000001ULL, false, false, true>(int32_t *,
                                                                                                            uint64_t,
                                                                                                            int32_t);
template void UpdateExtremesSIMD<uint64_t, uint64_t, uint64_t, 64, 1ULL, true, false, true>(uint64_t *, uint64_t,
                                                                                            uint64_t);
template void UpdateExtremesSIMD<int64_t, uint64_t, int64_t, 64, 1ULL, false, false, true>(int64_t *, uint64_t,
                                                                                           int64_t);
template void
UpdateExtremesSIMD<float, uint32_t, int32_t, 32, 0x0000000100000001ULL, true, true, false>(float *, uint64_t, float);
template void UpdateExtremesSIMD<double, uint64_t, int64_t, 64, 1ULL, true, true, false>(double *, uint64_t, double);
#else
// Specialization for hugeint_t - branchless 128-bit update
template <bool IS_MAX, bool SIGNED>
AUTOVECTORIZE inline void UpdateExtremesSIMD(hugeint_t *extremes, uint64_t key_hash, hugeint_t value) {
	for (int i = 0; i < 64; i++) {
		uint64_t mask = -((key_hash >> i) & 1);
		hugeint_t better = PAC_BETTER(value, extremes[i]);
		extremes[i].lower = (better.lower & mask) | (extremes[i].lower & ~mask);
		extremes[i].upper = (better.upper & mask) | (extremes[i].upper & ~mask);
	}
}

// Specialization for uhugeint_t - branchless 128-bit update
template <bool IS_MAX, bool SIGNED>
AUTOVECTORIZE inline void UpdateExtremesSIMD(uhugeint_t *extremes, uint64_t key_hash, uhugeint_t value) {
	for (int i = 0; i < 64; i++) {
		uint64_t mask = -((key_hash >> i) & 1);
		uhugeint_t better = PAC_BETTER(value, extremes[i]);
		extremes[i].lower = (better.lower & mask) | (extremes[i].lower & ~mask);
		extremes[i].upper = (better.upper & mask) | (extremes[i].upper & ~mask);
	}
}
#endif

#ifndef PAC_GODBOLT
// Simple branching update for PAC_NOSIMD benchmarking
template <typename T, bool IS_MAX>
inline void UpdateExtremesScalar(T *extremes, uint64_t key_hash, T value) {
	for (int i = 0; i < 64; i++) {
		if ((key_hash >> i) & 1) { // IF..THEN cannot be simd-ized (and is 50%:has heavy branch misprediction cost)
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
	uint64_t key_hash; // OR of all key_hashes seen (for PacNoiseInNull)
	T global_bound;
	T extremes[64];

	void Initialize() {
		T init = InitValue();
		for (int i = 0; i < 64; i++) {
			extremes[i] = init;
		}
		global_bound = init;
		update_count = 0;
		key_hash = 0;
		initialized = true;
	}

	AUTOVECTORIZE void Update(uint64_t key_hash, T value);

	void RecomputeBound() {
#ifndef PAC_NOBOUNDOPT
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
		for (int i = 0; i < 64; i++) {
			dst[i] = ToDouble(extremes[i]);
		}
	}

	void CombineWith(const PacMinMaxState &src) {
		if (!src.initialized) {
			return;
		}
		if (!initialized) {
			Initialize();
		}
		key_hash |= src.key_hash;
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
// PAC_NOSIMD: use simple branching scalar loop (for benchmarking)
// Default: use branchless SIMD-friendly updates
// ============================================================================

// Macro to define Update specializations for regular types (8/16/32/64-bit)
#ifdef PAC_NOSIMD
#define DEFINE_UPDATE(T, UINT_T, BITS_T, SHIFTS, MASK, FLOAT, SIGNED)                                                  \
	template <>                                                                                                        \
	inline void PacMinMaxState<T, true>::Update(uint64_t h, T v) {                                                     \
		UpdateExtremesScalar<T, true>(extremes, h, v);                                                                 \
	}                                                                                                                  \
	template <>                                                                                                        \
	inline void PacMinMaxState<T, false>::Update(uint64_t h, T v) {                                                    \
		UpdateExtremesScalar<T, false>(extremes, h, v);                                                                \
	}
#else
#define DEFINE_UPDATE(T, UINT_T, BITS_T, SHIFTS, MASK, FLOAT, SIGNED)                                                  \
	template <>                                                                                                        \
	inline void PacMinMaxState<T, true>::Update(uint64_t h, T v) {                                                     \
		UpdateExtremesSIMD<T, UINT_T, BITS_T, SHIFTS, MASK, true, FLOAT, SIGNED>(extremes, h, v);                      \
	}                                                                                                                  \
	template <>                                                                                                        \
	inline void PacMinMaxState<T, false>::Update(uint64_t h, T v) {                                                    \
		UpdateExtremesSIMD<T, UINT_T, BITS_T, SHIFTS, MASK, false, FLOAT, SIGNED>(extremes, h, v);                     \
	}
#endif

// Macro to define Update specializations for hugeint types (128-bit)
#ifdef PAC_NOSIMD
#define DEFINE_UPDATE_HUGE(T, SIGNED)                                                                                  \
	template <>                                                                                                        \
	inline void PacMinMaxState<T, true>::Update(uint64_t h, T v) {                                                     \
		UpdateExtremesScalar<T, true>(extremes, h, v);                                                                 \
	}                                                                                                                  \
	template <>                                                                                                        \
	inline void PacMinMaxState<T, false>::Update(uint64_t h, T v) {                                                    \
		UpdateExtremesScalar<T, false>(extremes, h, v);                                                                \
	}
#else
#define DEFINE_UPDATE_HUGE(T, SIGNED)                                                                                  \
	template <>                                                                                                        \
	inline void PacMinMaxState<T, true>::Update(uint64_t h, T v) {                                                     \
		UpdateExtremesSIMD<true, SIGNED>(extremes, h, v);                                                              \
	}                                                                                                                  \
	template <>                                                                                                        \
	inline void PacMinMaxState<T, false>::Update(uint64_t h, T v) {                                                    \
		UpdateExtremesSIMD<false, SIGNED>(extremes, h, v);                                                             \
	}
#endif

// 8-bit types (SWAR 8-way)
DEFINE_UPDATE(int8_t, uint8_t, int8_t, 8, 0x0101010101010101ULL, false, true)
DEFINE_UPDATE(uint8_t, uint8_t, int8_t, 8, 0x0101010101010101ULL, false, false)

// 16-bit types (SWAR 4-way)
DEFINE_UPDATE(int16_t, uint16_t, int16_t, 16, 0x0001000100010001ULL, false, true)
DEFINE_UPDATE(uint16_t, uint16_t, int16_t, 16, 0x0001000100010001ULL, false, false)

// 32-bit types (SWAR 2-way)
DEFINE_UPDATE(int32_t, uint32_t, int32_t, 32, 0x0000000100000001ULL, false, true)
DEFINE_UPDATE(uint32_t, uint32_t, int32_t, 32, 0x0000000100000001ULL, false, false)
DEFINE_UPDATE(float, uint32_t, int32_t, 32, 0x0000000100000001ULL, true, false)

// 64-bit types (SWAR 1-way)
DEFINE_UPDATE(int64_t, uint64_t, int64_t, 64, 1ULL, false, true)
DEFINE_UPDATE(uint64_t, uint64_t, int64_t, 64, 1ULL, false, false)
DEFINE_UPDATE(double, uint64_t, int64_t, 64, 1ULL, true, false)

// 128-bit types (hugeint)
DEFINE_UPDATE_HUGE(hugeint_t, true)
DEFINE_UPDATE_HUGE(uhugeint_t, false)

#undef DEFINE_UPDATE
#undef DEFINE_UPDATE_HUGE

// ============================================================================
// PacMinMaxUpdateOne: direct state update (always available)
// ============================================================================
template <typename T, bool IS_MAX>
inline void PacMinMaxUpdateOne(PacMinMaxState<T, IS_MAX> &state, uint64_t key_hash, T value, ArenaAllocator &a) {
	state.EnsureState(a);
	state.key_hash |= key_hash;
#ifndef PAC_NOBOUNDOPT
	if (!PAC_IS_BETTER(value, state.global_bound)) {
		return;
	}
#endif
	state.Update(key_hash, value);
	state.RecomputeBound();
}

#ifndef PAC_NOBUFFERING
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

	inline void FlushBuffer(PacMinMaxStateWrapper &dst_wrapper, ArenaAllocator &a) {
		// Flush THIS wrapper's buffer into dst_wrapper's inner state
		uint64_t cnt = n_buffered & BUF_MASK;
		if (cnt > 0) {
			State &dst = *dst_wrapper.EnsureState(a);
			FlushBufferInternal(dst, val_buf, hash_buf, cnt);
			n_buffered &= ~BUF_MASK;
		}
	}
};

// Wrapper update with buffering (for scatter/grouped updates)
template <typename T, bool IS_MAX>
AUTOVECTORIZE inline void PacMinMaxBufferOrUpdateOne(PacMinMaxStateWrapper<T, IS_MAX> &agg, uint64_t key_hash, T value,
                                                     ArenaAllocator &a) {
	using State = PacMinMaxState<T, IS_MAX>;
	using Wrapper = PacMinMaxStateWrapper<T, IS_MAX>;

#ifndef PAC_NOBOUNDOPT
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
			dst.key_hash |= (hashes[i] = agg.hash_buf[i]);
		}
		vals[Wrapper::BUF_SIZE] = value;
		dst.key_hash |= (hashes[Wrapper::BUF_SIZE] = key_hash);
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
#endif // PAC_NOBUFFERING

} // namespace duckdb
#endif // PAC_GODBOLT

#endif // PAC_MIN_MAX_HPP
