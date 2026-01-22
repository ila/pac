#include "include/pac_sum_avg.hpp"
#include "duckdb/common/types/decimal.hpp"
#include <cmath>
#include <limits>
#include <unordered_map>

namespace duckdb {

// ============================================================================
// State type selection for scatter updates
// ============================================================================
#ifdef PAC_NOBUFFERING
template <bool SIGNED>
using ScatterIntState = PacSumIntState<SIGNED>;
using ScatterDoubleState = PacSumDoubleState;
template <bool SIGNED>
using ScatterApproxState = PacSumApproxState<SIGNED>;
#else
template <bool SIGNED>
using ScatterIntState = PacSumIntStateWrapper<SIGNED>;
using ScatterDoubleState = PacSumDoubleStateWrapper;
template <bool SIGNED>
using ScatterApproxState = PacSumApproxStateWrapper<SIGNED>;
#endif

// SIGNED is compile-time known, so for unsigned the negative cases (value < 0) will be compiled away
#define ACCUMULATE_BITMARGIN      2 // val must be 2 bits shorter than the accumulator to allow >=4 updates without overflow
#define UPPERBOUND_BITWIDTH(bits) (1LL << ((bits - SIGNED) - ACCUMULATE_BITMARGIN))
#define LOWERBOUND_BITWIDTH(bits) -(static_cast<int64_t>(SIGNED) << ((bits - SIGNED) - ACCUMULATE_BITMARGIN))

// ============================================================================
// Inner state update functions (work directly on PacSumIntState/PacSumDoubleState)
// ============================================================================

template <bool SIGNED>
AUTOVECTORIZE inline void // main worker function for probabilistically adding one INTEGER to the 64 (sub)total
PacSumUpdateOneInternal(PacSumIntState<SIGNED> &state, uint64_t key_hash, typename PacSumIntState<SIGNED>::T64 value,
                        ArenaAllocator &allocator) {
	state.key_hash |= key_hash;
#ifdef PAC_NOCASCADING
	AddToTotalsSimple(state.probabilistic_total128, value, key_hash); // directly add the value to the final total
#else
	// decide based on the (integer) value, in which level to aggregate (ensure the level is allocated)
	// note that SIGNED stuff will be compiled away for unsigned types
	if ((SIGNED && value < 0) ? (value >= LOWERBOUND_BITWIDTH(8)) : (value < UPPERBOUND_BITWIDTH(8))) {
		state.exact_total8 =
		    PacSumIntState<SIGNED>::EnsureLevelAllocated(allocator, state.probabilistic_total8, 8, state.exact_total8);
		state.Flush8(allocator, value, false);
		AddToTotalsSWAR<int8_t, uint8_t, 0x0101010101010101ULL>(state.probabilistic_total8, value, key_hash);
	} else if ((SIGNED && value < 0) ? (value >= LOWERBOUND_BITWIDTH(16)) : (value < UPPERBOUND_BITWIDTH(16))) {
		state.exact_total16 = PacSumIntState<SIGNED>::EnsureLevelAllocated(allocator, state.probabilistic_total16, 16,
		                                                                   state.exact_total16);
		state.Flush16(allocator, value, false);
		AddToTotalsSWAR<int16_t, uint16_t, 0x0001000100010001ULL>(state.probabilistic_total16, value, key_hash);
	} else if ((SIGNED && value < 0) ? (value >= LOWERBOUND_BITWIDTH(32)) : (value < UPPERBOUND_BITWIDTH(32))) {
		state.exact_total32 = PacSumIntState<SIGNED>::EnsureLevelAllocated(allocator, state.probabilistic_total32, 32,
		                                                                   state.exact_total32);
		state.Flush32(allocator, value, false);
		AddToTotalsSWAR<int32_t, uint32_t, 0x0000000100000001ULL>(state.probabilistic_total32, value, key_hash);
	} else {
		state.exact_total64 = PacSumIntState<SIGNED>::EnsureLevelAllocated(allocator, state.probabilistic_total64, 64,
		                                                                   state.exact_total64);
		state.Flush64(allocator, value, false);
		AddToTotalsSimple(state.probabilistic_total64, value, key_hash);
	}
#endif
}

template <bool SIGNED>
AUTOVECTORIZE inline void // main worker function for probabilistically adding one DOUBLE to the 64 sum total
PacSumUpdateOneInternal(PacSumDoubleState &state, uint64_t key_hash, double value, ArenaAllocator &) {
	state.key_hash |= key_hash;
	AddToTotalsSimple(state.probabilistic_total, value, key_hash);
}

// Overload for HUGEINT input - adds directly to hugeint_t total (no cascading since values don't fit in subtotal)
template <bool SIGNED>
AUTOVECTORIZE inline void PacSumUpdateOneInternal(PacSumIntState<SIGNED> &state, uint64_t key_hash, hugeint_t value,
                                                  ArenaAllocator &allocator) {
	state.key_hash |= key_hash;
#ifndef PAC_NOCASCADING
	PacSumIntState<SIGNED>::EnsureLevelAllocated(allocator, state.probabilistic_total128, idx_t(64));
#endif
	for (int j = 0; j < 64; j++) {
		if ((key_hash >> j) & 1ULL) {
			state.probabilistic_total128[j] += value;
		}
	}
}

// ============================================================================
// Approximate pac_sum update function (adaptive 16-bit counters with scaling)
// ============================================================================
// Safe 16-bit range with margin: counters must stay in this range
constexpr int32_t APPROX_UPPER = 1L << 13; // 8192 (leaves 2-bit margin in int16)
constexpr int8_t MIN_SHIFT_INCREMENT = 3;  // Minimum shift increase to amortize rebalancing cost

// Rebalance counters by subtracting min (or max for negative total) and adding to base
// Returns the new maximum counter magnitude after rebalancing
// NOTE: base is stored in the ORIGINAL (unshifted) domain
template <bool SIGNED>
inline int32_t ApproxRebalance(PacSumApproxState<SIGNED> &state) {
	if (!state.probabilistic_total16) {
		return 0;
	}

	int16_t min_val, max_val;
	ApproxFindMinMax<SIGNED>(state.probabilistic_total16, min_val, max_val);

	// Determine which extreme to subtract based on which has larger magnitude
	int16_t delta;
	if (SIGNED) {
		// For signed, subtract the value that brings counters closest to zero
		// If max magnitude is larger, subtract that; otherwise subtract min
		if (max_val > 0 && max_val >= -min_val) {
			delta = min_val; // Subtract min to reduce max
		} else if (min_val < 0) {
			delta = max_val; // Subtract max to reduce min magnitude
		} else {
			delta = min_val;
		}
	} else {
		delta = min_val; // For unsigned, always subtract min
	}

	if (delta == 0) {
		// No rebalancing possible, return current max magnitude
		return SIGNED ? std::max(static_cast<int32_t>(max_val), static_cast<int32_t>(-min_val))
		              : static_cast<int32_t>(max_val);
	}

	// Subtract delta from all counters
	ApproxSubtractFromAll<SIGNED>(state.probabilistic_total16, delta);

	// Add 64*delta to base_rebal in ORIGINAL domain (delta subtracted from all 64 counters)
	// Rebalancing subtracts from all 64 counters uniformly
	state.base_rebal += static_cast<double>(delta) * 64.0 * static_cast<double>(1ULL << state.shift_amount);

	// Return new max magnitude
	int32_t new_max = SIGNED ? std::max(static_cast<int32_t>(max_val - delta), static_cast<int32_t>(-(min_val - delta)))
	                         : static_cast<int32_t>(max_val - delta);
	return new_max;
}

// Get current max counter magnitude
template <bool SIGNED>
inline int32_t ApproxGetMaxMagnitude(const PacSumApproxState<SIGNED> &state) {
	if (!state.probabilistic_total16) {
		return 0;
	}
	int16_t min_val, max_val;
	ApproxFindMinMax<SIGNED>(state.probabilistic_total16, min_val, max_val);
	return SIGNED ? std::max(static_cast<int32_t>(max_val), static_cast<int32_t>(-min_val))
	              : static_cast<int32_t>(max_val);
}

template <bool SIGNED>
AUTOVECTORIZE inline void PacApproxSumUpdateOneInternal(PacSumApproxState<SIGNED> &state, uint64_t key_hash,
                                                        typename PacSumApproxState<SIGNED>::T64 value,
                                                        ArenaAllocator &allocator) {
	using T32 = typename PacSumApproxState<SIGNED>::T32;
	using T64 = typename PacSumApproxState<SIGNED>::T64;
	state.key_hash |= key_hash;
	state.exact_count++;

	// Ensure array allocated
	if (!state.probabilistic_total16) {
		state.probabilistic_total16 = reinterpret_cast<uint64_t *>(allocator.Allocate(16 * sizeof(uint64_t)));
		memset(state.probabilistic_total16, 0, 16 * sizeof(uint64_t));
	}

	// Compute the shifted value we want to add
	constexpr int8_t MAX_SHIFT = 112;
	int64_t shifted_value = value >> state.shift_amount;
	int64_t abs_shifted = SIGNED ? (shifted_value < 0 ? -shifted_value : shifted_value) : shifted_value;

	// CRITICAL: Check if this value itself would overflow a 16-bit counter
	// This must happen BEFORE we add to counters, not just periodically
	if (abs_shifted > APPROX_UPPER && state.shift_amount < MAX_SHIFT) {
		// Calculate shift amount needed (starting at MIN_SHIFT_INCREMENT=3)
		int8_t extra_shift = MIN_SHIFT_INCREMENT;
		while ((abs_shifted >> extra_shift) > APPROX_UPPER && state.shift_amount + extra_shift < MAX_SHIFT) {
			extra_shift++;
		}

		// INVARIANT preservation: base_value += (low_bits_of_bound << shift_amount) before shifting bound
		T32 low_bits_mask = (static_cast<T32>(1) << extra_shift) - 1;
		T32 low_bits = state.bound & low_bits_mask;
		state.base_value += static_cast<double>(low_bits) * static_cast<double>(1ULL << state.shift_amount);
		state.bound >>= extra_shift;

		// Track lost bits from counters before shifting them (per-counter loss)
		int64_t counter_lost = ApproxSumLowBits<SIGNED>(state.probabilistic_total16, extra_shift);
		state.base_counter += static_cast<double>(counter_lost) * static_cast<double>(1ULL << state.shift_amount);

		ApproxShiftArrayRight<SIGNED>(state.probabilistic_total16, extra_shift);
		state.shift_amount += extra_shift;

		// Recompute shifted value with new shift_amount
		shifted_value = value >> state.shift_amount;
		abs_shifted = SIGNED ? (shifted_value < 0 ? -shifted_value : shifted_value) : shifted_value;
	}

	// Track lost bits from value shifting (distributed to ~32 counters)
	// lost = value - (shifted_value << shift_amount)
	if (state.shift_amount > 0) {
		T64 reconstructed = static_cast<T64>(shifted_value) << state.shift_amount;
		state.base_value += static_cast<double>(value - reconstructed);
	}

	// Update cumulative bound to track sum of |shifted_value|
	state.bound += static_cast<T32>(abs_shifted);

	// Check if bound exceeds safe range - need to rebalance or shift
	if (state.bound > APPROX_UPPER) {
		// First try rebalancing: subtract delta from all counters, add delta to base
		// ApproxRebalance adds delta << shift to base, we must subtract delta from bound
		int16_t min_val, max_val;
		ApproxFindMinMax<SIGNED>(state.probabilistic_total16, min_val, max_val);
		int16_t delta = 0;
		if (SIGNED) {
			if (max_val > -min_val) {
				delta = min_val;
			} else {
				delta = max_val;
			}
		} else {
			delta = min_val;
		}

		if (delta != 0) {
			ApproxSubtractFromAll<SIGNED>(state.probabilistic_total16, delta);
			// Add delta to base_rebal in ORIGINAL domain (rebalancing subtracts from all 64 counters)
			state.base_rebal += static_cast<double>(delta) * 64.0 * static_cast<double>(1ULL << state.shift_amount);
			// Recompute max after rebalancing
			ApproxFindMinMax<SIGNED>(state.probabilistic_total16, min_val, max_val);
		}

		int32_t new_max = SIGNED ? std::max(static_cast<int32_t>(max_val), static_cast<int32_t>(-min_val))
		                         : static_cast<int32_t>(max_val);
		// Reset bound to actual max counter value for tighter overflow detection
		state.bound = static_cast<T32>(new_max);

		// If rebalancing didn't bring us under the limit, shift right
		if (new_max > APPROX_UPPER && state.shift_amount < MAX_SHIFT) {
			int8_t extra_shift = MIN_SHIFT_INCREMENT;
			int64_t test_max = new_max;
			while (state.shift_amount + extra_shift < MAX_SHIFT) {
				if ((test_max >> extra_shift) <= APPROX_UPPER) {
					break;
				}
				extra_shift++;
			}

			// INVARIANT preservation: base_value += (low_bits_of_bound << shift_amount) before shifting bound
			T32 low_bits_mask = (static_cast<T32>(1) << extra_shift) - 1;
			T32 low_bits = state.bound & low_bits_mask;
			state.base_value += static_cast<double>(low_bits) * static_cast<double>(1ULL << state.shift_amount);
			state.bound >>= extra_shift;

			// Track lost bits from counters before shifting them (per-counter loss)
			int64_t counter_lost = ApproxSumLowBits<SIGNED>(state.probabilistic_total16, extra_shift);
			state.base_counter += static_cast<double>(counter_lost) * static_cast<double>(1ULL << state.shift_amount);

			ApproxShiftArrayRight<SIGNED>(state.probabilistic_total16, extra_shift);
			state.shift_amount += extra_shift;

			// Recompute shifted_value and abs_shifted with new shift_amount
			shifted_value = value >> state.shift_amount;
			abs_shifted = SIGNED ? (shifted_value < 0 ? -shifted_value : shifted_value) : shifted_value;
		}
	}

	// Add to counters
	AddToTotalsSWAR<int16_t, uint16_t, 0x0001000100010001ULL>(state.probabilistic_total16, shifted_value, key_hash);
}

// Overload for HUGEINT input in approx mode
template <bool SIGNED>
AUTOVECTORIZE inline void PacApproxSumUpdateOneInternal(PacSumApproxState<SIGNED> &state, uint64_t key_hash,
                                                        hugeint_t value, ArenaAllocator &allocator) {
	using T32 = typename PacSumApproxState<SIGNED>::T32;
	state.key_hash |= key_hash;
	state.exact_count++;

	// Ensure array allocated
	if (!state.probabilistic_total16) {
		state.probabilistic_total16 = reinterpret_cast<uint64_t *>(allocator.Allocate(16 * sizeof(uint64_t)));
		memset(state.probabilistic_total16, 0, 16 * sizeof(uint64_t));
	}

	constexpr int8_t MAX_SHIFT = 112;
	hugeint_t abs_value = value < 0 ? -value : value;
	hugeint_t shifted_huge = value >> state.shift_amount;
	int64_t shifted_value = Hugeint::Cast<int64_t>(shifted_huge);
	int64_t abs_shifted = Hugeint::Cast<int64_t>(abs_value >> state.shift_amount);

	// Check if this value would overflow - shift if needed
	while (abs_shifted > APPROX_UPPER && state.shift_amount < MAX_SHIFT) {
		int8_t extra_shift = MIN_SHIFT_INCREMENT;
		while ((abs_shifted >> extra_shift) > APPROX_UPPER && state.shift_amount + extra_shift < MAX_SHIFT) {
			extra_shift++;
		}

		// INVARIANT: base_value += (low_bits_of_bound << shift_amount) before shifting
		T32 low_bits = state.bound & ((static_cast<T32>(1) << extra_shift) - 1);
		state.base_value += static_cast<double>(low_bits) * static_cast<double>(1ULL << state.shift_amount);
		state.bound >>= extra_shift;

		// Track lost bits from counters before shifting them (per-counter loss)
		int64_t counter_lost = ApproxSumLowBits<SIGNED>(state.probabilistic_total16, extra_shift);
		state.base_counter += static_cast<double>(counter_lost) * static_cast<double>(1ULL << state.shift_amount);

		ApproxShiftArrayRight<SIGNED>(state.probabilistic_total16, extra_shift);
		state.shift_amount += extra_shift;

		shifted_huge = value >> state.shift_amount;
		shifted_value = Hugeint::Cast<int64_t>(shifted_huge);
		abs_shifted = Hugeint::Cast<int64_t>(abs_value >> state.shift_amount);
	}

	// Track lost bits from value shifting (distributed to ~32 counters)
	if (state.shift_amount > 0) {
		hugeint_t reconstructed = shifted_huge << state.shift_amount;
		state.base_value += Hugeint::Cast<double>(value - reconstructed);
	}

	// Update cumulative bound to track sum of |shifted_value|
	state.bound += static_cast<T32>(abs_shifted);

	// Check if bound exceeds safe range
	if (state.bound > APPROX_UPPER) {
		// Rebalance: subtract delta from counters, add to base_rebal, subtract from bound
		int16_t min_val, max_val;
		ApproxFindMinMax<SIGNED>(state.probabilistic_total16, min_val, max_val);
		int16_t delta = SIGNED ? (max_val > -min_val ? min_val : max_val) : min_val;

		if (delta != 0) {
			ApproxSubtractFromAll<SIGNED>(state.probabilistic_total16, delta);
			state.base_rebal += static_cast<double>(delta) * 64.0 * static_cast<double>(1ULL << state.shift_amount);
			ApproxFindMinMax<SIGNED>(state.probabilistic_total16, min_val, max_val);
		}

		int32_t new_max = SIGNED ? std::max(static_cast<int32_t>(max_val), static_cast<int32_t>(-min_val))
		                         : static_cast<int32_t>(max_val);
		// Reset bound to actual max counter value for tighter overflow detection
		state.bound = static_cast<T32>(new_max);

		// If still over limit, shift
		if (new_max > APPROX_UPPER && state.shift_amount < MAX_SHIFT) {
			int8_t extra_shift = MIN_SHIFT_INCREMENT;
			while ((new_max >> extra_shift) > APPROX_UPPER && state.shift_amount + extra_shift < MAX_SHIFT) {
				extra_shift++;
			}

			// INVARIANT: base_value += (low_bits_of_bound << shift_amount) before shifting
			T32 low_bits = state.bound & ((static_cast<T32>(1) << extra_shift) - 1);
			state.base_value += static_cast<double>(low_bits) * static_cast<double>(1ULL << state.shift_amount);
			state.bound >>= extra_shift;

			// Track lost bits from counters before shifting them (per-counter loss)
			int64_t counter_lost = ApproxSumLowBits<SIGNED>(state.probabilistic_total16, extra_shift);
			state.base_counter += static_cast<double>(counter_lost) * static_cast<double>(1ULL << state.shift_amount);

			ApproxShiftArrayRight<SIGNED>(state.probabilistic_total16, extra_shift);
			state.shift_amount += extra_shift;

			shifted_huge = value >> state.shift_amount;
			shifted_value = Hugeint::Cast<int64_t>(shifted_huge);
		}
	}

	AddToTotalsSWAR<int16_t, uint16_t, 0x0001000100010001ULL>(state.probabilistic_total16, shifted_value, key_hash);
}

// ============================================================================
// Unified PacSumUpdateOne - uses ifdefs to choose between buffering or direct update
// ============================================================================

#ifdef PAC_NOBUFFERING
// No buffering: ScatterState IS the inner state, just delegate to inner update
template <bool SIGNED>
AUTOVECTORIZE inline void PacSumUpdateOne(ScatterIntState<SIGNED> &state, uint64_t key_hash,
                                          typename PacSumIntState<SIGNED>::T64 value, ArenaAllocator &a) {
	PacSumUpdateOneInternal<SIGNED>(state, key_hash, value, a);
}

template <bool SIGNED>
AUTOVECTORIZE inline void PacSumUpdateOne(ScatterDoubleState &state, uint64_t key_hash, double value,
                                          ArenaAllocator &a) {
	PacSumUpdateOneInternal<SIGNED>(state, key_hash, value, a);
}

template <bool SIGNED>
AUTOVECTORIZE inline void PacSumUpdateOne(ScatterIntState<SIGNED> &state, uint64_t key_hash, hugeint_t value,
                                          ArenaAllocator &a) {
	PacSumUpdateOneInternal<SIGNED>(state, key_hash, value, a);
}
#else // Buffering enabled

//  FlushBuffer - flushes src's buffer into dst's inner state
// To flush into self, pass same wrapper for both src and dst
template <bool SIGNED, typename WrapperT>
inline void PacSumFlushBuffer(WrapperT &src, WrapperT &dst, ArenaAllocator &a) {
	uint64_t cnt = src.n_buffered & WrapperT::BUF_MASK;
	if (cnt > 0) {
		auto &dst_inner = *dst.EnsureState(a);
		for (uint64_t i = 0; i < cnt; i++) {
			PacSumUpdateOneInternal<SIGNED>(dst_inner, src.hash_buf[i], src.val_buf[i], a);
		}
		src.n_buffered &= ~WrapperT::BUF_MASK;
	}
}

// Unified buffering update - works for both int and double wrappers
template <bool SIGNED, typename WrapperT>
AUTOVECTORIZE inline void PacSumUpdateOne(WrapperT &agg, uint64_t key_hash, typename WrapperT::Value value,
                                          ArenaAllocator &a) {
	uint64_t cnt = agg.n_buffered & WrapperT::BUF_MASK;
	if (DUCKDB_UNLIKELY(cnt == WrapperT::BUF_SIZE)) {
		auto &dst = *agg.EnsureState(a);
		for (int i = 0; i < WrapperT::BUF_SIZE; i++) {
			PacSumUpdateOneInternal<SIGNED>(dst, agg.hash_buf[i], agg.val_buf[i], a);
		}
		PacSumUpdateOneInternal<SIGNED>(dst, key_hash, value, a);
		agg.n_buffered &= ~WrapperT::BUF_MASK;
	} else {
		agg.val_buf[cnt] = value;
		agg.hash_buf[cnt] = key_hash;
		agg.n_buffered++;
	}
}

// Hugeint doesn't benefit from buffering, just update directly
template <bool SIGNED>
inline void PacSumUpdateOne(PacSumIntStateWrapper<SIGNED> &agg, uint64_t key_hash, hugeint_t value, ArenaAllocator &a) {
	PacSumUpdateOneInternal<SIGNED>(*agg.EnsureState(a), key_hash, value, a);
}

// ============================================================================
// Approx state buffering support
// ============================================================================

// FlushBuffer for approx states - flushes src's buffer into dst's inner state
template <bool SIGNED>
inline void PacApproxFlushBuffer(PacSumApproxStateWrapper<SIGNED> &src, PacSumApproxStateWrapper<SIGNED> &dst,
                                 ArenaAllocator &a) {
	uint64_t cnt = src.n_buffered & PacSumApproxStateWrapper<SIGNED>::BUF_MASK;
	if (cnt > 0) {
		auto &dst_inner = *dst.EnsureState(a);
		for (uint64_t i = 0; i < cnt; i++) {
			PacApproxSumUpdateOneInternal<SIGNED>(dst_inner, src.hash_buf[i], src.val_buf[i], a);
		}
		src.n_buffered &= ~PacSumApproxStateWrapper<SIGNED>::BUF_MASK;
	}
}

// Unified buffering update for approx state
template <bool SIGNED>
AUTOVECTORIZE inline void PacApproxUpdateOne(PacSumApproxStateWrapper<SIGNED> &agg, uint64_t key_hash,
                                             typename PacSumApproxState<SIGNED>::T64 value, ArenaAllocator &a) {
	uint64_t cnt = agg.n_buffered & PacSumApproxStateWrapper<SIGNED>::BUF_MASK;
	if (DUCKDB_UNLIKELY(cnt == PacSumApproxStateWrapper<SIGNED>::BUF_SIZE)) {
		auto &dst = *agg.EnsureState(a);
		for (int i = 0; i < PacSumApproxStateWrapper<SIGNED>::BUF_SIZE; i++) {
			PacApproxSumUpdateOneInternal<SIGNED>(dst, agg.hash_buf[i], agg.val_buf[i], a);
		}
		PacApproxSumUpdateOneInternal<SIGNED>(dst, key_hash, value, a);
		agg.n_buffered &= ~PacSumApproxStateWrapper<SIGNED>::BUF_MASK;
	} else {
		agg.val_buf[cnt] = value;
		agg.hash_buf[cnt] = key_hash;
		agg.n_buffered++;
	}
}

// Hugeint doesn't benefit from buffering, just update directly
template <bool SIGNED>
inline void PacApproxUpdateOne(PacSumApproxStateWrapper<SIGNED> &agg, uint64_t key_hash, hugeint_t value,
                               ArenaAllocator &a) {
	PacApproxSumUpdateOneInternal<SIGNED>(*agg.EnsureState(a), key_hash, value, a);
}

#endif // PAC_NOBUFFERING

// ============================================================================
// Approx state no-buffering support
// ============================================================================

#ifdef PAC_NOBUFFERING
// No buffering: ScatterApproxState IS the inner state, just delegate to inner update
template <bool SIGNED>
AUTOVECTORIZE inline void PacApproxUpdateOne(ScatterApproxState<SIGNED> &state, uint64_t key_hash,
                                             typename PacSumApproxState<SIGNED>::T64 value, ArenaAllocator &a) {
	PacApproxSumUpdateOneInternal<SIGNED>(state, key_hash, value, a);
}

template <bool SIGNED>
AUTOVECTORIZE inline void PacApproxUpdateOne(ScatterApproxState<SIGNED> &state, uint64_t key_hash, hugeint_t value,
                                             ArenaAllocator &a) {
	PacApproxSumUpdateOneInternal<SIGNED>(state, key_hash, value, a);
}
#endif

template <class State, bool SIGNED, class VALUE_TYPE, class INPUT_TYPE>
static void PacSumUpdate(Vector inputs[], data_ptr_t state_p, idx_t count, ArenaAllocator &allocator,
                         uint64_t query_hash) {
	auto &state = *reinterpret_cast<State *>(state_p);
#ifdef PAC_UNSAFENULL
	if (state.seen_null) {
		return;
	}
#endif
	UnifiedVectorFormat hash_data, value_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);

	// Simple update: bypass buffering, update inner state directly
	auto &inner = *state.EnsureState(allocator);

	// Fast path: if both vectors have no nulls, skip per-row validity check
	if (hash_data.validity.AllValid() && value_data.validity.AllValid()) {
		state.exact_count += count; // increment count by batch size
		for (idx_t i = 0; i < count; i++) {
			auto h_idx = hash_data.sel->get_index(i);
			auto v_idx = value_data.sel->get_index(i);
			PacSumUpdateOneInternal<SIGNED>(inner, PacMixHash(hashes[h_idx], query_hash),
			                                ConvertValue<VALUE_TYPE>::convert(values[v_idx]), allocator);
		}
	} else {
		for (idx_t i = 0; i < count; i++) {
			auto h_idx = hash_data.sel->get_index(i);
			auto v_idx = value_data.sel->get_index(i);
			if (!hash_data.validity.RowIsValid(h_idx) || !value_data.validity.RowIsValid(v_idx)) {
#ifdef PAC_UNSAFENULL
				inner.seen_null = true;
				return;
#else
				continue; // safe mode: ignore NULLs
#endif
			}
			state.exact_count++;
			PacSumUpdateOneInternal<SIGNED>(inner, PacMixHash(hashes[h_idx], query_hash),
			                                ConvertValue<VALUE_TYPE>::convert(values[v_idx]), allocator);
		}
	}
}

template <class State, bool SIGNED, class VALUE_TYPE, class INPUT_TYPE>
static void PacSumScatterUpdate(Vector inputs[], Vector &states, idx_t count, ArenaAllocator &allocator,
                                uint64_t query_hash) {
	UnifiedVectorFormat hash_data, value_data, sdata;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	states.ToUnifiedFormat(count, sdata);

	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);
	auto state_ptrs = UnifiedVectorFormat::GetData<State *>(sdata);

	for (idx_t i = 0; i < count; i++) {
		auto h_idx = hash_data.sel->get_index(i);
		auto v_idx = value_data.sel->get_index(i);
		auto state = state_ptrs[sdata.sel->get_index(i)];
#ifdef PAC_UNSAFENULL
		auto *inner = state->GetState();
		if (inner && inner->seen_null) {
			continue; // result will be NULL anyway
		} else if (!hash_data.validity.RowIsValid(h_idx) || !value_data.validity.RowIsValid(v_idx)) {
			if (inner)
				inner->seen_null = true;
		} else {
#else
		if (!hash_data.validity.RowIsValid(h_idx) || !value_data.validity.RowIsValid(v_idx)) {
			continue; // safe mode: ignore NULLs
		} else {
#endif
			state->exact_count++;
			PacSumUpdateOne<SIGNED>(*state, PacMixHash(hashes[h_idx], query_hash),
			                        ConvertValue<VALUE_TYPE>::convert(values[v_idx]), allocator);
		}
	}
}

// ============================================================================
// Approx Update and ScatterUpdate templates
// ============================================================================

template <bool SIGNED, class VALUE_TYPE, class INPUT_TYPE>
static void PacApproxUpdate(Vector inputs[], data_ptr_t state_p, idx_t count, ArenaAllocator &allocator,
                            uint64_t query_hash) {
	auto &state = *reinterpret_cast<ScatterApproxState<SIGNED> *>(state_p);
	UnifiedVectorFormat hash_data, value_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);

	auto &inner = *state.EnsureState(allocator);

	// Note: exact_count is tracked in the inner state via PacApproxSumUpdateOneInternal, not in wrapper
	if (hash_data.validity.AllValid() && value_data.validity.AllValid()) {
		for (idx_t i = 0; i < count; i++) {
			auto h_idx = hash_data.sel->get_index(i);
			auto v_idx = value_data.sel->get_index(i);
			PacApproxSumUpdateOneInternal<SIGNED>(inner, PacMixHash(hashes[h_idx], query_hash),
			                                      ConvertValue<VALUE_TYPE>::convert(values[v_idx]), allocator);
		}
	} else {
		for (idx_t i = 0; i < count; i++) {
			auto h_idx = hash_data.sel->get_index(i);
			auto v_idx = value_data.sel->get_index(i);
			if (!hash_data.validity.RowIsValid(h_idx) || !value_data.validity.RowIsValid(v_idx)) {
				continue;
			}
			PacApproxSumUpdateOneInternal<SIGNED>(inner, PacMixHash(hashes[h_idx], query_hash),
			                                      ConvertValue<VALUE_TYPE>::convert(values[v_idx]), allocator);
		}
	}
}

template <bool SIGNED, class VALUE_TYPE, class INPUT_TYPE>
static void PacApproxScatterUpdate(Vector inputs[], Vector &states, idx_t count, ArenaAllocator &allocator,
                                   uint64_t query_hash) {
	UnifiedVectorFormat hash_data, value_data, sdata;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	states.ToUnifiedFormat(count, sdata);

	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);
	auto state_ptrs = UnifiedVectorFormat::GetData<ScatterApproxState<SIGNED> *>(sdata);

	for (idx_t i = 0; i < count; i++) {
		auto h_idx = hash_data.sel->get_index(i);
		auto v_idx = value_data.sel->get_index(i);
		auto state = state_ptrs[sdata.sel->get_index(i)];
		if (!hash_data.validity.RowIsValid(h_idx) || !value_data.validity.RowIsValid(v_idx)) {
			continue;
		}
		PacApproxUpdateOne<SIGNED>(*state, PacMixHash(hashes[h_idx], query_hash),
		                           ConvertValue<VALUE_TYPE>::convert(values[v_idx]), allocator);
	}
}

// Helper to combine src array into dst at a specific level
template <typename BUF_T>
static inline void CombineLevel(BUF_T *&src_buf, BUF_T *&dst_buf, idx_t count) {
	if (src_buf && dst_buf) {
		for (idx_t j = 0; j < count; j++) {
			dst_buf[j] += src_buf[j];
		}
	} else if (src_buf) {
		dst_buf = src_buf;
		src_buf = nullptr;
	}
}

// Combine for integer states - combines at each level without forcing to 128-bit
// If dst doesn't have a level that src has, we move the pointer (no copy needed)
template <bool SIGNED>
AUTOVECTORIZE static void PacSumCombineInt(Vector &src, Vector &dst, idx_t count, ArenaAllocator &allocator) {
	auto src_wrapper = FlatVector::GetData<ScatterIntState<SIGNED> *>(src);
	auto dst_wrapper = FlatVector::GetData<ScatterIntState<SIGNED> *>(dst);

	for (idx_t i = 0; i < count; i++) {
#ifndef PAC_NOBUFFERING
		// Flush src's buffer into dst's inner state (avoids allocating src inner)
		PacSumFlushBuffer<SIGNED>(*src_wrapper[i], *dst_wrapper[i], allocator);
#endif
		auto *s = src_wrapper[i]->GetState();
		auto *d_wrapper = dst_wrapper[i];
#ifdef PAC_UNSAFENULL
		if (s && s->seen_null) {
			auto *d = d_wrapper->EnsureState(allocator);
			d->seen_null = true;
		}
		auto *d_inner = d_wrapper->GetState();
		if (d_inner && d_inner->seen_null) {
			continue;
		}
#endif
		// Merge exact_count from wrapper
#ifndef PAC_NOBUFFERING
		d_wrapper->exact_count += src_wrapper[i]->exact_count;
#endif
		if (!s) {
			continue; // src has no state allocated, nothing to combine
		}
		auto *d = d_wrapper->EnsureState(allocator);
		d->key_hash |= s->key_hash;
#ifdef PAC_NOCASCADING
		for (int j = 0; j < 64; j++) {
			d->probabilistic_total128[j] += s->probabilistic_total128[j];
		}
#else
		// Handle exact_totals: if sum would overflow, flush dst (passing src's value so it becomes
		// the new exact_total after flush). Otherwise just add. Both must be allocated to overflow.
		// Note: we cast to int64_t for the overflow check to avoid truncation with small types.
		if (CHECK_BOUNDS_8(static_cast<int64_t>(s->exact_total8) + d->exact_total8)) {
			d->Flush8(allocator, s->exact_total8, true); // flushes d, sets d->exact_total8 = s->exact_total8
		} else {
			d->exact_total8 += s->exact_total8;
		}
		if (CHECK_BOUNDS_16(static_cast<int64_t>(s->exact_total16) + d->exact_total16)) {
			d->Flush16(allocator, s->exact_total16, true);
		} else {
			d->exact_total16 += s->exact_total16;
		}
		if (CHECK_BOUNDS_32(static_cast<int64_t>(s->exact_total32) + d->exact_total32)) {
			d->Flush32(allocator, s->exact_total32, true);
		} else {
			d->exact_total32 += s->exact_total32;
		}
		if (CHECK_BOUNDS_64(s->exact_total64 + d->exact_total64, s->exact_total64, d->exact_total64)) {
			d->Flush64(allocator, s->exact_total64, true);
		} else {
			d->exact_total64 += s->exact_total64;
		}
		// Combine arrays at each level
		CombineLevel(s->probabilistic_total8, d->probabilistic_total8, 8);
		CombineLevel(s->probabilistic_total16, d->probabilistic_total16, 16);
		CombineLevel(s->probabilistic_total32, d->probabilistic_total32, 32);
		CombineLevel(s->probabilistic_total64, d->probabilistic_total64, 64);
		CombineLevel(s->probabilistic_total128, d->probabilistic_total128, 64);
#endif
		d->exact_count += s->exact_count;
	}
}

// Combine for double state
AUTOVECTORIZE static void PacSumCombineDouble(Vector &src, Vector &dst, idx_t count, ArenaAllocator &allocator) {
	auto src_wrapper = FlatVector::GetData<ScatterDoubleState *>(src);
	auto dst_wrapper = FlatVector::GetData<ScatterDoubleState *>(dst);
	for (idx_t i = 0; i < count; i++) {
#ifndef PAC_NOBUFFERING
		// Flush src's buffer into dst's inner state (avoids allocating src inner)
		PacSumFlushBuffer<true>(*src_wrapper[i], *dst_wrapper[i], allocator);
#endif
		auto *s = src_wrapper[i]->GetState();
		auto *d_wrapper = dst_wrapper[i];
#ifdef PAC_UNSAFENULL
		if (s && s->seen_null) {
			auto *d = d_wrapper->EnsureState(allocator);
			d->seen_null = true;
		}
		auto *d_inner = d_wrapper->GetState();
		if (d_inner && d_inner->seen_null) {
			continue;
		}
#endif
		// Merge exact_count from wrapper
#ifndef PAC_NOBUFFERING
		d_wrapper->exact_count += src_wrapper[i]->exact_count;
#endif
		if (!s) {
			continue; // src has no state allocated
		}
		auto *d = d_wrapper->EnsureState(allocator);
		d->key_hash |= s->key_hash;
		d->exact_count += s->exact_count;
		for (int j = 0; j < 64; j++) {
			d->probabilistic_total[j] += s->probabilistic_total[j];
		}
	}
}

// Combine for approx state - handles shift amount merging and base accumulation
// NOTE: base and lost_precision are in ORIGINAL domain, so just add them directly
template <bool SIGNED>
AUTOVECTORIZE static void PacApproxSumCombine(Vector &src, Vector &dst, idx_t count, ArenaAllocator &allocator) {
	using T16 = typename std::conditional<SIGNED, int16_t, uint16_t>::type;
	using T32 = typename std::conditional<SIGNED, int32_t, uint32_t>::type;
	auto src_wrapper = FlatVector::GetData<ScatterApproxState<SIGNED> *>(src);
	auto dst_wrapper = FlatVector::GetData<ScatterApproxState<SIGNED> *>(dst);

	for (idx_t i = 0; i < count; i++) {
#ifndef PAC_NOBUFFERING
		// Flush src's buffer into dst's inner state (avoids allocating src inner)
		PacApproxFlushBuffer<SIGNED>(*src_wrapper[i], *dst_wrapper[i], allocator);
#endif
		auto *s = src_wrapper[i]->GetState();
		auto *d_wrapper = dst_wrapper[i];

		// Merge exact_count from wrapper
#ifndef PAC_NOBUFFERING
		d_wrapper->exact_count += src_wrapper[i]->exact_count;
#endif
		if (!s) {
			continue;
		}
		auto *d = d_wrapper->EnsureState(allocator);
		d->key_hash |= s->key_hash;
		d->exact_count += s->exact_count;

		// Merge bases (in original domain, just add)
		d->base_value += s->base_value;
		d->base_rebal += s->base_rebal;
		d->base_counter += s->base_counter;

		// Get actual max magnitudes from both states
		int32_t d_max = ApproxGetMaxMagnitude<SIGNED>(*d);
		int32_t s_max = ApproxGetMaxMagnitude<SIGNED>(*s);

		// Align shift amounts - bring dst up to src's shift if src is higher
		// INVARIANT: base_value += (low_bits << shift) before shifting bound
		if (s->shift_amount > d->shift_amount) {
			int8_t shift_diff = s->shift_amount - d->shift_amount;
			// Preserve low bits of d->bound in base_value before shifting
			T32 low_bits = d->bound & ((static_cast<T32>(1) << shift_diff) - 1);
			d->base_value += static_cast<double>(low_bits) * static_cast<double>(1ULL << d->shift_amount);
			d->bound >>= shift_diff;
			// Track lost bits from counters before shifting them (per-counter loss)
			int64_t counter_lost = ApproxSumLowBits<SIGNED>(d->probabilistic_total16, shift_diff);
			d->base_counter += static_cast<double>(counter_lost) * static_cast<double>(1ULL << d->shift_amount);
			ApproxShiftArrayRight<SIGNED>(d->probabilistic_total16, shift_diff);
			d->shift_amount = s->shift_amount;
			d_max >>= shift_diff;
		}

		// Compute src's max at dst's shift level
		int8_t shift_diff = d->shift_amount - s->shift_amount;
		int32_t s_max_adjusted = s_max >> shift_diff;

		// Estimate combined max (conservative: sum of both maxes)
		int64_t combined_max = static_cast<int64_t>(d_max) + s_max_adjusted;

		// Check if combined might overflow
		if (combined_max > APPROX_UPPER) {
			// Try rebalancing dst first
			if (d->probabilistic_total16) {
				int16_t min_val, max_val;
				ApproxFindMinMax<SIGNED>(d->probabilistic_total16, min_val, max_val);
				int16_t delta = SIGNED ? (max_val > -min_val ? min_val : max_val) : min_val;
				if (delta != 0) {
					ApproxSubtractFromAll<SIGNED>(d->probabilistic_total16, delta);
					d->base_rebal += static_cast<double>(delta) * 64.0 * static_cast<double>(1ULL << d->shift_amount);
					ApproxFindMinMax<SIGNED>(d->probabilistic_total16, min_val, max_val);
					d_max = SIGNED ? std::max(static_cast<int32_t>(max_val), static_cast<int32_t>(-min_val))
					               : static_cast<int32_t>(max_val);
					// Reset bound to actual max counter value for tighter overflow detection
					d->bound = static_cast<T32>(d_max);
				}
				combined_max = static_cast<int64_t>(d_max) + s_max_adjusted;
			}

			// If still would overflow, shift right
			if (combined_max > APPROX_UPPER && d->shift_amount < 112) {
				int8_t extra_shift = MIN_SHIFT_INCREMENT;
				while (d->shift_amount + extra_shift < 112) {
					if ((combined_max >> extra_shift) <= APPROX_UPPER) {
						break;
					}
					extra_shift++;
				}

				// INVARIANT: preserve low bits of bound in base_value
				T32 low_bits = d->bound & ((static_cast<T32>(1) << extra_shift) - 1);
				d->base_value += static_cast<double>(low_bits) * static_cast<double>(1ULL << d->shift_amount);
				d->bound >>= extra_shift;

				// Track lost bits from counters before shifting them (per-counter loss)
				int64_t counter_lost = ApproxSumLowBits<SIGNED>(d->probabilistic_total16, extra_shift);
				d->base_counter += static_cast<double>(counter_lost) * static_cast<double>(1ULL << d->shift_amount);

				ApproxShiftArrayRight<SIGNED>(d->probabilistic_total16, extra_shift);
				d->shift_amount += extra_shift;
				shift_diff = d->shift_amount - s->shift_amount;
				s_max_adjusted = s_max >> shift_diff;
			}
		}

		// Add src counters (shifted to match dst's shift_amount)
		shift_diff = d->shift_amount - s->shift_amount;
		if (s->probabilistic_total16 && d->probabilistic_total16) {
			const T16 *src_c = reinterpret_cast<const T16 *>(s->probabilistic_total16);
			T16 *dst_c = reinterpret_cast<T16 *>(d->probabilistic_total16);
			for (int j = 0; j < 64; j++) {
				dst_c[j] += static_cast<T16>(src_c[j] >> shift_diff);
			}
		} else if (s->probabilistic_total16) {
			if (shift_diff == 0) {
				d->probabilistic_total16 = s->probabilistic_total16;
			} else {
				d->probabilistic_total16 = reinterpret_cast<uint64_t *>(allocator.Allocate(16 * sizeof(uint64_t)));
				const T16 *src_c = reinterpret_cast<const T16 *>(s->probabilistic_total16);
				T16 *dst_c = reinterpret_cast<T16 *>(d->probabilistic_total16);
				for (int j = 0; j < 64; j++) {
					dst_c[j] = static_cast<T16>(src_c[j] >> shift_diff);
				}
			}
		}

		// Merge bounds: preserve low bits of s->bound in base_value, then add shifted bound
		if (shift_diff > 0) {
			T32 s_low_bits = s->bound & ((static_cast<T32>(1) << shift_diff) - 1);
			d->base_value += static_cast<double>(s_low_bits) * static_cast<double>(1ULL << s->shift_amount);
		}
		d->bound += static_cast<T32>(s->bound >> shift_diff);
	}
}

// Unified Finalize for both int and double states
template <class State, class ACC_TYPE, bool SIGNED, bool DIVIDE_BY_COUNT = false>
static void PacSumFinalize(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	auto state_ptrs = FlatVector::GetData<State *>(states);
	auto data = FlatVector::GetData<ACC_TYPE>(result);
	auto &result_mask = FlatVector::Validity(result);
	uint64_t seed = input.bind_data ? input.bind_data->Cast<PacBindData>().seed : std::random_device {}();
	std::mt19937_64 gen(seed);
	double mi = input.bind_data ? input.bind_data->Cast<PacBindData>().mi : 128.0;
	// scale_divisor is used by pac_avg on DECIMAL to convert internal integer representation back to decimal
	double scale_divisor = input.bind_data ? input.bind_data->Cast<PacBindData>().scale_divisor : 1.0;

	for (idx_t i = 0; i < count; i++) {
#ifndef PAC_NOBUFFERING
		PacSumFlushBuffer<SIGNED>(*state_ptrs[i], *state_ptrs[i], input.allocator);
#endif
		auto *s = state_ptrs[i]->GetState();
#ifdef PAC_UNSAFENULL
		if (s && s->seen_null) {
			result_mask.SetInvalid(offset + i);
			continue;
		}
#endif
		// Check if we should return NULL based on key_hash
		uint64_t key_hash = s ? s->key_hash : 0;
		if (PacNoiseInNull(key_hash, mi, gen)) {
			result_mask.SetInvalid(offset + i);
			continue;
		}
		double buf[64];
		if (s) {
			s->Flush(input.allocator);
			s->GetTotalsAsDouble(buf, &gen);
		} else {
			memset(buf, 0, sizeof(buf));
		}
		if (DIVIDE_BY_COUNT) {
			// Total count = wrapper's count + inner state's count
			uint64_t total_count = state_ptrs[i]->exact_count;
#ifndef PAC_NOBUFFERING
			if (s) {
				total_count += s->exact_count;
			}
#endif
			double divisor = static_cast<double>(total_count) * scale_divisor;
			for (int j = 0; j < 64; j++) {
				buf[j] /= divisor;
			}
		}
		data[offset + i] = FromDouble<ACC_TYPE>(PacNoisySampleFrom64Counters(buf, mi, gen, true, ~key_hash));
	}
}

// Finalize for approx states (uses PacApproxFlushBuffer instead of PacSumFlushBuffer)
template <class ACC_TYPE, bool SIGNED, bool DIVIDE_BY_COUNT = false>
static void PacApproxFinalize(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	auto state_ptrs = FlatVector::GetData<ScatterApproxState<SIGNED> *>(states);
	auto data = FlatVector::GetData<ACC_TYPE>(result);
	auto &result_mask = FlatVector::Validity(result);
	uint64_t seed = input.bind_data ? input.bind_data->Cast<PacBindData>().seed : std::random_device {}();
	std::mt19937_64 gen(seed);
	double mi = input.bind_data ? input.bind_data->Cast<PacBindData>().mi : 128.0;
	double scale_divisor = input.bind_data ? input.bind_data->Cast<PacBindData>().scale_divisor : 1.0;

	for (idx_t i = 0; i < count; i++) {
#ifndef PAC_NOBUFFERING
		PacApproxFlushBuffer<SIGNED>(*state_ptrs[i], *state_ptrs[i], input.allocator);
#endif
		auto *s = state_ptrs[i]->GetState();
		// Check if we should return NULL based on key_hash
		uint64_t key_hash = s ? s->key_hash : 0;
		if (PacNoiseInNull(key_hash, mi, gen)) {
			result_mask.SetInvalid(offset + i);
			continue;
		}
		double buf[64];
		if (s) {
			s->Flush(input.allocator);
			// DEBUG: uncomment to print shift_amount and bases
			// fprintf(stderr, "[approx] shift=%d base_value=%.0f base_rebal=%.0f base_counter=%.0f count=%llu\n",
			//         (int)s->shift_amount, s->base_value, s->base_rebal, s->base_counter, (unsigned long
			//         long)s->exact_count);
			s->GetTotalsAsDouble(buf, &gen);
		} else {
			memset(buf, 0, sizeof(buf));
		}
		if (DIVIDE_BY_COUNT) {
			// Total count = wrapper's count + inner state's count
			uint64_t total_count = state_ptrs[i]->exact_count;
#ifndef PAC_NOBUFFERING
			if (s) {
				total_count += s->exact_count;
			}
#endif
			double divisor = static_cast<double>(total_count) * scale_divisor;
			for (int j = 0; j < 64; j++) {
				buf[j] /= divisor;
			}
		}
		// For approx aggregates, ALWAYS use mean-based finalization.
		// The correction factor in GetTotalsAsDouble scales counters so their AVERAGE is correct,
		// but individual counters retain truncation-induced bias from right-shifts.
		// Using PacNoisySampleFrom64Counters (which picks ONE counter) would expose that bias.
		// Instead, we compute the mean (which averages out bias) and add PAC noise based on mi.
		double sum = 0.0;
		double sum_sq = 0.0;
		int valid_count = 0;
		for (int j = 0; j < 64; j++) {
			if (!((~key_hash >> j) & 1)) {
				sum += buf[j];
				sum_sq += buf[j] * buf[j];
				valid_count++;
			}
		}
		if (valid_count == 0) {
			data[offset + i] = FromDouble<ACC_TYPE>(0.0);
		} else {
			double mean = sum / valid_count;
			double result = 2.0 * mean; // Each counter estimates sum/2, so mean*2 = sum estimate

			if (mi > 0.0 && valid_count > 1) {
				// Add PAC noise: variance = sigma2 / (2 * mi), where sigma2 is counter variance
				// For mean of N counters, variance of mean = sigma2/N, so noise variance = sigma2/(2*mi*N)
				double variance = (sum_sq / valid_count) - (mean * mean);
				if (variance > 0.0) {
					double delta = variance / (2.0 * mi * valid_count);
					// Sample noise using deterministic Box-Muller (platform-agnostic)
					constexpr double inv2pow53 = 1.0 / 9007199254740992.0; // 1 / 2^53
					double u1 = static_cast<double>(gen() >> 11) * inv2pow53;
					double u2 = static_cast<double>(gen() >> 11) * inv2pow53;
					if (u1 <= 0.0)
						u1 = std::numeric_limits<double>::min();
					double noise = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2) * std::sqrt(delta);
					result += noise;
				}
			}
			data[offset + i] = FromDouble<ACC_TYPE>(result);
		}
	}
}

// instantiate Update methods
void PacSumUpdateTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<ScatterIntState<true>, true, int64_t, int8_t>(inputs, state_p, count, aggr.allocator,
	                                                           aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumUpdateSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<ScatterIntState<true>, true, int64_t, int16_t>(inputs, state_p, count, aggr.allocator,
	                                                            aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumUpdateInteger(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<ScatterIntState<true>, true, int64_t, int32_t>(inputs, state_p, count, aggr.allocator,
	                                                            aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumUpdateBigInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<ScatterIntState<true>, true, int64_t, int64_t>(inputs, state_p, count, aggr.allocator,
	                                                            aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumUpdateHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<ScatterIntState<true>, true, hugeint_t, hugeint_t>(inputs, state_p, count, aggr.allocator,
	                                                                aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumUpdateUTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<ScatterIntState<false>, false, uint64_t, uint8_t>(inputs, state_p, count, aggr.allocator,
	                                                               aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumUpdateUSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<ScatterIntState<false>, false, uint64_t, uint16_t>(inputs, state_p, count, aggr.allocator,
	                                                                aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumUpdateUInteger(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<ScatterIntState<false>, false, uint64_t, uint32_t>(inputs, state_p, count, aggr.allocator,
	                                                                aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumUpdateUBigInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<ScatterIntState<false>, false, uint64_t, uint64_t>(inputs, state_p, count, aggr.allocator,
	                                                                aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumUpdateUHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<ScatterDoubleState, true, double, uhugeint_t>(inputs, state_p, count, aggr.allocator,
	                                                           aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumUpdateFloat(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<ScatterDoubleState, true, double, float>(inputs, state_p, count, aggr.allocator,
	                                                      aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumUpdateDouble(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<ScatterDoubleState, true, double, double>(inputs, state_p, count, aggr.allocator,
	                                                       aggr.bind_data->Cast<PacBindData>().query_hash);
}

// instantiate ScatterUpdate methods
void PacSumScatterUpdateTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<ScatterIntState<true>, true, int64_t, int8_t>(inputs, states, count, aggr.allocator,
	                                                                  aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumScatterUpdateSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<ScatterIntState<true>, true, int64_t, int16_t>(inputs, states, count, aggr.allocator,
	                                                                   aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumScatterUpdateInteger(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<ScatterIntState<true>, true, int64_t, int32_t>(inputs, states, count, aggr.allocator,
	                                                                   aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumScatterUpdateBigInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<ScatterIntState<true>, true, int64_t, int64_t>(inputs, states, count, aggr.allocator,
	                                                                   aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumScatterUpdateHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<ScatterIntState<true>, true, hugeint_t, hugeint_t>(
	    inputs, states, count, aggr.allocator, aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumScatterUpdateUTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<ScatterIntState<false>, false, uint64_t, uint8_t>(
	    inputs, states, count, aggr.allocator, aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumScatterUpdateUSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<ScatterIntState<false>, false, uint64_t, uint16_t>(
	    inputs, states, count, aggr.allocator, aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumScatterUpdateUInteger(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<ScatterIntState<false>, false, uint64_t, uint32_t>(
	    inputs, states, count, aggr.allocator, aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumScatterUpdateUBigInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<ScatterIntState<false>, false, uint64_t, uint64_t>(
	    inputs, states, count, aggr.allocator, aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumScatterUpdateUHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<ScatterDoubleState, true, double, uhugeint_t>(inputs, states, count, aggr.allocator,
	                                                                  aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumScatterUpdateFloat(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<ScatterDoubleState, true, double, float>(inputs, states, count, aggr.allocator,
	                                                             aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacSumScatterUpdateDouble(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<ScatterDoubleState, true, double, double>(inputs, states, count, aggr.allocator,
	                                                              aggr.bind_data->Cast<PacBindData>().query_hash);
}

// instantiate Combine methods
void PacSumCombineSigned(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	PacSumCombineInt<true>(src, dst, count, aggr.allocator);
}
void PacSumCombineUnsigned(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	PacSumCombineInt<false>(src, dst, count, aggr.allocator);
}
void PacSumCombineDoubleWrapper(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	PacSumCombineDouble(src, dst, count, aggr.allocator);
}

// instantiate Finalize methods for pac_sum
void PacSumFinalizeSigned(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	PacSumFinalize<ScatterIntState<true>, hugeint_t, true>(states, input, result, count, offset);
}
void PacSumFinalizeUnsigned(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	PacSumFinalize<ScatterIntState<false>, hugeint_t, false>(states, input, result, count, offset);
}
void PacSumFinalizeDoubleWrapper(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	PacSumFinalize<ScatterDoubleState, double, true>(states, input, result, count, offset);
}

// instantiate Finalize methods for pac_avg (with DIVIDE_BY_COUNT=true)
void PacAvgFinalizeDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	PacSumFinalize<ScatterDoubleState, double, true, true>(states, input, result, count, offset);
}
void PacAvgFinalizeSignedDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	PacSumFinalize<ScatterIntState<true>, double, true, true>(states, input, result, count, offset);
}
void PacAvgFinalizeUnsignedDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count,
                                  idx_t offset) {
	PacSumFinalize<ScatterIntState<false>, double, false, true>(states, input, result, count, offset);
}

static unique_ptr<FunctionData> // Bind function for pac_sum with optional mi parameter (must be constant)
PacSumBind(ClientContext &ctx, AggregateFunction &, vector<unique_ptr<Expression>> &args) {
	double mi = 128.0; // default
	if (args.size() >= 3) {
		if (!args[2]->IsFoldable()) {
			throw InvalidInputException("pac_sum: mi parameter must be a constant");
		}
		auto mi_val = ExpressionExecutor::EvaluateScalar(ctx, *args[2]);
		mi = mi_val.GetValue<double>();
		if (mi < 0.0) {
			throw InvalidInputException("pac_sum: mi must be >= 0");
		}
	}
	// Read pac_seed setting (optional) to produce deterministic RNG seed for tests
	uint64_t seed = std::random_device {}();
	Value pac_seed_val;
	if (ctx.TryGetCurrentSetting("pac_seed", pac_seed_val) && !pac_seed_val.IsNull()) {
		seed = uint64_t(pac_seed_val.GetValue<int64_t>());
	}
	return make_uniq<PacBindData>(mi, seed);
}

static idx_t PacSumIntStateSize(const AggregateFunction &) {
	return sizeof(ScatterIntState<true>); // signed/unsigned have same wrapper size
}

static void PacSumIntInitialize(const AggregateFunction &, data_ptr_t state_p) {
	memset(state_p, 0, sizeof(ScatterIntState<true>));
}

static idx_t PacSumDoubleStateSize(const AggregateFunction &) {
	return sizeof(ScatterDoubleState);
}
static void PacSumDoubleInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	memset(state_ptr, 0, sizeof(ScatterDoubleState));
}

// Helper to register both 2-param and 3-param (with optional mi) versions
static void AddFcn(AggregateFunctionSet &set, const LogicalType &value_type, const LogicalType &result_type,
                   aggregate_size_t state_size, aggregate_initialize_t init, aggregate_update_t scatter,
                   aggregate_combine_t combine, aggregate_finalize_t finalize, aggregate_simple_update_t update,
                   aggregate_destructor_t destructor = nullptr) {
	set.AddFunction(AggregateFunction("pac_sum", {LogicalType::UBIGINT, value_type}, result_type, state_size, init,
	                                  scatter, combine, finalize, FunctionNullHandling::DEFAULT_NULL_HANDLING, update,
	                                  PacSumBind, destructor));
	set.AddFunction(AggregateFunction("pac_sum", {LogicalType::UBIGINT, value_type, LogicalType::DOUBLE}, result_type,
	                                  state_size, init, scatter, combine, finalize,
	                                  FunctionNullHandling::DEFAULT_NULL_HANDLING, update, PacSumBind, destructor));
}

// Helper to get the right pac_sum AggregateFunction for a given physical type (used by BindDecimalPacSum)
// Note: bind is set to nullptr - the caller (BindDecimalPacSum) handles binding
static AggregateFunction GetPacSumAggregate(PhysicalType type) {
	switch (type) {
	case PhysicalType::INT16:
		return AggregateFunction("pac_sum", {LogicalType::UBIGINT, LogicalType::SMALLINT}, LogicalType::HUGEINT,
		                         PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateSmallInt,
		                         PacSumCombineSigned, PacSumFinalizeSigned, FunctionNullHandling::DEFAULT_NULL_HANDLING,
		                         PacSumUpdateSmallInt);
	case PhysicalType::INT32:
		return AggregateFunction("pac_sum", {LogicalType::UBIGINT, LogicalType::INTEGER}, LogicalType::HUGEINT,
		                         PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateInteger,
		                         PacSumCombineSigned, PacSumFinalizeSigned, FunctionNullHandling::DEFAULT_NULL_HANDLING,
		                         PacSumUpdateInteger);
	case PhysicalType::INT64:
		return AggregateFunction("pac_sum", {LogicalType::UBIGINT, LogicalType::BIGINT}, LogicalType::HUGEINT,
		                         PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateBigInt,
		                         PacSumCombineSigned, PacSumFinalizeSigned, FunctionNullHandling::DEFAULT_NULL_HANDLING,
		                         PacSumUpdateBigInt);
	case PhysicalType::INT128:
		return AggregateFunction("pac_sum", {LogicalType::UBIGINT, LogicalType::HUGEINT}, LogicalType::HUGEINT,
		                         PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateHugeInt,
		                         PacSumCombineSigned, PacSumFinalizeSigned, FunctionNullHandling::DEFAULT_NULL_HANDLING,
		                         PacSumUpdateHugeInt);
	default:
		throw InternalException("Unsupported physical type for pac_sum decimal");
	}
}

// Dynamic dispatch for DECIMAL: selects the right integer implementation based on decimal width
static unique_ptr<FunctionData> BindDecimalPacSum(ClientContext &ctx, AggregateFunction &function,
                                                  vector<unique_ptr<Expression>> &args) {
	auto decimal_type = args[1]->return_type; // value is arg 1 (arg 0 is hash)
	function = GetPacSumAggregate(decimal_type.InternalType());
	function.name = "pac_sum";
	function.arguments[1] = decimal_type;
	function.return_type = LogicalType::DECIMAL(Decimal::MAX_WIDTH_DECIMAL, DecimalType::GetScale(decimal_type));
	// Get mi and seed
	return PacSumBind(ctx, function, args);
}

void RegisterPacSumFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet fcn_set("pac_sum");

	// Signed integers (accumulate to hugeint_t, return HUGEINT)
	AddFcn(fcn_set, LogicalType::TINYINT, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterUpdateTinyInt, PacSumCombineSigned, PacSumFinalizeSigned, PacSumUpdateTinyInt);
	AddFcn(fcn_set, LogicalType::BOOLEAN, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterUpdateTinyInt, PacSumCombineSigned, PacSumFinalizeSigned, PacSumUpdateTinyInt);
	AddFcn(fcn_set, LogicalType::SMALLINT, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterUpdateSmallInt, PacSumCombineSigned, PacSumFinalizeSigned, PacSumUpdateSmallInt);
	AddFcn(fcn_set, LogicalType::INTEGER, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterUpdateInteger, PacSumCombineSigned, PacSumFinalizeSigned, PacSumUpdateInteger);
	AddFcn(fcn_set, LogicalType::BIGINT, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterUpdateBigInt, PacSumCombineSigned, PacSumFinalizeSigned, PacSumUpdateBigInt);

	// Unsigned integers (idem)
	AddFcn(fcn_set, LogicalType::UTINYINT, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterUpdateUTinyInt, PacSumCombineUnsigned, PacSumFinalizeUnsigned, PacSumUpdateUTinyInt);
	AddFcn(fcn_set, LogicalType::USMALLINT, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterUpdateUSmallInt, PacSumCombineUnsigned, PacSumFinalizeUnsigned, PacSumUpdateUSmallInt);
	AddFcn(fcn_set, LogicalType::UINTEGER, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterUpdateUInteger, PacSumCombineUnsigned, PacSumFinalizeUnsigned, PacSumUpdateUInteger);
	AddFcn(fcn_set, LogicalType::UBIGINT, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterUpdateUBigInt, PacSumCombineUnsigned, PacSumFinalizeUnsigned, PacSumUpdateUBigInt);

	// HUGEINT: use int state, return HUGEINT (matches DuckDB's sum behavior)
	AddFcn(fcn_set, LogicalType::HUGEINT, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterUpdateHugeInt, PacSumCombineSigned, PacSumFinalizeSigned, PacSumUpdateHugeInt);
	// UHUGEINT: DuckDB's sum returns DOUBLE for uhugeint, so we do too
	AddFcn(fcn_set, LogicalType::UHUGEINT, LogicalType::DOUBLE, PacSumDoubleStateSize, PacSumDoubleInitialize,
	       PacSumScatterUpdateUHugeInt, PacSumCombineDoubleWrapper, PacSumFinalizeDoubleWrapper, PacSumUpdateUHugeInt);

	// Floating point (accumulate to double, return DOUBLE)
	AddFcn(fcn_set, LogicalType::FLOAT, LogicalType::DOUBLE, PacSumDoubleStateSize, PacSumDoubleInitialize,
	       PacSumScatterUpdateFloat, PacSumCombineDoubleWrapper, PacSumFinalizeDoubleWrapper, PacSumUpdateFloat);
	AddFcn(fcn_set, LogicalType::DOUBLE, LogicalType::DOUBLE, PacSumDoubleStateSize, PacSumDoubleInitialize,
	       PacSumScatterUpdateDouble, PacSumCombineDoubleWrapper, PacSumFinalizeDoubleWrapper, PacSumUpdateDouble);

	// DECIMAL: dynamic dispatch based on decimal width (like DuckDB's sum)
	// Uses BindDecimalPacSum to select INT16/INT32/INT64/INT128 implementation at bind time
	fcn_set.AddFunction(AggregateFunction({LogicalType::UBIGINT, LogicalTypeId::DECIMAL}, LogicalTypeId::DECIMAL,
	                                      nullptr, nullptr, nullptr, nullptr, nullptr,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, BindDecimalPacSum));
	fcn_set.AddFunction(AggregateFunction({LogicalType::UBIGINT, LogicalTypeId::DECIMAL, LogicalType::DOUBLE},
	                                      LogicalTypeId::DECIMAL, nullptr, nullptr, nullptr, nullptr, nullptr,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, BindDecimalPacSum));

	loader.RegisterFunction(fcn_set);
}

// Helper to get the right pac_avg AggregateFunction for a given physical type (used by BindDecimalPacAvg)
// Note: bind is set to nullptr - the caller (BindDecimalPacAvg) handles binding
static AggregateFunction GetPacAvgAggregate(PhysicalType type) {
	switch (type) {
	case PhysicalType::INT16:
		return AggregateFunction("pac_avg", {LogicalType::UBIGINT, LogicalType::SMALLINT}, LogicalType::DOUBLE,
		                         PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateSmallInt,
		                         PacSumCombineSigned, PacAvgFinalizeSignedDouble,
		                         FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateSmallInt);
	case PhysicalType::INT32:
		return AggregateFunction("pac_avg", {LogicalType::UBIGINT, LogicalType::INTEGER}, LogicalType::DOUBLE,
		                         PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateInteger,
		                         PacSumCombineSigned, PacAvgFinalizeSignedDouble,
		                         FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateInteger);
	case PhysicalType::INT64:
		return AggregateFunction("pac_avg", {LogicalType::UBIGINT, LogicalType::BIGINT}, LogicalType::DOUBLE,
		                         PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateBigInt,
		                         PacSumCombineSigned, PacAvgFinalizeSignedDouble,
		                         FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateBigInt);
	case PhysicalType::INT128:
		return AggregateFunction("pac_avg", {LogicalType::UBIGINT, LogicalType::HUGEINT}, LogicalType::DOUBLE,
		                         PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateHugeInt,
		                         PacSumCombineSigned, PacAvgFinalizeSignedDouble,
		                         FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateHugeInt);
	default:
		throw InternalException("Unsupported physical type for pac_avg decimal");
	}
}

// Dynamic dispatch for DECIMAL: selects the right integer implementation based on decimal width
static unique_ptr<FunctionData> BindDecimalPacAvg(ClientContext &ctx, AggregateFunction &function,
                                                  vector<unique_ptr<Expression>> &args) {
	auto decimal_type = args[1]->return_type; // value is arg 1 (arg 0 is hash)
	function = GetPacAvgAggregate(decimal_type.InternalType());
	function.name = "pac_avg";
	function.arguments[1] = decimal_type;
	// pac_avg always returns DOUBLE (like DuckDB's avg)
	function.return_type = LogicalType::DOUBLE;

	// Compute scale_divisor = 10^scale for DECIMAL types
	// This converts the internal integer representation back to the decimal value
	uint8_t scale = DecimalType::GetScale(decimal_type);
	double scale_divisor = std::pow(10.0, static_cast<double>(scale));

	// Get mi and seed (same as PacSumBind)
	double mi = 128.0;
	if (args.size() >= 3) {
		if (!args[2]->IsFoldable()) {
			throw InvalidInputException("pac_avg: mi parameter must be a constant");
		}
		auto mi_val = ExpressionExecutor::EvaluateScalar(ctx, *args[2]);
		mi = mi_val.GetValue<double>();
		if (mi < 0.0) {
			throw InvalidInputException("pac_avg: mi must be >= 0");
		}
	}
	uint64_t seed = std::random_device {}();
	Value pac_seed_val;
	if (ctx.TryGetCurrentSetting("pac_seed", pac_seed_val) && !pac_seed_val.IsNull()) {
		seed = uint64_t(pac_seed_val.GetValue<int64_t>());
	}
	return make_uniq<PacBindData>(mi, seed, scale_divisor);
}

// Helper to register both 2-param and 3-param (with optional mi) versions for pac_avg
static void AddAvgFcn(AggregateFunctionSet &set, const LogicalType &value_type, aggregate_size_t state_size,
                      aggregate_initialize_t init, aggregate_update_t scatter, aggregate_combine_t combine,
                      aggregate_finalize_t finalize, aggregate_simple_update_t update,
                      aggregate_destructor_t destructor = nullptr) {
	// pac_avg always returns DOUBLE (like DuckDB's avg)
	set.AddFunction(AggregateFunction("pac_avg", {LogicalType::UBIGINT, value_type}, LogicalType::DOUBLE, state_size,
	                                  init, scatter, combine, finalize, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	                                  update, PacSumBind, destructor));
	set.AddFunction(AggregateFunction("pac_avg", {LogicalType::UBIGINT, value_type, LogicalType::DOUBLE},
	                                  LogicalType::DOUBLE, state_size, init, scatter, combine, finalize,
	                                  FunctionNullHandling::DEFAULT_NULL_HANDLING, update, PacSumBind, destructor));
}

void RegisterPacAvgFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet fcn_set("pac_avg");

	// Signed integers (use int state, avg finalize returns DOUBLE)
	AddAvgFcn(fcn_set, LogicalType::TINYINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateTinyInt,
	          PacSumCombineSigned, PacAvgFinalizeSignedDouble, PacSumUpdateTinyInt);
	AddAvgFcn(fcn_set, LogicalType::BOOLEAN, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateTinyInt,
	          PacSumCombineSigned, PacAvgFinalizeSignedDouble, PacSumUpdateTinyInt);
	AddAvgFcn(fcn_set, LogicalType::SMALLINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateSmallInt,
	          PacSumCombineSigned, PacAvgFinalizeSignedDouble, PacSumUpdateSmallInt);
	AddAvgFcn(fcn_set, LogicalType::INTEGER, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateInteger,
	          PacSumCombineSigned, PacAvgFinalizeSignedDouble, PacSumUpdateInteger);
	AddAvgFcn(fcn_set, LogicalType::BIGINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateBigInt,
	          PacSumCombineSigned, PacAvgFinalizeSignedDouble, PacSumUpdateBigInt);

	// Unsigned integers
	AddAvgFcn(fcn_set, LogicalType::UTINYINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateUTinyInt,
	          PacSumCombineUnsigned, PacAvgFinalizeUnsignedDouble, PacSumUpdateUTinyInt);
	AddAvgFcn(fcn_set, LogicalType::USMALLINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateUSmallInt,
	          PacSumCombineUnsigned, PacAvgFinalizeUnsignedDouble, PacSumUpdateUSmallInt);
	AddAvgFcn(fcn_set, LogicalType::UINTEGER, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateUInteger,
	          PacSumCombineUnsigned, PacAvgFinalizeUnsignedDouble, PacSumUpdateUInteger);
	AddAvgFcn(fcn_set, LogicalType::UBIGINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateUBigInt,
	          PacSumCombineUnsigned, PacAvgFinalizeUnsignedDouble, PacSumUpdateUBigInt);

	// HUGEINT
	AddAvgFcn(fcn_set, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateHugeInt,
	          PacSumCombineSigned, PacAvgFinalizeSignedDouble, PacSumUpdateHugeInt);
	// UHUGEINT (uses double state)
	AddAvgFcn(fcn_set, LogicalType::UHUGEINT, PacSumDoubleStateSize, PacSumDoubleInitialize,
	          PacSumScatterUpdateUHugeInt, PacSumCombineDoubleWrapper, PacAvgFinalizeDouble, PacSumUpdateUHugeInt);

	// Floating point (uses double state)
	AddAvgFcn(fcn_set, LogicalType::FLOAT, PacSumDoubleStateSize, PacSumDoubleInitialize, PacSumScatterUpdateFloat,
	          PacSumCombineDoubleWrapper, PacAvgFinalizeDouble, PacSumUpdateFloat);
	AddAvgFcn(fcn_set, LogicalType::DOUBLE, PacSumDoubleStateSize, PacSumDoubleInitialize, PacSumScatterUpdateDouble,
	          PacSumCombineDoubleWrapper, PacAvgFinalizeDouble, PacSumUpdateDouble);

	// DECIMAL: dynamic dispatch based on decimal width (like DuckDB's avg)
	// Uses BindDecimalPacAvg to select INT16/INT32/INT64/INT128 implementation at bind time
	fcn_set.AddFunction(AggregateFunction({LogicalType::UBIGINT, LogicalTypeId::DECIMAL}, LogicalType::DOUBLE, nullptr,
	                                      nullptr, nullptr, nullptr, nullptr,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, BindDecimalPacAvg));
	fcn_set.AddFunction(AggregateFunction({LogicalType::UBIGINT, LogicalTypeId::DECIMAL, LogicalType::DOUBLE},
	                                      LogicalType::DOUBLE, nullptr, nullptr, nullptr, nullptr, nullptr,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, BindDecimalPacAvg));

	loader.RegisterFunction(fcn_set);
}

// ============================================================================
// PAC_SUM_APPROX / PAC_AVG_APPROX registration
// ============================================================================

// State size and initialize for approx state
static idx_t PacApproxStateSize(const AggregateFunction &) {
	return sizeof(ScatterApproxState<true>); // signed/unsigned have same wrapper size
}

static void PacApproxInitialize(const AggregateFunction &, data_ptr_t state_p) {
	memset(state_p, 0, sizeof(ScatterApproxState<true>));
}

// Instantiate Update methods for approx
void PacApproxUpdateTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacApproxUpdate<true, int64_t, int8_t>(inputs, state_p, count, aggr.allocator,
	                                       aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacApproxUpdateSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacApproxUpdate<true, int64_t, int16_t>(inputs, state_p, count, aggr.allocator,
	                                        aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacApproxUpdateInteger(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacApproxUpdate<true, int64_t, int32_t>(inputs, state_p, count, aggr.allocator,
	                                        aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacApproxUpdateBigInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacApproxUpdate<true, int64_t, int64_t>(inputs, state_p, count, aggr.allocator,
	                                        aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacApproxUpdateHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacApproxUpdate<true, hugeint_t, hugeint_t>(inputs, state_p, count, aggr.allocator,
	                                            aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacApproxUpdateUTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacApproxUpdate<false, uint64_t, uint8_t>(inputs, state_p, count, aggr.allocator,
	                                          aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacApproxUpdateUSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacApproxUpdate<false, uint64_t, uint16_t>(inputs, state_p, count, aggr.allocator,
	                                           aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacApproxUpdateUInteger(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacApproxUpdate<false, uint64_t, uint32_t>(inputs, state_p, count, aggr.allocator,
	                                           aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacApproxUpdateUBigInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacApproxUpdate<false, uint64_t, uint64_t>(inputs, state_p, count, aggr.allocator,
	                                           aggr.bind_data->Cast<PacBindData>().query_hash);
}

// Instantiate ScatterUpdate methods for approx
void PacApproxScatterUpdateTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacApproxScatterUpdate<true, int64_t, int8_t>(inputs, states, count, aggr.allocator,
	                                              aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacApproxScatterUpdateSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacApproxScatterUpdate<true, int64_t, int16_t>(inputs, states, count, aggr.allocator,
	                                               aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacApproxScatterUpdateInteger(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacApproxScatterUpdate<true, int64_t, int32_t>(inputs, states, count, aggr.allocator,
	                                               aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacApproxScatterUpdateBigInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacApproxScatterUpdate<true, int64_t, int64_t>(inputs, states, count, aggr.allocator,
	                                               aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacApproxScatterUpdateHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacApproxScatterUpdate<true, hugeint_t, hugeint_t>(inputs, states, count, aggr.allocator,
	                                                   aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacApproxScatterUpdateUTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacApproxScatterUpdate<false, uint64_t, uint8_t>(inputs, states, count, aggr.allocator,
	                                                 aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacApproxScatterUpdateUSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacApproxScatterUpdate<false, uint64_t, uint16_t>(inputs, states, count, aggr.allocator,
	                                                  aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacApproxScatterUpdateUInteger(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacApproxScatterUpdate<false, uint64_t, uint32_t>(inputs, states, count, aggr.allocator,
	                                                  aggr.bind_data->Cast<PacBindData>().query_hash);
}
void PacApproxScatterUpdateUBigInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacApproxScatterUpdate<false, uint64_t, uint64_t>(inputs, states, count, aggr.allocator,
	                                                  aggr.bind_data->Cast<PacBindData>().query_hash);
}

// Instantiate Combine methods for approx
void PacApproxCombineSigned(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	PacApproxSumCombine<true>(src, dst, count, aggr.allocator);
}
void PacApproxCombineUnsigned(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	PacApproxSumCombine<false>(src, dst, count, aggr.allocator);
}

// Instantiate Finalize methods for pac_sum_approx (returns HUGEINT)
void PacApproxFinalizeSigned(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	PacApproxFinalize<hugeint_t, true>(states, input, result, count, offset);
}
void PacApproxFinalizeUnsigned(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	PacApproxFinalize<hugeint_t, false>(states, input, result, count, offset);
}

// Instantiate Finalize methods for pac_avg_approx (returns DOUBLE, with DIVIDE_BY_COUNT=true)
void PacAvgApproxFinalizeSignedDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count,
                                      idx_t offset) {
	PacApproxFinalize<double, true, true>(states, input, result, count, offset);
}
void PacAvgApproxFinalizeUnsignedDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count,
                                        idx_t offset) {
	PacApproxFinalize<double, false, true>(states, input, result, count, offset);
}

// Helper to register both 2-param and 3-param versions for pac_sum_approx
static void AddApproxFcn(AggregateFunctionSet &set, const LogicalType &value_type, const LogicalType &result_type,
                         aggregate_size_t state_size, aggregate_initialize_t init, aggregate_update_t scatter,
                         aggregate_combine_t combine, aggregate_finalize_t finalize, aggregate_simple_update_t update,
                         aggregate_destructor_t destructor = nullptr) {
	set.AddFunction(AggregateFunction("pac_sum_approx", {LogicalType::UBIGINT, value_type}, result_type, state_size,
	                                  init, scatter, combine, finalize, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	                                  update, PacSumBind, destructor));
	set.AddFunction(AggregateFunction("pac_sum_approx", {LogicalType::UBIGINT, value_type, LogicalType::DOUBLE},
	                                  result_type, state_size, init, scatter, combine, finalize,
	                                  FunctionNullHandling::DEFAULT_NULL_HANDLING, update, PacSumBind, destructor));
}

// Helper to get the right pac_sum_approx AggregateFunction for a given physical type (used by BindDecimalPacSumApprox)
static AggregateFunction GetPacSumApproxAggregate(PhysicalType type) {
	switch (type) {
	case PhysicalType::INT16:
		return AggregateFunction("pac_sum_approx", {LogicalType::UBIGINT, LogicalType::SMALLINT}, LogicalType::HUGEINT,
		                         PacApproxStateSize, PacApproxInitialize, PacApproxScatterUpdateSmallInt,
		                         PacApproxCombineSigned, PacApproxFinalizeSigned,
		                         FunctionNullHandling::DEFAULT_NULL_HANDLING, PacApproxUpdateSmallInt);
	case PhysicalType::INT32:
		return AggregateFunction("pac_sum_approx", {LogicalType::UBIGINT, LogicalType::INTEGER}, LogicalType::HUGEINT,
		                         PacApproxStateSize, PacApproxInitialize, PacApproxScatterUpdateInteger,
		                         PacApproxCombineSigned, PacApproxFinalizeSigned,
		                         FunctionNullHandling::DEFAULT_NULL_HANDLING, PacApproxUpdateInteger);
	case PhysicalType::INT64:
		return AggregateFunction("pac_sum_approx", {LogicalType::UBIGINT, LogicalType::BIGINT}, LogicalType::HUGEINT,
		                         PacApproxStateSize, PacApproxInitialize, PacApproxScatterUpdateBigInt,
		                         PacApproxCombineSigned, PacApproxFinalizeSigned,
		                         FunctionNullHandling::DEFAULT_NULL_HANDLING, PacApproxUpdateBigInt);
	case PhysicalType::INT128:
		return AggregateFunction("pac_sum_approx", {LogicalType::UBIGINT, LogicalType::HUGEINT}, LogicalType::HUGEINT,
		                         PacApproxStateSize, PacApproxInitialize, PacApproxScatterUpdateHugeInt,
		                         PacApproxCombineSigned, PacApproxFinalizeSigned,
		                         FunctionNullHandling::DEFAULT_NULL_HANDLING, PacApproxUpdateHugeInt);
	default:
		throw InternalException("Unsupported physical type for pac_sum_approx decimal");
	}
}

// Dynamic dispatch for DECIMAL: selects the right integer implementation based on decimal width
static unique_ptr<FunctionData> BindDecimalPacSumApprox(ClientContext &ctx, AggregateFunction &function,
                                                        vector<unique_ptr<Expression>> &args) {
	auto decimal_type = args[1]->return_type;
	function = GetPacSumApproxAggregate(decimal_type.InternalType());
	function.name = "pac_sum_approx";
	function.arguments[1] = decimal_type;
	function.return_type = LogicalType::DECIMAL(Decimal::MAX_WIDTH_DECIMAL, DecimalType::GetScale(decimal_type));
	return PacSumBind(ctx, function, args);
}

void RegisterPacSumApproxFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet fcn_set("pac_sum_approx");

	// Signed integers (return HUGEINT to match pac_sum)
	AddApproxFcn(fcn_set, LogicalType::TINYINT, LogicalType::HUGEINT, PacApproxStateSize, PacApproxInitialize,
	             PacApproxScatterUpdateTinyInt, PacApproxCombineSigned, PacApproxFinalizeSigned,
	             PacApproxUpdateTinyInt);
	AddApproxFcn(fcn_set, LogicalType::BOOLEAN, LogicalType::HUGEINT, PacApproxStateSize, PacApproxInitialize,
	             PacApproxScatterUpdateTinyInt, PacApproxCombineSigned, PacApproxFinalizeSigned,
	             PacApproxUpdateTinyInt);
	AddApproxFcn(fcn_set, LogicalType::SMALLINT, LogicalType::HUGEINT, PacApproxStateSize, PacApproxInitialize,
	             PacApproxScatterUpdateSmallInt, PacApproxCombineSigned, PacApproxFinalizeSigned,
	             PacApproxUpdateSmallInt);
	AddApproxFcn(fcn_set, LogicalType::INTEGER, LogicalType::HUGEINT, PacApproxStateSize, PacApproxInitialize,
	             PacApproxScatterUpdateInteger, PacApproxCombineSigned, PacApproxFinalizeSigned,
	             PacApproxUpdateInteger);
	AddApproxFcn(fcn_set, LogicalType::BIGINT, LogicalType::HUGEINT, PacApproxStateSize, PacApproxInitialize,
	             PacApproxScatterUpdateBigInt, PacApproxCombineSigned, PacApproxFinalizeSigned, PacApproxUpdateBigInt);

	// Unsigned integers
	AddApproxFcn(fcn_set, LogicalType::UTINYINT, LogicalType::HUGEINT, PacApproxStateSize, PacApproxInitialize,
	             PacApproxScatterUpdateUTinyInt, PacApproxCombineUnsigned, PacApproxFinalizeUnsigned,
	             PacApproxUpdateUTinyInt);
	AddApproxFcn(fcn_set, LogicalType::USMALLINT, LogicalType::HUGEINT, PacApproxStateSize, PacApproxInitialize,
	             PacApproxScatterUpdateUSmallInt, PacApproxCombineUnsigned, PacApproxFinalizeUnsigned,
	             PacApproxUpdateUSmallInt);
	AddApproxFcn(fcn_set, LogicalType::UINTEGER, LogicalType::HUGEINT, PacApproxStateSize, PacApproxInitialize,
	             PacApproxScatterUpdateUInteger, PacApproxCombineUnsigned, PacApproxFinalizeUnsigned,
	             PacApproxUpdateUInteger);
	AddApproxFcn(fcn_set, LogicalType::UBIGINT, LogicalType::HUGEINT, PacApproxStateSize, PacApproxInitialize,
	             PacApproxScatterUpdateUBigInt, PacApproxCombineUnsigned, PacApproxFinalizeUnsigned,
	             PacApproxUpdateUBigInt);

	// HUGEINT
	AddApproxFcn(fcn_set, LogicalType::HUGEINT, LogicalType::HUGEINT, PacApproxStateSize, PacApproxInitialize,
	             PacApproxScatterUpdateHugeInt, PacApproxCombineSigned, PacApproxFinalizeSigned,
	             PacApproxUpdateHugeInt);

	// DECIMAL: dynamic dispatch based on decimal width
	fcn_set.AddFunction(AggregateFunction(
	    {LogicalType::UBIGINT, LogicalTypeId::DECIMAL}, LogicalTypeId::DECIMAL, nullptr, nullptr, nullptr, nullptr,
	    nullptr, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, BindDecimalPacSumApprox));
	fcn_set.AddFunction(AggregateFunction(
	    {LogicalType::UBIGINT, LogicalTypeId::DECIMAL, LogicalType::DOUBLE}, LogicalTypeId::DECIMAL, nullptr, nullptr,
	    nullptr, nullptr, nullptr, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, BindDecimalPacSumApprox));

	loader.RegisterFunction(fcn_set);
}

// Helper to register both 2-param and 3-param versions for pac_avg_approx
static void AddAvgApproxFcn(AggregateFunctionSet &set, const LogicalType &value_type, aggregate_size_t state_size,
                            aggregate_initialize_t init, aggregate_update_t scatter, aggregate_combine_t combine,
                            aggregate_finalize_t finalize, aggregate_simple_update_t update,
                            aggregate_destructor_t destructor = nullptr) {
	// pac_avg_approx always returns DOUBLE (like pac_avg)
	set.AddFunction(AggregateFunction("pac_avg_approx", {LogicalType::UBIGINT, value_type}, LogicalType::DOUBLE,
	                                  state_size, init, scatter, combine, finalize,
	                                  FunctionNullHandling::DEFAULT_NULL_HANDLING, update, PacSumBind, destructor));
	set.AddFunction(AggregateFunction("pac_avg_approx", {LogicalType::UBIGINT, value_type, LogicalType::DOUBLE},
	                                  LogicalType::DOUBLE, state_size, init, scatter, combine, finalize,
	                                  FunctionNullHandling::DEFAULT_NULL_HANDLING, update, PacSumBind, destructor));
}

// Helper to get the right pac_avg_approx AggregateFunction for a given physical type
static AggregateFunction GetPacAvgApproxAggregate(PhysicalType type) {
	switch (type) {
	case PhysicalType::INT16:
		return AggregateFunction("pac_avg_approx", {LogicalType::UBIGINT, LogicalType::SMALLINT}, LogicalType::DOUBLE,
		                         PacApproxStateSize, PacApproxInitialize, PacApproxScatterUpdateSmallInt,
		                         PacApproxCombineSigned, PacAvgApproxFinalizeSignedDouble,
		                         FunctionNullHandling::DEFAULT_NULL_HANDLING, PacApproxUpdateSmallInt);
	case PhysicalType::INT32:
		return AggregateFunction("pac_avg_approx", {LogicalType::UBIGINT, LogicalType::INTEGER}, LogicalType::DOUBLE,
		                         PacApproxStateSize, PacApproxInitialize, PacApproxScatterUpdateInteger,
		                         PacApproxCombineSigned, PacAvgApproxFinalizeSignedDouble,
		                         FunctionNullHandling::DEFAULT_NULL_HANDLING, PacApproxUpdateInteger);
	case PhysicalType::INT64:
		return AggregateFunction("pac_avg_approx", {LogicalType::UBIGINT, LogicalType::BIGINT}, LogicalType::DOUBLE,
		                         PacApproxStateSize, PacApproxInitialize, PacApproxScatterUpdateBigInt,
		                         PacApproxCombineSigned, PacAvgApproxFinalizeSignedDouble,
		                         FunctionNullHandling::DEFAULT_NULL_HANDLING, PacApproxUpdateBigInt);
	case PhysicalType::INT128:
		return AggregateFunction("pac_avg_approx", {LogicalType::UBIGINT, LogicalType::HUGEINT}, LogicalType::DOUBLE,
		                         PacApproxStateSize, PacApproxInitialize, PacApproxScatterUpdateHugeInt,
		                         PacApproxCombineSigned, PacAvgApproxFinalizeSignedDouble,
		                         FunctionNullHandling::DEFAULT_NULL_HANDLING, PacApproxUpdateHugeInt);
	default:
		throw InternalException("Unsupported physical type for pac_avg_approx decimal");
	}
}

// Dynamic dispatch for DECIMAL pac_avg_approx
static unique_ptr<FunctionData> BindDecimalPacAvgApprox(ClientContext &ctx, AggregateFunction &function,
                                                        vector<unique_ptr<Expression>> &args) {
	auto decimal_type = args[1]->return_type;
	function = GetPacAvgApproxAggregate(decimal_type.InternalType());
	function.name = "pac_avg_approx";
	function.arguments[1] = decimal_type;
	function.return_type = LogicalType::DOUBLE;

	// Compute scale_divisor = 10^scale for DECIMAL types
	uint8_t scale = DecimalType::GetScale(decimal_type);
	double scale_divisor = std::pow(10.0, static_cast<double>(scale));

	// Get mi and seed (same as PacSumBind)
	double mi = 128.0;
	if (args.size() >= 3) {
		if (!args[2]->IsFoldable()) {
			throw InvalidInputException("pac_avg_approx: mi parameter must be a constant");
		}
		auto mi_val = ExpressionExecutor::EvaluateScalar(ctx, *args[2]);
		mi = mi_val.GetValue<double>();
		if (mi < 0.0) {
			throw InvalidInputException("pac_avg_approx: mi must be >= 0");
		}
	}
	uint64_t seed = std::random_device {}();
	Value pac_seed_val;
	if (ctx.TryGetCurrentSetting("pac_seed", pac_seed_val) && !pac_seed_val.IsNull()) {
		seed = uint64_t(pac_seed_val.GetValue<int64_t>());
	}
	return make_uniq<PacBindData>(mi, seed, scale_divisor);
}

void RegisterPacAvgApproxFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet fcn_set("pac_avg_approx");

	// Signed integers (use approx state, avg finalize returns DOUBLE)
	AddAvgApproxFcn(fcn_set, LogicalType::TINYINT, PacApproxStateSize, PacApproxInitialize,
	                PacApproxScatterUpdateTinyInt, PacApproxCombineSigned, PacAvgApproxFinalizeSignedDouble,
	                PacApproxUpdateTinyInt);
	AddAvgApproxFcn(fcn_set, LogicalType::BOOLEAN, PacApproxStateSize, PacApproxInitialize,
	                PacApproxScatterUpdateTinyInt, PacApproxCombineSigned, PacAvgApproxFinalizeSignedDouble,
	                PacApproxUpdateTinyInt);
	AddAvgApproxFcn(fcn_set, LogicalType::SMALLINT, PacApproxStateSize, PacApproxInitialize,
	                PacApproxScatterUpdateSmallInt, PacApproxCombineSigned, PacAvgApproxFinalizeSignedDouble,
	                PacApproxUpdateSmallInt);
	AddAvgApproxFcn(fcn_set, LogicalType::INTEGER, PacApproxStateSize, PacApproxInitialize,
	                PacApproxScatterUpdateInteger, PacApproxCombineSigned, PacAvgApproxFinalizeSignedDouble,
	                PacApproxUpdateInteger);
	AddAvgApproxFcn(fcn_set, LogicalType::BIGINT, PacApproxStateSize, PacApproxInitialize, PacApproxScatterUpdateBigInt,
	                PacApproxCombineSigned, PacAvgApproxFinalizeSignedDouble, PacApproxUpdateBigInt);

	// Unsigned integers
	AddAvgApproxFcn(fcn_set, LogicalType::UTINYINT, PacApproxStateSize, PacApproxInitialize,
	                PacApproxScatterUpdateUTinyInt, PacApproxCombineUnsigned, PacAvgApproxFinalizeUnsignedDouble,
	                PacApproxUpdateUTinyInt);
	AddAvgApproxFcn(fcn_set, LogicalType::USMALLINT, PacApproxStateSize, PacApproxInitialize,
	                PacApproxScatterUpdateUSmallInt, PacApproxCombineUnsigned, PacAvgApproxFinalizeUnsignedDouble,
	                PacApproxUpdateUSmallInt);
	AddAvgApproxFcn(fcn_set, LogicalType::UINTEGER, PacApproxStateSize, PacApproxInitialize,
	                PacApproxScatterUpdateUInteger, PacApproxCombineUnsigned, PacAvgApproxFinalizeUnsignedDouble,
	                PacApproxUpdateUInteger);
	AddAvgApproxFcn(fcn_set, LogicalType::UBIGINT, PacApproxStateSize, PacApproxInitialize,
	                PacApproxScatterUpdateUBigInt, PacApproxCombineUnsigned, PacAvgApproxFinalizeUnsignedDouble,
	                PacApproxUpdateUBigInt);

	// HUGEINT
	AddAvgApproxFcn(fcn_set, LogicalType::HUGEINT, PacApproxStateSize, PacApproxInitialize,
	                PacApproxScatterUpdateHugeInt, PacApproxCombineSigned, PacAvgApproxFinalizeSignedDouble,
	                PacApproxUpdateHugeInt);

	// DECIMAL: dynamic dispatch based on decimal width
	fcn_set.AddFunction(AggregateFunction(
	    {LogicalType::UBIGINT, LogicalTypeId::DECIMAL}, LogicalType::DOUBLE, nullptr, nullptr, nullptr, nullptr,
	    nullptr, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, BindDecimalPacAvgApprox));
	fcn_set.AddFunction(AggregateFunction(
	    {LogicalType::UBIGINT, LogicalTypeId::DECIMAL, LogicalType::DOUBLE}, LogicalType::DOUBLE, nullptr, nullptr,
	    nullptr, nullptr, nullptr, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, BindDecimalPacAvgApprox));

	loader.RegisterFunction(fcn_set);
}

} // namespace duckdb
