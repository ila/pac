#include "include/pac_sum.hpp"
#include "duckdb/common/types/decimal.hpp"
#include <cmath>
#include <limits>
#include <unordered_map>
#include <atomic>

namespace duckdb {

// ============================================================================
// State type selection for scatter updates
// ============================================================================
#ifdef PAC_NOBUFFERING
template <bool SIGNED>
using ScatterIntState = PacSumIntState<SIGNED>;
using ScatterDoubleState = PacSumDoubleState;
#else
template <bool SIGNED>
using ScatterIntState = PacSumIntStateWrapper<SIGNED>;
using ScatterDoubleState = PacSumDoubleStateWrapper;
#endif

// ============================================================================
// Inner state update functions (work directly on PacSumIntState/PacSumDoubleState)
// ============================================================================

template <bool SIGNED>
AUTOVECTORIZE inline void // main worker function for probabilistically adding one INTEGER to the 64 (sub)total
PacSumUpdateOneInternal(PacSumIntState<SIGNED> &state, uint64_t key_hash, typename PacSumIntState<SIGNED>::T64 value,
                        ArenaAllocator &allocator) {
	state.key_hash |= key_hash;
#ifndef PAC_EXACTSUM
	// APPROX ALGORITHM: route by bit position, shift value
	// 16-bit counters with 4-bit level shift
	uint64_t abs_val = SIGNED && value < 0 ? static_cast<uint64_t>(-value) : static_cast<uint64_t>(value);
	int level = PacSumIntState<SIGNED>::GetLevel(static_cast<int64_t>(abs_val));
	uint64_t shift = level << 2; // multiply level by 4
	int64_t shifted_val = static_cast<int64_t>(value) >> shift;
	state.AddToExactTotal(level, static_cast<int32_t>(shifted_val), allocator);
	AddToTotalsSWAR<int16_t, uint16_t, PAC_APPROX_SWAR_MASK>(state.levels[level], shifted_val, key_hash);
#elif defined(PAC_NOCASCADING)
	AddToTotalsSimple(state.probabilistic_total128, value, key_hash); // directly add the value to the final total
#else
	// EXACT ALGORITHM: decide based on the (integer) value, in which level to aggregate
	// (ensure the level is allocated). Note that SIGNED stuff will be compiled away for unsigned types.
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

#endif // PAC_NOBUFFERING

template <class State, bool SIGNED, class VALUE_TYPE, class INPUT_TYPE>
static void PacSumUpdate(Vector inputs[], data_ptr_t state_p, idx_t count, ArenaAllocator &allocator,
                         uint64_t query_hash) {
	auto &state = *reinterpret_cast<State *>(state_p);
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
			PacSumUpdateOneInternal<SIGNED>(inner, hashes[h_idx] ^ query_hash,
			                                ConvertValue<VALUE_TYPE>::convert(values[v_idx]), allocator);
		}
	} else {
		for (idx_t i = 0; i < count; i++) {
			auto h_idx = hash_data.sel->get_index(i);
			auto v_idx = value_data.sel->get_index(i);
			if (!hash_data.validity.RowIsValid(h_idx) || !value_data.validity.RowIsValid(v_idx)) {
				continue; // ignore NULLs
			}
			state.exact_count++;
			PacSumUpdateOneInternal<SIGNED>(inner, hashes[h_idx] ^ query_hash,
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
		if (!hash_data.validity.RowIsValid(h_idx) || !value_data.validity.RowIsValid(v_idx)) {
			continue; // ignore NULLs
		}
		state->exact_count++;
		PacSumUpdateOne<SIGNED>(*state, hashes[h_idx] ^ query_hash, ConvertValue<VALUE_TYPE>::convert(values[v_idx]),
		                        allocator);
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
		// Merge exact_count from wrapper
#ifndef PAC_NOBUFFERING
		d_wrapper->exact_count += src_wrapper[i]->exact_count;
#endif
		if (!s) {
			continue; // src has no state allocated, nothing to combine
		}
		auto *d = d_wrapper->EnsureState(allocator);
		d->key_hash |= s->key_hash;
#ifndef PAC_EXACTSUM
		// APPROX COMBINE: merge levels from src into dst
		d->exact_count += s->exact_count;

		// Combine each level up to src's max_level_used (only allocated levels)
		for (int k = 0; k <= s->max_level_used; k++) {
			if (!s->levels[k]) {
				continue;
			}
			if (k > d->max_level_used) {
				// dst doesn't have this level - can we just take src's pointer?
				// Only safe if k < 9 or dst already spilled inline storage (max_level_used >= 9)
				// because levels[9..24] overlap with inline_level in the union
				if (k < 9 || d->max_level_used >= 9) {
					d->levels[k] = s->levels[k];
					d->exact_total[k] = s->exact_total[k];
					d->max_level_used = k;
					continue;
				}
				// Must allocate properly to handle inline storage spill
				d->EnsureLevelAllocated(allocator, k);
				// Fall through to add src's counters to the newly allocated (zeroed) level
			}

			// 16-bit counters: threshold must match counter width
			constexpr int32_t OVERFLOW_THRESHOLD = SIGNED ? 32767 : 65535;
			constexpr int32_t UNDERFLOW_THRESHOLD = SIGNED ? -32768 : 0;
			int32_t new_total = d->exact_total[k] + s->exact_total[k];

			if (new_total > OVERFLOW_THRESHOLD || new_total < UNDERFLOW_THRESHOLD) {
				d->Cascade(k, allocator);
				d->exact_total[k] = s->exact_total[k];
			} else {
				d->exact_total[k] = new_total;
			}

			// Add counters (SWAR layout: 16 uint64_t elements per level)
			uint64_t *src_level = s->levels[k];
			uint64_t *dst_level = d->levels[k];
			for (int j = 0; j < PAC_APPROX_SWAR_ELEMENTS; j++) {
				dst_level[j] += src_level[j];
			}
		}
#elif defined(PAC_NOCASCADING)
		for (int j = 0; j < 64; j++) {
			d->probabilistic_total128[j] += s->probabilistic_total128[j];
		}
#else
		// EXACT COMBINE: Handle exact_totals - if sum would overflow, flush dst (passing src's value
		// so it becomes the new exact_total after flush). Otherwise just add.
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
#ifdef PAC_EXACTSUM
		d->exact_count += s->exact_count;
#endif
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

// Unified Finalize for both int and double states
template <class State, class ACC_TYPE, bool SIGNED, bool DIVIDE_BY_COUNT>
void PacSumFinalize(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
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
		// Check if we should return NULL based on key_hash
		uint64_t key_hash = s ? s->key_hash : 0;
		if (PacNoiseInNull(key_hash, mi, gen)) {
			result_mask.SetInvalid(offset + i);
			continue;
		}
		double buf[64];
		if (s) {
			s->Flush(input.allocator);
			s->GetTotalsAsDouble(buf);
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

// ============================================================================
// X-macro definitions for type mappings
// ============================================================================
// X(NAME, VALUE_T, INPUT_T, SIGNED) - for integer types
// Note: HugeInt uses double state in approx mode, so it's excluded from this macro
#ifndef PAC_EXACTSUM
#define PAC_INT_TYPES_SIGNED                                                                                           \
	X(TinyInt, int64_t, int8_t, true)                                                                                  \
	X(SmallInt, int64_t, int16_t, true)                                                                                \
	X(Integer, int64_t, int32_t, true)                                                                                 \
	X(BigInt, int64_t, int64_t, true)
#else
#define PAC_INT_TYPES_SIGNED                                                                                           \
	X(TinyInt, int64_t, int8_t, true)                                                                                  \
	X(SmallInt, int64_t, int16_t, true)                                                                                \
	X(Integer, int64_t, int32_t, true)                                                                                 \
	X(BigInt, int64_t, int64_t, true)                                                                                  \
	X(HugeInt, hugeint_t, hugeint_t, true)
#endif

#define PAC_INT_TYPES_UNSIGNED                                                                                         \
	X(UTinyInt, uint64_t, uint8_t, false)                                                                              \
	X(USmallInt, uint64_t, uint16_t, false)                                                                            \
	X(UInteger, uint64_t, uint32_t, false)                                                                             \
	X(UBigInt, uint64_t, uint64_t, false)

// ============================================================================
// Generate exact pac_sum Update/ScatterUpdate functions via X-macros
// ============================================================================
#define X(NAME, VALUE_T, INPUT_T, SIGNED)                                                                              \
	void PacSumUpdate##NAME(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {       \
		PacSumUpdate<ScatterIntState<SIGNED>, SIGNED, VALUE_T, INPUT_T>(                                               \
		    inputs, state_p, count, aggr.allocator, aggr.bind_data->Cast<PacBindData>().query_hash);                   \
	}                                                                                                                  \
	void PacSumScatterUpdate##NAME(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {    \
		PacSumScatterUpdate<ScatterIntState<SIGNED>, SIGNED, VALUE_T, INPUT_T>(                                        \
		    inputs, states, count, aggr.allocator, aggr.bind_data->Cast<PacBindData>().query_hash);                    \
	}
PAC_INT_TYPES_SIGNED
PAC_INT_TYPES_UNSIGNED
#undef X

// Double/Float/[U]HugeInt use double state (not generated by X-macro)
void PacSumUpdateHugeIntDouble(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<ScatterDoubleState, true, double, hugeint_t>(inputs, state_p, count, aggr.allocator,
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
void PacSumScatterUpdateHugeIntDouble(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<ScatterDoubleState, true, double, hugeint_t>(inputs, states, count, aggr.allocator,
	                                                                 aggr.bind_data->Cast<PacBindData>().query_hash);
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
void PacSumFinalizeDoubleToHugeint(Vector &states, AggregateInputData &input, Vector &result, idx_t count,
                                   idx_t offset) {
	PacSumFinalize<ScatterDoubleState, hugeint_t, true>(states, input, result, count, offset);
}

unique_ptr<FunctionData> // Bind function for pac_sum with optional mi parameter (must be constant)
PacSumBind(ClientContext &ctx, AggregateFunction &, vector<unique_ptr<Expression>> &args) {
	// Check if pac_mi is explicitly set by the user - if so, use it and don't allow override
	double mi = 128.0;
	bool mi_from_setting = false;
	Value pac_mi_val;
	if (ctx.TryGetCurrentSetting("pac_mi", pac_mi_val) && !pac_mi_val.IsNull()) {
		mi = pac_mi_val.GetValue<double>();
		mi_from_setting = true;
	}

	// Only allow override from function argument if pac_mi was not explicitly set
	if (!mi_from_setting && args.size() >= 3) {
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

idx_t PacSumIntStateSize(const AggregateFunction &) {
	return sizeof(ScatterIntState<true>); // signed/unsigned have same wrapper size
}

void PacSumIntInitialize(const AggregateFunction &, data_ptr_t state_p) {
	memset(state_p, 0, sizeof(ScatterIntState<true>));
#ifndef PAC_EXACTSUM
	// For approx mode, set max_level_used to -1 (no levels have received data yet)
	// This applies when PAC_NOBUFFERING is set (state is directly PacSumIntState)
	// When buffering is enabled, wrapper's EnsureState handles this for the inner state
#ifdef PAC_NOBUFFERING
	auto &state = *reinterpret_cast<ScatterIntState<true> *>(state_p);
	state.max_level_used = -1;
	state.inline_level_idx = -1;
#endif
#endif
}

idx_t PacSumDoubleStateSize(const AggregateFunction &) {
	return sizeof(ScatterDoubleState);
}
void PacSumDoubleInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
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
#ifndef PAC_EXACTSUM
		return AggregateFunction("pac_sum", {LogicalType::UBIGINT, LogicalType::HUGEINT}, LogicalType::HUGEINT,
		                         PacSumDoubleStateSize, PacSumDoubleInitialize, PacSumScatterUpdateHugeIntDouble,
		                         PacSumCombineDoubleWrapper, PacSumFinalizeDoubleToHugeint,
		                         FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateHugeIntDouble);
#else
		return AggregateFunction("pac_sum", {LogicalType::UBIGINT, LogicalType::HUGEINT}, LogicalType::HUGEINT,
		                         PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateHugeInt,
		                         PacSumCombineSigned, PacSumFinalizeSigned, FunctionNullHandling::DEFAULT_NULL_HANDLING,
		                         PacSumUpdateHugeInt);
#endif
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

	// HUGEINT: uses double state with finalize converting back to hugeint in approx mode (default),
#ifndef PAC_EXACTSUM
	AddFcn(fcn_set, LogicalType::HUGEINT, LogicalType::HUGEINT, PacSumDoubleStateSize, PacSumDoubleInitialize,
	       PacSumScatterUpdateHugeIntDouble, PacSumCombineDoubleWrapper, PacSumFinalizeDoubleToHugeint,
	       PacSumUpdateHugeIntDouble);
#else
	// uses int state with exact cascading when PAC_EXACTSUM (can handle hugeint in probabilistic_total128)
	AddFcn(fcn_set, LogicalType::HUGEINT, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterUpdateHugeInt, PacSumCombineSigned, PacSumFinalizeSigned, PacSumUpdateHugeInt);
#endif
	// UHUGEINT: DuckDB's sum returns DOUBLE for uhugeint, so we do too (uses double state always)
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

// ============================================================================
// PAC_SUM_COUNTERS: Returns all 64 counters as LIST<DOUBLE> for categorical queries
// ============================================================================
// This variant is used when the sum result will be used in a comparison
// in an outer categorical query. Instead of picking one counter and adding noise,
// it returns all 64 counters so the outer query can evaluate the comparison
// against all subsamples and produce a mask.

template <class State, bool SIGNED>
static void PacSumFinalizeCounters(Vector &states, AggregateInputData &input, Vector &result, idx_t count,
                                   idx_t offset) {
	auto state_ptrs = FlatVector::GetData<State *>(states);

	// Result is LIST<DOUBLE>
	auto list_entries = FlatVector::GetData<list_entry_t>(result);
	auto &child_vec = ListVector::GetEntry(result);

	// Reserve space for all lists (64 elements each)
	idx_t total_elements = count * 64;
	ListVector::Reserve(result, total_elements);
	ListVector::SetListSize(result, total_elements);

	auto child_data = FlatVector::GetData<double>(child_vec);

	for (idx_t i = 0; i < count; i++) {
#ifndef PAC_NOBUFFERING
		PacSumFlushBuffer<SIGNED>(*state_ptrs[i], *state_ptrs[i], input.allocator);
#endif
		auto *s = state_ptrs[i]->GetState();

		// Set up the list entry
		list_entries[offset + i].offset = i * 64;
		list_entries[offset + i].length = 64;

		double *dst = &child_data[i * 64];
		if (s) {
			s->Flush(input.allocator);
			s->GetTotalsAsDouble(dst); // write directly to destination
		} else {
			memset(dst, 0, 64 * sizeof(double));
		}
	}
}

// Instantiate counter finalize methods
static void PacSumFinalizeCountersSigned(Vector &states, AggregateInputData &input, Vector &result, idx_t count,
                                         idx_t offset) {
	PacSumFinalizeCounters<ScatterIntState<true>, true>(states, input, result, count, offset);
}
static void PacSumFinalizeCountersUnsigned(Vector &states, AggregateInputData &input, Vector &result, idx_t count,
                                           idx_t offset) {
	PacSumFinalizeCounters<ScatterIntState<false>, false>(states, input, result, count, offset);
}
static void PacSumFinalizeCountersDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count,
                                         idx_t offset) {
	PacSumFinalizeCounters<ScatterDoubleState, true>(states, input, result, count, offset);
}

// Helper to register both 2-param and 3-param versions for pac_sum_counters
static void AddCountersFcn(AggregateFunctionSet &set, const LogicalType &value_type, aggregate_size_t state_size,
                           aggregate_initialize_t init, aggregate_update_t scatter, aggregate_combine_t combine,
                           aggregate_finalize_t finalize, aggregate_simple_update_t update) {
	auto list_double_type = LogicalType::LIST(LogicalType::DOUBLE);
	set.AddFunction(AggregateFunction("pac_sum_counters", {LogicalType::UBIGINT, value_type}, list_double_type,
	                                  state_size, init, scatter, combine, finalize,
	                                  FunctionNullHandling::DEFAULT_NULL_HANDLING, update, PacSumBind));
	set.AddFunction(AggregateFunction("pac_sum_counters", {LogicalType::UBIGINT, value_type, LogicalType::DOUBLE},
	                                  list_double_type, state_size, init, scatter, combine, finalize,
	                                  FunctionNullHandling::DEFAULT_NULL_HANDLING, update, PacSumBind));
}

void RegisterPacSumCountersFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet counters_set("pac_sum_counters");

	// Signed integers
	AddCountersFcn(counters_set, LogicalType::TINYINT, PacSumIntStateSize, PacSumIntInitialize,
	               PacSumScatterUpdateTinyInt, PacSumCombineSigned, PacSumFinalizeCountersSigned, PacSumUpdateTinyInt);
	AddCountersFcn(counters_set, LogicalType::BOOLEAN, PacSumIntStateSize, PacSumIntInitialize,
	               PacSumScatterUpdateTinyInt, PacSumCombineSigned, PacSumFinalizeCountersSigned, PacSumUpdateTinyInt);
	AddCountersFcn(counters_set, LogicalType::SMALLINT, PacSumIntStateSize, PacSumIntInitialize,
	               PacSumScatterUpdateSmallInt, PacSumCombineSigned, PacSumFinalizeCountersSigned,
	               PacSumUpdateSmallInt);
	AddCountersFcn(counters_set, LogicalType::INTEGER, PacSumIntStateSize, PacSumIntInitialize,
	               PacSumScatterUpdateInteger, PacSumCombineSigned, PacSumFinalizeCountersSigned, PacSumUpdateInteger);
	AddCountersFcn(counters_set, LogicalType::BIGINT, PacSumIntStateSize, PacSumIntInitialize,
	               PacSumScatterUpdateBigInt, PacSumCombineSigned, PacSumFinalizeCountersSigned, PacSumUpdateBigInt);

	// Unsigned integers
	AddCountersFcn(counters_set, LogicalType::UTINYINT, PacSumIntStateSize, PacSumIntInitialize,
	               PacSumScatterUpdateUTinyInt, PacSumCombineUnsigned, PacSumFinalizeCountersUnsigned,
	               PacSumUpdateUTinyInt);
	AddCountersFcn(counters_set, LogicalType::USMALLINT, PacSumIntStateSize, PacSumIntInitialize,
	               PacSumScatterUpdateUSmallInt, PacSumCombineUnsigned, PacSumFinalizeCountersUnsigned,
	               PacSumUpdateUSmallInt);
	AddCountersFcn(counters_set, LogicalType::UINTEGER, PacSumIntStateSize, PacSumIntInitialize,
	               PacSumScatterUpdateUInteger, PacSumCombineUnsigned, PacSumFinalizeCountersUnsigned,
	               PacSumUpdateUInteger);
	AddCountersFcn(counters_set, LogicalType::UBIGINT, PacSumIntStateSize, PacSumIntInitialize,
	               PacSumScatterUpdateUBigInt, PacSumCombineUnsigned, PacSumFinalizeCountersUnsigned,
	               PacSumUpdateUBigInt);

	// HUGEINT - uses double state in approx mode (same as UHUGEINT)
#ifndef PAC_EXACTSUM
	AddCountersFcn(counters_set, LogicalType::HUGEINT, PacSumDoubleStateSize, PacSumDoubleInitialize,
	               PacSumScatterUpdateHugeIntDouble, PacSumCombineDoubleWrapper, PacSumFinalizeCountersDouble,
	               PacSumUpdateHugeIntDouble);
#else
	AddCountersFcn(counters_set, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	               PacSumScatterUpdateHugeInt, PacSumCombineSigned, PacSumFinalizeCountersSigned, PacSumUpdateHugeInt);
#endif
	// UHUGEINT (uses double state always)
	AddCountersFcn(counters_set, LogicalType::UHUGEINT, PacSumDoubleStateSize, PacSumDoubleInitialize,
	               PacSumScatterUpdateUHugeInt, PacSumCombineDoubleWrapper, PacSumFinalizeCountersDouble,
	               PacSumUpdateUHugeInt);

	// Floating point
	AddCountersFcn(counters_set, LogicalType::FLOAT, PacSumDoubleStateSize, PacSumDoubleInitialize,
	               PacSumScatterUpdateFloat, PacSumCombineDoubleWrapper, PacSumFinalizeCountersDouble,
	               PacSumUpdateFloat);
	AddCountersFcn(counters_set, LogicalType::DOUBLE, PacSumDoubleStateSize, PacSumDoubleInitialize,
	               PacSumScatterUpdateDouble, PacSumCombineDoubleWrapper, PacSumFinalizeCountersDouble,
	               PacSumUpdateDouble);

	loader.RegisterFunction(counters_set);
}

// Explicit template instantiations for pac_avg (with DIVIDE_BY_COUNT=true)
#ifdef PAC_NOBUFFERING
// Without buffering, use raw state types
template void PacSumFinalize<PacSumDoubleState, double, true, true>(Vector &states, AggregateInputData &input,
                                                                    Vector &result, idx_t count, idx_t offset);
template void PacSumFinalize<PacSumIntState<true>, double, true, true>(Vector &states, AggregateInputData &input,
                                                                       Vector &result, idx_t count, idx_t offset);
template void PacSumFinalize<PacSumIntState<false>, double, false, true>(Vector &states, AggregateInputData &input,
                                                                         Vector &result, idx_t count, idx_t offset);
#else
// With buffering, use wrapper type names to match what pac_avg.cpp calls
template void PacSumFinalize<PacSumDoubleStateWrapper, double, true, true>(Vector &states, AggregateInputData &input,
                                                                           Vector &result, idx_t count, idx_t offset);
template void PacSumFinalize<PacSumIntStateWrapper<true>, double, true, true>(Vector &states, AggregateInputData &input,
                                                                              Vector &result, idx_t count,
                                                                              idx_t offset);
template void PacSumFinalize<PacSumIntStateWrapper<false>, double, false, true>(Vector &states,
                                                                                AggregateInputData &input,
                                                                                Vector &result, idx_t count,
                                                                                idx_t offset);

// Explicit template instantiations for PacSumFlushBuffer (used by pac_avg_counters)
template void PacSumFlushBuffer<true, PacSumIntStateWrapper<true>>(PacSumIntStateWrapper<true> &,
                                                                   PacSumIntStateWrapper<true> &, ArenaAllocator &);
template void PacSumFlushBuffer<false, PacSumIntStateWrapper<false>>(PacSumIntStateWrapper<false> &,
                                                                     PacSumIntStateWrapper<false> &, ArenaAllocator &);
template void PacSumFlushBuffer<true, PacSumDoubleStateWrapper>(PacSumDoubleStateWrapper &, PacSumDoubleStateWrapper &,
                                                                ArenaAllocator &);
#endif

} // namespace duckdb
