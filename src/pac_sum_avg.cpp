#include "include/pac_sum_avg.hpp"
#include "duckdb/common/types/decimal.hpp"
#include <cmath>
#include <unordered_map>

namespace duckdb {

// SIGNED is compile-time known, so for unsigned the negative cases (value < 0) will be compiled away
#define ACCUMULATE_BITMARGIN      2 // val must be 2 bits shorter than the accumulator to allow >=4 updates without overflow
#define UPPERBOUND_BITWIDTH(bits) (1LL << ((bits - SIGNED) - ACCUMULATE_BITMARGIN))
#define LOWERBOUND_BITWIDTH(bits) -(static_cast<int64_t>(SIGNED) << ((bits - SIGNED) - ACCUMULATE_BITMARGIN))

#ifdef PAC_SUMAVG_TENFOLDOPT
// Helper to insert a value at a specific level (used by tenfold optimization)
template <bool SIGNED, int LEVEL>
AUTOVECTORIZE static inline void InsertAtLevel(PacSumIntState<SIGNED> &state, uint64_t key_hash, int64_t value) {
	if constexpr (LEVEL == 8) {
		state.exact_total8 = state.EnsureLevelAllocated(state.probabilistic_total8, 8, state.exact_total8);
		state.Flush8(value, false);
		AddToTotalsSWAR<int8_t, uint8_t, 0x0101010101010101ULL>(state.probabilistic_total8, value, key_hash);
	} else if constexpr (LEVEL == 16) {
		state.exact_total16 = state.EnsureLevelAllocated(state.probabilistic_total16, 16, state.exact_total16);
		state.Flush16(value, false);
		AddToTotalsSWAR<int16_t, uint16_t, 0x0001000100010001ULL>(state.probabilistic_total16, value, key_hash);
	} else if constexpr (LEVEL == 32) {
		state.exact_total32 = state.EnsureLevelAllocated(state.probabilistic_total32, 32, state.exact_total32);
		state.Flush32(value, false);
		AddToTotalsSWAR<int32_t, uint32_t, 0x0000000100000001ULL>(state.probabilistic_total32, value, key_hash);
	}
}

// Helper to flush a specific level (used by tenfold optimization)
template <bool SIGNED>
static inline void FlushLevel(PacSumIntState<SIGNED> &state, int level_idx) {
	if (level_idx == 0 && state.probabilistic_total8) {
		state.Flush8(0, true);
	} else if (level_idx == 1 && state.probabilistic_total16) {
		state.Flush16(0, true);
	} else if (level_idx == 2 && state.probabilistic_total32) {
		state.Flush32(0, true);
	}
}

// Try tenfold optimization: returns true if value was inserted via tenfold path
template <bool SIGNED>
static inline bool TryTenfoldInsert(PacSumIntState<SIGNED> &state, uint64_t key_hash, int64_t value,
                                    ArenaAllocator &allocator) {
	constexpr uint8_t UNINIT = PacSumIntState<SIGNED>::TENFOLD_UNINITIALIZED;

	uint8_t z = is_tenfold(value);
	if (z == 0) {
		return false; // Value is not a multiple of 10
	}

	// Compute reduced value: w = value / 10^z (using fast division)
	int64_t w = fast_div10(value, z);

	// Find which level w would fit in
	int target_level = -1;
	if ((w >= LOWERBOUND_BITWIDTH(8)) && (w < UPPERBOUND_BITWIDTH(8))) {
		target_level = 0; // level 8
	} else if ((w >= LOWERBOUND_BITWIDTH(16)) && (w < UPPERBOUND_BITWIDTH(16))) {
		target_level = 1; // level 16
	} else if ((w >= LOWERBOUND_BITWIDTH(32)) && (w < UPPERBOUND_BITWIDTH(32))) {
		target_level = 2; // level 32
	}

	// Only apply tenfold to levels 0, 1, 2 (i.e., 8, 16, 32 bit)
	if (target_level < 0 || target_level > 2) {
		return false;
	}

	uint8_t &tf = state.tenfold[target_level];

	if (tf == UNINIT) {
		// First value at this level - establish tenfold factor
		tf = z;
		if (target_level == 0) {
			InsertAtLevel<SIGNED, 8>(state, key_hash, w);
		} else if (target_level == 1) {
			InsertAtLevel<SIGNED, 16>(state, key_hash, w);
		} else {
			InsertAtLevel<SIGNED, 32>(state, key_hash, w);
		}
		return true;
	} else if (tf == z) {
		// Same factor - just insert reduced value
		if (target_level == 0) {
			InsertAtLevel<SIGNED, 8>(state, key_hash, w);
		} else if (target_level == 1) {
			InsertAtLevel<SIGNED, 16>(state, key_hash, w);
		} else {
			InsertAtLevel<SIGNED, 32>(state, key_hash, w);
		}
		return true;
	} else if (tf > z && tf != 0) {
		// New value has smaller factor - flush level and adopt new factor
		// Must ensure next level is allocated before flushing
		if (target_level == 0) {
			state.exact_total16 = state.EnsureLevelAllocated(state.probabilistic_total16, 16, state.exact_total16);
		} else if (target_level == 1) {
			state.exact_total32 = state.EnsureLevelAllocated(state.probabilistic_total32, 32, state.exact_total32);
		} else if (target_level == 2) {
			state.exact_total64 = state.EnsureLevelAllocated(state.probabilistic_total64, 64, state.exact_total64);
		}
		FlushLevel<SIGNED>(state, target_level);
		tf = z;
		if (target_level == 0) {
			InsertAtLevel<SIGNED, 8>(state, key_hash, w);
		} else if (target_level == 1) {
			InsertAtLevel<SIGNED, 16>(state, key_hash, w);
		} else {
			InsertAtLevel<SIGNED, 32>(state, key_hash, w);
		}
		return true;
	} else if (tf != 0 && tf < z) {
		// tf < z: Existing factor is smaller - try to scale w up to match
		int64_t scale = POWERS_OF_10[z - tf];
		int64_t w_scaled = w * scale;

		// Check if scaled value still fits in target level
		if (target_level == 0 && (w_scaled >= LOWERBOUND_BITWIDTH(8)) && (w_scaled < UPPERBOUND_BITWIDTH(8))) {
			InsertAtLevel<SIGNED, 8>(state, key_hash, w_scaled);
			return true;
		} else if (target_level == 1 && (w_scaled >= LOWERBOUND_BITWIDTH(16)) && (w_scaled < UPPERBOUND_BITWIDTH(16))) {
			InsertAtLevel<SIGNED, 16>(state, key_hash, w_scaled);
			return true;
		} else if (target_level == 2 && (w_scaled >= LOWERBOUND_BITWIDTH(32)) && (w_scaled < UPPERBOUND_BITWIDTH(32))) {
			InsertAtLevel<SIGNED, 32>(state, key_hash, w_scaled);
			return true;
		}
		// Scaled value doesn't fit - fall through to normal insert
	}
	// tf == 0 means level has non-tenfold data, can't use tenfold optimization
	return false;
}
#endif // PAC_SUMAVG_TENFOLDOPT

template <bool SIGNED>
AUTOVECTORIZE inline void // main worker function for probabilistically adding one INTEGER to the 64 (sub)total
PacSumUpdateOne(PacSumIntState<SIGNED> &state, uint64_t key_hash, typename PacSumIntState<SIGNED>::T64 value,
                ArenaAllocator &allocator) {
#ifdef PAC_SUMAVG_NONCASCADING
	AddToTotalsSimple(state.probabilistic_total128, value, key_hash);
#else
	// Store allocator pointer for use in Flush/EnsureLevelAllocated methods
	if (!state.allocator) {
		state.allocator = &allocator;
#ifdef PAC_SUMAVG_NONLAZY
		state.InitializeAllLevels(allocator);
#endif
	}

#ifdef PAC_SUMAVG_TENFOLDOPT
	// Tenfold optimization: try to store multiples of 10^z as reduced values
	if (TryTenfoldInsert<SIGNED>(state, key_hash, static_cast<int64_t>(value), allocator)) {
		return;
	}
#endif

	// Normal insert path
	if ((SIGNED && value < 0) ? (value >= LOWERBOUND_BITWIDTH(8)) : (value < UPPERBOUND_BITWIDTH(8))) {
		state.exact_total8 = state.EnsureLevelAllocated(state.probabilistic_total8, 8, state.exact_total8);
#ifdef PAC_SUMAVG_TENFOLDOPT
		// If level has tenfold data (factor > 0), flush it first before mixing with non-tenfold
		if (state.tenfold[0] > 0 && state.tenfold[0] != PacSumIntState<SIGNED>::TENFOLD_UNINITIALIZED) {
			state.exact_total16 = state.EnsureLevelAllocated(state.probabilistic_total16, 16, state.exact_total16);
			state.Flush8(0, true);
		}
		state.tenfold[0] = 0; // Mark as non-tenfold (regardless of previous state)
#endif
		state.Flush8(value, false);
		AddToTotalsSWAR<int8_t, uint8_t, 0x0101010101010101ULL>(state.probabilistic_total8, value, key_hash);
	} else if ((SIGNED && value < 0) ? (value >= LOWERBOUND_BITWIDTH(16)) : (value < UPPERBOUND_BITWIDTH(16))) {
		state.exact_total16 = state.EnsureLevelAllocated(state.probabilistic_total16, 16, state.exact_total16);
#ifdef PAC_SUMAVG_TENFOLDOPT
		// If level has tenfold data (factor > 0), flush it first before mixing with non-tenfold
		if (state.tenfold[1] > 0 && state.tenfold[1] != PacSumIntState<SIGNED>::TENFOLD_UNINITIALIZED) {
			state.exact_total32 = state.EnsureLevelAllocated(state.probabilistic_total32, 32, state.exact_total32);
			state.Flush16(0, true);
		}
		state.tenfold[1] = 0; // Mark as non-tenfold (regardless of previous state)
#endif
		state.Flush16(value, false);
		AddToTotalsSWAR<int16_t, uint16_t, 0x0001000100010001ULL>(state.probabilistic_total16, value, key_hash);
	} else if ((SIGNED && value < 0) ? (value >= LOWERBOUND_BITWIDTH(32)) : (value < UPPERBOUND_BITWIDTH(32))) {
		state.exact_total32 = state.EnsureLevelAllocated(state.probabilistic_total32, 32, state.exact_total32);
#ifdef PAC_SUMAVG_TENFOLDOPT
		// If level has tenfold data (factor > 0), flush it first before mixing with non-tenfold
		if (state.tenfold[2] > 0 && state.tenfold[2] != PacSumIntState<SIGNED>::TENFOLD_UNINITIALIZED) {
			state.exact_total64 = state.EnsureLevelAllocated(state.probabilistic_total64, 64, state.exact_total64);
			state.Flush32(0, true);
		}
		state.tenfold[2] = 0; // Mark as non-tenfold (regardless of previous state)
#endif
		state.Flush32(value, false);
		AddToTotalsSWAR<int32_t, uint32_t, 0x0000000100000001ULL>(state.probabilistic_total32, value, key_hash);
	} else {
		state.exact_total64 = state.EnsureLevelAllocated(state.probabilistic_total64, 64, state.exact_total64);
		state.Flush64(value, false);
		AddToTotalsSimple(state.probabilistic_total64, value, key_hash);
	}
#endif
}

template <bool SIGNED>
AUTOVECTORIZE inline void // main worker function for probabilistically adding one DOUBLE to the 64 sum total
PacSumUpdateOne(PacSumDoubleState &state, uint64_t key_hash, double value, ArenaAllocator &) {
	// Note: ArenaAllocator not used for double state (no lazy allocation)
#ifdef PAC_SUMAVG_FLOAT_CASCADING
	if (PacSumDoubleState::FloatSubtotalFitsDouble(value)) {
		AddToTotalsFloat(state.probabilistic_total_float, static_cast<float>(value), key_hash);
		state.Flush32(value);
		return;
	}
#endif
	AddToTotalsSimple(state.probabilistic_total, value, key_hash);
}

// Overload for HUGEINT input - adds directly to hugeint_t total (no cascading since values don't fit in subtotal)
template <bool SIGNED>
AUTOVECTORIZE inline void PacSumUpdateOne(PacSumIntState<SIGNED> &state, uint64_t key_hash, hugeint_t value,
                                          ArenaAllocator &allocator) {
#ifndef PAC_SUMAVG_NONCASCADING
	state.allocator = &allocator;
	state.EnsureLevelAllocated(state.probabilistic_total128, idx_t(64));
#endif
	for (int j = 0; j < 64; j++) {
		if ((key_hash >> j) & 1ULL) {
			state.probabilistic_total128[j] += value;
		}
	}
}

template <class State, bool SIGNED, class VALUE_TYPE, class INPUT_TYPE>
static void PacSumUpdate(Vector inputs[], data_ptr_t state_p, idx_t count, ArenaAllocator &allocator) {
	auto &state = *reinterpret_cast<State *>(state_p);
#ifdef PAC_SUMAVG_UNSAFENULL
	if (state.seen_null) {
		return;
	}
#endif
	UnifiedVectorFormat hash_data, value_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);

	// Fast path: if both vectors have no nulls, skip per-row validity check
	if (hash_data.validity.AllValid() && value_data.validity.AllValid()) {
		state.exact_count += count; // increment count by batch size
		for (idx_t i = 0; i < count; i++) {
			auto h_idx = hash_data.sel->get_index(i);
			auto v_idx = value_data.sel->get_index(i);
			PacSumUpdateOne<SIGNED>(state, hashes[h_idx], ConvertValue<VALUE_TYPE>::convert(values[v_idx]), allocator);
		}
	} else {
		for (idx_t i = 0; i < count; i++) {
			auto h_idx = hash_data.sel->get_index(i);
			auto v_idx = value_data.sel->get_index(i);
			if (!hash_data.validity.RowIsValid(h_idx) || !value_data.validity.RowIsValid(v_idx)) {
#ifdef PAC_SUMAVG_UNSAFENULL
				state.seen_null = true;
				return;
#else
				continue; // safe mode: ignore NULLs
#endif
			}
			state.exact_count++;
			PacSumUpdateOne<SIGNED>(state, hashes[h_idx], ConvertValue<VALUE_TYPE>::convert(values[v_idx]), allocator);
		}
	}
}

template <class State, bool SIGNED, class VALUE_TYPE, class INPUT_TYPE>
static void PacSumScatterUpdate(Vector inputs[], Vector &states, idx_t count, ArenaAllocator &allocator) {
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
#ifdef PAC_SUMAVG_UNSAFENULL
		if (state->seen_null) {
			continue; // result will be NULL anyway
		} else if (!hash_data.validity.RowIsValid(h_idx) || !value_data.validity.RowIsValid(v_idx)) {
			state->seen_null = true;
		} else {
#else
		if (!hash_data.validity.RowIsValid(h_idx) || !value_data.validity.RowIsValid(v_idx)) {
			continue; // safe mode: ignore NULLs
		} else {
#endif
			state->exact_count++;
			PacSumUpdateOne<SIGNED>(*state, hashes[h_idx], ConvertValue<VALUE_TYPE>::convert(values[v_idx]), allocator);
		}
	}
}
// Check if combining two exact totals would risk overflow at a given bit width
template <int BITS, bool SIGNED>
static inline bool NeedsFlushBeforeCombine(int64_t src_exact, int64_t dst_exact) {
	constexpr int64_t MAX_VAL = SIGNED ? (1LL << (BITS - 2)) : (1LL << (BITS - 1));
	constexpr int64_t MIN_VAL = SIGNED ? -MAX_VAL : 0;
	int64_t combined = src_exact + dst_exact;
	return combined > MAX_VAL || combined < MIN_VAL;
}

#ifdef PAC_SUMAVG_TENFOLDOPT
static inline bool TenfoldCompatible(uint8_t src_tf, uint8_t dst_tf, uint8_t UNINIT) {
	return src_tf == UNINIT || dst_tf == UNINIT || src_tf == dst_tf;
}
#endif

// Combine level: returns false if flush needed, true if combined successfully.
// tf_ok: tenfold compatibility (true if no tenfold or compatible factors)
template <int BITS, bool SIGNED, typename EXACT_T>
static inline bool CombineLevel(uint64_t *&s_buf, uint64_t *&d_buf, EXACT_T &s_exact, EXACT_T &d_exact, bool tf_ok = true) {
	if (s_buf && d_buf) {
		if (NeedsFlushBeforeCombine<BITS, SIGNED>(s_exact, d_exact) || !tf_ok) {
			return false; // Caller should flush both
		}
		for (idx_t j = 0; j < BITS; j++) {
			d_buf[j] += s_buf[j];
		}
		d_exact += s_exact;
	} else if (s_buf) {
		d_buf = s_buf;
		d_exact = s_exact;
		s_buf = nullptr;
	}
	return true;
}

// Combine for integer states - combines at each level without forcing to 128-bit
// If dst doesn't have a level that src has, we move the pointer (no copy needed)
template <bool SIGNED>
AUTOVECTORIZE static void PacSumCombineInt(Vector &src, Vector &dst, idx_t count, ArenaAllocator &allocator) {
	auto src_state = FlatVector::GetData<PacSumIntState<SIGNED> *>(src);
	auto dst_state = FlatVector::GetData<PacSumIntState<SIGNED> *>(dst);

	for (idx_t i = 0; i < count; i++) {
#ifdef PAC_SUMAVG_UNSAFENULL
		if (src_state[i]->seen_null) {
			dst_state[i]->seen_null = true;
		}
		if (dst_state[i]->seen_null) {
			continue;
		}
#endif
		{
#ifdef PAC_SUMAVG_NONCASCADING
			for (int j = 0; j < 64; j++) {
				dst_state[i]->probabilistic_total128[j] += src_state[i]->probabilistic_total128[j];
			}
#else
			auto *s = src_state[i];
			auto *d = dst_state[i];
			if (!d->allocator) {
				d->allocator = &allocator; // Ensure dst has allocator pointer
			}

			// Combine each level: flush first if overflow risk or tenfold incompatible, then merge
#ifdef PAC_SUMAVG_TENFOLDOPT
			constexpr uint8_t U = PacSumIntState<SIGNED>::TENFOLD_UNINITIALIZED;
#define TF_COMPAT(lvl) TenfoldCompatible(s->tenfold[lvl], d->tenfold[lvl], U)
#else
#define TF_COMPAT(lvl) true
#endif
			if (!CombineLevel<8, SIGNED>(s->probabilistic_total8, d->probabilistic_total8, s->exact_total8, d->exact_total8, TF_COMPAT(0))) {
				s->exact_total16 = s->EnsureLevelAllocated(s->probabilistic_total16, 16, s->exact_total16);
				d->exact_total16 = d->EnsureLevelAllocated(d->probabilistic_total16, 16, d->exact_total16);
				s->Flush8(0, true); d->Flush8(0, true);
				CombineLevel<8, SIGNED>(s->probabilistic_total8, d->probabilistic_total8, s->exact_total8, d->exact_total8);
			}
			if (!CombineLevel<16, SIGNED>(s->probabilistic_total16, d->probabilistic_total16, s->exact_total16, d->exact_total16, TF_COMPAT(1))) {
				s->exact_total32 = s->EnsureLevelAllocated(s->probabilistic_total32, 32, s->exact_total32);
				d->exact_total32 = d->EnsureLevelAllocated(d->probabilistic_total32, 32, d->exact_total32);
				s->Flush16(0, true); d->Flush16(0, true);
				CombineLevel<16, SIGNED>(s->probabilistic_total16, d->probabilistic_total16, s->exact_total16, d->exact_total16);
			}
			if (!CombineLevel<32, SIGNED>(s->probabilistic_total32, d->probabilistic_total32, s->exact_total32, d->exact_total32, TF_COMPAT(2))) {
				s->exact_total64 = s->EnsureLevelAllocated(s->probabilistic_total64, 64, s->exact_total64);
				d->exact_total64 = d->EnsureLevelAllocated(d->probabilistic_total64, 64, d->exact_total64);
				s->Flush32(0, true); d->Flush32(0, true);
				CombineLevel<32, SIGNED>(s->probabilistic_total32, d->probabilistic_total32, s->exact_total32, d->exact_total32);
			}
			if (!CombineLevel<64, SIGNED>(s->probabilistic_total64, d->probabilistic_total64, s->exact_total64, d->exact_total64)) {
				s->EnsureLevelAllocated(s->probabilistic_total128, idx_t(64));
				d->EnsureLevelAllocated(d->probabilistic_total128, idx_t(64));
				s->Flush64(0, true); d->Flush64(0, true);
				CombineLevel<64, SIGNED>(s->probabilistic_total64, d->probabilistic_total64, s->exact_total64, d->exact_total64);
			}
			// Level 128: hugeint_t* type, no next level
			if (s->probabilistic_total128) {
				if (d->probabilistic_total128) {
					for (idx_t j = 0; j < 64; j++) {
						d->probabilistic_total128[j] += s->probabilistic_total128[j];
					}
				} else {
					d->probabilistic_total128 = s->probabilistic_total128;
					s->probabilistic_total128 = nullptr;
				}
			}
#ifdef PAC_SUMAVG_TENFOLDOPT
			// Merge tenfold factors: if dst adopted src's level, adopt its factor too
			constexpr uint8_t UNINIT = PacSumIntState<SIGNED>::TENFOLD_UNINITIALIZED;
			for (int lvl = 0; lvl < 3; lvl++) {
				if (d->tenfold[lvl] == UNINIT && s->tenfold[lvl] != UNINIT) {
					d->tenfold[lvl] = s->tenfold[lvl];
				}
			}
#endif
#undef TF_COMPAT
#endif
			dst_state[i]->exact_count += src_state[i]->exact_count;
		}
	}
}

// Combine for double state
AUTOVECTORIZE static void PacSumCombineDouble(Vector &src, Vector &dst, idx_t count) {
	auto src_state = FlatVector::GetData<PacSumDoubleState *>(src);
	auto dst_state = FlatVector::GetData<PacSumDoubleState *>(dst);
	for (idx_t i = 0; i < count; i++) {
#ifdef PAC_SUMAVG_UNSAFENULL
		if (src_state[i]->seen_null) {
			dst_state[i]->seen_null = true;
		}
		if (dst_state[i]->seen_null) {
			continue;
		}
#endif
#ifdef PAC_SUMAVG_FLOAT_CASCADING
		src_state[i]->Flush();
#endif
		dst_state[i]->exact_count += src_state[i]->exact_count;
		for (int j = 0; j < 64; j++) {
			dst_state[i]->probabilistic_total[j] += src_state[i]->probabilistic_total[j];
		}
	}
}

// Unified Finalize for both int and double states
template <class State, class ACC_TYPE, bool DIVIDE_BY_COUNT = false>
static void PacSumFinalize(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	auto state = FlatVector::GetData<State *>(states);
	auto data = FlatVector::GetData<ACC_TYPE>(result);
	auto &result_mask = FlatVector::Validity(result);
	uint64_t seed = input.bind_data ? input.bind_data->Cast<PacBindData>().seed : std::random_device {}();
	std::mt19937_64 gen(seed);
	double mi = input.bind_data ? input.bind_data->Cast<PacBindData>().mi : 128.0;
	// scale_divisor is used by pac_avg on DECIMAL to convert internal integer representation back to decimal
	double scale_divisor = input.bind_data ? input.bind_data->Cast<PacBindData>().scale_divisor : 1.0;

	for (idx_t i = 0; i < count; i++) {
#ifdef PAC_SUMAVG_UNSAFENULL
		if (state[i]->seen_null) {
			result_mask.SetInvalid(offset + i);
			continue;
		}
#endif
		double buf[64];
		state[i]->Flush();
		state[i]->GetTotalsAsDouble(buf);
		if (DIVIDE_BY_COUNT) {
			double divisor = static_cast<double>(state[i]->exact_count) * scale_divisor;
			for (int j = 0; j < 64; j++) {
				buf[j] /= divisor;
			}
		}
		// the random counter we choose to read is #42 (but we start counting from 0, so [41])
		data[offset + i] = FromDouble<ACC_TYPE>(PacNoisySampleFrom64Counters(buf, mi, gen) + buf[41]);
	}
}

// instantiate Update methods
void PacSumUpdateTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<true>, true, int64_t, int8_t>(inputs, state_p, count, aggr.allocator);
}
void PacSumUpdateSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<true>, true, int64_t, int16_t>(inputs, state_p, count, aggr.allocator);
}
void PacSumUpdateInteger(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<true>, true, int64_t, int32_t>(inputs, state_p, count, aggr.allocator);
}
void PacSumUpdateBigInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<true>, true, int64_t, int64_t>(inputs, state_p, count, aggr.allocator);
}
void PacSumUpdateHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<true>, true, hugeint_t, hugeint_t>(inputs, state_p, count, aggr.allocator);
}
void PacSumUpdateUTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<false>, false, uint64_t, uint8_t>(inputs, state_p, count, aggr.allocator);
}
void PacSumUpdateUSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<false>, false, uint64_t, uint16_t>(inputs, state_p, count, aggr.allocator);
}
void PacSumUpdateUInteger(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<false>, false, uint64_t, uint32_t>(inputs, state_p, count, aggr.allocator);
}
void PacSumUpdateUBigInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<false>, false, uint64_t, uint64_t>(inputs, state_p, count, aggr.allocator);
}
void PacSumUpdateUHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumDoubleState, true, double, uhugeint_t>(inputs, state_p, count, aggr.allocator);
}
void PacSumUpdateFloat(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumDoubleState, true, double, float>(inputs, state_p, count, aggr.allocator);
}
void PacSumUpdateDouble(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumDoubleState, true, double, double>(inputs, state_p, count, aggr.allocator);
}

// instantiate ScatterUpdate methods
void PacSumScatterUpdateTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<true>, true, int64_t, int8_t>(inputs, states, count, aggr.allocator);
}
void PacSumScatterUpdateSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<true>, true, int64_t, int16_t>(inputs, states, count, aggr.allocator);
}
void PacSumScatterUpdateInteger(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<true>, true, int64_t, int32_t>(inputs, states, count, aggr.allocator);
}
void PacSumScatterUpdateBigInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<true>, true, int64_t, int64_t>(inputs, states, count, aggr.allocator);
}
void PacSumScatterUpdateHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<true>, true, hugeint_t, hugeint_t>(inputs, states, count, aggr.allocator);
}
void PacSumScatterUpdateUTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<false>, false, uint64_t, uint8_t>(inputs, states, count, aggr.allocator);
}
void PacSumScatterUpdateUSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<false>, false, uint64_t, uint16_t>(inputs, states, count, aggr.allocator);
}
void PacSumScatterUpdateUInteger(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<false>, false, uint64_t, uint32_t>(inputs, states, count, aggr.allocator);
}
void PacSumScatterUpdateUBigInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<false>, false, uint64_t, uint64_t>(inputs, states, count, aggr.allocator);
}
void PacSumScatterUpdateUHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumDoubleState, true, double, uhugeint_t>(inputs, states, count, aggr.allocator);
}
void PacSumScatterUpdateFloat(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumDoubleState, true, double, float>(inputs, states, count, aggr.allocator);
}
void PacSumScatterUpdateDouble(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumDoubleState, true, double, double>(inputs, states, count, aggr.allocator);
}

// instantiate Combine methods
void PacSumCombineSigned(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	PacSumCombineInt<true>(src, dst, count, aggr.allocator);
}
void PacSumCombineUnsigned(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	PacSumCombineInt<false>(src, dst, count, aggr.allocator);
}
void PacSumCombineDoubleWrapper(Vector &src, Vector &dst, AggregateInputData &, idx_t count) {
	PacSumCombineDouble(src, dst, count);
}

// instantiate Finalize methods for pac_sum
void PacSumFinalizeSigned(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	PacSumFinalize<PacSumIntState<true>, hugeint_t>(states, input, result, count, offset);
}
void PacSumFinalizeUnsigned(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	PacSumFinalize<PacSumIntState<false>, hugeint_t>(states, input, result, count, offset);
}
void PacSumFinalizeDoubleWrapper(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	PacSumFinalize<PacSumDoubleState, double>(states, input, result, count, offset);
}

// instantiate Finalize methods for pac_avg (with DIVIDE_BY_COUNT=true)
void PacAvgFinalizeDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	PacSumFinalize<PacSumDoubleState, double, true>(states, input, result, count, offset);
}
void PacAvgFinalizeSignedDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	PacSumFinalize<PacSumIntState<true>, double, true>(states, input, result, count, offset);
}
void PacAvgFinalizeUnsignedDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count,
                                  idx_t offset) {
	PacSumFinalize<PacSumIntState<false>, double, true>(states, input, result, count, offset);
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
		if (mi <= 0.0) {
			throw InvalidInputException("pac_sum: mi must be > 0");
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
	return sizeof(PacSumIntState<true>); // signed (true) and unsigned (false) have the same size
}

static void PacSumIntInitialize(const AggregateFunction &, data_ptr_t state_p) {
	memset(state_p, 0, sizeof(PacSumIntState<true>)); // memset to 0 works for both signed and unsigned
#ifdef PAC_SUMAVG_TENFOLDOPT
	// Initialize tenfold factors to UNINITIALIZED (255), not 0 (which means non-tenfold data)
	auto *state = reinterpret_cast<PacSumIntState<true> *>(state_p);
	state->anytenfolds = 0xFFFFFFFF; // Sets all 4 tenfold bytes to 255 (UNINIT)
#endif
}

static idx_t PacSumDoubleStateSize(const AggregateFunction &) {
	return sizeof(PacSumDoubleState);
}
static void PacSumDoubleInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	memset(state_ptr, 0, sizeof(PacSumDoubleState));
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
		if (mi <= 0.0) {
			throw InvalidInputException("pac_avg: mi must be > 0");
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

} // namespace duckdb
