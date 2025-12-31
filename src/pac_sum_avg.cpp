#include "include/pac_sum_avg.hpp"
#include "duckdb/common/types/decimal.hpp"
#include <cmath>
#include <unordered_map>

namespace duckdb {

// SIGNED is compile-time known, so for unsigned the negative cases (value < 0) will be compiled away
#define ACCUMULATE_BITMARGIN      2 // val must be 2 bits shorter than the accumulator to allow >=4 updates without overflow
#define UPPERBOUND_BITWIDTH(bits) (1LL << ((bits - SIGNED) - ACCUMULATE_BITMARGIN))
#define LOWERBOUND_BITWIDTH(bits) -(static_cast<int64_t>(SIGNED) << ((bits - SIGNED) - ACCUMULATE_BITMARGIN))

template <bool SIGNED>
AUTOVECTORIZE inline void // main worker function for probabilistically adding one INTEGER to the 64 (sub)total
PacSumUpdateOne(PacSumIntState<SIGNED> &state, uint64_t key_hash, typename PacSumIntState<SIGNED>::T64 value,
                ArenaAllocator &allocator) {
#ifdef PAC_SUMAVG_NONCASCADING
	AddToTotalsSimple(state.probabilistic_total128, value, key_hash);
#else
#ifdef PAC_SUMAVG_NONLAZY
	if (!state.probabilistic_total8) { // Use as proxy for "not yet initialized"
		state.InitializeAllLevels(allocator);
		state.min_allocated_level = 8;
	}
#endif
	// Route to appropriate level based on value magnitude, skipping levels smaller than min_allocated_level
	if (state.min_allocated_level <= 8 &&
	    ((SIGNED && value < 0) ? (value >= LOWERBOUND_BITWIDTH(8)) : (value < UPPERBOUND_BITWIDTH(8)))) {
		if (!state.probabilistic_total8) {
			state.exact_total8 = PacSumIntState<SIGNED>::EnsureLevelAllocated(allocator, state.probabilistic_total8, 8,
			                                                                  state.exact_total8);
			state.min_allocated_level = 8;
		}
		state.Flush8(allocator, value, false);
		AddToTotalsSWAR<int8_t, uint8_t, 0x0101010101010101ULL>(state.probabilistic_total8, value, key_hash);
	} else if (state.min_allocated_level <= 16 &&
	           ((SIGNED && value < 0) ? (value >= LOWERBOUND_BITWIDTH(16)) : (value < UPPERBOUND_BITWIDTH(16)))) {
		if (!state.probabilistic_total16) {
			state.exact_total16 = PacSumIntState<SIGNED>::EnsureLevelAllocated(allocator, state.probabilistic_total16,
			                                                                   16, state.exact_total16);
			if (state.min_allocated_level == 0)
				state.min_allocated_level = 16;
		}
		state.Flush16(allocator, value, false);
		AddToTotalsSWAR<int16_t, uint16_t, 0x0001000100010001ULL>(state.probabilistic_total16, value, key_hash);
	} else if (state.min_allocated_level <= 32 &&
	           ((SIGNED && value < 0) ? (value >= LOWERBOUND_BITWIDTH(32)) : (value < UPPERBOUND_BITWIDTH(32)))) {
		if (!state.probabilistic_total32) {
			state.exact_total32 = PacSumIntState<SIGNED>::EnsureLevelAllocated(allocator, state.probabilistic_total32,
			                                                                   32, state.exact_total32);
			if (state.min_allocated_level == 0)
				state.min_allocated_level = 32;
		}
		state.Flush32(allocator, value, false);
		AddToTotalsSWAR<int32_t, uint32_t, 0x0000000100000001ULL>(state.probabilistic_total32, value, key_hash);
	} else {
		if (!state.probabilistic_total64) {
			state.exact_total64 = PacSumIntState<SIGNED>::EnsureLevelAllocated(allocator, state.probabilistic_total64,
			                                                                   64, state.exact_total64);
			if (state.min_allocated_level == 0)
				state.min_allocated_level = 64;
		}
		state.Flush64(allocator, value, false);
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
	PacSumIntState<SIGNED>::EnsureLevelAllocated(allocator, state.probabilistic_total128, idx_t(64));
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
// Helper to combine src array into dst at a specific level
// If src is null, does nothing. If dst is null, moves pointer from src to dst.
// Note: exact_totals only used in move case (when both exist, they're handled separately)
template <typename BUF_T, typename EXACT_T>
static inline void CombineLevel(BUF_T *&src_buf, BUF_T *&dst_buf, EXACT_T &src_exact, EXACT_T &dst_exact, idx_t count) {
	if (src_buf) {
		if (dst_buf) {
			for (idx_t j = 0; j < count; j++) {
				dst_buf[j] += src_buf[j];
			}
			// exact_totals handled separately (before this call) to avoid overflow
		} else {
			dst_buf = src_buf;
			dst_exact = src_exact;
			src_buf = nullptr;
		}
	}
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

			// Combine arrays at each level (exact_totals handled above, but passed for move case)
			hugeint_t dummy = 0;
			CombineLevel(s->probabilistic_total8, d->probabilistic_total8, s->exact_total8, d->exact_total8, 8);
			CombineLevel(s->probabilistic_total16, d->probabilistic_total16, s->exact_total16, d->exact_total16, 16);
			CombineLevel(s->probabilistic_total32, d->probabilistic_total32, s->exact_total32, d->exact_total32, 32);
			CombineLevel(s->probabilistic_total64, d->probabilistic_total64, s->exact_total64, d->exact_total64, 64);
			CombineLevel(s->probabilistic_total128, d->probabilistic_total128, dummy, dummy, 64);
#endif
			dst_state[i]->exact_count += src_state[i]->exact_count;
		}
	}
}

// Combine for double state
AUTOVECTORIZE static void PacSumCombineDouble(Vector &src, Vector &dst, idx_t count, ArenaAllocator &allocator) {
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
		src_state[i]->Flush(allocator);
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
		state[i]->Flush(input.allocator);
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
void PacSumCombineDoubleWrapper(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	PacSumCombineDouble(src, dst, count, aggr.allocator);
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
	return sizeof(PacSumIntState<true>); // signed (true) and unsigned (false) have the same size
}

static void PacSumIntInitialize(const AggregateFunction &, data_ptr_t state_p) {
	memset(state_p, 0, sizeof(PacSumIntState<true>)); // memset to 0 works for both signed and unsigned
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

} // namespace duckdb
