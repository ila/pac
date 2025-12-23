#include "include/pac_sum.hpp"

namespace duckdb {

// SIGNED is compile-time known, so for unsigned the negative cases (value < 0) will be compiled away
#define ACCUMULATE_BITMARGIN      2 // val must be 2 bits shorter than the accumulator to allow >=4 updates without overflow
#define UPPERBOUND_BITWIDTH(bits) (1LL << ((bits - SIGNED) - ACCUMULATE_BITMARGIN))
#define LOWERBOUND_BITWIDTH(bits) -(static_cast<int64_t>(SIGNED) << ((bits - SIGNED) - ACCUMULATE_BITMARGIN))

template <bool SIGNED>
AUTOVECTORIZE inline void // main worker function for probabilistically adding one INTEGER to the 64 (sub)totals
PacSumUpdateOne(PacSumIntState<SIGNED> &state, uint64_t key_hash, typename PacSumIntState<SIGNED>::T64 value) {
#ifndef PAC_SUM_NONCASCADING
	if ((SIGNED && value < 0) ? (value >= LOWERBOUND_BITWIDTH(8)) : (value < UPPERBOUND_BITWIDTH(8))) {
		state.Flush8(value);
		AddToTotalsSWAR<int8_t, uint8_t, 0x0101010101010101ULL>(state.subtotals8, value, key_hash);
	} else if ((SIGNED && value < 0) ? (value >= LOWERBOUND_BITWIDTH(16)) : (value < UPPERBOUND_BITWIDTH(16))) {
		state.Flush16(value);
		AddToTotalsSWAR<int16_t, uint16_t, 0x0001000100010001ULL>(state.subtotals16, value, key_hash);
	} else if ((SIGNED && value < 0) ? (value >= LOWERBOUND_BITWIDTH(32)) : (value < UPPERBOUND_BITWIDTH(32))) {
		state.Flush32(value);
		AddToTotalsSWAR<int32_t, uint32_t, 0x0000000100000001ULL>(state.subtotals32, value, key_hash);
	} else {
		state.Flush64(value);
		AddToTotalsSimple(state.subtotals64, value, key_hash);
	}
#else
	AddToTotalsSimple(state.probabilistic_totals, value, key_hash);
#endif
}

template <bool SIGNED>
AUTOVECTORIZE inline void // main worker function for probabilistically adding one DOUBLE to the 64 sum totals
PacSumUpdateOne(PacSumDoubleState &state, uint64_t key_hash, double value) {
#ifndef PAC_SUM_NONCASCADING
	if (FloatSubtotalFitsDouble(value)) {
		AddToTotalsSimple(state.probabilistic_subtotals, static_cast<float>(value), key_hash);
		state.Flush32(value);
		return;
	}
#endif
	AddToTotalsSimple(state.probabilistic_totals, value, key_hash);
}

// Overload for HUGEINT input - adds directly to hugeint_t totals (no cascading since values don't fit in subtotals)
template <bool SIGNED>
AUTOVECTORIZE inline void PacSumUpdateOne(PacSumIntState<SIGNED> &state, uint64_t key_hash, hugeint_t value) {
	for (int j = 0; j < 64; j++) {
		if ((key_hash >> j) & 1ULL) {
			state.probabilistic_totals[j] += value;
		}
	}
}

template <class State, bool SIGNED, class VALUE_TYPE, class INPUT_TYPE>
static void PacSumUpdate(Vector inputs[], data_ptr_t state_p, idx_t count) {
	auto &state = *reinterpret_cast<State *>(state_p);
	if (state.seen_null) {
		return;
	}
	UnifiedVectorFormat hash_data, value_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);

	for (idx_t i = 0; i < count; i++) {
		auto hash_idx = hash_data.sel->get_index(i);
		auto value_idx = value_data.sel->get_index(i);
		if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
			state.seen_null = true;
			return;
		}
		PacSumUpdateOne<SIGNED>(state, hashes[hash_idx], ConvertValue<VALUE_TYPE>::convert(values[value_idx]));
	}
}

template <class State, bool SIGNED, class VALUE_TYPE, class INPUT_TYPE>
static void PacSumScatterUpdate(Vector inputs[], Vector &states, idx_t count) {
	UnifiedVectorFormat hash_data, value_data, sdata;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	states.ToUnifiedFormat(count, sdata);

	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);
	auto state_ptrs = UnifiedVectorFormat::GetData<State *>(sdata);

	for (idx_t i = 0; i < count; i++) {
		auto hash_idx = hash_data.sel->get_index(i);
		auto value_idx = value_data.sel->get_index(i);
		auto state = state_ptrs[sdata.sel->get_index(i)];
		if (!state->seen_null) {
			if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
				state->seen_null = true;
			} else {
				PacSumUpdateOne<SIGNED>(*state, hashes[hash_idx], ConvertValue<VALUE_TYPE>::convert(values[value_idx]));
			}
		}
	}
}

template <class State>
AUTOVECTORIZE static void PacSumCombine(Vector &src, Vector &dst, idx_t count) {
	auto src_state = FlatVector::GetData<State *>(src);
	auto dst_state = FlatVector::GetData<State *>(dst);
	for (idx_t i = 0; i < count; i++) {
		if (src_state[i]->seen_null) {
			dst_state[i]->seen_null = true;
		}
		if (!dst_state[i]->seen_null) {
#ifndef PAC_SUM_NONCASCADING
			src_state[i]->Flush(); // flush source before reading from it
#endif
			for (int j = 0; j < 64; j++) {
				dst_state[i]->probabilistic_totals[j] += src_state[i]->probabilistic_totals[j];
			}
		}
	}
}

template <class State, class ACC_TYPE>
static void PacSumFinalize(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	auto state = FlatVector::GetData<State *>(states);
	auto data = FlatVector::GetData<ACC_TYPE>(result);
	auto &result_mask = FlatVector::Validity(result);
	thread_local std::mt19937_64 gen(std::random_device {}());
	double mi = input.bind_data ? input.bind_data->Cast<PacBindData>().mi : 128.0; // 128 is default

	for (idx_t i = 0; i < count; i++) {
		if (state[i]->seen_null) {
			result_mask.SetInvalid(offset + i);
		} else {
#ifndef PAC_SUM_NONCASCADING
			state[i]->Flush();
#endif
			double buf[64];
			ToDoubleArray(state[i]->probabilistic_totals, buf);
			data[offset + i] = // when choosing any one of the totals we go for #42 (but one counts from 0 ofc)
			    FromDouble<ACC_TYPE>(PacNoisySampleFrom64Counters(buf, mi, gen)) + state[i]->probabilistic_totals[41];
		}
	}
}

// instantiate Update methods
void PacSumUpdateTinyInt(Vector inputs[], AggregateInputData &, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<true>, true, int64_t, int8_t>(inputs, state_p, count);
}
void PacSumUpdateSmallInt(Vector inputs[], AggregateInputData &, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<true>, true, int64_t, int16_t>(inputs, state_p, count);
}
void PacSumUpdateInteger(Vector inputs[], AggregateInputData &, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<true>, true, int64_t, int32_t>(inputs, state_p, count);
}
void PacSumUpdateBigInt(Vector inputs[], AggregateInputData &, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<true>, true, int64_t, int64_t>(inputs, state_p, count);
}
void PacSumUpdateHugeInt(Vector inputs[], AggregateInputData &, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<true>, true, hugeint_t, hugeint_t>(inputs, state_p, count);
}
void PacSumUpdateUTinyInt(Vector inputs[], AggregateInputData &, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<false>, false, uint64_t, uint8_t>(inputs, state_p, count);
}
void PacSumUpdateUSmallInt(Vector inputs[], AggregateInputData &, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<false>, false, uint64_t, uint16_t>(inputs, state_p, count);
}
void PacSumUpdateUInteger(Vector inputs[], AggregateInputData &, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<false>, false, uint64_t, uint32_t>(inputs, state_p, count);
}
void PacSumUpdateUBigInt(Vector inputs[], AggregateInputData &, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumIntState<false>, false, uint64_t, uint64_t>(inputs, state_p, count);
}
void PacSumUpdateUHugeInt(Vector inputs[], AggregateInputData &, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumDoubleState, true, double, uhugeint_t>(inputs, state_p, count);
}
void PacSumUpdateFloat(Vector inputs[], AggregateInputData &, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumDoubleState, true, double, float>(inputs, state_p, count);
}
void PacSumUpdateDouble(Vector inputs[], AggregateInputData &, idx_t, data_ptr_t state_p, idx_t count) {
	PacSumUpdate<PacSumDoubleState, true, double, double>(inputs, state_p, count);
}

// instantiate ScatterUpdate methods
void PacSumScatterUpdateTinyInt(Vector inputs[], AggregateInputData &, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<true>, true, int64_t, int8_t>(inputs, states, count);
}
void PacSumScatterUpdateSmallInt(Vector inputs[], AggregateInputData &, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<true>, true, int64_t, int16_t>(inputs, states, count);
}
void PacSumScatterUpdateInteger(Vector inputs[], AggregateInputData &, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<true>, true, int64_t, int32_t>(inputs, states, count);
}
void PacSumScatterUpdateBigInt(Vector inputs[], AggregateInputData &, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<true>, true, int64_t, int64_t>(inputs, states, count);
}
void PacSumScatterUpdateHugeInt(Vector inputs[], AggregateInputData &, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<true>, true, hugeint_t, hugeint_t>(inputs, states, count);
}
void PacSumScatterUpdateUTinyInt(Vector inputs[], AggregateInputData &, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<false>, false, uint64_t, uint8_t>(inputs, states, count);
}
void PacSumScatterUpdateUSmallInt(Vector inputs[], AggregateInputData &, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<false>, false, uint64_t, uint16_t>(inputs, states, count);
}
void PacSumScatterUpdateUInteger(Vector inputs[], AggregateInputData &, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<false>, false, uint64_t, uint32_t>(inputs, states, count);
}
void PacSumScatterUpdateUBigInt(Vector inputs[], AggregateInputData &, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumIntState<false>, false, uint64_t, uint64_t>(inputs, states, count);
}
void PacSumScatterUpdateUHugeInt(Vector inputs[], AggregateInputData &, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumDoubleState, true, double, uhugeint_t>(inputs, states, count);
}
void PacSumScatterUpdateFloat(Vector inputs[], AggregateInputData &, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumDoubleState, true, double, float>(inputs, states, count);
}
void PacSumScatterUpdateDouble(Vector inputs[], AggregateInputData &, idx_t, Vector &states, idx_t count) {
	PacSumScatterUpdate<PacSumDoubleState, true, double, double>(inputs, states, count);
}

// instantiate Combine methods
void PacSumCombineSigned(Vector &src, Vector &dst, AggregateInputData &, idx_t count) {
	PacSumCombine<PacSumIntState<true>>(src, dst, count);
}
void PacSumCombineUnsigned(Vector &src, Vector &dst, AggregateInputData &, idx_t count) {
	PacSumCombine<PacSumIntState<false>>(src, dst, count);
}
void PacSumCombineDouble(Vector &src, Vector &dst, AggregateInputData &, idx_t count) {
	PacSumCombine<PacSumDoubleState>(src, dst, count);
}

// instantiate Finalize methods
void PacSumFinalizeSigned(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	PacSumFinalize<PacSumIntState<true>, hugeint_t>(states, input, result, count, offset);
}
void PacSumFinalizeUnsigned(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	PacSumFinalize<PacSumIntState<false>, hugeint_t>(states, input, result, count, offset);
}
void PacSumFinalizeDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	PacSumFinalize<PacSumDoubleState, double>(states, input, result, count, offset);
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
	return make_uniq<PacBindData>(mi);
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
                   aggregate_combine_t combine, aggregate_finalize_t finalize, aggregate_simple_update_t update) {
	set.AddFunction(AggregateFunction("pac_sum", {LogicalType::UBIGINT, value_type}, result_type, state_size, init,
	                                  scatter, combine, finalize, FunctionNullHandling::DEFAULT_NULL_HANDLING, update,
	                                  PacSumBind));
	set.AddFunction(AggregateFunction("pac_sum", {LogicalType::UBIGINT, value_type, LogicalType::DOUBLE}, result_type,
	                                  state_size, init, scatter, combine, finalize,
	                                  FunctionNullHandling::DEFAULT_NULL_HANDLING, update, PacSumBind));
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
	       PacSumScatterUpdateUHugeInt, PacSumCombineDouble, PacSumFinalizeDouble, PacSumUpdateUHugeInt);

	// Floating point (accumulate to double, return DOUBLE)
	AddFcn(fcn_set, LogicalType::FLOAT, LogicalType::DOUBLE, PacSumDoubleStateSize, PacSumDoubleInitialize,
	       PacSumScatterUpdateFloat, PacSumCombineDouble, PacSumFinalizeDouble, PacSumUpdateFloat);
	AddFcn(fcn_set, LogicalType::DOUBLE, LogicalType::DOUBLE, PacSumDoubleStateSize, PacSumDoubleInitialize,
	       PacSumScatterUpdateDouble, PacSumCombineDouble, PacSumFinalizeDouble, PacSumUpdateDouble);

	loader.RegisterFunction(fcn_set);
}

} // namespace duckdb
