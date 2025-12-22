#include "include/pac_sum.hpp"

namespace duckdb {

// Bind function for pac_sum with optional mi parameter
static unique_ptr<FunctionData> PacSumBind(ClientContext &context, AggregateFunction &,
                                           vector<unique_ptr<Expression>> &arguments) {
	double mi = 128.0; // default
	if (arguments.size() >= 3) {
		if (!arguments[2]->IsFoldable()) {
			throw InvalidInputException("pac_sum: mi parameter must be a constant");
		}
		auto mi_val = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
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

// Templated update internal - SIGNED selects signed/unsigned overflow checks
template <bool SIGNED>
AUTOVECTORIZE static inline void
PacSumIntUpdateInternal(PacSumIntState<SIGNED> &state, uint64_t key_hash,
                        typename std::conditional<SIGNED, int64_t, uint64_t>::type val) {
#ifndef PAC_SUM_NONCASCADING
	if (SIGNED ? VALUE_FITS_SIGNED(val, 8) : VALUE_FITS_UNSIGNED(val, 8)) {
		AddToTotalsINT(state.totals8, val, key_hash);
		state.Flush8(false);
	} else if (SIGNED ? VALUE_FITS_SIGNED(val, 16) : VALUE_FITS_UNSIGNED(val, 16)) {
		AddToTotalsINT(state.totals16, val, key_hash);
		state.Flush16(false);
	} else if (SIGNED ? VALUE_FITS_SIGNED(val, 32) : VALUE_FITS_UNSIGNED(val, 32)) {
		AddToTotalsINT(state.totals32, val, key_hash);
		state.Flush32(false);
	} else {
		AddToTotalsINT(state.totals64, val, key_hash);
		state.Flush64(false);
	}
#else
	AddToTotalsINT(state.totals128, val, key_hash); // directly add into int128_t (slow..)
#endif
}

// Templated integer update - SIGNED selects state type and value cast
template <bool SIGNED, class INPUT_TYPE>
static inline void PacSumIntUpdate(Vector param[], AggregateInputData &, idx_t np, data_ptr_t state_p, idx_t cnt) {
	using ValueType = typename std::conditional<SIGNED, int64_t, uint64_t>::type;
	D_ASSERT(np == 2 || np == 3); // optional mi param (unused here) can make it 3
	auto &state = *reinterpret_cast<PacSumIntState<SIGNED> *>(state_p);
	if (!state.seen_null) {
		UnifiedVectorFormat hash_data, value_data;
		param[0].ToUnifiedFormat(cnt, hash_data);
		param[1].ToUnifiedFormat(cnt, value_data);
		auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
		auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);

		for (idx_t i = 0; i < cnt; i++) {
			auto hash_idx = hash_data.sel->get_index(i);
			auto value_idx = value_data.sel->get_index(i);
			if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
				state.seen_null = true;
			} else {
				PacSumIntUpdateInternal<SIGNED>(state, hashes[hash_idx], static_cast<ValueType>(values[value_idx]));
			}
		}
	}
}

// Templated integer scatter update - SIGNED selects state type and value cast
template <bool SIGNED, class INPUT_TYPE>
static inline void PacSumIntScatterUpdate(Vector param[], AggregateInputData &, idx_t np, Vector &states, idx_t cnt) {
	using ValueType = typename std::conditional<SIGNED, int64_t, uint64_t>::type;
	D_ASSERT(np == 2 || np == 3); // optional mi param (unused here) can make it 3

	UnifiedVectorFormat hash_data, value_data, sdata;
	param[0].ToUnifiedFormat(cnt, hash_data);
	param[1].ToUnifiedFormat(cnt, value_data);
	states.ToUnifiedFormat(cnt, sdata);

	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);
	auto state_ptrs = UnifiedVectorFormat::GetData<PacSumIntState<SIGNED> *>(sdata);

	for (idx_t i = 0; i < cnt; i++) {
		auto hash_idx = hash_data.sel->get_index(i);
		auto value_idx = value_data.sel->get_index(i);
		auto state = state_ptrs[sdata.sel->get_index(i)];
		if (!state->seen_null) {
			if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
				state->seen_null = true;
			} else {
				PacSumIntUpdateInternal<SIGNED>(*state, hashes[hash_idx], static_cast<ValueType>(values[value_idx]));
			}
		}
	}
}

template <bool SIGNED, class ACC_TYPE>
AUTOVECTORIZE static inline void PacSumIntCombine(Vector &src, Vector &dst, AggregateInputData &, idx_t cnt) {
	auto src_state = FlatVector::GetData<PacSumIntState<SIGNED> *>(src);
	auto dst_state = FlatVector::GetData<PacSumIntState<SIGNED> *>(dst);
	for (idx_t i = 0; i < cnt; i++) {
		if (src_state[i]->seen_null) {
			dst_state[i]->seen_null = true;
		}
		if (!dst_state[i]->seen_null) {
#ifndef PAC_SUM_NONCASCADING
			dst_state[i]->Flush8(true); // Force flush all levels to totals128
#endif
			for (int j = 0; j < 64; j++) {
				dst_state[i]->totals128[j] += src_state[i]->totals128[j]; // combine totals128
			}
		}
	}
}

template <bool SIGNED, class ACC_TYPE>
static inline void PacSumIntFinalize(Vector &states, AggregateInputData &input, Vector &res, idx_t cnt, idx_t off) {
	auto state = FlatVector::GetData<PacSumIntState<SIGNED> *>(states);
	auto data = FlatVector::GetData<ACC_TYPE>(res);
	auto &result_mask = FlatVector::Validity(res);
	thread_local std::mt19937_64 gen(std::random_device {}());
	double mi = input.bind_data ? input.bind_data->Cast<PacBindData>().mi : 128.0; // 128 is default

	for (idx_t i = 0; i < cnt; i++) {
		if (state[i]->seen_null) {
			result_mask.SetInvalid(off + i);
		} else {
#ifndef PAC_SUM_NONCASCADING
			state[i]->Flush8(true); // Force flush all levels to totals128
#endif
			double totals_d[64]; // Convert totals128 to double array
			ToDoubleArray(state[i]->totals128, totals_d);
			data[off + i] = // when choosing any one of the totals we go for #42 (but one counts from 0 ofc)
			    FromDouble<ACC_TYPE>(PacNoisySampleFrom64Counters(totals_d, mi, gen)) + state[i]->totals128[41];
		}
	}
}

static idx_t PacSumDoubleStateSize(const AggregateFunction &) {
	return sizeof(PacSumDoubleState);
}
static void PacSumDoubleInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	memset(state_ptr, 0, sizeof(PacSumDoubleState));
}

static inline void PacSumDoubleUpdateInternal(PacSumDoubleState &state, uint64_t key_hash, double val) {
#ifndef PAC_SUM_NONCASCADING
	if (AdditionFitsInFloat(val)) {
		AddToTotalsDOUBLE(state.totals32, static_cast<float>(val), key_hash);
		state.Flush32(false);
		return;
	}
#endif
	AddToTotalsDOUBLE(state.totals64, val, key_hash);
}

// Double cascade update (ungrouped global aggregate) - also handles hugeint_t/uhugeint_t via ToDouble
template <class INPUT_TYPE>
static void PacSumDoubleUpdate(Vector param[], AggregateInputData &, idx_t np, data_ptr_t state_p, idx_t cnt) {
	D_ASSERT(np == 2 || np == 3); // optional mi param (unused here) can make it 3
	auto &state = *reinterpret_cast<PacSumDoubleState *>(state_p);
	if (state.seen_null) {
		return; // aggregate returns NULL after seeing a NULL value, whatever follows
	}
	UnifiedVectorFormat hash_data, value_data;
	param[0].ToUnifiedFormat(cnt, hash_data);
	param[1].ToUnifiedFormat(cnt, value_data);

	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);
	for (idx_t i = 0; i < cnt; i++) {
		auto hash_idx = hash_data.sel->get_index(i);
		auto value_idx = value_data.sel->get_index(i);
		if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
			state.seen_null = true;
			return;
		}
		PacSumDoubleUpdateInternal(state, hashes[hash_idx], ToDouble(values[value_idx]));
	}
}

// Double cascade scatter update (GROUP BY case) - also handles hugeint_t/uhugeint_t via ToDouble
template <class INPUT_TYPE>
static void PacSumDoubleScatterUpdate(Vector param[], AggregateInputData &, idx_t np, Vector &states, idx_t cnt) {
	D_ASSERT(np == 2 || np == 3); // optional mi param (unused here) can make it 3
	UnifiedVectorFormat hash_data, value_data, sdata;
	param[0].ToUnifiedFormat(cnt, hash_data);
	param[1].ToUnifiedFormat(cnt, value_data);
	states.ToUnifiedFormat(cnt, sdata);

	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);
	auto state_ptrs = UnifiedVectorFormat::GetData<PacSumDoubleState *>(sdata);

	for (idx_t i = 0; i < cnt; i++) {
		auto state_idx = sdata.sel->get_index(i);
		auto state = state_ptrs[state_idx];
		if (!state->seen_null) {
			auto hash_idx = hash_data.sel->get_index(i);
			auto value_idx = value_data.sel->get_index(i);
			if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
				state->seen_null = true;
			} else {
				PacSumDoubleUpdateInternal(*state, hashes[hash_idx], ToDouble(values[value_idx]));
			}
		}
	}
}

AUTOVECTORIZE static void PacSumDoubleCombine(Vector &src, Vector &dst, AggregateInputData &, idx_t cnt) {
	auto src_state = FlatVector::GetData<PacSumDoubleState *>(src);
	auto dst_state = FlatVector::GetData<PacSumDoubleState *>(dst);
	for (idx_t i = 0; i < cnt; i++) {
		if (src_state[i]->seen_null) {
			dst_state[i]->seen_null = true;
		}
		if (!dst_state[i]->seen_null) {
#ifndef PAC_SUM_NONCASCADING
			dst_state[i]->Flush32(true); // flush all data to totals64
#endif
			for (int j = 0; j < 64; j++) {
				dst_state[i]->totals64[j] += src_state[i]->totals64[j];
			}
		}
	}
}

static void PacSumDoubleFinalize(Vector &states, AggregateInputData &input, Vector &res, idx_t cnt, idx_t off) {
	auto state = FlatVector::GetData<PacSumDoubleState *>(states);
	auto data = FlatVector::GetData<double>(res);
	auto &result_mask = FlatVector::Validity(res);
	thread_local std::mt19937_64 gen(std::random_device {}());
	double mi = input.bind_data->Cast<PacBindData>().mi;

	for (idx_t i = 0; i < cnt; i++) {
		if (state[i]->seen_null) {
			result_mask.SetInvalid(off + i);
		} else {
#ifndef PAC_SUM_NONCASCADING
			state[i]->Flush32(true); // flush all data to totals64
#endif
			double totals_d[64];
			ToDoubleArray(state[i]->totals64, totals_d); // (fast) dummy copy for double, conversion for hugeint
			data[off + i] = // when choosing any one of the totals we go for #42 (but one counts from 0 ofc)
			    FromDouble<double>(PacNoisySampleFrom64Counters(totals_d, mi, gen)) + state[i]->totals64[41];
		}
	}
}

// instantiate Update methods
static void PacSumUpdateTinyInt(Vector param[], AggregateInputData &aggr, idx_t np, data_ptr_t state_p, idx_t cnt) {
	PacSumIntUpdate<true, int8_t>(param, aggr, np, state_p, cnt);
}
static void PacSumUpdateSmallInt(Vector param[], AggregateInputData &aggr, idx_t np, data_ptr_t state_p, idx_t cnt) {
	PacSumIntUpdate<true, int16_t>(param, aggr, np, state_p, cnt);
}
static void PacSumUpdateInteger(Vector param[], AggregateInputData &aggr, idx_t np, data_ptr_t state_p, idx_t cnt) {
	PacSumIntUpdate<true, int32_t>(param, aggr, np, state_p, cnt);
}
static void PacSumUpdateBigInt(Vector param[], AggregateInputData &aggr, idx_t np, data_ptr_t state_p, idx_t cnt) {
	PacSumIntUpdate<true, int64_t>(param, aggr, np, state_p, cnt);
}
static void PacSumUpdateHugeInt(Vector param[], AggregateInputData &aggr, idx_t np, data_ptr_t state_p, idx_t cnt) {
	PacSumDoubleUpdate<hugeint_t>(param, aggr, np, state_p, cnt); // double update with conversion from hugeint_t
}
static void PacSumUpdateUTinyInt(Vector param[], AggregateInputData &aggr, idx_t np, data_ptr_t state_p, idx_t cnt) {
	PacSumIntUpdate<false, uint8_t>(param, aggr, np, state_p, cnt);
}
static void PacSumUpdateUSmallInt(Vector param[], AggregateInputData &aggr, idx_t np, data_ptr_t state_p, idx_t cnt) {
	PacSumIntUpdate<false, uint16_t>(param, aggr, np, state_p, cnt);
}
static void PacSumUpdateUInteger(Vector param[], AggregateInputData &aggr, idx_t np, data_ptr_t state_p, idx_t cnt) {
	PacSumIntUpdate<false, uint32_t>(param, aggr, np, state_p, cnt);
}
static void PacSumUpdateUBigInt(Vector param[], AggregateInputData &aggr, idx_t np, data_ptr_t state_p, idx_t cnt) {
	PacSumIntUpdate<false, uint64_t>(param, aggr, np, state_p, cnt);
}
static void PacSumUpdateUHugeInt(Vector param[], AggregateInputData &aggr, idx_t np, data_ptr_t state_p, idx_t cnt) {
	PacSumDoubleUpdate<uhugeint_t>(param, aggr, np, state_p, cnt); // double update with conversion from uhugeint_t
}
static void PacSumUpdateFloat(Vector param[], AggregateInputData &aggr, idx_t np, data_ptr_t state_p, idx_t cnt) {
	PacSumDoubleUpdate<float>(param, aggr, np, state_p, cnt);
}
static void PacSumUpdateDouble(Vector param[], AggregateInputData &aggr, idx_t np, data_ptr_t state_p, idx_t cnt) {
	PacSumDoubleUpdate<double>(param, aggr, np, state_p, cnt);
}

// instantiate ScatterUpdate methods
static void PacSumScatterTinyInt(Vector param[], AggregateInputData &aggr, idx_t np, Vector &state_p, idx_t cnt) {
	PacSumIntScatterUpdate<true, int8_t>(param, aggr, np, state_p, cnt);
}
static void PacSumScatterSmallInt(Vector param[], AggregateInputData &aggr, idx_t np, Vector &state_p, idx_t cnt) {
	PacSumIntScatterUpdate<true, int16_t>(param, aggr, np, state_p, cnt);
}
static void PacSumScatterInteger(Vector param[], AggregateInputData &aggr, idx_t np, Vector &state_p, idx_t cnt) {
	PacSumIntScatterUpdate<true, int32_t>(param, aggr, np, state_p, cnt);
}
static void PacSumScatterBigInt(Vector param[], AggregateInputData &aggr, idx_t np, Vector &state_p, idx_t cnt) {
	PacSumIntScatterUpdate<true, int64_t>(param, aggr, np, state_p, cnt);
}
static void PacSumScatterHugeInt(Vector param[], AggregateInputData &aggr, idx_t np, Vector &state_p, idx_t cnt) {
	PacSumDoubleScatterUpdate<hugeint_t>(param, aggr, np, state_p, cnt); // scatter doubles after cast from hugeint_t
}
static void PacSumScatterUTinyInt(Vector param[], AggregateInputData &aggr, idx_t np, Vector &state_p, idx_t cnt) {
	PacSumIntScatterUpdate<false, uint8_t>(param, aggr, np, state_p, cnt);
}
static void PacSumScatterUSmallInt(Vector param[], AggregateInputData &aggr, idx_t np, Vector &state_p, idx_t cnt) {
	PacSumIntScatterUpdate<false, uint16_t>(param, aggr, np, state_p, cnt);
}
static void PacSumScatterUInteger(Vector param[], AggregateInputData &aggr, idx_t np, Vector &state_p, idx_t cnt) {
	PacSumIntScatterUpdate<false, uint32_t>(param, aggr, np, state_p, cnt);
}
static void PacSumScatterUBigInt(Vector param[], AggregateInputData &aggr, idx_t np, Vector &state_p, idx_t cnt) {
	PacSumIntScatterUpdate<false, uint64_t>(param, aggr, np, state_p, cnt);
}
static void PacSumScatterUHugeInt(Vector param[], AggregateInputData &aggr, idx_t np, Vector &state_p, idx_t cnt) {
	PacSumDoubleScatterUpdate<uhugeint_t>(param, aggr, np, state_p, cnt); // scatter doubles after cast from uhugeint_t
}
static void PacSumScatterFloat(Vector param[], AggregateInputData &aggr, idx_t np, Vector &states, idx_t cnt) {
	PacSumDoubleScatterUpdate<float>(param, aggr, np, states, cnt);
}
static void PacSumScatterDouble(Vector param[], AggregateInputData &aggr, idx_t np, Vector &states, idx_t cnt) {
	PacSumDoubleScatterUpdate<double>(param, aggr, np, states, cnt);
}

// instantiate Combine methods
static void PacSumCombineTinyInt(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t cnt) {
	PacSumIntCombine<true, hugeint_t>(src, dst, aggr, cnt);
}
static void PacSumCombineSmallInt(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t cnt) {
	PacSumIntCombine<true, hugeint_t>(src, dst, aggr, cnt);
}
static void PacSumCombineInteger(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t cnt) {
	PacSumIntCombine<true, hugeint_t>(src, dst, aggr, cnt);
}
static void PacSumCombineBigInt(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t cnt) {
	PacSumIntCombine<true, hugeint_t>(src, dst, aggr, cnt);
}
static void PacSumCombineUTinyInt(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t cnt) {
	PacSumIntCombine<false, hugeint_t>(src, dst, aggr, cnt);
}
static void PacSumCombineUSmallInt(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t cnt) {
	PacSumIntCombine<false, hugeint_t>(src, dst, aggr, cnt);
}
static void PacSumCombineUInteger(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t cnt) {
	PacSumIntCombine<false, hugeint_t>(src, dst, aggr, cnt);
}
static void PacSumCombineUBigInt(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t cnt) {
	PacSumIntCombine<false, hugeint_t>(src, dst, aggr, cnt);
}

// instantiate Finalize methods
static void PacSumFinalizeTinyInt(Vector &states, AggregateInputData &aggr, Vector &res, idx_t cnt, idx_t off) {
	PacSumIntFinalize<true, hugeint_t>(states, aggr, res, cnt, off);
}
static void PacSumFinalizeSmallInt(Vector &states, AggregateInputData &aggr, Vector &res, idx_t cnt, idx_t off) {
	PacSumIntFinalize<true, hugeint_t>(states, aggr, res, cnt, off);
}
static void PacSumFinalizeInteger(Vector &states, AggregateInputData &aggr, Vector &res, idx_t cnt, idx_t off) {
	PacSumIntFinalize<true, hugeint_t>(states, aggr, res, cnt, off);
}
static void PacSumFinalizeBigInt(Vector &states, AggregateInputData &aggr, Vector &res, idx_t cnt, idx_t off) {
	PacSumIntFinalize<true, hugeint_t>(states, aggr, res, cnt, off);
}
static void PacSumFinalizeUTinyInt(Vector &states, AggregateInputData &aggr, Vector &res, idx_t cnt, idx_t off) {
	PacSumIntFinalize<false, hugeint_t>(states, aggr, res, cnt, off);
}
static void PacSumFinalizeUSmallInt(Vector &states, AggregateInputData &aggr, Vector &res, idx_t cnt, idx_t off) {
	PacSumIntFinalize<false, hugeint_t>(states, aggr, res, cnt, off);
}
static void PacSumFinalizeUInteger(Vector &states, AggregateInputData &aggr, Vector &res, idx_t cnt, idx_t off) {
	PacSumIntFinalize<false, hugeint_t>(states, aggr, res, cnt, off);
}
static void PacSumFinalizeUBigInt(Vector &states, AggregateInputData &aggr, Vector &res, idx_t cnt, idx_t off) {
	PacSumIntFinalize<false, hugeint_t>(states, aggr, res, cnt, off);
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
	       PacSumScatterTinyInt, PacSumCombineTinyInt, PacSumFinalizeTinyInt, PacSumUpdateTinyInt);
	AddFcn(fcn_set, LogicalType::BOOLEAN, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterTinyInt, PacSumCombineTinyInt, PacSumFinalizeTinyInt, PacSumUpdateTinyInt);
	AddFcn(fcn_set, LogicalType::SMALLINT, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterSmallInt, PacSumCombineSmallInt, PacSumFinalizeSmallInt, PacSumUpdateSmallInt);
	AddFcn(fcn_set, LogicalType::INTEGER, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterInteger, PacSumCombineInteger, PacSumFinalizeInteger, PacSumUpdateInteger);
	AddFcn(fcn_set, LogicalType::BIGINT, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterBigInt, PacSumCombineBigInt, PacSumFinalizeBigInt, PacSumUpdateBigInt);

	// Unsigned integers (idem)
	AddFcn(fcn_set, LogicalType::UTINYINT, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterUTinyInt, PacSumCombineUTinyInt, PacSumFinalizeUTinyInt, PacSumUpdateUTinyInt);
	AddFcn(fcn_set, LogicalType::USMALLINT, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterUSmallInt, PacSumCombineUSmallInt, PacSumFinalizeUSmallInt, PacSumUpdateUSmallInt);
	AddFcn(fcn_set, LogicalType::UINTEGER, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterUInteger, PacSumCombineUInteger, PacSumFinalizeUInteger, PacSumUpdateUInteger);
	AddFcn(fcn_set, LogicalType::UBIGINT, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize,
	       PacSumScatterUBigInt, PacSumCombineUBigInt, PacSumFinalizeUBigInt, PacSumUpdateUBigInt);

	// [u]hugeint_t converts during the [scatter]update to double, and otherwise just uses the double logic
	AddFcn(fcn_set, LogicalType::HUGEINT, LogicalType::DOUBLE, PacSumDoubleStateSize, PacSumDoubleInitialize,
	       PacSumScatterHugeInt, PacSumDoubleCombine, PacSumDoubleFinalize, PacSumUpdateHugeInt);
	AddFcn(fcn_set, LogicalType::UHUGEINT, LogicalType::DOUBLE, PacSumDoubleStateSize, PacSumDoubleInitialize,
	       PacSumScatterUHugeInt, PacSumDoubleCombine, PacSumDoubleFinalize, PacSumUpdateUHugeInt);

	// Floating point (accumulate to double, return DOUBLE)
	AddFcn(fcn_set, LogicalType::FLOAT, LogicalType::DOUBLE, PacSumDoubleStateSize, PacSumDoubleInitialize,
	       PacSumScatterFloat, PacSumDoubleCombine, PacSumDoubleFinalize, PacSumUpdateFloat);
	AddFcn(fcn_set, LogicalType::DOUBLE, LogicalType::DOUBLE, PacSumDoubleStateSize, PacSumDoubleInitialize,
	       PacSumScatterDouble, PacSumDoubleCombine, PacSumDoubleFinalize, PacSumUpdateDouble);

	loader.RegisterFunction(fcn_set);
}

} // namespace duckdb
