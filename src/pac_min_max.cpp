#include "include/pac_min_max.hpp"

namespace duckdb {

// Integer state alias: PacMinMaxIntState<SIGNED, IS_MAX, MAXLEVEL>
template <bool SIGNED, bool IS_MAX, int MAXLEVEL = 5>
using PacMinMaxIntState = PacMinMaxState<false, SIGNED, IS_MAX, MAXLEVEL>;

// Float state alias: PacMinMaxFloatState<IS_MAX, MAXLEVEL>
template <bool IS_MAX, int MAXLEVEL = 2>
using PacMinMaxFloatState = PacMinMaxState<true, true, IS_MAX, MAXLEVEL>;

// ============================================================================
// PacMinMaxUpdateOne - process one value into the aggregation state (64 extremes)
// Works for both int and float states via the common interface
// ============================================================================

template <class State, bool IS_MAX>
AUTOVECTORIZE static inline void PacMinMaxUpdateOne(State &state, uint64_t key_hash, typename State::ValueType value,
                                                    ArenaAllocator &allocator) {
#ifndef PAC_MINMAX_NONBANKED
	if (!state.initialized) {
		state.AllocateFirstLevel(allocator); // cascading mode allocates levels in increasing width, upgrades upon need
	}
#endif
#ifndef PAC_MINMAX_NOBOUNDOPT
	if (!PAC_IS_BETTER(value, state.global_bound)) { // Compare value against global_bound
		return;                                      // early out
	}
#endif
#ifdef PAC_MINMAX_NONBANKED
	UpdateExtremes<typename State::ValueType, IS_MAX>(state.extremes, key_hash, value);
	state.RecomputeBound();
#else
	state.MaybeUpgrade(allocator, value);        // Upgrade level if value doesn't fit in current level
	state.UpdateAtCurrentLevel(key_hash, value); // here the SIMD magic happens
	state.RecomputeBound();                      // once every 2048 calls recomputes the bound
#endif
}

// DuckDB method for processing one vector for  aggregation without GROUP BY (there is only a single state)
template <class State, bool IS_MAX, class INPUT_TYPE>
static void PacMinMaxUpdate(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	auto &state = *reinterpret_cast<State *>(state_p);
#ifdef PAC_MINMAX_UNSAFENULL
	if (state.seen_null)
		return;
#endif

	UnifiedVectorFormat hash_data, value_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);

	if (hash_data.validity.AllValid() && value_data.validity.AllValid()) {
		for (idx_t i = 0; i < count; i++) {
			PacMinMaxUpdateOne<State, IS_MAX>(
			    state, hashes[hash_data.sel->get_index(i)],
			    static_cast<typename State::ValueType>(values[value_data.sel->get_index(i)]), aggr.allocator);
		}
	} else {
		for (idx_t i = 0; i < count; i++) {
			auto h_idx = hash_data.sel->get_index(i);
			auto v_idx = value_data.sel->get_index(i);
			if (!hash_data.validity.RowIsValid(h_idx) || !value_data.validity.RowIsValid(v_idx)) {
#ifdef PAC_MINMAX_UNSAFENULL
				state.seen_null = true;
				return;
#else
				continue;                        // safe mode: ignore NULLs
#endif
			}
			PacMinMaxUpdateOne<State, IS_MAX>(state, hashes[h_idx],
			                                  static_cast<typename State::ValueType>(values[v_idx]), aggr.allocator);
		}
	}
}

// DuckDB method for processing one vector for GROUP BY aggregation (so there are many states))
template <class State, bool IS_MAX, class INPUT_TYPE>
static void PacMinMaxScatterUpdate(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
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
#ifdef PAC_MINMAX_UNSAFENULL
		if (state->seen_null) {
			continue;
		}
		if (!hash_data.validity.RowIsValid(h_idx) || !value_data.validity.RowIsValid(v_idx)) {
			state->seen_null = true;
			continue;
		}
#else
		if (!hash_data.validity.RowIsValid(h_idx) || !value_data.validity.RowIsValid(v_idx)) {
			continue; // safe mode: ignore NULLs
		}
#endif
		PacMinMaxUpdateOne<State, IS_MAX>(*state, hashes[h_idx], static_cast<typename State::ValueType>(values[v_idx]),
		                                  aggr.allocator);
	}
}

// combine two Min/Max aggregate states (used in parallel query processing and partitioned aggregation)
template <class State, bool IS_MAX>
AUTOVECTORIZE static void PacMinMaxCombine(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	auto src_state = FlatVector::GetData<State *>(src);
	auto dst_state = FlatVector::GetData<State *>(dst);

	for (idx_t i = 0; i < count; i++) {
#ifdef PAC_MINMAX_UNSAFENULL
		if (src_state[i]->seen_null) {
			dst_state[i]->seen_null = true;
		}
		if (dst_state[i]->seen_null) {
			continue;
		}
#endif
#ifdef PAC_MINMAX_NONBANKED
		// States are already initialized by PacMinMaxInitialize
		auto *s = src_state[i];
		auto *d = dst_state[i];
		for (int j = 0; j < 64; j++) {
			d->extremes[j] = PAC_BETTER(d->extremes[j], s->extremes[j]);
		}
		d->global_bound = ComputeGlobalBound<typename State::ValueType, typename State::ValueType, IS_MAX>(d->extremes);
#else
		dst_state[i]->CombineWith(*src_state[i], aggr.allocator);
#endif
	}
}

// Finalize computes the final result from the aggregate state
template <class State, class RESULT_TYPE>
static void PacMinMaxFinalize(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	auto state = FlatVector::GetData<State *>(states);
	auto data = FlatVector::GetData<RESULT_TYPE>(result);
	auto &result_mask = FlatVector::Validity(result);

	uint64_t seed = input.bind_data ? input.bind_data->Cast<PacBindData>().seed : std::random_device {}();
	std::mt19937_64 gen(seed);
	double mi = input.bind_data ? input.bind_data->Cast<PacBindData>().mi : 128.0;

	for (idx_t i = 0; i < count; i++) {
#ifdef PAC_MINMAX_UNSAFENULL
		if (state[i]->seen_null) {
			result_mask.SetInvalid(offset + i);
			continue;
		}
#endif
		double buf[64];
		state[i]->GetTotalsAsDouble(buf);
		for (int j = 0; j < 64; j++) {
			buf[j] /= 2.0;
		}
		data[offset + i] = FromDouble<RESULT_TYPE>(PacNoisySampleFrom64Counters(buf, mi, gen) + buf[41]);
	}
}

// Unified State size and initialize (StateSize() is  a member of each state class)
template <class State>
static idx_t PacMinMaxStateSize(const AggregateFunction &) {
	return State::StateSize();
}

template <class State>
static void PacMinMaxInitialize(const AggregateFunction &, data_ptr_t p) {
	memset(p, 0, State::StateSize());
#ifdef PAC_MINMAX_NONBANKED
	reinterpret_cast<State *>(p)->Initialize();
#endif
}

// ============================================================================
//  Update/ScatterUpdate wrappers that instantiate the templates
// ============================================================================

// Integer updates (MAXLEVEL: 4=int64, 5=hugeint)
#define INT_UPDATE_WRAPPER(NAME, SIGNED, MAXLEVEL, INTYPE)                                                             \
	template <bool IS_MAX>                                                                                             \
	void NAME(Vector inputs[], AggregateInputData &aggr, idx_t n, data_ptr_t state_p, idx_t count) {                   \
		PacMinMaxUpdate<PacMinMaxIntState<SIGNED, IS_MAX, MAXLEVEL>, IS_MAX, INTYPE>(inputs, aggr, n, state_p, count); \
	}
// Use type-appropriate MAXLEVEL: each type cascades up to its own width only
INT_UPDATE_WRAPPER(PacMinMaxUpdateInt8, true, 1, int8_t)
INT_UPDATE_WRAPPER(PacMinMaxUpdateInt16, true, 2, int16_t)
INT_UPDATE_WRAPPER(PacMinMaxUpdateInt32, true, 3, int32_t)
INT_UPDATE_WRAPPER(PacMinMaxUpdateInt64, true, 4, int64_t)
INT_UPDATE_WRAPPER(PacMinMaxUpdateUInt8, false, 1, uint8_t)
INT_UPDATE_WRAPPER(PacMinMaxUpdateUInt16, false, 2, uint16_t)
INT_UPDATE_WRAPPER(PacMinMaxUpdateUInt32, false, 3, uint32_t)
INT_UPDATE_WRAPPER(PacMinMaxUpdateUInt64, false, 4, uint64_t)
INT_UPDATE_WRAPPER(PacMinMaxUpdateHugeInt, true, 5, hugeint_t)
INT_UPDATE_WRAPPER(PacMinMaxUpdateUHugeInt, false, 5, uhugeint_t)
#undef INT_UPDATE_WRAPPER

// Float/Double updates (MAXLEVEL=2: float+double cascading)
template <bool IS_MAX>
void PacMinMaxUpdateFloat(Vector in[], AggregateInputData &a, idx_t n, data_ptr_t s, idx_t c) {
	PacMinMaxUpdate<PacMinMaxFloatState<IS_MAX, 2>, IS_MAX, float>(in, a, n, s, c);
}
template <bool IS_MAX>
void PacMinMaxUpdateDoubleW(Vector in[], AggregateInputData &a, idx_t n, data_ptr_t s, idx_t c) {
	PacMinMaxUpdate<PacMinMaxFloatState<IS_MAX, 2>, IS_MAX, double>(in, a, n, s, c);
}

// Integer scatter updates (MAXLEVEL: 4=int64, 5=hugeint)
#define INT_SCATTER_WRAPPER(NAME, SIGNED, MAXLEVEL, INTYPE)                                                            \
	template <bool IS_MAX>                                                                                             \
	void NAME(Vector input[], AggregateInputData &aggr, idx_t n, Vector &states, idx_t count) {                        \
		PacMinMaxScatterUpdate<PacMinMaxIntState<SIGNED, IS_MAX, MAXLEVEL>, IS_MAX, INTYPE>(input, aggr, n, states,    \
		                                                                                    count);                    \
	}
// Use type-appropriate MAXLEVEL: each type cascades up to its own width only
INT_SCATTER_WRAPPER(PacMinMaxScatterUpdateInt8, true, 1, int8_t)
INT_SCATTER_WRAPPER(PacMinMaxScatterUpdateInt16, true, 2, int16_t)
INT_SCATTER_WRAPPER(PacMinMaxScatterUpdateInt32, true, 3, int32_t)
INT_SCATTER_WRAPPER(PacMinMaxScatterUpdateInt64, true, 4, int64_t)
INT_SCATTER_WRAPPER(PacMinMaxScatterUpdateUInt8, false, 1, uint8_t)
INT_SCATTER_WRAPPER(PacMinMaxScatterUpdateUInt16, false, 2, uint16_t)
INT_SCATTER_WRAPPER(PacMinMaxScatterUpdateUInt32, false, 3, uint32_t)
INT_SCATTER_WRAPPER(PacMinMaxScatterUpdateUInt64, false, 4, uint64_t)
INT_SCATTER_WRAPPER(PacMinMaxScatterUpdateHugeInt, true, 5, hugeint_t)
INT_SCATTER_WRAPPER(PacMinMaxScatterUpdateUHugeInt, false, 5, uhugeint_t)
#undef INT_SCATTER_WRAPPER

// Float/Double scatter updates
template <bool IS_MAX>
void PacMinMaxScatterUpdateFloat(Vector in[], AggregateInputData &a, idx_t n, Vector &s, idx_t c) {
	PacMinMaxScatterUpdate<PacMinMaxFloatState<IS_MAX, 2>, IS_MAX, float>(in, a, n, s, c);
}
template <bool IS_MAX>
void PacMinMaxScatterUpdateDoubleW(Vector in[], AggregateInputData &a, idx_t n, Vector &s, idx_t c) {
	PacMinMaxScatterUpdate<PacMinMaxFloatState<IS_MAX, 2>, IS_MAX, double>(in, a, n, s, c);
}

// Combine wrappers - one per (SIGNED, MAXLEVEL) combination
template <bool IS_MAX>
void PacMinMaxCombineInt8Signed(Vector &src, Vector &dst, AggregateInputData &a, idx_t c) {
	PacMinMaxCombine<PacMinMaxIntState<true, IS_MAX, 1>, IS_MAX>(src, dst, a, c);
}
template <bool IS_MAX>
void PacMinMaxCombineInt16Signed(Vector &src, Vector &dst, AggregateInputData &a, idx_t c) {
	PacMinMaxCombine<PacMinMaxIntState<true, IS_MAX, 2>, IS_MAX>(src, dst, a, c);
}
template <bool IS_MAX>
void PacMinMaxCombineInt32Signed(Vector &src, Vector &dst, AggregateInputData &a, idx_t c) {
	PacMinMaxCombine<PacMinMaxIntState<true, IS_MAX, 3>, IS_MAX>(src, dst, a, c);
}
template <bool IS_MAX>
void PacMinMaxCombineInt64Signed(Vector &src, Vector &dst, AggregateInputData &a, idx_t c) {
	PacMinMaxCombine<PacMinMaxIntState<true, IS_MAX, 4>, IS_MAX>(src, dst, a, c);
}
template <bool IS_MAX>
void PacMinMaxCombineInt8Unsigned(Vector &src, Vector &dst, AggregateInputData &a, idx_t c) {
	PacMinMaxCombine<PacMinMaxIntState<false, IS_MAX, 1>, IS_MAX>(src, dst, a, c);
}
template <bool IS_MAX>
void PacMinMaxCombineInt16Unsigned(Vector &src, Vector &dst, AggregateInputData &a, idx_t c) {
	PacMinMaxCombine<PacMinMaxIntState<false, IS_MAX, 2>, IS_MAX>(src, dst, a, c);
}
template <bool IS_MAX>
void PacMinMaxCombineInt32Unsigned(Vector &src, Vector &dst, AggregateInputData &a, idx_t c) {
	PacMinMaxCombine<PacMinMaxIntState<false, IS_MAX, 3>, IS_MAX>(src, dst, a, c);
}
template <bool IS_MAX>
void PacMinMaxCombineInt64Unsigned(Vector &src, Vector &dst, AggregateInputData &a, idx_t c) {
	PacMinMaxCombine<PacMinMaxIntState<false, IS_MAX, 4>, IS_MAX>(src, dst, a, c);
}
template <bool IS_MAX>
void PacMinMaxCombineDoubleWrapper(Vector &src, Vector &dst, AggregateInputData &a, idx_t c) {
	PacMinMaxCombine<PacMinMaxFloatState<IS_MAX, 2>, IS_MAX>(src, dst, a, c);
}
template <bool IS_MAX>
void PacMinMaxCombineHugeIntSigned(Vector &src, Vector &dst, AggregateInputData &a, idx_t c) {
	PacMinMaxCombine<PacMinMaxIntState<true, IS_MAX, 5>, IS_MAX>(src, dst, a, c);
}
template <bool IS_MAX>
void PacMinMaxCombineHugeIntUnsigned(Vector &src, Vector &dst, AggregateInputData &a, idx_t c) {
	PacMinMaxCombine<PacMinMaxIntState<false, IS_MAX, 5>, IS_MAX>(src, dst, a, c);
}

// Bind function: decides from the LogicalType which implementation function to instantiate
template <bool IS_MAX>
static unique_ptr<FunctionData> PacMinMaxBind(ClientContext &ctx, AggregateFunction &function,
                                              vector<unique_ptr<Expression>> &args) {
	// Get the value type (arg 1, arg 0 is hash)
	auto &value_type = args[1]->return_type;
	auto physical_type = value_type.InternalType();

	// Set return type to match input type
	function.return_type = value_type;
	function.arguments[1] = value_type;

	// Select implementation based on physical type
	// NOTE: function.update = scatter update (Vector &states)
	//       function.simple_update = simple update (data_ptr_t state_p)
	// Each type uses MAXLEVEL matching its width: int8=1, int16=2, int32=3, int64=4, hugeint=5
	switch (physical_type) {
	case PhysicalType::INT8:
		function.state_size = PacMinMaxStateSize<PacMinMaxIntState<true, IS_MAX, 1>>;
		function.initialize = PacMinMaxInitialize<PacMinMaxIntState<true, IS_MAX, 1>>;
		function.update = PacMinMaxScatterUpdateInt8<IS_MAX>;
		function.simple_update = PacMinMaxUpdateInt8<IS_MAX>;
		function.combine = PacMinMaxCombineInt8Signed<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<true, IS_MAX, 1>, int8_t>;
		break;

	case PhysicalType::INT16:
		function.state_size = PacMinMaxStateSize<PacMinMaxIntState<true, IS_MAX, 2>>;
		function.initialize = PacMinMaxInitialize<PacMinMaxIntState<true, IS_MAX, 2>>;
		function.update = PacMinMaxScatterUpdateInt16<IS_MAX>;
		function.simple_update = PacMinMaxUpdateInt16<IS_MAX>;
		function.combine = PacMinMaxCombineInt16Signed<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<true, IS_MAX, 2>, int16_t>;
		break;

	case PhysicalType::INT32:
		function.state_size = PacMinMaxStateSize<PacMinMaxIntState<true, IS_MAX, 3>>;
		function.initialize = PacMinMaxInitialize<PacMinMaxIntState<true, IS_MAX, 3>>;
		function.update = PacMinMaxScatterUpdateInt32<IS_MAX>;
		function.simple_update = PacMinMaxUpdateInt32<IS_MAX>;
		function.combine = PacMinMaxCombineInt32Signed<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<true, IS_MAX, 3>, int32_t>;
		break;

	case PhysicalType::INT64:
		function.state_size = PacMinMaxStateSize<PacMinMaxIntState<true, IS_MAX, 4>>;
		function.initialize = PacMinMaxInitialize<PacMinMaxIntState<true, IS_MAX, 4>>;
		function.update = PacMinMaxScatterUpdateInt64<IS_MAX>;
		function.simple_update = PacMinMaxUpdateInt64<IS_MAX>;
		function.combine = PacMinMaxCombineInt64Signed<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<true, IS_MAX, 4>, int64_t>;
		break;

	case PhysicalType::UINT8:
		function.state_size = PacMinMaxStateSize<PacMinMaxIntState<false, IS_MAX, 1>>;
		function.initialize = PacMinMaxInitialize<PacMinMaxIntState<false, IS_MAX, 1>>;
		function.update = PacMinMaxScatterUpdateUInt8<IS_MAX>;
		function.simple_update = PacMinMaxUpdateUInt8<IS_MAX>;
		function.combine = PacMinMaxCombineInt8Unsigned<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<false, IS_MAX, 1>, uint8_t>;
		break;

	case PhysicalType::UINT16:
		function.state_size = PacMinMaxStateSize<PacMinMaxIntState<false, IS_MAX, 2>>;
		function.initialize = PacMinMaxInitialize<PacMinMaxIntState<false, IS_MAX, 2>>;
		function.update = PacMinMaxScatterUpdateUInt16<IS_MAX>;
		function.simple_update = PacMinMaxUpdateUInt16<IS_MAX>;
		function.combine = PacMinMaxCombineInt16Unsigned<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<false, IS_MAX, 2>, uint16_t>;
		break;

	case PhysicalType::UINT32:
		function.state_size = PacMinMaxStateSize<PacMinMaxIntState<false, IS_MAX, 3>>;
		function.initialize = PacMinMaxInitialize<PacMinMaxIntState<false, IS_MAX, 3>>;
		function.update = PacMinMaxScatterUpdateUInt32<IS_MAX>;
		function.simple_update = PacMinMaxUpdateUInt32<IS_MAX>;
		function.combine = PacMinMaxCombineInt32Unsigned<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<false, IS_MAX, 3>, uint32_t>;
		break;

	case PhysicalType::UINT64:
		function.state_size = PacMinMaxStateSize<PacMinMaxIntState<false, IS_MAX, 4>>;
		function.initialize = PacMinMaxInitialize<PacMinMaxIntState<false, IS_MAX, 4>>;
		function.update = PacMinMaxScatterUpdateUInt64<IS_MAX>;
		function.simple_update = PacMinMaxUpdateUInt64<IS_MAX>;
		function.combine = PacMinMaxCombineInt64Unsigned<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<false, IS_MAX, 4>, uint64_t>;
		break;

	case PhysicalType::FLOAT:
		function.state_size = PacMinMaxStateSize<PacMinMaxFloatState<IS_MAX, 2>>;
		function.initialize = PacMinMaxInitialize<PacMinMaxFloatState<IS_MAX, 2>>;
		function.update = PacMinMaxScatterUpdateFloat<IS_MAX>;
		function.simple_update = PacMinMaxUpdateFloat<IS_MAX>;
		function.combine = PacMinMaxCombineDoubleWrapper<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxFloatState<IS_MAX, 2>, float>;
		break;

	case PhysicalType::DOUBLE:
		function.state_size = PacMinMaxStateSize<PacMinMaxFloatState<IS_MAX, 2>>;
		function.initialize = PacMinMaxInitialize<PacMinMaxFloatState<IS_MAX, 2>>;
		function.update = PacMinMaxScatterUpdateDoubleW<IS_MAX>;
		function.simple_update = PacMinMaxUpdateDoubleW<IS_MAX>;
		function.combine = PacMinMaxCombineDoubleWrapper<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxFloatState<IS_MAX, 2>, double>;
		break;

	case PhysicalType::INT128:
		function.state_size = PacMinMaxStateSize<PacMinMaxIntState<true, IS_MAX, 5>>;
		function.initialize = PacMinMaxInitialize<PacMinMaxIntState<true, IS_MAX, 5>>;
		function.update = PacMinMaxScatterUpdateHugeInt<IS_MAX>;
		function.simple_update = PacMinMaxUpdateHugeInt<IS_MAX>;
		function.combine = PacMinMaxCombineHugeIntSigned<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<true, IS_MAX, 5>, hugeint_t>;
		break;

	case PhysicalType::UINT128:
		function.state_size = PacMinMaxStateSize<PacMinMaxIntState<false, IS_MAX, 5>>;
		function.initialize = PacMinMaxInitialize<PacMinMaxIntState<false, IS_MAX, 5>>;
		function.update = PacMinMaxScatterUpdateUHugeInt<IS_MAX>;
		function.simple_update = PacMinMaxUpdateUHugeInt<IS_MAX>;
		function.combine = PacMinMaxCombineHugeIntUnsigned<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<false, IS_MAX, 5>, uhugeint_t>;
		break;

	default:
		throw NotImplementedException("pac_%s not implemented for type %s", IS_MAX ? "max" : "min",
		                              value_type.ToString());
	}

	// Get mi and seed
	double mi = 128.0;
	if (args.size() >= 3) {
		if (!args[2]->IsFoldable()) {
			throw InvalidInputException("pac_%s: mi parameter must be a constant", IS_MAX ? "max" : "min");
		}
		auto mi_val = ExpressionExecutor::EvaluateScalar(ctx, *args[2]);
		mi = mi_val.GetValue<double>();
		if (mi < 0.0) {
			throw InvalidInputException("pac_%s: mi must be >= 0", IS_MAX ? "max" : "min");
		}
	}

	uint64_t seed = std::random_device {}();
	Value pac_seed_val;
	if (ctx.TryGetCurrentSetting("pac_seed", pac_seed_val) && !pac_seed_val.IsNull()) {
		seed = uint64_t(pac_seed_val.GetValue<int64_t>());
	}

	return make_uniq<PacBindData>(mi, seed);
}

// Registration
void RegisterPacMinFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet fcn_set("pac_min");

	// Register with ANY type - bind callback handles specialization
	// 2-param version: pac_min(hash, value)
	fcn_set.AddFunction(AggregateFunction("pac_min", {LogicalType::UBIGINT, LogicalType::ANY}, LogicalType::ANY,
	                                      nullptr, nullptr, nullptr, nullptr, nullptr,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, PacMinMaxBind<false>));

	// 3-param version: pac_min(hash, value, mi)
	fcn_set.AddFunction(AggregateFunction("pac_min", {LogicalType::UBIGINT, LogicalType::ANY, LogicalType::DOUBLE},
	                                      LogicalType::ANY, nullptr, nullptr, nullptr, nullptr, nullptr,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, PacMinMaxBind<false>));

	loader.RegisterFunction(fcn_set);
}

void RegisterPacMaxFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet fcn_set("pac_max");

	// Register with ANY type - bind callback handles specialization
	// 2-param version: pac_max(hash, value)
	fcn_set.AddFunction(AggregateFunction("pac_max", {LogicalType::UBIGINT, LogicalType::ANY}, LogicalType::ANY,
	                                      nullptr, nullptr, nullptr, nullptr, nullptr,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, PacMinMaxBind<true>));

	// 3-param version: pac_max(hash, value, mi)
	fcn_set.AddFunction(AggregateFunction("pac_max", {LogicalType::UBIGINT, LogicalType::ANY, LogicalType::DOUBLE},
	                                      LogicalType::ANY, nullptr, nullptr, nullptr, nullptr, nullptr,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, PacMinMaxBind<true>));

	loader.RegisterFunction(fcn_set);
}

// Explicit template instantiations
#define INST_U(NAME)                                                                                                   \
	template void NAME<false>(Vector *, AggregateInputData &, idx_t, data_ptr_t, idx_t);                               \
	template void NAME<true>(Vector *, AggregateInputData &, idx_t, data_ptr_t, idx_t)
#define INST_S(NAME)                                                                                                   \
	template void NAME<false>(Vector *, AggregateInputData &, idx_t, Vector &, idx_t);                                 \
	template void NAME<true>(Vector *, AggregateInputData &, idx_t, Vector &, idx_t)
#define INST_C(NAME)                                                                                                   \
	template void NAME<false>(Vector &, Vector &, AggregateInputData &, idx_t);                                        \
	template void NAME<true>(Vector &, Vector &, AggregateInputData &, idx_t)

INST_U(PacMinMaxUpdateInt8);
INST_U(PacMinMaxUpdateInt16);
INST_U(PacMinMaxUpdateInt32);
INST_U(PacMinMaxUpdateInt64);
INST_U(PacMinMaxUpdateUInt8);
INST_U(PacMinMaxUpdateUInt16);
INST_U(PacMinMaxUpdateUInt32);
INST_U(PacMinMaxUpdateUInt64);
INST_U(PacMinMaxUpdateFloat);
INST_U(PacMinMaxUpdateDoubleW);
INST_U(PacMinMaxUpdateHugeInt);
INST_U(PacMinMaxUpdateUHugeInt);

INST_S(PacMinMaxScatterUpdateInt8);
INST_S(PacMinMaxScatterUpdateInt16);
INST_S(PacMinMaxScatterUpdateInt32);
INST_S(PacMinMaxScatterUpdateInt64);
INST_S(PacMinMaxScatterUpdateUInt8);
INST_S(PacMinMaxScatterUpdateUInt16);
INST_S(PacMinMaxScatterUpdateUInt32);
INST_S(PacMinMaxScatterUpdateUInt64);
INST_S(PacMinMaxScatterUpdateFloat);
INST_S(PacMinMaxScatterUpdateDoubleW);
INST_S(PacMinMaxScatterUpdateHugeInt);
INST_S(PacMinMaxScatterUpdateUHugeInt);

INST_C(PacMinMaxCombineInt8Signed);
INST_C(PacMinMaxCombineInt16Signed);
INST_C(PacMinMaxCombineInt32Signed);
INST_C(PacMinMaxCombineInt64Signed);
INST_C(PacMinMaxCombineInt8Unsigned);
INST_C(PacMinMaxCombineInt16Unsigned);
INST_C(PacMinMaxCombineInt32Unsigned);
INST_C(PacMinMaxCombineInt64Unsigned);
INST_C(PacMinMaxCombineDoubleWrapper);
INST_C(PacMinMaxCombineHugeIntSigned);
INST_C(PacMinMaxCombineHugeIntUnsigned);
} // namespace duckdb
