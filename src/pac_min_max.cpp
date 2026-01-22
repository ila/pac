#include "include/pac_min_max.hpp"

namespace duckdb {

// ============================================================================
// State type selection
// ============================================================================
#ifdef PAC_NOBUFFERING
template <typename T, bool IS_MAX>
using MinMaxState = PacMinMaxState<T, IS_MAX>;
#else
template <typename T, bool IS_MAX>
using MinMaxState = PacMinMaxStateWrapper<T, IS_MAX>;
#endif

// ============================================================================
// Update functions
// ============================================================================

// Non-grouped update
template <typename T, bool IS_MAX>
static void PacMinMaxUpdate(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	auto &agg = *reinterpret_cast<MinMaxState<T, IS_MAX> *>(state_p);
	uint64_t query_hash = aggr.bind_data->Cast<PacBindData>().query_hash;

	UnifiedVectorFormat hash_data, value_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<T>(value_data);

	for (idx_t i = 0; i < count; i++) {
		auto h_idx = hash_data.sel->get_index(i);
		auto v_idx = value_data.sel->get_index(i);
		if (hash_data.validity.RowIsValid(h_idx) && value_data.validity.RowIsValid(v_idx)) {
			// min/max do not use buffering for global aggregates (without GROUP BY) because it was  slower
			PacMinMaxUpdateOne<T, IS_MAX>(agg, PacMixHash(hashes[h_idx], query_hash), values[v_idx], aggr.allocator);
		}
	}
}

// Grouped (scatter) update
template <typename T, bool IS_MAX>
static void PacMinMaxScatterUpdate(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	uint64_t query_hash = aggr.bind_data->Cast<PacBindData>().query_hash;
	UnifiedVectorFormat hash_data, value_data, sdata;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	states.ToUnifiedFormat(count, sdata);

	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<T>(value_data);
	auto state_ptrs = UnifiedVectorFormat::GetData<MinMaxState<T, IS_MAX> *>(sdata);

	for (idx_t i = 0; i < count; i++) {
		auto h_idx = hash_data.sel->get_index(i);
		auto v_idx = value_data.sel->get_index(i);
		if (hash_data.validity.RowIsValid(h_idx) && value_data.validity.RowIsValid(v_idx)) {
			auto state = state_ptrs[sdata.sel->get_index(i)];
#ifdef PAC_NOBUFFERING
			PacMinMaxUpdateOne<T, IS_MAX>(*state, PacMixHash(hashes[h_idx], query_hash), values[v_idx], aggr.allocator);
#else
			PacMinMaxBufferOrUpdateOne<T, IS_MAX>(*state, PacMixHash(hashes[h_idx], query_hash), values[v_idx],
			                                      aggr.allocator);
#endif
		}
	}
}

// ============================================================================
// Combine and Finalize
// ============================================================================

template <typename T, bool IS_MAX>
static void PacMinMaxCombine(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	auto src_states = FlatVector::GetData<MinMaxState<T, IS_MAX> *>(src);
	auto dst_states = FlatVector::GetData<MinMaxState<T, IS_MAX> *>(dst);

	for (idx_t i = 0; i < count; i++) {
#ifndef PAC_NOBUFFERING
		// flush the src buffer into dst (not into src: that could cause allocations there, we only want that in dst)
		src_states[i]->FlushBuffer(*dst_states[i], aggr.allocator);
#endif
		auto *ss = src_states[i]->GetState();
		if (ss && ss->initialized) { // propagate the min/max of src into dst
			auto &ds = *dst_states[i]->EnsureState(aggr.allocator);
			ds.CombineWith(*ss);
		}
	}
}

template <typename T, bool IS_MAX>
static void PacMinMaxFinalize(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	auto state_ptrs = FlatVector::GetData<MinMaxState<T, IS_MAX> *>(states);
	auto data = FlatVector::GetData<T>(result);
	auto &result_mask = FlatVector::Validity(result);

	uint64_t seed = input.bind_data ? input.bind_data->Cast<PacBindData>().seed : std::random_device {}();
	std::mt19937_64 gen(seed);
	double mi = input.bind_data ? input.bind_data->Cast<PacBindData>().mi : 128.0;

	for (idx_t i = 0; i < count; i++) {
#ifndef PAC_NOBUFFERING
		// flush the buffer into yourself, possibly allocating the state (array with 64 totals)
		state_ptrs[i]->FlushBuffer(*state_ptrs[i], input.allocator);
#endif
		auto *s = state_ptrs[i]->GetState();
		// Check if we should return NULL based on key_hash
		uint64_t key_hash = s ? s->key_hash : 0;
		if (PacNoiseInNull(key_hash, mi, gen)) {
			result_mask.SetInvalid(offset + i);
			continue;
		}
		double buf[64];
		if (s && s->initialized) {
			s->GetTotalsAsDouble(buf);
		} else {
			memset(buf, 0, sizeof(buf));
		}
		for (int j = 0; j < 64; j++) {
			buf[j] /= 2.0; // if we add noise of the same scale to a min or max, it gets twice too big
		}
		data[offset + i] = FromDouble<T>(PacNoisySampleFrom64Counters(buf, mi, gen, true, ~key_hash));
	}
}

// ============================================================================
// State management
// ============================================================================

template <typename T, bool IS_MAX>
static idx_t PacMinMaxStateSize(const AggregateFunction &) {
	return sizeof(MinMaxState<T, IS_MAX>);
}

template <typename T, bool IS_MAX>
static void PacMinMaxInitialize(const AggregateFunction &, data_ptr_t p) {
	memset(p, 0, sizeof(MinMaxState<T, IS_MAX>));
}

// ============================================================================
// Bind function - selects implementation based on input type
// ============================================================================

template <bool IS_MAX>
static unique_ptr<FunctionData> PacMinMaxBind(ClientContext &ctx, AggregateFunction &function,
                                              vector<unique_ptr<Expression>> &args) {
	auto &value_type = args[1]->return_type;
	auto physical_type = value_type.InternalType();

	function.return_type = value_type;
	function.arguments[1] = value_type;

	// Select implementation based on physical type
#define BIND_TYPE(PHYS_TYPE, CPP_TYPE)                                                                                 \
	case PhysicalType::PHYS_TYPE:                                                                                      \
		function.state_size = PacMinMaxStateSize<CPP_TYPE, IS_MAX>;                                                    \
		function.initialize = PacMinMaxInitialize<CPP_TYPE, IS_MAX>;                                                   \
		function.update = PacMinMaxScatterUpdate<CPP_TYPE, IS_MAX>;                                                    \
		function.simple_update = PacMinMaxUpdate<CPP_TYPE, IS_MAX>;                                                    \
		function.combine = PacMinMaxCombine<CPP_TYPE, IS_MAX>;                                                         \
		function.finalize = PacMinMaxFinalize<CPP_TYPE, IS_MAX>;                                                       \
		break

	switch (physical_type) {
		BIND_TYPE(INT8, int8_t);
		BIND_TYPE(INT16, int16_t);
		BIND_TYPE(INT32, int32_t);
		BIND_TYPE(INT64, int64_t);
		BIND_TYPE(UINT8, uint8_t);
		BIND_TYPE(UINT16, uint16_t);
		BIND_TYPE(UINT32, uint32_t);
		BIND_TYPE(UINT64, uint64_t);
		BIND_TYPE(FLOAT, float);
		BIND_TYPE(DOUBLE, double);
		BIND_TYPE(INT128, hugeint_t);
		BIND_TYPE(UINT128, uhugeint_t);
	default:
		throw NotImplementedException("pac_%s not implemented for type %s", IS_MAX ? "max" : "min",
		                              value_type.ToString());
	}
#undef BIND_TYPE

	// Get mi parameter
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

// ============================================================================
// Registration
// ============================================================================

void RegisterPacMinFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet fcn_set("pac_min");

	fcn_set.AddFunction(AggregateFunction("pac_min", {LogicalType::UBIGINT, LogicalType::ANY}, LogicalType::ANY,
	                                      nullptr, nullptr, nullptr, nullptr, nullptr,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, PacMinMaxBind<false>));

	fcn_set.AddFunction(AggregateFunction("pac_min", {LogicalType::UBIGINT, LogicalType::ANY, LogicalType::DOUBLE},
	                                      LogicalType::ANY, nullptr, nullptr, nullptr, nullptr, nullptr,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, PacMinMaxBind<false>));

	loader.RegisterFunction(fcn_set);
}

void RegisterPacMaxFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet fcn_set("pac_max");

	fcn_set.AddFunction(AggregateFunction("pac_max", {LogicalType::UBIGINT, LogicalType::ANY}, LogicalType::ANY,
	                                      nullptr, nullptr, nullptr, nullptr, nullptr,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, PacMinMaxBind<true>));

	fcn_set.AddFunction(AggregateFunction("pac_max", {LogicalType::UBIGINT, LogicalType::ANY, LogicalType::DOUBLE},
	                                      LogicalType::ANY, nullptr, nullptr, nullptr, nullptr, nullptr,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, PacMinMaxBind<true>));

	loader.RegisterFunction(fcn_set);
}

// Explicit template instantiations
#define INST_ALL(T)                                                                                                    \
	template void PacMinMaxUpdate<T, false>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);                 \
	template void PacMinMaxUpdate<T, true>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);                  \
	template void PacMinMaxScatterUpdate<T, false>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);            \
	template void PacMinMaxScatterUpdate<T, true>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);             \
	template void PacMinMaxCombine<T, false>(Vector &, Vector &, AggregateInputData &, idx_t);                         \
	template void PacMinMaxCombine<T, true>(Vector &, Vector &, AggregateInputData &, idx_t)

INST_ALL(int8_t);
INST_ALL(int16_t);
INST_ALL(int32_t);
INST_ALL(int64_t);
INST_ALL(uint8_t);
INST_ALL(uint16_t);
INST_ALL(uint32_t);
INST_ALL(uint64_t);
INST_ALL(float);
INST_ALL(double);
INST_ALL(hugeint_t);
INST_ALL(uhugeint_t);

} // namespace duckdb
