#include "include/pac_min_max.hpp"

namespace duckdb {

// ============================================================================
// Update functions - process one value into the 64 extremes
// ============================================================================

// Integer update: cascading version
template <bool SIGNED, bool IS_MAX>
AUTOVECTORIZE static inline void PacMinMaxUpdateOne(PacMinMaxIntState<SIGNED, IS_MAX> &state, uint64_t key_hash,
                                                    typename PacMinMaxIntState<SIGNED, IS_MAX>::T64 value,
                                                    ArenaAllocator &allocator) {
	using State = PacMinMaxIntState<SIGNED, IS_MAX>;
	using T64 = typename State::T64;

#ifdef PAC_MINMAX_NONCASCADING
	if (!state.initialized) {
		state.Initialize();
	}
	// Early skip: if value can't improve any extreme, skip
	if (!State::IsBetter(value, state.global_bound)) {
		return;
	}
	state.global_bound = UpdateExtremesAtLevel<T64, T64, IS_MAX>(state.extremes, key_hash, value);
#else
	if (!state.allocator) {
		state.allocator = &allocator;
#ifdef PAC_MINMAX_NONLAZY
		state.InitializeAllLevels(allocator);
#endif
	}

	// Early skip optimization (only if initialized)
	if (state.current_level > 0 && !State::IsBetter(value, state.global_bound)) {
		return;
	}

	// Ensure we have the right level for this value
	if (state.current_level == 0) {
		state.EnsureLevel8();
	}

	// Upgrade level if value doesn't fit
	if (!State::FitsIn8(value) && state.current_level == 8) {
		state.UpgradeTo16();
	}
	if (!State::FitsIn16(value) && state.current_level == 16) {
		state.UpgradeTo32();
	}
	if (!State::FitsIn32(value) && state.current_level == 32) {
		state.UpgradeTo64();
	}

	// Update at current level using templated helper
	switch (state.current_level) {
	case 8:
		state.global_bound = UpdateExtremesAtLevel<typename State::T8, T64, IS_MAX>(
		    state.extremes8, key_hash, static_cast<typename State::T8>(value));
		break;
	case 16:
		state.global_bound = UpdateExtremesAtLevel<typename State::T16, T64, IS_MAX>(
		    state.extremes16, key_hash, static_cast<typename State::T16>(value));
		break;
	case 32:
		state.global_bound = UpdateExtremesAtLevel<typename State::T32, T64, IS_MAX>(
		    state.extremes32, key_hash, static_cast<typename State::T32>(value));
		break;
	case 64:
	default:
		state.global_bound = UpdateExtremesAtLevel<T64, T64, IS_MAX>(state.extremes64, key_hash, value);
		break;
	}
#endif
}

// Double update (SIGNED parameter ignored for doubles)
template <bool SIGNED, bool IS_MAX>
AUTOVECTORIZE static inline void PacMinMaxUpdateOne(PacMinMaxDoubleState<IS_MAX> &state, uint64_t key_hash,
                                                    double value, ArenaAllocator &allocator) {
	using State = PacMinMaxDoubleState<IS_MAX>;

#ifdef PAC_MINMAX_NONCASCADING
	if (!state.initialized) {
		state.Initialize();
	}

	// Early skip
	if (!State::IsBetter(value, state.global_bound)) {
		return;
	}

	state.global_bound = UpdateExtremesAtLevel<double, double, IS_MAX>(state.extremes, key_hash, value);
#else
	if (!state.allocator) {
		state.allocator = &allocator;
#ifdef PAC_MINMAX_NONLAZY
		state.InitializeAllLevels(allocator);
#endif
	}

	// Early skip optimization (only if initialized)
	if (state.current_level > 0 && !State::IsBetter(value, state.global_bound)) {
		return;
	}

	// Ensure we have the float level
	if (state.current_level == 0) {
		state.EnsureLevelFloat();
	}

	// Upgrade to double if value doesn't fit in float range
	if (!State::FitsInFloat(value) && state.current_level == 32) {
		state.UpgradeTo64();
	}

	// Update at current level using templated helper
	switch (state.current_level) {
	case 32:
		state.global_bound =
		    UpdateExtremesAtLevel<float, double, IS_MAX>(state.extremesF, key_hash, static_cast<float>(value));
		break;
	case 64:
	default:
		state.global_bound = UpdateExtremesAtLevel<double, double, IS_MAX>(state.extremesD, key_hash, value);
		break;
	}
#endif
}

// ============================================================================
// Batch Update functions (simple_update - single state pointer)
// ============================================================================

template <class State, bool SIGNED, bool IS_MAX, class VALUE_TYPE, class INPUT_TYPE>
static void PacMinMaxUpdate(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_p, idx_t count) {
	auto &state = *reinterpret_cast<State *>(state_p);
	if (state.seen_null) {
		return;
	}

	UnifiedVectorFormat hash_data, value_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, value_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);

	if (hash_data.validity.AllValid() && value_data.validity.AllValid()) {
		for (idx_t i = 0; i < count; i++) {
			auto h_idx = hash_data.sel->get_index(i);
			auto v_idx = value_data.sel->get_index(i);
			PacMinMaxUpdateOne<SIGNED, IS_MAX>(state, hashes[h_idx], static_cast<VALUE_TYPE>(values[v_idx]),
			                                   aggr.allocator);
		}
	} else {
		for (idx_t i = 0; i < count; i++) {
			auto h_idx = hash_data.sel->get_index(i);
			auto v_idx = value_data.sel->get_index(i);
			if (!hash_data.validity.RowIsValid(h_idx) || !value_data.validity.RowIsValid(v_idx)) {
				state.seen_null = true;
				return;
			}
			PacMinMaxUpdateOne<SIGNED, IS_MAX>(state, hashes[h_idx], static_cast<VALUE_TYPE>(values[v_idx]),
			                                   aggr.allocator);
		}
	}
}

// ============================================================================
// Scatter Update functions (update - vector of state pointers)
// ============================================================================

template <class State, bool SIGNED, bool IS_MAX, class VALUE_TYPE, class INPUT_TYPE>
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
		if (state->seen_null) {
			continue;
		} else if (!hash_data.validity.RowIsValid(h_idx) || !value_data.validity.RowIsValid(v_idx)) {
			state->seen_null = true;
		} else {
			PacMinMaxUpdateOne<SIGNED, IS_MAX>(*state, hashes[h_idx], static_cast<VALUE_TYPE>(values[v_idx]),
			                                   aggr.allocator);
		}
	}
}

// ============================================================================
// Combine functions
// ============================================================================

template <bool SIGNED, bool IS_MAX>
AUTOVECTORIZE static void PacMinMaxCombineInt(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	using State = PacMinMaxIntState<SIGNED, IS_MAX>;
	auto src_state = FlatVector::GetData<State *>(src);
	auto dst_state = FlatVector::GetData<State *>(dst);

	for (idx_t i = 0; i < count; i++) {
		if (src_state[i]->seen_null) {
			dst_state[i]->seen_null = true;
		}
		if (dst_state[i]->seen_null) {
			continue;
		}

		auto *s = src_state[i];
		auto *d = dst_state[i];

#ifdef PAC_MINMAX_NONCASCADING
		if (!s->initialized) {
			continue;
		}
		if (!d->initialized) {
			d->Initialize();
		}
		typename State::T64 new_bound = d->extremes[0];
		for (int j = 0; j < 64; j++) {
			d->extremes[j] = State::Better(d->extremes[j], s->extremes[j]);
			new_bound = State::Worse(new_bound, d->extremes[j]);
		}
		d->global_bound = new_bound;
#else
		if (s->current_level == 0) {
			continue; // src not initialized
		}
		if (!d->allocator) {
			d->allocator = &aggr.allocator;
		}

		// Ensure dst is at least at src's level
		while (d->current_level < s->current_level) {
			if (d->current_level == 0) {
				d->EnsureLevel8();
			} else if (d->current_level == 8) {
				d->UpgradeTo16();
			} else if (d->current_level == 16) {
				d->UpgradeTo32();
			} else if (d->current_level == 32) {
				d->UpgradeTo64();
			}
		}

		// Combine at the highest level
		typename State::T64 new_bound;
		switch (d->current_level) {
		case 8:
			new_bound = d->extremes8[0];
			for (int j = 0; j < 64; j++) {
				d->extremes8[j] = State::IsBetter(s->extremes8[j], d->extremes8[j]) ? s->extremes8[j] : d->extremes8[j];
				new_bound = State::Worse(new_bound, static_cast<typename State::T64>(d->extremes8[j]));
			}
			break;
		case 16:
			new_bound = d->extremes16[0];
			for (int j = 0; j < 64; j++) {
				auto src_val =
				    (s->current_level >= 16) ? s->extremes16[j] : static_cast<typename State::T16>(s->extremes8[j]);
				d->extremes16[j] = State::IsBetter(src_val, d->extremes16[j]) ? src_val : d->extremes16[j];
				new_bound = State::Worse(new_bound, static_cast<typename State::T64>(d->extremes16[j]));
			}
			break;
		case 32:
			new_bound = d->extremes32[0];
			for (int j = 0; j < 64; j++) {
				typename State::T32 src_val;
				if (s->current_level >= 32) {
					src_val = s->extremes32[j];
				} else if (s->current_level >= 16) {
					src_val = static_cast<typename State::T32>(s->extremes16[j]);
				} else {
					src_val = static_cast<typename State::T32>(s->extremes8[j]);
				}
				d->extremes32[j] = State::IsBetter(src_val, d->extremes32[j]) ? src_val : d->extremes32[j];
				new_bound = State::Worse(new_bound, static_cast<typename State::T64>(d->extremes32[j]));
			}
			break;
		case 64:
		default:
			new_bound = d->extremes64[0];
			for (int j = 0; j < 64; j++) {
				typename State::T64 src_val;
				if (s->current_level >= 64) {
					src_val = s->extremes64[j];
				} else if (s->current_level >= 32) {
					src_val = static_cast<typename State::T64>(s->extremes32[j]);
				} else if (s->current_level >= 16) {
					src_val = static_cast<typename State::T64>(s->extremes16[j]);
				} else {
					src_val = static_cast<typename State::T64>(s->extremes8[j]);
				}
				d->extremes64[j] = State::Better(d->extremes64[j], src_val);
				new_bound = State::Worse(new_bound, d->extremes64[j]);
			}
			break;
		}
		d->global_bound = new_bound;
#endif
	}
}

template <bool IS_MAX>
AUTOVECTORIZE static void PacMinMaxCombineDouble(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	using State = PacMinMaxDoubleState<IS_MAX>;
	auto src_state = FlatVector::GetData<State *>(src);
	auto dst_state = FlatVector::GetData<State *>(dst);

	for (idx_t i = 0; i < count; i++) {
		if (src_state[i]->seen_null) {
			dst_state[i]->seen_null = true;
		}
		if (dst_state[i]->seen_null) {
			continue;
		}

		auto *s = src_state[i];
		auto *d = dst_state[i];

#ifdef PAC_MINMAX_NONCASCADING
		if (!s->initialized) {
			continue;
		}
		if (!d->initialized) {
			d->Initialize();
		}

		double new_bound = d->extremes[0];
		for (int j = 0; j < 64; j++) {
			d->extremes[j] = State::Better(d->extremes[j], s->extremes[j]);
			new_bound = State::Worse(new_bound, d->extremes[j]);
		}
		d->global_bound = new_bound;
#else
		if (s->current_level == 0) {
			continue; // src not initialized
		}
		if (!d->allocator) {
			d->allocator = &aggr.allocator;
		}

		// Ensure dst is at least at src's level
		while (d->current_level < s->current_level) {
			if (d->current_level == 0) {
				d->EnsureLevelFloat();
			} else if (d->current_level == 32) {
				d->UpgradeTo64();
			}
		}

		// Combine at the highest level
		double new_bound;
		switch (d->current_level) {
		case 32:
			new_bound = d->extremesF[0];
			for (int j = 0; j < 64; j++) {
				d->extremesF[j] = State::IsBetter(s->extremesF[j], d->extremesF[j]) ? s->extremesF[j] : d->extremesF[j];
				new_bound = State::Worse(new_bound, static_cast<double>(d->extremesF[j]));
			}
			break;
		case 64:
		default:
			new_bound = d->extremesD[0];
			for (int j = 0; j < 64; j++) {
				double src_val;
				if (s->current_level >= 64) {
					src_val = s->extremesD[j];
				} else {
					src_val = static_cast<double>(s->extremesF[j]);
				}
				d->extremesD[j] = State::Better(d->extremesD[j], src_val);
				new_bound = State::Worse(new_bound, d->extremesD[j]);
			}
			break;
		}
		d->global_bound = new_bound;
#endif
	}
}

// ============================================================================
// Finalize function
// ============================================================================

template <class State, class RESULT_TYPE>
static void PacMinMaxFinalize(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	auto state = FlatVector::GetData<State *>(states);
	auto data = FlatVector::GetData<RESULT_TYPE>(result);
	auto &result_mask = FlatVector::Validity(result);

	uint64_t seed = input.bind_data ? input.bind_data->Cast<PacBindData>().seed : std::random_device {}();
	std::mt19937_64 gen(seed);
	double mi = input.bind_data ? input.bind_data->Cast<PacBindData>().mi : 128.0;

	for (idx_t i = 0; i < count; i++) {
		if (state[i]->seen_null) {
			result_mask.SetInvalid(offset + i);
		} else {
			double buf[64];
			state[i]->Flush();
			state[i]->GetTotalsAsDouble(buf);
			data[offset + i] = FromDouble<RESULT_TYPE>(PacNoisySampleFrom64Counters(buf, mi, gen) + buf[41]);
		}
	}
}

// ============================================================================
// State size and initialize
// ============================================================================

template <bool SIGNED, bool IS_MAX>
static idx_t PacMinMaxIntStateSize(const AggregateFunction &) {
	return sizeof(PacMinMaxIntState<SIGNED, IS_MAX>);
}

template <bool SIGNED, bool IS_MAX>
static void PacMinMaxIntInitialize(const AggregateFunction &, data_ptr_t state_p) {
	memset(state_p, 0, sizeof(PacMinMaxIntState<SIGNED, IS_MAX>));
}

template <bool IS_MAX>
static idx_t PacMinMaxDoubleStateSize(const AggregateFunction &) {
	return sizeof(PacMinMaxDoubleState<IS_MAX>);
}

template <bool IS_MAX>
static void PacMinMaxDoubleInitialize(const AggregateFunction &, data_ptr_t state_p) {
	memset(state_p, 0, sizeof(PacMinMaxDoubleState<IS_MAX>));
}

// ============================================================================
// Instantiated Update/ScatterUpdate wrappers
// ============================================================================

// Signed integer updates
template <bool IS_MAX>
void PacMinMaxUpdateInt8(Vector inputs[], AggregateInputData &aggr, idx_t n, data_ptr_t state_p, idx_t count) {
	PacMinMaxUpdate<PacMinMaxIntState<true, IS_MAX>, true, IS_MAX, int64_t, int8_t>(inputs, aggr, n, state_p, count);
}
template <bool IS_MAX>
void PacMinMaxUpdateInt16(Vector inputs[], AggregateInputData &aggr, idx_t n, data_ptr_t state_p, idx_t count) {
	PacMinMaxUpdate<PacMinMaxIntState<true, IS_MAX>, true, IS_MAX, int64_t, int16_t>(inputs, aggr, n, state_p, count);
}
template <bool IS_MAX>
void PacMinMaxUpdateInt32(Vector inputs[], AggregateInputData &aggr, idx_t n, data_ptr_t state_p, idx_t count) {
	PacMinMaxUpdate<PacMinMaxIntState<true, IS_MAX>, true, IS_MAX, int64_t, int32_t>(inputs, aggr, n, state_p, count);
}
template <bool IS_MAX>
void PacMinMaxUpdateInt64(Vector inputs[], AggregateInputData &aggr, idx_t n, data_ptr_t state_p, idx_t count) {
	PacMinMaxUpdate<PacMinMaxIntState<true, IS_MAX>, true, IS_MAX, int64_t, int64_t>(inputs, aggr, n, state_p, count);
}

// Unsigned integer updates
template <bool IS_MAX>
void PacMinMaxUpdateUInt8(Vector inputs[], AggregateInputData &aggr, idx_t n, data_ptr_t state_p, idx_t count) {
	PacMinMaxUpdate<PacMinMaxIntState<false, IS_MAX>, false, IS_MAX, uint64_t, uint8_t>(inputs, aggr, n, state_p,
	                                                                                    count);
}
template <bool IS_MAX>
void PacMinMaxUpdateUInt16(Vector inputs[], AggregateInputData &aggr, idx_t n, data_ptr_t state_p, idx_t count) {
	PacMinMaxUpdate<PacMinMaxIntState<false, IS_MAX>, false, IS_MAX, uint64_t, uint16_t>(inputs, aggr, n, state_p,
	                                                                                     count);
}
template <bool IS_MAX>
void PacMinMaxUpdateUInt32(Vector inputs[], AggregateInputData &aggr, idx_t n, data_ptr_t state_p, idx_t count) {
	PacMinMaxUpdate<PacMinMaxIntState<false, IS_MAX>, false, IS_MAX, uint64_t, uint32_t>(inputs, aggr, n, state_p,
	                                                                                     count);
}
template <bool IS_MAX>
void PacMinMaxUpdateUInt64(Vector inputs[], AggregateInputData &aggr, idx_t n, data_ptr_t state_p, idx_t count) {
	PacMinMaxUpdate<PacMinMaxIntState<false, IS_MAX>, false, IS_MAX, uint64_t, uint64_t>(inputs, aggr, n, state_p,
	                                                                                     count);
}

// Float/Double updates
template <bool IS_MAX>
void PacMinMaxUpdateFloat(Vector inputs[], AggregateInputData &aggr, idx_t n, data_ptr_t state_p, idx_t count) {
	PacMinMaxUpdate<PacMinMaxDoubleState<IS_MAX>, true, IS_MAX, double, float>(inputs, aggr, n, state_p, count);
}
template <bool IS_MAX>
void PacMinMaxUpdateDouble(Vector inputs[], AggregateInputData &aggr, idx_t n, data_ptr_t state_p, idx_t count) {
	PacMinMaxUpdate<PacMinMaxDoubleState<IS_MAX>, true, IS_MAX, double, double>(inputs, aggr, n, state_p, count);
}

// Signed integer scatter updates
template <bool IS_MAX>
void PacMinMaxScatterUpdateInt8(Vector inputs[], AggregateInputData &aggr, idx_t n, Vector &states, idx_t count) {
	PacMinMaxScatterUpdate<PacMinMaxIntState<true, IS_MAX>, true, IS_MAX, int64_t, int8_t>(inputs, aggr, n, states,
	                                                                                       count);
}
template <bool IS_MAX>
void PacMinMaxScatterUpdateInt16(Vector inputs[], AggregateInputData &aggr, idx_t n, Vector &states, idx_t count) {
	PacMinMaxScatterUpdate<PacMinMaxIntState<true, IS_MAX>, true, IS_MAX, int64_t, int16_t>(inputs, aggr, n, states,
	                                                                                        count);
}
template <bool IS_MAX>
void PacMinMaxScatterUpdateInt32(Vector inputs[], AggregateInputData &aggr, idx_t n, Vector &states, idx_t count) {
	PacMinMaxScatterUpdate<PacMinMaxIntState<true, IS_MAX>, true, IS_MAX, int64_t, int32_t>(inputs, aggr, n, states,
	                                                                                        count);
}
template <bool IS_MAX>
void PacMinMaxScatterUpdateInt64(Vector inputs[], AggregateInputData &aggr, idx_t n, Vector &states, idx_t count) {
	PacMinMaxScatterUpdate<PacMinMaxIntState<true, IS_MAX>, true, IS_MAX, int64_t, int64_t>(inputs, aggr, n, states,
	                                                                                        count);
}

// Unsigned integer scatter updates
template <bool IS_MAX>
void PacMinMaxScatterUpdateUInt8(Vector inputs[], AggregateInputData &aggr, idx_t n, Vector &states, idx_t count) {
	PacMinMaxScatterUpdate<PacMinMaxIntState<false, IS_MAX>, false, IS_MAX, uint64_t, uint8_t>(inputs, aggr, n, states,
	                                                                                           count);
}
template <bool IS_MAX>
void PacMinMaxScatterUpdateUInt16(Vector inputs[], AggregateInputData &aggr, idx_t n, Vector &states, idx_t count) {
	PacMinMaxScatterUpdate<PacMinMaxIntState<false, IS_MAX>, false, IS_MAX, uint64_t, uint16_t>(inputs, aggr, n, states,
	                                                                                            count);
}
template <bool IS_MAX>
void PacMinMaxScatterUpdateUInt32(Vector inputs[], AggregateInputData &aggr, idx_t n, Vector &states, idx_t count) {
	PacMinMaxScatterUpdate<PacMinMaxIntState<false, IS_MAX>, false, IS_MAX, uint64_t, uint32_t>(inputs, aggr, n, states,
	                                                                                            count);
}
template <bool IS_MAX>
void PacMinMaxScatterUpdateUInt64(Vector inputs[], AggregateInputData &aggr, idx_t n, Vector &states, idx_t count) {
	PacMinMaxScatterUpdate<PacMinMaxIntState<false, IS_MAX>, false, IS_MAX, uint64_t, uint64_t>(inputs, aggr, n, states,
	                                                                                            count);
}

// Float/Double scatter updates
template <bool IS_MAX>
void PacMinMaxScatterUpdateFloat(Vector inputs[], AggregateInputData &aggr, idx_t n, Vector &states, idx_t count) {
	PacMinMaxScatterUpdate<PacMinMaxDoubleState<IS_MAX>, true, IS_MAX, double, float>(inputs, aggr, n, states, count);
}
template <bool IS_MAX>
void PacMinMaxScatterUpdateDouble(Vector inputs[], AggregateInputData &aggr, idx_t n, Vector &states, idx_t count) {
	PacMinMaxScatterUpdate<PacMinMaxDoubleState<IS_MAX>, true, IS_MAX, double, double>(inputs, aggr, n, states, count);
}

// Combine wrappers
template <bool IS_MAX>
void PacMinMaxCombineIntSigned(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	PacMinMaxCombineInt<true, IS_MAX>(src, dst, aggr, count);
}
template <bool IS_MAX>
void PacMinMaxCombineIntUnsigned(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	PacMinMaxCombineInt<false, IS_MAX>(src, dst, aggr, count);
}
template <bool IS_MAX>
void PacMinMaxCombineDoubleWrapper(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	PacMinMaxCombineDouble<IS_MAX>(src, dst, aggr, count);
}

// ============================================================================
// Bind function
// ============================================================================

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
	switch (physical_type) {
	case PhysicalType::INT8:
		function.state_size = PacMinMaxIntStateSize<true, IS_MAX>;
		function.initialize = PacMinMaxIntInitialize<true, IS_MAX>;
		function.update = PacMinMaxScatterUpdateInt8<IS_MAX>;
		function.simple_update = PacMinMaxUpdateInt8<IS_MAX>;
		function.combine = PacMinMaxCombineIntSigned<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<true, IS_MAX>, int8_t>;
		break;

	case PhysicalType::INT16:
		function.state_size = PacMinMaxIntStateSize<true, IS_MAX>;
		function.initialize = PacMinMaxIntInitialize<true, IS_MAX>;
		function.update = PacMinMaxScatterUpdateInt16<IS_MAX>;
		function.simple_update = PacMinMaxUpdateInt16<IS_MAX>;
		function.combine = PacMinMaxCombineIntSigned<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<true, IS_MAX>, int16_t>;
		break;

	case PhysicalType::INT32:
		function.state_size = PacMinMaxIntStateSize<true, IS_MAX>;
		function.initialize = PacMinMaxIntInitialize<true, IS_MAX>;
		function.update = PacMinMaxScatterUpdateInt32<IS_MAX>;
		function.simple_update = PacMinMaxUpdateInt32<IS_MAX>;
		function.combine = PacMinMaxCombineIntSigned<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<true, IS_MAX>, int32_t>;
		break;

	case PhysicalType::INT64:
		function.state_size = PacMinMaxIntStateSize<true, IS_MAX>;
		function.initialize = PacMinMaxIntInitialize<true, IS_MAX>;
		function.update = PacMinMaxScatterUpdateInt64<IS_MAX>;
		function.simple_update = PacMinMaxUpdateInt64<IS_MAX>;
		function.combine = PacMinMaxCombineIntSigned<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<true, IS_MAX>, int64_t>;
		break;

	case PhysicalType::UINT8:
		function.state_size = PacMinMaxIntStateSize<false, IS_MAX>;
		function.initialize = PacMinMaxIntInitialize<false, IS_MAX>;
		function.update = PacMinMaxScatterUpdateUInt8<IS_MAX>;
		function.simple_update = PacMinMaxUpdateUInt8<IS_MAX>;
		function.combine = PacMinMaxCombineIntUnsigned<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<false, IS_MAX>, uint8_t>;
		break;

	case PhysicalType::UINT16:
		function.state_size = PacMinMaxIntStateSize<false, IS_MAX>;
		function.initialize = PacMinMaxIntInitialize<false, IS_MAX>;
		function.update = PacMinMaxScatterUpdateUInt16<IS_MAX>;
		function.simple_update = PacMinMaxUpdateUInt16<IS_MAX>;
		function.combine = PacMinMaxCombineIntUnsigned<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<false, IS_MAX>, uint16_t>;
		break;

	case PhysicalType::UINT32:
		function.state_size = PacMinMaxIntStateSize<false, IS_MAX>;
		function.initialize = PacMinMaxIntInitialize<false, IS_MAX>;
		function.update = PacMinMaxScatterUpdateUInt32<IS_MAX>;
		function.simple_update = PacMinMaxUpdateUInt32<IS_MAX>;
		function.combine = PacMinMaxCombineIntUnsigned<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<false, IS_MAX>, uint32_t>;
		break;

	case PhysicalType::UINT64:
		function.state_size = PacMinMaxIntStateSize<false, IS_MAX>;
		function.initialize = PacMinMaxIntInitialize<false, IS_MAX>;
		function.update = PacMinMaxScatterUpdateUInt64<IS_MAX>;
		function.simple_update = PacMinMaxUpdateUInt64<IS_MAX>;
		function.combine = PacMinMaxCombineIntUnsigned<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxIntState<false, IS_MAX>, uint64_t>;
		break;

	case PhysicalType::FLOAT:
		function.state_size = PacMinMaxDoubleStateSize<IS_MAX>;
		function.initialize = PacMinMaxDoubleInitialize<IS_MAX>;
		function.update = PacMinMaxScatterUpdateFloat<IS_MAX>;
		function.simple_update = PacMinMaxUpdateFloat<IS_MAX>;
		function.combine = PacMinMaxCombineDoubleWrapper<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxDoubleState<IS_MAX>, float>;
		break;

	case PhysicalType::DOUBLE:
		function.state_size = PacMinMaxDoubleStateSize<IS_MAX>;
		function.initialize = PacMinMaxDoubleInitialize<IS_MAX>;
		function.update = PacMinMaxScatterUpdateDouble<IS_MAX>;
		function.simple_update = PacMinMaxUpdateDouble<IS_MAX>;
		function.combine = PacMinMaxCombineDoubleWrapper<IS_MAX>;
		function.finalize = PacMinMaxFinalize<PacMinMaxDoubleState<IS_MAX>, double>;
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
		if (mi <= 0.0) {
			throw InvalidInputException("pac_%s: mi must be > 0", IS_MAX ? "max" : "min");
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

// Explicit template instantiations for the wrapper functions
template void PacMinMaxUpdateInt8<false>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateInt8<true>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateInt16<false>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateInt16<true>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateInt32<false>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateInt32<true>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateInt64<false>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateInt64<true>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateUInt8<false>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateUInt8<true>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateUInt16<false>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateUInt16<true>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateUInt32<false>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateUInt32<true>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateUInt64<false>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateUInt64<true>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateFloat<false>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateFloat<true>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateDouble<false>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);
template void PacMinMaxUpdateDouble<true>(Vector[], AggregateInputData &, idx_t, data_ptr_t, idx_t);

template void PacMinMaxScatterUpdateInt8<false>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateInt8<true>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateInt16<false>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateInt16<true>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateInt32<false>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateInt32<true>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateInt64<false>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateInt64<true>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateUInt8<false>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateUInt8<true>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateUInt16<false>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateUInt16<true>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateUInt32<false>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateUInt32<true>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateUInt64<false>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateUInt64<true>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateFloat<false>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateFloat<true>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateDouble<false>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);
template void PacMinMaxScatterUpdateDouble<true>(Vector[], AggregateInputData &, idx_t, Vector &, idx_t);

template void PacMinMaxCombineIntSigned<false>(Vector &, Vector &, AggregateInputData &, idx_t);
template void PacMinMaxCombineIntSigned<true>(Vector &, Vector &, AggregateInputData &, idx_t);
template void PacMinMaxCombineIntUnsigned<false>(Vector &, Vector &, AggregateInputData &, idx_t);
template void PacMinMaxCombineIntUnsigned<true>(Vector &, Vector &, AggregateInputData &, idx_t);
template void PacMinMaxCombineDoubleWrapper<false>(Vector &, Vector &, AggregateInputData &, idx_t);
template void PacMinMaxCombineDoubleWrapper<true>(Vector &, Vector &, AggregateInputData &, idx_t);

} // namespace duckdb
