#include "include/pac_count.hpp"

namespace duckdb {

static unique_ptr<FunctionData> // Bind function for pac_count with optional mi parameter (must be constant)
PacCountBind(ClientContext &ctx, AggregateFunction &func, vector<unique_ptr<Expression>> &args) {
	double mi = 128.0; // default

	// Handle mi parameter - check each argument position for a foldable numeric constant
	// This handles both explicit DOUBLE signature and ANY signature where user passes a constant
	for (idx_t i = 1; i < args.size(); i++) {
		auto &arg = args[i];
		// Check if this is a foldable constant that could be mi
		if (arg->IsFoldable() && (arg->return_type.IsNumeric() || arg->return_type.id() == LogicalTypeId::UNKNOWN)) {
			auto val = ExpressionExecutor::EvaluateScalar(ctx, *arg);
			if (val.type().IsNumeric()) {
				mi = val.GetValue<double>();
				if (mi < 0.0) {
					throw InvalidInputException("pac_count: mi must be >= 0");
				}
				break; // Found mi, stop looking
			}
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
static idx_t PacCountStateSize(const AggregateFunction &) {
	return sizeof(PacCountState);
}

static void PacCountInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	memset(state_ptr, 0, sizeof(PacCountState));
}

#ifdef PAC_COUNT_NONCASCADING
AUTOVECTORIZE static inline void // worker function that probabilistically counts one hash into 64 counters
PacCountUpdateHash(uint64_t key_hash, PacCountState &state) {
	// Direct update to uint64_t[64] counters
	for (int j = 0; j < 64; j++) {
		state.probabilistic_total[j] += (key_hash >> j) & 1ULL;
	}
}
#else
AUTOVECTORIZE static inline void // worker function that probabilistically counts one hash into 64 subtotal
PacCountUpdateHash(uint64_t key_hash, PacCountState &state, ArenaAllocator &allocator) {
#ifdef PAC_COUNT_NONLAZY
	if (!state.probabilistic_total16) { // Use as proxy for "not yet initialized"
		state.InitializeAllLevels(allocator);
	}
#endif
	// Level 8 is inline (always available), no allocation needed
	// the SIMD-friendly performance-critical loop: adding to bytecounters using SWAR
	// prototyped here: https://godbolt.org/z/8r6x8s17P
	for (int j = 0; j < 8; j++) {
		state.probabilistic_total8[j] += (key_hash >> j) & PAC_COUNT_MASK; // Add to SWAR-packed uint8 counters
	}
	state.Flush8(allocator, 1, false); // increment exact_total8 by 1, flush if needed
}
#endif

#ifdef PAC_COUNT_NONCASCADING
AUTOVECTORIZE static inline void // worker function that probabilistically counts one tuple into 64 counters
PacCountUpdateOne(const UnifiedVectorFormat &idata, idx_t i, const uint64_t *input_data, PacCountState &state) {
	auto idx = idata.sel->get_index(i);
	if (idata.validity.RowIsValid(idx)) {
		PacCountUpdateHash(input_data[idx], state);
	}
}

void PacCountUpdate(Vector inputs[], AggregateInputData &, idx_t input_total, data_ptr_t state_ptr, idx_t count) {
	D_ASSERT(input_total == 1 || input_total == 2); // optional mi param (unused here) can make it 2
	auto &state = *reinterpret_cast<PacCountState *>(state_ptr);

	UnifiedVectorFormat idata;
	inputs[0].ToUnifiedFormat(count, idata);
	auto input_data = UnifiedVectorFormat::GetData<uint64_t>(idata);
	for (idx_t i = 0; i < count; i++) {
		PacCountUpdateOne(idata, i, input_data, state);
	}
}

void PacCountScatterUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &states, idx_t count) {
	UnifiedVectorFormat idata;
	inputs[0].ToUnifiedFormat(count, idata);
	auto input_data = UnifiedVectorFormat::GetData<uint64_t>(idata);
	UnifiedVectorFormat sdata;
	states.ToUnifiedFormat(count, sdata);

	auto state_p = UnifiedVectorFormat::GetData<PacCountState *>(sdata);
	for (idx_t i = 0; i < count; i++) {
		PacCountUpdateOne(idata, i, input_data, *state_p[sdata.sel->get_index(i)]);
	}
}

// Column-based update: counts non-null values in the column
void PacCountColumnUpdate(Vector inputs[], AggregateInputData &, idx_t input_total, data_ptr_t state_ptr, idx_t count) {
	D_ASSERT(input_total >= 2); // hash + column (+ optional mi)
	auto &state = *reinterpret_cast<PacCountState *>(state_ptr);

	UnifiedVectorFormat hash_data, col_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, col_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);

	for (idx_t i = 0; i < count; i++) {
		auto h_idx = hash_data.sel->get_index(i);
		auto c_idx = col_data.sel->get_index(i);
		if (hash_data.validity.RowIsValid(h_idx) && col_data.validity.RowIsValid(c_idx)) {
			PacCountUpdateHash(hashes[h_idx], state);
		}
	}
}

void PacCountColumnScatterUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &states, idx_t count) {
	UnifiedVectorFormat hash_data, col_data, sdata;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, col_data);
	states.ToUnifiedFormat(count, sdata);

	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto state_p = UnifiedVectorFormat::GetData<PacCountState *>(sdata);

	for (idx_t i = 0; i < count; i++) {
		auto h_idx = hash_data.sel->get_index(i);
		auto c_idx = col_data.sel->get_index(i);
		if (hash_data.validity.RowIsValid(h_idx) && col_data.validity.RowIsValid(c_idx)) {
			PacCountUpdateHash(hashes[h_idx], *state_p[sdata.sel->get_index(i)]);
		}
	}
}
#else
AUTOVECTORIZE static inline void // worker function that probabilistically counts one tuple into 64 subtotal
PacCountUpdateOne(const UnifiedVectorFormat &idata, idx_t i, const uint64_t *input_data, PacCountState &state,
                  ArenaAllocator &allocator) {
	auto idx = idata.sel->get_index(i);
	if (idata.validity.RowIsValid(idx)) {
		PacCountUpdateHash(input_data[idx], state, allocator);
	}
}

void PacCountUpdate(Vector inputs[], AggregateInputData &aggr, idx_t input_total, data_ptr_t state_ptr, idx_t count) {
	D_ASSERT(input_total == 1 || input_total == 2); // optional mi param (unused here) can make it 2
	auto &state = *reinterpret_cast<PacCountState *>(state_ptr);
	UnifiedVectorFormat idata;
	inputs[0].ToUnifiedFormat(count, idata);
	auto input_data = UnifiedVectorFormat::GetData<uint64_t>(idata);
	for (idx_t i = 0; i < count; i++) {
		PacCountUpdateOne(idata, i, input_data, state, aggr.allocator);
	}
}

void PacCountScatterUpdate(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	UnifiedVectorFormat idata;
	inputs[0].ToUnifiedFormat(count, idata);
	auto input_data = UnifiedVectorFormat::GetData<uint64_t>(idata);
	UnifiedVectorFormat sdata;
	states.ToUnifiedFormat(count, sdata);

	auto state_p = UnifiedVectorFormat::GetData<PacCountState *>(sdata);
	for (idx_t i = 0; i < count; i++) {
		PacCountUpdateOne(idata, i, input_data, *state_p[sdata.sel->get_index(i)], aggr.allocator);
	}
}

// Column-based update: counts non-null values in the column
void PacCountColumnUpdate(Vector inputs[], AggregateInputData &aggr, idx_t input_total, data_ptr_t state_ptr,
                          idx_t count) {
	D_ASSERT(input_total >= 2); // hash + column (+ optional mi)
	auto &state = *reinterpret_cast<PacCountState *>(state_ptr);

	UnifiedVectorFormat hash_data, col_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, col_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);

	for (idx_t i = 0; i < count; i++) {
		auto h_idx = hash_data.sel->get_index(i);
		auto c_idx = col_data.sel->get_index(i);
		if (hash_data.validity.RowIsValid(h_idx) && col_data.validity.RowIsValid(c_idx)) {
			PacCountUpdateHash(hashes[h_idx], state, aggr.allocator);
		}
	}
}

void PacCountColumnScatterUpdate(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	UnifiedVectorFormat hash_data, col_data, sdata;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, col_data);
	states.ToUnifiedFormat(count, sdata);

	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto state_p = UnifiedVectorFormat::GetData<PacCountState *>(sdata);

	for (idx_t i = 0; i < count; i++) {
		auto h_idx = hash_data.sel->get_index(i);
		auto c_idx = col_data.sel->get_index(i);
		if (hash_data.validity.RowIsValid(h_idx) && col_data.validity.RowIsValid(c_idx)) {
			PacCountUpdateHash(hashes[h_idx], *state_p[sdata.sel->get_index(i)], aggr.allocator);
		}
	}
}
#endif

#ifdef PAC_COUNT_NONCASCADING
AUTOVECTORIZE void PacCountCombine(Vector &src, Vector &dst, AggregateInputData &, idx_t count) {
	auto src_state = FlatVector::GetData<PacCountState *>(src);
	auto dst_state = FlatVector::GetData<PacCountState *>(dst);
	for (idx_t i = 0; i < count; i++) {
		for (int j = 0; j < 64; j++) {
			dst_state[i]->probabilistic_total[j] += src_state[i]->probabilistic_total[j];
		}
	}
}
#else
// Helper to combine one level of counters (moves source pointer ownership to dest if dest is null)
template <typename T, typename EXACT_T>
static inline void CombineLevel(uint64_t *&src_buf, uint64_t *&dst_buf, EXACT_T &src_exact, EXACT_T &dst_exact) {
	if (src_buf) {
		if (dst_buf) { // Both have data - add element by element
			T *src = reinterpret_cast<T *>(src_buf);
			T *dst = reinterpret_cast<T *>(dst_buf);
			for (int i = 0; i < 64; i++) {
				dst[i] += src[i];
			}
			dst_exact += src_exact;
		} else { // Move ownership from src to dst
			dst_buf = src_buf;
			dst_exact = src_exact;
			src_buf = nullptr;
			src_exact = 0;
		}
	}
}

AUTOVECTORIZE void PacCountCombine(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	auto src_state = FlatVector::GetData<PacCountState *>(src);
	auto dst_state = FlatVector::GetData<PacCountState *>(dst);
	for (idx_t i = 0; i < count; i++) {
		auto *s = src_state[i];
		auto *d = dst_state[i];

		// Flush source to ensure all data is cascaded up
		s->Flush(aggr.allocator);

		// Before combining each level, check if dst would overflow - if so, flush dst first
		// Level 8 is always present (inline)
		if (static_cast<uint16_t>(s->exact_total8) + static_cast<uint16_t>(d->exact_total8) > UINT8_MAX) {
			d->Flush8(aggr.allocator, 0, true);
		}
		if (d->probabilistic_total16 &&
		    static_cast<uint32_t>(s->exact_total16) + static_cast<uint32_t>(d->exact_total16) > UINT16_MAX) {
			d->Flush16(aggr.allocator, 0, true);
		}
		if (d->probabilistic_total32 &&
		    static_cast<uint64_t>(s->exact_total32) + static_cast<uint64_t>(d->exact_total32) > UINT32_MAX) {
			d->Flush32(aggr.allocator, 0, true);
		}
		// Level 64 cannot overflow within uint64_t range

		// Combine at each level - use dummy pointers for inline level 8
		uint64_t *src8 = s->probabilistic_total8, *dst8 = d->probabilistic_total8;
		CombineLevel<uint8_t>(src8, dst8, s->exact_total8, d->exact_total8);
		CombineLevel<uint16_t>(s->probabilistic_total16, d->probabilistic_total16, s->exact_total16, d->exact_total16);
		CombineLevel<uint32_t>(s->probabilistic_total32, d->probabilistic_total32, s->exact_total32, d->exact_total32);
		// Level 64 has no exact_total tracking - use dummy
		uint64_t dummy = 0;
		CombineLevel<uint64_t>(s->probabilistic_total64, d->probabilistic_total64, dummy, dummy);
	}
}
#endif

void PacCountFinalize(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	auto state = FlatVector::GetData<PacCountState *>(states);
	auto data = FlatVector::GetData<int64_t>(result);
	// Use deterministic seed from bind_data if present
	uint64_t seed = input.bind_data ? input.bind_data->Cast<PacBindData>().seed : std::random_device {}();
	std::mt19937_64 gen(seed);
	double mi = input.bind_data->Cast<PacBindData>().mi;

	for (idx_t i = 0; i < count; i++) {
		state[i]->Flush(input.allocator); // force flush any remaining small total into the big total
		double buf[64];
		state[i]->GetTotalsAsDouble(buf);
		data[offset + i] = // when choosing any one of the total we go for #42 (but one counts from 0 ofc)
		    static_cast<int64_t>(PacNoisySampleFrom64Counters(buf, mi, gen)) + static_cast<int64_t>(buf[41]);
	}
}

void RegisterPacCountFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet fcn_set("pac_count");

	// pac_count(hash) - count all rows
	fcn_set.AddFunction(AggregateFunction("pac_count", {LogicalType::UBIGINT}, LogicalType::BIGINT, PacCountStateSize,
	                                      PacCountInitialize, PacCountScatterUpdate, PacCountCombine, PacCountFinalize,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, PacCountUpdate, PacCountBind));

	// pac_count(hash, mi) - count all rows with custom mi
	fcn_set.AddFunction(AggregateFunction("pac_count", {LogicalType::UBIGINT, LogicalType::DOUBLE}, LogicalType::BIGINT,
	                                      PacCountStateSize, PacCountInitialize, PacCountScatterUpdate, PacCountCombine,
	                                      PacCountFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING, PacCountUpdate,
	                                      PacCountBind));

	// pac_count(hash, column) - count non-null values in column
	fcn_set.AddFunction(AggregateFunction("pac_count", {LogicalType::UBIGINT, LogicalType::ANY}, LogicalType::BIGINT,
	                                      PacCountStateSize, PacCountInitialize, PacCountColumnScatterUpdate,
	                                      PacCountCombine, PacCountFinalize, FunctionNullHandling::SPECIAL_HANDLING,
	                                      PacCountColumnUpdate, PacCountBind));

	// pac_count(hash, column, mi) - count non-null values with custom mi
	fcn_set.AddFunction(AggregateFunction("pac_count", {LogicalType::UBIGINT, LogicalType::ANY, LogicalType::DOUBLE},
	                                      LogicalType::BIGINT, PacCountStateSize, PacCountInitialize,
	                                      PacCountColumnScatterUpdate, PacCountCombine, PacCountFinalize,
	                                      FunctionNullHandling::SPECIAL_HANDLING, PacCountColumnUpdate, PacCountBind));

	loader.RegisterFunction(fcn_set);
}

} // namespace duckdb
