#include "include/pac_count.hpp"

namespace duckdb {

static unique_ptr<FunctionData> // Bind function for pac_count with optional mi parameter (must be constant)
PacCountBind(ClientContext &ctx, AggregateFunction &func, vector<unique_ptr<Expression>> &args) {
	double mi = 128.0; // default

	// Handle mi parameter based on function signature:
	// - {UBIGINT, DOUBLE}: mi is args[1]
	// - {UBIGINT, ANY, DOUBLE}: mi is args[2]
	// Check if the last declared argument type is DOUBLE (the mi parameter)
	if (args.size() >= 2 && func.arguments.back().id() == LogicalTypeId::DOUBLE) {
		idx_t mi_arg_idx = args.size() - 1;
		if (!args[mi_arg_idx]->IsFoldable()) {
			throw InvalidInputException("pac_count: mi parameter must be a constant");
		}
		auto mi_val = ExpressionExecutor::EvaluateScalar(ctx, *args[mi_arg_idx]);
		mi = mi_val.GetValue<double>();
		if (mi <= 0.0) {
			throw InvalidInputException("pac_count: mi must be > 0");
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

AUTOVECTORIZE static inline void // worker function that probabilistically counts one hash into 64 subtotals
PacCountUpdateHash(uint64_t key_hash, PacCountState &state, ArenaAllocator &allocator) {
	// Ensure allocator is set and level8 is allocated
	if (!state.allocator) {
		state.allocator = &allocator;
	}
	if (!state.probabilistic_totals8) {
		state.EnsureLevelAllocated(state.probabilistic_totals8, 8);
	}

	// Add to SWAR-packed uint8 counters
	for (int j = 0; j < 8; j++) {
		state.probabilistic_totals8[j] += (key_hash >> j) & PAC_COUNT_MASK;
	}
	state.Flush8(1, false); // increment exact_total8 by 1, flush if needed
}

AUTOVECTORIZE static inline void // worker function that probabilistically counts one tuple into 64 subtotals
PacCountUpdateOne(const UnifiedVectorFormat &idata, idx_t i, const uint64_t *input_data, PacCountState &state,
                  ArenaAllocator &allocator) {
	auto idx = idata.sel->get_index(i);
	if (idata.validity.RowIsValid(idx)) {
		PacCountUpdateHash(input_data[idx], state, allocator);
	}
}

void PacCountUpdate(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state_ptr, idx_t count) {
	D_ASSERT(input_count == 1 || input_count == 2); // optional mi param (unused here) can make it 2
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
void PacCountColumnUpdate(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state_ptr,
                          idx_t count) {
	D_ASSERT(input_count >= 2); // hash + column (+ optional mi)
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

// Helper to combine one level of counters (moves source pointer ownership to dest if dest is null)
template <typename T>
static inline void CombineLevel(uint64_t *&src_buf, uint64_t *&dst_buf, uint32_t &src_exact, uint32_t &dst_exact,
                                int count) {
	if (!src_buf) {
		return; // nothing to combine
	}
	if (!dst_buf) {
		// Move ownership from src to dst
		dst_buf = src_buf;
		dst_exact = src_exact;
		src_buf = nullptr;
		src_exact = 0;
		return;
	}
	// Both have data - add element by element
	T *src = reinterpret_cast<T *>(src_buf);
	T *dst = reinterpret_cast<T *>(dst_buf);
	for (int i = 0; i < 64; i++) {
		dst[i] += src[i];
	}
	dst_exact += src_exact;
}

// Overload for uint64_t level (no exact_total needed)
static inline void CombineLevel64(uint64_t *&src_buf, uint64_t *&dst_buf) {
	if (!src_buf) {
		return;
	}
	if (!dst_buf) {
		dst_buf = src_buf;
		src_buf = nullptr;
		return;
	}
	for (int i = 0; i < 64; i++) {
		dst_buf[i] += src_buf[i];
	}
}

AUTOVECTORIZE void PacCountCombine(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	auto src_state = FlatVector::GetData<PacCountState *>(src);
	auto dst_state = FlatVector::GetData<PacCountState *>(dst);
	for (idx_t i = 0; i < count; i++) {
		auto *s = src_state[i];
		auto *d = dst_state[i];

		// Flush source to ensure all data is cascaded up
		s->Flush();

		// Ensure dst has allocator pointer
		if (!d->allocator) {
			d->allocator = &aggr.allocator;
		}

		// Combine at each level
		CombineLevel<uint8_t>(s->probabilistic_totals8, d->probabilistic_totals8, s->exact_total8, d->exact_total8, 8);
		CombineLevel<uint16_t>(s->probabilistic_totals16, d->probabilistic_totals16, s->exact_total16, d->exact_total16,
		                       16);
		uint32_t src32 = static_cast<uint32_t>(s->exact_total32);
		uint32_t dst32 = static_cast<uint32_t>(d->exact_total32);
		CombineLevel<uint32_t>(s->probabilistic_totals32, d->probabilistic_totals32, src32, dst32, 32);
		d->exact_total32 = dst32;
		CombineLevel64(s->probabilistic_totals64, d->probabilistic_totals64);
	}
}

void PacCountFinalize(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	auto state = FlatVector::GetData<PacCountState *>(states);
	auto data = FlatVector::GetData<int64_t>(result);
	// Use deterministic seed from bind_data if present
	uint64_t seed = input.bind_data ? input.bind_data->Cast<PacBindData>().seed : std::random_device {}();
	std::mt19937_64 gen(seed);
	double mi = input.bind_data->Cast<PacBindData>().mi;

	for (idx_t i = 0; i < count; i++) {
		state[i]->Flush(); // force flush any remaining small totals into the big totals
		double buf[64];
		state[i]->GetTotalsAsDouble(buf);
		data[offset + i] = // when choosing any one of the totals we go for #42 (but one counts from 0 ofc)
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
