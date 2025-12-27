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
PacCountUpdateHash(uint64_t key_hash, PacCountState &state) {
	for (int j = 0; j < 8; j++) {
		state.probabilistic_subtotals[j] += (key_hash >> j) & PAC_COUNT_MASK;
	}
	state.Flush();
}

AUTOVECTORIZE static inline void // worker function that probabilistically counts one tuple into 64 subtotals
PacCountUpdateOne(const UnifiedVectorFormat &idata, idx_t i, const uint64_t *input_data, PacCountState &state) {
	auto idx = idata.sel->get_index(i);
	if (idata.validity.RowIsValid(idx)) {
		PacCountUpdateHash(input_data[idx], state);
	}
}

void PacCountUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, data_ptr_t state_ptr, idx_t count) {
	D_ASSERT(input_count == 1 || input_count == 2); // optional mi param (unused here) can make it 2
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
void PacCountColumnUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, data_ptr_t state_ptr, idx_t count) {
	D_ASSERT(input_count >= 2); // hash + column (+ optional mi)
	auto &state = *reinterpret_cast<PacCountState *>(state_ptr);

	UnifiedVectorFormat hash_data, col_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, col_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);

	for (idx_t i = 0; i < count; i++) {
		auto hash_idx = hash_data.sel->get_index(i);
		auto col_idx = col_data.sel->get_index(i);
		if (hash_data.validity.RowIsValid(hash_idx) && col_data.validity.RowIsValid(col_idx)) {
			PacCountUpdateHash(hashes[hash_idx], state);
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
		auto hash_idx = hash_data.sel->get_index(i);
		auto col_idx = col_data.sel->get_index(i);
		if (hash_data.validity.RowIsValid(hash_idx) && col_data.validity.RowIsValid(col_idx)) {
			PacCountUpdateHash(hashes[hash_idx], *state_p[sdata.sel->get_index(i)]);
		}
	}
}

AUTOVECTORIZE void PacCountCombine(Vector &src, Vector &dst, AggregateInputData &, idx_t count) {
	auto src_state = FlatVector::GetData<PacCountState *>(src);
	auto dst_state = FlatVector::GetData<PacCountState *>(dst);
	for (idx_t i = 0; i < count; i++) {
		src_state[i]->Flush(true); // force flush source before reading from it
		for (int j = 0; j < 64; j++) {
			dst_state[i]->probabilistic_totals[j] += src_state[i]->probabilistic_totals[j];
		}
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
		state[i]->Flush(true); // force flush any remaining small totals into the big totals
		double buf[64];
		ToDoubleArray(state[i]->probabilistic_totals, buf); // Convert uint64_t totals64 to double array
		data[offset + i] = // when choosing any one of the totals we go for #42 (but one counts from 0 ofc)
		    static_cast<int64_t>(PacNoisySampleFrom64Counters(buf, mi, gen)) + state[i]->probabilistic_totals[41];
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
