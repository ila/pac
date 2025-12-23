#include "include/pac_count.hpp"

namespace duckdb {

static unique_ptr<FunctionData> // Bind function for pac_count with optional mi parameter (must be constant)
PacCountBind(ClientContext &ctx, AggregateFunction &, vector<unique_ptr<Expression>> &args) {
	double mi = 128.0; // default
	if (args.size() >= 3) {
		if (!args[2]->IsFoldable()) {
			throw InvalidInputException("pac_count: mi parameter must be a constant");
		}
		auto mi_val = ExpressionExecutor::EvaluateScalar(ctx, *args[1]);
		if (mi_val.GetValue<double>() <= 0.0) {
			throw InvalidInputException("pac_count: mi must be > 0");
		}
	}
	return make_uniq<PacBindData>(mi);
}
static idx_t PacCountStateSize(const AggregateFunction &) {
	return sizeof(PacCountState);
}

static void PacCountInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	memset(state_ptr, 0, sizeof(PacCountState));
}

AUTOVECTORIZE static inline void // worker function that probabilistically counts one tuple into 64 subtotals
PacCountUpdateOne(const UnifiedVectorFormat &idata, idx_t i, const uint64_t *input_data, PacCountState &state) {
	auto idx = idata.sel->get_index(i);
	if (idata.validity.RowIsValid(idx)) {
		uint64_t key_hash = input_data[idx];
		for (int j = 0; j < 8; j++) {
			state.probabilistic_subtotals[j] += (key_hash >> j) & PAC_COUNT_MASK;
		}
		state.Flush();
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

AUTOVECTORIZE void PacCountCombine(Vector &src, Vector &dst, AggregateInputData &, idx_t count) {
	auto src_state = FlatVector::GetData<PacCountState *>(src);
	auto dst_state = FlatVector::GetData<PacCountState *>(dst);
	for (idx_t i = 0; i < count; i++) {
		src_state[i]->Flush(); // flush source before reading from it
		for (int j = 0; j < 64; j++) {
			dst_state[i]->probabilistic_totals[j] += src_state[i]->probabilistic_totals[j];
		}
	}
}

void PacCountFinalize(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	auto state = FlatVector::GetData<PacCountState *>(states);
	auto data = FlatVector::GetData<int64_t>(result);
	thread_local std::mt19937_64 gen(std::random_device {}());
	double mi = input.bind_data->Cast<PacBindData>().mi;

	for (idx_t i = 0; i < count; i++) {
		state[i]->Flush(); // flush any remaining small totals into the big totals
		double buf[64];
		ToDoubleArray(state[i]->probabilistic_totals, buf); // Convert uint64_t totals64 to double array
		data[offset + i] = // when choosing any one of the totals we go for #42 (but one counts from 0 ofc)
		    static_cast<int64_t>(PacNoisySampleFrom64Counters(buf, mi, gen)) + state[i]->probabilistic_totals[41];
	}
}

void RegisterPacCountFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet fcn_set("pac_count");

	fcn_set.AddFunction(AggregateFunction("pac_count", {LogicalType::UBIGINT}, LogicalType::BIGINT, PacCountStateSize,
	                                      PacCountInitialize, PacCountScatterUpdate, PacCountCombine, PacCountFinalize,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, PacCountUpdate, PacCountBind));

	fcn_set.AddFunction(AggregateFunction("pac_count", {LogicalType::UBIGINT, LogicalType::DOUBLE},
	                                      LogicalType::BIGINT, PacCountStateSize, PacCountInitialize,
	                                      PacCountScatterUpdate, PacCountCombine, PacCountFinalize,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, PacCountUpdate, PacCountBind));

	loader.RegisterFunction(fcn_set);
}

} // namespace duckdb
