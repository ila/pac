#include "include/pac_count.hpp"
#include "include/pac_aggregate.hpp"

#include "duckdb.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

#include <cstring>

namespace duckdb {

unique_ptr<FunctionData>
PacCountBind(ClientContext &context, AggregateFunction &, vector<unique_ptr<Expression>> &arguments) {
	double mi = 128.0; // default
	if (arguments.size() >= 2) {
		if (!arguments[1]->IsFoldable()) {
			throw InvalidInputException("pac_count: mi parameter must be a constant");
		}
		auto mi_val = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
		if (mi_val.GetValue<double>() <= 0.0) {
			throw InvalidInputException("pac_count: mi must be > 0");
		}
	}
	return make_uniq<PacBindData>(mi);
}

idx_t PacCountStateSize(const AggregateFunction &) {
	return sizeof(PacCountState);
}

void PacCountInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	memset(state_ptr, 0, sizeof(PacCountState));
}

AUTOVECTORIZE static inline void
PacCountUpdateInternal(const UnifiedVectorFormat &idata, idx_t i, const uint64_t *input_data, PacCountState &state) {
	auto idx = idata.sel->get_index(i);
	if (!idata.validity.RowIsValid(idx)) {
		return;
	}
	if (++state.update_count == 0) {
		state.Flush(); // every 255 iterations flush to avoid overflow
		state.update_count = 1;
	}
	uint64_t key_hash = input_data[idx];
	for (int j = 0; j < 8; j++) {
		state.totals8[j] += (key_hash >> j) & PAC_COUNT_MASK;
	}
}

void PacCountUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, data_ptr_t state_ptr, idx_t count) {
	D_ASSERT(input_count == 1 || input_count == 2); // optional mi param (unused here) can make it 2
	auto &state = *reinterpret_cast<PacCountState *>(state_ptr);

	UnifiedVectorFormat idata;
	inputs[0].ToUnifiedFormat(count, idata);
	auto input_data = UnifiedVectorFormat::GetData<uint64_t>(idata);
	for (idx_t i = 0; i < count; i++) {
		PacCountUpdateInternal(idata, i, input_data, state);
	}
}

void PacCountScatterUpdate(Vector param[], AggregateInputData &, idx_t np, Vector &states, idx_t cnt) {
	D_ASSERT(np == 1 || np == 2); // optional mi param (unused here) can make it 2

	UnifiedVectorFormat idata;
	param[0].ToUnifiedFormat(cnt, idata);
	auto input_data = UnifiedVectorFormat::GetData<uint64_t>(idata);
	UnifiedVectorFormat sdata;
	states.ToUnifiedFormat(cnt, sdata);

	auto state_p = UnifiedVectorFormat::GetData<PacCountState *>(sdata);
	for (idx_t i = 0; i < cnt; i++) {
		PacCountUpdateInternal(idata, i, input_data, *state_p[sdata.sel->get_index(i)]);
	}
}

AUTOVECTORIZE void PacCountCombine(Vector &src, Vector &dst, AggregateInputData &, idx_t cnt) {
	auto src_state = FlatVector::GetData<PacCountState *>(src);
	auto dst_state = FlatVector::GetData<PacCountState *>(dst);
	for (idx_t i = 0; i < cnt; i++) {
		src_state[i]->Flush();
		for (int j = 0; j < 64; j++) {
			dst_state[i]->totals64[j] += src_state[i]->totals64[j];
		}
	}
}

void PacCountFinalize(Vector &states, AggregateInputData &aggr_input, Vector &res, idx_t cnt, idx_t off) {
    auto state = FlatVector::GetData<PacCountState *>(states);
    auto data = FlatVector::GetData<uint64_t>(res);
    thread_local std::mt19937_64 gen(std::random_device{}());
	double mi = aggr_input.bind_data->Cast<PacBindData>().mi;

    for (idx_t i = 0; i < cnt; i++) {
        state[i]->Flush(); // flush any remaining small totals into the big totals
        double counters_d[64];
        ToDoubleArray(state[i]->totals64, counters_d); // Convert uint64_t totals64 to double array
		data[off + i] = static_cast<uint64_t>(PacNoisySampleFrom64Counters(counters_d, mi, gen))
		                + state[i]->totals64[42];
    }
}

void RegisterPacCountFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet fcn_set("pac_count");

	fcn_set.AddFunction(AggregateFunction(
	    "pac_count", {LogicalType::UBIGINT}, LogicalType::UBIGINT, PacCountStateSize,
	    PacCountInitialize, PacCountScatterUpdate, PacCountCombine, PacCountFinalize,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacCountUpdate, PacCountBind));

	fcn_set.AddFunction(AggregateFunction(
	    "pac_count", {LogicalType::UBIGINT, LogicalType::DOUBLE}, LogicalType::UBIGINT, PacCountStateSize,
	    PacCountInitialize, PacCountScatterUpdate, PacCountCombine, PacCountFinalize,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING, PacCountUpdate, PacCountBind));

	loader.RegisterFunction(fcn_set);
}

} // namespace duckdb
