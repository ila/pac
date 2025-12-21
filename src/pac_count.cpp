#include "include/pac_count.hpp"
#include "include/pac_aggregate.hpp"

#include "duckdb.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "pac_aggregate.hpp"

#include <cstring>

namespace duckdb {

AUTOVECTORIZE
void PacCountState::Flush() {
	const uint8_t *small = reinterpret_cast<const uint8_t *>(totals8);
	for (int i = 0; i < 64; i++) {
		totals64[i] += small[i];
	}
	memset(totals8, 0, sizeof(totals8));
}

idx_t PacCountStateSize(const AggregateFunction &) {
	return sizeof(PacCountState);
}

void PacCountInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	memset(state_ptr, 0, sizeof(PacCountState));
	auto &state = *reinterpret_cast<PacCountState *>(state_ptr);
	state.mi = 128.0; // default, may be overwritten in Update from bind_data
}

// Internal update helper
AUTOVECTORIZE static inline void
PacCountUpdateInternal(const UnifiedVectorFormat &idata, idx_t i, const uint64_t *input_data, PacCountState &state) {
	auto idx = idata.sel->get_index(i);
	if (!idata.validity.RowIsValid(idx)) {
		return;
	}
	uint64_t key_hash = input_data[idx];
	for (int j = 0; j < 8; j++) {
		state.totals8[j] += (key_hash >> j) & PAC_COUNT_MASK;
	}
	if (++state.update_count == 0) {
		state.Flush(); // every 256 iterations flush to avoid overflow
	}
}

void PacCountUpdate(Vector inputs[], AggregateInputData &aggr_input, idx_t input_count, data_ptr_t state_ptr, idx_t count) {
	D_ASSERT(input_count == 1 || input_count == 2);
	auto &state = *reinterpret_cast<PacCountState *>(state_ptr);

	// Store mi from bind_data into state
	if (aggr_input.bind_data) {
		state.mi = aggr_input.bind_data->Cast<PacBindData>().mi;
	}

	UnifiedVectorFormat idata;
	inputs[0].ToUnifiedFormat(count, idata);
	auto input_data = UnifiedVectorFormat::GetData<uint64_t>(idata);

	for (idx_t i = 0; i < count; i++) {
		PacCountUpdateInternal(idata, i, input_data, state);
	}
}

void PacCountScatterUpdate(Vector inputs[], AggregateInputData &aggr_input, idx_t input_count, Vector &states, idx_t count) {
	D_ASSERT(input_count == 1 || input_count == 2);

	// Get mi from bind_data
	double mi = 128.0;
	if (aggr_input.bind_data) {
		mi = aggr_input.bind_data->Cast<PacBindData>().mi;
	}

	UnifiedVectorFormat idata;
	inputs[0].ToUnifiedFormat(count, idata);
	auto input_data = UnifiedVectorFormat::GetData<uint64_t>(idata);
	UnifiedVectorFormat sdata;
	states.ToUnifiedFormat(count, sdata);
	auto state_ptrs = UnifiedVectorFormat::GetData<PacCountState *>(sdata);

	for (idx_t i = 0; i < count; i++) {
		auto state = state_ptrs[sdata.sel->get_index(i)];
		state->mi = mi;  // Store mi in state
		PacCountUpdateInternal(idata, i, input_data, *state);
	}
}

AUTOVECTORIZE void PacCountCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	auto sdata = FlatVector::GetData<PacCountState *>(source);
	auto tdata = FlatVector::GetData<PacCountState *>(target);
	for (idx_t i = 0; i < count; i++) {
		auto src = sdata[i];
		auto tgt = tdata[i];
		src->Flush();
		tgt->Flush();
		for (int j = 0; j < 64; j++) {
			tgt->totals64[j] += src->totals64[j];
		}
	}
}

unique_ptr<FunctionData> PacCountBind(ClientContext &context, AggregateFunction &function,
                                      vector<unique_ptr<Expression>> &arguments) {
	double mi = 128.0; // default
	if (arguments.size() >= 2) {
		if (!arguments[1]->IsFoldable()) {
			throw InvalidInputException("pac_count: mi parameter must be a constant");
		}
		auto mi_val = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
		mi = mi_val.GetValue<double>();
		if (mi <= 0.0) {
			throw InvalidInputException("pac_count: mi must be > 0");
		}
	}
	return make_uniq<PacBindData>(mi);
}

// pac_count finalize - uses mi from state (set during Update from bind_data)
void PacCountFinalize(Vector &states, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
    auto sdata = FlatVector::GetData<PacCountState *>(states);
    auto rdata = FlatVector::GetData<uint64_t>(result);
    thread_local std::mt19937_64 gen(std::random_device{}());

    for (idx_t i = 0; i < count; i++) {
        auto state = sdata[i];
        if (state->update_count != 0) {
            state->Flush();
        }
        // Convert uint64_t totals64 to double array
        double counters_d[64];
        ToDoubleArray(state->totals64, counters_d);
        // Compute noisy sampled result using PacNoisySampleFrom64Counters (returns yJ + noise)
        double noisy = PacNoisySampleFrom64Counters(counters_d, state->mi, gen);
        uint64_t res = static_cast<uint64_t>(noisy);
        rdata[offset + i] = res;
    }
}

void RegisterPacCountFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet pac_count_set("pac_count");

	AggregateFunction pac_count_1("pac_count", {LogicalType::UBIGINT}, LogicalType::UBIGINT, PacCountStateSize,
	                              PacCountInitialize, PacCountScatterUpdate, PacCountCombine, PacCountFinalize,
	                              FunctionNullHandling::DEFAULT_NULL_HANDLING, PacCountUpdate, PacCountBind);
	pac_count_set.AddFunction(pac_count_1);

	AggregateFunction pac_count_2("pac_count", {LogicalType::UBIGINT, LogicalType::DOUBLE}, LogicalType::UBIGINT,
	                              PacCountStateSize, PacCountInitialize, PacCountScatterUpdate, PacCountCombine,
	                              PacCountFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING, PacCountUpdate,
	                              PacCountBind);
	pac_count_set.AddFunction(pac_count_2);

	loader.RegisterFunction(pac_count_set);
}

} // namespace duckdb
