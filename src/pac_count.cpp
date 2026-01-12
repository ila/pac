#include <locale>
#include "include/pac_count.hpp"

namespace duckdb {

static unique_ptr<FunctionData> PacCountBind(ClientContext &ctx, AggregateFunction &func,
                                             vector<unique_ptr<Expression>> &args) {
	double mi = 128.0;
	for (idx_t i = 1; i < args.size(); i++) {
		auto &arg = args[i];
		if (arg->IsFoldable() && (arg->return_type.IsNumeric() || arg->return_type.id() == LogicalTypeId::UNKNOWN)) {
			auto val = ExpressionExecutor::EvaluateScalar(ctx, *arg);
			if (val.type().IsNumeric()) {
				mi = val.GetValue<double>();
				if (mi < 0.0) {
					throw InvalidInputException("pac_count: mi must be >= 0");
				}
				break;
			}
		}
	}
	uint64_t seed = std::random_device {}();
	Value pac_seed_val;
	if (ctx.TryGetCurrentSetting("pac_seed", pac_seed_val) && !pac_seed_val.IsNull()) {
		seed = uint64_t(pac_seed_val.GetValue<int64_t>());
	}
	return make_uniq<PacBindData>(mi, seed);
}

// State types: simple (non-scatter) always uses PacCountState directly
// Scatter uses PacCountStateWrapper for buffering (unless NOBUFFERING or NOCASCADING)
#if defined(PAC_NOBUFFERING) || defined(PAC_NOCASCADING)
using ScatterState = PacCountState;
#else
using ScatterState = PacCountStateWrapper;
#endif

// State functions - uses ScatterState for both simple and scatter
static idx_t PacCountStateSize(const AggregateFunction &) {
	return sizeof(ScatterState);
}

static void PacCountInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	memset(state_ptr, 0, sizeof(ScatterState));
}

void PacCountUpdate(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_ptr, idx_t count) {
	ScatterState &agg = *reinterpret_cast<ScatterState *>(state_ptr);
	uint64_t query_hash = aggr.bind_data->Cast<PacBindData>().query_hash;
	UnifiedVectorFormat idata;
	inputs[0].ToUnifiedFormat(count, idata);
	auto input_data = UnifiedVectorFormat::GetData<uint64_t>(idata);
	for (idx_t i = 0; i < count; i++) {
		auto idx = idata.sel->get_index(i);
		if (idata.validity.RowIsValid(idx)) {
			PacCountUpdateOne(agg, input_data[idx] ^ query_hash,
			                  aggr.allocator); // ungrouped: direct update (no buffering)
		}
	}
}

void PacCountColumnUpdate(Vector inputs[], AggregateInputData &aggr, idx_t, data_ptr_t state_ptr, idx_t count) {
	ScatterState &agg = *reinterpret_cast<ScatterState *>(state_ptr);
	uint64_t query_hash = aggr.bind_data->Cast<PacBindData>().query_hash;
	UnifiedVectorFormat hash_data, col_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, col_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	for (idx_t i = 0; i < count; i++) {
		auto h_idx = hash_data.sel->get_index(i);
		auto c_idx = col_data.sel->get_index(i);
		if (hash_data.validity.RowIsValid(h_idx) && col_data.validity.RowIsValid(c_idx)) {
			PacCountUpdateOne(agg, hashes[h_idx] ^ query_hash,
			                  aggr.allocator); // ungrouped: direct update (no buffering)
		}
	}
}

void PacCountScatterUpdate(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	uint64_t query_hash = aggr.bind_data->Cast<PacBindData>().query_hash;
	UnifiedVectorFormat idata, sdata;
	inputs[0].ToUnifiedFormat(count, idata);
	states.ToUnifiedFormat(count, sdata);
	auto input_data = UnifiedVectorFormat::GetData<uint64_t>(idata);
	auto state_p = UnifiedVectorFormat::GetData<ScatterState *>(sdata);
	for (idx_t i = 0; i < count; i++) {
		auto idx = idata.sel->get_index(i);
		if (idata.validity.RowIsValid(idx)) { // to protect against very many groups, thus uses buffering
#if defined(PAC_NOBUFFERING) || defined(PAC_NOCASCADING)
			PacCountUpdateOne(*state_p[sdata.sel->get_index(i)], input_data[idx] ^ query_hash, aggr.allocator);
#else
			PacCountBufferOrUpdateOne(*state_p[sdata.sel->get_index(i)], input_data[idx] ^ query_hash, aggr.allocator);
#endif
		}
	}
}

void PacCountColumnScatterUpdate(Vector inputs[], AggregateInputData &aggr, idx_t, Vector &states, idx_t count) {
	uint64_t query_hash = aggr.bind_data->Cast<PacBindData>().query_hash;
	UnifiedVectorFormat hash_data, col_data, sdata;
	inputs[0].ToUnifiedFormat(count, hash_data);
	inputs[1].ToUnifiedFormat(count, col_data);
	states.ToUnifiedFormat(count, sdata);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);
	auto state_p = UnifiedVectorFormat::GetData<ScatterState *>(sdata);
	for (idx_t i = 0; i < count; i++) {
		auto h_idx = hash_data.sel->get_index(i);
		auto c_idx = col_data.sel->get_index(i);
		if (hash_data.validity.RowIsValid(h_idx) && col_data.validity.RowIsValid(c_idx)) {
			// to protect against very many groups, thus uses buffering
#if defined(PAC_NOBUFFERING) || defined(PAC_NOCASCADING)
			PacCountUpdateOne(*state_p[sdata.sel->get_index(i)], hashes[h_idx] ^ query_hash, aggr.allocator);
#else
			PacCountBufferOrUpdateOne(*state_p[sdata.sel->get_index(i)], hashes[h_idx] ^ query_hash, aggr.allocator);
#endif
		}
	}
}

// Combine - flush src's buffer into dst (don't allocate src), then merge states
void PacCountCombine(Vector &src, Vector &dst, AggregateInputData &aggr, idx_t count) {
	auto sa = FlatVector::GetData<ScatterState *>(src);
	auto da = FlatVector::GetData<ScatterState *>(dst);
	for (idx_t i = 0; i < count; i++) {
#if !defined(PAC_NOBUFFERING) && !defined(PAC_NOCASCADING)
		// flush buffered values into dst (not into src -- it would trigger allocations)
		sa[i]->FlushBuffer(*da[i], aggr.allocator);
#endif
		PacCountState *ss = sa[i]->GetState();
		if (ss) { // we have an allocated state: flush it into dst
			PacCountState &ds = *da[i]->EnsureState(aggr.allocator);
			ss->FlushLevel();
			for (int j = 0; j < 64; j++) {
				ds.probabilistic_total[j] += ss->probabilistic_total[j];
			}
		}
	}
}

void PacCountFinalize(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	auto aggs = FlatVector::GetData<ScatterState *>(states);
	auto data = FlatVector::GetData<int64_t>(result);
	uint64_t seed = input.bind_data ? input.bind_data->Cast<PacBindData>().seed : std::random_device {}();
	std::mt19937_64 gen(seed);
	double mi = input.bind_data->Cast<PacBindData>().mi;
	double buf[64];
	for (idx_t i = 0; i < count; i++) {
#if !defined(PAC_NOBUFFERING) && !defined(PAC_NOCASCADING)
		aggs[i]->FlushBuffer(*aggs[i], input.allocator); // flush values into yourself
#endif
		PacCountState *s = aggs[i]->GetState();
		if (s) {
			s->FlushLevel(); // flush uint8_t level into uint64_t totals
			s->GetTotalsAsDouble(buf);
		} else {
			memset(buf, 0, sizeof(buf));
		}
		data[offset + i] =
		    static_cast<int64_t>(PacNoisySampleFrom64Counters(buf, mi, gen)) + static_cast<int64_t>(buf[41]);
	}
}

void RegisterPacCountFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet fcn_set("pac_count");

	fcn_set.AddFunction(AggregateFunction("pac_count", {LogicalType::UBIGINT}, LogicalType::BIGINT, PacCountStateSize,
	                                      PacCountInitialize, PacCountScatterUpdate, PacCountCombine, PacCountFinalize,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, PacCountUpdate, PacCountBind));

	fcn_set.AddFunction(AggregateFunction("pac_count", {LogicalType::UBIGINT, LogicalType::DOUBLE}, LogicalType::BIGINT,
	                                      PacCountStateSize, PacCountInitialize, PacCountScatterUpdate, PacCountCombine,
	                                      PacCountFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING, PacCountUpdate,
	                                      PacCountBind));

	fcn_set.AddFunction(AggregateFunction("pac_count", {LogicalType::UBIGINT, LogicalType::ANY}, LogicalType::BIGINT,
	                                      PacCountStateSize, PacCountInitialize, PacCountColumnScatterUpdate,
	                                      PacCountCombine, PacCountFinalize, FunctionNullHandling::SPECIAL_HANDLING,
	                                      PacCountColumnUpdate, PacCountBind));

	fcn_set.AddFunction(AggregateFunction("pac_count", {LogicalType::UBIGINT, LogicalType::ANY, LogicalType::DOUBLE},
	                                      LogicalType::BIGINT, PacCountStateSize, PacCountInitialize,
	                                      PacCountColumnScatterUpdate, PacCountCombine, PacCountFinalize,
	                                      FunctionNullHandling::SPECIAL_HANDLING, PacCountColumnUpdate, PacCountBind));

	loader.RegisterFunction(fcn_set);
}

} // namespace duckdb
