#include "include/pac_sum.hpp"
#include "include/pac_avg.hpp"
#include "duckdb/common/types/decimal.hpp"
#include <cmath>

namespace duckdb {

// instantiate Finalize methods for pac_avg (with DIVIDE_BY_COUNT=true)
void PacAvgFinalizeDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	PacSumFinalize<ScatterDoubleState, double, true, true>(states, input, result, count, offset);
}
void PacAvgFinalizeSignedDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count, idx_t offset) {
	PacSumFinalize<ScatterIntState<true>, double, true, true>(states, input, result, count, offset);
}
void PacAvgFinalizeUnsignedDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count,
                                  idx_t offset) {
	PacSumFinalize<ScatterIntState<false>, double, false, true>(states, input, result, count, offset);
}

// ============================================================================
// PAC_AVG_COUNTERS: Returns all 64 counters as LIST<DOUBLE> for categorical queries
// ============================================================================
// This variant is used when the avg result will be used in a comparison
// in an outer categorical query. Instead of picking one counter and adding noise,
// it returns all 64 counters so the outer query can evaluate the comparison
// against all subsamples and produce a mask.

template <class State, bool SIGNED>
static void PacAvgFinalizeCounters(Vector &states, AggregateInputData &input, Vector &result, idx_t count,
                                   idx_t offset) {
	auto state_ptrs = FlatVector::GetData<State *>(states);

	// Result is LIST<DOUBLE>
	auto list_entries = FlatVector::GetData<list_entry_t>(result);
	auto &child_vec = ListVector::GetEntry(result);

	// Reserve space for all lists (64 elements each)
	idx_t total_elements = count * 64;
	ListVector::Reserve(result, total_elements);
	ListVector::SetListSize(result, total_elements);

	auto child_data = FlatVector::GetData<double>(child_vec);
	double buf[64];

	// scale_divisor is used by pac_avg on DECIMAL to convert internal integer representation back to decimal
	double scale_divisor = input.bind_data ? input.bind_data->Cast<PacBindData>().scale_divisor : 1.0;

	for (idx_t i = 0; i < count; i++) {
#ifndef PAC_NOBUFFERING
		PacSumFlushBuffer<SIGNED>(*state_ptrs[i], *state_ptrs[i], input.allocator);
#endif
		auto *s = state_ptrs[i]->GetState();

		// Set up the list entry
		list_entries[offset + i].offset = i * 64;
		list_entries[offset + i].length = 64;

		if (s) {
			s->Flush(input.allocator);
			s->GetTotalsAsDouble(buf);
		} else {
			memset(buf, 0, sizeof(buf));
		}

		// Divide by count for average
		uint64_t total_count = state_ptrs[i]->exact_count;
#ifndef PAC_NOBUFFERING
		if (s) {
			total_count += s->exact_count;
		}
#endif
		double divisor = static_cast<double>(total_count) * scale_divisor;

		// Copy the 64 counters to the list, divided by count
		for (idx_t j = 0; j < 64; j++) {
			child_data[i * 64 + j] = buf[j] / divisor;
		}
	}
}

// Instantiate counter finalize methods for pac_avg
void PacAvgFinalizeCountersSignedDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count,
                                        idx_t offset) {
	PacAvgFinalizeCounters<ScatterIntState<true>, true>(states, input, result, count, offset);
}
void PacAvgFinalizeCountersUnsignedDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count,
                                          idx_t offset) {
	PacAvgFinalizeCounters<ScatterIntState<false>, false>(states, input, result, count, offset);
}
void PacAvgFinalizeCountersDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count,
                                  idx_t offset) {
	PacAvgFinalizeCounters<ScatterDoubleState, true>(states, input, result, count, offset);
}

// Helper to get the right pac_avg AggregateFunction for a given physical type (used by BindDecimalPacAvg)
// Note: bind is set to nullptr - the caller (BindDecimalPacAvg) handles binding
static AggregateFunction GetPacAvgAggregate(PhysicalType type) {
	switch (type) {
	case PhysicalType::INT16:
		return AggregateFunction("pac_avg", {LogicalType::UBIGINT, LogicalType::SMALLINT}, LogicalType::DOUBLE,
		                         PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateSmallInt,
		                         PacSumCombineSigned, PacAvgFinalizeSignedDouble,
		                         FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateSmallInt);
	case PhysicalType::INT32:
		return AggregateFunction("pac_avg", {LogicalType::UBIGINT, LogicalType::INTEGER}, LogicalType::DOUBLE,
		                         PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateInteger,
		                         PacSumCombineSigned, PacAvgFinalizeSignedDouble,
		                         FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateInteger);
	case PhysicalType::INT64:
		return AggregateFunction("pac_avg", {LogicalType::UBIGINT, LogicalType::BIGINT}, LogicalType::DOUBLE,
		                         PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateBigInt,
		                         PacSumCombineSigned, PacAvgFinalizeSignedDouble,
		                         FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateBigInt);
	case PhysicalType::INT128:
		return AggregateFunction("pac_avg", {LogicalType::UBIGINT, LogicalType::HUGEINT}, LogicalType::DOUBLE,
		                         PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateHugeInt,
		                         PacSumCombineSigned, PacAvgFinalizeSignedDouble,
		                         FunctionNullHandling::DEFAULT_NULL_HANDLING, PacSumUpdateHugeInt);
	default:
		throw InternalException("Unsupported physical type for pac_avg decimal");
	}
}

// Dynamic dispatch for DECIMAL: selects the right integer implementation based on decimal width
static unique_ptr<FunctionData> BindDecimalPacAvg(ClientContext &ctx, AggregateFunction &function,
                                                  vector<unique_ptr<Expression>> &args) {
	auto decimal_type = args[1]->return_type; // value is arg 1 (arg 0 is hash)
	function = GetPacAvgAggregate(decimal_type.InternalType());
	function.name = "pac_avg";
	function.arguments[1] = decimal_type;
	// pac_avg always returns DOUBLE (like DuckDB's avg)
	function.return_type = LogicalType::DOUBLE;

	// Compute scale_divisor = 10^scale for DECIMAL types
	// This converts the internal integer representation back to the decimal value
	uint8_t scale = DecimalType::GetScale(decimal_type);
	double scale_divisor = std::pow(10.0, static_cast<double>(scale));

	// Get mi and seed (same as PacSumBind)
	double mi = 128.0;
	if (args.size() >= 3) {
		if (!args[2]->IsFoldable()) {
			throw InvalidInputException("pac_avg: mi parameter must be a constant");
		}
		auto mi_val = ExpressionExecutor::EvaluateScalar(ctx, *args[2]);
		mi = mi_val.GetValue<double>();
		if (mi < 0.0) {
			throw InvalidInputException("pac_avg: mi must be >= 0");
		}
	}
	uint64_t seed = std::random_device {}();
	Value pac_seed_val;
	if (ctx.TryGetCurrentSetting("pac_seed", pac_seed_val) && !pac_seed_val.IsNull()) {
		seed = uint64_t(pac_seed_val.GetValue<int64_t>());
	}
	return make_uniq<PacBindData>(mi, seed, scale_divisor);
}

// Helper to register both 2-param and 3-param (with optional mi) versions for pac_avg
static void AddAvgFcn(AggregateFunctionSet &set, const LogicalType &value_type, aggregate_size_t state_size,
                      aggregate_initialize_t init, aggregate_update_t scatter, aggregate_combine_t combine,
                      aggregate_finalize_t finalize, aggregate_simple_update_t update,
                      aggregate_destructor_t destructor = nullptr) {
	// pac_avg always returns DOUBLE (like DuckDB's avg)
	set.AddFunction(AggregateFunction("pac_avg", {LogicalType::UBIGINT, value_type}, LogicalType::DOUBLE, state_size,
	                                  init, scatter, combine, finalize, FunctionNullHandling::DEFAULT_NULL_HANDLING,
	                                  update, PacSumBind, destructor));
	set.AddFunction(AggregateFunction("pac_avg", {LogicalType::UBIGINT, value_type, LogicalType::DOUBLE},
	                                  LogicalType::DOUBLE, state_size, init, scatter, combine, finalize,
	                                  FunctionNullHandling::DEFAULT_NULL_HANDLING, update, PacSumBind, destructor));
}

void RegisterPacAvgFunctions(ExtensionLoader &loader) {
	AggregateFunctionSet fcn_set("pac_avg");

	// Signed integers (use int state, avg finalize returns DOUBLE)
	AddAvgFcn(fcn_set, LogicalType::TINYINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateTinyInt,
	          PacSumCombineSigned, PacAvgFinalizeSignedDouble, PacSumUpdateTinyInt);
	AddAvgFcn(fcn_set, LogicalType::BOOLEAN, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateTinyInt,
	          PacSumCombineSigned, PacAvgFinalizeSignedDouble, PacSumUpdateTinyInt);
	AddAvgFcn(fcn_set, LogicalType::SMALLINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateSmallInt,
	          PacSumCombineSigned, PacAvgFinalizeSignedDouble, PacSumUpdateSmallInt);
	AddAvgFcn(fcn_set, LogicalType::INTEGER, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateInteger,
	          PacSumCombineSigned, PacAvgFinalizeSignedDouble, PacSumUpdateInteger);
	AddAvgFcn(fcn_set, LogicalType::BIGINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateBigInt,
	          PacSumCombineSigned, PacAvgFinalizeSignedDouble, PacSumUpdateBigInt);

	// Unsigned integers
	AddAvgFcn(fcn_set, LogicalType::UTINYINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateUTinyInt,
	          PacSumCombineUnsigned, PacAvgFinalizeUnsignedDouble, PacSumUpdateUTinyInt);
	AddAvgFcn(fcn_set, LogicalType::USMALLINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateUSmallInt,
	          PacSumCombineUnsigned, PacAvgFinalizeUnsignedDouble, PacSumUpdateUSmallInt);
	AddAvgFcn(fcn_set, LogicalType::UINTEGER, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateUInteger,
	          PacSumCombineUnsigned, PacAvgFinalizeUnsignedDouble, PacSumUpdateUInteger);
	AddAvgFcn(fcn_set, LogicalType::UBIGINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateUBigInt,
	          PacSumCombineUnsigned, PacAvgFinalizeUnsignedDouble, PacSumUpdateUBigInt);

	// HUGEINT
	AddAvgFcn(fcn_set, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateHugeInt,
	          PacSumCombineSigned, PacAvgFinalizeSignedDouble, PacSumUpdateHugeInt);
	// UHUGEINT (uses double state)
	AddAvgFcn(fcn_set, LogicalType::UHUGEINT, PacSumDoubleStateSize, PacSumDoubleInitialize,
	          PacSumScatterUpdateUHugeInt, PacSumCombineDoubleWrapper, PacAvgFinalizeDouble, PacSumUpdateUHugeInt);

	// Floating point (uses double state)
	AddAvgFcn(fcn_set, LogicalType::FLOAT, PacSumDoubleStateSize, PacSumDoubleInitialize, PacSumScatterUpdateFloat,
	          PacSumCombineDoubleWrapper, PacAvgFinalizeDouble, PacSumUpdateFloat);
	AddAvgFcn(fcn_set, LogicalType::DOUBLE, PacSumDoubleStateSize, PacSumDoubleInitialize, PacSumScatterUpdateDouble,
	          PacSumCombineDoubleWrapper, PacAvgFinalizeDouble, PacSumUpdateDouble);

	// DECIMAL: dynamic dispatch based on decimal width (like DuckDB's avg)
	// Uses BindDecimalPacAvg to select INT16/INT32/INT64/INT128 implementation at bind time
	fcn_set.AddFunction(AggregateFunction({LogicalType::UBIGINT, LogicalTypeId::DECIMAL}, LogicalType::DOUBLE, nullptr,
	                                      nullptr, nullptr, nullptr, nullptr,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, BindDecimalPacAvg));
	fcn_set.AddFunction(AggregateFunction({LogicalType::UBIGINT, LogicalTypeId::DECIMAL, LogicalType::DOUBLE},
	                                      LogicalType::DOUBLE, nullptr, nullptr, nullptr, nullptr, nullptr,
	                                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, BindDecimalPacAvg));

	loader.RegisterFunction(fcn_set);
}

// ============================================================================
// PAC_AVG_COUNTERS: Returns all 64 counters as LIST<DOUBLE> for categorical queries
// ============================================================================
void RegisterPacAvgCountersFunctions(ExtensionLoader &loader) {
	auto list_double_type = LogicalType::LIST(LogicalType::DOUBLE);
	AggregateFunctionSet counters_set("pac_avg_counters");

	// Helper to add both 2-param and 3-param versions for counters
	auto AddCountersFcn = [&](const LogicalType &value_type, aggregate_size_t state_size, aggregate_initialize_t init,
	                          aggregate_update_t scatter, aggregate_combine_t combine, aggregate_finalize_t finalize,
	                          aggregate_simple_update_t update) {
		counters_set.AddFunction(AggregateFunction("pac_avg_counters", {LogicalType::UBIGINT, value_type},
		                                           list_double_type, state_size, init, scatter, combine, finalize,
		                                           FunctionNullHandling::DEFAULT_NULL_HANDLING, update, PacSumBind));
		counters_set.AddFunction(AggregateFunction(
		    "pac_avg_counters", {LogicalType::UBIGINT, value_type, LogicalType::DOUBLE}, list_double_type, state_size,
		    init, scatter, combine, finalize, FunctionNullHandling::DEFAULT_NULL_HANDLING, update, PacSumBind));
	};

	// Signed integers
	AddCountersFcn(LogicalType::TINYINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateTinyInt,
	               PacSumCombineSigned, PacAvgFinalizeCountersSignedDouble, PacSumUpdateTinyInt);
	AddCountersFcn(LogicalType::BOOLEAN, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateTinyInt,
	               PacSumCombineSigned, PacAvgFinalizeCountersSignedDouble, PacSumUpdateTinyInt);
	AddCountersFcn(LogicalType::SMALLINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateSmallInt,
	               PacSumCombineSigned, PacAvgFinalizeCountersSignedDouble, PacSumUpdateSmallInt);
	AddCountersFcn(LogicalType::INTEGER, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateInteger,
	               PacSumCombineSigned, PacAvgFinalizeCountersSignedDouble, PacSumUpdateInteger);
	AddCountersFcn(LogicalType::BIGINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateBigInt,
	               PacSumCombineSigned, PacAvgFinalizeCountersSignedDouble, PacSumUpdateBigInt);

	// Unsigned integers
	AddCountersFcn(LogicalType::UTINYINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateUTinyInt,
	               PacSumCombineUnsigned, PacAvgFinalizeCountersUnsignedDouble, PacSumUpdateUTinyInt);
	AddCountersFcn(LogicalType::USMALLINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateUSmallInt,
	               PacSumCombineUnsigned, PacAvgFinalizeCountersUnsignedDouble, PacSumUpdateUSmallInt);
	AddCountersFcn(LogicalType::UINTEGER, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateUInteger,
	               PacSumCombineUnsigned, PacAvgFinalizeCountersUnsignedDouble, PacSumUpdateUInteger);
	AddCountersFcn(LogicalType::UBIGINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateUBigInt,
	               PacSumCombineUnsigned, PacAvgFinalizeCountersUnsignedDouble, PacSumUpdateUBigInt);

	// HUGEINT
	AddCountersFcn(LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateHugeInt,
	               PacSumCombineSigned, PacAvgFinalizeCountersSignedDouble, PacSumUpdateHugeInt);
	// UHUGEINT (uses double state)
	AddCountersFcn(LogicalType::UHUGEINT, PacSumDoubleStateSize, PacSumDoubleInitialize, PacSumScatterUpdateUHugeInt,
	               PacSumCombineDoubleWrapper, PacAvgFinalizeCountersDouble, PacSumUpdateUHugeInt);

	// Floating point
	AddCountersFcn(LogicalType::FLOAT, PacSumDoubleStateSize, PacSumDoubleInitialize, PacSumScatterUpdateFloat,
	               PacSumCombineDoubleWrapper, PacAvgFinalizeCountersDouble, PacSumUpdateFloat);
	AddCountersFcn(LogicalType::DOUBLE, PacSumDoubleStateSize, PacSumDoubleInitialize, PacSumScatterUpdateDouble,
	               PacSumCombineDoubleWrapper, PacAvgFinalizeCountersDouble, PacSumUpdateDouble);

	loader.RegisterFunction(counters_set);
}

} // namespace duckdb
