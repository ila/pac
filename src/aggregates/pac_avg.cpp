#include "aggregates/pac_sum.hpp"
#include "aggregates/pac_avg.hpp"

namespace duckdb {

// Instantiate counter finalize methods for pac_avg (using shared PacSumAvgFinalizeCounters with DIVIDE_BY_COUNT=true)
static void PacAvgFinalizeCountersSignedDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count,
                                               idx_t offset) {
	PacSumAvgFinalizeCounters<ScatterIntState<true>, true, true>(states, input, result, count, offset);
}
static void PacAvgFinalizeCountersUnsignedDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count,
                                                 idx_t offset) {
	PacSumAvgFinalizeCounters<ScatterIntState<false>, false, true>(states, input, result, count, offset);
}
static void PacAvgFinalizeCountersDouble(Vector &states, AggregateInputData &input, Vector &result, idx_t count,
                                         idx_t offset) {
	PacSumAvgFinalizeCounters<ScatterDoubleState, true, true>(states, input, result, count, offset);
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

	// HUGEINT - uses double state in approx mode
#ifndef PAC_EXACTSUM
	AddAvgFcn(fcn_set, LogicalType::HUGEINT, PacSumDoubleStateSize, PacSumDoubleInitialize,
	          PacSumScatterUpdateHugeIntDouble, PacSumCombineDoubleWrapper, PacAvgFinalizeDouble,
	          PacSumUpdateHugeIntDouble);
#else
	AddAvgFcn(fcn_set, LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateHugeInt,
	          PacSumCombineSigned, PacAvgFinalizeSignedDouble, PacSumUpdateHugeInt);
#endif
	// UHUGEINT (uses double state always)
	AddAvgFcn(fcn_set, LogicalType::UHUGEINT, PacSumDoubleStateSize, PacSumDoubleInitialize,
	          PacSumScatterUpdateUHugeInt, PacSumCombineDoubleWrapper, PacAvgFinalizeDouble, PacSumUpdateUHugeInt);

	// Floating point (uses double state)
	AddAvgFcn(fcn_set, LogicalType::FLOAT, PacSumDoubleStateSize, PacSumDoubleInitialize, PacSumScatterUpdateFloat,
	          PacSumCombineDoubleWrapper, PacAvgFinalizeDouble, PacSumUpdateFloat);
	AddAvgFcn(fcn_set, LogicalType::DOUBLE, PacSumDoubleStateSize, PacSumDoubleInitialize, PacSumScatterUpdateDouble,
	          PacSumCombineDoubleWrapper, PacAvgFinalizeDouble, PacSumUpdateDouble);

	// DECIMAL: dynamic dispatch based on decimal width (like DuckDB's avg)
	// Uses BindDecimalPacSumAvg<true> to select INT16/INT32/INT64/INT128 implementation at bind time
	fcn_set.AddFunction(AggregateFunction(
	    {LogicalType::UBIGINT, LogicalTypeId::DECIMAL}, LogicalType::DOUBLE, nullptr, nullptr, nullptr, nullptr,
	    nullptr, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, BindDecimalPacSumAvg<true>));
	fcn_set.AddFunction(AggregateFunction(
	    {LogicalType::UBIGINT, LogicalTypeId::DECIMAL, LogicalType::DOUBLE}, LogicalType::DOUBLE, nullptr, nullptr,
	    nullptr, nullptr, nullptr, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, BindDecimalPacSumAvg<true>));

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

	// HUGEINT - uses double state in approx mode
#ifndef PAC_EXACTSUM
	AddCountersFcn(LogicalType::HUGEINT, PacSumDoubleStateSize, PacSumDoubleInitialize,
	               PacSumScatterUpdateHugeIntDouble, PacSumCombineDoubleWrapper, PacAvgFinalizeCountersDouble,
	               PacSumUpdateHugeIntDouble);
#else
	AddCountersFcn(LogicalType::HUGEINT, PacSumIntStateSize, PacSumIntInitialize, PacSumScatterUpdateHugeInt,
	               PacSumCombineSigned, PacAvgFinalizeCountersSignedDouble, PacSumUpdateHugeInt);
#endif
	// UHUGEINT (uses double state always)
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
