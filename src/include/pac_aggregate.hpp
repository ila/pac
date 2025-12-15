#ifndef PAC_AGGREGATE_HPP
#define PAC_AGGREGATE_HPP

#include "duckdb.hpp"
#include <vector>

namespace duckdb {

// Forward-declare local state type (defined in pac_aggregate.cpp)
struct PacAggregateLocalState;

// Compute the PAC noise variance (delta) from per-sample values and mutual information budget mi.
// Throws InvalidInputException if mi <= 0.
DUCKDB_API double ComputeDeltaFromValues(const std::vector<double> &values, double mi);

// Initialize thread-local state for pac_aggregate (reads pac_seed setting).
DUCKDB_API unique_ptr<FunctionLocalState> PacAggregateInit(ExpressionState &state,
                                                               const BoundFunctionExpression &expr,
                                                               FunctionData *bind_data);

// Scalar function entry point used by the DuckDB runtime. Accepts
// (LIST<DOUBLE> values, LIST<INT> counts, DOUBLE mi, INT k) -> DOUBLE
DUCKDB_API void PacAggregateScalar(DataChunk &args, ExpressionState &state, Vector &result);

// Register pac_aggregate scalar function(s) with the extension loader
void RegisterPacAggregateFunctions(ExtensionLoader &loader);

} // namespace duckdb

#endif // PAC_AGGREGATE_HPP
