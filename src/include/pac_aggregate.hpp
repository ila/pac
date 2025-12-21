#ifndef PAC_AGGREGATE_HPP
#define PAC_AGGREGATE_HPP

#include "duckdb.hpp"
#include <vector>

// Enable AVX2 vectorization for functions that get this preappended (useful for x86, harmless for arm)
#define AUTOVECTORIZE __attribute__((target("avx2")))

namespace duckdb {

// Header for PAC aggregate helpers and public declarations used across pac_* files.
// Contains bindings and small helpers shared between pac_aggregate, pac_count and pac_sum implementations.

// Forward-declare local state type (defined in pac_aggregate.cpp)
struct PacAggregateLocalState;

// Compute the PAC noise variance (delta) from per-sample values and mutual information budget mi.
// Throws InvalidInputException if mi <= 0.
DUCKDB_API double ComputeDeltaFromValues(const std::vector<double> &values, double mi);

// Initialize thread-local state for pac_aggregate (reads pac_seed setting).
DUCKDB_API unique_ptr<FunctionLocalState> PacAggregateInit(ExpressionState &state,
                                                               const BoundFunctionExpression &expr,
                                                               FunctionData *bind_data);

// NOTE: PacAggregateScalar and PacAggregateScalarBigint concrete symbols were removed.
// We now use the templated implementation PacAggregateScalar<T,C> instantiated at registration time.

// Register pac_aggregate scalar function(s) with the extension loader
void RegisterPacAggregateFunctions(ExtensionLoader &loader);

// Declare the noisy-sample helper so other translation units (pac_count.cpp) can call it.
DUCKDB_API double PacNoisySampleFrom64Counters(const double counters[64], double mi, std::mt19937_64 &gen);

// Bind data used by PAC aggregates to carry the `mi` parameter.
struct PacBindData : public FunctionData {
	double mi;
	explicit PacBindData(double mi_val) : mi(mi_val) {}
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<PacBindData>(mi);
	}
	bool Equals(const FunctionData &other) const override {
		return mi == other.Cast<PacBindData>().mi;
	}
};

// Helper to convert double to accumulator type (used by pac_sum finalizers)
template <class T>
static inline T FromDouble(double val) {
    return static_cast<T>(val);
}

// Specializations for hugeint_t and uhugeint_t
template <>
inline hugeint_t FromDouble<hugeint_t>(double val) {
	return Hugeint::Convert(static_cast<int64_t>(val));
}

// Helper to convert any numeric type to double for variance calculation
template <class T>
static inline double ToDouble(const T &val) {
	return static_cast<double>(val);
}

template <>
inline double ToDouble<hugeint_t>(const hugeint_t &val) {
	return Hugeint::Cast<double>(val);
}

// Specialization for unsigned hugeint (uhugeint_t)
template <>
inline double ToDouble<uhugeint_t>(const uhugeint_t &val) {
	return Uhugeint::Cast<double>(val);
}


// Helper to convert any totals array to double[64]
template <class T>
static inline void ToDoubleArray(const T *src, double *dst) {
	for (int i = 0; i < 64; i++) {
		dst[i] = ToDouble(src[i]);
	}
}
} // namespace duckdb

#endif // PAC_AGGREGATE_HPP
