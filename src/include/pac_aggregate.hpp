#ifndef PAC_AGGREGATE_HPP
#define PAC_AGGREGATE_HPP

#include "duckdb.hpp"

// Enable AVX2 vectorization for functions that get this preappended (useful for x86, harmless for arm)
// Only use __attribute__ on x86 with GCC/Clang - MSVC doesn't support this syntax
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
// On x86 targets with GCC/Clang, enable the attribute to allow function-level AVX2 codegen when available.
#define AUTOVECTORIZE __attribute__((target("avx2")))
#else
// On non-x86 targets (ARM, etc.) or Windows/MSVC, the attribute is invalid — make it a no-op.
#define AUTOVECTORIZE
#endif

#define PAC_MAGIC_HASH 2983746509182734091ULL

namespace duckdb {

// Header for PAC aggregate helpers and public declarations used across pac_* files.
// Contains bindings and small helpers shared between pac_aggregate, pac_count and pac_sum implementations.

// Forward-declare local state type (defined in pac_aggregate.cpp)
struct PacAggregateLocalState;

// Compute the PAC noise variance (delta) from per-sample values and mutual information budget mi.
// Throws InvalidInputException if mi < 0.
double ComputeDeltaFromValues(const vector<double> &values, double mi);

// Initialize thread-local state for pac_aggregate (reads pac_seed setting).
unique_ptr<FunctionLocalState> PacAggregateInit(ExpressionState &state, const BoundFunctionExpression &expr,
                                                FunctionData *bind_data);

// Register pac_aggregate scalar function(s) with the extension loader
void RegisterPacAggregateFunctions(ExtensionLoader &loader);

// Declare the noisy-sample helper so other translation units (pac_count.cpp) can call it.
// is_null: bitmask where bit i=1 means counter i should be excluded (compacted out)
double PacNoisySampleFrom64Counters(const double counters[64], double mi, std::mt19937_64 &gen,
                                    bool use_deterministic_noise = true, uint64_t is_null = 0);

// PacNoisedSelect: returns true with probability proportional to popcount(key_hash)/64
// Uses rnd&63 as threshold, returns true if bitcount >= threshold
static inline bool PacNoisedSelect(uint64_t key_hash, uint64_t rnd) {
	return __builtin_popcountll(key_hash) >= (rnd & 63);
}

// PacNoiseInNull: probabilistically returns true based on bit count in key_hash.
// Probability = popcount(key_hash) / 64. If mi==0, returns deterministic bit 0.
bool PacNoiseInNull(uint64_t key_hash, double mi, std::mt19937_64 &gen);

// Bind data used by PAC aggregates to carry the `mi` parameter.
struct PacBindData : public FunctionData {
	double mi;
	uint64_t seed;                // deterministic RNG seed for PAC aggregates
	uint64_t query_hash;          // XOR'd with key_hash to randomize and avoid hash(0)==0 issue
	double scale_divisor;         // for DECIMAL pac_avg: divide result by 10^scale (default 1.0)
	bool use_deterministic_noise; // if true, use platform-agnostic Box-Muller noise generation
	explicit PacBindData(double mi_val, uint64_t seed_val = std::random_device {}(), double scale_div = 1.0,
	                     bool use_det_noise = false)
	    : mi(mi_val), seed(seed_val), query_hash((seed_val ^ PAC_MAGIC_HASH) * PAC_MAGIC_HASH),
	      scale_divisor(scale_div), use_deterministic_noise(use_det_noise) {
	}
	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<PacBindData>(mi, seed, scale_divisor, use_deterministic_noise);
		copy->query_hash = query_hash;
		return copy;
	}
	bool Equals(const FunctionData &other) const override {
		auto &o = other.Cast<PacBindData>();
		return mi == o.mi && seed == o.seed && query_hash == o.query_hash && scale_divisor == o.scale_divisor &&
		       use_deterministic_noise == o.use_deterministic_noise;
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
	return Hugeint::Convert(val); // Use direct double-to-hugeint conversion (handles values > INT64_MAX)
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

// Helper to convert input type to value type (for unified Update methods)
template <class VALUE_TYPE>
struct ConvertValue {
	template <class INPUT_TYPE>
	static inline VALUE_TYPE convert(const INPUT_TYPE &val) {
		return static_cast<VALUE_TYPE>(val);
	}
};

template <>
struct ConvertValue<double> {
	template <class INPUT_TYPE>
	static inline double convert(const INPUT_TYPE &val) {
		return ToDouble(val);
	}
};
} // namespace duckdb

#endif // PAC_AGGREGATE_HPP
