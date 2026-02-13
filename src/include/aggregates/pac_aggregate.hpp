#ifndef PAC_AGGREGATE_HPP
#define PAC_AGGREGATE_HPP

#include "duckdb.hpp"
#include "duckdb/common/random_engine.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/transaction/meta_transaction.hpp"
#include <type_traits>

// Cross-platform restrict keyword: MSVC uses __restrict, GCC/Clang use __restrict__
#if defined(_MSC_VER)
#define PAC_RESTRICT __restrict
#else
#define PAC_RESTRICT __restrict__
#endif

// Enable AVX2 vectorization for functions that get this preappended (useful for x86, harmless for arm)
// Only use __attribute__ on x86 with GCC/Clang - MSVC doesn't support this syntax
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
// On x86 targets with GCC/Clang, enable the attribute to allow function-level AVX2 codegen when available.
#define AUTOVECTORIZE __attribute__((target("avx2")))
#else
// On non-x86 targets (ARM, etc.) or Windows/MSVC, the attribute is invalid — make it a no-op.
#define AUTOVECTORIZE
#endif

#ifndef PAC_FLOAT
#define PAC_FLOAT float
#endif

#define PAC_MAGIC_HASH 2983746509182734091ULL

// Cross-platform popcount for 64-bit integers
// MSVC doesn't have __builtin_popcountll, so we need to handle it differently
#if defined(_MSC_VER)
#include <intrin.h>
static inline int pac_popcount64(uint64_t x) {
#if defined(_M_X64) || defined(_M_AMD64)
	return static_cast<int>(__popcnt64(x));
#else
	// Fallback for 32-bit MSVC: split into two 32-bit popcounts
	return static_cast<int>(__popcnt(static_cast<uint32_t>(x)) + __popcnt(static_cast<uint32_t>(x >> 32)));
#endif
}
// Cross-platform count leading zeros for 64-bit integers
static inline int pac_clzll(uint64_t x) {
	unsigned long index;
#if defined(_M_X64) || defined(_M_AMD64)
	if (_BitScanReverse64(&index, x)) {
		return 63 - static_cast<int>(index);
	}
#else
	// Fallback for 32-bit MSVC
	if (_BitScanReverse(&index, static_cast<uint32_t>(x >> 32))) {
		return 31 - static_cast<int>(index);
	}
	if (_BitScanReverse(&index, static_cast<uint32_t>(x))) {
		return 63 - static_cast<int>(index);
	}
#endif
	return 64; // x == 0
}
#else
// GCC/Clang have __builtin_popcountll
static inline int pac_popcount64(uint64_t x) {
	return __builtin_popcountll(x);
}
// GCC/Clang have __builtin_clzll
static inline int pac_clzll(uint64_t x) {
	return x ? __builtin_clzll(x) : 64;
}
#endif

namespace duckdb {

// Returns the DuckDB LogicalType corresponding to PAC_FLOAT (FLOAT or DOUBLE).
static inline LogicalType PacFloatLogicalType() {
	return std::is_same<PAC_FLOAT, float>::value ? LogicalType::FLOAT : LogicalType::DOUBLE;
}

// Header for PAC aggregate helpers and public declarations used across pac_* files.
// Contains bindings and small helpers shared between pac_aggregate, pac_count and pac_sum implementations.

// Forward-declare local state type (defined in pac_aggregate.cpp)
struct PacAggregateLocalState;

// Compute the PAC noise variance (delta) from per-sample values and mutual information budget mi.
// Throws InvalidInputException if mi < 0.
double ComputeDeltaFromValues(const vector<PAC_FLOAT> &values, double mi);

// Initialize thread-local state for pac_aggregate (reads pac_seed setting).
unique_ptr<FunctionLocalState> PacAggregateInit(ExpressionState &state, const BoundFunctionExpression &expr,
                                                FunctionData *bind_data);

// Register pac_aggregate scalar function(s) with the extension loader
void RegisterPacAggregateFunctions(ExtensionLoader &loader);

// Declare the noisy-sample helper so other translation units (pac_count.cpp) can call it.
// is_null: bitmask where bit i=1 means counter i should be excluded (compacted out)
// mi: mutual information parameter for noise calculation
// correction: factor to multiply values by after compacting NULLs but before adding noise
PAC_FLOAT PacNoisySampleFrom64Counters(const PAC_FLOAT counters[64], double mi, double correction, std::mt19937_64 &gen,
                                       bool use_deterministic_noise = true, uint64_t is_null = 0,
                                       uint64_t counter_selector = 0);

// PacNoisedSelect: returns true with probability proportional to popcount(key_hash)/64
// Uses rnd&63 as threshold, returns true if bitcount > threshold
static inline bool PacNoisedSelect(uint64_t key_hash, uint64_t rnd) {
	return pac_popcount64(key_hash) > static_cast<int>(rnd & 63);
}

// PacNoiseInNull: probabilistically returns true based on bit count in key_hash.
// mi: controls probabilistic (mi>0) vs deterministic (mi<=0) mode
// correction: reduces NULL probability by this factor (considers correction times more non-nulls)
// Probabilistic: P(NULL) = popcount(~key_hash) / (64 * correction)
// Deterministic: NULL when popcount(key_hash) * correction < 1
bool PacNoiseInNull(uint64_t key_hash, double mi, double correction, std::mt19937_64 &gen);

// Minimum total rows across all groups before the diversity check applies
#define PAC_SUSPICIOUS_THRESHOLD 100

struct PacBindData; // forward declaration

// Check for absence of sample diversity in PAC aggregates.
// Accumulates per-group exact_count into bind_data totals, classifies the group as
// suspicious or not, and throws if total_exact_count >= threshold AND suspicious > nonsuspicious.
void CheckPacSampleDiversity(uint64_t key_hash, const PAC_FLOAT *buf, uint64_t update_count, const char *aggr_name,
                             PacBindData &bind_data);

// Helper function to generate a random seed (defined in pac_aggregate.cpp)
// This avoids including <random> in the header for std::random_device
uint64_t PacGenerateRandomSeed();

// Helper to read pac_mi from session setting.
// Returns 1.0/128 if not found or null.
inline double GetPacMiFromSetting(ClientContext &ctx) {
	Value pac_mi_val;
	if (ctx.TryGetCurrentSetting("pac_mi", pac_mi_val) && !pac_mi_val.IsNull()) {
		return pac_mi_val.GetValue<double>();
	}
	return 1.0 / 128; // default
}

// Bind data used by PAC aggregates to carry `mi` and `correction` parameters.
// Reads seed from pac_seed setting (or uses query-id if not set) and computes query_hash.
// query_hash is used both for XOR'ing with per-row key_hash and as the counter selector
// for PacNoisySampleFrom64Counters.
struct PacBindData : public FunctionData {
	double mi;                    // mutual information parameter from pac_mi setting (controls noise/NULL probability)
	double correction;            // correction factor: multiplies sum/avg/count results, reduces NULL prob for min/max
	uint64_t seed;                // RNG seed: pac_seed setting value, or query-id if not set
	uint64_t query_hash;          // derived from seed: XOR'd with key_hash, also used as counter selector
	double scale_divisor;         // for DECIMAL pac_avg: divide result by 10^scale (default 1.0)
	bool use_deterministic_noise; // if true, use platform-agnostic Box-Muller noise generation

	// Runtime diversity tracking (accumulated across groups during finalize, not bind-time config)
	mutable uint64_t total_update_count;  // sum of update_counts across all finalized groups
	mutable uint64_t suspicious_count;    // groups with [29..35] zero bits AND all-same-value
	mutable uint64_t nonsuspicious_count; // groups that are not suspicious

	// Primary constructor - reads seed from pac_seed setting, or uses query-id if not set.
	// All aggregates in the same query get the same seed and query_hash.
	explicit PacBindData(ClientContext &ctx, double mi_val, double correction_val = 1.0, double scale_div = 1.0)
	    : mi(mi_val), correction(correction_val), scale_divisor(scale_div), use_deterministic_noise(false),
	      total_update_count(0), suspicious_count(0), nonsuspicious_count(0) {
		Value pac_seed_val;
		if (ctx.TryGetCurrentSetting("pac_seed", pac_seed_val) && !pac_seed_val.IsNull()) {
			seed = uint64_t(pac_seed_val.GetValue<int64_t>());
		} else {
			seed = 42;
		}
		if (mi != 0.0) { // randomize seed per query (not in deterministic mode aka mi==0)
			seed ^= PAC_MAGIC_HASH * static_cast<uint64_t>(ctx.ActiveTransaction().GetActiveQuery());
		}
		query_hash = (seed * PAC_MAGIC_HASH) ^ PAC_MAGIC_HASH;
	}

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<PacBindData>(*this); // uses implicit copy ctor (all fields are POD)
		copy->total_update_count = 0;              // reset runtime diversity counters
		copy->suspicious_count = 0;
		copy->nonsuspicious_count = 0;
		return copy;
	}
	bool Equals(const FunctionData &other) const override {
		auto &o = other.Cast<PacBindData>();
		return mi == o.mi && correction == o.correction && seed == o.seed && query_hash == o.query_hash &&
		       scale_divisor == o.scale_divisor && use_deterministic_noise == o.use_deterministic_noise;
	}
};

// Common bind helper: reads mi from setting, extracts correction from args[correction_arg_index],
// validates it (foldable constant >= 0), and returns a PacBindData.
// Returns correction=1.0 if correction_arg_index >= args.size().
inline unique_ptr<FunctionData> MakePacBindData(ClientContext &ctx, vector<unique_ptr<Expression>> &args,
                                                idx_t correction_arg_index, const char *func_name,
                                                double scale_divisor = 1.0) {
	double mi = GetPacMiFromSetting(ctx);
	double correction = 1.0;
	if (correction_arg_index < args.size()) {
		if (!args[correction_arg_index]->IsFoldable()) {
			throw InvalidInputException("%s: correction parameter must be a constant", func_name);
		}
		auto val = ExpressionExecutor::EvaluateScalar(ctx, *args[correction_arg_index]);
		correction = val.GetValue<double>();
		if (correction < 0.0) {
			throw InvalidInputException("%s: correction must be >= 0", func_name);
		}
	}
	return make_uniq<PacBindData>(ctx, mi, correction, scale_divisor);
}

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

// Helper to convert any totals array to PAC_FLOAT[64]
template <class T>
static inline void ToDoubleArray(const T *src, PAC_FLOAT *dst) {
	for (int i = 0; i < 64; i++) {
		dst[i] = static_cast<PAC_FLOAT>(ToDouble(src[i]));
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
