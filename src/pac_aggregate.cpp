#include "include/pac_aggregate.hpp"

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/aggregate_function.hpp"

#include <random>
#define _USE_MATH_DEFINES // for M_PI on MSVC
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cstring>
#include <type_traits>
#include <limits>

// Every argument to pac_aggregate is the output of a query evaluated on a random subsample of the privacy unit

namespace duckdb {

// ============================================================================
// NOTE: pac_count implementation was moved to src/pac_count.cpp / src/include/pac_count.hpp.
// The noisy-sample computation used by multiple aggregates is kept here and exported.
// ============================================================================

// Forward declaration for the internal variance helper so it can be used above
static double ComputeSecondMomentVariance(const std::vector<double> &values);

// Deterministic sampling helpers (use engine() bits directly so draws are identical across platforms
// for a given mt19937_64 seed). Function names follow repository style (no snake_case).
static inline uint64_t EngineNext(std::mt19937_64 &gen) {
	return gen();
}

// DeterministicUniformRange: unbiased integer in [0, n-1] using rejection sampling on 64-bit engine outputs
static inline idx_t DeterministicUniformRange(std::mt19937_64 &gen, idx_t n) {
	if (n <= 0) {
		return 0;
	}
	constexpr uint64_t maxv = std::numeric_limits<uint64_t>::max();
	// compute largest multiple of n that fits in uint64_t (limit = floor((maxv+1)/n)*n)
	// to avoid overflow use division
	uint64_t limit = (maxv / static_cast<uint64_t>(n)) * static_cast<uint64_t>(n);
	while (true) {
		uint64_t r = EngineNext(gen);
		if (r < limit) {
			return static_cast<idx_t>(r % static_cast<uint64_t>(n));
		}
		// otherwise retry
	}
}

// DeterministicUniformUnit: produce a uniform double in [0,1) using 53 bits from engine
static inline double DeterministicUniformUnit(std::mt19937_64 &gen) {
	uint64_t r = EngineNext(gen);
	uint64_t top53 = r >> 11;                              // keep top 53 bits
	constexpr double inv2pow53 = 1.0 / 9007199254740992.0; // 1 / 2^53
	return static_cast<double>(top53) * inv2pow53;
}

// DeterministicNormalSample: Box–Muller sampling using DeterministicUniformUnit
static inline double DeterministicNormalSample(std::mt19937_64 &gen, bool &has_spare, double &spare, double mean,
                                               double stddev) {
	if (has_spare) {
		has_spare = false;
		return mean + spare * stddev;
	}
	double u1 = DeterministicUniformUnit(gen);
	double u2 = DeterministicUniformUnit(gen);
	if (u1 <= 0.0) {
		u1 = std::numeric_limits<double>::min();
	}
	double r = std::sqrt(-2.0 * std::log(u1));
	double theta = 2.0 * M_PI * u2;
	double z0 = r * std::cos(theta);
	double z1 = r * std::sin(theta);
	spare = z1;
	has_spare = true;
	return mean + z0 * stddev;
}

// Finalize: compute noisy sample from the 64 counters (works on double array)
double PacNoisySampleFrom64Counters(const double counters[64], double mi, std::mt19937_64 &gen) {
	constexpr int N = 64;
	// Compute empirical (second-moment) variance across the 64 counters and use it
	// to determine the noise variance. We reuse ComputeSecondMomentVariance here.
	std::vector<double> vals(counters, counters + N);
	// Compute delta using the shared exported helper (validates mi as well)
	double delta = ComputeDeltaFromValues(vals, mi);

	// Pick random index J in [0, N-1] to select the base counter yJ (same semantics as before)
	int J = static_cast<int>(DeterministicUniformRange(gen, N));
	double yJ = counters[J];

	if (delta <= 0.0 || !std::isfinite(delta)) {
		// If there's no variance, return the selected counter value without noise.
		return yJ;
	}

	// Sample normal(0, sqrt(delta)) using deterministic Box–Muller
	// We need a spare buffer for the generated pair; but PacAggregateLocalState owns that state.
	// The helper above only uses engine/local spare passed by caller; here we can't access spare.
	// To keep the same signature, fall back to a stateless Box–Muller that consumes two engine draws
	// and returns one sample (no spare saved). This remains deterministic across platforms.

	// draw two uniforms
	double u1 = DeterministicUniformUnit(gen);
	double u2 = DeterministicUniformUnit(gen);
	if (u1 <= 0.0) {
		u1 = std::numeric_limits<double>::min();
	}
	double r = std::sqrt(-2.0 * std::log(u1));
	double theta = 2.0 * M_PI * u2;
	double z0 = r * std::cos(theta);
	return yJ + z0 * std::sqrt(delta);
}

struct PacAggregateLocalState : public FunctionLocalState {
	explicit PacAggregateLocalState(uint64_t seed) : gen(seed), has_spare(false), spare(0.0) {
	}
	std::mt19937_64 gen;
	bool has_spare;
	double spare;
};

// Compute second-moment variance (not unbiased estimator)
// todo - check if this is leave one out or n denominator
static double ComputeSecondMomentVariance(const std::vector<double> &values) {
	idx_t n = values.size();
	if (n <= 1) {
		return 0.0;
	}

	double mean = 0.0;
	for (auto v : values) {
		mean += v;
	}
	mean /= static_cast<double>(n);

	double var = 0.0;
	for (auto v : values) {
		double d = v - mean;
		var += d * d;
	}
	// Use sample variance (divide by n-1) to make the estimator unbiased for finite samples
	return var / static_cast<double>(n - 1);
}

// Exported helper: compute the PAC noise variance (delta) from values and mi.
// This implements the header-declared ComputeDeltaFromValues and is reused by other files.
double ComputeDeltaFromValues(const std::vector<double> &values, double mi) {
	if (mi <= 0.0) {
		throw InvalidInputException("ComputeDeltaFromValues: mi must be > 0");
	}
	double sigma2 = ComputeSecondMomentVariance(values);
	double delta = sigma2 / (2.0 * mi);
	return delta;
}

unique_ptr<FunctionLocalState> PacAggregateInit(ExpressionState &state, const BoundFunctionExpression &,
                                                FunctionData *bind_data) {
	// Single source-of-truth: prefer seed supplied via bind_data (PacBindData). Do NOT read the session
	// setting here. If bind_data is not provided, fall back to non-deterministic random_device.
	uint64_t seed = std::random_device {}();
	if (bind_data) {
		seed = bind_data->Cast<PacBindData>().seed;
	}
	return make_uniq<PacAggregateLocalState>(seed);
}

// Templated implementation that reads list entries of arbitrary numeric type T for values
// and arbitrary integer-like type C for counts. We use the name PacAggregateScalar as the
// template so callers can instantiate PacAggregateScalar<T,C> directly.
template <class T, class C>
static void PacAggregateScalar(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &vals = args.data[0];
	auto &cnts = args.data[1];
	auto &mi_vec = args.data[2];
	auto &k_vec = args.data[3];

	idx_t count = args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto res = FlatVector::GetData<double>(result);
	FlatVector::Validity(result).SetAllValid(count);

	auto &local = ExecuteFunctionState::GetFunctionState(state)->Cast<PacAggregateLocalState>();
	auto &gen = local.gen;

	// --- extract lists ---
	UnifiedVectorFormat vvals, vcnts;
	vals.ToUnifiedFormat(count, vvals);
	cnts.ToUnifiedFormat(count, vcnts);

	// Prepare unified format for k (integer per-row parameter) once per chunk
	UnifiedVectorFormat kvals;
	k_vec.ToUnifiedFormat(count, kvals);
	auto *kdata = UnifiedVectorFormat::GetData<int32_t>(kvals);

	for (idx_t row = 0; row < count; row++) {
		bool refuse = false;

		// --- read mi, k ---
		double mi = mi_vec.GetValue(row).GetValue<double>();
		if (mi <= 0.0) {
			throw InvalidInputException("pac_aggregate: mi must be > 0");
		}
		idx_t kidx = kvals.sel ? kvals.sel->get_index(row) : row;
		int k = kdata[kidx];

		idx_t r = vvals.sel ? vvals.sel->get_index(row) : row;
		if (!vvals.validity.RowIsValid(r) || !vcnts.validity.RowIsValid(r)) {
			result.SetValue(row, Value());
			continue;
		}

		auto *vals_entries = UnifiedVectorFormat::GetData<list_entry_t>(vvals);
		auto *cnts_entries = UnifiedVectorFormat::GetData<list_entry_t>(vcnts);

		auto ve = vals_entries[r];
		auto ce = cnts_entries[r];

		// Values and counts arrays must have the same length (one count per sample position).
		if (ve.length != ce.length) {
			throw InvalidInputException("pac_aggregate: values and counts length mismatch");
		}
		idx_t vals_len = ve.length;
		idx_t cnts_len = ce.length;

		auto &vals_child = ListVector::GetEntry(vals);
		auto &cnts_child = ListVector::GetEntry(cnts);
		vals_child.Flatten(ve.offset + ve.length);
		cnts_child.Flatten(ce.offset + ce.length);

		auto *vdata = FlatVector::GetData<T>(vals_child);
		auto *cdata = FlatVector::GetData<C>(cnts_child);

		std::vector<double> values;
		values.reserve(vals_len);

		int64_t max_count = 0;
		for (idx_t i = 0; i < vals_len; i++) {
			if (!FlatVector::Validity(vals_child).RowIsValid(ve.offset + i)) {
				refuse = true;
				break;
			}
			values.push_back(ToDouble<T>(vdata[ve.offset + i]));
			int64_t cnt_val = 0;
			if (i < cnts_len) {
				// Cast counts to int64 for the max_count check so BIGINT[] counts don't truncate.
				cnt_val = static_cast<int64_t>(cdata[ce.offset + i]);
			}
			max_count = std::max<int64_t>(max_count, cnt_val);
		}

		if (refuse || values.empty() || max_count < static_cast<int64_t>(k)) {
			result.SetValue(row, Value());
			continue;
		}

		// ---------------- PAC core ----------------

		// 1. pick J
		idx_t J = DeterministicUniformRange(gen, values.size());
		double yJ = values[J];

		// 2. empirical (second-moment) variance over the full list
		// Compute delta using the shared helper (may throw if mi <= 0)
		double delta = ComputeDeltaFromValues(values, mi);

		if (delta <= 0.0 || !std::isfinite(delta)) {
			res[row] = yJ;
			continue;
		}

		// 3. noise
		res[row] = yJ + DeterministicNormalSample(gen, local.has_spare, local.spare, 0.0, std::sqrt(delta));
	}

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<FunctionData> PacAggregateBind(ClientContext &ctx, ScalarFunction &,
                                                 vector<unique_ptr<Expression>> &args) {
	uint64_t seed = std::random_device {}();
	Value pac_seed_val;
	if (ctx.TryGetCurrentSetting("pac_seed", pac_seed_val) && !pac_seed_val.IsNull()) {
		seed = static_cast<uint64_t>(pac_seed_val.GetValue<int64_t>());
	}
	// mi is per-row for pac_aggregate; store default mi in bind data but it's unused here
	double mi = 128.0;
	return make_uniq<PacBindData>(mi, seed);
}

void RegisterPacAggregateFunctions(ExtensionLoader &loader) {
	// Helper that registers one overload given the element logical types and a function pointer
	auto make_and_register = [&](const LogicalType &vals_elem, const LogicalType &cnts_elem,
	                             const scalar_function_t &fn) {
		auto fun = ScalarFunction(
		    "pac_aggregate",
		    {LogicalType::LIST(vals_elem), LogicalType::LIST(cnts_elem), LogicalType::DOUBLE, LogicalType::INTEGER},
		    LogicalType::DOUBLE, fn, PacAggregateBind, nullptr, nullptr, PacAggregateInit);
		fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
		loader.RegisterFunction(fun);
	};

	// For each values element type we register overloads where the counts element type can be any integer-like type.
	// This addresses cases like BIGINT[] counts (the user's error).
#define REG_COUNTS_FOR(V_LOG, V_CPP)                                                                                   \
	make_and_register(V_LOG, LogicalType::TINYINT, PacAggregateScalar<V_CPP, int8_t>);                                 \
	make_and_register(V_LOG, LogicalType::SMALLINT, PacAggregateScalar<V_CPP, int16_t>);                               \
	make_and_register(V_LOG, LogicalType::INTEGER, PacAggregateScalar<V_CPP, int32_t>);                                \
	make_and_register(V_LOG, LogicalType::BIGINT, PacAggregateScalar<V_CPP, int64_t>);                                 \
	make_and_register(V_LOG, LogicalType::HUGEINT, PacAggregateScalar<V_CPP, hugeint_t>);                              \
	make_and_register(V_LOG, LogicalType::UTINYINT, PacAggregateScalar<V_CPP, uint8_t>);                               \
	make_and_register(V_LOG, LogicalType::USMALLINT, PacAggregateScalar<V_CPP, uint16_t>);                             \
	make_and_register(V_LOG, LogicalType::UINTEGER, PacAggregateScalar<V_CPP, uint32_t>);                              \
	make_and_register(V_LOG, LogicalType::UBIGINT, PacAggregateScalar<V_CPP, uint64_t>);                               \
	make_and_register(V_LOG, LogicalType::UHUGEINT, PacAggregateScalar<V_CPP, uhugeint_t>);                            \
	/* Also accept floating-point counts arrays (users may accidentally pass DOUBLE[]/FLOAT[]) */                      \
	make_and_register(V_LOG, LogicalType::FLOAT, PacAggregateScalar<V_CPP, float>);                                    \
	make_and_register(V_LOG, LogicalType::DOUBLE, PacAggregateScalar<V_CPP, double>);

	// Register for all common value element types (including floats)
	REG_COUNTS_FOR(LogicalType::BOOLEAN, int8_t)
	REG_COUNTS_FOR(LogicalType::TINYINT, int8_t)
	REG_COUNTS_FOR(LogicalType::SMALLINT, int16_t)
	REG_COUNTS_FOR(LogicalType::INTEGER, int32_t)
	REG_COUNTS_FOR(LogicalType::BIGINT, int64_t)
	// HUGEINT values
	REG_COUNTS_FOR(LogicalType::HUGEINT, hugeint_t)
	// Unsigned values
	REG_COUNTS_FOR(LogicalType::UTINYINT, uint8_t)
	REG_COUNTS_FOR(LogicalType::USMALLINT, uint16_t)
	REG_COUNTS_FOR(LogicalType::UINTEGER, uint32_t)
	REG_COUNTS_FOR(LogicalType::UBIGINT, uint64_t)
	REG_COUNTS_FOR(LogicalType::UHUGEINT, uhugeint_t)
	// Floating point values: counts are still integer-like
	REG_COUNTS_FOR(LogicalType::FLOAT, float)
	REG_COUNTS_FOR(LogicalType::DOUBLE, double)

#undef REG_COUNTS_FOR
}

} // namespace duckdb
