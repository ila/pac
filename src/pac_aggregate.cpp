#include "include/pac_aggregate.hpp"

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/aggregate_function.hpp"

#include <random>
#include <cmath>
#include <limits>
#include <cstring>

// Every argument to pac_aggregate is the output of a query evaluated on a random subsample of the privacy unit

namespace duckdb {

// ============================================================================
// pac_count aggregate function
// ============================================================================
// State: 64 counters, one for each bit position
// Update: for each input key_hash, increment count[i] if bit i is set
// Finalize: compute the variance of the 64 counters
//
// Uses SIMD-friendly intermediate accumulators (small_totals) that get
// flushed to the main totals every 256 updates.

struct PacCountState {
	uint8_t update_count;     // Counts updates, flushes when wraps to 0
	uint64_t small_totals[8]; // SIMD-friendly intermediate accumulators
	uint64_t totals[64];      // Final counters

	void Flush() {
		const uint8_t *small = reinterpret_cast<const uint8_t *>(small_totals);
		for (int i = 0; i < 64; i++) {
			totals[i] += small[i];
		}
		memset(small_totals, 0, sizeof(small_totals));
	}
};

// SIMD-friendly mask: extracts bits at positions 0, 8, 16, 24, 32, 40, 48, 56
#define PAC_COUNT_MASK                                                                                                 \
	((1ULL << 0) | (1ULL << 8) | (1ULL << 16) | (1ULL << 24) | (1ULL << 32) | (1ULL << 40) | (1ULL << 48) |            \
	 (1ULL << 56))

struct PacCountOperation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.update_count = 0;
		memset(state.small_totals, 0, sizeof(state.small_totals));
		memset(state.totals, 0, sizeof(state.totals));
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &) {
		for (int i = 0; i < 8; i++) {
			state.small_totals[i] += (input >> i) & PAC_COUNT_MASK;
		}
		if (++state.update_count == 0) {
			// update_count wrapped (256 updates), flush to totals
			state.Flush();
		}
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &unary_input,
	                              idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, input, unary_input);
		}
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &) {
		// Flush any unflushed small_totals from source (need non-const copy)
		STATE src_copy = source;
		if (src_copy.update_count != 0) {
			src_copy.Flush();
		}
		if (target.update_count != 0) {
			target.Flush();
		}
		// Combine the totals
		for (int j = 0; j < 64; j++) {
			target.totals[j] += src_copy.totals[j];
		}
	}

	template <class RESULT_TYPE, class STATE>
	static void Finalize(STATE &state, RESULT_TYPE &result, AggregateFinalizeData &finalize_data) {
		// Flush any unflushed small_totals
		if (state.update_count != 0) {
			state.Flush();
		}

		// Compute the mean of the 64 counters
		double sum = 0.0;
		for (int j = 0; j < 64; j++) {
			sum += static_cast<double>(state.totals[j]);
		}
		double mean = sum / 64.0;

		// Compute variance (population variance / second moment)
		double variance = 0.0;
		for (int j = 0; j < 64; j++) {
			double diff = static_cast<double>(state.totals[j]) - mean;
			variance += diff * diff;
		}
		variance /= 64.0;

		result = variance;
	}

	static bool IgnoreNull() {
		return true;
	}
};


struct PacAggregateLocalState : public FunctionLocalState {
	explicit PacAggregateLocalState(uint64_t seed) : gen(seed) {}
	std::mt19937_64 gen;
};

// Compute second-moment variance (not unbiased estimator)
static double ComputeSecondMomentVariance(const std::vector<double> &values) {
	idx_t n = values.size();
	if (n <= 1) {
		return 0.0;
	}

	double mean = 0.0;
	for (auto v : values) mean += v;
	mean /= double(n);

	double var = 0.0;
	for (auto v : values) {
		double d = v - mean;
		var += d * d;
	}
	return var / double(n);
}

DUCKDB_API unique_ptr<FunctionLocalState>
PacAggregateInit(ExpressionState &state, const BoundFunctionExpression &, FunctionData *) {
	uint64_t seed = std::random_device{}();
	Value pac_seed;
	if (state.GetContext().TryGetCurrentSetting("pac_seed", pac_seed) && !pac_seed.IsNull()) {
		seed = uint64_t(pac_seed.GetValue<int64_t>());
	}
	return make_uniq<PacAggregateLocalState>(seed);
}

DUCKDB_API void PacAggregateScalar(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &vals = args.data[0];
	auto &cnts = args.data[1];
	auto &mi_vec = args.data[2];
	auto &k_vec  = args.data[3];

	idx_t count = args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto res = FlatVector::GetData<double>(result);
	FlatVector::Validity(result).SetAllValid(count);

	auto &local = ExecuteFunctionState::GetFunctionState(state)->Cast<PacAggregateLocalState>();
	auto &gen = local.gen;

	for (idx_t row = 0; row < count; row++) {
		bool refuse = false;

		// --- read mi, k ---
		double mi = mi_vec.GetValue(row).GetValue<double>();
		if (mi <= 0.0) {
			throw InvalidInputException("pac_aggregate: mi must be > 0");
		}
		int k = FlatVector::GetData<int32_t>(k_vec)[row];

		// --- extract lists ---
		UnifiedVectorFormat vvals, vcnts;
		vals.ToUnifiedFormat(count, vvals);
		cnts.ToUnifiedFormat(count, vcnts);

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

		// Read configured m from session settings (default 128)
		int64_t m_cfg = 128;
		Value m_val;
		if (state.GetContext().TryGetCurrentSetting("pac_m", m_val) && !m_val.IsNull()) {
			m_cfg = m_val.GetValue<int64_t>();
			if (m_cfg <= 0) {
				m_cfg = 128;
			}
		}

		// Read enforce_m_values flag (default true)
		bool enforce_m_values = true;
		Value enforce_val;
		if (state.GetContext().TryGetCurrentSetting("enforce_m_values", enforce_val) && !enforce_val.IsNull()) {
			enforce_m_values = enforce_val.GetValue<bool>();
		}

		// Enforce per-sample array length equals configured m (only if enabled)
		if (enforce_m_values && (int64_t)vals_len != m_cfg) {
			throw InvalidInputException(StringUtil::Format("pac_aggregate: expected per-sample array length %lld but got %llu", (long long)m_cfg, (unsigned long long)vals_len));
		}

		auto &vals_child = ListVector::GetEntry(vals);
		auto &cnts_child = ListVector::GetEntry(cnts);
		vals_child.Flatten(ve.offset + ve.length);
		cnts_child.Flatten(ce.offset + ce.length);

		auto *vdata = FlatVector::GetData<double>(vals_child);
		auto *cdata = FlatVector::GetData<int32_t>(cnts_child);

		std::vector<double> values;
		values.reserve(vals_len);

		int32_t max_count = 0;
		for (idx_t i = 0; i < vals_len; i++) {
			if (!FlatVector::Validity(vals_child).RowIsValid(ve.offset + i)) {
				refuse = true;
				break;
			}
			values.push_back(vdata[ve.offset + i]);
			int32_t cnt_val = 0;
			if (i < cnts_len) {
				cnt_val = cdata[ce.offset + i];
			}
			max_count = std::max(max_count, cnt_val);
		}

		if (refuse || values.empty() || max_count < k) {
			result.SetValue(row, Value());
			continue;
		}

		// ---------------- PAC core ----------------

		// 1. pick J
		std::uniform_int_distribution<idx_t> unif(0, values.size() - 1);
		idx_t J = unif(gen);
		double yJ = values[J];

		// 2. leave-one-out variance
		std::vector<double> loo;
		loo.reserve(values.size() - 1);
		for (idx_t i = 0; i < values.size(); i++) {
			if (i != J) {
				loo.push_back(values[i]);
			}
		}

		double sigma2 = ComputeSecondMomentVariance(loo);
		double delta  = sigma2 / (2.0 * mi);

		if (delta <= 0.0 || !std::isfinite(delta)) {
			res[row] = yJ;
			continue;
		}

		// 3. noise
		std::normal_distribution<double> gauss(0.0, std::sqrt(delta));
		res[row] = yJ + gauss(gen);
	}

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

// Overload that accepts LIST<BIGINT> for both values and counts.
DUCKDB_API void PacAggregateScalarBigint(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &vals = args.data[0];
	auto &cnts = args.data[1];
	auto &mi_vec = args.data[2];
	auto &k_vec  = args.data[3];

	idx_t count = args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto res = FlatVector::GetData<double>(result);
	FlatVector::Validity(result).SetAllValid(count);

	auto &local = ExecuteFunctionState::GetFunctionState(state)->Cast<PacAggregateLocalState>();
	auto &gen = local.gen;

	for (idx_t row = 0; row < count; row++) {
		bool refuse = false;

		// --- read mi, k ---
		double mi = mi_vec.GetValue(row).GetValue<double>();
		if (mi <= 0.0) {
			throw InvalidInputException("pac_aggregate: mi must be > 0");
		}
		int k = FlatVector::GetData<int32_t>(k_vec)[row];

		// --- extract lists ---
		UnifiedVectorFormat vvals, vcnts;
		vals.ToUnifiedFormat(count, vvals);
		cnts.ToUnifiedFormat(count, vcnts);

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

		// Read configured m from session settings (default 128)
		int64_t m_cfg = 128;
		Value m_val;
		if (state.GetContext().TryGetCurrentSetting("pac_m", m_val) && !m_val.IsNull()) {
			m_cfg = m_val.GetValue<int64_t>();
			if (m_cfg <= 0) {
				m_cfg = 128;
			}
		}

		// Read enforce_m_values flag (default true)
		// bool enforce_m_values = true;
		// Value enforce_val;
		// if (state.GetContext().TryGetCurrentSetting("enforce_m_values", enforce_val) && !enforce_val.IsNull()) {
		// 	enforce_m_values = enforce_val.GetValue<bool>();
		// }
		//
		// // Enforce per-sample array length equals configured m (only if enabled)
		// if (enforce_m_values && (int64_t)vals_len != m_cfg) {
		// 	throw InvalidInputException(StringUtil::Format("pac_aggregate: expected per-sample array length %lld but got %llu", (long long)m_cfg, (unsigned long long)vals_len));
		// }

		auto &vals_child = ListVector::GetEntry(vals);
		auto &cnts_child = ListVector::GetEntry(cnts);
		vals_child.Flatten(ve.offset + ve.length);
		cnts_child.Flatten(ce.offset + ce.length);

		auto *vdata = FlatVector::GetData<int64_t>(vals_child);
		auto *cdata = FlatVector::GetData<int64_t>(cnts_child);

		std::vector<double> values;
		values.reserve(vals_len);

		int32_t max_count = 0;
		for (idx_t i = 0; i < vals_len; i++) {
			if (!FlatVector::Validity(vals_child).RowIsValid(ve.offset + i)) {
				refuse = true;
				break;
			}
			// convert bigint value to double
			values.push_back(static_cast<double>(vdata[ve.offset + i]));
			int32_t cnt_val = 0;
			if (i < cnts_len) {
				cnt_val = static_cast<int32_t>(cdata[ce.offset + i]);
			}
			max_count = std::max(max_count, cnt_val);
		}

		if (refuse || values.empty() || max_count < k) {
			result.SetValue(row, Value());
			continue;
		}

		// ---------------- PAC core ----------------

		// 1. pick J
		std::uniform_int_distribution<idx_t> unif(0, values.size() - 1);
		idx_t J = unif(gen);
		double yJ = values[J];

		// 2. leave-one-out variance
		std::vector<double> loo;
		loo.reserve(values.size() - 1);
		for (idx_t i = 0; i < values.size(); i++) {
			if (i != J) {
				loo.push_back(values[i]);
			}
		}

		double sigma2 = ComputeSecondMomentVariance(loo);
		double delta  = sigma2 / (2.0 * mi);

		if (delta <= 0.0 || !std::isfinite(delta)) {
			res[row] = yJ;
			continue;
		}

		// 3. noise
		std::normal_distribution<double> gauss(0.0, std::sqrt(delta));
		res[row] = yJ + gauss(gen);
	}

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

void RegisterPacAggregateFunctions(ExtensionLoader &loader) {
	auto fun = ScalarFunction(
		"pac_aggregate",
		{LogicalType::LIST(LogicalType::DOUBLE),
		 LogicalType::LIST(LogicalType::INTEGER),
		 LogicalType::DOUBLE,
		 LogicalType::INTEGER},
		LogicalType::DOUBLE,
		PacAggregateScalar,
		nullptr, nullptr, nullptr,
		PacAggregateInit);

	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	loader.RegisterFunction(fun);

	// Register pac_count aggregate function
	// Input: UBIGINT key_hash
	// Output: DOUBLE (variance of the 64 bit counters)
	// Uses custom SIMD-friendly update that processes 128 values at a time
	AggregateFunction pac_count(
	    "pac_count",
	    {LogicalType::UBIGINT},
	    LogicalType::DOUBLE,
	    PacCountStateSize,
	    PacCountInitialize,
	    nullptr,  // update (scatter) - not used, we use simple_update
	    PacCountCombine,
	    PacCountFinalize,
	    FunctionNullHandling::DEFAULT_NULL_HANDLING,
	    PacCountUpdate  // simple_update - processes entire vector at once
	);
	loader.RegisterFunction(pac_count);

	// Register BIGINT overload: (LIST<BIGINT>, LIST<BIGINT>, DOUBLE, INT) -> DOUBLE
	auto fun_bigint = ScalarFunction(
		"pac_aggregate",
		{LogicalType::LIST(LogicalType::BIGINT),
		 LogicalType::LIST(LogicalType::BIGINT),
		 LogicalType::DOUBLE,
		 LogicalType::INTEGER},
		LogicalType::DOUBLE,
		PacAggregateScalarBigint,
		nullptr, nullptr, nullptr,
		PacAggregateInit);
	fun_bigint.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	loader.RegisterFunction(fun_bigint);
}

} // namespace duckdb
