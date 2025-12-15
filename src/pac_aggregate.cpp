#include "include/pac_aggregate.hpp"

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/unary_executor.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/common/exception.hpp"

#include <random>
#include <cmath>
#include <limits>

namespace duckdb {
// Local state stored per-thread for the scalar function: RNG.
struct PacAggregateLocalState : public FunctionLocalState {
	explicit PacAggregateLocalState(uint64_t seed)
		: gen(seed) {}
	std::mt19937_64 gen;
};

// Compute the PAC noise variance (delta) from values and mi.
// Implements the mapping from empirical sample variance (unbiased estimator) to noise variance
// delta := sigma^2_m / (2 * mi)
DUCKDB_API double ComputeDeltaFromValues(const std::vector<double> &values, double mi) {
	// values: per-sample numeric outputs (length m)
	// mi: mutual information budget (beta), must be > 0
	if (values.empty()) {
		return 0.0;
	}
	if (mi <= 0.0) {
		throw InvalidInputException("pac_aggregate: mi must be > 0");
	}
	idx_t n = values.size();
	// compute mean
	double mean = 0.0;
	for (auto v : values) mean += v;
	mean /= double(n);
	// unbiased sample variance
	double var = 0.0;
	if (n > 1) {
		for (auto v : values) {
			double d = v - mean;
			var += d * d;
		}
		var /= double(n - 1);
	} else {
		var = 0.0;
	}
	// map to noise variance according to thesis Algorithm 2: delta := var / (2 * mi)
	double delta = var / (2.0 * mi);
	return delta;
}

// init_local_state: set up RNG seeded from session option `pac_seed`.
DUCKDB_API unique_ptr<FunctionLocalState> PacAggregateInit(ExpressionState &state, const BoundFunctionExpression &expr,
													   FunctionData *bind_data) {
	// Try to read pac_seed for deterministic RNG in tests
	Value pac_seed_value;
	uint64_t seed = std::random_device{}();
	if (state.GetContext().TryGetCurrentSetting("pac_seed", pac_seed_value)) {
		if (!pac_seed_value.IsNull()) {
			// Use provided seed value (cast to 64-bit)
			seed = uint64_t(pac_seed_value.GetValue<int64_t>());
		}
	}

	return make_uniq<PacAggregateLocalState>(seed);
}

// Scalar function implementation: expects list<double>, list<int>, double mi, int k
DUCKDB_API void PacAggregateScalar(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &vals = args.data[0];
	auto &cnts = args.data[1];
	auto &mi_vec = args.data[2];
	auto &k_vec = args.data[3];

	idx_t count = args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto res_data = FlatVector::GetData<double>(result);
	FlatVector::Validity(result).SetAllValid(count);

	// Obtain per-thread local state (RNG)
	auto local_state_ptr = ExecuteFunctionState::GetFunctionState(state);
	PacAggregateLocalState *local_state = nullptr;
	if (local_state_ptr) {
		local_state = &local_state_ptr->Cast<PacAggregateLocalState>();
	}

	for (idx_t row = 0; row < count; row++) {
		// Handle mi and k as scalars (they should be constant per-call but we handle generality)
		double mi = 1.0 / 128.0;
		if (mi_vec.GetType().id() != LogicalTypeId::SQLNULL) {
			mi = FlatVector::GetData<double>(mi_vec)[row];
		}
		int k = 3;
		if (k_vec.GetType().id() != LogicalTypeId::SQLNULL) {
			k = (int)FlatVector::GetData<int32_t>(k_vec)[row];
		}

		// Extract lists for this row using UnifiedVectorFormat
		std::vector<double> values;
		std::vector<int32_t> counts_vec;

		UnifiedVectorFormat vvals;
		vals.ToUnifiedFormat(count, vvals);
		idx_t sel_vals = vvals.sel ? vvals.sel->get_index(row) : row;
		if (!vvals.validity.RowIsValid(sel_vals)) {
			// values is NULL => strict refusal
			result.SetValue(row, Value()); // NULL
			continue;
		}
		auto *vals_entries = UnifiedVectorFormat::GetData<list_entry_t>(vvals);
		auto vals_entry = vals_entries[sel_vals];
		idx_t vals_offset = vals_entry.offset;
		idx_t vals_len = vals_entry.length;

		// counts
		UnifiedVectorFormat vcnts;
		cnts.ToUnifiedFormat(count, vcnts);
		idx_t sel_cnts = vcnts.sel ? vcnts.sel->get_index(row) : row;
		if (!vcnts.validity.RowIsValid(sel_cnts)) {
			result.SetValue(row, Value());
			continue;
		}
		auto *cnts_entries = UnifiedVectorFormat::GetData<list_entry_t>(vcnts);
		auto cnts_entry = cnts_entries[sel_cnts];
		idx_t cnts_offset = cnts_entry.offset;
		idx_t cnts_len = cnts_entry.length;

		if (vals_len != cnts_len) {
			throw InvalidInputException("pac_aggregate: values and counts arrays must have the same length");
		}

		// Read child vectors
		auto &vals_child = ListVector::GetEntry(vals);
		auto &cnts_child = ListVector::GetEntry(cnts);

		// Flatten child vectors to allow direct data access
		vals_child.Flatten(vals_len + vals_offset);
		cnts_child.Flatten(cnts_len + cnts_offset);

		auto vals_data = FlatVector::GetData<double>(vals_child);
		auto cnts_data = FlatVector::GetData<int32_t>(cnts_child);

		bool any_null = false;
		values.resize(vals_len);
		counts_vec.resize(cnts_len);
		for (idx_t i = 0; i < vals_len; ++i) {
			if (!FlatVector::Validity(vals_child).RowIsValid(vals_offset + i)) {
				any_null = true;
				break;
			}
			values[i] = vals_data[vals_offset + i];
			counts_vec[i] = cnts_data[cnts_offset + i];
		}
		if (any_null) {
			// strict null-based refusal
			result.SetValue(row, Value());
			continue;
		}

		// Cardinality check: v1 simple rule -> if maximum counts < k -> refuse
		int32_t max_count = 0;
		for (auto c : counts_vec) {
			if (c > max_count) max_count = c;
		}
		if (max_count < k) {
			result.SetValue(row, Value());
			continue;
		}

		idx_t n = values.size();
		if (n == 0) {
			result.SetValue(row, Value());
			continue;
		}

		// Compute noise variance (delta) from empirical variance
		double delta = ComputeDeltaFromValues(values, mi);

		// If local_state available, use its RNG; otherwise create a temporary RNG
		std::mt19937_64 *gen_ptr = nullptr;
		std::mt19937_64 temp_gen(std::random_device{}());
		if (local_state) {
			gen_ptr = &local_state->gen;
		} else {
			gen_ptr = &temp_gen;
		}

		// pick random index uniformly from 0..n-1
		std::uniform_int_distribution<idx_t> unif(0, n - 1);
		idx_t chosen = unif(*gen_ptr);
		double chosen_val = values[chosen];

		// If delta is zero/invalid, just return chosen value
		if (delta <= 0.0 || std::isnan(delta) || std::isinf(delta)) {
			res_data[row] = chosen_val;
			continue;
		}

		// Draw Gaussian noise with variance delta (stddev = sqrt(delta)) and add to chosen value
		double sd = std::sqrt(delta);
		std::normal_distribution<double> nd(0.0, sd);
		double noise = nd(*gen_ptr);
		res_data[row] = chosen_val + noise;
	}

	// If all input arguments are constant, mark the result as constant to satisfy the expression
	// evaluator's expectations during constant folding.
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

void RegisterPacAggregateFunctions(ExtensionLoader &loader) {
	// Build function signature: (LIST<DOUBLE>, LIST<INT>, DOUBLE, INT) -> DOUBLE
	auto fun = ScalarFunction(
		"pac_aggregate",
		{LogicalType::LIST(LogicalType::DOUBLE), LogicalType::LIST(LogicalType::INTEGER), LogicalType::DOUBLE,
		 LogicalType::INTEGER},
		LogicalType::DOUBLE, PacAggregateScalar, nullptr, nullptr, nullptr, PacAggregateInit);
	// set null handling as special since we need to inspect NULLs ourselves
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	loader.RegisterFunction(fun);
}
}
