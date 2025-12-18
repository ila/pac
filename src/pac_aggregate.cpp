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

// Explicit function definitions for pac_count (used by manual AggregateFunction registration)
static idx_t PacCountStateSize(const AggregateFunction &) {
	return sizeof(PacCountState);
}

static void PacCountInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	auto &state = *reinterpret_cast<PacCountState *>(state_ptr);
	state.update_count = 0;
	memset(state.small_totals, 0, sizeof(state.small_totals));
	memset(state.totals, 0, sizeof(state.totals));
}

// Simple update for pac_count (ungrouped aggregation)
static void PacCountUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, data_ptr_t state_ptr, idx_t count) {
	D_ASSERT(input_count == 1);
	auto &state = *reinterpret_cast<PacCountState *>(state_ptr);
	auto &input = inputs[0];

	UnifiedVectorFormat idata;
	input.ToUnifiedFormat(count, idata);
	auto input_data = UnifiedVectorFormat::GetData<uint64_t>(idata);

	for (idx_t i = 0; i < count; i++) {
		auto idx = idata.sel->get_index(i);
		if (!idata.validity.RowIsValid(idx)) {
			continue;
		}
		uint64_t key_hash = input_data[idx];
		for (int j = 0; j < 8; j++) {
			state.small_totals[j] += (key_hash >> j) & PAC_COUNT_MASK;
		}
		if (++state.update_count == 0) {
			state.Flush();
		}
	}
}

// Scatter update for pac_count (grouped aggregation)
static void PacCountScatterUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &states,
                                  idx_t count) {
	D_ASSERT(input_count == 1);

	UnifiedVectorFormat idata;
	inputs[0].ToUnifiedFormat(count, idata);
	auto input_data = UnifiedVectorFormat::GetData<uint64_t>(idata);

	UnifiedVectorFormat sdata;
	states.ToUnifiedFormat(count, sdata);
	auto state_ptrs = UnifiedVectorFormat::GetData<PacCountState *>(sdata);

	for (idx_t i = 0; i < count; i++) {
		auto input_idx = idata.sel->get_index(i);
		if (!idata.validity.RowIsValid(input_idx)) {
			continue;
		}

		auto state_idx = sdata.sel->get_index(i);
		auto state = state_ptrs[state_idx];

		uint64_t key_hash = input_data[input_idx];
		for (int j = 0; j < 8; j++) {
			state->small_totals[j] += (key_hash >> j) & PAC_COUNT_MASK;
		}
		if (++state->update_count == 0) {
			state->Flush();
		}
	}
}

static void PacCountCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	auto sdata = FlatVector::GetData<PacCountState *>(source);
	auto tdata = FlatVector::GetData<PacCountState *>(target);
	for (idx_t i = 0; i < count; i++) {
		auto src = sdata[i];
		auto tgt = tdata[i];
		// Flush any unflushed small_totals
		if (src->update_count != 0) {
			src->Flush();
		}
		if (tgt->update_count != 0) {
			tgt->Flush();
		}
		for (int j = 0; j < 64; j++) {
			tgt->totals[j] += src->totals[j];
		}
	}
}

static void PacCountFinalize(Vector &states, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	auto sdata = FlatVector::GetData<PacCountState *>(states);
	auto rdata = FlatVector::GetData<double>(result);

	for (idx_t i = 0; i < count; i++) {
		auto state = sdata[i];
		if (state->update_count != 0) {
			state->Flush();
		}

		double sum = 0.0;
		for (int j = 0; j < 64; j++) {
			sum += static_cast<double>(state->totals[j]);
		}
		double mean = sum / 64.0;

		double variance = 0.0;
		for (int j = 0; j < 64; j++) {
			double diff = static_cast<double>(state->totals[j]) - mean;
			variance += diff * diff;
		}
		variance /= 64.0;

		rdata[offset + i] = variance;
	}
}

// ============================================================================
// pac_sum aggregate function
// ============================================================================
// State: 64 sums, one for each bit position
// Update: for each (key_hash, value), add value to sums[i] if bit i of key_hash is set
// Finalize: compute the variance of the 64 sums
//
// Templated to work with all numeric types (integers, floats, decimals)

struct PacSumState {
	double sums[64]; // 64 accumulators, one per bit position
};

static idx_t PacSumStateSize(const AggregateFunction &) {
	return sizeof(PacSumState);
}

static void PacSumInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	auto &state = *reinterpret_cast<PacSumState *>(state_ptr);
	memset(state.sums, 0, sizeof(state.sums));
}

// Helper to convert any numeric type to double
template <class T>
static inline double ToDouble(const T &val) {
	return static_cast<double>(val);
}

// Specialization for hugeint_t
template <>
inline double ToDouble<hugeint_t>(const hugeint_t &val) {
	return Hugeint::Cast<double>(val);
}

// Specialization for uhugeint_t
template <>
inline double ToDouble<uhugeint_t>(const uhugeint_t &val) {
	return Uhugeint::Cast<double>(val);
}

// Templated update function for pac_sum
template <class INPUT_TYPE>
static void PacSumUpdateImpl(Vector inputs[], AggregateInputData &, idx_t input_count, data_ptr_t state_ptr,
                             idx_t count) {
	D_ASSERT(input_count == 2);
	auto &state = *reinterpret_cast<PacSumState *>(state_ptr);

	// First input: key_hash (UBIGINT)
	UnifiedVectorFormat hash_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);

	// Second input: value (templated type)
	UnifiedVectorFormat value_data;
	inputs[1].ToUnifiedFormat(count, value_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);

	for (idx_t i = 0; i < count; i++) {
		auto hash_idx = hash_data.sel->get_index(i);
		auto value_idx = value_data.sel->get_index(i);

		// Skip if either is NULL
		if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
			continue;
		}

		uint64_t key_hash = hashes[hash_idx];
		double value = ToDouble<INPUT_TYPE>(values[value_idx]);

		// Add value to each sum where the corresponding bit is set
		for (int j = 0; j < 64; j++) {
			if ((key_hash >> j) & 1ULL) {
				state.sums[j] += value;
			}
		}
	}
}

// Specializations for different input types
static void PacSumUpdateTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                idx_t count) {
	PacSumUpdateImpl<int8_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                 idx_t count) {
	PacSumUpdateImpl<int16_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateInteger(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                idx_t count) {
	PacSumUpdateImpl<int32_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateBigInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                               idx_t count) {
	PacSumUpdateImpl<int64_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                idx_t count) {
	PacSumUpdateImpl<hugeint_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateUTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                 idx_t count) {
	PacSumUpdateImpl<uint8_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateUSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                  idx_t count) {
	PacSumUpdateImpl<uint16_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateUInteger(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                 idx_t count) {
	PacSumUpdateImpl<uint32_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateUBigInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                idx_t count) {
	PacSumUpdateImpl<uint64_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateUHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                                 idx_t count) {
	PacSumUpdateImpl<uhugeint_t>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateFloat(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                              idx_t count) {
	PacSumUpdateImpl<float>(inputs, aggr, input_count, state, count);
}
static void PacSumUpdateDouble(Vector inputs[], AggregateInputData &aggr, idx_t input_count, data_ptr_t state,
                               idx_t count) {
	PacSumUpdateImpl<double>(inputs, aggr, input_count, state, count);
}

static void PacSumCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	auto sdata = FlatVector::GetData<PacSumState *>(source);
	auto tdata = FlatVector::GetData<PacSumState *>(target);
	for (idx_t i = 0; i < count; i++) {
		for (int j = 0; j < 64; j++) {
			tdata[i]->sums[j] += sdata[i]->sums[j];
		}
	}
}

// Scatter update for pac_sum (used for grouped aggregation)
template <class INPUT_TYPE>
static void PacSumScatterUpdateImpl(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &states,
                                    idx_t count) {
	D_ASSERT(input_count == 2);

	// First input: key_hash (UBIGINT)
	UnifiedVectorFormat hash_data;
	inputs[0].ToUnifiedFormat(count, hash_data);
	auto hashes = UnifiedVectorFormat::GetData<uint64_t>(hash_data);

	// Second input: value (templated type)
	UnifiedVectorFormat value_data;
	inputs[1].ToUnifiedFormat(count, value_data);
	auto values = UnifiedVectorFormat::GetData<INPUT_TYPE>(value_data);

	// States vector
	UnifiedVectorFormat sdata;
	states.ToUnifiedFormat(count, sdata);
	auto state_ptrs = UnifiedVectorFormat::GetData<PacSumState *>(sdata);

	for (idx_t i = 0; i < count; i++) {
		auto hash_idx = hash_data.sel->get_index(i);
		auto value_idx = value_data.sel->get_index(i);
		auto state_idx = sdata.sel->get_index(i);

		// Skip if either input is NULL
		if (!hash_data.validity.RowIsValid(hash_idx) || !value_data.validity.RowIsValid(value_idx)) {
			continue;
		}

		auto state = state_ptrs[state_idx];
		uint64_t key_hash = hashes[hash_idx];
		double value = ToDouble<INPUT_TYPE>(values[value_idx]);

		// Add value to each sum where the corresponding bit is set
		for (int j = 0; j < 64; j++) {
			if ((key_hash >> j) & 1ULL) {
				state->sums[j] += value;
			}
		}
	}
}

// Scatter update specializations
static void PacSumScatterTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                 idx_t count) {
	PacSumScatterUpdateImpl<int8_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                  idx_t count) {
	PacSumScatterUpdateImpl<int16_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterInteger(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                 idx_t count) {
	PacSumScatterUpdateImpl<int32_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterBigInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                idx_t count) {
	PacSumScatterUpdateImpl<int64_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                 idx_t count) {
	PacSumScatterUpdateImpl<hugeint_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterUTinyInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                  idx_t count) {
	PacSumScatterUpdateImpl<uint8_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterUSmallInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                   idx_t count) {
	PacSumScatterUpdateImpl<uint16_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterUInteger(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                  idx_t count) {
	PacSumScatterUpdateImpl<uint32_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterUBigInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                 idx_t count) {
	PacSumScatterUpdateImpl<uint64_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterUHugeInt(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                  idx_t count) {
	PacSumScatterUpdateImpl<uhugeint_t>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterFloat(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                               idx_t count) {
	PacSumScatterUpdateImpl<float>(inputs, aggr, input_count, states, count);
}
static void PacSumScatterDouble(Vector inputs[], AggregateInputData &aggr, idx_t input_count, Vector &states,
                                idx_t count) {
	PacSumScatterUpdateImpl<double>(inputs, aggr, input_count, states, count);
}

static void PacSumFinalize(Vector &states, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	auto sdata = FlatVector::GetData<PacSumState *>(states);
	auto rdata = FlatVector::GetData<double>(result);

	for (idx_t i = 0; i < count; i++) {
		auto state = sdata[i];

		// Compute mean of the 64 sums
		double total = 0.0;
		for (int j = 0; j < 64; j++) {
			total += state->sums[j];
		}
		double mean = total / 64.0;

		// Compute variance
		double variance = 0.0;
		for (int j = 0; j < 64; j++) {
			double diff = state->sums[j] - mean;
			variance += diff * diff;
		}
		variance /= 64.0;

		rdata[offset + i] = variance;
	}
}

// Helper to create pac_sum aggregate function for a specific type
static AggregateFunction GetPacSumFunction(const LogicalType &input_type, aggregate_update_t scatter_update_func,
                                           aggregate_simple_update_t simple_update_func) {
	return AggregateFunction("pac_sum", {LogicalType::UBIGINT, input_type}, LogicalType::DOUBLE, PacSumStateSize,
	                         PacSumInitialize, scatter_update_func, PacSumCombine, PacSumFinalize,
	                         FunctionNullHandling::DEFAULT_NULL_HANDLING, simple_update_func);
}


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
	// Uses SIMD-friendly update with intermediate accumulators
	AggregateFunction pac_count("pac_count", {LogicalType::UBIGINT}, LogicalType::DOUBLE, PacCountStateSize,
	                            PacCountInitialize, PacCountScatterUpdate, PacCountCombine, PacCountFinalize,
	                            FunctionNullHandling::DEFAULT_NULL_HANDLING, PacCountUpdate);
	loader.RegisterFunction(pac_count);

	// Register pac_sum aggregate function set
	// Input: (UBIGINT key_hash, <numeric> value)
	// Output: DOUBLE (variance of the 64 sums)
	// Supports all numeric types that SUM supports
	AggregateFunctionSet pac_sum_set("pac_sum");

	// Signed integers
	pac_sum_set.AddFunction(GetPacSumFunction(LogicalType::TINYINT, PacSumScatterTinyInt, PacSumUpdateTinyInt));
	pac_sum_set.AddFunction(GetPacSumFunction(LogicalType::SMALLINT, PacSumScatterSmallInt, PacSumUpdateSmallInt));
	pac_sum_set.AddFunction(GetPacSumFunction(LogicalType::INTEGER, PacSumScatterInteger, PacSumUpdateInteger));
	pac_sum_set.AddFunction(GetPacSumFunction(LogicalType::BIGINT, PacSumScatterBigInt, PacSumUpdateBigInt));
	pac_sum_set.AddFunction(GetPacSumFunction(LogicalType::HUGEINT, PacSumScatterHugeInt, PacSumUpdateHugeInt));

	// Unsigned integers
	pac_sum_set.AddFunction(GetPacSumFunction(LogicalType::UTINYINT, PacSumScatterUTinyInt, PacSumUpdateUTinyInt));
	pac_sum_set.AddFunction(GetPacSumFunction(LogicalType::USMALLINT, PacSumScatterUSmallInt, PacSumUpdateUSmallInt));
	pac_sum_set.AddFunction(GetPacSumFunction(LogicalType::UINTEGER, PacSumScatterUInteger, PacSumUpdateUInteger));
	pac_sum_set.AddFunction(GetPacSumFunction(LogicalType::UBIGINT, PacSumScatterUBigInt, PacSumUpdateUBigInt));
	pac_sum_set.AddFunction(GetPacSumFunction(LogicalType::UHUGEINT, PacSumScatterUHugeInt, PacSumUpdateUHugeInt));

	// Floating point
	pac_sum_set.AddFunction(GetPacSumFunction(LogicalType::FLOAT, PacSumScatterFloat, PacSumUpdateFloat));
	pac_sum_set.AddFunction(GetPacSumFunction(LogicalType::DOUBLE, PacSumScatterDouble, PacSumUpdateDouble));

	loader.RegisterFunction(pac_sum_set);
}

} // namespace duckdb
