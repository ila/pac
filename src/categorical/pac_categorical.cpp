//
// PAC Categorical Query Support - Implementation
//
// See pac_categorical.hpp for design documentation.
//
// Created by ila on 1/22/26.
//

#include "categorical/pac_categorical.hpp"
#include "pac_debug.hpp"
#include "aggregates/pac_aggregate.hpp"

#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/vector_operations/generic_executor.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/execution/expression_executor.hpp"

#include <random>
#include <cmath>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace duckdb {

// ============================================================================
// Cross-platform popcount helper
// ============================================================================
static inline int Popcount64(uint64_t x) {
#ifdef _MSC_VER
	return static_cast<int>(__popcnt64(x));
#else
	return __builtin_popcountll(x);
#endif
}

// ============================================================================
// Bind data for PAC categorical functions
// ============================================================================
struct PacCategoricalBindData : public FunctionData {
	double mi;
	double correction;
	uint64_t seed;
	uint64_t counter_selector; // per-query deterministic counter index for NoisySample

	explicit PacCategoricalBindData(double mi_val = 0.0, double correction_val = 1.0,
	                                uint64_t seed_val = std::random_device {}())
	    : mi(mi_val), correction(correction_val), seed(seed_val), counter_selector(seed_val * PAC_MAGIC_HASH) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<PacCategoricalBindData>(mi, correction, seed);
		copy->counter_selector = counter_selector;
		return copy;
	}

	bool Equals(const FunctionData &other) const override {
		auto &o = other.Cast<PacCategoricalBindData>();
		return mi == o.mi && correction == o.correction && seed == o.seed && counter_selector == o.counter_selector;
	}
};

// ============================================================================
// Local state for PAC categorical functions (RNG)
// ============================================================================
struct PacCategoricalLocalState : public FunctionLocalState {
	std::mt19937_64 gen;
	explicit PacCategoricalLocalState(uint64_t seed) : gen(seed) {
	}
};

static unique_ptr<FunctionLocalState>
PacCategoricalInitLocal(ExpressionState &state, const BoundFunctionExpression &expr, FunctionData *bind_data) {
	uint64_t seed = std::random_device {}();
	if (bind_data) {
		seed = bind_data->Cast<PacCategoricalBindData>().seed;
	}
	return make_uniq<PacCategoricalLocalState>(seed);
}

// ============================================================================
// Helper: Convert list<bool> to UBIGINT mask
// ============================================================================
// NULL and false both result in bit=0, true results in bit=1
static inline uint64_t BoolListToMask(UnifiedVectorFormat &list_data, UnifiedVectorFormat &child_data,
                                      const bool *child_values, idx_t list_idx) {
	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(list_data);
	auto &entry = list_entries[list_idx];

	uint64_t mask = 0;
	idx_t len = entry.length > 64 ? 64 : entry.length; // cap at 64
	for (idx_t j = 0; j < len; j++) {
		auto child_idx = child_data.sel->get_index(entry.offset + j);
		// Only set bit if valid AND true; NULL is treated as false
		if (child_data.validity.RowIsValid(child_idx) && child_values[child_idx]) {
			mask |= (1ULL << j);
		}
	}
	return mask;
}

// ============================================================================
// Helper: Common filter logic for mask-based filtering
// ============================================================================
// Returns true with probability proportional to popcount(mask)/64
// mi: controls probabilistic (mi>0) vs deterministic (mi<=0) mode
// correction: considers correction times more non-nulls (increases true probability)
// When mi <= 0: deterministic, true when popcount(mask) * correction >= 32
// When mi > 0: probabilistic, P(true) = popcount(mask) * correction / 64
static inline bool FilterFromMask(uint64_t mask, double mi, double correction, std::mt19937_64 &gen) {
	int popcount = Popcount64(mask);
	double effective_popcount = popcount * correction;

	if (mi <= 0.0) {
		// Deterministic mode: true when effective_popcount >= 32
		return effective_popcount >= 32.0;
	} else {
		// Probabilistic mode: P(true) = effective_popcount / 64
		// Equivalently: true if popcount > threshold, where threshold is in [0, 64/correction)
		uint64_t range = static_cast<uint64_t>(64.0 / correction);
		if (range == 0) {
			return true; // correction is very large, always return true
		}
		int threshold = static_cast<int>(gen() % range);
		return popcount > threshold;
	}
}

// ============================================================================
// PAC_SELECT: Convert list<bool> to UBIGINT mask
// ============================================================================
// pac_select(list<bool>) -> UBIGINT
// Converts a list of booleans to a bitmask where true=1, false/NULL=0
static void PacSelectFromBoolListFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &list_vec = args.data[0];
	idx_t count = args.size();

	UnifiedVectorFormat list_data;
	list_vec.ToUnifiedFormat(count, list_data);

	auto result_data = FlatVector::GetData<uint64_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	auto &child_vec = ListVector::GetEntry(list_vec);
	UnifiedVectorFormat child_data;
	child_vec.ToUnifiedFormat(ListVector::GetListSize(list_vec), child_data);
	auto child_values = UnifiedVectorFormat::GetData<bool>(child_data);

	for (idx_t i = 0; i < count; i++) {
		auto list_idx = list_data.sel->get_index(i);

		if (!list_data.validity.RowIsValid(list_idx)) {
			// NULL list -> NULL mask
			result_validity.SetInvalid(i);
			continue;
		}

		auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(list_data);
		auto &entry = list_entries[list_idx];

		// Empty list -> mask of 0
		if (entry.length == 0) {
			result_data[i] = 0;
			continue;
		}

		result_data[i] = BoolListToMask(list_data, child_data, child_values, list_idx);
	}
}

// ============================================================================
// PAC_FILTER: Probabilistically filter based on mask
// ============================================================================
// pac_filter(UBIGINT mask) -> BOOLEAN
// Returns true with probability proportional to popcount(mask)/64
static void PacFilterFromMaskFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &mask_vec = args.data[0];
	idx_t count = args.size();

	auto &local_state = ExecuteFunctionState::GetFunctionState(state)->Cast<PacCategoricalLocalState>();
	auto &gen = local_state.gen;

	double mi = 0.0;
	double correction = 1.0;
	auto &function = state.expr.Cast<BoundFunctionExpression>();
	if (function.bind_info) {
		auto &bind_data = function.bind_info->Cast<PacCategoricalBindData>();
		mi = bind_data.mi;
		correction = bind_data.correction;
	}

	UnifiedVectorFormat mask_data;
	mask_vec.ToUnifiedFormat(count, mask_data);
	auto masks = UnifiedVectorFormat::GetData<uint64_t>(mask_data);

	auto result_data = FlatVector::GetData<bool>(result);
	auto &result_validity = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto mask_idx = mask_data.sel->get_index(i);

		if (!mask_data.validity.RowIsValid(mask_idx)) {
			result_validity.SetInvalid(i);
			continue;
		}

		result_data[i] = FilterFromMask(masks[mask_idx], mi, correction, gen);
	}
}

// pac_filter(list<bool>) -> BOOLEAN
// Converts list to mask, then applies filter logic
static void PacFilterFromBoolListFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &list_vec = args.data[0];
	idx_t count = args.size();

	auto &local_state = ExecuteFunctionState::GetFunctionState(state)->Cast<PacCategoricalLocalState>();
	auto &gen = local_state.gen;

	double mi = 0.0;
	double correction = 1.0;
	auto &function = state.expr.Cast<BoundFunctionExpression>();
	if (function.bind_info) {
		auto &bind_data = function.bind_info->Cast<PacCategoricalBindData>();
		mi = bind_data.mi;
		correction = bind_data.correction;
	}

	UnifiedVectorFormat list_data;
	list_vec.ToUnifiedFormat(count, list_data);

	auto result_data = FlatVector::GetData<bool>(result);

	auto &child_vec = ListVector::GetEntry(list_vec);
	UnifiedVectorFormat child_data;
	child_vec.ToUnifiedFormat(ListVector::GetListSize(list_vec), child_data);
	auto child_values = UnifiedVectorFormat::GetData<bool>(child_data);

	for (idx_t i = 0; i < count; i++) {
		auto list_idx = list_data.sel->get_index(i);

		if (!list_data.validity.RowIsValid(list_idx)) {
			// NULL list -> return false
			result_data[i] = false;
			continue;
		}

		auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(list_data);
		auto &entry = list_entries[list_idx];

		// Empty list -> return false
		if (entry.length == 0) {
			result_data[i] = false;
			continue;
		}

		uint64_t mask = BoolListToMask(list_data, child_data, child_values, list_idx);
		result_data[i] = FilterFromMask(mask, mi, correction, gen);
	}
}

// pac_filter(UBIGINT mask, DOUBLE correction) -> BOOLEAN (explicit correction parameter)
static void PacFilterWithCorrectionFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &mask_vec = args.data[0];
	auto &correction_vec = args.data[1];
	idx_t count = args.size();

	auto &local_state = ExecuteFunctionState::GetFunctionState(state)->Cast<PacCategoricalLocalState>();
	auto &gen = local_state.gen;

	// Get mi from bind data
	double mi = 0.0;
	auto &function = state.expr.Cast<BoundFunctionExpression>();
	if (function.bind_info) {
		mi = function.bind_info->Cast<PacCategoricalBindData>().mi;
	}

	UnifiedVectorFormat mask_data;
	mask_vec.ToUnifiedFormat(count, mask_data);
	auto masks = UnifiedVectorFormat::GetData<uint64_t>(mask_data);

	UnifiedVectorFormat correction_data;
	correction_vec.ToUnifiedFormat(count, correction_data);
	auto corrections = UnifiedVectorFormat::GetData<double>(correction_data);

	auto result_data = FlatVector::GetData<bool>(result);
	auto &result_validity = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto mask_idx = mask_data.sel->get_index(i);
		auto correction_idx = correction_data.sel->get_index(i);

		if (!mask_data.validity.RowIsValid(mask_idx)) {
			result_validity.SetInvalid(i);
			continue;
		}

		double correction = correction_data.validity.RowIsValid(correction_idx) ? corrections[correction_idx] : 1.0;
		result_data[i] = FilterFromMask(masks[mask_idx], mi, correction, gen);
	}
}

// ============================================================================
// PAC_MASK_AND / PAC_MASK_OR: Combine two masks with binary operation
// ============================================================================
enum class MaskBinaryOp { AND, OR };

template <MaskBinaryOp OP>
static void PacMaskBinaryFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &mask1_vec = args.data[0];
	auto &mask2_vec = args.data[1];
	idx_t count = args.size();

	UnifiedVectorFormat mask1_data, mask2_data;
	mask1_vec.ToUnifiedFormat(count, mask1_data);
	mask2_vec.ToUnifiedFormat(count, mask2_data);
	auto masks1 = UnifiedVectorFormat::GetData<uint64_t>(mask1_data);
	auto masks2 = UnifiedVectorFormat::GetData<uint64_t>(mask2_data);

	auto result_data = FlatVector::GetData<uint64_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto idx1 = mask1_data.sel->get_index(i);
		auto idx2 = mask2_data.sel->get_index(i);

		if (!mask1_data.validity.RowIsValid(idx1) || !mask2_data.validity.RowIsValid(idx2)) {
			result_validity.SetInvalid(i);
			continue;
		}

		if (OP == MaskBinaryOp::AND) {
			result_data[i] = masks1[idx1] & masks2[idx2];
		} else {
			result_data[i] = masks1[idx1] | masks2[idx2];
		}
	}
}

// ============================================================================
// PAC_MASK_NOT: Negate a mask
// ============================================================================
static void PacMaskNotFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &mask_vec = args.data[0];
	idx_t count = args.size();

	UnifiedVectorFormat mask_data;
	mask_vec.ToUnifiedFormat(count, mask_data);
	auto masks = UnifiedVectorFormat::GetData<uint64_t>(mask_data);

	auto result_data = FlatVector::GetData<uint64_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto idx = mask_data.sel->get_index(i);

		if (!mask_data.validity.RowIsValid(idx)) {
			result_validity.SetInvalid(i);
			continue;
		}

		result_data[i] = ~masks[idx];
	}
}

// ============================================================================
// PAC_NOISED: Apply noise to a list of 64 counter values
// ============================================================================
// pac_noised(list<double> counters) -> DOUBLE
// Takes a list of 64 counter values, reconstructs key_hash from NULL/non-NULL pattern,
// and returns a single noised value using PacNoisySampleFrom64Counters.
// This is essentially what pac_sum/avg/count/min/max aggregates do in their finalize.
static void PacNoisedFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &list_vec = args.data[0];
	idx_t count = args.size();

	// Get mi and correction from bind data
	double mi = 0.0;
	double correction = 1.0;
	uint64_t seed = 0;
	uint64_t counter_selector = 0;
	auto &function = state.expr.Cast<BoundFunctionExpression>();
	if (function.bind_info) {
		auto &bind_data = function.bind_info->Cast<PacCategoricalBindData>();
		mi = bind_data.mi;
		correction = bind_data.correction;
		seed = bind_data.seed;
		counter_selector = bind_data.counter_selector;
	}

	UnifiedVectorFormat list_data;
	list_vec.ToUnifiedFormat(count, list_data);

	auto result_data = FlatVector::GetData<double>(result);
	auto &result_validity = FlatVector::Validity(result);

	auto &child_vec = ListVector::GetEntry(list_vec);
	UnifiedVectorFormat child_data;
	child_vec.ToUnifiedFormat(ListVector::GetListSize(list_vec), child_data);
	auto child_values = UnifiedVectorFormat::GetData<double>(child_data);

	for (idx_t i = 0; i < count; i++) {
		auto list_idx = list_data.sel->get_index(i);

		if (!list_data.validity.RowIsValid(list_idx)) {
			// NULL list -> NULL result
			result_validity.SetInvalid(i);
			continue;
		}

		auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(list_data);
		auto &entry = list_entries[list_idx];

		// Must have exactly 64 elements
		if (entry.length != 64) {
			result_validity.SetInvalid(i);
			continue;
		}

		// Reconstruct key_hash from NULL pattern and extract counter values
		uint64_t key_hash = 0;
		double counters[64];
		for (idx_t j = 0; j < 64; j++) {
			auto child_idx = child_data.sel->get_index(entry.offset + j);
			if (child_data.validity.RowIsValid(child_idx)) {
				// Non-NULL: set bit in key_hash and store value
				key_hash |= (1ULL << j);
				counters[j] = child_values[child_idx];
			} else {
				// NULL: bit stays 0, value doesn't matter (will be filtered out)
				counters[j] = 0.0;
			}
		}

		// If no valid counters, return NULL
		if (key_hash == 0) {
			result_validity.SetInvalid(i);
			continue;
		}

		// Use per-row deterministic RNG seeded by both seed and key_hash
		std::mt19937_64 gen(seed ^ key_hash);

		// Check if we should return NULL based on key_hash (uses mi and correction)
		if (PacNoiseInNull(key_hash, mi, correction, gen)) {
			result_validity.SetInvalid(i);
			continue;
		}

		// Get noised sample from counters
		// Note: No 2x multiplier here because _counters functions already apply it
		double noised = PacNoisySampleFrom64Counters(counters, mi, correction, gen, true, ~key_hash, counter_selector);
		result_data[i] = noised;
	}
}

// ============================================================================
// Bind function for pac_filter/pac_select (reads pac_seed and pac_mi settings)
// ============================================================================
static unique_ptr<FunctionData> PacCategoricalBind(ClientContext &ctx, ScalarFunction &func,
                                                   vector<unique_ptr<Expression>> &args) {
	// Read mi from pac_mi setting
	double mi = GetPacMiFromSetting(ctx);
	// Default correction is 1.0 (can be overridden via explicit parameter in some functions)
	double correction = 1.0;
	uint64_t seed = std::random_device {}();

	// Try to get seed from session setting
	Value pac_seed_val;
	if (ctx.TryGetCurrentSetting("pac_seed", pac_seed_val) && !pac_seed_val.IsNull()) {
		seed = static_cast<uint64_t>(pac_seed_val.GetValue<int64_t>());
	}

#if PAC_DEBUG
	PAC_DEBUG_PRINT("PacCategoricalBind: mi=" + std::to_string(mi) + ", correction=" + std::to_string(correction) +
	                ", seed=" + std::to_string(seed));
#endif

	auto result = make_uniq<PacCategoricalBindData>(mi, correction, seed);
	result->counter_selector = GetQueryCounterSelector(ctx, seed);
	// When mi > 0, also vary the seed used for per-row RNG so subsequent queries diverge
	// (When mi == 0, keep seed-based behavior for deterministic test reproducibility)
	return result;
}

// ============================================================================
// PAC List Aggregates: Aggregate over LIST<DOUBLE> inputs element-wise
// ============================================================================
// These aggregates take LIST<DOUBLE> inputs (from PAC _counters results) and
// produce LIST<DOUBLE> outputs. They aggregate element-wise across all input lists.
//
// Design: Uses template specialization for SIMD-friendly code generation.
// Each aggregate type has specialized ops that compile to tight vectorizable loops.

enum class PacListAggType { SUM, AVG, COUNT, MIN, MAX };

struct PacListAggregateState {
	uint64_t key_hash;   // Bitmap: bit i = 1 if we've seen a non-null at position i
	double values[64];   // Accumulated values
	uint64_t counts[64]; // Count of non-null values (for avg/count)
};

// ============================================================================
// Template-specialized operations for each aggregate type
// These are designed to be inlined and auto-vectorized by the compiler
// ============================================================================

template <PacListAggType AGG_TYPE>
struct PacListOps {
	static constexpr double InitValue();
	static void UpdateDense(double *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT counts, const double *PAC_RESTRICT src,
	                        idx_t len);
	static void UpdateScatter(double *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT counts, const double *PAC_RESTRICT src,
	                          const idx_t *PAC_RESTRICT indices, const uint64_t *PAC_RESTRICT valid_mask, idx_t len);
	static void Combine(double *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT dst_counts, const double *PAC_RESTRICT src,
	                    const uint64_t *PAC_RESTRICT src_counts, uint64_t src_mask, uint64_t dst_mask);
	static double Finalize(double value, uint64_t count);
};

// SUM specialization
template <>
struct PacListOps<PacListAggType::SUM> {
	static constexpr double InitValue() {
		return 0.0;
	}
	static void UpdateDense(double *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const double *PAC_RESTRICT src,
	                        idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			dst[i] += src[i];
		}
	}
	static void UpdateScatter(double *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const double *PAC_RESTRICT src,
	                          const idx_t *PAC_RESTRICT indices, const uint64_t *PAC_RESTRICT, idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			dst[indices[i]] += src[i];
		}
	}
	static void Combine(double *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const double *PAC_RESTRICT src,
	                    const uint64_t *PAC_RESTRICT, uint64_t, uint64_t) {
		for (idx_t i = 0; i < 64; i++) {
			dst[i] += src[i];
		}
	}
	static double Finalize(double value, uint64_t) {
		return value;
	}
};

// AVG specialization
template <>
struct PacListOps<PacListAggType::AVG> {
	static constexpr double InitValue() {
		return 0.0;
	}
	static void UpdateDense(double *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT counts, const double *PAC_RESTRICT src,
	                        idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			dst[i] += src[i];
			counts[i]++;
		}
	}
	static void UpdateScatter(double *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT counts, const double *PAC_RESTRICT src,
	                          const idx_t *PAC_RESTRICT indices, const uint64_t *PAC_RESTRICT, idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			dst[indices[i]] += src[i];
			counts[indices[i]]++;
		}
	}
	static void Combine(double *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT dst_counts, const double *PAC_RESTRICT src,
	                    const uint64_t *PAC_RESTRICT src_counts, uint64_t, uint64_t) {
		for (idx_t i = 0; i < 64; i++) {
			dst[i] += src[i];
			dst_counts[i] += src_counts[i];
		}
	}
	static double Finalize(double value, uint64_t count) {
		return count > 0 ? value / static_cast<double>(count) : 0.0;
	}
};

// COUNT specialization
template <>
struct PacListOps<PacListAggType::COUNT> {
	static constexpr double InitValue() {
		return 0.0;
	}
	static void UpdateDense(double *PAC_RESTRICT, uint64_t *PAC_RESTRICT counts, const double *PAC_RESTRICT,
	                        idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			counts[i]++;
		}
	}
	static void UpdateScatter(double *PAC_RESTRICT, uint64_t *PAC_RESTRICT counts, const double *PAC_RESTRICT,
	                          const idx_t *PAC_RESTRICT indices, const uint64_t *PAC_RESTRICT, idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			counts[indices[i]]++;
		}
	}
	static void Combine(double *PAC_RESTRICT, uint64_t *PAC_RESTRICT dst_counts, const double *PAC_RESTRICT,
	                    const uint64_t *PAC_RESTRICT src_counts, uint64_t, uint64_t) {
		for (idx_t i = 0; i < 64; i++) {
			dst_counts[i] += src_counts[i];
		}
	}
	static double Finalize(double, uint64_t count) {
		return static_cast<double>(count);
	}
};

// MIN specialization
template <>
struct PacListOps<PacListAggType::MIN> {
	static constexpr double InitValue() {
		return std::numeric_limits<double>::max();
	}
	static void UpdateDense(double *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const double *PAC_RESTRICT src,
	                        idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			dst[i] = std::min(dst[i], src[i]);
		}
	}
	static void UpdateScatter(double *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const double *PAC_RESTRICT src,
	                          const idx_t *PAC_RESTRICT indices, const uint64_t *PAC_RESTRICT, idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			dst[indices[i]] = std::min(dst[indices[i]], src[i]);
		}
	}
	static void Combine(double *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const double *PAC_RESTRICT src,
	                    const uint64_t *PAC_RESTRICT, uint64_t src_mask, uint64_t dst_mask) {
		// For MIN: take source if target doesn't have the value yet, or if source is smaller
		for (idx_t i = 0; i < 64; i++) {
			bool src_has = (src_mask & (1ULL << i)) != 0;
			bool dst_has = (dst_mask & (1ULL << i)) != 0;
			if (src_has && (!dst_has || src[i] < dst[i])) {
				dst[i] = src[i];
			}
		}
	}
	static double Finalize(double value, uint64_t) {
		return value;
	}
};

// MAX specialization
template <>
struct PacListOps<PacListAggType::MAX> {
	static constexpr double InitValue() {
		return std::numeric_limits<double>::lowest();
	}
	static void UpdateDense(double *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const double *PAC_RESTRICT src,
	                        idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			dst[i] = std::max(dst[i], src[i]);
		}
	}
	static void UpdateScatter(double *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const double *PAC_RESTRICT src,
	                          const idx_t *PAC_RESTRICT indices, const uint64_t *PAC_RESTRICT, idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			dst[indices[i]] = std::max(dst[indices[i]], src[i]);
		}
	}
	static void Combine(double *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const double *PAC_RESTRICT src,
	                    const uint64_t *PAC_RESTRICT, uint64_t src_mask, uint64_t dst_mask) {
		// For MAX: take source if target doesn't have the value yet, or if source is larger
		for (idx_t i = 0; i < 64; i++) {
			bool src_has = (src_mask & (1ULL << i)) != 0;
			bool dst_has = (dst_mask & (1ULL << i)) != 0;
			if (src_has && (!dst_has || src[i] > dst[i])) {
				dst[i] = src[i];
			}
		}
	}
	static double Finalize(double value, uint64_t) {
		return value;
	}
};

// ============================================================================
// Aggregate function implementations using the specialized ops
// ============================================================================

template <PacListAggType AGG_TYPE>
static void PacListAggregateInit(const AggregateFunction &, data_ptr_t state_ptr) {
	auto &state = *reinterpret_cast<PacListAggregateState *>(state_ptr);
	state.key_hash = 0;
	double init_val = PacListOps<AGG_TYPE>::InitValue();
	for (idx_t i = 0; i < 64; i++) {
		state.values[i] = init_val;
		state.counts[i] = 0;
	}
}

// Dense update: when child vector is flat and contiguous (no validity gaps)
template <PacListAggType AGG_TYPE>
static void PacListAggregateUpdateDense(PacListAggregateState &state, const double *child_values, idx_t offset,
                                        idx_t len) {
	state.key_hash |= (len == 64) ? ~0ULL : ((1ULL << len) - 1);
	PacListOps<AGG_TYPE>::UpdateDense(state.values, state.counts, child_values + offset, len);
}

// Scatter update: when child has validity gaps or non-contiguous access
template <PacListAggType AGG_TYPE>
static void PacListAggregateUpdateScatter(PacListAggregateState &state, const double *child_values,
                                          UnifiedVectorFormat &child_data, idx_t offset, idx_t len) {
	// Temporary buffers for gathering valid values and their target indices
	double valid_values[64];
	idx_t valid_indices[64];
	idx_t valid_count = 0;

	for (idx_t j = 0; j < len; j++) {
		auto child_idx = child_data.sel->get_index(offset + j);
		if (child_data.validity.RowIsValid(child_idx)) {
			valid_values[valid_count] = child_values[child_idx];
			valid_indices[valid_count] = j;
			state.key_hash |= (1ULL << j);
			valid_count++;
		}
	}

	if (valid_count > 0) {
		PacListOps<AGG_TYPE>::UpdateScatter(state.values, state.counts, valid_values, valid_indices, nullptr,
		                                    valid_count);
	}
}

template <PacListAggType AGG_TYPE>
static void PacListAggregateUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector,
                                   idx_t count) {
	D_ASSERT(input_count == 1);
	auto &list_vec = inputs[0];

	UnifiedVectorFormat list_data;
	list_vec.ToUnifiedFormat(count, list_data);

	auto &child_vec = ListVector::GetEntry(list_vec);
	UnifiedVectorFormat child_data;
	child_vec.ToUnifiedFormat(ListVector::GetListSize(list_vec), child_data);
	auto child_values = UnifiedVectorFormat::GetData<double>(child_data);

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<PacListAggregateState **>(sdata.data);

	// Check if child vector is flat with all valid entries (enables dense path)
	bool child_is_flat = child_vec.GetVectorType() == VectorType::FLAT_VECTOR && child_data.validity.AllValid();

	for (idx_t i = 0; i < count; i++) {
		auto list_idx = list_data.sel->get_index(i);
		if (!list_data.validity.RowIsValid(list_idx)) {
			continue;
		}

		auto &state = *states[sdata.sel->get_index(i)];
		auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(list_data);
		auto &entry = list_entries[list_idx];
		idx_t len = entry.length > 64 ? 64 : entry.length;

		if (child_is_flat) {
			// Dense path: direct array access, fully vectorizable
			PacListAggregateUpdateDense<AGG_TYPE>(state, child_values, entry.offset, len);
		} else {
			// Scatter path: handle validity and indirection
			PacListAggregateUpdateScatter<AGG_TYPE>(state, child_values, child_data, entry.offset, len);
		}
	}
}

template <PacListAggType AGG_TYPE>
static void PacListAggregateCombine(Vector &source_vec, Vector &target_vec, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata, tdata;
	source_vec.ToUnifiedFormat(count, sdata);
	target_vec.ToUnifiedFormat(count, tdata);
	auto sources = reinterpret_cast<PacListAggregateState **>(sdata.data);
	auto targets = reinterpret_cast<PacListAggregateState **>(tdata.data);

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[sdata.sel->get_index(i)];
		auto &target = *targets[tdata.sel->get_index(i)];

		uint64_t src_mask = source.key_hash;
		uint64_t dst_mask = target.key_hash;
		target.key_hash |= src_mask;

		PacListOps<AGG_TYPE>::Combine(target.values, target.counts, source.values, source.counts, src_mask, dst_mask);
	}
}

template <PacListAggType AGG_TYPE>
static void PacListAggregateFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                                     idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<PacListAggregateState **>(sdata.data);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &result_validity = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto result_idx = i + offset;

		if (state.key_hash == 0) {
			result_validity.SetInvalid(result_idx);
			continue;
		}

		vector<Value> list_values;
		list_values.reserve(64);
		for (idx_t j = 0; j < 64; j++) {
			if (!(state.key_hash & (1ULL << j))) {
				list_values.push_back(Value());
			} else {
				list_values.push_back(Value::DOUBLE(PacListOps<AGG_TYPE>::Finalize(state.values[j], state.counts[j])));
			}
		}
		result.SetValue(result_idx, Value::LIST(LogicalType::DOUBLE, std::move(list_values)));
	}
}

static void PacListAggregateDestructor(Vector &, AggregateInputData &, idx_t) {
}

template <PacListAggType AGG_TYPE>
static AggregateFunction CreatePacListAggregate(const string &name) {
	auto list_double_type = LogicalType::LIST(LogicalType::DOUBLE);
	return AggregateFunction(name, {list_double_type}, list_double_type,
	                         AggregateFunction::StateSize<PacListAggregateState>, PacListAggregateInit<AGG_TYPE>,
	                         PacListAggregateUpdate<AGG_TYPE>, PacListAggregateCombine<AGG_TYPE>,
	                         PacListAggregateFinalize<AGG_TYPE>, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr,
	                         nullptr, PacListAggregateDestructor);
}

// ============================================================================
// Registration
// ============================================================================
void RegisterPacCategoricalFunctions(ExtensionLoader &loader) {
	auto list_bool_type = LogicalType::LIST(LogicalType::BOOLEAN);

	// pac_select(list<bool>) -> UBIGINT : Convert list of booleans to bitmask
	ScalarFunction pac_select_list("pac_select", {list_bool_type}, LogicalType::UBIGINT, PacSelectFromBoolListFunction,
	                               PacCategoricalBind);
	loader.RegisterFunction(pac_select_list);

	// pac_filter(UBIGINT mask) -> BOOLEAN : Probabilistic filter from mask
	ScalarFunction pac_filter_mask("pac_filter", {LogicalType::UBIGINT}, LogicalType::BOOLEAN,
	                               PacFilterFromMaskFunction, PacCategoricalBind, nullptr, nullptr,
	                               PacCategoricalInitLocal);
	loader.RegisterFunction(pac_filter_mask);

	// pac_filter(UBIGINT mask, DOUBLE correction) -> BOOLEAN : With explicit correction parameter
	ScalarFunction pac_filter_mask_correction("pac_filter", {LogicalType::UBIGINT, LogicalType::DOUBLE},
	                                          LogicalType::BOOLEAN, PacFilterWithCorrectionFunction, PacCategoricalBind,
	                                          nullptr, nullptr, PacCategoricalInitLocal);
	loader.RegisterFunction(pac_filter_mask_correction);

	// pac_filter(list<bool>) -> BOOLEAN : Probabilistic filter from list (convenience)
	ScalarFunction pac_filter_list("pac_filter", {list_bool_type}, LogicalType::BOOLEAN, PacFilterFromBoolListFunction,
	                               PacCategoricalBind, nullptr, nullptr, PacCategoricalInitLocal);
	loader.RegisterFunction(pac_filter_list);

	// Mask combination functions (kept for potential future use)
	ScalarFunction pac_mask_and("pac_mask_and", {LogicalType::UBIGINT, LogicalType::UBIGINT}, LogicalType::UBIGINT,
	                            PacMaskBinaryFunction<MaskBinaryOp::AND>);
	loader.RegisterFunction(pac_mask_and);

	ScalarFunction pac_mask_or("pac_mask_or", {LogicalType::UBIGINT, LogicalType::UBIGINT}, LogicalType::UBIGINT,
	                           PacMaskBinaryFunction<MaskBinaryOp::OR>);
	loader.RegisterFunction(pac_mask_or);

	ScalarFunction pac_mask_not("pac_mask_not", {LogicalType::UBIGINT}, LogicalType::UBIGINT, PacMaskNotFunction);
	loader.RegisterFunction(pac_mask_not);

	// pac_noised(list<double>) -> DOUBLE : Apply noise to 64 counter values
	auto list_double_type = LogicalType::LIST(LogicalType::DOUBLE);
	ScalarFunction pac_noised("pac_noised", {list_double_type}, LogicalType::DOUBLE, PacNoisedFunction,
	                          PacCategoricalBind, nullptr, nullptr, PacCategoricalInitLocal);
	loader.RegisterFunction(pac_noised);

	// List aggregates: aggregate over LIST<DOUBLE> inputs element-wise
	// These handle cases where PAC _counters results are used as input to another aggregate
	loader.RegisterFunction(CreatePacListAggregate<PacListAggType::SUM>("pac_sum_list"));
	loader.RegisterFunction(CreatePacListAggregate<PacListAggType::AVG>("pac_avg_list"));
	loader.RegisterFunction(CreatePacListAggregate<PacListAggType::COUNT>("pac_count_list"));
	loader.RegisterFunction(CreatePacListAggregate<PacListAggType::MIN>("pac_min_list"));
	loader.RegisterFunction(CreatePacListAggregate<PacListAggType::MAX>("pac_max_list"));
}

} // namespace duckdb
