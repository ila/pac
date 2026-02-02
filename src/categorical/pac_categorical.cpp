//
// PAC Categorical Query Support - Implementation
//
// See pac_categorical.hpp for design documentation.
//
// Created by ila on 1/22/26.
//

#include "categorical/pac_categorical.hpp"
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
// Helper: Proportional noised select
// ============================================================================
// Returns true with probability proportional to popcount(mask)/count
// For count=64, this is equivalent to PacNoisedSelect
static inline bool PacNoisedSelectWithCount(uint64_t mask, uint64_t rnd, idx_t count) {
	if (count == 0) {
		return false;
	}
	int popcount = Popcount64(mask);
	// Probability = popcount / count
	// True if random value in [0, count) < popcount
	return (rnd % count) < static_cast<uint64_t>(popcount);
}

// ============================================================================
// Bind data for PAC categorical functions
// ============================================================================
struct PacCategoricalBindData : public FunctionData {
	double mi;
	uint64_t seed;

	explicit PacCategoricalBindData(double mi_val = 128.0, uint64_t seed_val = std::random_device {}())
	    : mi(mi_val), seed(seed_val) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<PacCategoricalBindData>(mi, seed);
	}

	bool Equals(const FunctionData &other) const override {
		auto &o = other.Cast<PacCategoricalBindData>();
		return mi == o.mi && seed == o.seed;
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
// When mi=0: deterministic majority voting (popcount >= 32)
static inline bool FilterFromMask(uint64_t mask, double mi, std::mt19937_64 &gen) {
	if (mi == 0.0) {
		return Popcount64(mask) >= 32;
	} else {
		return PacNoisedSelect(mask, gen());
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

	double mi = 128.0; // default
	auto &function = state.expr.Cast<BoundFunctionExpression>();
	if (function.bind_info) {
		auto &bind_data = function.bind_info->Cast<PacCategoricalBindData>();
		mi = bind_data.mi;
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

		result_data[i] = FilterFromMask(masks[mask_idx], mi, gen);
	}
}

// pac_filter(list<bool>) -> BOOLEAN
// Converts list to mask, then applies filter logic
static void PacFilterFromBoolListFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &list_vec = args.data[0];
	idx_t count = args.size();

	auto &local_state = ExecuteFunctionState::GetFunctionState(state)->Cast<PacCategoricalLocalState>();
	auto &gen = local_state.gen;

	double mi = 128.0; // default
	auto &function = state.expr.Cast<BoundFunctionExpression>();
	if (function.bind_info) {
		auto &bind_data = function.bind_info->Cast<PacCategoricalBindData>();
		mi = bind_data.mi;
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
		result_data[i] = FilterFromMask(mask, mi, gen);
	}
}

// pac_filter(UBIGINT mask, DOUBLE mi) -> BOOLEAN (explicit mi parameter)
static void PacFilterWithMiFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &mask_vec = args.data[0];
	auto &mi_vec = args.data[1];
	idx_t count = args.size();

	auto &local_state = ExecuteFunctionState::GetFunctionState(state)->Cast<PacCategoricalLocalState>();
	auto &gen = local_state.gen;

	UnifiedVectorFormat mask_data;
	mask_vec.ToUnifiedFormat(count, mask_data);
	auto masks = UnifiedVectorFormat::GetData<uint64_t>(mask_data);

	UnifiedVectorFormat mi_data;
	mi_vec.ToUnifiedFormat(count, mi_data);
	auto mis = UnifiedVectorFormat::GetData<double>(mi_data);

	auto result_data = FlatVector::GetData<bool>(result);
	auto &result_validity = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto mask_idx = mask_data.sel->get_index(i);
		auto mi_idx = mi_data.sel->get_index(i);

		if (!mask_data.validity.RowIsValid(mask_idx)) {
			result_validity.SetInvalid(i);
			continue;
		}

		double mi = mi_data.validity.RowIsValid(mi_idx) ? mis[mi_idx] : 128.0;
		result_data[i] = FilterFromMask(masks[mask_idx], mi, gen);
	}
}

// ============================================================================
// PAC_MASK_AND: Combine two masks with AND
// ============================================================================
static void PacMaskAndFunction(DataChunk &args, ExpressionState &state, Vector &result) {
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

		result_data[i] = masks1[idx1] & masks2[idx2];
	}
}

// ============================================================================
// PAC_MASK_OR: Combine two masks with OR
// ============================================================================
static void PacMaskOrFunction(DataChunk &args, ExpressionState &state, Vector &result) {
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

		result_data[i] = masks1[idx1] | masks2[idx2];
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
// Bind function for pac_filter/pac_select (reads pac_seed and pac_mi settings)
// ============================================================================
static unique_ptr<FunctionData> PacCategoricalBind(ClientContext &ctx, ScalarFunction &func,
                                                   vector<unique_ptr<Expression>> &args) {
	double mi = 128.0;
	uint64_t seed = std::random_device {}();

	// Try to get mi from session setting
	Value pac_mi_val;
	if (ctx.TryGetCurrentSetting("pac_mi", pac_mi_val) && !pac_mi_val.IsNull()) {
		mi = pac_mi_val.GetValue<double>();
	}

	// Try to get seed from session setting
	Value pac_seed_val;
	if (ctx.TryGetCurrentSetting("pac_seed", pac_seed_val) && !pac_seed_val.IsNull()) {
		seed = static_cast<uint64_t>(pac_seed_val.GetValue<int64_t>());
	}

#ifdef DEBUG
	Printer::Print("PacCategoricalBind: mi=" + std::to_string(mi) + ", seed=" + std::to_string(seed));
#endif

	return make_uniq<PacCategoricalBindData>(mi, seed);
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

	// pac_filter(UBIGINT mask, DOUBLE mi) -> BOOLEAN : With explicit mi parameter
	ScalarFunction pac_filter_mask_mi("pac_filter", {LogicalType::UBIGINT, LogicalType::DOUBLE}, LogicalType::BOOLEAN,
	                                  PacFilterWithMiFunction, PacCategoricalBind, nullptr, nullptr,
	                                  PacCategoricalInitLocal);
	loader.RegisterFunction(pac_filter_mask_mi);

	// pac_filter(list<bool>) -> BOOLEAN : Probabilistic filter from list (convenience)
	ScalarFunction pac_filter_list("pac_filter", {list_bool_type}, LogicalType::BOOLEAN, PacFilterFromBoolListFunction,
	                               PacCategoricalBind, nullptr, nullptr, PacCategoricalInitLocal);
	loader.RegisterFunction(pac_filter_list);

	// Mask combination functions (kept for potential future use)
	ScalarFunction pac_mask_and("pac_mask_and", {LogicalType::UBIGINT, LogicalType::UBIGINT}, LogicalType::UBIGINT,
	                            PacMaskAndFunction);
	loader.RegisterFunction(pac_mask_and);

	ScalarFunction pac_mask_or("pac_mask_or", {LogicalType::UBIGINT, LogicalType::UBIGINT}, LogicalType::UBIGINT,
	                           PacMaskOrFunction);
	loader.RegisterFunction(pac_mask_or);

	ScalarFunction pac_mask_not("pac_mask_not", {LogicalType::UBIGINT}, LogicalType::UBIGINT, PacMaskNotFunction);
	loader.RegisterFunction(pac_mask_not);
}

} // namespace duckdb
