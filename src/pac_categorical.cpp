//
// PAC Categorical Query Support - Implementation
//
// See pac_categorical.hpp for design documentation.
//
// Created by ila on 1/22/26.
//

#include "include/pac_categorical.hpp"
#include "include/pac_aggregate.hpp"

#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/vector_operations/generic_executor.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/execution/expression_executor.hpp"

#include <random>
#include <cmath>

namespace duckdb {

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
// PAC_GT: Compare scalar > counters, return 64-bit mask
// ============================================================================
// pac_gt(value, counters_list) -> UBIGINT mask
// bit i = 1 if value > counters[i]

static void PacGtFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &value_vec = args.data[0];
	auto &list_vec = args.data[1];
	idx_t count = args.size();

	UnifiedVectorFormat value_data;
	value_vec.ToUnifiedFormat(count, value_data);
	auto values = UnifiedVectorFormat::GetData<double>(value_data);

	UnifiedVectorFormat list_data;
	list_vec.ToUnifiedFormat(count, list_data);

	auto result_data = FlatVector::GetData<uint64_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(list_data);
	auto &child_vec = ListVector::GetEntry(list_vec);

	UnifiedVectorFormat child_data;
	child_vec.ToUnifiedFormat(ListVector::GetListSize(list_vec), child_data);
	auto child_values = UnifiedVectorFormat::GetData<double>(child_data);

	for (idx_t i = 0; i < count; i++) {
		auto value_idx = value_data.sel->get_index(i);
		auto list_idx = list_data.sel->get_index(i);

		if (!value_data.validity.RowIsValid(value_idx) || !list_data.validity.RowIsValid(list_idx)) {
			result_validity.SetInvalid(i);
			continue;
		}

		double val = values[value_idx];
		auto &entry = list_entries[list_idx];

		if (entry.length != 64) {
			throw InvalidInputException("pac_gt: counters list must have exactly 64 elements");
		}

		uint64_t mask = 0;
		for (idx_t j = 0; j < 64; j++) {
			auto child_idx = child_data.sel->get_index(entry.offset + j);
			if (child_data.validity.RowIsValid(child_idx)) {
				double counter_val = child_values[child_idx];
				if (val > counter_val) {
					mask |= (1ULL << j);
				}
			}
		}
		result_data[i] = mask;
	}
}

// ============================================================================
// PAC_LT: Compare scalar < counters, return 64-bit mask
// ============================================================================
static void PacLtFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &value_vec = args.data[0];
	auto &list_vec = args.data[1];
	idx_t count = args.size();

	UnifiedVectorFormat value_data;
	value_vec.ToUnifiedFormat(count, value_data);
	auto values = UnifiedVectorFormat::GetData<double>(value_data);

	UnifiedVectorFormat list_data;
	list_vec.ToUnifiedFormat(count, list_data);

	auto result_data = FlatVector::GetData<uint64_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(list_data);
	auto &child_vec = ListVector::GetEntry(list_vec);

	UnifiedVectorFormat child_data;
	child_vec.ToUnifiedFormat(ListVector::GetListSize(list_vec), child_data);
	auto child_values = UnifiedVectorFormat::GetData<double>(child_data);

	for (idx_t i = 0; i < count; i++) {
		auto value_idx = value_data.sel->get_index(i);
		auto list_idx = list_data.sel->get_index(i);

		if (!value_data.validity.RowIsValid(value_idx) || !list_data.validity.RowIsValid(list_idx)) {
			result_validity.SetInvalid(i);
			continue;
		}

		double val = values[value_idx];
		auto &entry = list_entries[list_idx];

		if (entry.length != 64) {
			throw InvalidInputException("pac_lt: counters list must have exactly 64 elements");
		}

		uint64_t mask = 0;
		for (idx_t j = 0; j < 64; j++) {
			auto child_idx = child_data.sel->get_index(entry.offset + j);
			if (child_data.validity.RowIsValid(child_idx)) {
				double counter_val = child_values[child_idx];
				if (val < counter_val) {
					mask |= (1ULL << j);
				}
			}
		}
		result_data[i] = mask;
	}
}

// ============================================================================
// PAC_GTE: Compare scalar >= counters, return 64-bit mask
// ============================================================================
static void PacGteFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &value_vec = args.data[0];
	auto &list_vec = args.data[1];
	idx_t count = args.size();

	UnifiedVectorFormat value_data;
	value_vec.ToUnifiedFormat(count, value_data);
	auto values = UnifiedVectorFormat::GetData<double>(value_data);

	UnifiedVectorFormat list_data;
	list_vec.ToUnifiedFormat(count, list_data);

	auto result_data = FlatVector::GetData<uint64_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(list_data);
	auto &child_vec = ListVector::GetEntry(list_vec);

	UnifiedVectorFormat child_data;
	child_vec.ToUnifiedFormat(ListVector::GetListSize(list_vec), child_data);
	auto child_values = UnifiedVectorFormat::GetData<double>(child_data);

	for (idx_t i = 0; i < count; i++) {
		auto value_idx = value_data.sel->get_index(i);
		auto list_idx = list_data.sel->get_index(i);

		if (!value_data.validity.RowIsValid(value_idx) || !list_data.validity.RowIsValid(list_idx)) {
			result_validity.SetInvalid(i);
			continue;
		}

		double val = values[value_idx];
		auto &entry = list_entries[list_idx];

		if (entry.length != 64) {
			throw InvalidInputException("pac_gte: counters list must have exactly 64 elements");
		}

		uint64_t mask = 0;
		for (idx_t j = 0; j < 64; j++) {
			auto child_idx = child_data.sel->get_index(entry.offset + j);
			if (child_data.validity.RowIsValid(child_idx)) {
				double counter_val = child_values[child_idx];
				if (val >= counter_val) {
					mask |= (1ULL << j);
				}
			}
		}
		result_data[i] = mask;
	}
}

// ============================================================================
// PAC_LTE: Compare scalar <= counters, return 64-bit mask
// ============================================================================
static void PacLteFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &value_vec = args.data[0];
	auto &list_vec = args.data[1];
	idx_t count = args.size();

	UnifiedVectorFormat value_data;
	value_vec.ToUnifiedFormat(count, value_data);
	auto values = UnifiedVectorFormat::GetData<double>(value_data);

	UnifiedVectorFormat list_data;
	list_vec.ToUnifiedFormat(count, list_data);

	auto result_data = FlatVector::GetData<uint64_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(list_data);
	auto &child_vec = ListVector::GetEntry(list_vec);

	UnifiedVectorFormat child_data;
	child_vec.ToUnifiedFormat(ListVector::GetListSize(list_vec), child_data);
	auto child_values = UnifiedVectorFormat::GetData<double>(child_data);

	for (idx_t i = 0; i < count; i++) {
		auto value_idx = value_data.sel->get_index(i);
		auto list_idx = list_data.sel->get_index(i);

		if (!value_data.validity.RowIsValid(value_idx) || !list_data.validity.RowIsValid(list_idx)) {
			result_validity.SetInvalid(i);
			continue;
		}

		double val = values[value_idx];
		auto &entry = list_entries[list_idx];

		if (entry.length != 64) {
			throw InvalidInputException("pac_lte: counters list must have exactly 64 elements");
		}

		uint64_t mask = 0;
		for (idx_t j = 0; j < 64; j++) {
			auto child_idx = child_data.sel->get_index(entry.offset + j);
			if (child_data.validity.RowIsValid(child_idx)) {
				double counter_val = child_values[child_idx];
				if (val <= counter_val) {
					mask |= (1ULL << j);
				}
			}
		}
		result_data[i] = mask;
	}
}

// ============================================================================
// PAC_EQ: Compare scalar == counters (approximate equality), return 64-bit mask
// ============================================================================
static void PacEqFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &value_vec = args.data[0];
	auto &list_vec = args.data[1];
	idx_t count = args.size();

	UnifiedVectorFormat value_data;
	value_vec.ToUnifiedFormat(count, value_data);
	auto values = UnifiedVectorFormat::GetData<double>(value_data);

	UnifiedVectorFormat list_data;
	list_vec.ToUnifiedFormat(count, list_data);

	auto result_data = FlatVector::GetData<uint64_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(list_data);
	auto &child_vec = ListVector::GetEntry(list_vec);

	UnifiedVectorFormat child_data;
	child_vec.ToUnifiedFormat(ListVector::GetListSize(list_vec), child_data);
	auto child_values = UnifiedVectorFormat::GetData<double>(child_data);

	for (idx_t i = 0; i < count; i++) {
		auto value_idx = value_data.sel->get_index(i);
		auto list_idx = list_data.sel->get_index(i);

		if (!value_data.validity.RowIsValid(value_idx) || !list_data.validity.RowIsValid(list_idx)) {
			result_validity.SetInvalid(i);
			continue;
		}

		double val = values[value_idx];
		auto &entry = list_entries[list_idx];

		if (entry.length != 64) {
			throw InvalidInputException("pac_eq: counters list must have exactly 64 elements");
		}

		uint64_t mask = 0;
		for (idx_t j = 0; j < 64; j++) {
			auto child_idx = child_data.sel->get_index(entry.offset + j);
			if (child_data.validity.RowIsValid(child_idx)) {
				double counter_val = child_values[child_idx];
				// Use approximate equality for floating point comparison
				if (val == counter_val) {
					mask |= (1ULL << j);
				}
			}
		}
		result_data[i] = mask;
	}
}

// ============================================================================
// PAC_NEQ: Compare scalar != counters, return 64-bit mask
// ============================================================================
static void PacNeqFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &value_vec = args.data[0];
	auto &list_vec = args.data[1];
	idx_t count = args.size();

	UnifiedVectorFormat value_data;
	value_vec.ToUnifiedFormat(count, value_data);
	auto values = UnifiedVectorFormat::GetData<double>(value_data);

	UnifiedVectorFormat list_data;
	list_vec.ToUnifiedFormat(count, list_data);

	auto result_data = FlatVector::GetData<uint64_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(list_data);
	auto &child_vec = ListVector::GetEntry(list_vec);

	UnifiedVectorFormat child_data;
	child_vec.ToUnifiedFormat(ListVector::GetListSize(list_vec), child_data);
	auto child_values = UnifiedVectorFormat::GetData<double>(child_data);

	for (idx_t i = 0; i < count; i++) {
		auto value_idx = value_data.sel->get_index(i);
		auto list_idx = list_data.sel->get_index(i);

		if (!value_data.validity.RowIsValid(value_idx) || !list_data.validity.RowIsValid(list_idx)) {
			result_validity.SetInvalid(i);
			continue;
		}

		double val = values[value_idx];
		auto &entry = list_entries[list_idx];

		if (entry.length != 64) {
			throw InvalidInputException("pac_neq: counters list must have exactly 64 elements");
		}

		uint64_t mask = 0;
		for (idx_t j = 0; j < 64; j++) {
			auto child_idx = child_data.sel->get_index(entry.offset + j);
			if (child_data.validity.RowIsValid(child_idx)) {
				double counter_val = child_values[child_idx];
				if (val != counter_val) {
					mask |= (1ULL << j);
				}
			}
		}
		result_data[i] = mask;
	}
}

// ============================================================================
// PAC_SELECT: Probabilistically select based on mask
// ============================================================================
// Returns true with probability proportional to popcount(mask)/64

struct PacSelectLocalState : public FunctionLocalState {
	std::mt19937_64 gen;
	explicit PacSelectLocalState(uint64_t seed) : gen(seed) {
	}
};

static unique_ptr<FunctionLocalState> PacSelectInitLocal(ExpressionState &state, const BoundFunctionExpression &expr,
                                                         FunctionData *bind_data) {
	uint64_t seed = std::random_device {}();
	if (bind_data) {
		seed = bind_data->Cast<PacCategoricalBindData>().seed;
	}
	return make_uniq<PacSelectLocalState>(seed);
}

static void PacSelectFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &mask_vec = args.data[0];
	idx_t count = args.size();

	auto &local_state = ExecuteFunctionState::GetFunctionState(state)->Cast<PacSelectLocalState>();
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

		uint64_t mask = masks[mask_idx];

		// If mi == 0, use deterministic selection (bit 0)
		if (mi == 0.0) {
			result_data[i] = (mask & 1ULL) != 0;
		} else {
			// Probabilistic selection based on popcount
			result_data[i] = PacNoisedSelect(mask, gen());
		}
	}
}

// Version with explicit mi parameter
static void PacSelectWithMiFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &mask_vec = args.data[0];
	auto &mi_vec = args.data[1];
	idx_t count = args.size();

	auto &local_state = ExecuteFunctionState::GetFunctionState(state)->Cast<PacSelectLocalState>();
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

		uint64_t mask = masks[mask_idx];
		double mi = mi_data.validity.RowIsValid(mi_idx) ? mis[mi_idx] : 128.0;

		// If mi == 0, use deterministic selection (bit 0)
		if (mi == 0.0) {
			result_data[i] = (mask & 1ULL) != 0;
		} else {
			// Probabilistic selection based on popcount
			result_data[i] = PacNoisedSelect(mask, gen());
		}
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
// PAC_SCALE_COUNTERS: Multiply each element of a counters list by a scalar
// Used in categorical queries where the aggregate is wrapped with arithmetic
// e.g., 0.5 * sum(...) becomes pac_scale_counters(0.5, pac_sum_counters(...))
// ============================================================================
static void PacScaleCountersFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &scale_vec = args.data[0];
	auto &counters_vec = args.data[1];
	idx_t count = args.size();

	UnifiedVectorFormat scale_data;
	scale_vec.ToUnifiedFormat(count, scale_data);
	auto scales = UnifiedVectorFormat::GetData<double>(scale_data);

	// Result is a list of doubles
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &result_validity = FlatVector::Validity(result);
	auto list_entries = FlatVector::GetData<list_entry_t>(result);
	auto &child_vector = ListVector::GetEntry(result);

	idx_t total_elements = 0;

	// First pass: count total elements needed
	for (idx_t i = 0; i < count; i++) {
		auto counters_child = ListVector::GetEntry(counters_vec);
		UnifiedVectorFormat counters_data;
		counters_vec.ToUnifiedFormat(count, counters_data);

		if (!counters_data.validity.RowIsValid(counters_data.sel->get_index(i))) {
			continue;
		}

		auto list_data = UnifiedVectorFormat::GetData<list_entry_t>(counters_data);
		auto idx = counters_data.sel->get_index(i);
		total_elements += list_data[idx].length;
	}

	ListVector::Reserve(result, total_elements);
	auto child_data = FlatVector::GetData<double>(child_vector);

	idx_t current_offset = 0;

	UnifiedVectorFormat counters_format;
	counters_vec.ToUnifiedFormat(count, counters_format);
	auto list_entries_input = UnifiedVectorFormat::GetData<list_entry_t>(counters_format);

	auto &input_child = ListVector::GetEntry(counters_vec);
	UnifiedVectorFormat input_child_data;
	input_child.ToUnifiedFormat(ListVector::GetListSize(counters_vec), input_child_data);
	auto input_values = UnifiedVectorFormat::GetData<double>(input_child_data);

	for (idx_t i = 0; i < count; i++) {
		auto scale_idx = scale_data.sel->get_index(i);
		auto counters_idx = counters_format.sel->get_index(i);

		if (!scale_data.validity.RowIsValid(scale_idx) || !counters_format.validity.RowIsValid(counters_idx)) {
			result_validity.SetInvalid(i);
			list_entries[i].offset = current_offset;
			list_entries[i].length = 0;
			continue;
		}

		double scale = scales[scale_idx];
		auto &input_entry = list_entries_input[counters_idx];

		list_entries[i].offset = current_offset;
		list_entries[i].length = input_entry.length;

		// Scale each element
		for (idx_t j = 0; j < input_entry.length; j++) {
			auto input_idx = input_child_data.sel->get_index(input_entry.offset + j);
			child_data[current_offset + j] = scale * input_values[input_idx];
		}

		current_offset += input_entry.length;
	}

	ListVector::SetListSize(result, current_offset);
}

// ============================================================================
// Bind function for pac_select (reads pac_seed setting)
// ============================================================================
static unique_ptr<FunctionData> PacSelectBind(ClientContext &ctx, ScalarFunction &func,
                                              vector<unique_ptr<Expression>> &args) {
	double mi = 128.0;
	uint64_t seed = std::random_device {}();

	// Try to get seed from session setting
	Value pac_seed_val;
	if (ctx.TryGetCurrentSetting("pac_seed", pac_seed_val) && !pac_seed_val.IsNull()) {
		seed = static_cast<uint64_t>(pac_seed_val.GetValue<int64_t>());
	}

	return make_uniq<PacCategoricalBindData>(mi, seed);
}

// ============================================================================
// Registration
// ============================================================================
void RegisterPacCategoricalFunctions(ExtensionLoader &loader) {
	// Comparison functions: pac_gt, pac_lt, pac_gte, pac_lte
	// All take (DOUBLE, LIST<DOUBLE>) and return UBIGINT mask

	auto list_double_type = LogicalType::LIST(LogicalType::DOUBLE);

	// pac_gt(value, counters) -> mask
	ScalarFunction pac_gt("pac_gt", {LogicalType::DOUBLE, list_double_type}, LogicalType::UBIGINT, PacGtFunction);
	loader.RegisterFunction(pac_gt);

	// pac_lt(value, counters) -> mask
	ScalarFunction pac_lt("pac_lt", {LogicalType::DOUBLE, list_double_type}, LogicalType::UBIGINT, PacLtFunction);
	loader.RegisterFunction(pac_lt);

	// pac_gte(value, counters) -> mask
	ScalarFunction pac_gte("pac_gte", {LogicalType::DOUBLE, list_double_type}, LogicalType::UBIGINT, PacGteFunction);
	loader.RegisterFunction(pac_gte);

	// pac_lte(value, counters) -> mask
	ScalarFunction pac_lte("pac_lte", {LogicalType::DOUBLE, list_double_type}, LogicalType::UBIGINT, PacLteFunction);
	loader.RegisterFunction(pac_lte);

	// pac_eq(value, counters) -> mask
	ScalarFunction pac_eq("pac_eq", {LogicalType::DOUBLE, list_double_type}, LogicalType::UBIGINT, PacEqFunction);
	loader.RegisterFunction(pac_eq);

	// pac_neq(value, counters) -> mask
	ScalarFunction pac_neq("pac_neq", {LogicalType::DOUBLE, list_double_type}, LogicalType::UBIGINT, PacNeqFunction);
	loader.RegisterFunction(pac_neq);

	// pac_select(mask) -> bool
	ScalarFunction pac_select_1("pac_select", {LogicalType::UBIGINT}, LogicalType::BOOLEAN, PacSelectFunction,
	                            PacSelectBind, nullptr, nullptr, PacSelectInitLocal);
	loader.RegisterFunction(pac_select_1);

	// pac_select(mask, mi) -> bool
	ScalarFunction pac_select_2("pac_select", {LogicalType::UBIGINT, LogicalType::DOUBLE}, LogicalType::BOOLEAN,
	                            PacSelectWithMiFunction, PacSelectBind, nullptr, nullptr, PacSelectInitLocal);
	loader.RegisterFunction(pac_select_2);

	// Mask combination functions
	// pac_mask_and(mask1, mask2) -> mask
	ScalarFunction pac_mask_and("pac_mask_and", {LogicalType::UBIGINT, LogicalType::UBIGINT}, LogicalType::UBIGINT,
	                            PacMaskAndFunction);
	loader.RegisterFunction(pac_mask_and);

	// pac_mask_or(mask1, mask2) -> mask
	ScalarFunction pac_mask_or("pac_mask_or", {LogicalType::UBIGINT, LogicalType::UBIGINT}, LogicalType::UBIGINT,
	                           PacMaskOrFunction);
	loader.RegisterFunction(pac_mask_or);

	// pac_mask_not(mask) -> mask
	ScalarFunction pac_mask_not("pac_mask_not", {LogicalType::UBIGINT}, LogicalType::UBIGINT, PacMaskNotFunction);
	loader.RegisterFunction(pac_mask_not);

	// pac_scale_counters(scale, counters) -> scaled_counters
	ScalarFunction pac_scale_counters("pac_scale_counters", {LogicalType::DOUBLE, list_double_type}, list_double_type,
	                                  PacScaleCountersFunction);
	loader.RegisterFunction(pac_scale_counters);
}

} // namespace duckdb
