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

namespace duckdb {

// ============================================================================
// Bind data for PAC categorical functions
// ============================================================================
struct PacCategoricalBindData : public FunctionData {
	double mi;
	double correction;
	uint64_t seed;
	uint64_t query_hash; // derived from seed: used as counter selector for NoisySample

	// Primary constructor - reads seed from pac_seed setting, or uses default 42 if not set.
	// When mi > 0, seed is randomized per query via the query number.
	explicit PacCategoricalBindData(ClientContext &ctx, double mi_val = 0.0, double correction_val = 1.0)
	    : mi(mi_val), correction(correction_val) {
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
		auto copy = make_uniq<PacCategoricalBindData>(*this); // uses implicit copy ctor (all fields are POD)
		return copy;
	}

	bool Equals(const FunctionData &other) const override {
		auto &o = other.Cast<PacCategoricalBindData>();
		return mi == o.mi && correction == o.correction && seed == o.seed && query_hash == o.query_hash;
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

// Forward declaration
static unique_ptr<FunctionData> PacCategoricalBind(ClientContext &ctx, ScalarFunction &func,
                                                   vector<unique_ptr<Expression>> &args);

// ============================================================================
// PAC_NOISED: Apply noise to a list of 64 counter values
// ============================================================================
// pac_noised(list<double> counters) -> DOUBLE
// Takes a list of 64 counter values, reconstructs key_hash from NULL/non-NULL pattern,
// and returns a single noised value using PacNoisySampleFrom64Counters.
// This is essentially what pac_sum/avg/count/min/max aggregates do in their finalize.
// pac_coalesce(LIST<DOUBLE>) -> LIST<DOUBLE>
// If the input list is NULL, returns a list of 64 NULL doubles.
// Otherwise returns the input unchanged. This is needed because COALESCE
// with a constant fallback list would have only 1 element, but pac_noised
// needs exactly 64 for noise sampling.
static void PacCoalesceFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input = args.data[0];
	idx_t count = args.size();

	UnifiedVectorFormat input_data;
	input.ToUnifiedFormat(count, input_data);

	// Fast path: if no NULL lists, zero-copy pass-through (common case)
	if (input_data.validity.AllValid()) {
		result.Reference(input);
		return;
	}

	// Slow path: some NULLs present — must materialize
	auto input_entries = UnifiedVectorFormat::GetData<list_entry_t>(input_data);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto list_entries = FlatVector::GetData<list_entry_t>(result);
	auto &child_vec = ListVector::GetEntry(result);

	// First pass: count total elements needed
	idx_t total_elements = 0;
	for (idx_t i = 0; i < count; i++) {
		auto idx = input_data.sel->get_index(i);
		if (input_data.validity.RowIsValid(idx)) {
			total_elements += input_entries[idx].length;
		} else {
			total_elements += 64; // will fill with NULLs
		}
	}

	ListVector::Reserve(result, total_elements);
	ListVector::SetListSize(result, total_elements);
	auto child_data = FlatVector::GetData<PAC_FLOAT>(child_vec);
	auto &child_validity = FlatVector::Validity(child_vec);
	child_validity.EnsureWritable();

	// Get the source child vector for non-NULL lists
	auto &input_child = ListVector::GetEntry(input);
	bool child_is_flat = (input_child.GetVectorType() == VectorType::FLAT_VECTOR);
	auto flat_src_data = child_is_flat ? FlatVector::GetData<PAC_FLOAT>(input_child) : nullptr;
	auto *flat_src_validity = child_is_flat ? &FlatVector::Validity(input_child) : nullptr;

	// Fallback for non-flat child
	UnifiedVectorFormat input_child_data;
	const PAC_FLOAT *src_data = nullptr;
	if (!child_is_flat) {
		input_child.ToUnifiedFormat(ListVector::GetListSize(input), input_child_data);
		src_data = UnifiedVectorFormat::GetData<PAC_FLOAT>(input_child_data);
	}

	idx_t offset = 0;
	for (idx_t i = 0; i < count; i++) {
		auto idx = input_data.sel->get_index(i);
		list_entries[i].offset = offset;

		if (input_data.validity.RowIsValid(idx)) {
			// Non-NULL: copy the input list
			auto &entry = input_entries[idx];
			list_entries[i].length = entry.length;

			if (child_is_flat) {
				// Bulk memcpy for flat child data
				memcpy(child_data + offset, flat_src_data + entry.offset, entry.length * sizeof(PAC_FLOAT));
				// Copy validity bits for this range
				for (idx_t j = 0; j < entry.length; j++) {
					if (!flat_src_validity->RowIsValid(entry.offset + j)) {
						child_validity.SetInvalid(offset + j);
					}
				}
			} else {
				// Non-flat child: element-by-element with selection vector
				for (idx_t j = 0; j < entry.length; j++) {
					auto src_idx = input_child_data.sel->get_index(entry.offset + j);
					if (input_child_data.validity.RowIsValid(src_idx)) {
						child_data[offset + j] = src_data[src_idx];
					} else {
						child_validity.SetInvalid(offset + j);
					}
				}
			}
			offset += entry.length;
		} else {
			// NULL input: produce 64 NULL PAC_FLOATs
			list_entries[i].length = 64;
			// 64 bits = 1 validity word — zero in one operation when aligned
			if (offset % ValidityMask::BITS_PER_VALUE == 0) {
				child_validity.GetData()[offset / ValidityMask::BITS_PER_VALUE] = 0;
			} else {
				for (idx_t j = 0; j < 64; j++) {
					child_validity.SetInvalid(offset + j);
				}
			}
			offset += 64;
		}
	}
}

static void PacNoisedFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &list_vec = args.data[0];
	idx_t count = args.size();

	// Get mi and correction from bind data
	double mi = 0.0;
	double correction = 1.0;
	uint64_t seed = 0;
	uint64_t query_hash = 0;
	auto &function = state.expr.Cast<BoundFunctionExpression>();
	if (function.bind_info) {
		auto &bind_data = function.bind_info->Cast<PacCategoricalBindData>();
		mi = bind_data.mi;
		correction = bind_data.correction;
		seed = bind_data.seed;
		query_hash = bind_data.query_hash;
	}

	UnifiedVectorFormat list_data;
	list_vec.ToUnifiedFormat(count, list_data);

	auto result_data = FlatVector::GetData<PAC_FLOAT>(result);
	auto &result_validity = FlatVector::Validity(result);

	auto &child_vec = ListVector::GetEntry(list_vec);
	UnifiedVectorFormat child_data;
	child_vec.ToUnifiedFormat(ListVector::GetListSize(list_vec), child_data);
	auto child_values = UnifiedVectorFormat::GetData<PAC_FLOAT>(child_data);

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
		PAC_FLOAT counters[64];
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
		PAC_FLOAT noised = PacNoisySampleFrom64Counters(counters, mi, correction, gen, true, ~key_hash, query_hash);
		result_data[i] = noised;
	}
}

// ============================================================================
// Bind function for pac_noised (reads pac_seed and pac_mi settings)
// ============================================================================
static unique_ptr<FunctionData> PacCategoricalBind(ClientContext &ctx, ScalarFunction &func,
                                                   vector<unique_ptr<Expression>> &args) {
	// Read mi from pac_mi setting
	double mi = GetPacMiFromSetting(ctx);
	// Default correction is 1.0 (can be overridden via explicit parameter in some functions)
	double correction = 1.0;

	auto result = make_uniq<PacCategoricalBindData>(ctx, mi, correction);

#if PAC_DEBUG
	PAC_DEBUG_PRINT("PacCategoricalBind: mi=" + std::to_string(mi) + ", correction=" + std::to_string(correction) +
	                ", seed=" + std::to_string(result->seed));
#endif

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
	uint64_t key_hash;    // Bitmap: bit i = 1 if we've seen a non-null at position i
	PAC_FLOAT values[64]; // Accumulated values
	uint64_t counts[64];  // Count of non-null values (for avg/count)
};

// ============================================================================
// Template-specialized operations for each aggregate type
// These are designed to be inlined and auto-vectorized by the compiler
// ============================================================================

template <PacListAggType AGG_TYPE>
struct PacListOps {
	static constexpr PAC_FLOAT InitValue();
	static void UpdateDense(PAC_FLOAT *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT counts,
	                        const PAC_FLOAT *PAC_RESTRICT src, idx_t len);
	static void UpdateScatter(PAC_FLOAT *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT counts,
	                          const PAC_FLOAT *PAC_RESTRICT src, const idx_t *PAC_RESTRICT indices,
	                          const uint64_t *PAC_RESTRICT valid_mask, idx_t len);
	static void Combine(PAC_FLOAT *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT dst_counts,
	                    const PAC_FLOAT *PAC_RESTRICT src, const uint64_t *PAC_RESTRICT src_counts, uint64_t src_mask,
	                    uint64_t dst_mask);
	static PAC_FLOAT Finalize(PAC_FLOAT value, uint64_t count);
};

// SUM specialization
template <>
struct PacListOps<PacListAggType::SUM> {
	static constexpr PAC_FLOAT InitValue() {
		return PAC_FLOAT(0);
	}
	static void UpdateDense(PAC_FLOAT *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const PAC_FLOAT *PAC_RESTRICT src,
	                        idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			dst[i] += src[i];
		}
	}
	static void UpdateScatter(PAC_FLOAT *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const PAC_FLOAT *PAC_RESTRICT src,
	                          const idx_t *PAC_RESTRICT indices, const uint64_t *PAC_RESTRICT, idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			dst[indices[i]] += src[i];
		}
	}
	static void Combine(PAC_FLOAT *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const PAC_FLOAT *PAC_RESTRICT src,
	                    const uint64_t *PAC_RESTRICT, uint64_t, uint64_t) {
		for (idx_t i = 0; i < 64; i++) {
			dst[i] += src[i];
		}
	}
	static PAC_FLOAT Finalize(PAC_FLOAT value, uint64_t) {
		return value;
	}
};

// AVG specialization
template <>
struct PacListOps<PacListAggType::AVG> {
	static constexpr PAC_FLOAT InitValue() {
		return PAC_FLOAT(0);
	}
	static void UpdateDense(PAC_FLOAT *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT counts,
	                        const PAC_FLOAT *PAC_RESTRICT src, idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			dst[i] += src[i];
			counts[i]++;
		}
	}
	static void UpdateScatter(PAC_FLOAT *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT counts,
	                          const PAC_FLOAT *PAC_RESTRICT src, const idx_t *PAC_RESTRICT indices,
	                          const uint64_t *PAC_RESTRICT, idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			dst[indices[i]] += src[i];
			counts[indices[i]]++;
		}
	}
	static void Combine(PAC_FLOAT *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT dst_counts,
	                    const PAC_FLOAT *PAC_RESTRICT src, const uint64_t *PAC_RESTRICT src_counts, uint64_t,
	                    uint64_t) {
		for (idx_t i = 0; i < 64; i++) {
			dst[i] += src[i];
			dst_counts[i] += src_counts[i];
		}
	}
	static PAC_FLOAT Finalize(PAC_FLOAT value, uint64_t count) {
		return count > 0 ? static_cast<PAC_FLOAT>(value / static_cast<double>(count)) : PAC_FLOAT(0);
	}
};

// COUNT specialization
template <>
struct PacListOps<PacListAggType::COUNT> {
	static constexpr PAC_FLOAT InitValue() {
		return PAC_FLOAT(0);
	}
	static void UpdateDense(PAC_FLOAT *PAC_RESTRICT, uint64_t *PAC_RESTRICT counts, const PAC_FLOAT *PAC_RESTRICT,
	                        idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			counts[i]++;
		}
	}
	static void UpdateScatter(PAC_FLOAT *PAC_RESTRICT, uint64_t *PAC_RESTRICT counts, const PAC_FLOAT *PAC_RESTRICT,
	                          const idx_t *PAC_RESTRICT indices, const uint64_t *PAC_RESTRICT, idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			counts[indices[i]]++;
		}
	}
	static void Combine(PAC_FLOAT *PAC_RESTRICT, uint64_t *PAC_RESTRICT dst_counts, const PAC_FLOAT *PAC_RESTRICT,
	                    const uint64_t *PAC_RESTRICT src_counts, uint64_t, uint64_t) {
		for (idx_t i = 0; i < 64; i++) {
			dst_counts[i] += src_counts[i];
		}
	}
	static PAC_FLOAT Finalize(PAC_FLOAT, uint64_t count) {
		return static_cast<PAC_FLOAT>(count);
	}
};

// MIN specialization
template <>
struct PacListOps<PacListAggType::MIN> {
	static constexpr PAC_FLOAT InitValue() {
		return std::numeric_limits<PAC_FLOAT>::max();
	}
	static void UpdateDense(PAC_FLOAT *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const PAC_FLOAT *PAC_RESTRICT src,
	                        idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			dst[i] = std::min(dst[i], src[i]);
		}
	}
	static void UpdateScatter(PAC_FLOAT *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const PAC_FLOAT *PAC_RESTRICT src,
	                          const idx_t *PAC_RESTRICT indices, const uint64_t *PAC_RESTRICT, idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			dst[indices[i]] = std::min(dst[indices[i]], src[i]);
		}
	}
	static void Combine(PAC_FLOAT *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const PAC_FLOAT *PAC_RESTRICT src,
	                    const uint64_t *PAC_RESTRICT, uint64_t src_mask, uint64_t dst_mask) {
		for (idx_t i = 0; i < 64; i++) {
			bool src_has = (src_mask & (1ULL << i)) != 0;
			bool dst_has = (dst_mask & (1ULL << i)) != 0;
			if (src_has && (!dst_has || src[i] < dst[i])) {
				dst[i] = src[i];
			}
		}
	}
	static PAC_FLOAT Finalize(PAC_FLOAT value, uint64_t) {
		return value;
	}
};

// MAX specialization
template <>
struct PacListOps<PacListAggType::MAX> {
	static constexpr PAC_FLOAT InitValue() {
		return std::numeric_limits<PAC_FLOAT>::lowest();
	}
	static void UpdateDense(PAC_FLOAT *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const PAC_FLOAT *PAC_RESTRICT src,
	                        idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			dst[i] = std::max(dst[i], src[i]);
		}
	}
	static void UpdateScatter(PAC_FLOAT *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const PAC_FLOAT *PAC_RESTRICT src,
	                          const idx_t *PAC_RESTRICT indices, const uint64_t *PAC_RESTRICT, idx_t len) {
		for (idx_t i = 0; i < len; i++) {
			dst[indices[i]] = std::max(dst[indices[i]], src[i]);
		}
	}
	static void Combine(PAC_FLOAT *PAC_RESTRICT dst, uint64_t *PAC_RESTRICT, const PAC_FLOAT *PAC_RESTRICT src,
	                    const uint64_t *PAC_RESTRICT, uint64_t src_mask, uint64_t dst_mask) {
		for (idx_t i = 0; i < 64; i++) {
			bool src_has = (src_mask & (1ULL << i)) != 0;
			bool dst_has = (dst_mask & (1ULL << i)) != 0;
			if (src_has && (!dst_has || src[i] > dst[i])) {
				dst[i] = src[i];
			}
		}
	}
	static PAC_FLOAT Finalize(PAC_FLOAT value, uint64_t) {
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
	PAC_FLOAT init_val = PacListOps<AGG_TYPE>::InitValue();
	for (idx_t i = 0; i < 64; i++) {
		state.values[i] = init_val;
		state.counts[i] = 0;
	}
}

// Dense update: when child vector is flat and contiguous (no validity gaps)
template <PacListAggType AGG_TYPE>
static void PacListAggregateUpdateDense(PacListAggregateState &state, const PAC_FLOAT *child_values, idx_t offset,
                                        idx_t len) {
	state.key_hash |= (len == 64) ? ~0ULL : ((1ULL << len) - 1);
	PacListOps<AGG_TYPE>::UpdateDense(state.values, state.counts, child_values + offset, len);
}

// Scatter update: when child has validity gaps or non-contiguous access
template <PacListAggType AGG_TYPE>
static void PacListAggregateUpdateScatter(PacListAggregateState &state, const PAC_FLOAT *child_values,
                                          UnifiedVectorFormat &child_data, idx_t offset, idx_t len) {
	// Temporary buffers for gathering valid values and their target indices
	PAC_FLOAT valid_values[64];
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
	auto child_values = UnifiedVectorFormat::GetData<PAC_FLOAT>(child_data);

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

		// Compute finalized values and check for sample diversity
		PAC_FLOAT buf[64] = {0};
		for (idx_t j = 0; j < 64; j++) {
			if (state.key_hash & (1ULL << j)) {
				buf[j] = PacListOps<AGG_TYPE>::Finalize(state.values[j], state.counts[j]);
			}
		}
		vector<Value> list_values;
		list_values.reserve(64);
		for (idx_t j = 0; j < 64; j++) {
			if (!(state.key_hash & (1ULL << j))) {
				list_values.push_back(Value());
			} else {
				list_values.push_back(std::is_same<PAC_FLOAT, float>::value
				                          ? Value::FLOAT(static_cast<float>(buf[j]))
				                          : Value::DOUBLE(static_cast<double>(buf[j])));
			}
		}
		result.SetValue(result_idx, Value::LIST(PacFloatLogicalType(), std::move(list_values)));
	}
}

static void PacListAggregateDestructor(Vector &, AggregateInputData &, idx_t) {
}

template <PacListAggType AGG_TYPE>
static AggregateFunction CreatePacListAggregate(const string &name) {
	auto list_double_type = LogicalType::LIST(PacFloatLogicalType());
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
	// pac_coalesce(list<PAC_FLOAT>) -> list<PAC_FLOAT> : Replace NULL list with 64 NULLs
	auto list_double_type = LogicalType::LIST(PacFloatLogicalType());
	ScalarFunction pac_coalesce("pac_coalesce", {list_double_type}, list_double_type, PacCoalesceFunction);
	pac_coalesce.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	loader.RegisterFunction(pac_coalesce);

	// pac_noised(list<PAC_FLOAT>) -> PAC_FLOAT : Apply noise to 64 counter values
	ScalarFunction pac_noised("pac_noised", {list_double_type}, PacFloatLogicalType(), PacNoisedFunction,
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
