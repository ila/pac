//
// PAC Categorical Query Support
//
// This file implements the mask-based approach for handling categorical queries
// (queries that don't return aggregates but use PAC aggregates in subquery predicates).
//
// Problem: When an inner query has a PAC aggregate and the outer query uses it in a comparison
// (e.g., WHERE value > pac_sum(...)), picking ONE subsample for the inner aggregate leaks privacy
// because the outer query's filter decision is based on that specific subsample.
//
// Solution: The inner PAC aggregate returns ALL 64 counter values. The comparison is evaluated
// against all 64 values, producing a 64-bit mask where bit i = 1 if subsample i satisfies the
// condition. The final selection happens at the outermost categorical boundary using pac_filter().
//
// Key Functions:
// - pac_counters(hash, value) -> LIST[DOUBLE] : Returns all 64 counter values (no noise/selection yet)
// - pac_gt(value, counters) -> UBIGINT : Returns mask where bit i = 1 if value > counters[i]
// - pac_lt(value, counters) -> UBIGINT : Returns mask where bit i = 1 if value < counters[i]
// - pac_gte(value, counters) -> UBIGINT : Returns mask where bit i = 1 if value >= counters[i]
// - pac_lte(value, counters) -> UBIGINT : Returns mask where bit i = 1 if value <= counters[i]
// - pac_select(list<bool>) -> UBIGINT : Convert boolean list to bitmask
// - pac_filter(mask) -> BOOLEAN : Probabilistically filter based on popcount(mask)/64
// - pac_filter(list<bool>) -> BOOLEAN : Convert to mask, then filter
//
// Mask operations:
// - AND: mask1 & mask2 (both conditions true)
// - OR:  mask1 | mask2 (either condition true)
// - NOT: ~mask (negate condition)
//
// Created by ila on 1/22/26.
//

#ifndef PAC_CATEGORICAL_HPP
#define PAC_CATEGORICAL_HPP

#include "duckdb.hpp"
#include "aggregates/pac_aggregate.hpp"

namespace duckdb {

// Register all PAC categorical functions with the extension loader
void RegisterPacCategoricalFunctions(ExtensionLoader &loader);

// ============================================================================
// PAC_COUNTERS aggregate: Returns all 64 counters as a LIST for categorical queries
// ============================================================================
// This is a variant of pac_sum that returns the raw counters instead of picking one.
// Used when the aggregate result will be used in a comparison in an outer categorical query.

// ============================================================================
// Comparison functions: Compare scalar value against 64 PAC counters
// ============================================================================
// Each returns a UBIGINT mask where bit i = 1 if the comparison holds for counter i

// pac_gt(value, counters) -> mask where bit i = 1 if value > counters[i]
// pac_lt(value, counters) -> mask where bit i = 1 if value < counters[i]
// pac_gte(value, counters) -> mask where bit i = 1 if value >= counters[i]
// pac_lte(value, counters) -> mask where bit i = 1 if value <= counters[i]
// pac_eq(value, counters) -> mask where bit i = 1 if value == counters[i] (approx equality for floats)

// ============================================================================
// PAC_SELECT: Convert list<bool> to bitmask
// ============================================================================
// pac_select(list<bool>) -> UBIGINT
// Converts a list of booleans to a 64-bit mask (true=1, false/NULL=0)

// ============================================================================
// PAC_FILTER: Final probabilistic selection based on mask
// ============================================================================
// pac_filter(mask) -> BOOLEAN
// Returns true with probability proportional to popcount(mask)/64
// This should be called at the outermost categorical query boundary

// pac_filter(mask, mi) -> BOOLEAN
// Same as above but with explicit mi parameter (mi=0 means deterministic: majority voting)

// pac_filter(list<bool>) -> BOOLEAN
// Convenience: converts list to mask, then applies filter logic

// ============================================================================
// Helper: Check if a mask has any bits set (for NULL handling)
// ============================================================================
static inline bool PacMaskHasAnyBit(uint64_t mask) {
	return mask != 0;
}

// ============================================================================
// Helper: Combine masks for AND/OR operations
// ============================================================================
static inline uint64_t PacMaskAnd(uint64_t mask1, uint64_t mask2) {
	return mask1 & mask2;
}

static inline uint64_t PacMaskOr(uint64_t mask1, uint64_t mask2) {
	return mask1 | mask2;
}

static inline uint64_t PacMaskNot(uint64_t mask) {
	return ~mask;
}

} // namespace duckdb

#endif // PAC_CATEGORICAL_HPP
