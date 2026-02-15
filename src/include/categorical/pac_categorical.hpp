//
// PAC Categorical Query Support
//
// This file implements the counter-based approach for handling categorical queries
// (queries that don't return aggregates but use PAC aggregates in subquery predicates).
//
// Problem: When an inner query has a PAC aggregate and the outer query uses it in a comparison
// (e.g., WHERE value > pac_sum(...)), picking ONE subsample for the inner aggregate leaks privacy
// because the outer query's filter decision is based on that specific subsample.
//
// Solution: The inner PAC aggregate returns ALL 64 counter values via _counters variants.
// For filters/joins, comparisons are algebraically simplified, then the PAC side is noised
// with pac_noised and compared normally. For projections, arithmetic over counters is
// computed via list_transform and then pac_noised produces the final scalar value.
//
// Key Functions:
// - pac_*_counters(hash, value) -> LIST[DOUBLE] : Returns all 64 counter values (no noise yet)
// - pac_noised(LIST<DOUBLE>) -> DOUBLE : Sample one noised value from 64 counters
// - pac_coalesce(LIST<DOUBLE>) -> LIST<DOUBLE> : Replace NULL list with 64 NULLs
// - pac_*_list : Element-wise list aggregates over counter lists
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

} // namespace duckdb

#endif // PAC_CATEGORICAL_HPP
