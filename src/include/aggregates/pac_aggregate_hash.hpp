//
// PAC Aggregate Hash Builder
//
// This file provides utilities for building hash expressions for PAC aggregates
// and transforming aggregates to use PAC functions. It extracts the aggregate
// modification logic from pac_bitslice_compiler.cpp.
//
// Created by refactoring pac_bitslice_compiler.cpp
//

#ifndef PAC_AGGREGATE_HASH_HPP
#define PAC_AGGREGATE_HASH_HPP

#include "duckdb.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "metadata/pac_compatibility_check.hpp"

namespace duckdb {

// Result of building a hash expression for an aggregate
struct AggregateHashResult {
	unique_ptr<Expression> hash_expr;
	bool success;

	AggregateHashResult() : success(false) {
	}
	explicit AggregateHashResult(unique_ptr<Expression> expr) : hash_expr(std::move(expr)), success(true) {
	}
};

// Build hash expression for an aggregate from an FK table
// This handles ensuring columns are projected, building the hash, and propagating through projections
AggregateHashResult BuildHashForAggregateFromFKTable(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                                                     LogicalAggregate *target_agg, LogicalGet &fk_get,
                                                     const vector<string> &fk_columns);

// Build hash expression for an aggregate from a PU table (using PK or rowid)
AggregateHashResult BuildHashForAggregateFromPU(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                                                LogicalAggregate *target_agg, LogicalGet &pu_get,
                                                const vector<string> &pk_columns, bool use_rowid);

// Build hash expression using DELIM_JOIN for correlated subqueries
// This is used when the FK table is in the outer query and not directly accessible
AggregateHashResult BuildHashForAggregateViaDelimJoin(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                                                      LogicalAggregate *target_agg, LogicalGet &fk_get,
                                                      const vector<string> &fk_columns);

// Process a single aggregate: find the appropriate FK/PU table and build hash
// Returns true if the aggregate was successfully processed
bool ProcessAggregateForPAC(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                            LogicalAggregate *target_agg, const std::unordered_map<idx_t, idx_t> &connecting_to_fk_map,
                            const vector<string> &gets_present, const vector<string> &privacy_units,
                            const vector<string> &fk_path, const PACCompatibilityResult &check);

// Process all target aggregates in a plan (for ModifyPlanWithoutPU)
void ProcessAllAggregatesWithoutPU(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                                   const vector<LogicalAggregate *> &target_aggregates,
                                   const std::unordered_map<idx_t, idx_t> &connecting_to_fk_map,
                                   const vector<string> &gets_present, const vector<string> &privacy_units,
                                   const vector<string> &fk_path, const PACCompatibilityResult &check);

// Process all target aggregates in a plan (for ModifyPlanWithPU)
void ProcessAllAggregatesWithPU(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                                const vector<LogicalAggregate *> &target_aggregates,
                                const vector<string> &pu_table_names, const PACCompatibilityResult &check);

} // namespace duckdb

#endif // PAC_AGGREGATE_HASH_HPP
