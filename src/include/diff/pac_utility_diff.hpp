#pragma once

#include "duckdb.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"

namespace duckdb {

// Apply utility diff rewrite to the PAC-compiled plan.
// Wraps the plan in a FULL OUTER JOIN with the deep-copied reference plan and
// rewrites the top projection to encode diff information (utility %).
// Must be called AFTER CompilePacBitsliceQuery, within the ReplanGuard scope.
void ApplyUtilityDiff(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                      unique_ptr<LogicalOperator> ref_plan, idx_t num_key_cols, const string &output_path = "");

} // namespace duckdb
