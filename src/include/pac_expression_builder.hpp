//
// Created by ila on 1/6/26.
//

#ifndef PAC_EXPRESSION_BUILDER_HPP
#define PAC_EXPRESSION_BUILDER_HPP

#include "duckdb.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"

namespace duckdb {

// Ensure a column is projected in a LogicalGet and return its projection index
// Returns DConstants::INVALID_INDEX if the column cannot be found
idx_t EnsureProjectedColumn(LogicalGet &g, const string &col_name);

// Ensure PK columns are present in a LogicalGet's column_ids and projection_ids
void AddPKColumns(LogicalGet &get, const vector<string> &pks);

// Helper to ensure rowid is present in the output columns of a LogicalGet
void AddRowIDColumn(LogicalGet &get);

// Build XOR(pk1, pk2, ...) then hash(...) bound expression for the given LogicalGet's PKs
unique_ptr<Expression> BuildXorHashFromPKs(OptimizerExtensionInput &input, LogicalGet &get, const vector<string> &pks);

// Modify aggregate expressions to use PAC functions (replaces the aggregate loop logic)
void ModifyAggregatesWithPacFunctions(OptimizerExtensionInput &input, LogicalAggregate *agg,
                                      unique_ptr<Expression> &hash_input_expr);

} // namespace duckdb

#endif // PAC_EXPRESSION_BUILDER_HPP
