//
// Created by ila on 12/23/25.
//

#ifndef PAC_COMPILER_HELPERS_HPP
#define PAC_COMPILER_HELPERS_HPP

#include "duckdb.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "pac_compatibility_check.hpp"
#include "pac_expression_builder.hpp"
#include "pac_plan_traversal.hpp"

namespace duckdb {

// Replan the provided SQL query into `plan` after disabling several optimizers. The function
// performs the SET transaction, reparses and replans, and prints the resulting plan if present.
void ReplanWithoutOptimizers(ClientContext &context, const string &query, unique_ptr<LogicalOperator> &plan);

// Build join conditions from FK columns to PK columns
void BuildJoinConditions(LogicalGet *left_get, LogicalGet *right_get, const vector<string> &left_cols,
                         const vector<string> &right_cols, const string &left_table_name,
                         const string &right_table_name, vector<JoinCondition> &conditions);

// Create a logical join operator based on FK relationships in the compatibility check metadata
unique_ptr<LogicalOperator> CreateLogicalJoin(const PACCompatibilityResult &check, ClientContext &context,
                                              unique_ptr<LogicalOperator> left_operator, unique_ptr<LogicalGet> right);

// Create a LogicalGet operator for a table by name
unique_ptr<LogicalGet> CreateLogicalGet(ClientContext &context, unique_ptr<LogicalOperator> &plan, const string &table,
                                        idx_t idx);

// Examine PACCompatibilityResult.fk_paths and populate gets_present / gets_missing
void PopulateGetsFromFKPath(const PACCompatibilityResult &check, vector<string> &gets_present,
                            vector<string> &gets_missing, string &start_table_out, string &target_pu_out);

} // namespace duckdb

#endif // PAC_COMPILER_HELPERS_HPP
