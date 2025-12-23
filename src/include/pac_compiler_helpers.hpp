//
// Created by ila on 12/23/25.
//

#ifndef PAC_COMPILER_HELPERS_HPP
#define PAC_COMPILER_HELPERS_HPP

#include <memory>
#include <string>
#include "duckdb.hpp"

namespace duckdb {

// Replan the provided SQL query into `plan` after disabling several optimizers. The function
// performs the SET transaction, reparses and replans, and prints the resulting plan if present.
void ReplanWithoutOptimizers(ClientContext &context, const std::string &query, unique_ptr<LogicalOperator> &plan);

// Find the first LogicalGet node in `plan`. Returns a pointer to the unique_ptr that holds the
// found node (so it can be replaced) and sets out_table_idx to the LogicalGet.table_index.
// If not found, returns nullptr and sets out_table_idx to DConstants::INVALID_INDEX.
unique_ptr<LogicalOperator> *FindPrivacyUnitGetNode(unique_ptr<LogicalOperator> &plan);

void AddRowIDColumn(LogicalGet &get);

} // namespace duckdb

#endif // PAC_COMPILER_HELPERS_HPP
