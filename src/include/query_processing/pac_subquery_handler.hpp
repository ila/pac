//
// PAC Subquery Handler
//
// This file contains functions for handling correlated subqueries and DELIM_JOIN operations
// in PAC query compilation. These functions help manage column accessibility and propagation
// across subquery boundaries.
//
// Created by ila on 1/22/26.
//

#ifndef PAC_SUBQUERY_HANDLER_HPP
#define PAC_SUBQUERY_HANDLER_HPP

#include "duckdb.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"

namespace duckdb {

// Result of AddColumnToDelimJoin - contains both the binding and the column type
struct DelimColumnResult {
	ColumnBinding binding;
	LogicalType type;

	bool IsValid() const {
		return binding.table_index != DConstants::INVALID_INDEX;
	}
};

// Add a column to a DELIM_JOIN's duplicate_eliminated_columns and update corresponding DELIM_GETs
// Returns the ColumnBinding and LogicalType for accessing the column via DELIM_GET
DelimColumnResult AddColumnToDelimJoin(unique_ptr<LogicalOperator> &plan, LogicalGet &source_get,
                                       const string &column_name, LogicalAggregate *target_agg);

} // namespace duckdb

#endif // PAC_SUBQUERY_HANDLER_HPP
