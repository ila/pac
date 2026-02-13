//
// Created by ila on 1/16/26.
//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

class LogicalGet;
class LogicalAggregate;
struct OptimizerExtensionInput;

/**
 * PropagatePKThroughProjections: Propagates primary key columns through projection operators
 *
 * Purpose: When we have projections between a table scan (e.g., privacy unit table) and an aggregate,
 * we need to ensure the PK columns used in hash expressions are available at the aggregate level.
 * This function traces the path from the table scan to the aggregate, adds necessary columns to
 * intermediate projections, and updates the hash expression's column bindings accordingly.
 *
 * Arguments:
 * @param plan - The logical plan being modified
 * @param pu_get - The LogicalGet node for the privacy unit (or FK-linked) table containing the source columns
 * @param hash_expr - The hash expression built from PK columns at the table scan level
 * @param target_agg - The target aggregate node where the hash expression will be used
 *
 * Returns: An updated hash expression with column bindings that reference the correct projection outputs
 *
 * Logic:
 * 1. Walk from the aggregate down to the table scan, collecting all projection operators
 * 2. Extract all column references from the hash expression (these are the PK columns we need)
 * 3. For each projection (processing bottom-up from table scan to aggregate):
 *    - Check if each required column is already projected
 *    - If not, add it to the projection's expression list
 *    - Update the binding map to track how column bindings change through this projection
 * 4. Apply the final binding map to the hash expression, updating all column references
 *
 * Example: If hash(customer.c_custkey) is built at the table scan level with binding [5, 0],
 * but there's a projection between the scan and aggregate that remaps it to [8, 3],
 * this function ensures the projection includes c_custkey and returns hash([8, 3]).
 */
unique_ptr<Expression> PropagatePKThroughProjections(LogicalOperator &plan, LogicalGet &pu_get,
                                                     unique_ptr<Expression> hash_expr, LogicalAggregate *target_agg);

} // namespace duckdb
