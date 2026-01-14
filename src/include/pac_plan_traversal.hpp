//
// Created by ila on 1/6/26.
//

#ifndef PAC_PLAN_TRAVERSAL_HPP
#define PAC_PLAN_TRAVERSAL_HPP

#include "duckdb.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

namespace duckdb {

// Find the first LogicalGet node in `plan`. Returns a pointer to the unique_ptr that holds the
// found node (so it can be replaced). If pu_table_name is provided, only returns a LogicalGet
// matching that table name. Throws InternalException if not found.
unique_ptr<LogicalOperator> *FindPrivacyUnitGetNode(unique_ptr<LogicalOperator> &plan,
                                                    const string &pu_table_name = "");

// Find the first LogicalAggregate node in the plan tree.
// Throws InternalException if not found.
LogicalAggregate *FindTopAggregate(unique_ptr<LogicalOperator> &op);

// Find all LogicalAggregate nodes in the plan tree.
// Returns a vector of pointers to all aggregates found.
void FindAllAggregates(unique_ptr<LogicalOperator> &op, vector<LogicalAggregate *> &aggregates);

// Find the parent LogicalProjection of a given child node.
// Returns nullptr if not found.
LogicalProjection *FindParentProjection(unique_ptr<LogicalOperator> &root, LogicalOperator *target_child);

// Find the unique_ptr reference to a LogicalGet node by table name.
// Optionally returns the parent node and child index.
// Returns nullptr if not found.
unique_ptr<LogicalOperator> *FindNodeRefByTable(unique_ptr<LogicalOperator> *root, const string &table_name,
                                                LogicalOperator **parent_out = nullptr, idx_t *child_idx_out = nullptr);

} // namespace duckdb

#endif // PAC_PLAN_TRAVERSAL_HPP
