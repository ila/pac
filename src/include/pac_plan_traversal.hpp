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

// Check if an operator has any LogicalGet nodes (base table scans) in its subtree.
// Returns false if the subtree only contains CTE scans or no table scans at all.
bool HasBaseTableInSubtree(LogicalOperator *op);

// Check if an operator has a specific table (by name) in its subtree.
// Returns true if there's a LogicalGet for the given table name in the subtree.
bool HasTableInSubtree(LogicalOperator *op, const string &table_name);

// Find all LogicalGet nodes for a specific table name in the plan tree.
// Returns a vector of pointers to the unique_ptrs holding the LogicalGet nodes.
void FindAllNodesByTable(unique_ptr<LogicalOperator> *root, const string &table_name,
                         vector<unique_ptr<LogicalOperator> *> &results);

// Check if an operator has a LogicalGet with a specific table index in its subtree.
bool HasTableIndexInSubtree(LogicalOperator *op, idx_t table_index);

// Find all LogicalGet nodes with a specific table index in the plan tree.
void FindAllNodesByTableIndex(unique_ptr<LogicalOperator> *root, idx_t table_index,
                              vector<unique_ptr<LogicalOperator> *> &results);

// Filter aggregates to only those that have specified tables in their subtree
// AND have base tables in their DIRECT children (not through nested aggregates).
// This filters out outer aggregates that only depend on inner aggregate results.
vector<LogicalAggregate *> FilterTargetAggregates(const vector<LogicalAggregate *> &all_aggregates,
                                                  const vector<string> &target_table_names);

// Check if a target node is inside a DELIM_JOIN's subquery branch (children[1]).
// This is important for correlated subqueries where nodes in the subquery branch
// cannot directly access tables from the outer query.
bool IsInDelimJoinSubqueryBranch(unique_ptr<LogicalOperator> *root, LogicalOperator *target_node);

} // namespace duckdb

#endif // PAC_PLAN_TRAVERSAL_HPP
