//
// Created by ila on 1/6/26.
//

#include "include/pac_plan_traversal.hpp"

#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"

namespace duckdb {

unique_ptr<LogicalOperator> *FindPrivacyUnitGetNode(unique_ptr<LogicalOperator> &plan, const string &pu_table_name) {
	unique_ptr<LogicalOperator> *found_ptr = nullptr;
	if (!plan) {
		return nullptr;
	}

	vector<unique_ptr<LogicalOperator> *> stack;
	stack.push_back(&plan);
	while (!stack.empty()) {
		auto cur_ptr = stack.back();
		stack.pop_back();
		auto &cur = *cur_ptr;
		if (!cur) {
			continue;
		}
		if (cur->type == LogicalOperatorType::LOGICAL_GET) {
			// If a specific table name is provided, only match that table
			if (!pu_table_name.empty()) {
				auto &get = cur->Cast<LogicalGet>();
				auto tblptr = get.GetTable();
				if (tblptr && tblptr->name == pu_table_name) {
					found_ptr = cur_ptr;
					break;
				}
			} else {
				// No specific table name provided, return the first LogicalGet
				found_ptr = cur_ptr;
				break;
			}
		}
		for (auto &c : cur->children) {
			stack.push_back(&c);
		}
	}

	if (!found_ptr) {
		if (!pu_table_name.empty()) {
			throw InternalException("PAC Compiler: could not find LogicalGet node for table " + pu_table_name +
			                        " in plan");
		} else {
			throw InternalException("PAC Compiler: could not find LogicalGet node in plan");
		}
	}

	return found_ptr;
}

LogicalAggregate *FindTopAggregate(unique_ptr<LogicalOperator> &op) {
	if (!op) {
		throw InternalException("PAC Compiler: could not find LogicalAggregate node in plan");
	}
	if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return &op->Cast<LogicalAggregate>();
	}
	for (auto &child : op->children) {
		if (auto *agg = FindTopAggregate(child)) {
			return agg;
		}
	}
	throw InternalException("PAC Compiler: could not find LogicalAggregate node in plan");
}

// Find all LogicalAggregate nodes in the plan tree
void FindAllAggregates(unique_ptr<LogicalOperator> &op, vector<LogicalAggregate *> &aggregates) {
	if (!op) {
		return;
	}
	if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		aggregates.push_back(&op->Cast<LogicalAggregate>());
	}
	for (auto &child : op->children) {
		FindAllAggregates(child, aggregates);
	}
}

LogicalProjection *FindParentProjection(unique_ptr<LogicalOperator> &root, LogicalOperator *target_child) {
	if (!root) {
		return nullptr;
	}
	for (auto &child : root->children) {
		if (child.get() == target_child && root->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			return &root->Cast<LogicalProjection>();
		}
		if (auto *proj = FindParentProjection(child, target_child)) {
			return proj;
		}
	}
	return nullptr;
}

unique_ptr<LogicalOperator> *FindNodeRefByTable(unique_ptr<LogicalOperator> *root, const string &table_name,
                                                LogicalOperator **parent_out, idx_t *child_idx_out) {
	if (!root || !root->get()) {
		return nullptr;
	}

	struct StackEntry {
		unique_ptr<LogicalOperator> *ptr;
		LogicalOperator *parent;
		idx_t child_idx;
	};

	vector<StackEntry> stack;
	stack.push_back({root, nullptr, 0});

	while (!stack.empty()) {
		auto entry = stack.back();
		stack.pop_back();

		auto &cur = *entry.ptr;
		if (!cur) {
			continue;
		}

		if (cur->type == LogicalOperatorType::LOGICAL_GET) {
			auto &get = cur->Cast<LogicalGet>();
			auto tblptr = get.GetTable();
			if (tblptr && tblptr->name == table_name) {
				if (parent_out) {
					*parent_out = entry.parent;
				}
				if (child_idx_out) {
					*child_idx_out = entry.child_idx;
				}
				return entry.ptr;
			}
		}

		for (idx_t i = 0; i < cur->children.size(); i++) {
			stack.push_back({&cur->children[i], cur.get(), i});
		}
	}

	return nullptr;
}

// Check if an operator has any LogicalGet nodes (base table scans) in its subtree.
// Returns false if the subtree only contains CTE scans or no table scans at all.
// IMPORTANT: This function stops at aggregates, because aggregates consume base table
// bindings and produce new output bindings. Base tables behind an aggregate are not
// directly accessible from operators above the aggregate.
bool HasBaseTableInSubtree(LogicalOperator *op) {
	if (!op) {
		return false;
	}

	// Check if this is a base table scan
	if (op->type == LogicalOperatorType::LOGICAL_GET) {
		return true;
	}

	// Don't traverse through aggregates - they consume base table bindings
	// and produce new output bindings
	if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return false;
	}

	// Recursively check children
	for (auto &child : op->children) {
		if (HasBaseTableInSubtree(child.get())) {
			return true;
		}
	}

	return false;
}

// Check if an operator has a specific table (by name) in its subtree.
// Returns true if there's a LogicalGet for the given table name in the subtree.
bool HasTableInSubtree(LogicalOperator *op, const string &table_name) {
	if (!op) {
		return false;
	}

	// Check if this is a LogicalGet for the target table
	if (op->type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op->Cast<LogicalGet>();
		auto tblptr = get.GetTable();
		if (tblptr && tblptr->name == table_name) {
			return true;
		}
	}

	// Recursively check children
	for (auto &child : op->children) {
		if (HasTableInSubtree(child.get(), table_name)) {
			return true;
		}
	}

	return false;
}

// Find all LogicalGet nodes for a specific table name in the plan tree.
void FindAllNodesByTable(unique_ptr<LogicalOperator> *root, const string &table_name,
                         vector<unique_ptr<LogicalOperator> *> &results) {
	if (!root || !root->get()) {
		return;
	}

	auto &cur = *root;

	// Check if this is a LogicalGet for the target table
	if (cur->type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = cur->Cast<LogicalGet>();
		auto tblptr = get.GetTable();
		if (tblptr && tblptr->name == table_name) {
			results.push_back(root);
		}
	}

	// Recursively check children
	for (auto &child : cur->children) {
		FindAllNodesByTable(&child, table_name, results);
	}
}

// Check if an operator has a LogicalGet with a specific table index in its subtree.
bool HasTableIndexInSubtree(LogicalOperator *op, idx_t table_index) {
	if (!op) {
		return false;
	}

	// Check if this is a LogicalGet with the target table index
	if (op->type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op->Cast<LogicalGet>();
		if (get.table_index == table_index) {
			return true;
		}
	}

	// Recursively check children
	for (auto &child : op->children) {
		if (HasTableIndexInSubtree(child.get(), table_index)) {
			return true;
		}
	}

	return false;
}

// Find all LogicalGet nodes with a specific table index in the plan tree.
void FindAllNodesByTableIndex(unique_ptr<LogicalOperator> *root, idx_t table_index,
                              vector<unique_ptr<LogicalOperator> *> &results) {
	if (!root || !root->get()) {
		return;
	}

	auto &cur = *root;

	// Check if this is a LogicalGet with the target table index
	if (cur->type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = cur->Cast<LogicalGet>();
		if (get.table_index == table_index) {
			results.push_back(root);
		}
	}

	// Recursively check children
	for (auto &child : cur->children) {
		FindAllNodesByTableIndex(&child, table_index, results);
	}
}

// Filter aggregates to only those that have specified tables in their subtree
// AND have base tables in their DIRECT children (not through nested aggregates).
// This filters out outer aggregates that only depend on inner aggregate results.
vector<LogicalAggregate *> FilterTargetAggregates(const vector<LogicalAggregate *> &all_aggregates,
                                                  const vector<string> &target_table_names) {
	vector<LogicalAggregate *> target_aggregates;

	for (auto *agg : all_aggregates) {
		// Check if this aggregate has at least one target table in its subtree
		bool has_target_table = false;
		for (auto &table_name : target_table_names) {
			if (HasTableInSubtree(agg, table_name)) {
				has_target_table = true;
				break;
			}
		}

		if (!has_target_table) {
			continue;
		}

		// Check if this aggregate has base tables in its DIRECT children (not nested aggregates)
		// If it only depends on another aggregate's output, skip it
		bool has_direct_base_table = false;
		for (auto &child : agg->children) {
			// Skip if the child is another aggregate
			if (child->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
				continue;
			}
			if (HasBaseTableInSubtree(child.get())) {
				has_direct_base_table = true;
				break;
			}
		}

		if (has_direct_base_table) {
			target_aggregates.push_back(agg);
		}
	}

	return target_aggregates;
}

// Check if a target node is inside a DELIM_JOIN's subquery branch (children[1]).
// This is important for correlated subqueries where nodes in the subquery branch
// cannot directly access tables from the outer query.
bool IsInDelimJoinSubqueryBranch(unique_ptr<LogicalOperator> *root, LogicalOperator *target_node) {
	if (!root || !root->get() || !target_node) {
		return false;
	}

	auto &cur = *root;

	// If this is a DELIM_JOIN, check if target is in children[1] (subquery side)
	if (cur->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		if (cur->children.size() >= 2) {
			// Check if target_node is in the subquery branch (children[1])
			std::function<bool(LogicalOperator *)> find_target = [&](LogicalOperator *op) -> bool {
				if (op == target_node) {
					return true;
				}
				for (auto &child : op->children) {
					if (find_target(child.get())) {
						return true;
					}
				}
				return false;
			};

			if (find_target(cur->children[1].get())) {
				return true;
			}
		}
	}

	// Recursively check children
	for (auto &child : cur->children) {
		if (IsInDelimJoinSubqueryBranch(&child, target_node)) {
			return true;
		}
	}

	return false;
}

} // namespace duckdb
