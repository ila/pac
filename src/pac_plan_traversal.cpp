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

} // namespace duckdb
