//
// PAC Subquery Handler
//
// This file contains functions for handling correlated subqueries and DELIM_JOIN operations
// in PAC query compilation. These functions help manage column accessibility and propagation
// across subquery boundaries.
//
// Created by ila on 1/22/26.
//

#include "include/pac_subquery_handler.hpp"
#include "include/pac_expression_builder.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"

namespace duckdb {

DelimColumnResult AddColumnToDelimJoin(unique_ptr<LogicalOperator> &plan, LogicalGet &source_get,
                                       const string &column_name, LogicalAggregate *target_agg) {
	// Find the DELIM_JOIN that is an ancestor of the target aggregate
	// The DELIM_JOIN connects the outer query (containing source_get) to the subquery (containing target_agg)

	// First, find the DELIM_JOIN by walking up from the root
	LogicalComparisonJoin *delim_join = nullptr;
	std::function<bool(LogicalOperator *)> find_delim_join = [&](LogicalOperator *op) -> bool {
		if (!op)
			return false;

		if (op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			// Check if this DELIM_JOIN has the target aggregate in its subtree
			// and the source_get in the other subtree
			auto &join = op->Cast<LogicalComparisonJoin>();

			// Check if target_agg is in children[1] (the subquery side)
			bool agg_in_right = false;
			std::function<bool(LogicalOperator *)> find_agg = [&](LogicalOperator *child) -> bool {
				if (child == target_agg)
					return true;
				for (auto &c : child->children) {
					if (find_agg(c.get()))
						return true;
				}
				return false;
			};

			if (join.children.size() >= 2) {
				agg_in_right = find_agg(join.children[1].get());
			}

			// Check if source_get is in children[0] (the outer query side)
			bool source_in_left = false;
			std::function<bool(LogicalOperator *)> find_source = [&](LogicalOperator *child) -> bool {
				if (child->type == LogicalOperatorType::LOGICAL_GET) {
					auto &get = child->Cast<LogicalGet>();
					if (get.table_index == source_get.table_index)
						return true;
				}
				for (auto &c : child->children) {
					if (find_source(c.get()))
						return true;
				}
				return false;
			};

			if (!join.children.empty()) {
				source_in_left = find_source(join.children[0].get());
			}

			if (agg_in_right && source_in_left) {
				delim_join = &join;
				return true;
			}
		}

		for (auto &child : op->children) {
			if (find_delim_join(child.get()))
				return true;
		}
		return false;
	};

	find_delim_join(plan.get());

	DelimColumnResult invalid_result;
	invalid_result.binding = ColumnBinding(DConstants::INVALID_INDEX, DConstants::INVALID_INDEX);
	invalid_result.type = LogicalType::INVALID;

	if (!delim_join) {
		// No DELIM_JOIN found - return invalid result
		return invalid_result;
	}

	// Ensure the column is projected in source_get
	idx_t col_proj_idx = EnsureProjectedColumn(source_get, column_name);
	if (col_proj_idx == DConstants::INVALID_INDEX) {
		return invalid_result;
	}

	// Get the column type
	auto col_index = source_get.GetColumnIds()[col_proj_idx];
	auto col_type = source_get.GetColumnType(col_index);

	// Create a column reference expression for the source column
	auto source_binding = ColumnBinding(source_get.table_index, col_proj_idx);
	auto col_ref = make_uniq<BoundColumnRefExpression>(col_type, source_binding);

	// Add to DELIM_JOIN's duplicate_eliminated_columns
	idx_t new_col_idx = delim_join->duplicate_eliminated_columns.size();
	delim_join->duplicate_eliminated_columns.push_back(std::move(col_ref));

	// Find and update all DELIM_GETs in the subquery that reference this DELIM_JOIN
	// We need to add the new column type to their chunk_types
	std::function<void(LogicalOperator *)> update_delim_gets = [&](LogicalOperator *op) {
		if (!op)
			return;

		if (op->type == LogicalOperatorType::LOGICAL_DELIM_GET) {
			auto &delim_get = op->Cast<LogicalDelimGet>();
			// Add the new column type
			delim_get.chunk_types.push_back(col_type);
		}

		for (auto &child : op->children) {
			update_delim_gets(child.get());
		}
	};

	// Only update DELIM_GETs in the subquery side (children[1])
	if (delim_join->children.size() >= 2) {
		update_delim_gets(delim_join->children[1].get());
	}

	// Find the DELIM_GET that the aggregate can access and return the binding for the new column
	// Walk from aggregate to find the closest DELIM_GET
	std::function<LogicalDelimGet *(LogicalOperator *)> find_delim_get = [&](LogicalOperator *op) -> LogicalDelimGet * {
		if (!op)
			return nullptr;

		if (op->type == LogicalOperatorType::LOGICAL_DELIM_GET) {
			return &op->Cast<LogicalDelimGet>();
		}

		for (auto &child : op->children) {
			auto result = find_delim_get(child.get());
			if (result)
				return result;
		}
		return nullptr;
	};

	auto *delim_get = find_delim_get(target_agg);
	if (!delim_get) {
		return invalid_result;
	}

	// Return binding and type for the new column in the DELIM_GET
	DelimColumnResult result;
	result.binding = ColumnBinding(delim_get->table_index, new_col_idx);
	result.type = col_type;
	return result;
}

} // namespace duckdb
