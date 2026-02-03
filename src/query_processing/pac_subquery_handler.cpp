//
// PAC Subquery Handler
//
// This file contains functions for handling correlated subqueries and DELIM_JOIN operations
// in PAC query compilation. These functions help manage column accessibility and propagation
// across subquery boundaries.
//
// Created by ila on 1/22/26.
//

#include "query_processing/pac_subquery_handler.hpp"
#include "query_processing/pac_expression_builder.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"

namespace duckdb {

// Helper: Ensure a column flows through all operators between source_get and target_op
// Returns the final output binding, or invalid if the column cannot flow through
// This function may modify operators (e.g., add to aggregate groups) to ensure the column flows through
static ColumnBinding EnsureColumnFlowsThrough(LogicalOperator *target_op, LogicalGet &source_get,
                                              const string &column_name, idx_t source_col_proj_idx,
                                              LogicalType &out_type) {
	// First, get the column's type from source
	auto col_index = source_get.GetColumnIds()[source_col_proj_idx];
	out_type = source_get.GetColumnType(col_index);

	// Determine the correct output binding for the source_get
	// When projection_ids is non-empty, bindings use projection_ids values
	// When projection_ids is empty, bindings are sequential [0, 1, 2, ...]
	idx_t output_col_idx;
	if (source_get.projection_ids.empty()) {
		output_col_idx = source_col_proj_idx;
	} else {
		// Find the projection_id that corresponds to source_col_proj_idx
		// projection_ids[i] = j means "the i-th output column comes from column_ids[j]"
		// We need to find i where projection_ids[i] == source_col_proj_idx
		output_col_idx = DConstants::INVALID_INDEX;
		for (idx_t i = 0; i < source_get.projection_ids.size(); i++) {
			if (source_get.projection_ids[i] == source_col_proj_idx) {
				output_col_idx = source_get.projection_ids[i]; // Use the projection_id value as binding
				break;
			}
		}
		if (output_col_idx == DConstants::INVALID_INDEX) {
			// Column is in column_ids but not in projection_ids - we need to add it
#ifdef DEBUG
			Printer::Print("EnsureColumnFlowsThrough: Column at position " + std::to_string(source_col_proj_idx) +
			               " not in projection_ids, adding it");
#endif
			source_get.projection_ids.push_back(source_col_proj_idx);
			output_col_idx = source_col_proj_idx; // The binding uses the projection_id value
			// Resolve types to update the types vector
			source_get.ResolveOperatorTypes();
		}
	}

	ColumnBinding source_binding(source_get.table_index, output_col_idx);

	// Find path from target_op down to source_get and ensure column flows through
	std::function<ColumnBinding(LogicalOperator *)> ensure_flow = [&](LogicalOperator *op) -> ColumnBinding {
		if (op->type == LogicalOperatorType::LOGICAL_GET) {
			auto &get = op->Cast<LogicalGet>();
			if (get.table_index == source_get.table_index) {
				return source_binding;
			}
			return ColumnBinding(DConstants::INVALID_INDEX, DConstants::INVALID_INDEX);
		}

		// Recursively find and trace through children
		ColumnBinding child_result(DConstants::INVALID_INDEX, DConstants::INVALID_INDEX);
		for (auto &child : op->children) {
			child_result = ensure_flow(child.get());
			if (child_result.table_index != DConstants::INVALID_INDEX) {
				break;
			}
		}

		if (child_result.table_index == DConstants::INVALID_INDEX) {
			return child_result;
		}

		// Handle how the column passes through this operator
		switch (op->type) {
		case LogicalOperatorType::LOGICAL_FILTER: {
			// Filters pass through all columns
			return child_result;
		}
		case LogicalOperatorType::LOGICAL_PROJECTION: {
			auto &proj = op->Cast<LogicalProjection>();
			for (idx_t i = 0; i < proj.expressions.size(); i++) {
				if (proj.expressions[i]->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = proj.expressions[i]->Cast<BoundColumnRefExpression>();
					if (col_ref.binding == child_result) {
						return ColumnBinding(proj.table_index, i);
					}
				}
			}
			// Column not projected - return invalid
			return ColumnBinding(DConstants::INVALID_INDEX, DConstants::INVALID_INDEX);
		}
		case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
			auto &agg = op->Cast<LogicalAggregate>();
			// Check if column is already in groups
			for (idx_t i = 0; i < agg.groups.size(); i++) {
				if (agg.groups[i]->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = agg.groups[i]->Cast<BoundColumnRefExpression>();
					if (col_ref.binding == child_result) {
						return ColumnBinding(agg.group_index, i);
					}
				}
			}
			// Column not in groups - we need to ADD it to the groups
			// This is safe when the column is functionally dependent on existing groups
			// (e.g., user_id is functionally dependent on id if id is the PK)
			//
			// We add the column to the aggregate's groups so it passes through
#ifdef DEBUG
			Printer::Print("AddColumnToDelimJoin: Adding column to aggregate groups (binding [" +
			               std::to_string(child_result.table_index) + "." + std::to_string(child_result.column_index) +
			               "])");
#endif
			// Create a column ref expression for the group
			auto group_col_ref = make_uniq<BoundColumnRefExpression>(out_type, child_result);
			idx_t new_group_idx = agg.groups.size();
			agg.groups.push_back(std::move(group_col_ref));

			// After modifying groups, we need to resolve types again
			agg.ResolveOperatorTypes();

#ifdef DEBUG
			Printer::Print("AddColumnToDelimJoin: Added column as group " + std::to_string(new_group_idx) +
			               ", output binding [" + std::to_string(agg.group_index) + "." +
			               std::to_string(new_group_idx) + "]");
#endif
			return ColumnBinding(agg.group_index, new_group_idx);
		}
		default:
			return child_result;
		}
	};

	return ensure_flow(target_op);
}

DelimColumnResult AddColumnToDelimJoin(unique_ptr<LogicalOperator> &plan, LogicalGet &source_get,
                                       const string &column_name, LogicalAggregate *target_agg) {
	// Find the DELIM_JOIN that is an ancestor of the target aggregate
	// The DELIM_JOIN connects the outer query (containing source_get) to the subquery (containing target_agg)

	// First, find the DELIM_JOIN by walking up from the root
	LogicalComparisonJoin *delim_join = nullptr;
	std::function<bool(LogicalOperator *)> find_delim_join = [&](LogicalOperator *op) -> bool {
		if (!op) {
			return false;
		}

		if (op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			// Check if this DELIM_JOIN has the target aggregate in its subtree
			// and the source_get in the other subtree
			auto &join = op->Cast<LogicalComparisonJoin>();

			// Check if target_agg is in children[1] (the subquery side)
			bool agg_in_right = false;
			std::function<bool(LogicalOperator *)> find_agg = [&](LogicalOperator *child) -> bool {
				if (child == target_agg) {
					return true;
				}
				for (auto &c : child->children) {
					if (find_agg(c.get())) {
						return true;
					}
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
					if (get.table_index == source_get.table_index) {
						return true;
					}
				}
				for (auto &c : child->children) {
					if (find_source(c.get())) {
						return true;
					}
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
			if (find_delim_join(child.get())) {
				return true;
			}
		}
		return false;
	};

	find_delim_join(plan.get());

	DelimColumnResult invalid_result;
	invalid_result.binding = ColumnBinding(DConstants::INVALID_INDEX, DConstants::INVALID_INDEX);
	invalid_result.type = LogicalType::INVALID;

	if (!delim_join) {
		// No DELIM_JOIN found - return invalid result
#ifdef DEBUG
		Printer::Print("AddColumnToDelimJoin: No DELIM_JOIN found");
#endif
		return invalid_result;
	}

#ifdef DEBUG
	Printer::Print("AddColumnToDelimJoin: Found DELIM_JOIN, source_get.table_index=" +
	               std::to_string(source_get.table_index) + ", column=" + column_name);
	Printer::Print("AddColumnToDelimJoin: DELIM_JOIN has " +
	               std::to_string(delim_join->duplicate_eliminated_columns.size()) +
	               " existing duplicate_eliminated_columns");
	for (idx_t i = 0; i < delim_join->duplicate_eliminated_columns.size(); i++) {
		Printer::Print("  dup_elim_col[" + std::to_string(i) +
		               "]: " + delim_join->duplicate_eliminated_columns[i]->ToString());
	}
#endif

	// Ensure the column is projected in source_get
	idx_t col_proj_idx = EnsureProjectedColumn(source_get, column_name);
	if (col_proj_idx == DConstants::INVALID_INDEX) {
#ifdef DEBUG
		Printer::Print("AddColumnToDelimJoin: Failed to project column " + column_name + " in source_get");
#endif
		return invalid_result;
	}

#ifdef DEBUG
	Printer::Print("AddColumnToDelimJoin: Column " + column_name + " projected at index " +
	               std::to_string(col_proj_idx) + " in source_get");
#endif

	// Get the column type
	auto col_index = source_get.GetColumnIds()[col_proj_idx];
	auto col_type = source_get.GetColumnType(col_index);

	// CRITICAL FIX: The duplicate_eliminated_columns expressions must reference the OUTPUT
	// of the DELIM_JOIN's left child, not the original scan. If there are intermediate
	// operators (like aggregates) between the scan and the DELIM_JOIN, we need to trace
	// the column through them (and potentially add to aggregate groups).

	// Get the left child of DELIM_JOIN
	auto *left_child = delim_join->children[0].get();

#ifdef DEBUG
	Printer::Print("AddColumnToDelimJoin: Left child type=" + std::to_string(static_cast<int>(left_child->type)));
#endif

	// Trace the column through the left child to find the output binding
	// This may ADD the column to aggregate groups if needed
	LogicalType traced_type;
	ColumnBinding output_binding =
	    EnsureColumnFlowsThrough(left_child, source_get, column_name, col_proj_idx, traced_type);

	if (output_binding.table_index == DConstants::INVALID_INDEX) {
#ifdef DEBUG
		Printer::Print("AddColumnToDelimJoin: Column " + column_name +
		               " does not flow through to DELIM_JOIN left child output");
#endif
		return invalid_result;
	}

#ifdef DEBUG
	Printer::Print("AddColumnToDelimJoin: Column flows through with output binding [" +
	               std::to_string(output_binding.table_index) + "." + std::to_string(output_binding.column_index) +
	               "]");
#endif

	// Create a column reference expression using the OUTPUT binding (not the scan binding)
	auto col_ref = make_uniq<BoundColumnRefExpression>(col_type, output_binding);

#ifdef DEBUG
	Printer::Print("AddColumnToDelimJoin: Adding column ref " + col_ref->ToString() +
	               " to duplicate_eliminated_columns");
#endif

	// Add to DELIM_JOIN's duplicate_eliminated_columns
	idx_t new_col_idx = delim_join->duplicate_eliminated_columns.size();
	delim_join->duplicate_eliminated_columns.push_back(std::move(col_ref));

	// Find and update all DELIM_GETs in the subquery that reference this DELIM_JOIN
	// We need to add the new column type to their chunk_types
	std::function<void(LogicalOperator *)> update_delim_gets = [&](LogicalOperator *op) {
		if (!op) {
			return;
		}

		if (op->type == LogicalOperatorType::LOGICAL_DELIM_GET) {
			auto &delim_get = op->Cast<LogicalDelimGet>();
			// Add the new column type
			delim_get.chunk_types.push_back(col_type);
#ifdef DEBUG
			Printer::Print("AddColumnToDelimJoin: Updated DELIM_GET #" + std::to_string(delim_get.table_index) +
			               " with new column type");
#endif
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
		if (!op) {
			return nullptr;
		}

		if (op->type == LogicalOperatorType::LOGICAL_DELIM_GET) {
			return &op->Cast<LogicalDelimGet>();
		}

		for (auto &child : op->children) {
			auto result = find_delim_get(child.get());
			if (result) {
				return result;
			}
		}
		return nullptr;
	};

	auto *delim_get = find_delim_get(target_agg);
	if (!delim_get) {
#ifdef DEBUG
		Printer::Print("AddColumnToDelimJoin: No DELIM_GET found in aggregate subtree");
#endif
		return invalid_result;
	}

#ifdef DEBUG
	Printer::Print("AddColumnToDelimJoin: Found DELIM_GET #" + std::to_string(delim_get->table_index) +
	               ", returning binding [" + std::to_string(delim_get->table_index) + "." +
	               std::to_string(new_col_idx) + "]");
#endif

	// Return binding and type for the new column in the DELIM_GET
	DelimColumnResult result;
	result.binding = ColumnBinding(delim_get->table_index, new_col_idx);
	result.type = col_type;
	return result;
}

} // namespace duckdb
