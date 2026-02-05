//
// Created by ila on 1/16/26.
//

#include "query_processing/pac_projection_propagation.hpp"
#include "pac_debug.hpp"
#include "utils/pac_helpers.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"

namespace duckdb {

// Helper to check if an operator is a join type that has projection maps
// Only joins inheriting from LogicalJoin have left_projection_map/right_projection_map
// Unconditional joins (CROSS_PRODUCT, POSITIONAL_JOIN) do NOT have projection maps
static bool IsJoinWithProjectionMap(LogicalOperatorType type) {
	return type == LogicalOperatorType::LOGICAL_JOIN || type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	       type == LogicalOperatorType::LOGICAL_DELIM_JOIN || type == LogicalOperatorType::LOGICAL_ANY_JOIN ||
	       type == LogicalOperatorType::LOGICAL_ASOF_JOIN || type == LogicalOperatorType::LOGICAL_DEPENDENT_JOIN;
}

// Helper: Find the path from an operator to a specific table scan (by table_index)
// Returns true if found, and fills 'path' with ALL operators on the path (excluding the table scan itself)
// Also tracks which child index led to the target (0 = left, 1 = right for joins)
struct PathEntry {
	LogicalOperator *op;
	idx_t child_idx; // Which child led to the target table scan
};

// FindDirectPathToTableScan: Similar to the general path finder, but STOPS at nested aggregates.
// This is used when we need to propagate columns only within the current aggregate's
// direct subtree, not through nested aggregates (which have their own column scope).
static bool FindDirectPathToTableScan(LogicalOperator *current, idx_t target_table_index, vector<PathEntry> &path,
                                      bool is_start = true) {
	if (!current) {
		return false;
	}

	// Check if this is the target table scan
	if (current->type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = current->Cast<LogicalGet>();
		if (get.table_index == target_table_index) {
			return true;
		}
	}

	// STOP at nested aggregates - they have their own column scope
	// The is_start flag allows us to skip the starting aggregate itself
	if (!is_start && current->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return false;
	}

	// Try each child
	for (idx_t child_idx = 0; child_idx < current->children.size(); child_idx++) {
		if (FindDirectPathToTableScan(current->children[child_idx].get(), target_table_index, path, false)) {
			// Found it in this subtree - add current to path ONLY if it's not an aggregate
			if (current->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
				path.push_back({current, child_idx});
			}
			return true;
		}
	}

	return false;
}

unique_ptr<Expression> PropagatePKThroughProjections(LogicalOperator &plan, LogicalGet &pu_get,
                                                     unique_ptr<Expression> hash_expr, LogicalAggregate *target_agg) {
	// Find ALL operators on the path between the PU table scan and the target aggregate
	// IMPORTANT: Use FindDirectPathToTableScan which STOPS at nested aggregates.
	// This prevents us from trying to propagate columns through nested aggregates,
	// which would fail because aggregates create a new column scope.
	vector<PathEntry> path_ops;

	// Find the path from aggregate to the specific table scan, stopping at nested aggregates
	if (!FindDirectPathToTableScan(target_agg, pu_get.table_index, path_ops, true)) {
		// No direct path found - table scan is not in this aggregate's direct subtree
		// (it may be behind a nested aggregate or in a different branch of the plan).
		// Return nullptr to signal that this aggregate should NOT be transformed.
#if PAC_DEBUG
		PAC_DEBUG_PRINT("PropagatePKThroughProjections: No direct path found from aggregate to table #" +
		                std::to_string(pu_get.table_index) + ", returning nullptr to skip transformation");
#endif
		return nullptr;
	}

#if PAC_DEBUG
	PAC_DEBUG_PRINT("PropagatePKThroughProjections: Found path with " + std::to_string(path_ops.size()) +
	                " operators from aggregate to table #" + std::to_string(pu_get.table_index));
#endif

	// If there are no operators on the path (aggregate reads directly from scan),
	// we still need to ensure the aggregate can access the new column.
	// The aggregate's child (the scan) has the column, so the binding is valid.
	// Just return the original hash expression - it references the scan's output correctly.
	if (path_ops.empty()) {
#if PAC_DEBUG
		PAC_DEBUG_PRINT("PropagatePKThroughProjections: Empty path - aggregate reads directly from scan");
		PAC_DEBUG_PRINT("PropagatePKThroughProjections: Hash expression: " + hash_expr->ToString());

		// Debug: verify the scan has the column we're referencing
		PAC_DEBUG_PRINT("PropagatePKThroughProjections: Scan table_index=" + std::to_string(pu_get.table_index) +
		                ", column_ids.size=" + std::to_string(pu_get.GetColumnIds().size()));
		auto bindings = pu_get.GetColumnBindings();
		string bindings_str;
		for (auto &b : bindings) {
			bindings_str += "[" + std::to_string(b.table_index) + "." + std::to_string(b.column_index) + "] ";
		}
		PAC_DEBUG_PRINT("PropagatePKThroughProjections: Scan output bindings: " + bindings_str);
#endif
		return hash_expr->Copy();
	}

	// path_ops are collected from bottom to top (closest to table scan first),
	// which is the order we want for processing

	// Extract all column references from the hash expression, including their types
	struct BindingInfo {
		ColumnBinding binding;
		LogicalType type;
	};
	vector<BindingInfo> hash_bindings;
	ExpressionIterator::EnumerateExpression(hash_expr, [&](Expression &expr) {
		if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
			auto &col_ref = expr.Cast<BoundColumnRefExpression>();
			hash_bindings.push_back({col_ref.binding, col_ref.return_type});
		}
	});

	// Map from old binding to new binding as we propagate through operators
	std::unordered_map<uint64_t, ColumnBinding> binding_map;
	// Also track types for each binding
	std::unordered_map<uint64_t, LogicalType> type_map;
	auto hash_binding_key = [](const ColumnBinding &b) -> uint64_t {
		return (static_cast<uint64_t>(b.table_index) << 32) | static_cast<uint64_t>(b.column_index);
	};

	// Initialize maps with identity mappings for the original bindings
	for (auto &info : hash_bindings) {
		auto key = hash_binding_key(info.binding);
		binding_map[key] = info.binding;
		type_map[key] = info.type;
	}

	// Propagate through each operator (from bottom to top, closest to table scan first)
	for (auto &entry : path_ops) {
		auto *op = entry.op;
		idx_t child_idx = entry.child_idx;

		if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			auto *proj = &op->Cast<LogicalProjection>();
#if PAC_DEBUG
			PAC_DEBUG_PRINT("PropagatePKThroughProjections: Processing projection #" +
			                std::to_string(proj->table_index));
#endif
			// For each binding we're tracking, add it to this projection if not already present
			std::unordered_map<uint64_t, ColumnBinding> new_binding_map;

			for (auto &kv : binding_map) {
				auto original_key = kv.first;
				auto old_binding = kv.second;

				// Check if this binding is already in the projection's expressions
				idx_t existing_idx = DConstants::INVALID_INDEX;
				for (idx_t i = 0; i < proj->expressions.size(); i++) {
					if (proj->expressions[i]->type == ExpressionType::BOUND_COLUMN_REF) {
						auto &expr_ref = proj->expressions[i]->Cast<BoundColumnRefExpression>();
						if (expr_ref.binding.table_index == old_binding.table_index &&
						    expr_ref.binding.column_index == old_binding.column_index) {
							existing_idx = i;
							break;
						}
					}
				}

				idx_t new_idx;
				if (existing_idx != DConstants::INVALID_INDEX) {
					// Already in projection, use existing index
					new_idx = existing_idx;
				} else {
					// Add to projection using the type we stored from the original expression
					auto col_type = type_map[original_key];
					auto col_binding = ColumnBinding(old_binding.table_index, old_binding.column_index);
					auto col_ref = make_uniq<BoundColumnRefExpression>(col_type, col_binding);
					proj->expressions.push_back(std::move(col_ref));
					new_idx = proj->expressions.size() - 1;
				}

				// Update mapping: old binding -> new binding in this projection's output
				ColumnBinding new_binding(proj->table_index, new_idx);
				new_binding_map[original_key] = new_binding;
			}

			binding_map = std::move(new_binding_map);
			// After modifying projection expressions, we MUST call ResolveOperatorTypes to sync the types vector
			proj->ResolveOperatorTypes();
		} else if (IsJoinWithProjectionMap(op->type)) {
			// Handle join operators - they may have projection maps that filter columns
			auto &join = op->Cast<LogicalJoin>();
			bool is_delim_join = (op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN);
#if PAC_DEBUG
			PAC_DEBUG_PRINT("PropagatePKThroughProjections: Processing join (type=" +
			                std::to_string(static_cast<int>(join.join_type)) + ", child_idx=" +
			                std::to_string(child_idx) + ", op_type=" + std::to_string(static_cast<int>(op->type)) +
			                ", is_delim_join=" + std::to_string(is_delim_join) + ")");
#endif

			// Check for incompatible join types:
			// - RIGHT_SEMI and RIGHT_ANTI joins only output columns from the RIGHT child
			// - If our column is on the LEFT side (child_idx=0), we cannot propagate through
			// NOTE: This applies to DELIM_JOIN too! DELIM_JOIN with RIGHT_SEMI still only
			// outputs columns from the right child (the subquery side)
			if ((join.join_type == JoinType::RIGHT_SEMI || join.join_type == JoinType::RIGHT_ANTI) && child_idx == 0) {
#if PAC_DEBUG
				PAC_DEBUG_PRINT("PropagatePKThroughProjections: Cannot propagate through RIGHT_SEMI/RIGHT_ANTI join "
				                "from left child - returning nullptr");
#endif
				return nullptr;
			}
			// Similarly, SEMI and ANTI joins only output columns from the LEFT child
			// If our column is on the RIGHT side (child_idx=1), we cannot propagate through
			if ((join.join_type == JoinType::SEMI || join.join_type == JoinType::ANTI) && child_idx == 1) {
#if PAC_DEBUG
				PAC_DEBUG_PRINT("PropagatePKThroughProjections: Cannot propagate through SEMI/ANTI join "
				                "from right child - returning nullptr");
#endif
				return nullptr;
			}

			// Special handling for DELIM_JOIN from left child with INNER/LEFT join type
			// DELIM_JOIN passes through all columns from the left child directly to its output
			// (The duplicate_eliminated_columns is for the RIGHT/subquery side to access outer columns via DELIM_GET)
			// So for left child propagation, we need to ensure the columns are in the left_projection_map
			if (is_delim_join && child_idx == 0) {
				// For DELIM_JOIN with INNER/LEFT join types, left child columns pass through
				// But we may need to add them to the left_projection_map if not already present

				// First, ensure columns are in the left_projection_map
				for (auto &kv : binding_map) {
					auto old_binding = kv.second;

					if (!join.left_projection_map.empty()) {
						// Check if this column is already in the projection map
						bool found = false;
						for (auto &proj_idx : join.left_projection_map) {
							if (proj_idx == old_binding.column_index) {
								found = true;
								break;
							}
						}
						if (!found) {
							// Add to projection map
							join.left_projection_map.push_back(old_binding.column_index);
#if PAC_DEBUG
							PAC_DEBUG_PRINT("PropagatePKThroughProjections: Added column " +
							                std::to_string(old_binding.column_index) +
							                " to DELIM_JOIN left_projection_map");
#endif
						}
					}
					// If left_projection_map is empty, all columns pass through by default
				}

				// Re-resolve types to pick up any changes
				join.ResolveOperatorTypes();

				// Verify bindings are in the output
				auto join_bindings = join.GetColumnBindings();
				std::unordered_map<uint64_t, ColumnBinding> new_binding_map;

				for (auto &kv : binding_map) {
					auto old_binding = kv.second;
					auto original_key = kv.first;

					// Find this binding in the join's output bindings
					bool found = false;
					for (idx_t i = 0; i < join_bindings.size(); i++) {
						if (join_bindings[i].table_index == old_binding.table_index &&
						    join_bindings[i].column_index == old_binding.column_index) {
							new_binding_map[original_key] = old_binding;
							found = true;
#if PAC_DEBUG
							PAC_DEBUG_PRINT("PropagatePKThroughProjections: DELIM_JOIN preserves left child binding [" +
							                std::to_string(old_binding.table_index) + "." +
							                std::to_string(old_binding.column_index) + "]");
#endif
							break;
						}
					}

					if (!found) {
						// Binding still not found - this can happen if the DELIM_JOIN's output
						// doesn't include this column even after adding to projection map.
						// This might occur with certain join types or plan structures.
						// Keep the old binding and continue - the column should be accessible
						// from the child operator.
#if PAC_DEBUG
						PAC_DEBUG_PRINT("PropagatePKThroughProjections: DELIM_JOIN left child binding [" +
						                std::to_string(old_binding.table_index) + "." +
						                std::to_string(old_binding.column_index) +
						                "] not found in output after adding to projection map - keeping old binding");
#endif
						new_binding_map[original_key] = old_binding;
					}
				}

				binding_map = std::move(new_binding_map);
				continue; // Skip the normal join handling below
			}

			// Determine which projection map applies based on which child the table is in
			vector<idx_t> *proj_map = nullptr;
			if (child_idx == 0) {
				proj_map = &join.left_projection_map;
			} else if (child_idx == 1) {
				proj_map = &join.right_projection_map;
			}

			if (proj_map && !proj_map->empty()) {
				// The join has a projection map - we need to ensure our columns are included
				for (auto &kv : binding_map) {
					auto old_binding = kv.second;

					// Find where this column index appears in the projection map
					bool found = false;
					for (idx_t i = 0; i < proj_map->size(); i++) {
						if ((*proj_map)[i] == old_binding.column_index) {
							found = true;
							break;
						}
					}

					if (!found) {
						// Column not in projection map - add it
						proj_map->push_back(old_binding.column_index);
#if PAC_DEBUG
						PAC_DEBUG_PRINT("PropagatePKThroughProjections: Added column " +
						                std::to_string(old_binding.column_index) + " to join projection map");
#endif
					}
				}
			}
			// Re-resolve types to pick up any changes from children
			join.ResolveOperatorTypes();

			// Update binding map to reflect column positions in join's output
			// Joins output columns from their children - we need to find where our binding
			// ends up in the join's output
			auto join_bindings = join.GetColumnBindings();
			std::unordered_map<uint64_t, ColumnBinding> new_binding_map;

			for (auto &kv : binding_map) {
				auto original_key = kv.first;
				auto old_binding = kv.second;

				// Find this binding in the join's output bindings
				bool found = false;
				for (idx_t i = 0; i < join_bindings.size(); i++) {
					if (join_bindings[i].table_index == old_binding.table_index &&
					    join_bindings[i].column_index == old_binding.column_index) {
						// The binding passes through unchanged in the join's output
						new_binding_map[original_key] = old_binding;
						found = true;
#if PAC_DEBUG
						PAC_DEBUG_PRINT("PropagatePKThroughProjections: Join preserves binding [" +
						                std::to_string(old_binding.table_index) + "." +
						                std::to_string(old_binding.column_index) + "]");
#endif
						break;
					}
				}

				if (!found) {
					// Binding not found in join output - this shouldn't happen if we added it correctly
					// Keep the old binding and hope for the best
#if PAC_DEBUG
					PAC_DEBUG_PRINT("PropagatePKThroughProjections: WARNING - binding [" +
					                std::to_string(old_binding.table_index) + "." +
					                std::to_string(old_binding.column_index) + "] not found in join output");
#endif
					new_binding_map[original_key] = old_binding;
				}
			}

			binding_map = std::move(new_binding_map);
		} else if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
			// Handle filter operators - they also have projection maps that can filter columns
			auto &filter = op->Cast<LogicalFilter>();
#if PAC_DEBUG
			PAC_DEBUG_PRINT("PropagatePKThroughProjections: Processing filter with projection_map.size=" +
			                std::to_string(filter.projection_map.size()));
#endif

			if (!filter.projection_map.empty()) {
				// The filter has a projection map - we need to ensure our columns are included
				for (auto &kv : binding_map) {
					auto old_binding = kv.second;

					// Find where this column index appears in the projection map
					bool found = false;
					for (idx_t i = 0; i < filter.projection_map.size(); i++) {
						if (filter.projection_map[i] == old_binding.column_index) {
							found = true;
							break;
						}
					}

					if (!found) {
						// Column not in projection map - add it
						filter.projection_map.push_back(old_binding.column_index);
#if PAC_DEBUG
						PAC_DEBUG_PRINT("PropagatePKThroughProjections: Added column " +
						                std::to_string(old_binding.column_index) + " to filter projection map");
#endif
					}
				}
			}
			// Re-resolve types to pick up any changes from children
			filter.ResolveOperatorTypes();

			// Update binding map to reflect column positions in filter's output
			// Filters pass through columns from their child - find where our binding ends up
			auto filter_bindings = filter.GetColumnBindings();
			std::unordered_map<uint64_t, ColumnBinding> new_binding_map;

			for (auto &kv : binding_map) {
				auto original_key = kv.first;
				auto old_binding = kv.second;

				// Find this binding in the filter's output bindings
				bool found = false;
				for (idx_t i = 0; i < filter_bindings.size(); i++) {
					if (filter_bindings[i].table_index == old_binding.table_index &&
					    filter_bindings[i].column_index == old_binding.column_index) {
						// The binding passes through unchanged in the filter's output
						new_binding_map[original_key] = old_binding;
						found = true;
#if PAC_DEBUG
						PAC_DEBUG_PRINT("PropagatePKThroughProjections: Filter preserves binding [" +
						                std::to_string(old_binding.table_index) + "." +
						                std::to_string(old_binding.column_index) + "]");
#endif
						break;
					}
				}

				if (!found) {
					// Binding not found in filter output - this shouldn't happen if we added it correctly
#if PAC_DEBUG
					PAC_DEBUG_PRINT("PropagatePKThroughProjections: WARNING - binding [" +
					                std::to_string(old_binding.table_index) + "." +
					                std::to_string(old_binding.column_index) + "] not found in filter output");
#endif
					new_binding_map[original_key] = old_binding;
				}
			}

			binding_map = std::move(new_binding_map);
		} else {
			// For all other operators, just re-resolve their types
			// so they pick up the new columns from their children
#if PAC_DEBUG
			PAC_DEBUG_PRINT("PropagatePKThroughProjections: Re-resolving types for operator type " +
			                std::to_string(static_cast<int>(op->type)));
#endif
			op->ResolveOperatorTypes();
		}
	}

	// Now update the hash expression to use the new bindings
	auto updated_hash = hash_expr->Copy();
	ExpressionIterator::EnumerateExpression(updated_hash, [&](Expression &expr) {
		if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
			auto &col_ref = expr.Cast<BoundColumnRefExpression>();
			auto key = hash_binding_key(col_ref.binding);
			if (binding_map.find(key) != binding_map.end()) {
#if PAC_DEBUG
				auto old_binding = col_ref.binding;
#endif
				col_ref.binding = binding_map[key];
#if PAC_DEBUG
				PAC_DEBUG_PRINT(
				    "PropagatePKThroughProjections: Updated binding [" + std::to_string(old_binding.table_index) + "." +
				    std::to_string(old_binding.column_index) + "] -> [" + std::to_string(col_ref.binding.table_index) +
				    "." + std::to_string(col_ref.binding.column_index) + "]");
#endif
			}
		}
	});

	return updated_hash;
}

} // namespace duckdb
