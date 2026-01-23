//
// Created by ila on 1/16/26.
//

#include "include/pac_projection_propagation.hpp"
#include "include/pac_helpers.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
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

static bool FindPathToTableScan(LogicalOperator *current, idx_t target_table_index, vector<PathEntry> &path) {
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

	// Try each child
	for (idx_t child_idx = 0; child_idx < current->children.size(); child_idx++) {
		if (FindPathToTableScan(current->children[child_idx].get(), target_table_index, path)) {
			// Found it in this subtree - add current to path ONLY if it's not an aggregate
			// The aggregate is the starting point and shouldn't be included in the path
			// We only want operators BETWEEN the aggregate and the table scan
			if (current->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
				path.push_back({current, child_idx});
			}
			return true;
		}
	}

	return false;
}

// Similar to FindPathToTableScan, but STOPS at nested aggregates.
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
#ifdef DEBUG
		Printer::Print("PropagatePKThroughProjections: No direct path found from aggregate to table #" +
		               std::to_string(pu_get.table_index) + ", returning nullptr to skip transformation");
#endif
		return nullptr;
	}

#ifdef DEBUG
	Printer::Print("PropagatePKThroughProjections: Found path with " + std::to_string(path_ops.size()) +
	               " operators from aggregate to table #" + std::to_string(pu_get.table_index));
#endif

	// If there are no operators on the path (aggregate reads directly from scan),
	// we still need to ensure the aggregate can access the new column.
	// The aggregate's child (the scan) has the column, so the binding is valid.
	// Just return the original hash expression - it references the scan's output correctly.
	if (path_ops.empty()) {
#ifdef DEBUG
		Printer::Print("PropagatePKThroughProjections: Empty path - aggregate reads directly from scan");
		Printer::Print("PropagatePKThroughProjections: Hash expression: " + hash_expr->ToString());

		// Debug: verify the scan has the column we're referencing
		Printer::Print("PropagatePKThroughProjections: Scan table_index=" + std::to_string(pu_get.table_index) +
		               ", column_ids.size=" + std::to_string(pu_get.GetColumnIds().size()));
		auto bindings = pu_get.GetColumnBindings();
		string bindings_str;
		for (auto &b : bindings) {
			bindings_str += "[" + std::to_string(b.table_index) + "." + std::to_string(b.column_index) + "] ";
		}
		Printer::Print("PropagatePKThroughProjections: Scan output bindings: " + bindings_str);
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
#ifdef DEBUG
			Printer::Print("PropagatePKThroughProjections: Processing projection #" +
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
			// Projections resolve their own types from expressions, so ResolveOperatorTypes is automatic
		} else if (IsJoinWithProjectionMap(op->type)) {
			// Handle join operators - they may have projection maps that filter columns
			auto &join = op->Cast<LogicalJoin>();
#ifdef DEBUG
			Printer::Print("PropagatePKThroughProjections: Processing join (type=" +
			               std::to_string(static_cast<int>(join.join_type)) +
			               ", child_idx=" + std::to_string(child_idx) + ")");
#endif

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
#ifdef DEBUG
						Printer::Print("PropagatePKThroughProjections: Added column " +
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
#ifdef DEBUG
						Printer::Print("PropagatePKThroughProjections: Join preserves binding [" +
						               std::to_string(old_binding.table_index) + "." +
						               std::to_string(old_binding.column_index) + "]");
#endif
						break;
					}
				}

				if (!found) {
					// Binding not found in join output - this shouldn't happen if we added it correctly
					// Keep the old binding and hope for the best
#ifdef DEBUG
					Printer::Print("PropagatePKThroughProjections: WARNING - binding [" +
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
#ifdef DEBUG
			Printer::Print("PropagatePKThroughProjections: Processing filter with projection_map.size=" +
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
#ifdef DEBUG
						Printer::Print("PropagatePKThroughProjections: Added column " +
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
#ifdef DEBUG
						Printer::Print("PropagatePKThroughProjections: Filter preserves binding [" +
						               std::to_string(old_binding.table_index) + "." +
						               std::to_string(old_binding.column_index) + "]");
#endif
						break;
					}
				}

				if (!found) {
					// Binding not found in filter output - this shouldn't happen if we added it correctly
#ifdef DEBUG
					Printer::Print("PropagatePKThroughProjections: WARNING - binding [" +
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
#ifdef DEBUG
			Printer::Print("PropagatePKThroughProjections: Re-resolving types for operator type " +
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
				auto old_binding = col_ref.binding;
				col_ref.binding = binding_map[key];
#ifdef DEBUG
				Printer::Print(
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
