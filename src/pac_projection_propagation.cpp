//
// Created by ila on 1/16/26.
//

#include "include/pac_projection_propagation.hpp"
#include "include/pac_helpers.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"

namespace duckdb {

unique_ptr<Expression> PropagatePKThroughProjections(LogicalOperator &plan, LogicalGet &pu_get,
                                                     unique_ptr<Expression> hash_expr, LogicalAggregate *target_agg) {
	// Find all projections between the PU table scan and the target aggregate
	vector<LogicalProjection *> projections;

	// Walk up from the aggregate to find projections
	LogicalOperator *current = target_agg;
	while (current) {
		if (current->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			projections.push_back(&current->Cast<LogicalProjection>());
		}

		// Check if we've reached the table scan
		if (current->type == LogicalOperatorType::LOGICAL_GET) {
			auto &get = current->Cast<LogicalGet>();
			if (get.table_index == pu_get.table_index) {
				break;
			}
		}

		// Move to first child (depth-first search)
		if (!current->children.empty()) {
			current = current->children[0].get();
		} else {
			break;
		}
	}

	// If there are no projections, return the original hash expression
	if (projections.empty()) {
		return hash_expr->Copy();
	}

	// Process projections from bottom to top (closest to table scan first)
	// We need to add the hash expression columns to each projection
	std::reverse(projections.begin(), projections.end());

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

	// Map from old binding to new binding as we propagate through projections
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

	// Propagate through each projection
	for (auto *proj : projections) {
#ifdef DEBUG
		Printer::Print("PropagatePKThroughProjections: Processing projection #" + std::to_string(proj->table_index));
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
