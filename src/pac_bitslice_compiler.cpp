//
// Created by ila on 12/21/25.
//

#include "include/pac_bitslice_compiler.hpp"
#include "include/pac_helpers.hpp"
#include "include/pac_compatibility_check.hpp"
#include "include/pac_compiler_helpers.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/optimizer/optimizer.hpp"

namespace duckdb {

// Helper: Get boolean setting value with default
static bool GetBooleanSetting(ClientContext &context, const string &setting_name, bool default_value) {
	Value val;
	if (context.TryGetCurrentSetting(setting_name, val) && !val.IsNull()) {
		return val.GetValue<bool>();
	}
	return default_value;
}

// Helper: Propagate PK columns through projections between table scan and aggregate
// Returns an updated hash expression that references the correct bindings at the aggregate level
static unique_ptr<Expression> PropagatePKThroughProjections(LogicalOperator &plan, LogicalGet &pu_get,
                                                            unique_ptr<Expression> hash_expr,
                                                            LogicalAggregate *target_agg) {
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

void ModifyPlanWithoutPU(const PACCompatibilityResult &check, OptimizerExtensionInput &input,
                         unique_ptr<LogicalOperator> &plan, const vector<string> &gets_missing,
                         const vector<string> &gets_present, const vector<string> &fk_path,
                         const vector<string> &privacy_units) {
	// Note: we assume we don't use rowid

	// Check if join elimination is enabled
	bool join_elimination = GetBooleanSetting(input.context, "pac_join_elimination", false);

#ifdef DEBUG
	Printer::Print("ModifyPlanWithoutPU: join_elimination = " + std::to_string(join_elimination));
	Printer::Print("ModifyPlanWithoutPU: privacy_units:");
	for (auto &pu : privacy_units) {
		Printer::Print("  " + pu);
	}
#endif

	// Create the necessary LogicalGets for missing tables
	// IMPORTANT: We need to preserve the FK path ordering when creating joins
	// Use fk_path as the canonical ordering, filter to only missing tables
	std::unordered_set<string> missing_set(gets_missing.begin(), gets_missing.end());

	// If join elimination is enabled, skip the PU tables themselves
	if (join_elimination) {
		for (auto &pu : privacy_units) {
			missing_set.erase(pu);
		}
	}

	// Check if any "missing" tables are actually already present in the plan
	// This can happen with correlated subqueries where the FK path starts from a subquery table
	// but the outer query already has the connecting table
	std::unordered_set<string> actually_present;
	for (auto &table : missing_set) {
		if (FindNodeRefByTable(&plan, table) != nullptr) {
			actually_present.insert(table);
#ifdef DEBUG
			Printer::Print("ModifyPlanWithoutPU: Table " + table + " marked as missing but already present in plan");
#endif
		}
	}

	// Remove actually present tables from missing_set
	for (auto &table : actually_present) {
		missing_set.erase(table);
	}

	std::unordered_map<string, unique_ptr<LogicalGet>> get_map;
	vector<string> ordered_table_names;
	// Track tables that were marked as missing but are actually present - these can serve as connection points
	vector<string> actually_present_in_fk_order;
	auto idx = GetNextTableIndex(plan);

	// Build ordered_table_names based on fk_path order, only including missing tables
	for (auto &table : fk_path) {
		if (missing_set.find(table) != missing_set.end()) {
			auto it = check.table_metadata.find(table);
			if (it == check.table_metadata.end()) {
				throw InternalException("PAC compiler: missing table metadata for missing GET: " + table);
			}
			vector<string> pks = it->second.pks;
			auto get = CreateLogicalGet(input.context, plan, table, idx);
			get_map[table] = std::move(get);
			ordered_table_names.push_back(table);
#ifdef DEBUG
			Printer::Print("ModifyPlanWithoutPU: Added table " + table + " to join chain");
#endif
			idx++;
		} else if (actually_present.find(table) != actually_present.end()) {
			// Track the order of already-present tables in the FK path
			actually_present_in_fk_order.push_back(table);
		}
	}

	// Find the unique_ptr reference to the existing table that connects to the missing tables
	// We need to find the last present table in the FK path, which is the connection point
	LogicalOperator *parent_join = nullptr;
	idx_t child_idx_in_parent = 0;
	unique_ptr<LogicalOperator> *target_ref = nullptr;

	// Find the last present table in the FK path (ordered)
	string connecting_table;
	if (!fk_path.empty()) {
		// Go through the FK path and find the last table that's in gets_present
		for (auto &table_in_path : fk_path) {
			bool is_present = false;
			for (auto &present : gets_present) {
				if (table_in_path == present) {
					is_present = true;
					connecting_table = table_in_path;
					break;
				}
			}
			// Once we hit a missing table, we've found our connection point
			if (!is_present) {
				break;
			}
		}
	}

	// If we couldn't find a connecting table, fall back to the first present table
	if (connecting_table.empty() && !gets_present.empty()) {
		connecting_table = gets_present[0];
	}

	target_ref = FindNodeRefByTable(&plan, connecting_table, &parent_join, &child_idx_in_parent);
	if (!target_ref) {
		throw InternalException("PAC compiler: could not find existing LogicalGet for table " +
		                        (connecting_table.empty() ? string("<none>") : connecting_table));
	}

	// Check if the target node is part of a LEFT JOIN
	bool is_left_join = false;
	JoinType original_join_type = JoinType::INNER;
	if (parent_join && parent_join->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		auto &parent_join_op = parent_join->Cast<LogicalComparisonJoin>();
		original_join_type = parent_join_op.join_type;
		is_left_join = (original_join_type == JoinType::LEFT);
	}

	// We create the joins in this order:
	// the existing node joins with ordered_table_names[0], then this join node joins with ordered_table_names[1], ...
	// All intermediate joins are INNER joins to reach the privacy unit
	unique_ptr<LogicalOperator> final_join;

	// Only create joins if there are tables to join
	if (!ordered_table_names.empty()) {
		unique_ptr<LogicalOperator> existing_node = (*target_ref)->Copy(input.context);

		for (size_t i = 0; i < ordered_table_names.size(); ++i) {
			auto &tbl_name = ordered_table_names[i];
			// move the LogicalGet from get_map into a unique_ptr<LogicalGet>
			unique_ptr<LogicalGet> right_op = std::move(get_map[tbl_name]);
			if (!right_op) {
				throw InternalException("PAC compiler: failed to transfer ownership of LogicalGet for " + tbl_name);
			}

			if (i == 0) {
				auto join = CreateLogicalJoin(check, input.context, std::move(existing_node), std::move(right_op));
				final_join = std::move(join);
			} else {
				auto join = CreateLogicalJoin(check, input.context, std::move(final_join), std::move(right_op));
				final_join = std::move(join);
			}
		}

		// Use ReplaceNode to insert the join and handle column binding remapping
		ReplaceNode(plan, *target_ref, final_join);

		// If the original was a LEFT JOIN, we need to restore it at the top level
		// The INNER joins to reach the PU are now inside the LEFT JOIN structure
		if (is_left_join && parent_join && parent_join->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
			auto &parent_join_op = parent_join->Cast<LogicalComparisonJoin>();
			// The join type should already be LEFT, but make sure the structure is correct
			// The INNER join chain is now in place of the original child
			parent_join_op.join_type = original_join_type;
		}
	}

#if DEBUG
	plan->Print();

#endif

	// Build hash expressions for each privacy unit
	vector<unique_ptr<Expression>> hash_exprs;

	for (auto &privacy_unit : privacy_units) {
		unique_ptr<Expression> hash_input_expr;

		if (join_elimination) {
			// Find the last table in the FK chain for this PU
			string last_table_name;
			if (!ordered_table_names.empty()) {
				last_table_name = ordered_table_names.back();
			} else if (!actually_present_in_fk_order.empty()) {
				// No new joins were added, but we have tables that were already present in the FK path
				// Use the last one as it's closest to the PU
				last_table_name = actually_present_in_fk_order.back();
			} else {
				last_table_name = connecting_table;
			}

			// Check if the PU table itself is in the query
			// If it is, it's simpler and more reliable to use it directly
			unique_ptr<LogicalOperator> *pu_table_ref = FindNodeRefByTable(&plan, privacy_unit);

			if (pu_table_ref && pu_table_ref->get()) {
				// PU table is in the query - use it directly (simpler path)
				auto pu_get = &pu_table_ref->get()->Cast<LogicalGet>();

				vector<string> pu_pks;
				for (auto &pk : check.table_metadata.at(privacy_unit).pks) {
					pu_pks.push_back(pk);
				}

				// Ensure PK columns are projected
				for (auto &pk : pu_pks) {
					idx_t proj_idx = EnsureProjectedColumn(*pu_get, pk);
					if (proj_idx == DConstants::INVALID_INDEX) {
						throw InternalException("PAC compiler: failed to project PK column " + pk);
					}
				}

				// Build hash from PU's PK columns
				auto base_hash_expr = BuildXorHashFromPKs(input, *pu_get, pu_pks);

				// Find the aggregate to propagate through
				auto *agg = FindTopAggregate(plan);
				if (!agg) {
					throw InternalException("PAC compiler: could not find aggregate");
				}

				// Propagate through projections
				hash_input_expr = PropagatePKThroughProjections(*plan, *pu_get, std::move(base_hash_expr), agg);
			} else {
				// PU table is NOT in the query - use FK column from connecting table
				// Find the FK column(s) that reference the PU
				auto it = check.table_metadata.find(last_table_name);
				if (it == check.table_metadata.end()) {
					throw InternalException("PAC compiler: missing metadata for table " + last_table_name);
				}

				vector<string> fk_cols;
				for (auto &fk : it->second.fks) {
					if (fk.first == privacy_unit) {
						fk_cols = fk.second;
						break;
					}
				}

				if (fk_cols.empty()) {
					throw InternalException("PAC compiler: no FK found from " + last_table_name + " to " +
					                        privacy_unit);
				}

				// Find the LogicalGet for the last table and ensure FK columns are projected
				unique_ptr<LogicalOperator> *last_table_ref = FindNodeRefByTable(&plan, last_table_name);
				if (!last_table_ref || !last_table_ref->get()) {
					throw InternalException("PAC compiler: could not find LogicalGet for last table " +
					                        last_table_name);
				}
				auto last_get = &last_table_ref->get()->Cast<LogicalGet>();

				// Ensure FK columns are projected in the table scan
				for (auto &fk_col : fk_cols) {
					idx_t proj_idx = EnsureProjectedColumn(*last_get, fk_col);
					if (proj_idx == DConstants::INVALID_INDEX) {
						throw InternalException("PAC compiler: failed to project FK column " + fk_col);
					}
				}

				// Build hash expression from FK columns at the table scan level
				auto base_hash_expr = BuildXorHashFromPKs(input, *last_get, fk_cols);

				// Find the aggregate to propagate through
				auto *agg = FindTopAggregate(plan);
				if (!agg) {
					throw InternalException("PAC compiler: could not find aggregate");
				}

				// Use PropagatePKThroughProjections to propagate the hash expression through all projections
				hash_input_expr = PropagatePKThroughProjections(*plan, *last_get, std::move(base_hash_expr), agg);
			}
		} else {
			// Original behavior: locate the privacy-unit LogicalGet
			unique_ptr<LogicalOperator> *pu_ref = nullptr;
			if (!privacy_unit.empty()) {
				pu_ref = FindNodeRefByTable(&plan, privacy_unit);
			}
			if (!pu_ref || !pu_ref->get()) {
				throw InternalException("PAC compiler: could not find LogicalGet for privacy unit " + privacy_unit);
			}
			auto pu_get = &pu_ref->get()->Cast<LogicalGet>();

			vector<string> pu_pks;
			for (auto &pk : check.table_metadata.at(privacy_unit).pks) {
				pu_pks.push_back(pk);
			}

			hash_input_expr = BuildXorHashFromPKs(input, *pu_get, pu_pks);
		}

		hash_exprs.push_back(std::move(hash_input_expr));
	}

	// Combine all hash expressions with AND
	auto combined_hash_expr = BuildAndFromHashes(input, hash_exprs);

	// Find ALL aggregate nodes in the plan (not just the topmost one)
	// For queries with nested aggregates like TPC-H Q13, we need to find all of them
	// but only modify the BOTTOMMOST one (closest to the PU table)
	vector<LogicalAggregate *> all_aggregates;
	FindAllAggregates(plan, all_aggregates);

	if (all_aggregates.empty()) {
		throw InternalException("PAC Compiler: no aggregate nodes found in plan");
	}

	// For nested aggregates, we only want to modify the bottommost aggregate
	// (the one closest to the base table scan). The outer aggregates operate on
	// already-aggregated data and should be left as-is.
	// Since FindAllAggregates does a pre-order traversal, the last aggregate in the
	// vector is the bottommost one.
	auto *bottommost_agg = all_aggregates.back();
	ModifyAggregatesWithPacFunctions(input, bottommost_agg, combined_hash_expr);
}

void ModifyPlanWithPU(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                      const vector<string> &pu_table_names, const PACCompatibilityResult &check) {

	// Find ALL aggregate nodes in the plan first
	// For nested aggregates (like Q13), we need to use the bottommost one
	vector<LogicalAggregate *> all_aggregates;
	FindAllAggregates(plan, all_aggregates);

	if (all_aggregates.empty()) {
		throw InternalException("PAC Compiler: no aggregate nodes found in plan");
	}

	// Use the bottommost aggregate (closest to table scans) for both propagation and modification
	auto *target_agg = all_aggregates.back();

	// Build hash expressions for each privacy unit
	vector<unique_ptr<Expression>> hash_exprs;

	for (auto &pu_table_name : pu_table_names) {
		auto pu_scan_ptr = FindPrivacyUnitGetNode(plan, pu_table_name);
		auto &get = pu_scan_ptr->get()->Cast<LogicalGet>();

		// Determine if we should use rowid or PKs
		bool use_rowid = false;
		vector<string> pks;

		auto it = check.table_metadata.find(pu_table_name);
		if (it != check.table_metadata.end() && !it->second.pks.empty()) {
			pks = it->second.pks;
		} else {
			use_rowid = true;
		}

		if (use_rowid) {
			AddRowIDColumn(get);
		} else {
			// Ensure primary key columns are present in the LogicalGet (add them if necessary)
			AddPKColumns(get, pks);
		}

		// Build the hash expression for this PU
		unique_ptr<Expression> hash_input_expr;
		if (use_rowid) {
			// rowid is the last column added
			auto rowid_binding = ColumnBinding(get.table_index, get.GetColumnIds().size() - 1);
			auto rowid_col = make_uniq<BoundColumnRefExpression>(LogicalType::BIGINT, rowid_binding);
			auto bound_hash = input.optimizer.BindScalarFunction("hash", std::move(rowid_col));
			hash_input_expr = std::move(bound_hash);
		} else {
			hash_input_expr = BuildXorHashFromPKs(input, get, pks);
		}

		// Propagate the hash expression through all projections between table scan and the target aggregate
		hash_input_expr = PropagatePKThroughProjections(*plan, get, std::move(hash_input_expr), target_agg);

		hash_exprs.push_back(std::move(hash_input_expr));
	}

	// Combine all hash expressions with AND
	auto combined_hash_expr = BuildAndFromHashes(input, hash_exprs);

	// Modify the bottommost aggregate with PAC functions
	ModifyAggregatesWithPacFunctions(input, target_agg, combined_hash_expr);
}

void CompilePacBitsliceQuery(const PACCompatibilityResult &check, OptimizerExtensionInput &input,
                             unique_ptr<LogicalOperator> &plan, const vector<string> &privacy_units,
                             const string &query, const string &query_hash) {

#ifdef DEBUG
	Printer::Print("CompilePacBitsliceQuery called for " + std::to_string(privacy_units.size()) +
	               " PUs, hash=" + query_hash);
	for (auto &pu : privacy_units) {
		Printer::Print("  PU: " + pu);
	}
#endif

	// Generate filename with all PU names concatenated
	string path = GetPacCompiledPath(input.context, ".");
	if (!path.empty() && path.back() != '/') {
		path.push_back('/');
	}
	string pu_names_joined;
	for (size_t i = 0; i < privacy_units.size(); ++i) {
		if (i > 0) {
			pu_names_joined += "_";
		}
		pu_names_joined += privacy_units[i];
	}
	string filename = path + pu_names_joined + "_" + query_hash + "_bitslice.sql";

	// The bitslice compiler works in the following way:
	// a) the query scans PU table(s):
	// a.1) each PU table has 1 PK: we hash it
	// a.2) each PU table has multiple PKs: we XOR them and hash the result
	// a.3) each PU table has no PK: we hash rowid
	// a.4) we AND all the hashes together for multiple PUs
	// b) the query does not scan PU table(s):
	// b.1) we follow the FK path to find the PK(s) of each PU table
	// b.2) we join the chain of tables from the scanned table to each PU table (deduplicating)
	// b.3) we hash the PK(s) as in a) and AND them together

	bool pu_present_in_tree = false;

	if (!check.scanned_pu_tables.empty()) {
		pu_present_in_tree = true;
	}

	// Replan with selected optimizers disabled
	ReplanWithoutOptimizers(input.context, input.context.GetCurrentQuery(), plan);

	// Build two vectors: present (GETs already in the plan) and missing (GETs to create)
	vector<string> gets_present;
	vector<string> gets_missing;

	// For multi-PU support, we need to gather FK paths and missing tables for all PUs
	// and deduplicate the tables that need to be joined
	std::unordered_set<string> all_missing_tables;
	vector<string> fk_path_to_use; // We'll use the first FK path as the base

	if (pu_present_in_tree) {
		// Case a) query scans PU table(s) - all PUs are in scanned_pu_tables
		ModifyPlanWithPU(input, plan, check.scanned_pu_tables, check);
	} else if (!check.fk_paths.empty()) {
		// Case b) query does not scan PU table(s): follow FK paths
		string start_table;
		vector<string> target_pus;
		PopulateGetsFromFKPath(check, gets_present, gets_missing, start_table, target_pus);

		// Collect all missing tables across all FK paths (deduplicate)
		for (auto &table : gets_missing) {
			all_missing_tables.insert(table);
		}

		// Extract the ordered fk_path from the compatibility result (use first path as base)
		auto it = check.fk_paths.find(start_table);
		if (it == check.fk_paths.end() || it->second.empty()) {
			throw InternalException("PAC compiler: expected fk_path for start table " + start_table);
		}
		fk_path_to_use = it->second;

		// Convert set back to vector for ModifyPlanWithoutPU
		vector<string> unique_gets_missing(all_missing_tables.begin(), all_missing_tables.end());

#ifdef DEBUG
		Printer::Print("PAC bitslice: FK path detection for multi-PU");
		Printer::Print("start_table: " + start_table);
		Printer::Print("target_pus:");
		for (auto &pu : target_pus) {
			Printer::Print("  " + pu);
		}
		Printer::Print("fk_path:");
		for (auto &p : fk_path_to_use) {
			Printer::Print("  " + p);
		}
		Printer::Print("gets_present:");
		for (auto &g : gets_present) {
			Printer::Print("  " + g);
		}
		Printer::Print("unique_gets_missing:");
		for (auto &g : unique_gets_missing) {
			Printer::Print("  " + g);
		}
#endif

		ModifyPlanWithoutPU(check, input, plan, unique_gets_missing, gets_present, fk_path_to_use, privacy_units);
	}

	plan->ResolveOperatorTypes();
	plan->Verify(input.context);
#ifdef DEBUG
	plan->Print();
#endif
}

} // namespace duckdb
