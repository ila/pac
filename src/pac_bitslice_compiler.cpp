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

	std::unordered_map<string, unique_ptr<LogicalGet>> get_map;
	vector<string> ordered_table_names;
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
			} else {
				last_table_name = connecting_table;
			}

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
				throw InternalException("PAC compiler: no FK found from " + last_table_name + " to " + privacy_unit);
			}

			// Find the LogicalGet for the last table and ensure FK columns are projected
			unique_ptr<LogicalOperator> *last_table_ref = FindNodeRefByTable(&plan, last_table_name);
			if (!last_table_ref || !last_table_ref->get()) {
				throw InternalException("PAC compiler: could not find LogicalGet for last table " + last_table_name);
			}
			auto last_get = &last_table_ref->get()->Cast<LogicalGet>();

			// Ensure FK columns are projected in the table scan
			for (auto &fk_col : fk_cols) {
				idx_t proj_idx = EnsureProjectedColumn(*last_get, fk_col);
				if (proj_idx == DConstants::INVALID_INDEX) {
					throw InternalException("PAC compiler: failed to project FK column " + fk_col);
				}
			}

			// Now we need to find the projection that feeds the aggregate and add the FK columns there
			auto *agg = FindTopAggregate(plan);
			if (!agg || agg->children.empty()) {
				throw InternalException("PAC compiler: could not find aggregate");
			}

			// The aggregate's input should be a projection (either #7 or #8 based on the plan output)
			LogicalOperator *agg_input = agg->children[0].get();
			LogicalProjection *input_projection = nullptr;

			// Find the DEEPEST (last) projection in the aggregate's input chain
			// We need to add the FK column to the projection closest to the join
			LogicalOperator *current = agg_input;
			while (current) {
				if (current->type == LogicalOperatorType::LOGICAL_PROJECTION) {
					input_projection = &current->Cast<LogicalProjection>();
					// Keep going deeper to find the last projection
					if (!current->children.empty()) {
						LogicalOperator *next = current->children[0].get();
						// Check if the next operator is also a projection
						if (next->type == LogicalOperatorType::LOGICAL_PROJECTION) {
							current = next;
							continue;
						}
					}
					// Found the deepest projection
					break;
				}
				if (!current->children.empty()) {
					current = current->children[0].get();
				} else {
					break;
				}
			}

			// Handle two cases: with projections and without projections
			if (!input_projection) {
				// Case 1: No projection in the aggregate's input chain
				// We can directly reference FK columns from the table scan
				// This is the simple case - just use BuildXorHashFromPKs like the original code did
				hash_input_expr = BuildXorHashFromPKs(input, *last_get, fk_cols);
			} else {
				// Case 2: There are projections in the aggregate's input chain
				// We need to add FK columns to projections and propagate them through all layers
				vector<unique_ptr<Expression>> fk_col_exprs;
				idx_t projection_table_idx = input_projection->table_index;

				for (auto &fk_col : fk_cols) {
					// Find this FK column in the orders table scan
					idx_t proj_idx = DConstants::INVALID_INDEX;
					for (idx_t i = 0; i < last_get->GetColumnIds().size(); i++) {
						auto col_idx = last_get->GetColumnIds()[i];
						if (!col_idx.IsVirtualColumn()) {
							idx_t primary = col_idx.GetPrimaryIndex();
							if (primary < last_get->names.size() && last_get->names[primary] == fk_col) {
								proj_idx = i;
								break;
							}
						}
					}

					if (proj_idx == DConstants::INVALID_INDEX) {
						throw InternalException("PAC compiler: FK column not found in table scan");
					}

					auto col_binding = ColumnBinding(last_get->table_index, proj_idx);
					auto col_index_obj = last_get->GetColumnIds()[proj_idx];
					auto &col_type = last_get->GetColumnType(col_index_obj);

					// Add this as a new expression in the deepest projection
					input_projection->expressions.push_back(make_uniq<BoundColumnRefExpression>(col_type, col_binding));

					// Now propagate this column through all parent projections up to the aggregate
					idx_t deepest_proj_idx = input_projection->expressions.size() - 1;
					LogicalProjection *current_proj = input_projection;

					// Walk up from the deepest projection to the aggregate
					LogicalOperator *parent = agg_input;
					while (parent && parent != current_proj) {
						if (parent->type == LogicalOperatorType::LOGICAL_PROJECTION) {
							auto &parent_proj = parent->Cast<LogicalProjection>();
							// Check if parent_proj's child is current_proj
							if (!parent->children.empty() && parent->children[0].get() == current_proj) {
								// Add a pass-through expression for the FK column
								auto pass_through_binding = ColumnBinding(current_proj->table_index, deepest_proj_idx);
								parent_proj.expressions.push_back(
								    make_uniq<BoundColumnRefExpression>(col_type, pass_through_binding));

								// Update for next iteration
								deepest_proj_idx = parent_proj.expressions.size() - 1;
								current_proj = &parent_proj;

								// Continue searching for more parent projections
								parent = agg_input;
								continue;
							}
						}

						// Traverse down to find projections
						if (!parent->children.empty()) {
							bool found_child = false;
							for (auto &child : parent->children) {
								if (child.get() == current_proj) {
									found_child = true;
									break;
								}
							}
							if (found_child) {
								break;
							}
							parent = parent->children[0].get();
						} else {
							break;
						}
					}

					// Create a reference to the projected column for our hash expression
					// Use the column from the projection that directly feeds the aggregate
					LogicalProjection *agg_input_proj = nullptr;
					if (agg_input->type == LogicalOperatorType::LOGICAL_PROJECTION) {
						agg_input_proj = &agg_input->Cast<LogicalProjection>();
					}

					if (agg_input_proj) {
						idx_t final_proj_idx = agg_input_proj->expressions.size() - 1;
						auto new_col_binding = ColumnBinding(agg_input_proj->table_index, final_proj_idx);
						fk_col_exprs.push_back(make_uniq<BoundColumnRefExpression>(col_type, new_col_binding));
					} else {
						// Fallback: use the deepest projection
						auto new_col_binding = ColumnBinding(projection_table_idx, deepest_proj_idx);
						fk_col_exprs.push_back(make_uniq<BoundColumnRefExpression>(col_type, new_col_binding));
					}
				}

				// Build XOR of FK columns
				unique_ptr<Expression> xor_expr;
				if (fk_col_exprs.size() == 1) {
					xor_expr = std::move(fk_col_exprs[0]);
				} else {
					auto left = std::move(fk_col_exprs[0]);
					for (size_t i = 1; i < fk_col_exprs.size(); ++i) {
						auto right = std::move(fk_col_exprs[i]);
						left = input.optimizer.BindScalarFunction("^", std::move(left), std::move(right));
					}
					xor_expr = std::move(left);
				}

				// Hash the XOR result
				hash_input_expr = input.optimizer.BindScalarFunction("hash", std::move(xor_expr));
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

		hash_exprs.push_back(std::move(hash_input_expr));
	}

	// Combine all hash expressions with AND
	auto combined_hash_expr = BuildAndFromHashes(input, hash_exprs);

	// Find ALL aggregate nodes in the plan (not just the topmost one)
	// This is needed for queries with nested aggregates like TPC-H Q13
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
		if (i > 0)
			pu_names_joined += "_";
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
	vector<string> fk_path_to_use; // We'll use the first FK path as the base path

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
