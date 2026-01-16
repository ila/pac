//
// Created by ila on 12/21/25.
//

#include "include/pac_bitslice_compiler.hpp"
#include "include/pac_helpers.hpp"
#include "include/pac_compatibility_check.hpp"
#include "include/pac_compiler_helpers.hpp"
#include "include/pac_projection_propagation.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
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

/**
 * ModifyPlanWithoutPU: Transforms a query plan when the privacy unit (PU) table is NOT scanned directly
 *
 * Purpose: When the query doesn't directly scan the PU table, we need to join tables along the FK path
 * from the scanned tables to the PU table, then build hash expressions from the FK columns that reference the PU.
 *
 * Arguments:
 * @param check - Compatibility check result containing table metadata and FK relationships
 * @param input - Optimizer extension input containing context and optimizer
 * @param plan - The logical plan to modify
 * @param gets_missing - Tables in the FK path that are NOT in the original query (need to be added as joins)
 * @param gets_present - Tables in the FK path that ARE already in the original query
 * @param fk_path - Ordered list of tables from the scanned table to the PU (e.g., [lineitem, orders, customer])
 * @param privacy_units - List of privacy unit table names (e.g., ["customer"])
 *
 * Logic:
 * 1. Identify which tables need to be joined (those in gets_missing)
 * 2. Find the "connecting table" - the last present table in the FK path (e.g., lineitem)
 * 3. For each instance of the connecting table in the plan (handles correlated subqueries):
 *    - Create a fresh join chain: connecting_table -> missing_table_1 -> ... -> missing_table_N
 *    - Replace the connecting table with this join chain
 *    - Track the table index of each "orders" table (or equivalent) for hash generation
 * 4. Find all aggregates that have FK-linked tables in their subtree
 * 5. For each aggregate:
 *    - Determine which "orders" table instance it should use (critical for correlated subqueries)
 *    - Build hash expression from the FK columns in "orders" that reference the PU
 *    - Propagate the hash expression through projections
 *    - Transform the aggregate to use PAC functions (pac_sum, pac_avg, etc.)
 *
 * Correlated Subquery Handling:
 * - If a table appears in BOTH outer query and inner subquery, we find ALL instances and add joins to each
 * - Each aggregate gets the hash from its "closest" orders table (not crossing subquery boundaries)
 * - Example: In TPC-H Q17, lineitem appears in both outer and inner aggregate:
 *   * Outer aggregate gets hash from outer orders table
 *   * Inner aggregate gets hash from inner orders table (same lineitem.l_partkey correlation)
 *
 * Join Addition Rules:
 * - If the table referencing the PU is in both outer and subquery: join and add pac_aggregate in BOTH
 * - If inner query has no aggregate: still join if it references a table in the FK path to PU
 */
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

	// If STILL no connecting table, it means we're querying a leaf table that's not in the FK path
	// In this case, we should use the scanned table itself as the connection point
	if (connecting_table.empty()) {
		// Find any scanned non-PU table
		if (!check.scanned_non_pu_tables.empty()) {
			connecting_table = check.scanned_non_pu_tables[0];
		}
	}

	// Find ALL instances of the connecting table (for correlated subqueries, there may be multiple)
	vector<unique_ptr<LogicalOperator> *> all_connecting_nodes;
	if (!connecting_table.empty()) {
		FindAllNodesByTable(&plan, connecting_table, all_connecting_nodes);
	}

	// If no connecting nodes found, we may be in a case where the query scans a leaf table
	// that's not directly in gets_present but is in the FK chain
	if (all_connecting_nodes.empty() && !gets_present.empty()) {
		// Try to find any present table in the plan
		for (auto &present : gets_present) {
			FindAllNodesByTable(&plan, present, all_connecting_nodes);
			if (!all_connecting_nodes.empty()) {
				connecting_table = present;
				break;
			}
		}
	}

	if (all_connecting_nodes.empty() && !connecting_table.empty()) {
		throw InternalException("PAC compiler: could not find any LogicalGet for table " + connecting_table);
	}

	if (all_connecting_nodes.empty()) {
		throw InternalException("PAC compiler: could not find any connecting table in the plan");
	}

	// For each instance of the connecting table, add the join chain
	// Store the mapping from each instance to its corresponding orders table for hash generation
	std::unordered_map<idx_t, idx_t> connecting_table_to_orders_table;

	for (auto *target_ref : all_connecting_nodes) {
		// Get the table index of this instance
		auto &target_op = (*target_ref)->Cast<LogicalGet>();
		idx_t connecting_table_idx = target_op.table_index;

		// Only create joins if there are tables to join
		if (!ordered_table_names.empty()) {
			unique_ptr<LogicalOperator> existing_node = (*target_ref)->Copy(input.context);

			// Create fresh LogicalGet nodes for this instance
			auto local_idx = GetNextTableIndex(plan);
			std::unordered_map<string, unique_ptr<LogicalGet>> local_get_map;

			for (auto &table : ordered_table_names) {
				auto it = check.table_metadata.find(table);
				if (it == check.table_metadata.end()) {
					throw InternalException("PAC compiler: missing table metadata for table: " + table);
				}
				vector<string> pks = it->second.pks;
				auto get = CreateLogicalGet(input.context, plan, table, local_idx);

				// Remember the table index of the orders table for this instance
				if (table == "orders") {
					connecting_table_to_orders_table[connecting_table_idx] = local_idx;
				}

				local_get_map[table] = std::move(get);
				local_idx++;
			}

			unique_ptr<LogicalOperator> final_join;
			for (size_t i = 0; i < ordered_table_names.size(); ++i) {
				auto &tbl_name = ordered_table_names[i];
				unique_ptr<LogicalGet> right_op = std::move(local_get_map[tbl_name]);
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

			// Replace this instance with the join chain
			ReplaceNode(plan, *target_ref, final_join);
		} else {
			// No tables to join - the FK-linked table (e.g., orders) must already be in the plan
			// Find it and map the connecting table to it
			// Search BOTH gets_present AND actually_present tables (tables marked missing but found in plan)
			vector<string> all_present_tables;
			all_present_tables.insert(all_present_tables.end(), gets_present.begin(), gets_present.end());
			all_present_tables.insert(all_present_tables.end(), actually_present.begin(), actually_present.end());

			for (auto &present_table : all_present_tables) {
				// Check if this present table has an FK to the PU
				auto it = check.table_metadata.find(present_table);
				if (it != check.table_metadata.end()) {
					bool has_fk_to_pu = false;
					for (auto &fk : it->second.fks) {
						for (auto &pu : privacy_units) {
							if (fk.first == pu) {
								has_fk_to_pu = true;
								break;
							}
						}
						if (has_fk_to_pu)
							break;
					}

					if (has_fk_to_pu) {
						// This is the table we need for hashing - find its table index
						vector<unique_ptr<LogicalOperator> *> fk_table_nodes;
						FindAllNodesByTable(&plan, present_table, fk_table_nodes);
						if (!fk_table_nodes.empty()) {
							// For correlated subqueries, we need to map to ALL instances of the FK table
							// not just the first one, so the aggregate can find its matching instance
							for (auto *fk_node : fk_table_nodes) {
								auto &fk_table_get = fk_node->get()->Cast<LogicalGet>();
								// Map connecting table to this FK table instance
								// Use the FK table's own index as both key and value since it IS the orders table
								connecting_table_to_orders_table[fk_table_get.table_index] = fk_table_get.table_index;
#ifdef DEBUG
								Printer::Print("ModifyPlanWithoutPU: Mapped FK table " + present_table + " #" +
								               std::to_string(fk_table_get.table_index) + " to itself for hashing");
#endif
							}
							break;
						}
					}
				}
			}
		}
	}

#if DEBUG
	plan->Print();
#endif

	// Now find all aggregates and modify them with PAC functions
	// Each aggregate needs a hash expression based on the orders table in its subtree
	vector<LogicalAggregate *> all_aggregates;
	FindAllAggregates(plan, all_aggregates);

	if (all_aggregates.empty()) {
		throw InternalException("PAC Compiler: no aggregate nodes found in plan");
	}

	// For correlated subqueries, the FK-linked table (e.g., lineitem) may appear in multiple contexts
	// Filter aggregates to only those that have FK-linked tables in their subtree
	std::unordered_set<string> fk_linked_tables(gets_present.begin(), gets_present.end());

	vector<string> fk_linked_tables_vec(fk_linked_tables.begin(), fk_linked_tables.end());
	vector<LogicalAggregate *> target_aggregates = FilterTargetAggregates(all_aggregates, fk_linked_tables_vec);

	if (target_aggregates.empty()) {
		throw InternalException("PAC Compiler: no aggregate nodes with FK-linked tables found in plan");
	}

#ifdef DEBUG
	Printer::Print("ModifyPlanWithoutPU: Found " + std::to_string(target_aggregates.size()) +
	               " aggregates with FK-linked tables");
#endif

	// For each target aggregate, find which orders table it has access to and build hash expression
	for (auto *target_agg : target_aggregates) {
		// Find which connecting table (lineitem) this aggregate has in its DIRECT path
		// (not in a nested subquery), and use the corresponding orders table
		idx_t orders_table_idx = DConstants::INVALID_INDEX;

		// We need to find the "closest" connecting table to this aggregate
		// For nested queries, the outer aggregate might contain both inner and outer tables
		// So we need to find which connecting table is in the aggregate's direct path
		// Strategy: check which connecting table indices exist, and use the one that's NOT in a DELIM_GET

		// First, collect all connecting table indices that appear in the aggregate's subtree
		vector<idx_t> candidate_conn_tables;
		for (auto &kv : connecting_table_to_orders_table) {
			idx_t conn_table_idx = kv.first;
			if (HasTableIndexInSubtree(target_agg, conn_table_idx)) {
				candidate_conn_tables.push_back(conn_table_idx);
			}
		}

#ifdef DEBUG
		Printer::Print("ModifyPlanWithoutPU: Aggregate has " + std::to_string(candidate_conn_tables.size()) +
		               " candidate connecting tables");
#endif

		// If no candidate connecting tables found, the scanned table itself must have an FK to PU
		// This happens when we query a leaf table that's far from the PU via FK chain
		// OR when we have a correlated subquery that accesses the outer table via DELIM_GET
		bool handled_via_direct_fk = false;
		if (candidate_conn_tables.empty()) {
			// First, check if this aggregate can access an already-present FK table via DELIM_GET
			// This happens in correlated subqueries where the outer query has the FK table
			// and the inner subquery accesses it via DELIM_GET
			// Search BOTH gets_present AND actually_present tables
			vector<string> all_searchable_tables;
			all_searchable_tables.insert(all_searchable_tables.end(), gets_present.begin(), gets_present.end());
			all_searchable_tables.insert(all_searchable_tables.end(), actually_present.begin(), actually_present.end());

			for (auto &present_table : all_searchable_tables) {
				// Check if this table has FK to PU
				auto it = check.table_metadata.find(present_table);
				if (it != check.table_metadata.end()) {
					bool has_fk_to_pu = false;
					for (auto &fk : it->second.fks) {
						for (auto &pu : privacy_units) {
							if (fk.first == pu) {
								has_fk_to_pu = true;
								break;
							}
						}
						if (has_fk_to_pu)
							break;
					}

					if (has_fk_to_pu) {
						// This present table has FK to PU - check if aggregate can access it
						// Find all instances of this table and check which one is accessible
						vector<unique_ptr<LogicalOperator> *> fk_table_nodes;
						FindAllNodesByTable(&plan, present_table, fk_table_nodes);

						for (auto *node : fk_table_nodes) {
							auto &node_get = node->get()->Cast<LogicalGet>();
							idx_t node_table_idx = node_get.table_index;

							// Check if this table index is in the aggregate's subtree
							if (HasTableIndexInSubtree(target_agg, node_table_idx)) {
								// Found an accessible FK table - use it for hashing
								vector<string> fk_cols;
								for (auto &fk : it->second.fks) {
									for (auto &pu : privacy_units) {
										if (fk.first == pu) {
											fk_cols = fk.second;
											break;
										}
									}
									if (!fk_cols.empty())
										break;
								}

								if (!fk_cols.empty()) {
									// Ensure FK columns are projected
									for (auto &fk_col : fk_cols) {
										idx_t proj_idx = EnsureProjectedColumn(node_get, fk_col);
										if (proj_idx == DConstants::INVALID_INDEX) {
											throw InternalException("PAC compiler: failed to project FK column " +
											                        fk_col);
										}
									}

									// Build hash expression
									auto base_hash_expr = BuildXorHashFromPKs(input, node_get, fk_cols);
									auto hash_input_expr = PropagatePKThroughProjections(
									    *plan, node_get, std::move(base_hash_expr), target_agg);

									// Modify this aggregate with PAC functions
									ModifyAggregatesWithPacFunctions(input, target_agg, hash_input_expr);

									handled_via_direct_fk = true;
									break;
								}
							}
						}

						if (handled_via_direct_fk) {
							break;
						}
					}
				}
			}

			// If not handled via direct FK table access, check for deep FK chains
			if (!handled_via_direct_fk) {
				// Find the scanned table in this aggregate's subtree that has FK to any table in the FK path
				// This handles deep FK chains where the leaf table doesn't directly reference the PU
				for (auto &present_table : gets_present) {
					if (HasTableInSubtree(target_agg, present_table)) {
						auto it = check.table_metadata.find(present_table);
						if (it != check.table_metadata.end()) {
							// Look for FK to any table in the FK path, not just PUs
							bool has_fk_in_path = false;
							vector<string> fk_cols;
							string fk_target;

							for (auto &fk : it->second.fks) {
								// Check if this FK references any table in the FK path
								for (auto &path_table : fk_path) {
									if (fk.first == path_table) {
										has_fk_in_path = true;
										fk_cols = fk.second;
										fk_target = fk.first;
										break;
									}
								}
								if (has_fk_in_path) {
									break;
								}
							}

							if (has_fk_in_path) {
								// This table is in the FK chain - we need to find the hash source table
								// that has FK to PU and is accessible within this aggregate's subtree

								// Find the last table in the FK path that has an FK to a PU
								string hash_source_table;
								vector<string> hash_source_fk_cols;

								for (size_t i = fk_path.size(); i > 0; i--) {
									auto &path_table = fk_path[i - 1];
									auto path_it = check.table_metadata.find(path_table);
									if (path_it != check.table_metadata.end()) {
										bool found_pu_fk = false;
										for (auto &path_fk : path_it->second.fks) {
											for (auto &pu : privacy_units) {
												if (path_fk.first == pu) {
													hash_source_table = path_table;
													hash_source_fk_cols = path_fk.second;
													found_pu_fk = true;
													break;
												}
											}
											if (found_pu_fk) {
												break;
											}
										}
										if (found_pu_fk) {
											break;
										}
									}
								}

								if (!hash_source_table.empty()) {
									// Find this table in the plan - search ALL instances
									vector<unique_ptr<LogicalOperator> *> hash_source_nodes;
									FindAllNodesByTable(&plan, hash_source_table, hash_source_nodes);

									// Find which instance is accessible from this aggregate's subtree
									// Check by table index to see which one the aggregate can access
									unique_ptr<LogicalOperator> *accessible_node = nullptr;
									for (auto *node : hash_source_nodes) {
										auto &node_get = node->get()->Cast<LogicalGet>();
										idx_t node_table_idx = node_get.table_index;

										// Check if this table index is in the aggregate's subtree
										if (HasTableIndexInSubtree(target_agg, node_table_idx)) {
											accessible_node = node;
											break;
										}
									}

									if (accessible_node) {
										auto &hash_source_get = accessible_node->get()->Cast<LogicalGet>();

										// Ensure FK columns are projected
										for (auto &hash_fk_col : hash_source_fk_cols) {
											idx_t proj_idx = EnsureProjectedColumn(hash_source_get, hash_fk_col);
											if (proj_idx == DConstants::INVALID_INDEX) {
												throw InternalException("PAC compiler: failed to project FK column " +
												                        hash_fk_col);
											}
										}

										// Build hash expression
										auto base_hash_expr =
										    BuildXorHashFromPKs(input, hash_source_get, hash_source_fk_cols);
										auto hash_input_expr = PropagatePKThroughProjections(
										    *plan, hash_source_get, std::move(base_hash_expr), target_agg);

										// Modify this aggregate with PAC functions
										ModifyAggregatesWithPacFunctions(input, target_agg, hash_input_expr);

										// Mark as handled and break
										handled_via_direct_fk = true;
										break;
									} else if (!hash_source_nodes.empty()) {
										// The hash source table exists but is not in the aggregate's subtree
										// This happens in correlated subqueries where the FK table is in the outer
										// query We need to add the FK column(s) to the DELIM_JOIN correlation
										auto &hash_source_get = hash_source_nodes[0]->get()->Cast<LogicalGet>();

										// For each FK column, add it to the DELIM_JOIN and build hash from DELIM_GET
										// binding
										vector<DelimColumnResult> delim_results;
										for (auto &hash_fk_col : hash_source_fk_cols) {
											auto delim_result =
											    AddColumnToDelimJoin(plan, hash_source_get, hash_fk_col, target_agg);
											if (!delim_result.IsValid()) {
												// Failed to add to DELIM_JOIN - skip this approach
												break;
											}
											delim_results.push_back(delim_result);
										}

										if (delim_results.size() == hash_source_fk_cols.size()) {
											// Successfully added all FK columns to DELIM_JOIN
											// Build hash expression using the DELIM_GET bindings with correct types
											unique_ptr<Expression> hash_input_expr;

											if (delim_results.size() == 1) {
												// Single FK column - just hash it
												auto col_ref = make_uniq<BoundColumnRefExpression>(
												    delim_results[0].type, delim_results[0].binding);
												hash_input_expr =
												    input.optimizer.BindScalarFunction("hash", std::move(col_ref));
											} else {
												// Multiple FK columns - XOR them together then hash
												auto first_col = make_uniq<BoundColumnRefExpression>(
												    delim_results[0].type, delim_results[0].binding);
												unique_ptr<Expression> xor_expr = std::move(first_col);

												for (size_t i = 1; i < delim_results.size(); i++) {
													auto next_col = make_uniq<BoundColumnRefExpression>(
													    delim_results[i].type, delim_results[i].binding);
													xor_expr = input.optimizer.BindScalarFunction(
													    "xor", std::move(xor_expr), std::move(next_col));
												}
												hash_input_expr =
												    input.optimizer.BindScalarFunction("hash", std::move(xor_expr));
											}

											// Modify this aggregate with PAC functions
											ModifyAggregatesWithPacFunctions(input, target_agg, hash_input_expr);

											// Mark as handled and break
											handled_via_direct_fk = true;
											break;
										}
									}
								}
							}
						}
					}
				}
			}

			if (!handled_via_direct_fk) {
				throw InternalException("PAC Compiler: could not find any connecting table for aggregate");
			}
		}

		// Skip to next aggregate if we already handled this one via direct FK
		if (handled_via_direct_fk) {
			continue;
		}

		// If there's only one candidate, use it
		if (candidate_conn_tables.size() == 1) {
			orders_table_idx = connecting_table_to_orders_table[candidate_conn_tables[0]];
		} else {
			// Multiple candidates - we need to find which one is in the aggregate's direct path
			// The heuristic: try each candidate's orders table and see which one can be propagated
			// The correct one will be accessible without going through DELIM_GET
			for (auto conn_table_idx : candidate_conn_tables) {
				idx_t ord_table_idx = connecting_table_to_orders_table[conn_table_idx];

				// Try to find the orders table - if it's directly accessible (not through DELIM_GET),
				// we can use it
				// Simple heuristic: the orders table with the SMALLER index is likely the outer one
				// For correlated subqueries, outer tables have smaller indices than inner tables
				if (orders_table_idx == DConstants::INVALID_INDEX || ord_table_idx < orders_table_idx) {
					orders_table_idx = ord_table_idx;
				}
			}
		}

		if (orders_table_idx == DConstants::INVALID_INDEX) {
			throw InternalException("PAC Compiler: could not find orders table for aggregate");
		}

#ifdef DEBUG
		Printer::Print("ModifyPlanWithoutPU: Selected orders table #" + std::to_string(orders_table_idx) +
		               " for aggregate");
#endif

		// Find the orders table LogicalGet with this index
		vector<unique_ptr<LogicalOperator> *> orders_nodes;
		FindAllNodesByTableIndex(&plan, orders_table_idx, orders_nodes);

		if (orders_nodes.empty()) {
			throw InternalException("PAC Compiler: could not find orders LogicalGet with index " +
			                        std::to_string(orders_table_idx));
		}

		auto &orders_get = orders_nodes[0]->get()->Cast<LogicalGet>();

		// Build hash expression from the FK-linked table's FK to the PU
		// Find which table this is and get its FK columns
		vector<string> fk_cols;
		string fk_table_name;

		// Get the table name from the LogicalGet
		auto orders_table_ptr = orders_get.GetTable();
		if (orders_table_ptr) {
			fk_table_name = orders_table_ptr->name;
		}

		if (!fk_table_name.empty()) {
			auto it = check.table_metadata.find(fk_table_name);
			if (it != check.table_metadata.end()) {
				for (auto &fk : it->second.fks) {
					// Find FK to any of the privacy units
					for (auto &pu : privacy_units) {
						if (fk.first == pu) {
							fk_cols = fk.second;
							break;
						}
					}
					if (!fk_cols.empty()) {
						break;
					}
				}
			}
		}

		if (fk_cols.empty()) {
			throw InternalException("PAC Compiler: no FK found from " + fk_table_name + " to any PU");
		}

		// Ensure FK columns are projected
		for (auto &fk_col : fk_cols) {
			idx_t proj_idx = EnsureProjectedColumn(orders_get, fk_col);
			if (proj_idx == DConstants::INVALID_INDEX) {
				throw InternalException("PAC compiler: failed to project FK column " + fk_col);
			}
		}

		// Build hash expression
		auto base_hash_expr = BuildXorHashFromPKs(input, orders_get, fk_cols);
		auto hash_input_expr = PropagatePKThroughProjections(*plan, orders_get, std::move(base_hash_expr), target_agg);

#ifdef DEBUG
		Printer::Print("ModifyPlanWithoutPU: Built hash expression for aggregate using orders table #" +
		               std::to_string(orders_table_idx));
#endif

		// Modify this aggregate with PAC functions
		ModifyAggregatesWithPacFunctions(input, target_agg, hash_input_expr);
	}
}

/**
 * ModifyPlanWithPU: Transforms a query plan when the privacy unit (PU) table IS scanned directly
 *
 * Purpose: When the query directly scans the PU table, we build hash expressions from the PU's
 * primary key columns and transform aggregates to use PAC functions.
 *
 * Arguments:
 * @param input - Optimizer extension input containing context and optimizer
 * @param plan - The logical plan to modify
 * @param pu_table_names - List of privacy unit table names that are scanned in the query
 * @param check - Compatibility check result containing table metadata
 *
 * Logic:
 * 1. Find ALL aggregate nodes in the plan
 * 2. Filter to aggregates that have at least one PU table in their subtree
 * 3. For each target aggregate:
 *    - For each PU table in the aggregate's subtree:
 *      * Determine whether to use rowid or primary key columns for hashing
 *      * Build hash expression: hash(pk) or hash(xor(pk1, pk2, ...)) for composite PKs
 *      * Propagate the hash expression through projections to the aggregate level
 *    - Combine all PU hash expressions with AND (for multi-PU queries)
 *    - Transform the aggregate to use PAC functions
 *
 * Correlated Subquery Handling:
 * - If the PU table appears in BOTH outer query and subquery, we transform aggregates in BOTH
 * - Each aggregate gets its own hash expression from the PU instance in its subtree
 * - Example: Query with outer aggregate on customer and subquery aggregate on customer:
 *   * Both aggregates are transformed with pac_sum(hash(c_custkey), value)
 *   * Each uses its respective customer table instance
 *
 * Nested Aggregate Rules (IMPORTANT):
 * - If we have 2 aggregates stacked on top of each other:
 *   * ONLY transform the inner aggregate if it directly operates on PU tables
 *   * The outer aggregate is NOT transformed if it only depends on the inner result
 *   * EXCEPTION: Transform the outer aggregate ONLY if it also has PU tables in its subtree
 *     (meaning it has its own FK path that needs joins and PAC transformation)
 *
 * Example from user's TPC-H Q17-style query:
 *   SELECT sum(l_extendedprice) / 7.0 AS avg_yearly
 *   FROM lineitem, part
 *   WHERE p_partkey = l_partkey AND ... AND l_quantity < (SELECT 0.2 * avg(l_quantity) FROM lineitem WHERE ...)
 *
 * Transformation:
 *   - Inner aggregate (avg in subquery): Has lineitem (PU table) -> pac_avg(hash(rowid), l_quantity)
 *   - Outer aggregate (sum): Also has lineitem (PU table) -> pac_sum(hash(rowid), l_extendedprice)
 *   - Division by 7.0 happens AFTER PAC aggregation, not transformed
 *
 * Counter-example where outer is NOT transformed:
 *   SELECT sum(inner_sum) FROM (SELECT sum(customer_col) FROM customer) AS subq
 *   - Inner: Has customer (PU) -> pac_sum(hash(c_custkey), customer_col)
 *   - Outer: No PU tables, only depends on subquery result -> Regular sum(), NOT transformed
 */
void ModifyPlanWithPU(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                      const vector<string> &pu_table_names, const PACCompatibilityResult &check) {

	// Find ALL aggregate nodes in the plan first
	vector<LogicalAggregate *> all_aggregates;
	FindAllAggregates(plan, all_aggregates);

	if (all_aggregates.empty()) {
		throw InternalException("PAC Compiler: no aggregate nodes found in plan");
	}

	// Build a list of all tables that should trigger PAC transformation:
	// 1. Privacy unit tables themselves
	// 2. Tables that are FK-linked to privacy unit tables
	vector<string> relevant_tables;
	std::unordered_set<string> relevant_tables_set;

	// Add PU tables
	for (auto &pu : pu_table_names) {
		relevant_tables.push_back(pu);
		relevant_tables_set.insert(pu);
	}

	// Add FK-linked tables from the compatibility check results
	for (auto &kv : check.fk_paths) {
		auto &path = kv.second;
		for (auto &table : path) {
			if (relevant_tables_set.find(table) == relevant_tables_set.end()) {
				relevant_tables.push_back(table);
				relevant_tables_set.insert(table);
			}
		}
	}

#ifdef DEBUG
	Printer::Print("ModifyPlanWithPU: relevant tables for PAC transformation:");
	for (auto &t : relevant_tables) {
		Printer::Print("  " + t);
	}
#endif

	// Filter aggregates to those that have at least one relevant table in their subtree
	vector<LogicalAggregate *> target_aggregates = FilterTargetAggregates(all_aggregates, relevant_tables);

	if (target_aggregates.empty()) {
		throw InternalException("PAC Compiler: no aggregate nodes with privacy unit tables found in plan");
	}

#ifdef DEBUG
	Printer::Print("ModifyPlanWithPU: Found " + std::to_string(target_aggregates.size()) +
	               " aggregates with privacy unit tables");
#endif

	// For each target aggregate, build hash expressions and modify it
	for (auto *target_agg : target_aggregates) {
		// Build hash expressions for each privacy unit
		vector<unique_ptr<Expression>> hash_exprs;

		for (auto &pu_table_name : pu_table_names) {
			// Check if this aggregate has the PU table in its subtree
			if (HasTableInSubtree(target_agg, pu_table_name)) {
				// Direct PU scan case: use PU's primary key
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

				// Propagate the hash expression through all projections between table scan and this aggregate
				hash_input_expr = PropagatePKThroughProjections(*plan, get, std::move(hash_input_expr), target_agg);

				hash_exprs.push_back(std::move(hash_input_expr));
			} else {
				// FK-linked table case: find FK columns that reference the PU
				// Check which FK-linked tables are in this aggregate's subtree
				for (auto &kv : check.fk_paths) {
					auto &fk_table = kv.first;
					auto &path = kv.second;

					// Skip if this FK path doesn't lead to the current PU
					if (path.empty() || path.back() != pu_table_name) {
						continue;
					}

					// Check if the FK table is in this aggregate's subtree
					if (!HasTableInSubtree(target_agg, fk_table)) {
						continue;
					}

					// Find the FK columns from fk_table that reference the next table in the path
					// The path is ordered: [fk_table, intermediate_table(s), pu_table]
					// We need the FK columns from fk_table that reference the next table in the path

					// Get metadata for the FK table
					auto fk_it = check.table_metadata.find(fk_table);
					if (fk_it == check.table_metadata.end()) {
						continue;
					}

					// Find the FK that references the PU (possibly through intermediate tables)
					// For now, let's find the FK that ultimately leads to the PU
					string next_table_in_path = path.size() > 1 ? path[1] : pu_table_name;

					vector<string> fk_cols;
					for (auto &fk : fk_it->second.fks) {
						if (fk.first == next_table_in_path) {
							fk_cols = fk.second;
							break;
						}
					}

					if (fk_cols.empty()) {
						continue;
					}

					// Find the LogicalGet for the FK table in this aggregate's subtree
					// Search for all FK table nodes and find which one is accessible from this aggregate
					vector<unique_ptr<LogicalOperator> *> fk_nodes;
					FindAllNodesByTable(&plan, fk_table, fk_nodes);

					// For each FK table instance, check if it's in this aggregate's subtree
					// by checking if the aggregate has access to that table index
					unique_ptr<LogicalOperator> *fk_scan_ptr = nullptr;
					for (auto *node : fk_nodes) {
						auto &node_get = node->get()->Cast<LogicalGet>();
						idx_t node_table_idx = node_get.table_index;

						// Check if this table index is in the aggregate's subtree
						if (HasTableIndexInSubtree(target_agg, node_table_idx)) {
							fk_scan_ptr = node;
							break;
						}
					}

					if (!fk_scan_ptr) {
						continue;
					}
					auto &fk_get = fk_scan_ptr->get()->Cast<LogicalGet>();

					// Ensure FK columns are present
					AddPKColumns(fk_get, fk_cols);

					// Build hash expression from FK columns
					auto fk_hash_expr = BuildXorHashFromPKs(input, fk_get, fk_cols);

					// Propagate through projections
					fk_hash_expr = PropagatePKThroughProjections(*plan, fk_get, std::move(fk_hash_expr), target_agg);

					hash_exprs.push_back(std::move(fk_hash_expr));
					break; // Only process one FK path per PU per aggregate
				}
			}
		}

		// Skip if no hash expressions were built for this aggregate
		if (hash_exprs.empty()) {
			continue;
		}

		// Combine all hash expressions with AND
		auto combined_hash_expr = BuildAndFromHashes(input, hash_exprs);

		// Modify this aggregate with PAC functions
		ModifyAggregatesWithPacFunctions(input, target_agg, combined_hash_expr);
	}
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
