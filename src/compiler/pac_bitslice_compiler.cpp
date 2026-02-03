//
// PAC Bitslice Compiler
//
// This file implements the bitslice compilation strategy for PAC (Privacy-Augmented Computation).
// The bitslice compiler transforms query plans to add necessary joins and hash expressions for
// computing PAC aggregates over privacy units.
//
// Key Concepts:
// - Privacy Unit (PU): The entity that defines privacy boundaries (e.g., customer)
// - FK Path: The chain of foreign keys from queried tables to the PU (e.g., lineitem -> orders -> customer)
// - Hash Expression: A hash computed from FK columns that reference the PU (e.g., hash(o_custkey))
// - Correlated Subqueries: Inner queries that reference outer query tables (require special handling)
//
// Created by ila on 12/21/25.
//

#include "compiler/pac_bitslice_compiler.hpp"
#include "utils/pac_helpers.hpp"
#include "metadata/pac_compatibility_check.hpp"
#include "compiler/pac_compiler_helpers.hpp"
#include "query_processing/pac_projection_propagation.hpp"
#include "query_processing/pac_subquery_handler.hpp"
#include "categorical/pac_categorical_rewriter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/optimizer/optimizer.hpp"

namespace duckdb {

/**
 * ModifyPlanWithoutPU: Transforms a query plan when the privacy unit (PU) table is NOT scanned directly
 *
 * PURPOSE:
 * When a query doesn't directly scan the PU table (e.g., SELECT FROM lineitem), we need to:
 * 1. Add joins to connect the scanned tables to the PU via foreign key relationships
 * 2. Build hash expressions from the FK columns that reference the PU
 * 3. Propagate these hash expressions to aggregates
 * 4. Transform aggregates to use PAC functions (pac_sum, pac_avg, etc.)
 *
 * EXAMPLE:
 * Query: SELECT SUM(l_quantity) FROM lineitem WHERE l_partkey = 100
 * FK Path: lineitem -> orders -> customer (customer is PU)
 * Missing: orders table (needs to be joined)
 * Result: lineitem JOIN orders -> compute hash(o_custkey) -> pac_sum(l_quantity, hash)
 *
 * ARGUMENTS:
 * @param check - Compatibility check result with:
 *   - table_metadata: Metadata for each table (PKs, FKs)
 *   - scanned_non_pu_tables: Tables actually scanned in the query
 *   - privacy_units: List of PU table names
 * @param input - Optimizer extension input (context, optimizer)
 * @param plan - The logical plan to modify (modified in-place)
 * @param gets_missing - Tables in FK path NOT in original query (need to be added as joins)
 * @param gets_present - Tables in FK path ALREADY in original query
 * @param fk_path - Ordered list from scanned table to PU (e.g., [lineitem, orders, customer])
 * @param privacy_units - List of PU table names (e.g., ["customer"])
 *
 * CORRELATED SUBQUERY HANDLING:
 * When a table appears in BOTH outer query AND inner subquery:
 * - We find ALL instances of the connecting table in the plan
 * - Add join chains to EACH instance independently
 * - Map each aggregate to its "closest" FK table (not crossing subquery boundaries)
 *
 * Example (TPC-H Q17):
 *   SELECT SUM(l_extendedprice) FROM lineitem l1
 *   WHERE l1.l_quantity < (
 *     SELECT AVG(l2.l_quantity) FROM lineitem l2
 *     WHERE l2.l_partkey = l1.l_partkey
 *   )
 * - Outer lineitem -> needs join to orders for outer aggregate
 * - Inner lineitem -> needs separate join to orders for inner aggregate
 * - Each aggregate gets hash from its own orders table instance
 *
 * ALGORITHM:
 * 1. Determine which tables need to be joined (filter gets_missing by FK path)
 * 2. If join_elimination enabled, skip joining PU tables themselves
 * 3. Find "connecting table" - last present table in FK path order (e.g., lineitem)
 * 4. For each instance of connecting table:
 *    a. Create fresh LogicalGet nodes for missing tables
 *    b. Build join chain: connecting_table -> table1 -> table2 -> ...
 *    c. Track table index of FK table (that references PU) for hash generation
 *    d. Replace connecting table with join chain
 * 5. Find all aggregates with FK-linked tables in their subtree
 * 6. For each aggregate:
 *    a. Determine which FK table instance it should use (closest accessible one)
 *    b. Build hash expression from FK columns referencing PU
 *    c. Propagate hash through projections
 *    d. Transform aggregate to PAC aggregate
 *
 * JOIN ELIMINATION OPTIMIZATION:
 * When enabled (pac_join_elimination=true), we skip joining the PU table itself
 * if it's only needed for the foreign key columns (not for PU data).
 * This reduces join overhead when the PU table is large.
 *
 * ACCESSIBILITY CHECKS:
 * We ensure FK table columns are ACCESSIBLE from each aggregate:
 * - Not blocked by MARK/SEMI/ANTI joins (which don't pass right-side columns)
 * - Not in a separate subquery branch (would need DELIM_GET)
 * If FK table is inaccessible, we add a fresh join to bring it into scope.
 */

// Helper function to compute required columns for a table in the FK path
// Returns the columns needed for joining and for the PU FK reference (for hashing)
static vector<string> GetRequiredColumnsForTable(const PACCompatibilityResult &check, const string &table_name,
                                                 const vector<string> &fk_path, const vector<string> &privacy_units) {
	vector<string> required_columns;
	std::unordered_set<string> added_columns;

	auto it = check.table_metadata.find(table_name);
	if (it == check.table_metadata.end()) {
#ifdef DEBUG
		Printer::Print("GetRequiredColumnsForTable WARNING: No metadata for table " + table_name);
#endif
		return {}; // Empty = project all columns (fallback)
	}

	const auto &meta = it->second;

	// 1. Add PK columns (needed for joins FROM other tables TO this table)
	for (auto &pk : meta.pks) {
		if (added_columns.insert(StringUtil::Lower(pk)).second) {
			required_columns.push_back(pk);
		}
	}

	// 2. Add FK columns (needed for joins FROM this table TO other tables, and for PU hash)
	for (auto &fk_pair : meta.fks) {
		const string &referenced_table = fk_pair.first;
		const vector<string> &fk_cols = fk_pair.second;

		// Check if this FK references a table in the FK path or a privacy unit
		bool is_relevant = false;
		for (auto &path_table : fk_path) {
			if (StringUtil::Lower(path_table) == StringUtil::Lower(referenced_table)) {
				is_relevant = true;
				break;
			}
		}
		if (!is_relevant) {
			for (auto &pu : privacy_units) {
				if (StringUtil::Lower(pu) == StringUtil::Lower(referenced_table)) {
					is_relevant = true;
					break;
				}
			}
		}

		if (is_relevant) {
			for (auto &fk_col : fk_cols) {
				if (added_columns.insert(StringUtil::Lower(fk_col)).second) {
					required_columns.push_back(fk_col);
				}
			}
		}
	}

#ifdef DEBUG
	if (!required_columns.empty()) {
		string cols_str;
		for (auto &col : required_columns) {
			if (!cols_str.empty())
				cols_str += ", ";
			cols_str += col;
		}
		Printer::Print("GetRequiredColumnsForTable: Table " + table_name + " requires columns: [" + cols_str + "]");
	}
#endif

	return required_columns;
}

void ModifyPlanWithoutPU(const PACCompatibilityResult &check, OptimizerExtensionInput &input,
                         unique_ptr<LogicalOperator> &plan, const vector<string> &gets_missing,
                         const vector<string> &gets_present, const vector<string> &fk_path,
                         const vector<string> &privacy_units) {
	// Note: we assume we don't use rowid

	// Check if join elimination is enabled
	bool join_elimination = GetBooleanSetting(input.context, "pac_join_elimination", false);

	// Create the necessary LogicalGets for missing tables
	// IMPORTANT: We need to preserve the FK path ordering when creating joins
	// Use fk_path as the canonical ordering, filter to only missing tables
	std::unordered_set<string> missing_set(gets_missing.begin(), gets_missing.end());

	// If join elimination is enabled, skip the PU tables themselves
	// We only need the FK-linked table (e.g., orders), not the PU (e.g., customer)
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
			// Get required columns for this table (PKs and relevant FKs)
			auto required_cols = GetRequiredColumnsForTable(check, table, fk_path, privacy_units);
			auto get = CreateLogicalGet(input.context, plan, table, idx, required_cols);
			get_map[table] = std::move(get);
			ordered_table_names.push_back(table);
			idx++;
		} else if (actually_present.find(table) != actually_present.end()) {
			// Track the order of already-present tables in the FK path
			actually_present_in_fk_order.push_back(table);
		}
	}

	// Find the unique_ptr reference to the existing table that connects to the missing tables
	// There are TWO different scenarios:
	// 1. When there ARE missing tables to join: use the LAST present table in FK path order
	//    (it's adjacent to the first missing table)
	// 2. When there are NO missing tables: use the FIRST present table in FK path order
	//    (it's the leaf table that may appear in subqueries and need joins to the FK table)

	// Determine which present table should be the "connecting table" for adding joins
	string connecting_table_for_joins;   // First present - for finding nodes that need joins added
	string connecting_table_for_missing; // Last present - for connecting to missing tables

	if (!fk_path.empty()) {
		for (auto &table_in_path : fk_path) {
			bool is_present = false;
			for (auto &present : gets_present) {
				if (table_in_path == present) {
					is_present = true;
					break;
				}
			}
			if (is_present) {
				if (connecting_table_for_joins.empty()) {
					connecting_table_for_joins = table_in_path; // First present table
				}
				connecting_table_for_missing = table_in_path; // Keep updating to get last present
			}
		}
	}

	// Fallback: if no connecting table found in FK path, use any present table
	if (connecting_table_for_joins.empty() && !gets_present.empty()) {
		connecting_table_for_joins = gets_present[0];
	}
	if (connecting_table_for_missing.empty() && !gets_present.empty()) {
		connecting_table_for_missing = gets_present[0];
	}
	if (connecting_table_for_joins.empty() && !check.scanned_non_pu_tables.empty()) {
		connecting_table_for_joins = check.scanned_non_pu_tables[0];
	}
	if (connecting_table_for_missing.empty() && !check.scanned_non_pu_tables.empty()) {
		connecting_table_for_missing = check.scanned_non_pu_tables[0];
	}

	// Decide which connecting table to use based on whether there are missing tables
	// If there are missing tables, we iterate over the LAST present table to add joins to missing tables
	// If there are NO missing tables, we iterate over the FIRST present table to add FK joins for subqueries
	string connecting_table = ordered_table_names.empty() ? connecting_table_for_joins : connecting_table_for_missing;

	// Find ALL instances of the connecting table (for correlated subqueries, there may be multiple)
	// Example: In TPC-H Q17, lineitem appears in both outer query and subquery
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
	// Store the mapping from each instance to its corresponding FK table (e.g., orders) for hash generation
	// This is critical for correlated subqueries: each instance gets its own FK table join
	std::unordered_map<idx_t, idx_t> connecting_table_to_orders_table;

	// First, find the FK table that has FK to PU (e.g., orders -> customer)
	// This is the table whose FK columns we'll hash
	string fk_table_with_pu_reference;
	for (auto &table : fk_path) {
		auto it = check.table_metadata.find(table);
		if (it != check.table_metadata.end()) {
			for (auto &fk : it->second.fks) {
				for (auto &pu : privacy_units) {
					if (fk.first == pu) {
						fk_table_with_pu_reference = table;
						break;
					}
				}
				if (fk_table_with_pu_reference.empty()) {
					break;
				}
			}
		}
	}

	// Iterate over each instance of the connecting table and add joins
	for (auto *target_ref : all_connecting_nodes) {
		// Get the table index of this instance
		auto &target_op = (*target_ref)->Cast<LogicalGet>();
		idx_t connecting_table_idx = target_op.table_index;

		// Check if this connecting table instance is inside a DELIM_JOIN subquery branch
		// If so, we MAY need to add a join to the FK table - BUT only if the FK table
		// is not already directly accessible in the outer query WITHOUT going through
		// a join with the connecting table
		bool is_in_subquery = IsInDelimJoinSubqueryBranch(&plan, target_ref->get());

		// Determine which tables need to be joined for THIS instance
		vector<string> tables_to_join_for_instance = ordered_table_names;

		// SUBQUERY SPECIAL CASE:
		// For subquery instances, check if we need to add a join to the FK table
		// even though it's "present" in outer query
		if (is_in_subquery && !fk_table_with_pu_reference.empty() &&
		    std::find(ordered_table_names.begin(), ordered_table_names.end(), fk_table_with_pu_reference) ==
		        ordered_table_names.end()) {
			// The FK table is not in the join list (because it's "present" in outer query)
			//
			// We should ONLY skip adding the join if:
			// 1. The FK table is in the outer query AND
			// 2. The connecting table is NOT in the outer query (meaning outer query scans FK table directly)
			//
			// If BOTH the FK table AND connecting table are in the outer query, the subquery
			// still needs its own FK join because:
			// - The outer query has: connecting_table JOIN fk_table
			// - The subquery has: connecting_table (correlated with outer connecting_table)
			// - The subquery needs its own fk_table join to get the FK columns for each row

			// IMPORTANT: If the connecting table IS the FK table, we don't need to add a join
			// (we can't join a table to itself via FK relationship)
			if (connecting_table == fk_table_with_pu_reference) {
				// The connecting table already IS the FK table - no join needed
				// Just map it to itself for hash generation
				connecting_table_to_orders_table[connecting_table_idx] = connecting_table_idx;
#ifdef DEBUG
				Printer::Print("ModifyPlanWithoutPU: Connecting table #" + std::to_string(connecting_table_idx) +
				               " is already the FK table (" + fk_table_with_pu_reference + "), no join needed");
#endif
			} else {
				vector<unique_ptr<LogicalOperator> *> fk_nodes;
				FindAllNodesByTable(&plan, fk_table_with_pu_reference, fk_nodes);

				bool fk_in_outer_query = false;
				for (auto *fk_node : fk_nodes) {
					if (!IsInDelimJoinSubqueryBranch(&plan, fk_node->get())) {
						fk_in_outer_query = true;
						break;
					}
				}

				// Check if the connecting table is ALSO in the outer query
				bool connecting_table_in_outer = false;
				for (auto *conn_node : all_connecting_nodes) {
					if (!IsInDelimJoinSubqueryBranch(&plan, conn_node->get())) {
						connecting_table_in_outer = true;
						break;
					}
				}

				// Only skip adding FK join if:
				// - FK table is in outer query AND
				// - Connecting table is NOT in outer query (outer scans FK table directly)
				// This means the subquery can access FK columns via DELIM_GET
				bool can_use_delim_get = fk_in_outer_query && !connecting_table_in_outer;

				if (!can_use_delim_get) {
					// Need to add FK join for this subquery instance
					tables_to_join_for_instance.push_back(fk_table_with_pu_reference);
#ifdef DEBUG
					Printer::Print("ModifyPlanWithoutPU: Adding " + fk_table_with_pu_reference +
					               " join for subquery instance #" + std::to_string(connecting_table_idx));
#endif
				}
#ifdef DEBUG
				else {
					Printer::Print("ModifyPlanWithoutPU: FK table " + fk_table_with_pu_reference +
					               " accessible via DELIM_GET for subquery instance #" +
					               std::to_string(connecting_table_idx));
				}
#endif
			}
		}

		// Only create joins if there are tables to join for this instance
		if (!tables_to_join_for_instance.empty()) {
			unique_ptr<LogicalOperator> existing_node = (*target_ref)->Copy(input.context);

			// Create fresh LogicalGet nodes for this instance
			auto local_idx = GetNextTableIndex(plan);
			std::unordered_map<string, unique_ptr<LogicalGet>> local_get_map;

			for (auto &table : tables_to_join_for_instance) {
				auto it = check.table_metadata.find(table);
				if (it == check.table_metadata.end()) {
					throw InternalException("PAC compiler: missing table metadata for table: " + table);
				}
				vector<string> pks = it->second.pks;
				// Get required columns for this table (PKs and relevant FKs)
				auto required_cols = GetRequiredColumnsForTable(check, table, fk_path, privacy_units);
				auto get = CreateLogicalGet(input.context, plan, table, local_idx, required_cols);

				// Remember the table index of the FK table for this instance
				if (table == fk_table_with_pu_reference) {
					connecting_table_to_orders_table[connecting_table_idx] = local_idx;
				}

				local_get_map[table] = std::move(get);
				local_idx++;
			}

			unique_ptr<LogicalOperator> final_join;
			for (size_t i = 0; i < tables_to_join_for_instance.size(); ++i) {
				auto &tbl_name = tables_to_join_for_instance[i];
				unique_ptr<LogicalGet> right_op = std::move(local_get_map[tbl_name]);
				if (!right_op) {
					throw InternalException("PAC compiler: failed to transfer ownership of LogicalGet for " + tbl_name);
				}

				if (i == 0) {
					final_join = CreateLogicalJoin(check, input.context, std::move(existing_node), std::move(right_op));
				} else {
					final_join = CreateLogicalJoin(check, input.context, std::move(final_join), std::move(right_op));
				}
			}

			// Replace this instance with the join chain
			ReplaceNode(plan, *target_ref, final_join);
		} else {
			// No tables to join - the FK-linked table (e.g., orders) must already be in the plan
			// and this instance can access it directly (not in a subquery branch)
			// Find it and map the connecting table to it
			// Search BOTH gets_present AND actually_present tables (tables marked missing but found in plan)
			// IMPORTANT: We must find an FK table that is in the SAME SUBTREE as the connecting table,
			// not just any accessible FK table. This is crucial for queries with multiple branches
			// (e.g., CROSS_PRODUCT with different aggregates in each branch).
			vector<string> all_present_tables;
			all_present_tables.insert(all_present_tables.end(), gets_present.begin(), gets_present.end());
			all_present_tables.insert(all_present_tables.end(), actually_present.begin(), actually_present.end());

			bool found_accessible_fk_table = false;

			// First, find the common ancestor of this connecting table in the plan
			// We need to find an FK table that shares a subtree with this connecting table
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
						if (has_fk_to_pu) {
							break;
						}
					}

					if (has_fk_to_pu) {
						// This is the table we need for hashing - find its table index
						vector<unique_ptr<LogicalOperator> *> fk_table_nodes;
						FindAllNodesByTable(&plan, present_table, fk_table_nodes);
						if (!fk_table_nodes.empty()) {
							// Find an FK table instance that is in the same subtree as the connecting table
							// This is done by checking if BOTH tables are reachable from some common ancestor
							for (auto *fk_node : fk_table_nodes) {
								auto &fk_table_get = fk_node->get()->Cast<LogicalGet>();
								idx_t fk_table_idx = fk_table_get.table_index;

								// IMPORTANT: If the FK table IS the connecting table (same index),
								// they trivially share a subtree - just map it to itself
								if (fk_table_idx == connecting_table_idx) {
									if (AreTableColumnsAccessible(plan.get(), fk_table_idx)) {
										connecting_table_to_orders_table[connecting_table_idx] = fk_table_idx;
#ifdef DEBUG
										Printer::Print("ModifyPlanWithoutPU: Connecting table #" +
										               std::to_string(connecting_table_idx) +
										               " IS the FK table - mapping to itself");
#endif
										found_accessible_fk_table = true;
										break;
									}
									continue;
								}

								// Check if this FK table is in the same subtree as the connecting table
								// by finding a common ancestor that has both table indices in its subtree
								// The simplest check: find an operator that has BOTH table indices in its subtree
								bool shares_subtree = false;

								// Walk up from plan root and find smallest subtree containing both tables
								std::function<bool(LogicalOperator *)> findCommonSubtree =
								    [&](LogicalOperator *op) -> bool {
									if (!op) {
										return false;
									}

									bool has_connecting = HasTableIndexInSubtree(op, connecting_table_idx);
									bool has_fk = HasTableIndexInSubtree(op, fk_table_idx);

									if (has_connecting && has_fk) {
										// Both tables are in this subtree - check if they're in the SAME child
										// (not in different branches of a join/cross product)
										for (auto &child : op->children) {
											bool child_has_connecting =
											    HasTableIndexInSubtree(child.get(), connecting_table_idx);
											bool child_has_fk = HasTableIndexInSubtree(child.get(), fk_table_idx);

											if (child_has_connecting && child_has_fk) {
												// Both in same child - recurse to find tighter common ancestor
												return findCommonSubtree(child.get());
											}
										}
										// Tables are in different children of this operator
										// Check if this is a JOIN that connects them (valid) vs CROSS_PRODUCT (separate
										// branches)
										if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
										    op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN ||
										    op->type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
											// Join connects them - they can reference each other
											shares_subtree = true;
											return true;
										}
										// CROSS_PRODUCT or other - tables are in independent branches
										return false;
									}
									return false;
								};

								findCommonSubtree(plan.get());

								if (shares_subtree) {
									// Also verify the FK table's columns are accessible (not blocked by MARK/SEMI/ANTI)
									if (AreTableColumnsAccessible(plan.get(), fk_table_idx)) {
										// Map connecting table to this FK table instance
										connecting_table_to_orders_table[connecting_table_idx] = fk_table_idx;
#ifdef DEBUG
										Printer::Print("ModifyPlanWithoutPU: Mapped connecting table #" +
										               std::to_string(connecting_table_idx) + " to FK table " +
										               present_table + " #" + std::to_string(fk_table_idx) +
										               " for hashing (same subtree)");
#endif
										found_accessible_fk_table = true;
										break;
									}
								}
#ifdef DEBUG
								else {
									Printer::Print("ModifyPlanWithoutPU: FK table " + present_table + " #" +
									               std::to_string(fk_table_idx) +
									               " is NOT in same subtree as connecting table #" +
									               std::to_string(connecting_table_idx));
								}
#endif
							}
							if (found_accessible_fk_table) {
								break;
							}
						}
					}
				}
			}

			// If no accessible FK table was found, we need to add a join to bring in the FK table
			// on the accessible (left) side of the query
			if (!found_accessible_fk_table && !fk_table_with_pu_reference.empty()) {
				// We need to join through ALL intermediate tables in the FK chain
				// to reach the FK table with PU reference.
				// Example: shipments -> order_items -> orders (FK to users)
				// We can't just join shipments directly to orders - we need order_items in between.

				// Find the connecting table's position in the FK path
				string connecting_table_name;
				auto table_ptr = target_op.GetTable();
				if (table_ptr) {
					connecting_table_name = table_ptr->name;
				}

				// IMPORTANT: If the connecting table IS the FK table with PU reference,
				// we need to check if its columns are actually accessible.
				// If blocked by SEMI/ANTI join, we need to find an accessible table and add a join from there.
				if (connecting_table_name == fk_table_with_pu_reference) {
					// Check if this table's columns are accessible
					if (AreTableColumnsAccessible(plan.get(), connecting_table_idx)) {
						// Columns are accessible - no join needed
						connecting_table_to_orders_table[connecting_table_idx] = connecting_table_idx;
#ifdef DEBUG
						Printer::Print(
						    "ModifyPlanWithoutPU: Connecting table #" + std::to_string(connecting_table_idx) + " (" +
						    connecting_table_name +
						    ") IS the FK table with PU reference and columns are accessible, no join needed");
#endif
						continue; // Skip to next connecting table instance
					}
#ifdef DEBUG
					Printer::Print("ModifyPlanWithoutPU: Connecting table #" + std::to_string(connecting_table_idx) +
					               " (" + connecting_table_name +
					               ") IS the FK table but columns are NOT accessible (blocked by SEMI/ANTI join)");
#endif
					// Columns are NOT accessible - we need to find an accessible table that has
					// a PAC LINK (FK) to the blocked FK table, and add a join from there.
					// Search in ALL scanned tables for tables that have FK to the blocked table.
					bool found_accessible_alternative = false;

					// Build list of all scanned tables to search
					vector<string> all_scanned_tables;
					all_scanned_tables.insert(all_scanned_tables.end(), gets_present.begin(), gets_present.end());
					all_scanned_tables.insert(all_scanned_tables.end(), check.scanned_non_pu_tables.begin(),
					                          check.scanned_non_pu_tables.end());

					for (auto &present_table : all_scanned_tables) {
						if (present_table == fk_table_with_pu_reference) {
							continue; // Skip the blocked table itself
						}

						// Check if this present table has an FK to the blocked FK table
						auto it = check.table_metadata.find(present_table);
						if (it == check.table_metadata.end()) {
							continue;
						}

						bool has_fk_to_blocked = false;
						for (auto &fk : it->second.fks) {
							if (fk.first == fk_table_with_pu_reference) {
								has_fk_to_blocked = true;
								break;
							}
						}

						if (!has_fk_to_blocked) {
							continue;
						}

						// Found a table with FK to the blocked table - check if it's accessible
						vector<unique_ptr<LogicalOperator> *> table_nodes;
						FindAllNodesByTable(&plan, present_table, table_nodes);

						for (auto *table_node : table_nodes) {
							auto &table_get = table_node->get()->Cast<LogicalGet>();
							if (AreTableColumnsAccessible(plan.get(), table_get.table_index)) {
								// Found an accessible table with FK to the blocked FK table
								// Add a join from this table to the FK table
#ifdef DEBUG
								Printer::Print("ModifyPlanWithoutPU: Found accessible table " + present_table + " #" +
								               std::to_string(table_get.table_index) +
								               " with FK to blocked table - adding join to " +
								               fk_table_with_pu_reference);
#endif
								unique_ptr<LogicalOperator> current_node = (*table_node)->Copy(input.context);
								auto local_idx = GetNextTableIndex(plan);

								// Create a join to the FK table
								// Get required columns for this table (PKs and relevant FKs)
								auto required_cols = GetRequiredColumnsForTable(check, fk_table_with_pu_reference,
								                                                fk_path, privacy_units);
								auto new_get = CreateLogicalGet(input.context, plan, fk_table_with_pu_reference,
								                                local_idx, required_cols);
								idx_t fk_table_idx = local_idx;

								auto join = CreateLogicalJoin(check, input.context, std::move(current_node),
								                              std::move(new_get));

								// Map the accessible table to the new FK table instance
								connecting_table_to_orders_table[table_get.table_index] = fk_table_idx;

								ReplaceNode(plan, *table_node, join);
								found_accessible_alternative = true;
								break;
							}
						}

						if (found_accessible_alternative) {
							break;
						}
					}

					if (found_accessible_alternative) {
						continue; // Move to next connecting table instance
					}

					// If still not found, this connecting table instance simply cannot be used
					// for PAC transformation - skip it (don't throw an exception, as other
					// code paths may handle this aggregate differently)
#ifdef DEBUG
					Printer::Print("ModifyPlanWithoutPU: No accessible alternative found for blocked FK table " +
					               fk_table_with_pu_reference + " - skipping this connecting table instance");
#endif
					continue;
				}

				// Build list of tables we need to join to reach fk_table_with_pu_reference
				vector<string> tables_to_add;
				bool found_connecting = false;
				bool found_fk_table = false;

				for (auto &table_in_path : fk_path) {
					if (table_in_path == connecting_table_name) {
						found_connecting = true;
						continue; // Skip the connecting table itself
					}
					if (found_connecting && !found_fk_table) {
						tables_to_add.push_back(table_in_path);
						if (table_in_path == fk_table_with_pu_reference) {
							found_fk_table = true;
						}
					}
				}

#ifdef DEBUG
				Printer::Print("ModifyPlanWithoutPU: No accessible FK table found, adding join chain for " +
				               std::to_string(tables_to_add.size()) + " tables to connecting table #" +
				               std::to_string(connecting_table_idx));
				for (auto &t : tables_to_add) {
					Printer::Print("  - " + t);
				}
#endif

				if (tables_to_add.empty()) {
					continue;
				}

				// IMPORTANT: Before adding the join chain to this connecting table,
				// check if its columns are actually accessible from the plan root.
				// If the connecting table is in the right branch of a SEMI/ANTI join,
				// any columns we add via joins won't be able to propagate to the aggregate.
				// In that case, we need to find an accessible table and add the FULL join chain there.
				if (!AreTableColumnsAccessible(plan.get(), connecting_table_idx)) {
#ifdef DEBUG
					Printer::Print("ModifyPlanWithoutPU: Connecting table #" + std::to_string(connecting_table_idx) +
					               " (" + connecting_table_name +
					               ") columns are NOT accessible - looking for accessible alternative");
#endif
					// Find an accessible table in the FK path and add the full join chain from there
					bool found_accessible_alternative = false;

					// Search all scanned tables for one that is accessible and has an FK path to the PU
					// IMPORTANT: We must check ALL FK paths, not just the single fk_path parameter,
					// because accessible tables might have their own FK paths (e.g., shipments -> order_items -> orders
					// -> users)
					for (auto &scanned_table : check.scanned_non_pu_tables) {
						// Skip the connecting table itself - we already know it's not accessible
						if (scanned_table == connecting_table_name) {
							continue;
						}

						// Check if this table has its own FK path to the PU
						auto scanned_fk_path_it = check.fk_paths.find(scanned_table);
						if (scanned_fk_path_it == check.fk_paths.end() || scanned_fk_path_it->second.empty()) {
							continue; // This table has no FK path to PU
						}
						const vector<string> &scanned_fk_path = scanned_fk_path_it->second;

						// Find all instances of this table and check if any is accessible
						vector<unique_ptr<LogicalOperator> *> table_nodes;
						FindAllNodesByTable(&plan, scanned_table, table_nodes);

						for (auto *table_node : table_nodes) {
							auto &table_get = table_node->get()->Cast<LogicalGet>();
							// Skip if this is actually the same node as our connecting table
							if (table_get.table_index == connecting_table_idx) {
								continue;
							}
							if (AreTableColumnsAccessible(plan.get(), table_get.table_index)) {
								// Found an accessible table with its own FK path
								// Build the full join chain from this table to the FK table with PU reference
								vector<string> full_tables_to_add;
								bool found_start = false;
								bool found_end = false;

								// Use this table's own FK path to build the join chain
								for (auto &path_table : scanned_fk_path) {
									if (path_table == scanned_table) {
										found_start = true;
										continue; // Skip the starting table itself
									}
									if (found_start && !found_end) {
										full_tables_to_add.push_back(path_table);
										if (path_table == fk_table_with_pu_reference) {
											found_end = true;
										}
									}
								}

								if (full_tables_to_add.empty()) {
									continue;
								}

#ifdef DEBUG
								Printer::Print("ModifyPlanWithoutPU: Found accessible table " + scanned_table + " #" +
								               std::to_string(table_get.table_index) + " - adding full join chain:");
								for (auto &t : full_tables_to_add) {
									Printer::Print("  - " + t);
								}
#endif
								// Create the full join chain
								unique_ptr<LogicalOperator> current_node = (*table_node)->Copy(input.context);
								auto local_idx = GetNextTableIndex(plan);
								idx_t fk_table_idx = DConstants::INVALID_INDEX;

								for (auto &table_to_add : full_tables_to_add) {
									auto it = check.table_metadata.find(table_to_add);
									if (it == check.table_metadata.end()) {
										throw InternalException("PAC compiler: missing table metadata for table: " +
										                        table_to_add);
									}

									auto required_cols =
									    GetRequiredColumnsForTable(check, table_to_add, fk_path, privacy_units);
									auto table_get_new =
									    CreateLogicalGet(input.context, plan, table_to_add, local_idx, required_cols);

									if (table_to_add == fk_table_with_pu_reference) {
										fk_table_idx = local_idx;
									}

									auto join = CreateLogicalJoin(check, input.context, std::move(current_node),
									                              std::move(table_get_new));
									current_node = std::move(join);
									local_idx++;
								}

								if (fk_table_idx != DConstants::INVALID_INDEX) {
									connecting_table_to_orders_table[table_get.table_index] = fk_table_idx;
								}

								ReplaceNode(plan, *table_node, current_node);
								found_accessible_alternative = true;
								break;
							}
						}

						if (found_accessible_alternative) {
							continue; // Move to next connecting table instance
						}
					}

					if (found_accessible_alternative) {
						continue; // Move to next connecting table instance
					}

#ifdef DEBUG
					Printer::Print("ModifyPlanWithoutPU: No accessible alternative found - skipping this instance");
#endif
					continue;
				}

				// Create joins for all intermediate tables
				// IMPORTANT: Use tables_to_add (NOT tables_to_join_for_instance) - this contains the path
				// from connecting_table to fk_table_with_pu_reference that we computed above
				unique_ptr<LogicalOperator> existing_node = (*target_ref)->Copy(input.context);

				// Create fresh LogicalGet nodes for this instance
				auto local_idx = GetNextTableIndex(plan);
				std::unordered_map<string, unique_ptr<LogicalGet>> local_get_map;

				for (auto &table : tables_to_add) {
					auto it = check.table_metadata.find(table);
					if (it == check.table_metadata.end()) {
						throw InternalException("PAC compiler: missing table metadata for table: " + table);
					}
					vector<string> pks = it->second.pks;
					// Get required columns for this table (PKs and relevant FKs)
					auto required_cols = GetRequiredColumnsForTable(check, table, fk_path, privacy_units);
					auto get = CreateLogicalGet(input.context, plan, table, local_idx, required_cols);

					// Remember the table index of the FK table for this instance
					if (table == fk_table_with_pu_reference) {
						connecting_table_to_orders_table[connecting_table_idx] = local_idx;
					}

					local_get_map[table] = std::move(get);
					local_idx++;
				}

				unique_ptr<LogicalOperator> final_join;
				for (size_t i = 0; i < tables_to_add.size(); ++i) {
					auto &tbl_name = tables_to_add[i];
					unique_ptr<LogicalGet> right_op = std::move(local_get_map[tbl_name]);
					if (!right_op) {
						throw InternalException("PAC compiler: failed to transfer ownership of LogicalGet for " +
						                        tbl_name);
					}

					if (i == 0) {
						final_join =
						    CreateLogicalJoin(check, input.context, std::move(existing_node), std::move(right_op));
					} else {
						final_join =
						    CreateLogicalJoin(check, input.context, std::move(final_join), std::move(right_op));
					}
				}

				// Replace this instance with the join chain
				ReplaceNode(plan, *target_ref, final_join);
			}
		}
	}

#if DEBUG
	plan->Print();
#endif

	// Now find all aggregates and modify them with PAC functions
	// Each aggregate needs a hash expression based on the FK table in its subtree
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

	// For each target aggregate, find which FK table it has access to and build hash expression
	// IMPORTANT: We need to find the "closest" FK table instance for each aggregate
	// to handle correlated subqueries correctly
	for (auto *target_agg : target_aggregates) {
		// Find which connecting table (lineitem) this aggregate has in its DIRECT path
		// (not in a nested subquery), and use the corresponding FK table (orders)
		idx_t orders_table_idx = DConstants::INVALID_INDEX;

		// We need to find the "closest" connecting table to this aggregate
		// For nested queries, the outer aggregate might contain both inner and outer tables
		// So we need to find which connecting table is in the aggregate's direct path
		// Strategy: check which connecting table indices exist, and use the one that's NOT in a DELIM_GET
		// IMPORTANT: Also check that the table's columns are actually accessible (not blocked by MARK/SEMI/ANTI joins)

		// First, collect all connecting table indices that appear in the aggregate's subtree
		// AND whose columns are actually accessible (not in right side of MARK/SEMI/ANTI join)
		// IMPORTANT: We must also verify that the MAPPED orders table is in the aggregate's subtree,
		// not just the connecting table. This is crucial for CROSS_PRODUCT queries where the
		// connecting table and orders table might be in different branches.
		vector<idx_t> candidate_conn_tables;
		for (auto &kv : connecting_table_to_orders_table) {
			idx_t conn_table_idx = kv.first;
			idx_t orders_idx = kv.second;

			// Check if BOTH the connecting table AND the orders table are in the aggregate's subtree
			if (HasTableIndexInSubtree(target_agg, conn_table_idx) && HasTableIndexInSubtree(target_agg, orders_idx)) {
				// Also check that both tables' columns are accessible (not blocked by MARK/SEMI/ANTI joins)
				if (AreTableColumnsAccessible(target_agg, conn_table_idx) &&
				    AreTableColumnsAccessible(target_agg, orders_idx)) {
					candidate_conn_tables.push_back(conn_table_idx);
				}
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
						if (has_fk_to_pu) {
							break;
						}
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
									if (!fk_cols.empty()) {
										break;
									}
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

									// ALWAYS propagate the hash expression through any intermediate operators
									// between the table scan and the aggregate. This ensures:
									// 1. Projection maps are updated to include the new column
									// 2. Operator types are re-resolved
									// 3. The binding is correctly mapped through the operator chain
									// Even if the aggregate reads directly from the scan, PropagatePKThroughProjections
									// will handle it correctly (returning the original expression if no intermediate
									// ops)
									auto hash_input_expr = PropagatePKThroughProjections(
									    *plan, node_get, std::move(base_hash_expr), target_agg);

									// Skip if no direct path found (e.g., table is behind nested aggregate)
									if (!hash_input_expr) {
										continue;
									}

#ifdef DEBUG
									Printer::Print("ModifyPlanWithoutPU: FK table #" + std::to_string(node_table_idx) +
									               " hash expression propagated");
#endif

									// Modify this aggregate with PAC functions
									ModifyAggregatesWithPacFunctions(input, target_agg, hash_input_expr);

									// Mark as handled so we don't process this aggregate again
									handled_via_direct_fk = true;
									break;
								}
							}

							if (handled_via_direct_fk) {
								break;
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

										// Skip if no direct path found
										if (!hash_input_expr) {
											continue;
										}

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
		}

		// Skip to next aggregate if we already handled this one via direct FK
		if (handled_via_direct_fk) {
			continue;
		}

		// Also skip if there are no candidate connecting tables (safety check)
		// This should not happen if the above logic is correct, but ensures we don't proceed with empty candidates
		if (candidate_conn_tables.empty()) {
#ifdef DEBUG
			Printer::Print("ModifyPlanWithoutPU: Skipping aggregate with no candidate connecting tables");
#endif
			continue;
		}

		// If there's only one candidate, use it
		if (candidate_conn_tables.size() == 1) {
			orders_table_idx = connecting_table_to_orders_table[candidate_conn_tables[0]];
		} else {
			// Multiple candidates - collect all possible orders table indices
			// We'll try each one and use the first where propagation actually succeeds
			vector<idx_t> candidate_orders_tables;
			for (auto conn_table_idx : candidate_conn_tables) {
				idx_t ord_table_idx = connecting_table_to_orders_table[conn_table_idx];
				candidate_orders_tables.push_back(ord_table_idx);
			}
			// Sort by table index (smaller indices first - outer query tables)
			std::sort(candidate_orders_tables.begin(), candidate_orders_tables.end());
			// Remove duplicates
			candidate_orders_tables.erase(std::unique(candidate_orders_tables.begin(), candidate_orders_tables.end()),
			                              candidate_orders_tables.end());

#ifdef DEBUG
			Printer::Print("ModifyPlanWithoutPU: Trying " + std::to_string(candidate_orders_tables.size()) +
			               " candidate orders tables for aggregate");
#endif

			// Try each candidate until propagation succeeds
			bool found_working_table = false;
			for (auto test_ord_idx : candidate_orders_tables) {
				vector<unique_ptr<LogicalOperator> *> test_nodes;
				FindAllNodesByTableIndex(&plan, test_ord_idx, test_nodes);
				if (test_nodes.empty()) {
					continue;
				}

				auto &test_get = test_nodes[0]->get()->Cast<LogicalGet>();

				// Get FK columns for this table
				vector<string> test_fk_cols;
				auto test_table_ptr = test_get.GetTable();
				if (test_table_ptr) {
					auto it = check.table_metadata.find(test_table_ptr->name);
					if (it != check.table_metadata.end()) {
						for (auto &fk : it->second.fks) {
							for (auto &pu : privacy_units) {
								if (fk.first == pu) {
									test_fk_cols = fk.second;
									break;
								}
							}
							if (!test_fk_cols.empty()) {
								break;
							}
						}
					}
				}
				if (test_fk_cols.empty()) {
					continue;
				}

				// Ensure FK columns are projected
				bool proj_ok = true;
				for (auto &fk_col : test_fk_cols) {
					if (EnsureProjectedColumn(test_get, fk_col) == DConstants::INVALID_INDEX) {
						proj_ok = false;
						break;
					}
				}
				if (!proj_ok) {
					continue;
				}

				// Try to propagate - this will return nullptr if blocked by DELIM_JOIN etc.
				auto test_hash = BuildXorHashFromPKs(input, test_get, test_fk_cols);
				auto test_result = PropagatePKThroughProjections(*plan, test_get, std::move(test_hash), target_agg);

				if (test_result) {
#ifdef DEBUG
					Printer::Print("ModifyPlanWithoutPU: Propagation succeeded for orders table #" +
					               std::to_string(test_ord_idx));
#endif
					ModifyAggregatesWithPacFunctions(input, target_agg, test_result);
					found_working_table = true;
					break;
				}
#ifdef DEBUG
				Printer::Print("ModifyPlanWithoutPU: Propagation failed for orders table #" +
				               std::to_string(test_ord_idx) + ", trying next candidate");
#endif
			}

			if (found_working_table) {
				continue; // Move to next aggregate - this one is handled
			}

			// No candidate worked - use first one (will likely fail in the code below)
			orders_table_idx = candidate_orders_tables.empty() ? DConstants::INVALID_INDEX : candidate_orders_tables[0];
		}

		if (orders_table_idx == DConstants::INVALID_INDEX) {
			throw InternalException("PAC Compiler: could not find orders table for aggregate");
		}

		// IMPORTANT: Check if the selected orders table is actually accessible from this aggregate's subtree
		// If not (e.g., it's in the outer query of a correlated subquery), we need to use AddColumnToDelimJoin
		bool orders_in_aggregate_subtree = HasTableIndexInSubtree(target_agg, orders_table_idx);

		if (!orders_in_aggregate_subtree) {
			// The selected orders table is not in this aggregate's subtree
			// This can happen when the connecting_table_to_orders_table mapping points to a table
			// in a different branch (e.g., correlated subquery vs uncorrelated subquery)
			//
			// First, try to find ANY FK table (table with FK to PU) directly in this aggregate's subtree
			// This handles uncorrelated subqueries that have their own FK table instance
			bool found_direct_fk = false;

			for (auto &present_table : gets_present) {
				auto it = check.table_metadata.find(present_table);
				if (it == check.table_metadata.end()) {
					continue;
				}

				// Check if this table has FK to PU
				bool has_fk_to_pu = false;
				vector<string> fk_cols;
				for (auto &fk : it->second.fks) {
					for (auto &pu : privacy_units) {
						if (fk.first == pu) {
							has_fk_to_pu = true;
							fk_cols = fk.second;
							break;
						}
					}
					if (has_fk_to_pu) {
						break;
					}
				}

				if (!has_fk_to_pu || fk_cols.empty()) {
					continue;
				}

				// Find all instances of this FK table and check which one is in the aggregate's subtree
				vector<unique_ptr<LogicalOperator> *> fk_table_nodes;
				FindAllNodesByTable(&plan, present_table, fk_table_nodes);

				for (auto *node : fk_table_nodes) {
					auto &node_get = node->get()->Cast<LogicalGet>();
					idx_t node_table_idx = node_get.table_index;

					if (HasTableIndexInSubtree(target_agg, node_table_idx)) {
						// Found an FK table directly in this aggregate's subtree - use it
#ifdef DEBUG
						Printer::Print("ModifyPlanWithoutPU: Found FK table " + present_table + " #" +
						               std::to_string(node_table_idx) + " directly in aggregate subtree");
#endif
						// Ensure FK columns are projected
						for (auto &fk_col : fk_cols) {
							idx_t proj_idx = EnsureProjectedColumn(node_get, fk_col);
							if (proj_idx == DConstants::INVALID_INDEX) {
								throw InternalException("PAC compiler: failed to project FK column " + fk_col);
							}
						}

						// Build hash expression
						auto base_hash_expr = BuildXorHashFromPKs(input, node_get, fk_cols);
						auto hash_input_expr =
						    PropagatePKThroughProjections(*plan, node_get, std::move(base_hash_expr), target_agg);

						// Modify this aggregate with PAC functions
						ModifyAggregatesWithPacFunctions(input, target_agg, hash_input_expr);

						found_direct_fk = true;
						break;
					}
				}

				if (found_direct_fk) {
					break;
				}
			}

			if (found_direct_fk) {
				continue; // Move to next aggregate
			}

			// If still not found, fall back to AddColumnToDelimJoin for correlated subqueries
#ifdef DEBUG
			Printer::Print("ModifyPlanWithoutPU: Orders table #" + std::to_string(orders_table_idx) +
			               " not in aggregate subtree, using AddColumnToDelimJoin");
#endif
			// Find the orders table LogicalGet
			vector<unique_ptr<LogicalOperator> *> orders_nodes;
			FindAllNodesByTableIndex(&plan, orders_table_idx, orders_nodes);

			if (orders_nodes.empty()) {
				throw InternalException("PAC Compiler: could not find orders LogicalGet with index " +
				                        std::to_string(orders_table_idx));
			}

			auto &orders_get = orders_nodes[0]->get()->Cast<LogicalGet>();

			// Get FK columns from this table
			vector<string> fk_cols;
			string fk_table_name;
			auto orders_table_ptr = orders_get.GetTable();
			if (orders_table_ptr) {
				fk_table_name = orders_table_ptr->name;
			}

			if (!fk_table_name.empty()) {
				auto it = check.table_metadata.find(fk_table_name);
				if (it != check.table_metadata.end()) {
					for (auto &fk : it->second.fks) {
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

			// Add FK columns to DELIM_JOIN and build hash from DELIM_GET binding
			vector<DelimColumnResult> delim_results;
			for (auto &fk_col : fk_cols) {
				auto delim_result = AddColumnToDelimJoin(plan, orders_get, fk_col, target_agg);
				if (!delim_result.IsValid()) {
					throw InternalException("PAC Compiler: failed to add " + fk_col + " to DELIM_JOIN");
				}
				delim_results.push_back(delim_result);
			}

			// Build hash expression using the DELIM_GET bindings
			unique_ptr<Expression> hash_input_expr;
			if (delim_results.size() == 1) {
				auto col_ref = make_uniq<BoundColumnRefExpression>(delim_results[0].type, delim_results[0].binding);
				hash_input_expr = input.optimizer.BindScalarFunction("hash", std::move(col_ref));
			} else {
				auto first_col = make_uniq<BoundColumnRefExpression>(delim_results[0].type, delim_results[0].binding);
				unique_ptr<Expression> xor_expr = std::move(first_col);

				for (size_t i = 1; i < delim_results.size(); i++) {
					auto next_col =
					    make_uniq<BoundColumnRefExpression>(delim_results[i].type, delim_results[i].binding);
					xor_expr = input.optimizer.BindScalarFunction("xor", std::move(xor_expr), std::move(next_col));
				}
				hash_input_expr = input.optimizer.BindScalarFunction("hash", std::move(xor_expr));
			}

			// Modify this aggregate with PAC functions
			ModifyAggregatesWithPacFunctions(input, target_agg, hash_input_expr);
			continue; // Move to next aggregate
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

		// If propagation failed (e.g., blocked by DELIM_JOIN with RIGHT_SEMI), try AddColumnToDelimJoin
		// This handles cases where the FK table is accessible but its columns can't be propagated
		// directly through the operator chain due to semi/anti join semantics
		if (!hash_input_expr) {
#ifdef DEBUG
			Printer::Print("ModifyPlanWithoutPU: Direct propagation failed for orders table #" +
			               std::to_string(orders_table_idx) + ", trying AddColumnToDelimJoin fallback");
#endif
			// Try to add FK columns via DELIM_JOIN
			vector<DelimColumnResult> delim_results;
			bool delim_fallback_ok = true;
			for (auto &fk_col : fk_cols) {
				auto delim_result = AddColumnToDelimJoin(plan, orders_get, fk_col, target_agg);
				if (!delim_result.IsValid()) {
					delim_fallback_ok = false;
					break;
				}
				delim_results.push_back(delim_result);
			}

			if (delim_fallback_ok && !delim_results.empty()) {
				// Successfully added FK columns via DELIM_JOIN - build hash from DELIM_GET bindings
				if (delim_results.size() == 1) {
					auto col_ref = make_uniq<BoundColumnRefExpression>(delim_results[0].type, delim_results[0].binding);
					hash_input_expr = input.optimizer.BindScalarFunction("hash", std::move(col_ref));
				} else {
					auto first_col =
					    make_uniq<BoundColumnRefExpression>(delim_results[0].type, delim_results[0].binding);
					unique_ptr<Expression> xor_expr = std::move(first_col);

					for (size_t i = 1; i < delim_results.size(); i++) {
						auto next_col =
						    make_uniq<BoundColumnRefExpression>(delim_results[i].type, delim_results[i].binding);
						xor_expr = input.optimizer.BindScalarFunction("xor", std::move(xor_expr), std::move(next_col));
					}
					hash_input_expr = input.optimizer.BindScalarFunction("hash", std::move(xor_expr));
				}
#ifdef DEBUG
				Printer::Print("ModifyPlanWithoutPU: AddColumnToDelimJoin fallback succeeded for orders table #" +
				               std::to_string(orders_table_idx));
#endif
			}
		}

		// Skip if still no hash expression - this can happen for nested aggregates that operate
		// on already-aggregated data (e.g., avg(total) over a GROUP BY sum). These aggregates
		// don't need PAC transformation because they consume already-protected data.
		if (!hash_input_expr) {
#ifdef DEBUG
			Printer::Print("ModifyPlanWithoutPU: Skipping aggregate - no direct path to orders table #" +
			               std::to_string(orders_table_idx) + " (likely a nested aggregate on aggregated data)");
#endif
			continue;
		}

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
 *      *      * Build hash expression: hash(pk) or hash(xor(pk1, pk2, ...)) for composite PKs
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
	// // 1. Privacy unit tables themselves
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
			// AND if the PU table's columns are actually accessible
			// (not blocked by MARK/SEMI/ANTI joins from IN/EXISTS subqueries)
			bool pu_in_subtree = HasTableInSubtree(target_agg, pu_table_name);
			bool pu_columns_accessible = false;
			LogicalGet *get_ptr = nullptr;

			if (pu_in_subtree) {
				get_ptr = FindTableScanInSubtree(target_agg, pu_table_name);
				if (get_ptr) {
					pu_columns_accessible = AreTableColumnsAccessible(target_agg, get_ptr->table_index);
				}
			}

			if (pu_in_subtree && pu_columns_accessible && get_ptr) {
				// Direct PU scan case: use PU's primary key
				// The PU table is in the subtree AND its columns are accessible
				auto &get = *get_ptr;

#ifdef DEBUG
				Printer::Print("ModifyPlanWithPU: Processing table " + pu_table_name + " #" +
				               std::to_string(get.table_index) +
				               " for aggregate, column_ids.size=" + std::to_string(get.GetColumnIds().size()) +
				               ", projection_ids.size=" + std::to_string(get.projection_ids.size()));
#endif

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

				// Skip if propagation failed (e.g., blocked by RIGHT_SEMI/RIGHT_ANTI join)
				if (!hash_input_expr) {
#ifdef DEBUG
					Printer::Print("ModifyPlanWithPU: Skipping PU table " + pu_table_name +
					               " - propagation failed (likely blocked by semi/anti join)");
#endif
					continue;
				}

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

					// Ensure FK columns are projected
					AddPKColumns(fk_get, fk_cols);

					// Build hash expression from FK columns
					auto fk_hash_expr = BuildXorHashFromPKs(input, fk_get, fk_cols);

					// Propagate through projections
					fk_hash_expr = PropagatePKThroughProjections(*plan, fk_get, std::move(fk_hash_expr), target_agg);

					// Skip if propagation failed (e.g., blocked by RIGHT_SEMI/RIGHT_ANTI join)
					if (!fk_hash_expr) {
#ifdef DEBUG
						Printer::Print("ModifyPlanWithPU: Skipping FK table " + fk_table +
						               " - propagation failed (likely blocked by semi/anti join)");
#endif
						continue;
					}

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
	Printer::Print("=== PAC BITSLICE COMPILATION ===");
	Printer::Print("Query hash: " + query_hash);
	Printer::Print("Query: " + query.substr(0, 100) + (query.length() > 100 ? "..." : ""));
	Printer::Print("Privacy units: " + std::to_string(privacy_units.size()));
	for (auto &pu : privacy_units) {
		Printer::Print("  " + pu);
	}
	Printer::Print("Scanned PU tables: " + std::to_string(check.scanned_pu_tables.size()));
	Printer::Print("Scanned non-PU tables: " + std::to_string(check.scanned_non_pu_tables.size()));
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

	// NOTE: The plan coming in is already optimized WITHOUT COLUMN_LIFETIME and COMPRESSED_MATERIALIZATION
	// because the pre-optimizer (PACPreOptimizeFunction) disabled them before built-in optimizers ran.
	// We'll run those optimizers ourselves at the end of this function after PAC transformation.

#ifdef DEBUG
	Printer::Print("=== PLAN BEFORE PAC TRANSFORMATION ===");
	plan->Print();
#endif

	// Build two vectors: present (GETs already in the plan) and missing (GETs to create)
	vector<string> gets_present;
	vector<string> gets_missing;

	// For multi-PU support, we need to gather FK paths and missing tables for all PUs
	// and deduplicate the tables that need to be joined
	std::unordered_set<string> all_missing_tables;
	// We'll use the first FK path as the base

	if (pu_present_in_tree) {
		// Case a) query scans PU table(s) - all PUs are in scanned_pu_tables
		ModifyPlanWithPU(input, plan, check.scanned_pu_tables, check);
	} else if (!check.fk_paths.empty()) {
		// Case b) query does not scan PU table(s): follow FK paths
		// Note: Tables with protected columns are now treated as implicit privacy units,
		// so their paths are included in fk_paths automatically.
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
		vector<string> fk_path_to_use = it->second;

		// Convert set back to vector for ModifyPlanWithoutPU
		vector<string> unique_gets_missing(all_missing_tables.begin(), all_missing_tables.end());

		ModifyPlanWithoutPU(check, input, plan, unique_gets_missing, gets_present, fk_path_to_use, privacy_units);
	}

	// ============================================================================
	// CATEGORICAL QUERY HANDLING
	// ============================================================================
	// After the standard PAC transformation, check if this is a categorical query
	// (outer query compares against inner PAC aggregate without its own aggregate).
	// If so, rewrite to use _counters variants and pac_filter for probabilistic filtering.
	vector<CategoricalPatternInfo> categorical_patterns;
	if (IsCategoricalQuery(plan, categorical_patterns)) {
#ifdef DEBUG
		Printer::Print("=== CATEGORICAL QUERY DETECTED ===");
		Printer::Print("Found " + std::to_string(categorical_patterns.size()) + " categorical pattern(s)");
#endif
		RewriteCategoricalQuery(input, plan, categorical_patterns);
#ifdef DEBUG
		Printer::Print("=== CATEGORICAL REWRITE COMPLETE ===");
#endif
	}

#ifdef DEBUG
	Printer::Print("=== PAC-OPTIMIZED PLAN ===");
	plan->Print();
	Printer::Print("=== END PAC-OPTIMIZED PLAN ===");
#endif

	// NOTE: The deferred optimizers (COLUMN_LIFETIME and COMPRESSED_MATERIALIZATION) are now
	// run centrally in PACRewriteRuleFunction after this function returns. This ensures they
	// run for ALL queries (not just PAC-compiled ones) since we disable them in pre-optimize.

#ifdef DEBUG
	Printer::Print("=== PAC COMPILATION END ===");
#endif
}
} // namespace duckdb
