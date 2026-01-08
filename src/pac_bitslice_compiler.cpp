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

void ModifyPlanWithoutPU(const PACCompatibilityResult &check, OptimizerExtensionInput &input,
                         unique_ptr<LogicalOperator> &plan, const vector<string> &gets_missing,
                         const vector<string> &gets_present, const vector<string> &fk_path,
                         const string &privacy_unit) {
	// Note: we assume we don't use rowid

	// Check if join elimination is enabled
	Value join_elim_val;
	bool join_elimination = false;
	if (input.context.TryGetCurrentSetting("pac_join_elimination", join_elim_val) && !join_elim_val.IsNull()) {
		join_elimination = join_elim_val.GetValue<bool>();
	}

#ifdef DEBUG
	Printer::Print("ModifyPlanWithoutPU: join_elimination = " + std::to_string(join_elimination));
	Printer::Print("ModifyPlanWithoutPU: privacy_unit = " + privacy_unit);
#endif

	// Create the necessary LogicalGets for missing tables
	std::unordered_map<string, unique_ptr<LogicalGet>> get_map;
	// Preserve creation order: store ordered table names (ownership kept in get_map)
	vector<string> ordered_table_names;
	auto idx = GetNextTableIndex(plan);
	for (auto &table : gets_missing) {
		// If join elimination is enabled, skip the PU table
		if (join_elimination && table == privacy_unit) {
#ifdef DEBUG
			Printer::Print("ModifyPlanWithoutPU: Skipping PU table " + table + " due to join elimination");
#endif
			continue;
		}

		auto it = check.table_metadata.find(table);
		if (it == check.table_metadata.end()) {
			throw InternalException("PAC compiler: missing table metadata for missing GET: " + table);
		}
		vector<string> pks = it->second.pks;
		auto get = CreateLogicalGet(input.context, plan, table, idx);
		// store in map (ownership) and record name to preserve order
		get_map[table] = std::move(get);
		ordered_table_names.push_back(table);
#ifdef DEBUG
		Printer::Print("ModifyPlanWithoutPU: Added table " + table + " to join chain");
#endif
		idx++;
	}

	// Find the unique_ptr reference to the existing table that connects to the missing tables
	// We need to find the last present table in the FK path, which is the connection point
	LogicalOperator *parent_join = nullptr;
	idx_t child_idx_in_parent = 0;
	unique_ptr<LogicalOperator> *target_ref = nullptr;

	// Find the last present table in the FK path (ordered)
	// This is the table that connects to the first missing table
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

	// When join elimination is enabled, we don't join to PU table
	// Instead, we use the FK column from the last table in the chain
	unique_ptr<Expression> hash_input_expr;

	if (join_elimination) {
		// Find the last table in the FK chain (the one that would link to PU)
		// This is the last table in ordered_table_names, or if empty, the connecting_table
		string last_table_name;
		if (!ordered_table_names.empty()) {
			last_table_name = ordered_table_names.back();
		} else {
			last_table_name = connecting_table;
		}

		// Find the LogicalGet for this table
		unique_ptr<LogicalOperator> *last_table_ref = FindNodeRefByTable(&plan, last_table_name);
		if (!last_table_ref || !last_table_ref->get()) {
			throw InternalException("PAC compiler: could not find LogicalGet for last table " + last_table_name);
		}
		auto last_get = &last_table_ref->get()->Cast<LogicalGet>();

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

		// Build hash from FK columns
		hash_input_expr = BuildXorHashFromPKs(input, *last_get, fk_cols);
	} else {
		// Original behavior: locate the privacy-unit LogicalGet we just inserted
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

		// Build the hash expression from PU PKs
		hash_input_expr = BuildXorHashFromPKs(input, *pu_get, pu_pks);
	}

	// Now we need to edit the aggregate node to use pac functions
	auto *agg = FindTopAggregate(plan);

	// Use the helper function to modify aggregates with PAC functions
	ModifyAggregatesWithPacFunctions(input, agg, hash_input_expr);
}

void ModifyPlanWithPU(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan, const vector<string> &pks,
                      bool use_rowid, const string &pu_table_name) {

	auto pu_scan_ptr = FindPrivacyUnitGetNode(plan, pu_table_name);
	auto &get = pu_scan_ptr->get()->Cast<LogicalGet>();
	if (use_rowid) {
		AddRowIDColumn(get);
	} else {
		// Ensure primary key columns are present in the LogicalGet (add them if necessary)
		AddPKColumns(get, pks);
	}

	// Now we need to edit the aggregate node to use pac functions
	auto *agg = FindTopAggregate(plan);

	// We create a hash expression over either the row id or the XOR of PK columns
	// Build the hash expression once (will be copied/reused for each aggregate)
	unique_ptr<Expression> hash_input_expr;
	if (use_rowid) {
		// rowid is the last column added
		auto rowid_binding = ColumnBinding(get.table_index, get.GetColumnIds().size() - 1);
		auto rowid_col = make_uniq<BoundColumnRefExpression>(LogicalType::BIGINT, rowid_binding);
		auto bound_hash = input.optimizer.BindScalarFunction("hash", std::move(rowid_col));
		hash_input_expr = std::move(bound_hash);
	} else {
		// Replaced duplicated inline logic with helper
		hash_input_expr = BuildXorHashFromPKs(input, get, pks);
	}

	// Use the helper function to modify aggregates with PAC functions
	ModifyAggregatesWithPacFunctions(input, agg, hash_input_expr);
}

void CompilePacBitsliceQuery(const PACCompatibilityResult &check, OptimizerExtensionInput &input,
                             unique_ptr<LogicalOperator> &plan, const string &privacy_unit, const string &query,
                             const string &query_hash) {

#ifdef DEBUG
	Printer::Print("CompilePacBitsliceQuery called for PU=" + privacy_unit + " hash=" + query_hash);
#endif

	string path = GetPacCompiledPath(input.context, ".");
	if (!path.empty() && path.back() != '/') {
		path.push_back('/');
	}
	string filename = path + privacy_unit + "_" + query_hash + "_bitslice.sql";

	// The bitslice compiler works in the following way:
	// a) the query scans PU table:
	// a.1) the PU table has 1 PK: we hash it
	// a.2) the PU table has multiple PKs: we XOR them and hash the result
	// a.3) the PU table has no PK: we hash rowid
	// b) the query does not scan PU table:
	// b.1) we follow the FK path to find the PK(s) of the PU table
	// b.2) we join the chain of tables from the scanned table to the PU table
	// b.3) we hash the PK(s) as in a)

	// Example: SELECT group_key, SUM(val) AS sum_val FROM t_single GROUP BY group_key;
	// Becomes: SELECT group_key, pac_sum(HASH(rowid), val) AS sum_val FROM t_single GROUP BY group_key;
	// todo- what is MI? what is k?

	bool pu_present_in_tree = false;
	bool use_rowid = false;

	if (!check.scanned_pu_tables.empty()) {
		pu_present_in_tree = true;
	}

	// Replan with selected optimizers disabled
	ReplanWithoutOptimizers(input.context, input.context.GetCurrentQuery(), plan);

	// Case a) query scans PU table (we assume only 1 PAC table for now)
	vector<string> pks;
	// Build two vectors: present (GETs already in the plan) and missing (GETs to create)
	vector<string> gets_present;
	vector<string> gets_missing;
	vector<LogicalGet *> new_gets;

	// fk_path_to_use will be populated when we detect an FK path (used for multi-hop joins)
	vector<string> fk_path_to_use;

	if (pu_present_in_tree) {
		// Look up PKs for the scanned PAC table via table_metadata (filled by the compatibility check)
		if (!check.scanned_pu_tables.empty()) {
			auto tbl = check.scanned_pu_tables[0];
			auto it = check.table_metadata.find(tbl);
			if (it != check.table_metadata.end() && !it->second.pks.empty()) {
				pks = it->second.pks;
			} else {
				// no PKs found -> use rowid
				use_rowid = true;
			}
		} else {
			use_rowid = true;
		}
	} else if (!check.fk_paths.empty()) {
		// The query does not scan the PU table: we need to follow the FK path
		string start_table;
		string target_pu;
		PopulateGetsFromFKPath(check, gets_present, gets_missing, start_table, target_pu);

		// extract the ordered fk_path from the compatibility result
		auto it = check.fk_paths.find(start_table);
		if (it == check.fk_paths.end() || it->second.empty()) {
			throw InternalException("PAC compiler: expected fk_path for start table " + start_table);
		}
		fk_path_to_use = it->second;

#ifdef DEBUG
		Printer::Print("PAC bitslice: FK path detection");
		Printer::Print("start_table: " + start_table);
		Printer::Print("target_pu: " + target_pu);
		Printer::Print("fk_path:");
		for (auto &p : fk_path_to_use) {
			Printer::Print("  " + p);
		}
		Printer::Print("gets_present:");
		for (auto &g : gets_present) {
			Printer::Print("  " + g);
		}
		Printer::Print("gets_missing:");
		for (auto &g : gets_missing) {
			Printer::Print("  " + g);
		}
#endif
		// fk_path_to_use is populated and will be passed into ModifyPlanWithoutPU below
	}

	if (pu_present_in_tree) {
		ModifyPlanWithPU(input, plan, pks, use_rowid, check.scanned_pu_tables[0]);
	} else {
		ModifyPlanWithoutPU(check, input, plan, gets_missing, gets_present, fk_path_to_use, privacy_unit);
	}

	plan->ResolveOperatorTypes();
	plan->Verify(input.context);
#ifdef DEBUG
	plan->Print();
#endif
}

} // namespace duckdb
