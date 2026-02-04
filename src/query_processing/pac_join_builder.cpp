//
// PAC Join Builder - Implementation
//
// See pac_join_builder.hpp for documentation.
//
// Created by refactoring pac_bitslice_compiler.cpp
//

#include "query_processing/pac_join_builder.hpp"
#include "compiler/pac_compiler_helpers.hpp"

namespace duckdb {

std::unordered_map<string, unique_ptr<LogicalGet>>
CreateLogicalGetsForTables(ClientContext &context, unique_ptr<LogicalOperator> &plan,
                           const PACCompatibilityResult &check, const vector<string> &tables, idx_t &next_table_index) {
	std::unordered_map<string, unique_ptr<LogicalGet>> get_map;

	for (auto &table : tables) {
		auto it = check.table_metadata.find(table);
		if (it == check.table_metadata.end()) {
			throw InternalException("PAC compiler: missing table metadata for table: " + table);
		}

		auto get = CreateLogicalGet(context, plan, table, next_table_index);
		get_map[table] = std::move(get);
		next_table_index++;
	}

	return get_map;
}

JoinChainResult BuildJoinChain(const PACCompatibilityResult &check, ClientContext &context,
                               unique_ptr<LogicalOperator> &plan, unique_ptr<LogicalOperator> existing_node,
                               const vector<string> &tables_to_join, const string &fk_table_with_pu_reference,
                               idx_t &next_table_index) {
	if (tables_to_join.empty()) {
		return JoinChainResult();
	}

	// Create LogicalGet nodes for all tables to join
	auto local_get_map = CreateLogicalGetsForTables(context, plan, check, tables_to_join, next_table_index);

	// Track the FK table index
	idx_t fk_table_index = DConstants::INVALID_INDEX;

	// Find the FK table index before we move the gets
	for (auto &table : tables_to_join) {
		if (table == fk_table_with_pu_reference) {
			// The index was assigned during CreateLogicalGetsForTables
			// We need to find it by looking at the get's table_index
			auto &get = local_get_map[table];
			if (get) {
				fk_table_index = get->table_index;
			}
			break;
		}
	}

	// Build the join chain
	unique_ptr<LogicalOperator> final_join = std::move(existing_node);
	for (size_t i = 0; i < tables_to_join.size(); ++i) {
		auto &tbl_name = tables_to_join[i];
		unique_ptr<LogicalGet> right_op = std::move(local_get_map[tbl_name]);
		if (!right_op) {
			throw InternalException("PAC compiler: failed to transfer ownership of LogicalGet for " + tbl_name);
		}

		final_join = CreateLogicalJoin(check, context, std::move(final_join), std::move(right_op));
	}

	return JoinChainResult(std::move(final_join), fk_table_index);
}

vector<string> DetermineTablesForJoinChain(const vector<string> &fk_path,
                                           const std::unordered_set<string> &missing_tables,
                                           const std::unordered_set<string> &present_tables, bool join_elimination,
                                           const vector<string> &privacy_units) {
	vector<string> ordered_tables;

	// Create a copy of missing_tables that we can modify
	std::unordered_set<string> tables_to_include = missing_tables;

	// If join elimination is enabled, remove PU tables
	if (join_elimination) {
		for (auto &pu : privacy_units) {
			tables_to_include.erase(pu);
		}
	}

	// Build ordered list based on FK path order, only including tables in tables_to_include
	for (auto &table : fk_path) {
		if (tables_to_include.find(table) != tables_to_include.end()) {
			ordered_tables.push_back(table);
		}
	}

	return ordered_tables;
}

string FindConnectingTable(const vector<string> &fk_path, const vector<string> &gets_present,
                           const vector<string> &scanned_non_pu_tables, bool for_missing_tables) {
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
	if (connecting_table_for_joins.empty() && !scanned_non_pu_tables.empty()) {
		connecting_table_for_joins = scanned_non_pu_tables[0];
	}
	if (connecting_table_for_missing.empty() && !scanned_non_pu_tables.empty()) {
		connecting_table_for_missing = scanned_non_pu_tables[0];
	}

	return for_missing_tables ? connecting_table_for_missing : connecting_table_for_joins;
}

} // namespace duckdb
