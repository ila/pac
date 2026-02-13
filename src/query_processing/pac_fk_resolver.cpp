//
// PAC FK Resolver - Implementation
//
// See pac_fk_resolver.hpp for documentation.
//
// Created by refactoring pac_bitslice_compiler.cpp
//

#include "query_processing/pac_fk_resolver.hpp"
#include "query_processing/pac_plan_traversal.hpp"

namespace duckdb {

FKToPUResult GetFKColumnsToPU(const string &table_name, const vector<string> &privacy_units,
                              const PACCompatibilityResult &check) {
	auto it = check.table_metadata.find(table_name);
	if (it != check.table_metadata.end()) {
		for (auto &fk : it->second.fks) {
			for (auto &pu : privacy_units) {
				if (fk.first == pu) {
					return FKToPUResult(fk.second, pu);
				}
			}
		}
	}
	return FKToPUResult();
}

AccessibleFKTableResult FindAccessibleFKTable(unique_ptr<LogicalOperator> &plan, LogicalOperator *search_root,
                                              const vector<string> &candidate_tables,
                                              const vector<string> &privacy_units,
                                              const PACCompatibilityResult &check) {
	AccessibleFKTableResult result;

	for (auto &table_name : candidate_tables) {
		// Check if this table has FK to any PU
		auto fk_result = GetFKColumnsToPU(table_name, privacy_units, check);
		if (!fk_result.found) {
			continue;
		}

		// Find all instances of this table in the plan
		vector<unique_ptr<LogicalOperator> *> table_nodes;
		FindAllNodesByTable(&plan, table_name, table_nodes);

		for (auto *node : table_nodes) {
			auto &node_get = node->get()->Cast<LogicalGet>();
			idx_t node_table_idx = node_get.table_index;

			// Check if this table is in the search root's subtree
			if (search_root && !HasTableIndexInSubtree(search_root, node_table_idx)) {
				continue;
			}

			// Check if this table's columns are accessible
			if (AreTableColumnsAccessible(plan.get(), node_table_idx)) {
				result.get_node = &node_get;
				result.table_index = node_table_idx;
				result.table_name = table_name;
				result.fk_columns = fk_result.fk_columns;
				result.found = true;
				return result;
			}
		}
	}

	return result;
}

} // namespace duckdb
