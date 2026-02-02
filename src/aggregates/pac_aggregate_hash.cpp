//
// PAC Aggregate Hash Builder - Implementation
//
// See pac_aggregate_hash.hpp for documentation.
//
// Created by refactoring pac_bitslice_compiler.cpp
//

#include "aggregates/pac_aggregate_hash.hpp"
#include "query_processing/pac_expression_builder.hpp"
#include "query_processing/pac_projection_propagation.hpp"
#include "query_processing/pac_plan_traversal.hpp"
#include "query_processing/pac_subquery_handler.hpp"
#include "query_processing/pac_fk_resolver.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"

namespace duckdb {

AggregateHashResult BuildHashForAggregateFromFKTable(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                                                     LogicalAggregate *target_agg, LogicalGet &fk_get,
                                                     const vector<string> &fk_columns) {
	// Ensure FK columns are projected
	for (auto &fk_col : fk_columns) {
		idx_t proj_idx = EnsureProjectedColumn(fk_get, fk_col);
		if (proj_idx == DConstants::INVALID_INDEX) {
			return AggregateHashResult();
		}
	}

	// Build hash expression
	auto base_hash_expr = BuildXorHashFromPKs(input, fk_get, fk_columns);

	// Propagate through projections to the aggregate
	auto hash_input_expr = PropagatePKThroughProjections(*plan, fk_get, std::move(base_hash_expr), target_agg);

	if (!hash_input_expr) {
		return AggregateHashResult();
	}

	return AggregateHashResult(std::move(hash_input_expr));
}

AggregateHashResult BuildHashForAggregateFromPU(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                                                LogicalAggregate *target_agg, LogicalGet &pu_get,
                                                const vector<string> &pk_columns, bool use_rowid) {
	unique_ptr<Expression> hash_input_expr;

	if (use_rowid) {
		AddRowIDColumn(pu_get);
		auto rowid_binding = ColumnBinding(pu_get.table_index, pu_get.GetColumnIds().size() - 1);
		auto rowid_col = make_uniq<BoundColumnRefExpression>(LogicalType::BIGINT, rowid_binding);
		hash_input_expr = input.optimizer.BindScalarFunction("hash", std::move(rowid_col));
	} else {
		AddPKColumns(pu_get, pk_columns);
		hash_input_expr = BuildXorHashFromPKs(input, pu_get, pk_columns);
	}

	// Propagate through projections
	hash_input_expr = PropagatePKThroughProjections(*plan, pu_get, std::move(hash_input_expr), target_agg);

	if (!hash_input_expr) {
		return AggregateHashResult();
	}

	return AggregateHashResult(std::move(hash_input_expr));
}

AggregateHashResult BuildHashForAggregateViaDelimJoin(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                                                      LogicalAggregate *target_agg, LogicalGet &fk_get,
                                                      const vector<string> &fk_columns) {
	// Add FK columns to DELIM_JOIN and build hash from DELIM_GET binding
	vector<DelimColumnResult> delim_results;
	for (auto &fk_col : fk_columns) {
		auto delim_result = AddColumnToDelimJoin(plan, fk_get, fk_col, target_agg);
		if (!delim_result.IsValid()) {
			return AggregateHashResult();
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
			auto next_col = make_uniq<BoundColumnRefExpression>(delim_results[i].type, delim_results[i].binding);
			xor_expr = input.optimizer.BindScalarFunction("xor", std::move(xor_expr), std::move(next_col));
		}
		hash_input_expr = input.optimizer.BindScalarFunction("hash", std::move(xor_expr));
	}

	return AggregateHashResult(std::move(hash_input_expr));
}

bool ProcessAggregateForPAC(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                            LogicalAggregate *target_agg, const std::unordered_map<idx_t, idx_t> &connecting_to_fk_map,
                            const vector<string> &gets_present, const vector<string> &privacy_units,
                            const vector<string> &fk_path, const PACCompatibilityResult &check) {
	// Collect candidate connecting tables that are in the aggregate's subtree
	vector<idx_t> candidate_conn_tables;
	for (auto &kv : connecting_to_fk_map) {
		idx_t conn_table_idx = kv.first;
		idx_t fk_table_idx = kv.second;

		if (HasTableIndexInSubtree(target_agg, conn_table_idx) && HasTableIndexInSubtree(target_agg, fk_table_idx)) {
			if (AreTableColumnsAccessible(target_agg, conn_table_idx) &&
			    AreTableColumnsAccessible(target_agg, fk_table_idx)) {
				candidate_conn_tables.push_back(conn_table_idx);
			}
		}
	}

	// If no candidates found, try to find a direct FK table
	if (candidate_conn_tables.empty()) {
		auto accessible_fk = FindAccessibleFKTable(plan, target_agg, gets_present, privacy_units, check);
		if (accessible_fk.found) {
			auto result = BuildHashForAggregateFromFKTable(input, plan, target_agg, *accessible_fk.get_node,
			                                               accessible_fk.fk_columns);
			if (result.success) {
				ModifyAggregatesWithPacFunctions(input, target_agg, result.hash_expr);
				return true;
			}
		}
		return false;
	}

	// Try each candidate
	for (auto conn_table_idx : candidate_conn_tables) {
		auto it = connecting_to_fk_map.find(conn_table_idx);
		if (it == connecting_to_fk_map.end()) {
			continue;
		}
		idx_t fk_table_idx = it->second;

		// Find the FK table LogicalGet
		vector<unique_ptr<LogicalOperator> *> fk_nodes;
		FindAllNodesByTableIndex(&plan, fk_table_idx, fk_nodes);
		if (fk_nodes.empty()) {
			continue;
		}

		auto &fk_get = fk_nodes[0]->get()->Cast<LogicalGet>();

		// Get FK columns
		auto fk_result = GetFKColumnsToPU(fk_get.GetTable()->name, privacy_units, check);
		if (!fk_result.found) {
			continue;
		}

		auto hash_result = BuildHashForAggregateFromFKTable(input, plan, target_agg, fk_get, fk_result.fk_columns);
		if (hash_result.success) {
			ModifyAggregatesWithPacFunctions(input, target_agg, hash_result.hash_expr);
			return true;
		}
	}

	return false;
}

void ProcessAllAggregatesWithoutPU(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                                   const vector<LogicalAggregate *> &target_aggregates,
                                   const std::unordered_map<idx_t, idx_t> &connecting_to_fk_map,
                                   const vector<string> &gets_present, const vector<string> &privacy_units,
                                   const vector<string> &fk_path, const PACCompatibilityResult &check) {
	for (auto *target_agg : target_aggregates) {
		ProcessAggregateForPAC(input, plan, target_agg, connecting_to_fk_map, gets_present, privacy_units, fk_path,
		                       check);
	}
}

void ProcessAllAggregatesWithPU(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
                                const vector<LogicalAggregate *> &target_aggregates,
                                const vector<string> &pu_table_names, const PACCompatibilityResult &check) {
	for (auto *target_agg : target_aggregates) {
		vector<unique_ptr<Expression>> hash_exprs;

		for (auto &pu_table_name : pu_table_names) {
			// Check if PU table is in subtree and accessible
			bool pu_in_subtree = HasTableInSubtree(target_agg, pu_table_name);
			if (!pu_in_subtree) {
				continue;
			}

			LogicalGet *get_ptr = FindTableScanInSubtree(target_agg, pu_table_name);
			if (!get_ptr) {
				continue;
			}

			if (!AreTableColumnsAccessible(target_agg, get_ptr->table_index)) {
				continue;
			}

			// Determine if we should use rowid or PKs
			bool use_rowid = false;
			vector<string> pks;

			auto it = check.table_metadata.find(pu_table_name);
			if (it != check.table_metadata.end() && !it->second.pks.empty()) {
				pks = it->second.pks;
			} else {
				use_rowid = true;
			}

			auto result = BuildHashForAggregateFromPU(input, plan, target_agg, *get_ptr, pks, use_rowid);
			if (result.success) {
				hash_exprs.push_back(std::move(result.hash_expr));
			}
		}

		if (!hash_exprs.empty()) {
			auto combined_hash_expr = BuildAndFromHashes(input, hash_exprs);
			ModifyAggregatesWithPacFunctions(input, target_agg, combined_hash_expr);
		}
	}
}

} // namespace duckdb
