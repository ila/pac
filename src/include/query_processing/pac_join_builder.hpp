//
// PAC Join Builder
//
// This file provides utilities for building join chains in PAC query compilation.
// It handles creating LogicalGet nodes for tables and chaining them together
// with appropriate join conditions based on foreign key relationships.
//
// Created by refactoring pac_bitslice_compiler.cpp
//

#ifndef PAC_JOIN_BUILDER_HPP
#define PAC_JOIN_BUILDER_HPP

#include "duckdb.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

#include "metadata/pac_compatibility_check.hpp"

namespace duckdb {

// Result of building a join chain
struct JoinChainResult {
	// The final joined operator (or nullptr if failed)
	unique_ptr<LogicalOperator> join_chain;
	// Table index of the FK table (table with FK to PU) in the chain
	idx_t fk_table_index;
	// Whether the build was successful
	bool success;

	JoinChainResult() : fk_table_index(DConstants::INVALID_INDEX), success(false) {
	}

	JoinChainResult(unique_ptr<LogicalOperator> chain, idx_t fk_idx)
	    : join_chain(std::move(chain)), fk_table_index(fk_idx), success(true) {
	}
};

// Chain joins from a pre-built map of LogicalGet nodes
// Takes ownership of gets from the map and returns the final joined operator
unique_ptr<LogicalOperator> ChainJoinsFromGetMap(const PACCompatibilityResult &check, ClientContext &context,
                                                 unique_ptr<LogicalOperator> left_node,
                                                 std::unordered_map<string, unique_ptr<LogicalGet>> &get_map,
                                                 const vector<string> &tables_to_join);

// Build a join chain from an existing node through a list of tables
// Returns the joined operator and the table index of the FK table
JoinChainResult BuildJoinChain(const PACCompatibilityResult &check, ClientContext &context,
                               unique_ptr<LogicalOperator> &plan, unique_ptr<LogicalOperator> existing_node,
                               const vector<string> &tables_to_join, const string &fk_table_with_pu_reference,
                               idx_t &next_table_index);

// Create LogicalGet nodes for a list of tables
// Returns a map from table name to LogicalGet
std::unordered_map<string, unique_ptr<LogicalGet>>
CreateLogicalGetsForTables(ClientContext &context, unique_ptr<LogicalOperator> &plan,
                           const PACCompatibilityResult &check, const vector<string> &tables, idx_t &next_table_index);

// Determine which tables need to be joined based on FK path and what's already present
// Returns the ordered list of tables to join
vector<string> DetermineTablesForJoinChain(const vector<string> &fk_path,
                                           const std::unordered_set<string> &missing_tables,
                                           const std::unordered_set<string> &present_tables, bool join_elimination,
                                           const vector<string> &privacy_units);

// Find the "connecting table" - the table that serves as the starting point for joins
// Returns empty string if no connecting table found
string FindConnectingTable(const vector<string> &fk_path, const vector<string> &gets_present,
                           const vector<string> &scanned_non_pu_tables, bool for_missing_tables);

} // namespace duckdb

#endif // PAC_JOIN_BUILDER_HPP
