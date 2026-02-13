//
// PAC FK Resolver
//
// This file provides utilities for resolving foreign key relationships in PAC query compilation.
// It handles finding which tables have FK references to privacy units, mapping connecting tables
// to their FK tables, and finding accessible FK tables when blocked by SEMI/ANTI joins.
//
// Created by refactoring pac_bitslice_compiler.cpp
//

#ifndef PAC_FK_RESOLVER_HPP
#define PAC_FK_RESOLVER_HPP

#include "duckdb.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "metadata/pac_compatibility_check.hpp"
namespace duckdb {

// Get the FK columns from a table that reference any of the privacy units
// Returns the FK columns and the referenced PU table name
struct FKToPUResult {
	vector<string> fk_columns;
	string pu_table;
	bool found;

	FKToPUResult() : found(false) {
	}
	FKToPUResult(vector<string> cols, string pu) : fk_columns(std::move(cols)), pu_table(std::move(pu)), found(true) {
	}
};

FKToPUResult GetFKColumnsToPU(const string &table_name, const vector<string> &privacy_units,
                              const PACCompatibilityResult &check);

// Find an accessible FK table in the plan that can be used for hashing
// This is used when the primary FK table is blocked by SEMI/ANTI joins
struct AccessibleFKTableResult {
	LogicalGet *get_node;
	idx_t table_index;
	string table_name;
	vector<string> fk_columns;
	bool found;

	AccessibleFKTableResult() : get_node(nullptr), table_index(DConstants::INVALID_INDEX), found(false) {
	}
};

AccessibleFKTableResult FindAccessibleFKTable(unique_ptr<LogicalOperator> &plan, LogicalOperator *search_root,
                                              const vector<string> &candidate_tables,
                                              const vector<string> &privacy_units, const PACCompatibilityResult &check);

} // namespace duckdb

#endif // PAC_FK_RESOLVER_HPP
