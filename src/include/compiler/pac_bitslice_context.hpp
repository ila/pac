//
// PAC Bitslice Context
//
// This file provides a context struct that bundles together all the commonly
// passed parameters during bitslice compilation. This reduces function parameter
// counts and makes the code more maintainable.
//
// Created by refactoring pac_bitslice_compiler.cpp
//

#ifndef PAC_BITSLICE_CONTEXT_HPP
#define PAC_BITSLICE_CONTEXT_HPP

#include "duckdb.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "metadata/pac_compatibility_check.hpp"

namespace duckdb {

// Context struct that holds all the state needed during bitslice compilation
// This reduces the number of parameters passed between functions
struct BitsliceCompilerContext {
	// Reference to the compatibility check result
	const PACCompatibilityResult &check;
	// Optimizer extension input (contains context and optimizer)
	OptimizerExtensionInput &input;
	// The plan being modified
	unique_ptr<LogicalOperator> &plan;
	// List of privacy unit table names
	const vector<string> &privacy_units;
	// FK path from scanned tables to PU
	const vector<string> &fk_path;
	// Tables present in the original query
	const vector<string> &gets_present;
	// Tables missing from the query that need to be joined
	const vector<string> &gets_missing;

	// Computed values (set during compilation)
	string fk_table_with_pu_reference;
	std::unordered_map<idx_t, idx_t> connecting_table_to_fk_table;
	bool join_elimination;

	BitsliceCompilerContext(const PACCompatibilityResult &check, OptimizerExtensionInput &input,
	                        unique_ptr<LogicalOperator> &plan, const vector<string> &privacy_units,
	                        const vector<string> &fk_path, const vector<string> &gets_present,
	                        const vector<string> &gets_missing)
	    : check(check), input(input), plan(plan), privacy_units(privacy_units), fk_path(fk_path),
	      gets_present(gets_present), gets_missing(gets_missing), join_elimination(false) {
	}

	// Convenience accessor for ClientContext
	ClientContext &GetContext() const {
		return input.context;
	}

	// Convenience accessor for Optimizer
	Optimizer &GetOptimizer() const {
		return input.optimizer;
	}
};

// Simplified context for ModifyPlanWithPU (fewer required fields)
struct BitsliceWithPUContext {
	OptimizerExtensionInput &input;
	unique_ptr<LogicalOperator> &plan;
	const vector<string> &pu_table_names;
	const PACCompatibilityResult &check;

	BitsliceWithPUContext(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan,
	                      const vector<string> &pu_table_names, const PACCompatibilityResult &check)
	    : input(input), plan(plan), pu_table_names(pu_table_names), check(check) {
	}

	ClientContext &GetContext() const {
		return input.context;
	}

	Optimizer &GetOptimizer() const {
		return input.optimizer;
	}
};

} // namespace duckdb

#endif // PAC_BITSLICE_CONTEXT_HPP
