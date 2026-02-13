#ifndef DUCKDB_OPENPAC_REWRITE_RULE_HPP
#define DUCKDB_OPENPAC_REWRITE_RULE_HPP

#include "duckdb.hpp"
#include <atomic>

namespace duckdb {

// PAC-specific optimizer info used to prevent re-entrant replanning from the extension
struct PACOptimizerInfo : public OptimizerExtensionInfo {
	std::atomic<bool> replan_in_progress {false};

	PACOptimizerInfo() = default;
	~PACOptimizerInfo() override = default;
};

class PACRewriteRule : public OptimizerExtension {
public:
	PACRewriteRule() {
		// PAC rewrites run in the pre-optimizer phase, BEFORE DuckDB's built-in optimizers.
		// This way DuckDB's join ordering, filter pushdown, column lifetime, compressed
		// materialization etc. all run on the PAC-transformed plan automatically.
		pre_optimize_function = PACPreOptimizeFunction;
	}

	// Pre-optimizer: performs PAC rewriting before built-in optimizers run
	static void PACPreOptimizeFunction(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan);
};

// Separate optimizer rule to handle DROP TABLE operations and clean up PAC metadata
class PACDropTableRule : public OptimizerExtension {
public:
	PACDropTableRule() {
		optimize_function = PACDropTableRuleFunction;
	}

	static void PACDropTableRuleFunction(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan);
};

} // namespace duckdb

#endif // DUCKDB_OPENPAC_REWRITE_RULE_HPP
