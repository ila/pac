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
		optimize_function = PACRewriteRuleFunction;
	}

	static unique_ptr<LogicalOperator> ModifyPlan(unique_ptr<LogicalOperator> &plan);

	static void PACRewriteRuleFunction(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan);

	// Checks if the query plan is PAC compatible according to the rules.
	// Throws a ParserException with an explanatory message when the plan is not compatible.
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
