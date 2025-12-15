#ifndef DUCKDB_OPENPAC_REWRITE_RULE_HPP
#define DUCKDB_OPENPAC_REWRITE_RULE_HPP

#include "duckdb.hpp"

namespace duckdb {

class PACRewriteRule : public OptimizerExtension {
public:
	PACRewriteRule() {
		optimize_function = PACRewriteRuleFunction;
	}

	static unique_ptr<LogicalOperator> ModifyPlan(unique_ptr<LogicalOperator> &plan);

	static void PACRewriteRuleFunction(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan);

	// Checks if the query plan is PAC compatible according to the rules.
	static bool IsPACCompatible(LogicalOperator &plan);
};

} // namespace duckdb

#endif // DUCKDB_OPENPAC_REWRITE_RULE_HPP

