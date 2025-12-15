//
// Created by ila on 12/12/25.
//

#include "include/pac_optimizer.hpp"
#include "include/pac_compatibility_check.hpp"

using namespace duckdb;

namespace duckdb {

bool PACRewriteRule::IsPACCompatible(LogicalOperator &plan) {
    // Delegate to the compatibility check implementation in pac_compatibility_check.cpp
    return CheckPACCompatibility(plan, "pac_tables.csv");
}

void PACRewriteRule::PACRewriteRuleFunction(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
    if (!plan) return;
    if (IsPACCompatible(*plan)) {
        // PAC compatible: do nothing for now
        return;
    } else {
        // Not PAC compatible: do nothing for now (future: could log or throw)
        return;
    }
}

} // namespace duckdb
