//
// Created by ila on 1/23/26.
//

#ifndef PAC_AVG_HPP
#define PAC_AVG_HPP

#include "duckdb.hpp"
#include "pac_sum.hpp"

namespace duckdb {

// Register pac_avg functions (uses pac_sum infrastructure with DIVIDE_BY_COUNT=true)
void RegisterPacAvgFunctions(ExtensionLoader &loader);
void RegisterPacAvgCountersFunctions(ExtensionLoader &loader);
} // namespace duckdb

#endif // PAC_AVG_HPP
