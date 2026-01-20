//
// Created by ila on 1/19/26.
//

#ifndef PAC_TPCH_COMPILER_BENCHMARK_HPP
#define PAC_TPCH_COMPILER_BENCHMARK_HPP

#include <string>

namespace duckdb {

// Run the TPC-H compiler benchmark
// - scale_factor: the TPC-H scale factor (e.g. 0.1, 1, 10, 100)
// - scale_factor_str: the original string representation for database naming
// Compares automatically compiled PAC queries (PRAGMA) against manually written PAC queries
void RunTPCHCompilerBenchmark(double scale_factor, const std::string &scale_factor_str);

} // namespace duckdb

#endif // PAC_TPCH_COMPILER_BENCHMARK_HPP
