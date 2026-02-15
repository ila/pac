//
// Created by ila on 12/24/25.
//

#ifndef PAC_TPCH_BENCHMARK_HPP
#define PAC_TPCH_BENCHMARK_HPP

#include <string>
#include <vector>
#include "duckdb.hpp"

namespace duckdb {

// Run the TPCH vs PAC benchmark. Defaults match the requested behavior:
// - db_path: path to DuckDB database file (default "tpch.db")
// - queries_dir: directory containing PAC query SQL files (default "benchmark/tpch_pac_queries")
// - sf: TPCH scale factor (default 10). This can be fractional (e.g. 0.1).
// - out_csv: output CSV path (if empty, a default name benchmark/tpch_bench_results_sf{sf}.csv will be used)
// Returns 0 on success, non-zero on error.
int RunTPCHBenchmark(const string &db_path = "tpch.db",
                     const string &queries_dir = "benchmark/tpch_pac_queries",
                     double sf = 10.0,
                     const string &out_csv = "",
                     bool run_naive = false,
                     bool run_simple_hash = false,
                     int threads = 8);

} // namespace duckdb

#endif // PAC_TPCH_BENCHMARK_HPP
