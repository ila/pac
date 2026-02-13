//
// Created by ila on 02/12/26.
//

#ifndef PAC_CLICKHOUSE_BENCHMARK_HPP
#define PAC_CLICKHOUSE_BENCHMARK_HPP

#include <string>
#include <vector>
#include "duckdb.hpp"

namespace duckdb {

// Run the ClickHouse Hits dataset benchmark (ClickBench).
// This benchmark:
// 1. Downloads the hits.csv.gz dataset from ClickHouse if not present
// 2. Extracts it
// 3. Creates and loads the hits table
// 4. Runs all queries (cold run + 3 warm runs)
// 5. Adds hits as a PAC table with UserID and ClientIP as protected columns
// 6. Runs all queries 3 more times under PAC protection
//
// Parameters:
// - db_path: path to DuckDB database file (default "clickbench.db")
// - queries_dir: directory containing create.sql, load.sql, and queries.sql (default "benchmark/clickbench_queries")
// - out_csv: output CSV path (if empty, auto-named)
// - micro: if true, use a smaller subset for quick testing
//
// Returns 0 on success, non-zero on error.
int RunClickHouseBenchmark(const string &db_path = "clickbench.db",
                           const string &queries_dir = "benchmark/clickbench_queries",
                           const string &out_csv = "",
                           bool micro = false);

} // namespace duckdb

#endif // PAC_CLICKHOUSE_BENCHMARK_HPP

