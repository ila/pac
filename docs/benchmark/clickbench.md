# ClickBench Benchmark

This document describes the ClickBench benchmark for measuring PAC overhead on web analytics workloads.

## Overview

[ClickBench](https://github.com/ClickHouse/ClickBench) is a benchmark based on the ClickHouse "hits" dataset, containing web analytics data with ~100 million page views. It provides a realistic workload for testing aggregate query performance on a single wide table.

The PAC ClickBench benchmark compares:
- **Baseline**: Standard DuckDB query execution
- **PAC**: Privacy-preserving query execution with differential privacy guarantees

## Dataset

The benchmark uses the `hits` table from the ClickBench dataset:
- **Rows**: ~100 million page views (full) or a micro subset for quick testing
- **Columns**: 105 columns including timestamps, user IDs, URLs, search phrases, etc.
- **Privacy Unit**: `UserID` - each user's contribution is bounded

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `WatchID` | BIGINT | Unique page view identifier |
| `UserID` | BIGINT | User identifier (privacy unit) |
| `EventTime` | TIMESTAMP | Page view timestamp |
| `URL` | TEXT | Page URL |
| `SearchPhrase` | TEXT | Search query if from search engine |
| `RegionID` | INTEGER | Geographic region |
| `ResolutionWidth` | SMALLINT | Screen resolution width |

## Building

```bash
# Build the ClickBench benchmark executable
cmake --build build/release --target pac_clickhouse_benchmark
```

## Running

```bash
# Run with default settings (uses clickbench_micro.db)
./build/release/benchmark/pac_clickhouse_benchmark

# Run with custom database
./build/release/benchmark/pac_clickhouse_benchmark --db path/to/clickbench.db

# Run with specific number of runs
./build/release/benchmark/pac_clickhouse_benchmark --runs 5
```

## Query Coverage

The benchmark includes 43 queries from the official ClickBench suite, testing:

| Query Type | Count | Examples |
|------------|-------|----------|
| Simple aggregates | 6 | `COUNT(*)`, `SUM()`, `AVG()` |
| `COUNT(DISTINCT)` | 8 | Unique users, search phrases |
| `GROUP BY` | 20 | Regional stats, device breakdowns |
| Filtered aggregates | 15 | Time-range filters, string matching |
| `ORDER BY` + `LIMIT` | 10 | Top-N queries |

### Query Compatibility

Some queries may be rejected by PAC due to privacy constraints:
- **Protected column access**: Direct access to `UserID` without aggregation
- **Unsupported aggregates**: `MIN`/`MAX` on `VARCHAR` columns
- **Sample diversity**: Insufficient diversity in grouped results

## Output

### Console Output

```
[2026-02-12 15:59:12] === ClickBench Benchmark ===
[2026-02-12 15:59:12] Database: clickbench_micro.db
[2026-02-12 15:59:12] Queries: 43, Runs: 3
[2026-02-12 15:59:12] --- Baseline (per run average) ---
[2026-02-12 15:59:12]   Successful queries: 43
[2026-02-12 15:59:12]   Total time (successful): 2567.2 ms
[2026-02-12 15:59:12]   Avg time per successful query: 59.7 ms
[2026-02-12 15:59:12] --- PAC (per run average) ---
[2026-02-12 15:59:12]   Successful queries: 26
[2026-02-12 15:59:12]   Rejected (privacy violations): 13
[2026-02-12 15:59:12]   Total time (successful): 2881.4 ms
[2026-02-12 15:59:12]   Avg time per successful query: 110.8 ms
[2026-02-12 15:59:12] --- PAC Overhead: 85.7% ---
```

### CSV Output

Results are saved to `clickbench_micro_results.csv`:

```csv
query,mode,run,time_ms,success
Q0,baseline,1,12.34,true
Q0,PAC,1,18.56,true
Q1,baseline,1,8.12,true
Q1,PAC,1,0,false
...
```

## Plotting Results

```bash
# Generate plots from results
cd benchmark
Rscript plot_clickbench_results.R

# Or specify custom input/output
Rscript plot_clickbench_results.R path/to/results.csv output_dir/
```

This generates:
- `clickbench_benchmark_plot_micro.png` - Standard plot with title
- `clickbench_benchmark_plot_micro_paper.png` - Publication-ready plot

## Overhead Calculation

The PAC overhead is calculated as:

```
overhead = (pac_avg_time / baseline_avg_time - 1.0) * 100%
```

Where:
- `pac_avg_time` = total successful PAC query time / number of successful PAC queries
- `baseline_avg_time` = total successful baseline query time / number of successful baseline queries

This measures the **percentage increase** in query execution time when using PAC.

## Directory Structure

```
benchmark/
├── clickbench_queries/
│   ├── create.sql          # Table schema
│   ├── load.sql            # Data loading script
│   └── queries.sql         # 43 benchmark queries
├── include/
│   └── pac_clickhouse_benchmark.hpp
├── pac_clickhouse_benchmark.cpp
├── plot_clickbench_results.R
└── clickbench_micro_results.csv
```

## See Also

- [TPC-H Benchmark](tpch.md) - Decision support benchmark
- [Microbenchmarks](microbenchmarks.md) - Individual aggregate tests

