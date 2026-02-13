# PAC Benchmarks

This document provides an overview of the benchmark suite for the PAC extension.

## Why Benchmarks?

PAC provides differential privacy guarantees by transforming SQL aggregates into privacy-preserving versions. This transformation introduces computational overhead that we need to measure and optimize. The benchmarks help us answer key questions:

1. **PAC Overhead**: How much slower are PAC queries compared to standard DuckDB queries? The bitslice compilation algorithm adds per-row hashing and probabilistic counting overhead.

2. **Join Overhead**: PAC requires joining through the privacy unit chain (e.g., `lineitem → orders → customer`) to obtain the privacy key hash. This adds join costs even when the original query doesn't need those joins.

## Benchmark Executables

PAC benchmarks are compiled as **separate standalone executables** (not run through DuckDB's extension loading). They are built using CMake targets:

| Executable | CMake Target | Description |
|------------|--------------|-------------|
| `pac_tpch_benchmark` | `pac_tpch_benchmark` | TPC-H benchmark comparing PAC vs baseline |
| `pac_tpch_compiler_benchmark` | `pac_tpch_compiler_benchmark` | Compiler benchmark comparing auto-compiled vs manual PAC queries |
| `pac_clickhouse_benchmark` | `pac_clickhouse_benchmark` | ClickBench benchmark for web analytics workloads |
### Building Benchmark Executables

```bash
# Build TPC-H benchmark
cmake --build build/release --target pac_tpch_benchmark

# Build compiler benchmark
cmake --build build/release --target pac_tpch_compiler_benchmark
```

# Build ClickBench benchmark
cmake --build build/release --target pac_clickhouse_benchmark
| Benchmark | Documentation | Purpose |
|-----------|---------------|---------|
| TPC-H Benchmark | [tpch.md](tpch.md) | Measure PAC overhead vs baseline DuckDB |
| TPC-H Compiler Benchmark | [tpch_compiler.md](tpch_compiler.md) | Compare auto-compiled vs manual PAC queries |
| Microbenchmarks | [microbenchmarks.md](microbenchmarks.md) | Test individual PAC aggregate optimizations |

| ClickBench | [clickbench.md](clickbench.md) | Measure PAC overhead on web analytics workloads |

```
benchmark/
├── include/                        # Benchmark headers
│   ├── pac_tpch_benchmark.hpp
│   └── pac_tpch_compiler_benchmark.hpp
├── pac_microbench/                 # Microbenchmark suite
├── pac_tpch_benchmark.cpp          # TPC-H benchmark runner
├── pac_tpch_compiler_benchmark.cpp # Compiler benchmark runner
│   ├── pac_clickhouse_benchmark.hpp
├── plot_tpch_results.R             # R script for plotting results
├── tpch_pac_queries/               # Hand-written optimized PAC queries
├── clickbench_queries/             # ClickBench query suite
└── tpch_pac_simple_hash_queries/   # Simple hash PAC query variants
├── pac_clickhouse_benchmark.cpp    # ClickBench benchmark runner

## Output

├── plot_clickbench_results.R       # R script for ClickBench plots
├── plot_tpch_results.R             # R script for TPC-H plots
- PNG plots are generated automatically if R and required packages are installed
