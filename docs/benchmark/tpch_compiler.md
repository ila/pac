# TPC-H Compiler Benchmark

The compiler benchmark compares automatically compiled PAC queries against manually written PAC queries, to identify regressions or discrepancies in the results.

## Overview

This benchmark tests the PAC query compiler by:
1. Running TPC-H queries through DuckDB's `PRAGMA tpch(N)` with PAC automatic compilation
2. Comparing results and performance against manually written PAC query variants

## Building

The benchmark is a **standalone executable**, not run through DuckDB extension loading:

```bash
# From repository root
cmake --build build/release --target pac_tpch_compiler_benchmark
```

## Running

```bash
# Basic usage
./build/release/pac_tpch_compiler_benchmark --sf 1

# Different scale factors
./build/release/pac_tpch_compiler_benchmark --sf 0.1
./build/release/pac_tpch_compiler_benchmark --sf 10
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--sf <factor>` | TPC-H scale factor | Required |

## What It Tests

The compiler benchmark validates that:
1. PAC's automatic query compilation produces correct results
2. Compiled queries match the output of manually written PAC queries
3. Performance characteristics of compiled vs manual queries

## Database Setup

The benchmark:
1. Opens/creates a database file `tpch_sf{SF}.db`
2. Installs and loads the TPC-H extension
3. Loads the PAC extension
4. Creates TPC-H data if the database doesn't exist
5. Loads PAC schema from `pac_tpch_schema.sql` (adds PAC_LINK and PROTECTED annotations)

## PAC Schema

The `pac_tpch_schema.sql` file defines:
- `PAC_KEY` for privacy unit tables
- `PAC_LINK` relationships between tables
- `PROTECTED` columns that require aggregate access

This allows the PAC compiler to automatically transform standard TPC-H queries into privacy-preserving versions.

## Query Comparison

For each TPC-H query, the benchmark:
1. Executes the auto-compiled version via `PRAGMA tpch(N)` with PAC enabled
2. Executes the manually written PAC query from `tpch_pac_queries/`
3. Compares row counts, column counts, and values
4. Reports timing for both versions

## Query Directories

| Directory | Description |
|-----------|-------------|
| `benchmark/tpch_pac_queries/` | Manually written PAC queries for comparison |
| `benchmark/tpch_pac_naive_queries/` | Naive PAC implementations |
| `benchmark/tpch_pac_simple_hash_queries/` | Simple hash PAC implementations |

