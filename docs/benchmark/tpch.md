# TPC-H Benchmark

The TPC-H benchmark compares PAC query performance against baseline DuckDB TPC-H queries.

## Overview

This benchmark runs TPC-H queries Q1-Q22 (excluding Q2, Q10, Q11, Q16, Q18) and measures:
- **Baseline**: Standard DuckDB TPC-H query execution time
- **PAC (bitslice)**: Optimized PAC queries using bitslice compilation
- **PAC (naive)**: Naive PAC query variants (optional)
- **PAC (simple hash)**: Simple hash PAC query variants (optional)


## PAC Schema Configuration

The benchmark uses `pac_tpch_schema.sql` to configure the TPC-H schema for PAC:

```sql
-- Mark customer as the privacy unit
ALTER TABLE customer SET PAC;

-- Protected columns in customer table
ALTER PAC TABLE customer ADD PROTECTED (c_custkey, c_comment, c_acctbal, c_name, c_address);

-- Orders -> Customer link (foreign key chain)
ALTER PAC TABLE orders ADD PAC_LINK (o_custkey) REFERENCES customer(c_custkey);

-- Lineitem -> Orders link
ALTER PAC TABLE lineitem ADD PAC_LINK (l_orderkey) REFERENCES orders(o_orderkey);
```

This creates a privacy unit chain: `lineitem → orders → customer`. PAC queries on `lineitem` must join through `orders` to get the customer's privacy key hash.

## Manual PAC Queries

The `benchmark/tpch_pac_queries/` directory contains **hand-written PAC query variants** for each TPC-H query. These serve as:

1. **Ground truth** for compiler correctness testing
2. **Performance baseline** for compiler optimization comparison
3. **Examples** of how to write PAC queries manually

### Query Structure

Manual PAC queries replace standard aggregates with PAC versions and explicitly include the privacy key hash. For example:

**Q1 (Pricing Summary Report)**:
```sql
SELECT l_returnflag, l_linestatus,
       pac_sum(hash(orders.o_custkey), l_quantity) AS sum_qty,
       pac_sum(hash(orders.o_custkey), l_extendedprice) AS sum_base_price,
       pac_avg(hash(orders.o_custkey), l_quantity) AS avg_qty,
       pac_count(hash(orders.o_custkey), 1) AS count_order
FROM lineitem JOIN orders ON lineitem.l_orderkey = orders.o_orderkey
WHERE l_shipdate <= DATE '1998-09-02'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;
```

Note how the query:
- Joins `lineitem` to `orders` to access `o_custkey` (the privacy unit FK)
- Uses `hash(orders.o_custkey)` as the first argument to each PAC aggregate
- Replaces `SUM` → `pac_sum`, `AVG` → `pac_avg`, `COUNT` → `pac_count`

**Q3 (Shipping Priority)**:
```sql
SELECT l_orderkey, 
       pac_sum(hash(customer.c_custkey), l_extendedprice * (1 - l_discount)) AS revenue, 
       o_orderdate, o_shippriority
FROM customer JOIN orders ON c_custkey = o_custkey 
              JOIN lineitem ON l_orderkey = o_orderkey
WHERE c_mktsegment = 'BUILDING' AND o_orderdate < DATE '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY revenue DESC, o_orderdate
LIMIT 10;
```

### Query Variants

| Directory | Description |
|-----------|-------------|
| `tpch_pac_queries/` | Optimized bitslice PAC queries (production algorithm) |
| `tpch_pac_naive_queries/` | Naive PAC implementation (simpler but slower) |
| `tpch_pac_simple_hash_queries/` | Simple hash-based PAC (alternative algorithm) |

## Building

The benchmark is a **standalone executable**, not run through DuckDB extension loading:

```bash
# From repository root
cmake --build build/release --target pac_tpch_benchmark
```

## Running

```bash
# Basic usage (SF=10 by default)
./build/release/pac_tpch_benchmark

# Specify scale factor
./build/release/pac_tpch_benchmark --sf 1
./build/release/pac_tpch_benchmark --sf 0.1
./build/release/pac_tpch_benchmark --sf 10

# Specify database path
./build/release/pac_tpch_benchmark --db tpch_sf10.db --sf 10

# Include naive PAC variant
./build/release/pac_tpch_benchmark --sf 1 --naive

# Include simple hash variant
./build/release/pac_tpch_benchmark --sf 1 --simple-hash

# Custom output CSV
./build/release/pac_tpch_benchmark --sf 1 --out results.csv
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--sf <factor>` | TPC-H scale factor (can be fractional) | `10.0` |
| `--db <path>` | Database file path | `tpch_sf{SF}.db` |
| `--queries <dir>` | Directory containing PAC query files | `benchmark` |
| `--out <path>` | Output CSV file path | `benchmark/tpch_benchmark_results_sf{SF}.csv` |
| `--naive` | Include naive PAC query variants | disabled |
| `--simple-hash` | Include simple hash PAC query variants | disabled |

## Database Creation

If the database file doesn't exist, the benchmark will:
1. Install and load the TPC-H extension
2. Generate TPC-H data using `CALL dbgen(sf=<scale_factor>)`
3. Load PAC schema from `pac_tpch_schema.sql`

If the database already exists, data generation is skipped.

## Output

### CSV Format

Results are written to a CSV file with columns:
- `query`: Query number (1-22)
- `mode`: `baseline`, `pac_bitslice`, `pac_naive`, or `pac_simple_hash`
- `run`: Run number (1-3)
- `time_ms`: Execution time in milliseconds

### Plotting

If R and the required packages are installed, the benchmark automatically generates a plot:
- `benchmark/tpch_benchmark_plot_sf{SF}.png`

To manually generate plots:
```bash
Rscript --vanilla benchmark/plot_tpch_results.R benchmark/tpch_benchmark_results_sf10.csv benchmark/
```

## Benchmark Methodology

For each query:
1. **Cold run**: Execute once (not timed) to load data into memory
2. **Warm-up run**: Execute once (not timed) to warm caches
3. **Timed runs**: Execute 3 times, recording each execution time

This is done for baseline and each enabled PAC variant.
