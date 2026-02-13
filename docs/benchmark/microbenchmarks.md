# PAC Microbenchmark Suite

Comprehensive performance benchmarks for PAC aggregate functions (pac_count, pac_sum, pac_avg, pac_min, pac_max) with various optimization configurations. The extension should be compiled with clang.

## Quick Start
Install the required dependencies (assuming a Debian-based system):
```bash
sudo apt install make ninja-build cmake clang
export CC=clang
export CXX=clang++
```
Then, run the scripts (from the `microbenchmark` directory):
```bash
./create_test_db.sh

# Build all variants and run all benchmarks
./run_all.sh

# Quick test (smaller data, single run)
./run_all.sh -q

# Build only
./run_all.sh -b

# Run specific benchmark only
./run_all.sh -r count
./run_all.sh -r min_max

# Analyze results
./analyze_results.py
```

## Build Variants

The suite builds multiple DuckDB binaries with different optimization flags:

| Variant | Flags | Description |
|---------|-------|-------------|
| `default` | (none) | All optimizations enabled |
| `count_nonbanked` | `-DPAC_COUNT_NONBANKED` | Disable banked counting |
| `sumavg_noncascading` | `-DPAC_SUMAVG_NONCASCADING` | Disable sum/avg cascading |
| `sumavg_nonlazy` | `-DPAC_SUMAVG_NONLAZY` | Pre-allocate all levels |
| `minmax_nonbanked` | `-DPAC_MINMAX_NONBANKED` | Disable banked min/max |
| `minmax_noboundopt` | `-DPAC_MINMAX_NOBOUNDOPT` | Disable bound optimization |
| `minmax_nonlazy` | `-DPAC_MINMAX_NONLAZY` | Pre-allocate all levels |

## Benchmarks

### pac_count (`bench_count.sh`)

Tests counter overflow handling and banked vs non-banked performance:

- **Ungrouped**: 10M, 100M, 500M, 1B+ rows (tests 16→32→64 bit counter upgrade)
- **Grouped**: 1 to 10M groups (tests per-group state overhead)

### pac_sum/avg (`bench_sum_avg.sh`)

Tests cascading accumulator performance:

- **Data types**: int8, int16, int32, int64, float, double
- **Distributions**: small (0-100), medium (0-10K), large (0-1M)
- **Grouped vs ungrouped**
- **Scaling**: 10M to 500M rows

### pac_min/max (`bench_min_max.sh`)

Tests bound optimization and bank allocation:

- **Data types**: int8-64, uint8-64, float, double
- **Distributions**: random, monotonic increasing, monotonic decreasing
- **Bound optimization impact**: worst case (random) vs best case (monotonic)
- **Grouped vs ungrouped**

## Data Generation

All data is generated using DuckDB's `range()` function:

```sql
-- Random values via hash
(hash(i) % domain)::TYPE

-- Monotonic increasing
(i % domain)::TYPE

-- Monotonic decreasing
((domain - i % domain))::TYPE
```

## Results

Results are saved to `results/` as CSV files with timestamps:

```
results/
├── count_YYYYMMDD_HHMMSS.csv
├── sum_avg_YYYYMMDD_HHMMSS.csv
└── min_max_YYYYMMDD_HHMMSS.csv
```

CSV columns:
- `test`: test category (ungrouped, grouped, scaling, etc.)
- `aggregate`: function (count, sum, avg, min, max)
- `variant`: build variant (default, nonbanked, etc.)
- `data_size_m`: data size in millions
- `groups`: number of groups
- `dtype`: data type
- `distribution`: value distribution
- `time_sec`: median time in seconds
- `times`: all run times

## Environment Variables

```bash
WARMUP_RUNS=1    # Warmup iterations (default: 1)
BENCH_RUNS=3     # Benchmark iterations (default: 3)
FORCE_REBUILD=1  # Force rebuild binaries
```
