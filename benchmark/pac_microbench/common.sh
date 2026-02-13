#!/bin/bash
# Common functions for PAC microbenchmarks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARIES_DIR="$SCRIPT_DIR/binaries"
RESULTS_DIR="$SCRIPT_DIR/results"
# Default binary (with all optimizations enabled)
DEFAULT_BINARY="$BINARIES_DIR/duckdb_default"
TEST_DB="$BINARIES_DIR/test_data.duckdb"

# Default parameters
WARMUP_RUNS=${WARMUP_RUNS:-1}
BENCH_RUNS=${BENCH_RUNS:-3}
THREADS=${THREADS:-1}  # Default to single-threaded to avoid partitioning issues

# Query setup commands
QUERY_SETUP="SET threads=4; SET max_memory='30G'; SET max_temp_directory_size='90GB';"

mkdir -p "$RESULTS_DIR"

# Run a benchmark query and extract timing
# Args: binary, query, [runs]
# Returns: wall_median,agg_median,wall_times,agg_times
run_bench() {
    local binary=$1
    local query=$2
    local runs=${3:-$BENCH_RUNS}

    if [[ ! -f "$binary" ]]; then
        echo "ERROR: Binary not found: $binary" >&2
        return 1
    fi

    # Warmup
    for ((i=0; i<WARMUP_RUNS; i++)); do
        "$binary" -c "$QUERY_SETUP $query" >/dev/null 2>&1
    done

    # Benchmark runs with profiling
    local wall_times=()
    local agg_times=()
    local profile_file=$(mktemp)
    local query_failed=0
    local timeout_threshold=25.0

    for ((i=0; i<runs; i++)); do
        local start=$(python3 -c 'import time; print(time.time())')
        if ! "$binary" -c "
            $QUERY_SETUP
            PRAGMA enable_profiling='json';
            PRAGMA profiling_output='$profile_file';
            $query
        " >/dev/null 2>&1; then
            query_failed=1
            break
        fi
        local end=$(python3 -c 'import time; print(time.time())')
        local elapsed=$(python3 -c "print(f'{$end - $start:.3f}')")
        wall_times+=("$elapsed")

        # Extract aggregate operator time from JSON profile
        local agg_time=$(python3 -c "
import json
import sys
try:
    with open('$profile_file') as f:
        data = json.load(f)

    def find_agg_time(node):
        total = 0.0
        op_type = node.get('operator_type', '')
        # Match aggregate operators
        if 'UNGROUPED' in op_type or 'GROUP_BY' in op_type or 'AGGREGATE' in op_type:
            total += node.get('operator_timing', 0.0)
        for child in node.get('children', []):
            total += find_agg_time(child)
        return total

    agg_sec = find_agg_time(data)
    print(f'{agg_sec:.3f}')
except:
    print('0.000')
" 2>/dev/null)
        agg_times+=("$agg_time")

        # Stop repeating if query took longer than threshold
        if (( $(echo "$elapsed > $timeout_threshold" | bc -l) )); then
            break
        fi
    done

    rm -f "$profile_file"

    # If any benchmark run failed, return error indicator
    if [[ $query_failed -eq 1 ]]; then
        echo "-1,-1,-1,-1"
        return 0
    fi

    # Calculate medians
    local wall_median=$(printf '%s\n' "${wall_times[@]}" | sort -n | awk 'NR==int((NR+1)/2)')
    local agg_median=$(printf '%s\n' "${agg_times[@]}" | sort -n | awk 'NR==int((NR+1)/2)')

    echo "$wall_median,$agg_median,${wall_times[*]},${agg_times[*]}"
}

# Run DuckDB with timing pragma
# Args: binary, query
run_timed() {
    local binary=$1
    local query=$2

    "$binary" -c "
        .timer on
        $query
    " 2>&1
}

# Generate data sizes for benchmarks (in millions)
# Small: 10M, Medium: 100M, Large: 500M, XL: 1B (1000M)
DATA_SIZES_M=(10 50 100 500 1000)

# Group counts for grouped aggregation benchmarks
GROUP_COUNTS=(1 10 100 1000 10000 100000 1000000)

# Get binary for a variant
get_binary() {
    local variant=$1
    echo "$BINARIES_DIR/duckdb_$variant"
}

# Check if binary exists
check_binary() {
    local variant=$1
    local binary=$(get_binary "$variant")
    if [[ ! -f "$binary" ]]; then
        echo "Binary not found: $binary" >&2
        echo "Run: ./build_variants.sh $variant" >&2
        return 1
    fi
}

# Format number with commas
fmt_num() {
    printf "%'d" "$1"
}

# Print benchmark header
print_header() {
    local test_name=$1
    echo ""
    echo "========================================"
    echo "$test_name"
    echo "========================================"
    echo "Date: $(date)"
    echo "Host: $(hostname)"
    echo "========================================"
}

# CSV output helpers
csv_header() {
    echo "test,variant,data_size,groups,dtype,distribution,time_sec,all_times"
}

append_result() {
    local file=$1
    shift
    echo "$@" >> "$file"
}

# Check if test database exists
check_test_db() {
    if [[ ! -f "$TEST_DB" ]]; then
        echo "Test database not found: $TEST_DB" >&2
        echo "Run: ./create_test_db.sh" >&2
        return 1
    fi
}

# Run a benchmark query against the test database
# Args: binary, query, [runs]
# Returns: wall_median,agg_median,wall_times,agg_times
# Returns -1,-1,-1,-1 if query fails (e.g., out of memory)
run_bench_db() {
    local binary=$1
    local query=$2
    local runs=${3:-$BENCH_RUNS}

    if [[ ! -f "$binary" ]]; then
        echo "ERROR: Binary not found: $binary" >&2
        return 1
    fi

    if [[ ! -f "$TEST_DB" ]]; then
        echo "ERROR: Test database not found: $TEST_DB" >&2
        echo "Run: ./create_test_db.sh" >&2
        return 1
    fi

    # Warmup - check if query succeeds
    local warmup_failed=0
    for ((i=0; i<WARMUP_RUNS; i++)); do
        if ! "$binary" "$TEST_DB" -c "$QUERY_SETUP $query" >/dev/null 2>&1; then
            warmup_failed=1
            break
        fi
    done

    # If warmup failed, return error indicator
    if [[ $warmup_failed -eq 1 ]]; then
        echo "-1,-1,-1,-1"
        return 0
    fi

    # Benchmark runs with profiling
    local wall_times=()
    local agg_times=()
    local profile_file=$(mktemp)
    local query_failed=0
    local timeout_threshold=25.0

    for ((i=0; i<runs; i++)); do
        local start=$(python3 -c 'import time; print(time.time())')
        if ! "$binary" "$TEST_DB" -c "
            $QUERY_SETUP
            PRAGMA enable_profiling='json';
            PRAGMA profiling_output='$profile_file';
            $query
        " >/dev/null 2>&1; then
            query_failed=1
            break
        fi
        local end=$(python3 -c 'import time; print(time.time())')
        local elapsed=$(python3 -c "print(f'{$end - $start:.3f}')")
        wall_times+=("$elapsed")

        # Extract aggregate operator time from JSON profile
        local agg_time=$(python3 -c "
import json
import sys
try:
    with open('$profile_file') as f:
        data = json.load(f)

    def find_agg_time(node):
        total = 0.0
        op_type = node.get('operator_type', '')
        # Match aggregate operators
        if 'UNGROUPED' in op_type or 'GROUP_BY' in op_type or 'AGGREGATE' in op_type:
            total += node.get('operator_timing', 0.0)
        for child in node.get('children', []):
            total += find_agg_time(child)
        return total

    agg_sec = find_agg_time(data)
    print(f'{agg_sec:.3f}')
except:
    print('0.000')
" 2>/dev/null)
        agg_times+=("$agg_time")

        # Stop repeating if query took longer than threshold
        if (( $(echo "$elapsed > $timeout_threshold" | bc -l) )); then
            break
        fi
    done

    rm -f "$profile_file"

    # If any benchmark run failed, return error indicator
    if [[ $query_failed -eq 1 ]]; then
        echo "-1,-1,-1,-1"
        return 0
    fi

    # Calculate medians
    local wall_median=$(printf '%s\n' "${wall_times[@]}" | sort -n | awk 'NR==int((NR+1)/2)')
    local agg_median=$(printf '%s\n' "${agg_times[@]}" | sort -n | awk 'NR==int((NR+1)/2)')

    echo "$wall_median,$agg_median,${wall_times[*]},${agg_times[*]}"
}
