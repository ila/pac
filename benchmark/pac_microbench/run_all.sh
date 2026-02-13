#!/bin/bash
# PAC Microbenchmark Suite - Main Runner
# Builds all variants and runs all benchmarks

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 [options] [benchmark...]"
    echo ""
    echo "Options:"
    echo "  -b, --build-only     Only build binaries, don't run benchmarks"
    echo "  -r, --run-only       Only run benchmarks (assume binaries exist)"
    echo "  -f, --force-rebuild  Force rebuild all binaries"
    echo "  -q, --quick          Run quick benchmarks (smaller data sizes)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Benchmarks (default: all main benchmarks):"
    echo "  count          Run pac_count benchmarks"
    echo "  sum_avg        Run pac_sum/avg benchmarks"
    echo "  min_max        Run pac_min/max benchmarks"
    echo ""
    echo "Examples:"
    echo "  $0                      # Build all and run all main benchmarks"
    echo "  $0 -b                   # Only build binaries"
    echo "  $0 -r count             # Only run count benchmark"
    echo "  $0 -q min_max           # Quick min/max benchmark"
    exit 0
}

# Parse arguments
BUILD=1
RUN=1
FORCE_REBUILD=0
QUICK=0
BENCHMARKS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--build-only)
            RUN=0
            shift
            ;;
        -r|--run-only)
            BUILD=0
            shift
            ;;
        -f|--force-rebuild)
            FORCE_REBUILD=1
            shift
            ;;
        -q|--quick)
            QUICK=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        count|sum_avg|min_max)
            BENCHMARKS+=("$1")
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Default to main benchmarks (not the focused experiments)
if [[ ${#BENCHMARKS[@]} -eq 0 ]]; then
    BENCHMARKS=(count sum_avg min_max)
fi

echo "========================================"
echo "PAC Microbenchmark Suite"
echo "========================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Script dir: $SCRIPT_DIR"
echo ""

# Build phase
if [[ $BUILD -eq 1 ]]; then
    echo "=== Building DuckDB variants ==="
    echo ""

    export FORCE_REBUILD
    "$SCRIPT_DIR/build_variants.sh"

    echo ""
fi

# Run phase
if [[ $RUN -eq 1 ]]; then
    echo "=== Running benchmarks ==="
    echo "Benchmarks to run: ${BENCHMARKS[*]}"
    echo ""

    # Export settings for quick mode
    if [[ $QUICK -eq 1 ]]; then
        export WARMUP_RUNS=0
        export BENCH_RUNS=1
        echo "Quick mode: WARMUP_RUNS=0, BENCH_RUNS=1"
        echo ""
    fi

    for bench in "${BENCHMARKS[@]}"; do
        echo "----------------------------------------"
        echo "Running: $bench"
        echo "----------------------------------------"

        case $bench in
            count)
                "$SCRIPT_DIR/bench_count.sh"
                ;;
            sum_avg)
                "$SCRIPT_DIR/bench_sum_avg.sh"
                ;;
            min_max)
                "$SCRIPT_DIR/bench_min_max.sh"
                ;;
        esac

        echo ""
    done
fi

echo "========================================"
echo "PAC Microbenchmark Suite Complete"
echo "========================================"
echo ""
echo "Results are in: $SCRIPT_DIR/results/"
ls -la "$SCRIPT_DIR/results/" 2>/dev/null || echo "(no results yet)"
