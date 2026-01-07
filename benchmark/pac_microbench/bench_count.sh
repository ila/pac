#!/bin/bash
# PAC COUNT Benchmark
# Uses data1/data10/data100 views (10M/100M/1B rows).
#
# Tests:
# 1. Ungrouped count with varying scale factors
# 2. Grouped count with sequential groups (grp_X)
# 3. Grouped count with scattered groups (prg_X) - stresses memory
#
# Variants:
#   - standard: DuckDB's native COUNT (baseline)
#   - default: PAC COUNT with all optimizations (buffering, cascading)
#   - nobuffering: PAC COUNT without buffering (lazy alloc)
#   - nocascading: PAC COUNT directly into uint64 totals
#   - nosimd: PAC COUNT nocascading with simd-unfriendly update kernel

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

RESULTS_FILE="$RESULTS_DIR/count_$(date +%Y%m%d_%H%M%S).csv"

# Variants for ungrouped tests
UNGROUPED_VARIANTS=(standard default nobuffering nocascading nosimd)

# Variants for grouped tests 
GROUPED_VARIANTS=(standard default nobuffering nocascading)

# Get binary for variant
get_binary() {
    local variant=$1
    if [[ "$variant" == "standard" ]]; then
        echo "$BINARIES_DIR/duckdb_default"
    else
        echo "$BINARIES_DIR/duckdb_$variant"
    fi
}

print_header "PAC COUNT Benchmark"
echo "Results: $RESULTS_FILE"
echo ""

# Check test database exists
check_test_db || exit 1

# Check binaries exist
for variant in "${GROUPED_VARIANTS[@]}"; do
    if [[ "$variant" == "standard" ]]; then
        continue
    fi
    binary=$(get_binary "$variant")
    if [[ ! -f "$binary" ]]; then
        echo "Binary not found: $binary" >&2
        echo "Run: ./build_variants.sh $variant" >&2
        exit 1
    fi
done

# CSV header
echo "test,variant,rows_m,groups,wall_sec,agg_sec,wall_times,agg_times" > "$RESULTS_FILE"

# Data views and their sizes
DATA_VIEWS=(data1 data10 data100)
DATA_SIZES_M=(10 100 1000)

# Sequential group columns and their cardinalities
SEQ_GROUP_COLS=(grp_10 grp_1000 grp_100000 grp_10000000)
# Scattered group columns (same cardinalities)
SCAT_GROUP_COLS=(prg_10 prg_1000 prg_100000 prg_10000000)
GROUP_COUNTS=(10 1000 100000 10000000)

echo ""
echo "=== Ungrouped COUNT ==="
echo "Tests counter cascading from 16-bit to 32-bit to 64-bit"
echo ""

for idx in "${!DATA_VIEWS[@]}"; do
    view="${DATA_VIEWS[$idx]}"
    rows_m="${DATA_SIZES_M[$idx]}"
    echo "--- $view (${rows_m}M rows) ---"

    for variant in "${UNGROUPED_VARIANTS[@]}"; do
        binary=$(get_binary "$variant")

        if [[ "$variant" == "standard" ]]; then
            query="SELECT COUNT(*) FROM $view;"
        else
            query="SELECT pac_count(hash(i), 0.0) FROM $view;"
        fi

        result=$(run_bench_db "$binary" "$query")
        wall=$(echo "$result" | cut -d',' -f1)
        agg=$(echo "$result" | cut -d',' -f2)
        wall_times=$(echo "$result" | cut -d',' -f3)
        agg_times=$(echo "$result" | cut -d',' -f4)
        if [[ "$wall" == "-1" ]]; then
            echo "  $variant: FAILED"
        else
            echo "  $variant: wall=${wall}s agg=${agg}s"
        fi
        echo "ungrouped,$variant,$rows_m,1,$wall,$agg,$wall_times,$agg_times" >> "$RESULTS_FILE"
    done
    echo ""
done

echo ""
echo "=== Grouped COUNT with sequential groups (grp_X) ==="
echo "Sequential groups have consecutive rows in same group."
echo ""

for idx in "${!DATA_VIEWS[@]}"; do
    view="${DATA_VIEWS[$idx]}"
    rows_m="${DATA_SIZES_M[$idx]}"
    echo "--- $view (${rows_m}M rows) ---"

    for grp_idx in "${!SEQ_GROUP_COLS[@]}"; do
        grp_col="${SEQ_GROUP_COLS[$grp_idx]}"
        num_groups="${GROUP_COUNTS[$grp_idx]}"

        rows_per_group=$((rows_m * 1000000 / num_groups))
        echo "  Groups: $num_groups (~$rows_per_group rows/group, sequential)"

        for variant in "${GROUPED_VARIANTS[@]}"; do
            binary=$(get_binary "$variant")

            if [[ "$variant" == "standard" ]]; then
                query="SELECT $grp_col, COUNT(*) FROM $view GROUP BY $grp_col;"
            else
                query="SELECT $grp_col, pac_count(hash(i), 0.0) FROM $view GROUP BY $grp_col;"
            fi

            result=$(run_bench_db "$binary" "$query")
            wall=$(echo "$result" | cut -d',' -f1)
            agg=$(echo "$result" | cut -d',' -f2)
            wall_times=$(echo "$result" | cut -d',' -f3)
            agg_times=$(echo "$result" | cut -d',' -f4)
            if [[ "$wall" == "-1" ]]; then
                echo "    $variant: FAILED"
            else
                echo "    $variant: wall=${wall}s agg=${agg}s"
            fi
            echo "grouped_seq,$variant,$rows_m,$num_groups,$wall,$agg,$wall_times,$agg_times" >> "$RESULTS_FILE"
        done
    done
    echo ""
done

echo ""
echo "=== Grouped COUNT with scattered groups (prg_X) ==="
echo "Scattered groups cycle through all groups (stresses memory)."
echo ""

for idx in "${!DATA_VIEWS[@]}"; do
    view="${DATA_VIEWS[$idx]}"
    rows_m="${DATA_SIZES_M[$idx]}"
    echo "--- $view (${rows_m}M rows) ---"

    for grp_idx in "${!SCAT_GROUP_COLS[@]}"; do
        grp_col="${SCAT_GROUP_COLS[$grp_idx]}"
        num_groups="${GROUP_COUNTS[$grp_idx]}"

        rows_per_group=$((rows_m * 1000000 / num_groups))
        echo "  Groups: $num_groups (~$rows_per_group rows/group, scattered)"

        for variant in "${GROUPED_VARIANTS[@]}"; do
            binary=$(get_binary "$variant")

            if [[ "$variant" == "standard" ]]; then
                query="SELECT $grp_col, COUNT(*) FROM $view GROUP BY $grp_col;"
            else
                query="SELECT $grp_col, pac_count(hash(i), 0.0) FROM $view GROUP BY $grp_col;"
            fi

            result=$(run_bench_db "$binary" "$query")
            wall=$(echo "$result" | cut -d',' -f1)
            agg=$(echo "$result" | cut -d',' -f2)
            wall_times=$(echo "$result" | cut -d',' -f3)
            agg_times=$(echo "$result" | cut -d',' -f4)
            if [[ "$wall" == "-1" ]]; then
                echo "    $variant: FAILED"
            else
                echo "    $variant: wall=${wall}s agg=${agg}s"
            fi
            echo "grouped_scat,$variant,$rows_m,$num_groups,$wall,$agg,$wall_times,$agg_times" >> "$RESULTS_FILE"
        done
    done
    echo ""
done

echo ""
echo "========================================"
echo "COUNT Benchmark Complete"
echo "Results saved to: $RESULTS_FILE"
echo "========================================"
