#!/bin/bash
# PAC SUM/AVG Benchmark
# Uses data1/data10/data100 views (10M/100M/1B rows).
#
# Tests:
# 1. Ungrouped SUM by value domain
# 2. Grouped SUM with sequential groups (grp_X) - consecutive rows in same group
# 3. Grouped SUM with scattered groups (prg_X) - rows cycle through groups (stresses buffering)
#
# Variants:
#   - standard: DuckDB's native SUM (baseline)
#   - default: PAC SUM with all optimizations (buffering, approx cascading)
#   - nobuffering: PAC SUM without input buffering (grouped tests only)
#   - exactsum: PAC SUM with exact cascading su (without the approximation optimization)
#   - nocascading: PAC SUM without cascading (direct to largest type -- implies exactsum)
#   - nosimd: PAC SUM nocascading with simd-unfriendly update kernel and auto-vectorization disabled

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

RESULTS_FILE="$RESULTS_DIR/sum_avg_$(date +%Y%m%d_%H%M%S).csv"

# Variants for ungrouped tests (eageralloc/nobuffering don't affect ungrouped)
UNGROUPED_VARIANTS=(standard default nobuffering exactsum nocascading nosimd)

# Variants for grouped tests (includes allocation/buffering variants)
GROUPED_VARIANTS=(standard default nobuffering exactsum nocascading)

# Get binary for variant
get_binary() {
    local variant=$1
    if [[ "$variant" == "standard" ]]; then
        # Use default binary for standard SQL
        echo "$BINARIES_DIR/duckdb_default"
    else
        echo "$BINARIES_DIR/duckdb_$variant"
    fi
}

print_header "PAC SUM/AVG Benchmark"
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
echo "test,aggregate,variant,rows_m,groups,dtype,wall_sec,agg_sec,wall_times,agg_times" > "$RESULTS_FILE"

# Data views and their sizes
DATA_VIEWS=(data1 data10 data100)
DATA_SIZES_M=(10 100 1000)

# Sequential group columns and their cardinalities
SEQ_GROUP_COLS=(grp_10 grp_1000 grp_100000 grp_10000000)
# Scattered group columns (same cardinalities)
SCAT_GROUP_COLS=(prg_10 prg_1000 prg_100000 prg_10000000)
GROUP_COUNTS=(10 1000 100000 10000000)

echo ""
echo "=== Ungrouped SUM by value domain (BIGINT columns) ==="
echo ""

DOMAIN_COLS=(tiny_64 small_64 medium_64 large_64)
DOMAIN_NAMES=(tiny small medium large)

for idx in "${!DATA_VIEWS[@]}"; do
    view="${DATA_VIEWS[$idx]}"
    rows_m="${DATA_SIZES_M[$idx]}"
    echo "--- $view (${rows_m}M rows) ---"

    for col_idx in "${!DOMAIN_COLS[@]}"; do
        col="${DOMAIN_COLS[$col_idx]}"
        domain="${DOMAIN_NAMES[$col_idx]}"

        echo "  Column: $col ($domain domain)"

        for variant in "${UNGROUPED_VARIANTS[@]}"; do
            binary=$(get_binary "$variant")

            if [[ "$variant" == "standard" ]]; then
                query="SELECT SUM($col) FROM $view;"
            else
                query="SELECT pac_sum(hash(i), $col, 0.0) FROM $view;"
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
            echo "ungrouped_domain,sum,$variant,$rows_m,1,$domain,$wall,$agg,$wall_times,$agg_times" >> "$RESULTS_FILE"
        done
    done
    echo ""
done

echo ""
echo "=== Grouped SUM with sequential groups (grp_X) ==="
echo ""

col="tiny_64"

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
                query="SELECT $grp_col, SUM($col) FROM $view GROUP BY $grp_col;"
            else
                query="SELECT $grp_col, pac_sum(hash(i), $col, 0.0) FROM $view GROUP BY $grp_col;"
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
            echo "grouped_seq,sum,$variant,$rows_m,$num_groups,large,$wall,$agg,$wall_times,$agg_times" >> "$RESULTS_FILE"
        done
    done
    echo ""
done

echo ""
echo "=== Grouped SUM with scattered groups (prg_X) ==="
echo "Using tiny_64 column. Scattered groups cycle through all groups (stresses buffering)."
echo "This is where NOBUFFERING should show significant slowdown due to memory leaks."
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
                query="SELECT $grp_col, SUM($col) FROM $view GROUP BY $grp_col;"
            else
                query="SELECT $grp_col, pac_sum(hash(i), $col, 0.0) FROM $view GROUP BY $grp_col;"
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
            echo "grouped_scat,sum,$variant,$rows_m,$num_groups,large,$wall,$agg,$wall_times,$agg_times" >> "$RESULTS_FILE"
        done
    done
    echo ""
done

echo ""
echo "========================================"
echo "SUM/AVG Benchmark Complete"
echo "Results saved to: $RESULTS_FILE"
echo "========================================"
