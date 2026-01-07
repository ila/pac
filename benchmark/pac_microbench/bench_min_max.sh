#!/bin/bash
# PAC MIN/MAX Benchmark
# Uses data1/data10/data100 views (10M/100M/1B rows).
#
# Tests:
# 1. Ungrouped MAX by data type
# 2. Value distribution impact on bound optimization (random vs monotonic)
# 3. Grouped MAX with sequential groups (grp_X)
# 4. Grouped MAX with scattered groups (prg_X) - stresses buffering
# 5. Float/Double MAX performance
#
# Variants:
#   - standard: DuckDB's native MAX (baseline)
#   - default: PAC MAX with all optimizations (buffering, bound opt)
#   - noboundopt: PAC MAX without bound optimization
#   - nobuffering: PAC MAX without input buffering (grouped tests only)
#   - nosimd: PAC MAX with simd-unfriendly update kernel and auto-vectorization disabled

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

RESULTS_FILE="$RESULTS_DIR/min_max_$(date +%Y%m%d_%H%M%S).csv"

# Variants for ungrouped tests (includes noboundopt since it affects min/max)
UNGROUPED_VARIANTS=(standard default noboundopt nobuffering nosimd)

# Variants for grouped tests (includes allocation/buffering variants)
GROUPED_VARIANTS=(standard default noboundopt nobuffering)

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

print_header "PAC MIN/MAX Benchmark"
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

# Typed columns
TYPED_COLS=(col_8 col_16 col_32 col_64 col_128 col_flt col_dbl)
TYPED_NAMES=(int8 int16 int32 int64 hugeint float double)

echo ""
echo "=== Ungrouped MAX by data type ==="
echo ""

for idx in "${!DATA_VIEWS[@]}"; do
    view="${DATA_VIEWS[$idx]}"
    rows_m="${DATA_SIZES_M[$idx]}"
    echo "--- $view (${rows_m}M rows) ---"

    for col_idx in "${!TYPED_COLS[@]}"; do
        col="${TYPED_COLS[$col_idx]}"
        dtype="${TYPED_NAMES[$col_idx]}"

        echo "  Column: $col ($dtype)"

        for variant in "${UNGROUPED_VARIANTS[@]}"; do
            binary=$(get_binary "$variant")

            if [[ "$variant" == "standard" ]]; then
                query="SELECT MAX($col) FROM $view;"
            else
                query="SELECT pac_max(hash(i), $col, 0.0) FROM $view;"
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
            echo "ungrouped_type,max,$variant,$rows_m,1,$dtype,$wall,$agg,$wall_times,$agg_times" >> "$RESULTS_FILE"
        done
    done
    echo ""
done

echo ""
echo "=== Value Distribution Impact on Bound Optimization ==="
echo "Comparing random (rand_8) vs monotonic increasing (inc_8) vs monotonic decreasing (dec_8)"
echo "All columns are UTINYINT (0-255). Bound optimization should help most with monotonic sequences."
echo ""

# Distribution columns: random vs monotonic (all UTINYINT 0-255)
DIST_COLS=(rand_8 inc_8 dec_8)
DIST_NAMES=(random monotonic_inc monotonic_dec)

for idx in "${!DATA_VIEWS[@]}"; do
    view="${DATA_VIEWS[$idx]}"
    rows_m="${DATA_SIZES_M[$idx]}"
    echo "--- $view (${rows_m}M rows) ---"

    for col_idx in "${!DIST_COLS[@]}"; do
        col="${DIST_COLS[$col_idx]}"
        dist="${DIST_NAMES[$col_idx]}"

        echo "  Distribution: $dist ($col)"

        for variant in "${UNGROUPED_VARIANTS[@]}"; do
            binary=$(get_binary "$variant")

            if [[ "$variant" == "standard" ]]; then
                query="SELECT MAX($col) FROM $view;"
            else
                query="SELECT pac_max(hash(i), $col, 0.0) FROM $view;"
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
            echo "dist_test,max,$variant,$rows_m,1,$dist,$wall,$agg,$wall_times,$agg_times" >> "$RESULTS_FILE"
        done
    done
    echo ""
done

echo ""
echo "=== Grouped MAX with sequential groups (grp_X) ==="
echo "Using tiny_64 (to limit spilling..). Sequential groups have consecutive rows in same group."
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
                query="SELECT $grp_col, MAX($col) FROM $view GROUP BY $grp_col;"
            else
                query="SELECT $grp_col, pac_max(hash(i), $col, 0.0) FROM $view GROUP BY $grp_col;"
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
            echo "grouped_seq,max,$variant,$rows_m,$num_groups,random,$wall,$agg,$wall_times,$agg_times" >> "$RESULTS_FILE"
        done
    done
    echo ""
done

echo ""
echo "=== Grouped MAX with scattered groups (prg_X) ==="
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
                query="SELECT $grp_col, MAX($col) FROM $view GROUP BY $grp_col;"
            else
                query="SELECT $grp_col, pac_max(hash(i), $col, 0.0) FROM $view GROUP BY $grp_col;"
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
            echo "grouped_scat,max,$variant,$rows_m,$num_groups,random,$wall,$agg,$wall_times,$agg_times" >> "$RESULTS_FILE"
        done
    done
    echo ""
done

echo ""
echo "=== Float/Double MAX performance ==="
echo ""

for idx in "${!DATA_VIEWS[@]}"; do
    view="${DATA_VIEWS[$idx]}"
    rows_m="${DATA_SIZES_M[$idx]}"

    for col in col_flt col_dbl; do
        dtype=$([[ "$col" == "col_flt" ]] && echo "float" || echo "double")
        echo "--- $view (${rows_m}M rows), $dtype ---"

        for variant in "${UNGROUPED_VARIANTS[@]}"; do
            binary=$(get_binary "$variant")

            if [[ "$variant" == "standard" ]]; then
                query="SELECT MAX($col) FROM $view;"
            else
                query="SELECT pac_max(hash(i), $col, 0.0) FROM $view;"
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
            echo "float_test,max,$variant,$rows_m,1,$dtype,$wall,$agg,$wall_times,$agg_times" >> "$RESULTS_FILE"
        done
    done
    echo ""
done

echo ""
echo "========================================"
echo "MIN/MAX Benchmark Complete"
echo "Results saved to: $RESULTS_FILE"
echo "========================================"
