#!/usr/bin/env python3
"""
Analyze and visualize PAC microbenchmark results.
Generates comparison tables and summary statistics.
"""

import csv
import sys
import os
from collections import defaultdict
from pathlib import Path

def load_csv(filepath):
    """Load a CSV file into a list of dicts."""
    with open(filepath, 'r') as f:
        return list(csv.DictReader(f))

def format_speedup(base, opt):
    """Format speedup ratio."""
    if base == 0 or opt == 0:
        return "N/A"
    ratio = base / opt
    if ratio >= 1:
        return f"{ratio:.2f}x faster"
    else:
        return f"{1/ratio:.2f}x slower"

def analyze_count(results):
    """Analyze COUNT benchmark results."""
    print("\n" + "="*60)
    print("PAC COUNT Analysis")
    print("="*60)

    # Group by test type
    ungrouped = [r for r in results if r['test'] == 'ungrouped']
    grouped = [r for r in results if r['test'] == 'grouped']

    if ungrouped:
        print("\n--- Ungrouped COUNT (banked vs nonbanked) ---")
        print(f"{'Size':<15} {'Banked (s)':<15} {'NonBanked (s)':<15} {'Speedup':<20}")
        print("-" * 65)

        by_size = defaultdict(dict)
        for r in ungrouped:
            by_size[r['data_size_m']][r['variant']] = float(r['time_sec'])

        for size in sorted(by_size.keys(), key=int):
            banked = by_size[size].get('default', 0)
            nonbanked = by_size[size].get('count_nonbanked', 0)
            speedup = format_speedup(nonbanked, banked)
            print(f"{size + 'M':<15} {banked:<15.3f} {nonbanked:<15.3f} {speedup:<20}")

    if grouped:
        print("\n--- Grouped COUNT (varying group counts) ---")
        print(f"{'Groups':<15} {'Banked (s)':<15} {'NonBanked (s)':<15} {'Speedup':<20}")
        print("-" * 65)

        by_groups = defaultdict(dict)
        for r in grouped:
            by_groups[r['groups']][r['variant']] = float(r['time_sec'])

        for groups in sorted(by_groups.keys(), key=int):
            banked = by_groups[groups].get('default', 0)
            nonbanked = by_groups[groups].get('count_nonbanked', 0)
            speedup = format_speedup(nonbanked, banked)
            print(f"{int(groups):,}".ljust(15) + f"{banked:<15.3f} {nonbanked:<15.3f} {speedup:<20}")

def analyze_sum_avg(results):
    """Analyze SUM/AVG benchmark results."""
    print("\n" + "="*60)
    print("PAC SUM/AVG Analysis")
    print("="*60)

    # Group by dtype
    by_dtype = defaultdict(lambda: defaultdict(dict))
    for r in results:
        if r['test'] == 'ungrouped' and r['aggregate'] == 'sum':
            key = (r['dtype'], r['distribution'])
            by_dtype[key][r['variant']] = float(r['time_sec'])

    if by_dtype:
        print("\n--- Ungrouped SUM by dtype/distribution (cascading vs non-cascading) ---")
        print(f"{'Type/Dist':<20} {'Cascading (s)':<15} {'NonCasc (s)':<15} {'Speedup':<20}")
        print("-" * 70)

        for (dtype, dist) in sorted(by_dtype.keys()):
            casc = by_dtype[(dtype, dist)].get('default', 0)
            noncasc = by_dtype[(dtype, dist)].get('sumavg_noncascading', 0)
            if casc and noncasc:
                speedup = format_speedup(noncasc, casc)
                label = f"{dtype}/{dist}"
                print(f"{label:<20} {casc:<15.3f} {noncasc:<15.3f} {speedup:<20}")

    # Float comparison
    float_results = [r for r in results if r['test'] == 'float_perf']
    if float_results:
        print("\n--- Float vs Double SUM Performance ---")
        by_dtype_size = defaultdict(dict)
        for r in float_results:
            key = (r['dtype'], r['data_size_m'])
            by_dtype_size[key][r['variant']] = float(r['time_sec'])

        print(f"{'Type/Size':<20} {'Cascading (s)':<15} {'NonCasc (s)':<15}")
        print("-" * 50)
        for (dtype, size) in sorted(by_dtype_size.keys()):
            casc = by_dtype_size[(dtype, size)].get('default', 0)
            noncasc = by_dtype_size[(dtype, size)].get('sumavg_noncascading', 0)
            label = f"{dtype}/{size}M"
            print(f"{label:<20} {casc:<15.3f} {noncasc:<15.3f}")

def analyze_min_max(results):
    """Analyze MIN/MAX benchmark results."""
    print("\n" + "="*60)
    print("PAC MIN/MAX Analysis")
    print("="*60)

    # Distribution impact
    dist_results = [r for r in results if r['test'] == 'dist_test']
    if dist_results:
        print("\n--- Distribution Impact on Bound Optimization ---")
        print("(Lower is better, comparing default vs no-bound-opt)")

        by_dist_agg = defaultdict(lambda: defaultdict(dict))
        for r in dist_results:
            by_dist_agg[(r['distribution'], r['aggregate'])][r['variant']] = float(r['time_sec'])

        print(f"{'Dist/Agg':<20} {'Default (s)':<12} {'NoBound (s)':<12} {'Bound Speedup':<20}")
        print("-" * 64)

        for (dist, agg) in sorted(by_dist_agg.keys()):
            default = by_dist_agg[(dist, agg)].get('default', 0)
            nobound = by_dist_agg[(dist, agg)].get('minmax_noboundopt', 0)
            if default and nobound:
                speedup = format_speedup(nobound, default)
                label = f"{dist}/{agg}"
                print(f"{label:<20} {default:<12.3f} {nobound:<12.3f} {speedup:<20}")

    # Type comparison
    dtype_results = [r for r in results if r['test'] == 'dtype_test']
    if dtype_results:
        print("\n--- Data Type Comparison (random distribution) ---")

        by_dtype_agg = defaultdict(lambda: defaultdict(dict))
        for r in dtype_results:
            by_dtype_agg[(r['dtype'], r['aggregate'])][r['variant']] = float(r['time_sec'])

        print(f"{'Type/Agg':<15} {'Default':<10} {'NonBanked':<10} {'NoBound':<10}")
        print("-" * 45)

        for (dtype, agg) in sorted(by_dtype_agg.keys()):
            default = by_dtype_agg[(dtype, agg)].get('default', 0)
            nonbanked = by_dtype_agg[(dtype, agg)].get('minmax_nonbanked', 0)
            nobound = by_dtype_agg[(dtype, agg)].get('minmax_noboundopt', 0)
            label = f"{dtype}/{agg}"
            print(f"{label:<15} {default:<10.3f} {nonbanked:<10.3f} {nobound:<10.3f}")

def main():
    results_dir = Path(__file__).parent / "results"

    if not results_dir.exists():
        print("No results directory found. Run benchmarks first.")
        sys.exit(1)

    # Find latest result files
    count_files = sorted(results_dir.glob("count_*.csv"))
    sum_avg_files = sorted(results_dir.glob("sum_avg_*.csv"))
    min_max_files = sorted(results_dir.glob("min_max_*.csv"))

    print("PAC Microbenchmark Results Analysis")
    print("=" * 60)

    if count_files:
        latest = count_files[-1]
        print(f"\nAnalyzing COUNT results: {latest.name}")
        analyze_count(load_csv(latest))

    if sum_avg_files:
        latest = sum_avg_files[-1]
        print(f"\nAnalyzing SUM/AVG results: {latest.name}")
        analyze_sum_avg(load_csv(latest))

    if min_max_files:
        latest = min_max_files[-1]
        print(f"\nAnalyzing MIN/MAX results: {latest.name}")
        analyze_min_max(load_csv(latest))

    if not (count_files or sum_avg_files or min_max_files):
        print("No result files found. Run benchmarks first:")
        print("  ./run_all.sh")

if __name__ == "__main__":
    main()
