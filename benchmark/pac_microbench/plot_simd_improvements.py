#!/usr/bin/env python3
"""Plot SIMD improvement factors across architectures."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Platform display names
PLATFORM_NAMES = {
    'epyc': 'AMD (EPYC 9645)',
    'granite-rapids': 'Intel Xeon (Granite Rapids)',
    'graviton': 'ARM (AWS Graviton4)',
    'macbook': 'ARM (Apple M2 Pro)'
}

PLATFORM_ORDER = ['graviton', 'macbook', 'granite-rapids', 'epyc']

# Colors for each aggregate
AGG_COLORS = {
    'pac_count': '#2ecc71',  # Green
    'pac_max': '#3498db',    # Blue
    'pac_sum': '#e74c3c'     # Red
}

AGG_ORDER = ['pac_max', 'pac_count', 'pac_sum']


def load_count_data(platform_dir):
    """Load count improvement factor for a platform."""
    count_files = list(platform_dir.glob('count_*.csv'))
    if not count_files:
        return None, 0
    df = pd.read_csv(count_files[0])

    # Filter to 100M rows, ungrouped (count has no dtype column)
    df = df[(df['rows_m'] == 100) & (df['test'] == 'ungrouped')]

    total_default = 0
    total_naive = 0
    n_experiments = 0

    # Group by experiment parameters to find matching pairs
    for test in df['test'].unique():
        test_df = df[df['test'] == test]

        # For grouped: compare default vs nocascading
        naive_variant = 'nocascading'

        # Get unique experiment configs (excluding variant)
        group_cols = [c for c in ['groups', 'dtype'] if c in test_df.columns]
        if group_cols:
            for _, group_df in test_df.groupby(group_cols):
                default_rows = group_df[group_df['variant'] == 'default']
                naive_rows = group_df[group_df['variant'] == naive_variant]

                if not default_rows.empty and not naive_rows.empty:
                    default_time = default_rows['agg_sec'].values[0]
                    naive_time = naive_rows['agg_sec'].values[0]

                    if default_time > 0 and naive_time > 0:
                        total_default += default_time
                        total_naive += naive_time
                        n_experiments += 1
        else:
            default_rows = test_df[test_df['variant'] == 'default']
            naive_rows = test_df[test_df['variant'] == naive_variant]

            if not default_rows.empty and not naive_rows.empty:
                default_time = default_rows['agg_sec'].values[0]
                naive_time = naive_rows['agg_sec'].values[0]

                if default_time > 0 and naive_time > 0:
                    total_default += default_time
                    total_naive += naive_time
                    n_experiments += 1

    if total_default == 0:
        return None, 0

    return total_naive / total_default, n_experiments


def load_max_data(platform_dir):
    """Load max improvement factor for a platform."""
    minmax_files = list(platform_dir.glob('min_max_*.csv'))
    if not minmax_files:
        return None, 0
    df = pd.read_csv(minmax_files[0])

    # Filter to 100M rows, max aggregate, ungrouped, int8 only
    df = df[(df['rows_m'] == 100) & (df['aggregate'] == 'max') & (df['test'].isin(['ungrouped', 'ungrouped_type', 'dist_test']))]
    if 'dtype' in df.columns:
        df = df[df['dtype'] == 'int8']

    total_default = 0
    total_naive = 0
    n_experiments = 0

    for test in df['test'].unique():
        test_df = df[df['test'] == test]

        # For ungrouped: compare default vs nosimd
        naive_variant = 'nosimd'

        # Group by experiment configs
        group_cols = [c for c in ['groups', 'dtype'] if c in test_df.columns]
        if group_cols:
            for _, group_df in test_df.groupby(group_cols):
                default_rows = group_df[group_df['variant'] == 'default']
                naive_rows = group_df[group_df['variant'] == naive_variant]

                if not default_rows.empty and not naive_rows.empty:
                    default_time = default_rows['agg_sec'].values[0]
                    naive_time = naive_rows['agg_sec'].values[0]

                    if default_time > 0 and naive_time > 0:
                        total_default += default_time
                        total_naive += naive_time
                        n_experiments += 1
        else:
            default_rows = test_df[test_df['variant'] == 'default']
            naive_rows = test_df[test_df['variant'] == naive_variant]

            if not default_rows.empty and not naive_rows.empty:
                default_time = default_rows['agg_sec'].values[0]
                naive_time = naive_rows['agg_sec'].values[0]

                if default_time > 0 and naive_time > 0:
                    total_default += default_time
                    total_naive += naive_time
                    n_experiments += 1

    if total_default == 0:
        return None, 0

    return total_naive / total_default, n_experiments


def load_sum_data(platform_dir):
    """Load sum improvement factor for a platform."""
    sumavg_files = list(platform_dir.glob('sum_avg_*.csv'))
    if not sumavg_files:
        return None, 0
    df = pd.read_csv(sumavg_files[0])

    # Handle different column names
    if 'itest' in df.columns:
        test_col = 'itest'
    else:
        test_col = 'test'

    # Filter to sum, 100M rows, ungrouped, tiny (int8)
    df = df[(df['aggregate'] == 'sum') & (df['rows_m'] == 100) & (df[test_col] == 'ungrouped_domain')]
    if 'dtype' in df.columns:
        df = df[df['dtype'] == 'tiny']

    total_default = 0
    total_naive = 0
    n_experiments = 0

    for test in df[test_col].unique():
        test_df = df[df[test_col] == test]

        # For grouped: compare default vs nocascading
        naive_variant = 'nocascading'

        # Group by experiment configs
        group_cols = [c for c in ['groups', 'dtype'] if c in test_df.columns]
        if group_cols:
            for _, group_df in test_df.groupby(group_cols):
                default_rows = group_df[group_df['variant'] == 'default']
                naive_rows = group_df[group_df['variant'] == naive_variant]

                if not default_rows.empty and not naive_rows.empty:
                    default_time = default_rows['agg_sec'].values[0]
                    naive_time = naive_rows['agg_sec'].values[0]

                    if default_time > 0 and naive_time > 0:
                        total_default += default_time
                        total_naive += naive_time
                        n_experiments += 1
        else:
            default_rows = test_df[test_df['variant'] == 'default']
            naive_rows = test_df[test_df['variant'] == naive_variant]

            if not default_rows.empty and not naive_rows.empty:
                default_time = default_rows['agg_sec'].values[0]
                naive_time = naive_rows['agg_sec'].values[0]

                if default_time > 0 and naive_time > 0:
                    total_default += default_time
                    total_naive += naive_time
                    n_experiments += 1

    if total_default == 0:
        return None, 0

    return total_naive / total_default, n_experiments


def main():
    results_dir = Path(__file__).parent / 'results'
    output_dir = Path(__file__).parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    # Collect data for all platforms
    data = {agg: [] for agg in AGG_ORDER}
    platforms_with_data = []

    for platform in PLATFORM_ORDER:
        platform_dir = results_dir / platform
        if not platform_dir.is_dir():
            continue

        count_factor, count_n = load_count_data(platform_dir)
        max_factor, max_n = load_max_data(platform_dir)
        sum_factor, sum_n = load_sum_data(platform_dir)

        print(f"{platform}: count={count_factor:.2f}x (n={count_n}), max={max_factor:.2f}x (n={max_n}), sum={sum_factor:.2f}x (n={sum_n})" if count_factor and max_factor and sum_factor else f"{platform}: count={count_factor}, max={max_factor}, sum={sum_factor}")

        if count_factor is not None or max_factor is not None or sum_factor is not None:
            platforms_with_data.append(platform)
            data['pac_count'].append(count_factor if count_factor else 0)
            data['pac_max'].append(max_factor if max_factor else 0)
            data['pac_sum'].append(sum_factor if sum_factor else 0)

    if not platforms_with_data:
        print("No data found")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    n_platforms = len(platforms_with_data)
    n_aggs = len(AGG_ORDER)
    bar_width = 0.25
    x = np.arange(n_platforms)

    for i, agg in enumerate(AGG_ORDER):
        offset = (i - n_aggs / 2 + 0.5) * bar_width
        positions = x + offset
        values = data[agg]

        bars = ax.bar(positions, values, bar_width,
                      label=agg, color=AGG_COLORS[agg],
                      edgecolor='black', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                       f'{val:.1f}x', ha='center', va='bottom', fontsize=9,
                       color=AGG_COLORS[agg], fontweight='bold')

    # Labels
    ax.set_xlabel('Architecture', fontsize=12)
    ax.set_ylabel('Improvement factor', fontsize=12)
    ax.set_title('SIMD improvements (100M tuples, ungrouped, aggregate cost)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([PLATFORM_NAMES[p] for p in platforms_with_data], fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3, axis='y')

    # Add baseline line at y=1
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    plt.tight_layout()
    output_path = output_dir / 'simd_improvements.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
