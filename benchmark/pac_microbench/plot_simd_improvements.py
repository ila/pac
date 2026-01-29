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
    'graviton': 'ARM (Graviton4)',
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
        return None
    df = pd.read_csv(count_files[0])

    # Filter to 1000M rows, ungrouped (nocascading times out at 10M groups)
    df = df[(df['rows_m'] == 1000) &
            (df['test'] == 'ungrouped')]

    default_row = df[df['variant'] == 'default']
    naive_row = df[df['variant'] == 'nocascading']

    if default_row.empty or naive_row.empty:
        return None

    default_time = default_row['wall_sec'].values[0]
    naive_time = naive_row['wall_sec'].values[0]

    # Handle timeouts
    if default_time < 0 or naive_time < 0:
        return None

    return naive_time / default_time


def load_max_data(platform_dir):
    """Load max improvement factor for a platform."""
    minmax_files = list(platform_dir.glob('min_max_*.csv'))
    if not minmax_files:
        return None
    df = pd.read_csv(minmax_files[0])

    # Filter to dist_test, max, 1000M rows, random distribution
    df = df[(df['test'] == 'dist_test') &
            (df['aggregate'] == 'max') &
            (df['rows_m'] == 1000) &
            (df['dtype'] == 'random')]

    default_row = df[df['variant'] == 'default']
    naive_row = df[df['variant'] == 'noboundopt']

    if default_row.empty or naive_row.empty:
        return None

    default_time = default_row['wall_sec'].values[0]
    naive_time = naive_row['wall_sec'].values[0]

    if default_time < 0 or naive_time < 0:
        return None

    return naive_time / default_time


def load_sum_data(platform_dir):
    """Load sum improvement factor for a platform."""
    sumavg_files = list(platform_dir.glob('sum_avg_*.csv'))
    if not sumavg_files:
        return None
    df = pd.read_csv(sumavg_files[0])

    # Handle different column names
    if 'itest' in df.columns:
        test_col = 'itest'
    else:
        test_col = 'test'

    # Filter to sum, 1000M rows, ungrouped (nosimd only available for ungrouped)
    df = df[(df['aggregate'] == 'sum') &
            (df['rows_m'] == 1000) &
            (df[test_col] == 'ungrouped_domain')]

    # Use dtype='large' if available
    if 'large' in df['dtype'].values:
        df = df[df['dtype'] == 'large']
    elif 'tiny' in df['dtype'].values:
        df = df[df['dtype'] == 'tiny']

    default_row = df[df['variant'] == 'default']
    naive_row = df[df['variant'] == 'nosimd']

    if default_row.empty or naive_row.empty:
        return None

    default_time = default_row['wall_sec'].values[0]
    naive_time = naive_row['wall_sec'].values[0]

    if default_time < 0 or naive_time < 0:
        return None

    return naive_time / default_time


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

        count_factor = load_count_data(platform_dir)
        max_factor = load_max_data(platform_dir)
        sum_factor = load_sum_data(platform_dir)

        print(f"{platform}: count={count_factor}, max={max_factor}, sum={sum_factor}")

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
    ax.set_title('SIMD improvements (over all experiments at 10M tuples)', fontsize=14)
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
