#!/usr/bin/env python3
"""Plot pac_max optimization benefits per platform using bar charts."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Variant renaming (CSV name -> display name)
VARIANT_NAMES = {
    'standard': 'DuckDB max (non-pac)',
    'default': 'pac_max (buffering+pruning)',
    'nobuffering': 'pac_max (pruning)',
    'noboundopt': 'pac_max (naive/simd-unfriendly)'
}

# Order for bars (DuckDB first, then most optimized to least)
VARIANT_ORDER = ['DuckDB max (non-pac)', 'pac_max (buffering+pruning)', 'pac_max (pruning)', 'pac_max (naive/simd-unfriendly)']

# Colors for each variant
VARIANT_COLORS = {
    'pac_max (buffering+pruning)': '#2ecc71',  # Green
    'pac_max (pruning)': '#3498db',             # Blue
    'pac_max (naive/simd-unfriendly)': '#e74c3c', # Red
    'DuckDB max (non-pac)': '#95a5a6'           # Gray
}

# Y-axis will be calculated dynamically per platform (max value at ~82%)

# Platform display names
PLATFORM_NAMES = {
    'epyc': 'AMD (EPYC 9645)',
    'granite-rapids': 'Intel Xeon (Granite Rapids)',
    'graviton': 'ARM (Graviton4)',
    'macbook': 'ARM (Apple M2 Pro)'
}


def load_platform_data(platform_dir):
    """Load min_max data for a platform."""
    minmax_files = list(platform_dir.glob('min_max_*.csv'))
    if not minmax_files:
        return None
    return pd.read_csv(minmax_files[0])


def prepare_data(df):
    """Filter and prepare data for plotting."""
    # Filter to dist_test, max aggregate, 1000M rows
    df = df[(df['test'] == 'dist_test') &
            (df['aggregate'] == 'max') &
            (df['rows_m'] == 1000)].copy()

    # Keep only relevant variants
    df = df[df['variant'].isin(VARIANT_NAMES.keys())]

    # Keep only random and monotonic_inc distributions
    df = df[df['dtype'].isin(['random', 'monotonic_inc'])]

    # Rename variants
    df['variant_name'] = df['variant'].map(VARIANT_NAMES)

    # Use wall_sec for display
    df['display_time'] = df['wall_sec']
    df['is_timeout'] = df['wall_sec'] < 0

    return df


def plot_platform(df, platform_name, output_path):
    """Create grouped bar chart for a single platform."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Two distribution types
    distributions = ['random', 'monotonic_inc']
    dist_labels = {'random': 'Random', 'monotonic_inc': 'Monotonic\nIncreasing'}
    n_dists = len(distributions)
    n_variants = len(VARIANT_ORDER)

    # Bar positioning
    bar_width = 0.18
    x = np.arange(n_dists)

    # Collect times
    variant_times = {}
    variant_timeouts = {}

    for variant in VARIANT_ORDER:
        variant_data = df[df['variant_name'] == variant]
        times = []
        timeouts = []
        for dist in distributions:
            data_point = variant_data[variant_data['dtype'] == dist]
            if not data_point.empty:
                t = data_point['display_time'].values[0]
                is_to = data_point['is_timeout'].values[0]
                times.append(t)
                timeouts.append(is_to)
            else:
                times.append(0)
                timeouts.append(False)
        variant_times[variant] = times
        variant_timeouts[variant] = timeouts

    # Calculate y_max dynamically: max non-timeout value at ~82% of y-axis
    all_valid_times = []
    for variant in VARIANT_ORDER:
        times = variant_times.get(variant, [])
        timeouts = variant_timeouts.get(variant, [False] * len(times))
        for t, is_to in zip(times, timeouts):
            if t > 0 and not is_to:
                all_valid_times.append(t)
    max_time = max(all_valid_times) if all_valid_times else 100
    y_max = max_time / 0.82

    # Plot each variant
    for i, variant in enumerate(VARIANT_ORDER):
        times = variant_times[variant]
        timeouts = variant_timeouts[variant]

        offset = (i - n_variants / 2 + 0.5) * bar_width
        positions = x + offset

        # Minimum bar height for visibility
        min_bar_height = y_max * 0.005
        display_times = []
        for t, is_to in zip(times, timeouts):
            if is_to:
                display_times.append(y_max * 1.1)
            elif t > y_max:
                display_times.append(y_max * 0.95)
            elif t > 0 and t < min_bar_height:
                display_times.append(min_bar_height)
            else:
                display_times.append(t)

        bars = ax.bar(positions, display_times, bar_width,
                      label=variant, color=VARIANT_COLORS[variant],
                      edgecolor='black', linewidth=0.5)

        # Add labels
        for j, (bar, actual_time, is_to) in enumerate(zip(bars, times, timeouts)):
            bx = bar.get_x() + bar.get_width() / 2

            if is_to:
                ax.text(bx, y_max * 0.5, 'QUERY FAILED (out of memory)',
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       rotation=90, color='white')
            elif actual_time > y_max:
                # Draw break marks
                by = y_max * 0.92
                break_width = bar.get_width() * 0.6
                break_height = y_max * 0.04
                ax.plot([bx - break_width/2, bx + break_width/2],
                       [by - break_height, by + break_height],
                       color='white', linewidth=2, zorder=10)
                ax.plot([bx - break_width/2, bx + break_width/2],
                       [by - break_height - y_max*0.03, by + break_height - y_max*0.03],
                       color='white', linewidth=2, zorder=10)

                ax.text(bx, y_max * 0.98, f'{actual_time:.0f}s',
                       ha='center', va='bottom', fontsize=7, fontweight='bold',
                       color=VARIANT_COLORS[variant])
            elif actual_time > 0:
                bar_top = bar.get_height()
                ax.text(bx, bar_top + y_max * 0.01, f'{actual_time:.1f}s',
                       ha='center', va='bottom', fontsize=6,
                       color=VARIANT_COLORS[variant])

    x_labels = [dist_labels[d] for d in distributions]

    ax.set_xlabel('Data Distribution', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'pac_max Optimization Impact — {platform_name}\n(1 billion rows, ungrouped)',
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0, y_max * 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    results_dir = Path(__file__).parent / 'results'
    output_dir = Path(__file__).parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    for platform_dir in results_dir.iterdir():
        if not platform_dir.is_dir():
            continue

        platform = platform_dir.name
        df = load_platform_data(platform_dir)

        if df is None:
            print(f"No min_max data for {platform}")
            continue

        df = prepare_data(df)

        if df.empty:
            print(f"No dist_test data for {platform}")
            continue

        platform_name = PLATFORM_NAMES.get(platform, platform)
        output_path = output_dir / f'minmax_optimizations_{platform}.png'

        plot_platform(df, platform_name, output_path)


if __name__ == '__main__':
    main()
