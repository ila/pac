#!/usr/bin/env python3
"""Plot pac_count optimization benefits per platform using bar charts."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Variant renaming (CSV name -> display name)
VARIANT_NAMES = {
    'standard': 'DuckDB count (non-pac)',
    'default': 'pac_count (buffering+cascading)',
    'nobuffering': 'pac_count (cascading)',
    'nocascading': 'pac_count (naive/simd-unfriendly)'
}

# Order for bars (DuckDB first, then most optimized to least)
VARIANT_ORDER = ['DuckDB count (non-pac)', 'pac_count (buffering+cascading)', 'pac_count (cascading)', 'pac_count (naive/simd-unfriendly)']

# Colors for each variant
VARIANT_COLORS = {
    'pac_count (buffering+cascading)': '#2ecc71',  # Green
    'pac_count (cascading)': '#3498db',             # Blue
    'pac_count (naive/simd-unfriendly)': '#e74c3c', # Red
    'DuckDB count (non-pac)': '#95a5a6'             # Gray
}

# Y-axis will be calculated dynamically per platform (max value at ~82%)

# Platform display names
PLATFORM_NAMES = {
    'epyc': 'AMD (EPYC 9645)',
    'granite-rapids': 'Intel Xeon (Granite Rapids)',
    'graviton': 'ARM (AWS Graviton4)',
    'macbook': 'ARM (Apple M2 Pro)'
}

# Timeout ceiling for display
TIMEOUT_CEILING = 300


def load_platform_data(platform_dir):
    """Load count data for a platform."""
    count_files = list(platform_dir.glob('count_*.csv'))
    if not count_files:
        return None
    return pd.read_csv(count_files[0])


def prepare_data(df):
    """Filter and prepare data for plotting."""
    # Filter to 1000M rows only
    df = df[df['rows_m'] == 1000].copy()

    # Keep only relevant variants
    df = df[df['variant'].isin(VARIANT_NAMES.keys())]

    # Keep ungrouped and grouped_scat tests
    df = df[df['test'].isin(['ungrouped', 'grouped_scat'])]

    # Rename variants
    df['variant_name'] = df['variant'].map(VARIANT_NAMES)

    # Map to group values (ungrouped = 0)
    df['group_values'] = df.apply(
        lambda r: 0 if r['test'] == 'ungrouped' else r['groups'],
        axis=1
    )

    # Handle timeouts: mark them but keep original value for display logic
    # Use wall_sec (full query time) instead of agg_sec (aggregation only)
    df['display_time'] = df['wall_sec']
    df['is_timeout'] = df['wall_sec'] < 0

    return df


def plot_platform(df, platform_name, output_path):
    """Create grouped bar chart for a single platform with interrupted bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique group values and sort
    group_values = sorted(df['group_values'].unique())
    n_groups = len(group_values)
    n_variants = len(VARIANT_ORDER)

    # Bar positioning
    bar_width = 0.2
    x = np.arange(n_groups)

    # Collect all times and timeout info
    variant_times = {}
    variant_timeouts = {}

    for variant in VARIANT_ORDER:
        variant_data = df[df['variant_name'] == variant]
        times = []
        timeouts = []
        for gv in group_values:
            data_point = variant_data[variant_data['group_values'] == gv]
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

        # Calculate bar positions
        offset = (i - n_variants / 2 + 0.5) * bar_width
        positions = x + offset

        # Clip times for display, handle timeouts as full-height bars
        # Minimum bar height for visibility (barely visible)
        min_bar_height = y_max * 0.005
        display_times = []
        for t, is_to in zip(times, timeouts):
            if is_to:
                display_times.append(y_max * 1.1)  # Full height for timeout
            elif t > y_max:
                display_times.append(y_max * 0.95)  # Show as interrupted bar
            elif t > 0 and t < min_bar_height:
                display_times.append(min_bar_height)  # Minimum visible height
            else:
                display_times.append(t)

        # Plot bars
        bars = ax.bar(positions, display_times, bar_width,
                      label=variant, color=VARIANT_COLORS[variant],
                      edgecolor='black', linewidth=0.5)

        # Add break marks and value labels for all bars
        for j, (bar, actual_time, is_to) in enumerate(zip(bars, times, timeouts)):
            bx = bar.get_x() + bar.get_width() / 2

            if is_to:
                # Write failure message vertically in white inside the bar
                ax.text(bx, y_max * 0.5, 'QUERY FAILED (out of memory)',
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       rotation=90, color='white')
            elif actual_time > y_max:
                # Draw break marks (two small diagonal lines)
                by = y_max * 0.92
                break_width = bar.get_width() * 0.6
                break_height = y_max * 0.04
                ax.plot([bx - break_width/2, bx + break_width/2],
                       [by - break_height, by + break_height],
                       color='white', linewidth=2, zorder=10)
                ax.plot([bx - break_width/2, bx + break_width/2],
                       [by - break_height - y_max*0.03, by + break_height - y_max*0.03],
                       color='white', linewidth=2, zorder=10)

                # Add value label on top
                ax.text(bx, y_max * 0.98, f'{actual_time:.0f}s',
                       ha='center', va='bottom', fontsize=7, fontweight='bold',
                       color=VARIANT_COLORS[variant])
            elif actual_time > 0:
                # Add value label on top of regular bars
                bar_top = bar.get_height()
                ax.text(bx, bar_top + y_max * 0.01, f'{actual_time:.1f}s',
                       ha='center', va='bottom', fontsize=6,
                       color=VARIANT_COLORS[variant])

    # X-axis labels
    x_labels = []
    for gv in group_values:
        if gv == 0:
            x_labels.append('0\n(ungrouped)')
        elif gv >= 1_000_000:
            x_labels.append(f'{gv // 1_000_000}M')
        elif gv >= 1_000:
            x_labels.append(f'{gv // 1_000}K')
        else:
            x_labels.append(str(gv))

    ax.set_xlabel('distinct GROUP BY values for a COUNT(*)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'pac_count Optimization Impact — {platform_name}\n(1 billion rows, scattered groups)',
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.legend(loc='upper left', fontsize=10)
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
            print(f"No count data for {platform}")
            continue

        df = prepare_data(df)

        if df.empty:
            print(f"No 1000M data for {platform}")
            continue

        platform_name = PLATFORM_NAMES.get(platform, platform)
        output_path = output_dir / f'count_optimizations_{platform}.png'

        plot_platform(df, platform_name, output_path)


if __name__ == '__main__':
    main()
