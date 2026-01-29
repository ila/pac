#!/usr/bin/env python3
"""Plot pac_sum optimization benefits per platform using bar charts."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Variant renaming (CSV name -> display name with newlines for narrower legend)
VARIANT_NAMES = {
    'standard': 'DuckDB sum\n(non-pac)',
    'default': 'pac_sum\n(approximate+buffering)',
    'nobuffering': 'pac_sum\n(approximate)',
    'exactsum': 'pac_sum\n(exact, cascading)',
    'nocascading': 'pac_sum\n(naive/simd-unfriendly)'
}

# Order for bars (DuckDB first, then most optimized to least)
VARIANT_ORDER = ['DuckDB sum\n(non-pac)', 'pac_sum\n(approximate+buffering)', 'pac_sum\n(approximate)', 'pac_sum\n(exact, cascading)', 'pac_sum\n(naive/simd-unfriendly)']

# Colors for each variant
VARIANT_COLORS = {
    'pac_sum\n(approximate+buffering)': '#2ecc71',  # Green
    'pac_sum\n(approximate)': '#3498db',             # Blue
    'pac_sum\n(exact, cascading)': '#f39c12',        # Orange
    'pac_sum\n(naive/simd-unfriendly)': '#e74c3c',   # Red
    'DuckDB sum\n(non-pac)': '#95a5a6'               # Gray
}

# Y-axis will be calculated dynamically per platform (max value at ~82%)

# Platform display names
PLATFORM_NAMES = {
    'epyc': 'AMD (EPYC 9645)',
    'granite-rapids': 'Intel Xeon (Granite Rapids)',
    'graviton': 'ARM (AWS Graviton4)',
    'macbook': 'ARM (Apple M2 Pro)'
}


def load_platform_data(platform_dir):
    """Load sum_avg data for a platform."""
    sumavg_files = list(platform_dir.glob('sum_avg_*.csv'))
    if not sumavg_files:
        return None
    return pd.read_csv(sumavg_files[0])


def prepare_data(df):
    """Filter and prepare data for plotting."""
    # Handle different column names across platforms
    if 'itest' in df.columns:
        test_col = 'itest'
    else:
        test_col = 'test'

    # Filter to sum aggregate, 1000M rows
    df = df[(df['aggregate'] == 'sum') & (df['rows_m'] == 1000)].copy()

    # For grouped_scat tests, use dtype='large'
    # For ungrouped, also use dtype='large' (or 'tiny' for nosimd consistency)

    # Get grouped_scat data - prefer 'large' dtype, fall back to 'tiny'
    grouped_df = df[df[test_col] == 'grouped_scat'].copy()
    if 'large' in grouped_df['dtype'].values:
        grouped_df = grouped_df[grouped_df['dtype'] == 'large']
    elif 'tiny' in grouped_df['dtype'].values:
        grouped_df = grouped_df[grouped_df['dtype'] == 'tiny']
    grouped_df['group_values'] = grouped_df['groups']

    # Get ungrouped data for nosimd (and as x=0 reference) - same dtype preference
    ungrouped_df = df[df[test_col] == 'ungrouped_domain'].copy()
    if 'large' in ungrouped_df['dtype'].values:
        ungrouped_df = ungrouped_df[ungrouped_df['dtype'] == 'large']
    elif 'tiny' in ungrouped_df['dtype'].values:
        ungrouped_df = ungrouped_df[ungrouped_df['dtype'] == 'tiny']
    ungrouped_df['group_values'] = 0
    ungrouped_df['groups'] = 1

    # Combine
    df = pd.concat([ungrouped_df, grouped_df], ignore_index=True)

    # Keep only relevant variants
    df = df[df['variant'].isin(VARIANT_NAMES.keys())]

    # Rename variants
    df['variant_name'] = df['variant'].map(VARIANT_NAMES)

    # Use wall_sec for display
    df['display_time'] = df['wall_sec']
    df['is_timeout'] = df['wall_sec'] < 0

    return df


def plot_platform(df, platform_name, output_path):
    """Create grouped bar chart for a single platform."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get unique group values and sort
    group_values = sorted(df['group_values'].unique())
    n_groups = len(group_values)
    n_variants = len(VARIANT_ORDER)

    # Bar positioning
    bar_width = 0.15
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

    # Calculate y_max: 10M approximate+buffering at 85% of y-axis
    approx_buff_times = variant_times.get('pac_sum\n(approximate+buffering)', [])
    approx_buff_timeouts = variant_timeouts.get('pac_sum\n(approximate+buffering)', [])
    approx_buff_10m = 100  # default
    if approx_buff_times and len(approx_buff_times) > 0:
        last_idx = len(approx_buff_times) - 1
        if not approx_buff_timeouts[last_idx] and approx_buff_times[last_idx] > 0:
            approx_buff_10m = approx_buff_times[last_idx]
    y_max = approx_buff_10m / 0.85

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
                display_times.append(y_max)  # Full height for timeout
            elif t > y_max:
                display_times.append(y_max * 0.95)  # 95% for cut bars
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

                bar_top = bar.get_height()
                ax.text(bx, bar_top + y_max * 0.01, f'{actual_time:.0f}s',
                       ha='center', va='bottom', fontsize=7, fontweight='bold',
                       color=VARIANT_COLORS[variant])
            elif actual_time > 0:
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

    ax.set_xlabel('distinct GROUP BY values for a SUM(*)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'pac_sum Optimization Impact — {platform_name}\n(1 billion rows, scattered groups)',
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlim(-1, n_groups - 0.5)  # Add padding on the left for legend
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0, y_max)
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
            print(f"No sum_avg data for {platform}")
            continue

        df = prepare_data(df)

        if df.empty:
            print(f"No 1000M data for {platform}")
            continue

        platform_name = PLATFORM_NAMES.get(platform, platform)
        output_path = output_dir / f'sum_optimizations_{platform}.png'

        plot_platform(df, platform_name, output_path)


if __name__ == '__main__':
    main()
