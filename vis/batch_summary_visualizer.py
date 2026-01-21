#!/usr/bin/env python
"""
Batch Summary Visualizer

Visualizes all metrics from batch_summary.csv grouped by melting regime.
Creates box/violin plots for each metric with regime color-coding.

Author: AI Assistant
Date: 2025-01-09
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path for tools import
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths, get_logbook, get_logbook_data, filter_logbook_tracks, get_regime_marker_dict, define_collumn_labels


def load_batch_summary(csv_path):
    """
    Load batch_summary.csv file.

    Args:
        csv_path (str or Path): Path to batch_summary.csv

    Returns:
        pd.DataFrame: Loaded dataframe

    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"batch_summary.csv not found at: {csv_path}\n"
            f"Expected location: {{hdf5_path}}/timeseries_compare_analysis_results/run_summary/batch_summary.csv"
        )

    df = pd.read_csv(csv_path, encoding='utf-8')
    print(f"Loaded batch_summary.csv: {len(df)} tracks, {len(df.columns)} columns")

    return df


def merge_regime_data(df, logbook, verbose=True):
    """
    Merge melting regime information from logbook into dataframe.

    Args:
        df (pd.DataFrame): Batch summary dataframe with 'trackid' column
        logbook (pd.DataFrame): Logbook dataframe
        verbose (bool): Print merge statistics

    Returns:
        pd.DataFrame: Dataframe with 'melting_regime' column added
    """
    # Create regime column
    regimes = []
    missing_trackids = []

    for trackid in df['trackid']:
        try:
            track_data = get_logbook_data(logbook, trackid, layer_n=1)
            regime = track_data.get('melting_regime', None)
            regimes.append(regime)
        except (KeyError, ValueError):
            regimes.append(None)
            missing_trackids.append(trackid)

    df['melting_regime'] = regimes

    # Print statistics
    if verbose:
        n_matched = df['melting_regime'].notna().sum()
        n_missing = len(missing_trackids)
        print(f"\nMerged with logbook:")
        print(f"  Matched: {n_matched}/{len(df)} tracks")

        if n_missing > 0:
            print(f"  Missing from logbook: {n_missing} tracks")
            if n_missing <= 10:
                print(f"    Missing trackids: {', '.join(missing_trackids)}")
            else:
                print(f"    Missing trackids (first 10): {', '.join(missing_trackids[:10])}")

        # Show regime distribution
        regime_counts = df['melting_regime'].value_counts()
        print(f"\nRegime distribution:")
        for regime, count in regime_counts.items():
            print(f"  {regime}: {count}")

    return df


def apply_filters(df, logbook, filter_dict, verbose=True):
    """
    Apply filters to dataframe using filter_logbook_tracks logic.

    Args:
        df (pd.DataFrame): Batch summary dataframe
        logbook (pd.DataFrame): Logbook dataframe
        filter_dict (dict): Dictionary of filters (see filter_logbook_tracks)
        verbose (bool): Print filter statistics

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if not filter_dict or all(v is None for v in filter_dict.values()):
        if verbose:
            print("\nNo filters applied (showing all data)")
        return df

    # Use filter_logbook_tracks to get filtered logbook
    filtered_logbook, active_filters = filter_logbook_tracks(logbook, filter_dict)

    if verbose:
        print(f"\n--- FILTER TROUBLESHOOTING ---")
        print(f"Filter dict: {filter_dict}")
        print(f"Logbook rows after filter_logbook_tracks: {len(filtered_logbook)}")
        print(f"Active filters: {active_filters}")

    # Get trackids from filtered logbook
    # Logbook stores trackids in format "0100" or "0100_01", batch_summary uses "0100_01"
    filtered_trackids = set()

    # Debug: Check column names in logbook
    if verbose:
        print(f"\nLogbook column names: {list(filtered_logbook.columns)}")
        if len(filtered_logbook) > 0:
            print(f"Sample row:\n{filtered_logbook.iloc[0]}")

    for idx, row in filtered_logbook.iterrows():
        trackid = row['trackid']
        filtered_trackids.add(trackid)

    if verbose:
        print(f"\nGenerated {len(filtered_trackids)} filtered trackids")
        print(f"Sample filtered trackids (first 10): {sorted(list(filtered_trackids))[:10]}")
        print(f"\nBatch summary trackids (first 10): {sorted(df['trackid'].tolist())[:10]}")

        # Check overlap
        csv_trackids = set(df['trackid'])
        overlap = filtered_trackids & csv_trackids
        print(f"\nOverlap analysis:")
        print(f"  Filtered trackids: {len(filtered_trackids)}")
        print(f"  CSV trackids: {len(csv_trackids)}")
        print(f"  Overlap: {len(overlap)}")

        if len(overlap) == 0:
            print("\nWARNING: No overlap! Sample comparison:")
            print(f"  From filter: {sorted(list(filtered_trackids))[:5]}")
            print(f"  From CSV: {sorted(list(csv_trackids))[:5]}")

    # Filter dataframe
    df_filtered = df[df['trackid'].isin(filtered_trackids)].copy()

    if verbose:
        print(f"\n--- FILTER RESULTS ---")
        print(f"Applied filters: {', '.join(active_filters)}")
        print(f"  Tracks before filtering: {len(df)}")
        print(f"  Tracks after filtering: {len(df_filtered)}")
        print(f"  Removed: {len(df) - len(df_filtered)} tracks")

    return df_filtered


def categorize_column(column_name):
    """
    Determine category and subcategory for a column.

    Args:
        column_name (str): Column name from batch_summary.csv

    Returns:
        tuple: (category, subcategory, display_name)
            category: 'statistics', 'correlations', 'clustering', or None
            subcategory: Signal name for statistics, None otherwise
            display_name: Human-readable name for the metric
    """
    # Skip trackid and melting_regime columns
    if column_name in ['trackid', 'melting_regime']:
        return None, None, None

    # Statistics: {signal}_{metric}
    stat_metrics = ['mean', 'median', 'std', 'var', 'min', 'max', 'range',
                   'skewness', 'kurtosis', 'rms', 'energy', 'zero_crossings']

    for metric in stat_metrics:
        if column_name.endswith(f'_{metric}'):
            signal = column_name.replace(f'_{metric}', '')
            display_name = f"{signal} - {metric.replace('_', ' ').title()}"
            return 'statistics', signal, display_name

    # Correlations: {sig1}_vs_{sig2}_{type}
    if '_vs_' in column_name:
        if column_name.endswith('_pearson') or column_name.endswith('_spearman'):
            display_name = column_name.replace('_', ' ').title()
            return 'correlations', None, display_name
        elif column_name.endswith('_pearson_p') or column_name.endswith('_spearman_p'):
            display_name = column_name.replace('_', ' ').replace(' P', ' p-value').title()
            return 'correlations', None, display_name
        elif 'silhouette' in column_name or 'optimal_k' in column_name:
            display_name = column_name.replace('_', ' ').title()
            return 'clustering', None, display_name

    # Unknown column format
    return 'other', None, column_name


def get_ylabel_from_column(column):
    """
    Get proper y-axis label for a column using define_collumn_labels().

    Args:
        column (str): Column name from batch_summary.csv

    Returns:
        str: Proper axis label with units
    """
    col_dict = define_collumn_labels()

    # Pattern matching priority:
    # 1. For statistics (signal_metric): extract signal and combine with metric
    # 2. For correlations: use correlation-specific labels
    # 3. For clustering: use clustering-specific labels
    # 4. Direct match from col_dict

    # Statistics pattern: {signal}_{metric}
    stat_metrics = {
        'mean': 'Mean',
        'median': 'Median',
        'std': 'Standard deviation',
        'var': 'Variance',
        'min': 'Minimum',
        'max': 'Maximum',
        'range': 'Range',
        'skewness': 'Skewness',
        'kurtosis': 'Kurtosis',
        'rms': 'RMS',
        'energy': 'Energy',
        'zero_crossings': 'Zero crossings'
    }

    # Check if this is a statistic column (ends with _metric)
    for metric, metric_label in stat_metrics.items():
        if column.endswith(f'_{metric}'):
            # Extract signal name (everything before _metric)
            signal = column.replace(f'_{metric}', '')

            # Map signal names to their proper labels from col_dict
            signal_map = {
                'PD1': 'PD1',
                'PD2': 'PD2',
                'KH_depth': col_dict.get('KH_depth', [None, 'KH depth'])[1],
                'KH_area': col_dict.get('KH_area', [None, 'KH area'])[1],
                'KH_length': col_dict.get('KH_length', [None, 'KH length'])[1],
                'FKW_angle': col_dict.get('fkw_angle', [None, 'FKW angle'])[1]
            }

            # If we have a specific signal label, use it with the metric
            if signal in signal_map:
                return f"{signal_map[signal]} - {metric_label}"
            else:
                # Fallback: use formatted signal name
                return f"{signal.replace('_', ' ').title()} - {metric_label}"

    # Correlation patterns: {sig1}_vs_{sig2}_{type}
    if '_vs_' in column:
        if 'pearson' in column:
            if '_p' in column:
                return 'Pearson p-value'
            return 'Pearson correlation coefficient'
        elif 'spearman' in column:
            if '_p' in column:
                return 'Spearman p-value'
            return 'Spearman correlation coefficient'
        elif 'silhouette' in column:
            if 'k2' in column:
                return 'Silhouette score (k=2)'
            elif 'k3' in column:
                return 'Silhouette score (k=3)'
        elif 'optimal_k' in column:
            return 'Optimal cluster count'

    # Check for direct match in col_dict
    if column in col_dict:
        return col_dict[column][1]

    # Fallback: use column name formatted
    return column.replace('_', ' ').title()


def create_regime_plot(df, column, plot_type, output_path, dpi=600):
    """
    Create box/violin plot for a single metric grouped by regime.

    Args:
        df (pd.DataFrame): Dataframe with 'melting_regime' and metric columns
        column (str): Column name to plot
        plot_type (str): 'box', 'violin', or 'both'
        output_path (Path): Path to save figure
        dpi (int): Figure DPI
    """
    # Filter out rows with missing regime or metric data
    plot_df = df[['melting_regime', column]].dropna()

    if len(plot_df) == 0:
        warnings.warn(f"No valid data for column '{column}' - skipping")
        return False

    # Check if column has variation
    if plot_df[column].std() == 0:
        warnings.warn(f"Column '{column}' has zero variance - skipping")
        return False

    # Get regime colors and labels
    marker_dict = get_regime_marker_dict()

    # Create ordered list of regimes (in dictionary order)
    regime_order = list(marker_dict.keys())

    # Filter to only regimes present in data (maintaining order)
    regime_order_present = [r for r in regime_order if r in plot_df['melting_regime'].unique()]

    # Create color palette and label mapping
    regime_colors = {regime: marker_dict[regime]['c'] for regime in regime_order_present}
    regime_labels = {regime: marker_dict[regime]['label'] for regime in marker_dict.keys()}

    # Replace regime names with shortened labels in dataframe for plotting
    plot_df['regime_label'] = plot_df['melting_regime'].map(regime_labels)

    # Set up figure with A4 half-width format
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9
    })

    fig, ax = plt.subplots(figsize=(4.13, 3.5))

    # Map labels to colors for seaborn (needs labels as keys)
    label_colors = {regime_labels[r]: regime_colors[r] for r in regime_order_present}
    label_order = [regime_labels[r] for r in regime_order_present]

    # Create plot based on type (fix seaborn deprecation warning by using hue)
    if plot_type == 'box':
        sns.boxplot(data=plot_df, x='regime_label', y=column, order=label_order,
                   hue='regime_label', palette=label_colors, legend=False, ax=ax)
    elif plot_type == 'violin':
        sns.violinplot(data=plot_df, x='regime_label', y=column, order=label_order,
                      hue='regime_label', palette=label_colors, legend=False, ax=ax)
    elif plot_type == 'both':
        # Violin plot with box plot overlay
        sns.violinplot(data=plot_df, x='regime_label', y=column, order=label_order,
                      hue='regime_label', palette=label_colors, legend=False,
                      ax=ax, inner=None, alpha=0.6)
        sns.boxplot(data=plot_df, x='regime_label', y=column, order=label_order,
                   hue='regime_label', palette=label_colors, legend=False,
                   ax=ax, width=0.3, boxprops=dict(alpha=0.7), showcaps=True,
                   showfliers=False, whiskerprops=dict(alpha=0.7))

    # Formatting
    _, display_name = categorize_column(column)[1:]
    if display_name is None:
        display_name = column.replace('_', ' ').title()

    # Get proper y-axis label from define_collumn_labels()
    ylabel = get_ylabel_from_column(column)

    ax.set_title(display_name, fontweight='bold')
    ax.set_xlabel('Melting Regime')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, color='#CCCCCC')

    # Rotate x-tick labels for readability (use plt.setp to avoid warning)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Tight layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return True


def visualize_all_columns(df, output_dir, plot_type='box', dpi=600, quiet=False):
    """
    Create visualizations for all columns in dataframe.

    Args:
        df (pd.DataFrame): Batch summary dataframe with melting_regime
        output_dir (Path): Output directory for plots
        plot_type (str): 'box', 'violin', or 'both'
        dpi (int): Figure DPI
        quiet (bool): Suppress progress bar
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Categorize all columns
    columns_by_category = {}
    for col in df.columns:
        category, subcategory, display_name = categorize_column(col)
        if category is None:
            continue  # Skip trackid and melting_regime

        if category not in columns_by_category:
            columns_by_category[category] = []
        columns_by_category[category].append((col, subcategory, display_name))

    # Count total plots
    total_plots = sum(len(cols) for cols in columns_by_category.values())

    print(f"\nGenerating {total_plots} visualizations...")
    print(f"  Statistics: {len(columns_by_category.get('statistics', []))}")
    print(f"  Correlations: {len(columns_by_category.get('correlations', []))}")
    print(f"  Clustering: {len(columns_by_category.get('clustering', []))}")

    # Create plots with progress bar
    plots_created = 0
    plots_skipped = 0

    iterator = tqdm(total=total_plots, desc="Creating plots", disable=quiet)

    for category, columns in columns_by_category.items():
        # Group statistics by signal
        if category == 'statistics':
            # Organize by subcategory (signal)
            signals = {}
            for col, subcategory, display_name in columns:
                if subcategory not in signals:
                    signals[subcategory] = []
                signals[subcategory].append(col)

            # Create plots organized by signal
            for signal, cols in signals.items():
                signal_dir = output_dir / category / signal
                for col in cols:
                    output_path = signal_dir / f"{col}.png"
                    iterator.set_description(f"[{category}/{signal}] {col}")
                    success = create_regime_plot(df, col, plot_type, output_path, dpi)
                    if success:
                        plots_created += 1
                    else:
                        plots_skipped += 1
                    iterator.update(1)
        else:
            # Other categories go directly in their folder
            category_dir = output_dir / category
            for col, _, display_name in columns:
                output_path = category_dir / f"{col}.png"
                iterator.set_description(f"[{category}] {col}")
                success = create_regime_plot(df, col, plot_type, output_path, dpi)
                if success:
                    plots_created += 1
                else:
                    plots_skipped += 1
                iterator.update(1)

    iterator.close()

    print(f"\n✓ Visualization complete!")
    print(f"  Created: {plots_created} plots")
    if plots_skipped > 0:
        print(f"  Skipped: {plots_skipped} plots (no data or zero variance)")
    print(f"  Output directory: {output_dir.absolute()}")


def main():
    """Main entry point for batch summary visualizer."""
    parser = argparse.ArgumentParser(
        description='Visualize batch_summary.csv metrics grouped by melting regime',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (all columns, box plots, no filters)
  python vis/batch_summary_visualizer.py

  # Violin plots with custom config
  python vis/batch_summary_visualizer.py --plot-type violin --config my_filters.json

  # Both plot types overlaid
  python vis/batch_summary_visualizer.py --plot-type both

  # Specify CSV path explicitly
  python vis/batch_summary_visualizer.py --csv /path/to/batch_summary.csv

  # Quiet mode (minimal output)
  python vis/batch_summary_visualizer.py --quiet
"""
    )

    parser.add_argument('--csv', type=str, default=None,
                       help='Path to batch_summary.csv (default: auto-detect from hdf5 path)')
    parser.add_argument('--config', type=str, default='vis/batch_summary_config.json',
                       help='Path to filter config JSON file (default: vis/batch_summary_config.json)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same as CSV location)')
    parser.add_argument('--plot-type', choices=['box', 'violin', 'both'], default='box',
                       help='Plot type: box, violin, or both overlaid (default: box)')
    parser.add_argument('--dpi', type=int, default=600,
                       help='Figure DPI (default: 600)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress bar and detailed output')

    args = parser.parse_args()

    # Determine CSV path
    if args.csv is None:
        hdf5_path = Path(get_paths()['hdf5'])
        csv_path = hdf5_path / 'timeseries_compare_analysis_results' / 'run_summary' / 'batch_summary.csv'
    else:
        csv_path = Path(args.csv)

    # Determine output directory
    if args.output_dir is None:
        output_dir = csv_path.parent / 'regime_analysis'
    else:
        output_dir = Path(args.output_dir)

    try:
        # Load data
        print("=" * 80)
        print("BATCH SUMMARY VISUALIZER")
        print("=" * 80)
        print(f"\nLoading data from: {csv_path}")

        df = load_batch_summary(csv_path)
        logbook = get_logbook()

        # Merge regime data
        df = merge_regime_data(df, logbook, verbose=not args.quiet)

        # Load and apply filters
        filter_dict = {}
        config_path = Path(args.config)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                filter_dict = config.get('filters', {})
            except json.JSONDecodeError as e:
                print(f"\nWarning: Invalid JSON in config file: {e}")
                print("Proceeding with no filters")
        else:
            if not args.quiet:
                print(f"\nConfig file not found: {config_path}")
                print("Proceeding with no filters")

        df = apply_filters(df, logbook, filter_dict, verbose=not args.quiet)

        # Check if we have data after filtering
        if len(df) == 0:
            print("\nError: No data remaining after filtering!")
            return 1

        # Generate visualizations
        visualize_all_columns(df, output_dir, args.plot_type, args.dpi, args.quiet)

        print("\n" + "=" * 80)
        print("COMPLETE")
        print("=" * 80)

        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
