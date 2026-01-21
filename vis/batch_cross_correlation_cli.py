#!/usr/bin/env python
"""
Batch Cross-Correlation Analysis CLI

Performs statistical cross-correlation analysis across multiple HDF5 files.
Concatenates data from filtered tracks and analyzes correlations between signals.

Usage:
    python vis/batch_cross_correlation_cli.py --config config.json --output-dir results/
    python vis/batch_cross_correlation_cli.py --material AlSi10Mg --layer 1 --base-type powder

Author: AI Assistant
Date: 2025-01-15
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

# Silence joblib/loky CPU count warning on Windows
# This warning occurs when joblib cannot determine physical core count
os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 4))

import pandas as pd

# Add parent directory to path for tools import
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths, get_logbook, filter_logbook_tracks

from vis.timeseries.batch_cross_correlation import BatchCrossCorrelator
from vis.timeseries.config import DatasetConfig, ProcessingConfig


def load_config(config_path):
    """
    Load configuration from JSON file.

    Args:
        config_path (Path): Path to JSON config file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def get_filtered_hdf5_files(hdf5_dir, logbook, filter_dict, verbose=True):
    """
    Get list of HDF5 files matching filter criteria.

    Args:
        hdf5_dir (Path): Directory containing HDF5 files
        logbook (pd.DataFrame): Logbook dataframe
        filter_dict (dict): Filter criteria (see filter_logbook_tracks)
        verbose (bool): Print filtering statistics

    Returns:
        List[Path]: List of HDF5 file paths matching criteria
    """
    # Get all HDF5 files
    all_hdf5_files = sorted(hdf5_dir.glob('*.hdf5'))

    if not filter_dict or all(v is None for v in filter_dict.values()):
        if verbose:
            print(f"\nNo filters applied - using all {len(all_hdf5_files)} HDF5 files")
        return all_hdf5_files

    # Apply filters to logbook
    filtered_logbook, active_filters = filter_logbook_tracks(logbook, filter_dict)

    if verbose:
        print(f"\n{'=' * 80}")
        print("APPLYING FILTERS")
        print('=' * 80)
        print(f"Active filters: {active_filters}")
        print(f"Matched tracks in logbook: {len(filtered_logbook)}")

    # Extract trackids from filtered logbook
    filtered_trackids = set(filtered_logbook['trackid'].values)

    # Match HDF5 files to filtered trackids
    matched_files = []
    for hdf5_file in all_hdf5_files:
        trackid = hdf5_file.stem  # Filename without extension
        if trackid in filtered_trackids:
            matched_files.append(hdf5_file)

    if verbose:
        print(f"Matched HDF5 files: {len(matched_files)}")
        if len(matched_files) > 0:
            print(f"Sample trackids: {', '.join([f.stem for f in matched_files[:10]])}")
        else:
            print("WARNING: No HDF5 files matched the filter criteria!")
            print("Available HDF5 files:", ', '.join([f.stem for f in all_hdf5_files[:10]]))
            print("Filtered trackids:", ', '.join(sorted(list(filtered_trackids))[:10]))

    return matched_files


def create_dataset_configs_from_config(config):
    """
    Create DatasetConfig objects from configuration.

    Args:
        config (dict): Configuration dictionary

    Returns:
        List[DatasetConfig]: List of dataset configurations
    """
    dataset_configs = []

    for ds_config in config.get('datasets', []):
        dataset_configs.append(DatasetConfig(
            group=ds_config.get('group'),
            name=ds_config.get('name'),
            label=ds_config.get('label'),
            color=ds_config.get('color'),
            linestyle=ds_config.get('linestyle'),
            time_group=ds_config.get('time_group'),
            time_name=ds_config.get('time_name'),
            sampling_rate=ds_config.get('sampling_rate'),
            time_units=ds_config.get('time_units', 's')
        ))

    return dataset_configs


def create_processing_config_from_config(config):
    """
    Create ProcessingConfig from configuration.

    Args:
        config (dict): Configuration dictionary

    Returns:
        ProcessingConfig: Processing configuration object
    """
    proc_config = config.get('processing', {})

    return ProcessingConfig(
        # Normalization
        apply_normalization=proc_config.get('apply_normalization', False),
        normalization_method=proc_config.get('normalization_method', 'standard'),
        use_global_normalization=proc_config.get('use_global_normalization', False),

        # Outlier removal
        remove_outliers=proc_config.get('remove_outliers', False),
        outlier_method=proc_config.get('outlier_method', 'iqr'),
        outlier_threshold=proc_config.get('outlier_threshold', 3.0),
        outlier_window=proc_config.get('outlier_window', 50),

        # Filtering (if needed)
        apply_savgol=proc_config.get('apply_savgol', False),
        apply_lowpass=proc_config.get('apply_lowpass', False),
        apply_highpass=proc_config.get('apply_highpass', False),

        # Detrending
        apply_detrend=proc_config.get('apply_detrend', False),
        detrend_method=proc_config.get('detrend_method', 'linear')
    )


def main():
    """Main entry point for batch cross-correlation analysis."""
    parser = argparse.ArgumentParser(
        description='Batch cross-correlation analysis across multiple HDF5 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file
  python vis/batch_cross_correlation_cli.py --config config.json --output-dir results/

  # Using command-line filters
  python vis/batch_cross_correlation_cli.py --material AlSi10Mg --layer 1 --base-type powder

  # Filter by melting regime (keywords: conduction, keyhole, not_cond)
  python vis/batch_cross_correlation_cli.py --regime keyhole --output-dir keyhole_results/
  python vis/batch_cross_correlation_cli.py --regime not_cond --output-dir non_conduction/

  # Filter by exact regime name (use quotes for multi-word names)
  python vis/batch_cross_correlation_cli.py --regime "unstable keyhole" --output-dir unstable/

  # Filter by beamtime number (single or multiple)
  python vis/batch_cross_correlation_cli.py --beamtime 1 --output-dir beamtime1/
  python vis/batch_cross_correlation_cli.py --beamtime 1,2 --output-dir beamtimes_1_and_2/

  # Combining config and filters
  python vis/batch_cross_correlation_cli.py --config config.json --material AlSi10Mg --regime keyhole

  # Disable normalization to show raw data values
  python vis/batch_cross_correlation_cli.py --regime keyhole --no-normalize --output-dir keyhole_raw
        """
    )

    # Input/output arguments
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--output-dir', type=str, default='batch_cross_correlation_results',
                       help='Output directory for results (default: batch_cross_correlation_results)')

    # Filter arguments (override config)
    parser.add_argument('--material', type=str, help='Filter by material (e.g., AlSi10Mg)')
    parser.add_argument('--layer', type=int, help='Filter by layer number')
    parser.add_argument('--laser-mode', type=str, help='Filter by laser mode (CW, PW)')
    parser.add_argument('--base-type', type=str, help='Filter by base type (powder, weld, solid)')
    parser.add_argument('--regime', type=str, help='Filter by melting regime (keywords: conduction, keyhole, not_cond; or exact match like "unstable keyhole")')
    parser.add_argument('--beamtime', type=str, help='Filter by beamtime number(s) - single value (e.g., 1) or comma-separated (e.g., 1,2)')
    parser.add_argument('--travel-direction', type=str, help='Filter by travel direction')

    # Processing arguments
    parser.add_argument('--no-normalize', action='store_true', help='Disable normalization - show raw data values in plots')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')
    parser.add_argument('--quiet', action='store_true', help='Suppress non-essential output')

    args = parser.parse_args()

    # Determine verbosity
    verbose = args.verbose and not args.quiet

    # Load configuration
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"ERROR: Config file not found: {config_path}")
            sys.exit(1)
        config = load_config(config_path)
        if verbose:
            print(f"Loaded configuration from: {config_path}")
    else:
        # Default configuration
        config = {
            'datasets': [],
            'processing': {},
            'filters': {}
        }

    # Override filters from command line
    filters = config.get('filters', {})
    if args.material:
        filters['material'] = args.material
    if args.layer is not None:
        filters['layer'] = args.layer
    if args.laser_mode:
        filters['laser_mode'] = args.laser_mode
    if args.base_type:
        filters['base_type'] = args.base_type
    if args.regime:
        filters['regime'] = args.regime
    if args.beamtime:
        # Parse comma-separated beamtime values
        beamtime_str = args.beamtime.strip()
        if ',' in beamtime_str:
            # Multiple beamtimes: convert to list of integers
            filters['beamtime'] = [int(bt.strip()) for bt in beamtime_str.split(',')]
        else:
            # Single beamtime: convert to integer
            filters['beamtime'] = int(beamtime_str)
    if args.travel_direction:
        filters['travel_direction'] = args.travel_direction

    # Get paths and logbook
    paths = get_paths()
    hdf5_dir = Path(paths['hdf5'])
    logbook = get_logbook()

    if verbose:
        print(f"\n{'=' * 80}")
        print("BATCH CROSS-CORRELATION ANALYSIS")
        print('=' * 80)
        print(f"HDF5 directory: {hdf5_dir}")

    # Get filtered HDF5 files
    hdf5_files = get_filtered_hdf5_files(hdf5_dir, logbook, filters, verbose=verbose)

    if len(hdf5_files) == 0:
        print("\nERROR: No HDF5 files matched the filter criteria")
        print("Please adjust your filters or check that HDF5 files exist")
        sys.exit(1)

    # Create dataset configurations
    dataset_configs = create_dataset_configs_from_config(config)

    if len(dataset_configs) == 0:
        print("\nERROR: No datasets specified in configuration")
        print("Please add 'datasets' to your config file or use a default config")
        sys.exit(1)

    # Create processing configuration
    processing_config = create_processing_config_from_config(config)

    # Override normalization if --no-normalize flag is set
    if args.no_normalize:
        processing_config.apply_normalization = False
        if verbose:
            print("Note: Normalization disabled - showing raw data values")

    # Initialize batch cross-correlator
    correlator = BatchCrossCorrelator(
        hdf5_files=hdf5_files,
        datasets=dataset_configs,
        processing_config=processing_config,
        verbose=verbose
    )

    # Load and concatenate data
    correlator.load_and_concatenate_data()

    # Calculate statistics
    print(f"\n{'=' * 80}")
    print("CALCULATING STATISTICS")
    print('=' * 80)
    correlator.calculate_statistics()
    print(f"✓ Statistics calculated for {len(correlator.statistics)} signals")

    # Calculate correlations
    print(f"\n{'=' * 80}")
    print("CALCULATING CORRELATIONS")
    print('=' * 80)
    correlator.calculate_correlations()
    print(f"✓ Correlations calculated for {len(correlator.correlations)} signal pairs")

    # Calculate silhouette scores
    if len(correlator.processed_data) >= 2:
        print(f"\n{'=' * 80}")
        print("CALCULATING CLUSTERING QUALITY")
        print('=' * 80)
        correlator.calculate_silhouette_scores()
        print(f"✓ Silhouette scores calculated for {len(correlator.silhouette_scores)} signal pairs")

    # Generate report
    # Place output in the same parent directory as timeseries_compare uses
    output_parent = Path(hdf5_dir, 'timeseries_compare_analysis_results')
    if Path(args.output_dir).is_absolute():
        # If user provided absolute path, use it as-is
        output_dir = Path(args.output_dir)
    else:
        # Otherwise, place it as a subdirectory in timeseries_compare_analysis_results
        output_dir = Path(output_parent, args.output_dir)

    correlator.generate_report(output_dir=str(output_dir))

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print('=' * 80)
    print(f"Results saved to: {output_dir.absolute()}")
    print(f"\nGenerated files:")
    print(f"  - statistics_summary.png")
    print(f"  - correlation_matrix.png")
    if len(correlator.processed_data) >= 2:
        print(f"  - scatterplot_matrix.png")
        print(f"  - scatterplot_matrix_compact.png")
    print(f"  - batch_analysis_report.txt")
    print(f"  - batch_analysis_log.json")


if __name__ == '__main__':
    main()
