#!/usr/bin/env python
"""
Compute statistics for all datasets in HDF5 files.

This script iterates through all HDF5 files, calculates statistics
(min, max, mean, std, count) for each dataset, and saves them to a CSV log file.
Overall statistics are computed using weighted averages based on counts.

Author: AI Assistant
Date: 2025-12-10
"""

import os
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows compatibility
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from tools import get_paths


def compute_dataset_statistics(dataset):
    """
    Compute statistics for a single dataset.

    Parameters:
    -----------
    dataset : h5py.Dataset
        HDF5 dataset to analyze

    Returns:
    --------
    dict
        Dictionary containing statistics: min, max, mean, std, count
    """
    try:
        # Read dataset into memory
        data = dataset[:]

        # Flatten if multi-dimensional
        data_flat = data.flatten()

        # Remove NaN and inf values
        data_clean = data_flat[np.isfinite(data_flat)]

        if len(data_clean) == 0:
            # Handle empty or all-NaN datasets
            return {
                'min': np.nan,
                'max': np.nan,
                'mean': np.nan,
                'std': np.nan,
                'count': 0
            }

        # Compute statistics
        stats = {
            'min': float(np.min(data_clean)),
            'max': float(np.max(data_clean)),
            'mean': float(np.mean(data_clean)),
            'std': float(np.std(data_clean)),
            'count': int(len(data_clean))
        }

        return stats

    except Exception as e:
        print(f"Warning: Error computing statistics for dataset: {e}")
        return {
            'min': np.nan,
            'max': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'count': 0
        }


def process_hdf5_file(hdf5_path):
    """
    Process a single HDF5 file and extract statistics for all datasets.

    Parameters:
    -----------
    hdf5_path : str or Path
        Path to HDF5 file

    Returns:
    --------
    dict
        Dictionary mapping dataset paths to their statistics
    """
    file_stats = {}

    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Iterate through all datasets
            def collect_stats(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # Compute statistics for this dataset
                    stats = compute_dataset_statistics(obj)
                    # Use full path as key (e.g., "AMPM/Photodiode1Bits")
                    file_stats[name] = stats

            f.visititems(collect_stats)

    except Exception as e:
        print(f"Error processing {hdf5_path}: {e}")

    return file_stats


def compute_overall_statistics(all_stats_df):
    """
    Compute overall statistics using weighted averages.

    For mean: weighted average using counts
    For std: combined standard deviation using counts
    For min/max: global min/max across all tracks

    Parameters:
    -----------
    all_stats_df : pd.DataFrame
        DataFrame with trackids as rows and dataset_stat columns

    Returns:
    --------
    pd.Series
        Series containing overall statistics
    """
    overall_stats = {}

    # Get unique dataset names (extract from column names like "AMPM/Photodiode1Bits_mean")
    dataset_names = set()
    for col in all_stats_df.columns:
        if col == 'trackid':
            continue
        # Extract dataset name (everything before the last underscore)
        dataset_name = '_'.join(col.split('_')[:-1])
        dataset_names.add(dataset_name)

    # Compute overall stats for each dataset
    for dataset in dataset_names:
        mean_col = f'{dataset}_mean'
        std_col = f'{dataset}_std'
        min_col = f'{dataset}_min'
        max_col = f'{dataset}_max'
        count_col = f'{dataset}_count'

        # Check if columns exist
        if count_col not in all_stats_df.columns:
            continue

        # Get counts and valid data
        counts = all_stats_df[count_col]
        valid_mask = (counts > 0) & (counts.notna())

        if not valid_mask.any():
            # No valid data for this dataset
            overall_stats[mean_col] = np.nan
            overall_stats[std_col] = np.nan
            overall_stats[min_col] = np.nan
            overall_stats[max_col] = np.nan
            overall_stats[count_col] = 0
            continue

        # Total count
        total_count = counts[valid_mask].sum()
        overall_stats[count_col] = int(total_count)

        # Global min and max
        if min_col in all_stats_df.columns:
            overall_stats[min_col] = all_stats_df[min_col][valid_mask].min()
        else:
            overall_stats[min_col] = np.nan

        if max_col in all_stats_df.columns:
            overall_stats[max_col] = all_stats_df[max_col][valid_mask].max()
        else:
            overall_stats[max_col] = np.nan

        # Weighted mean
        if mean_col in all_stats_df.columns:
            means = all_stats_df[mean_col][valid_mask]
            weights = counts[valid_mask]
            weighted_mean = (means * weights).sum() / total_count
            overall_stats[mean_col] = weighted_mean
        else:
            overall_stats[mean_col] = np.nan

        # Combined standard deviation
        # Formula: σ_total = sqrt( Σ(n_i * (σ_i^2 + (μ_i - μ_total)^2)) / Σ(n_i) )
        if std_col in all_stats_df.columns and mean_col in all_stats_df.columns:
            stds = all_stats_df[std_col][valid_mask]
            means = all_stats_df[mean_col][valid_mask]
            weights = counts[valid_mask]

            # Combined variance
            variances = stds ** 2
            mean_diffs_sq = (means - weighted_mean) ** 2
            combined_variance = (weights * (variances + mean_diffs_sq)).sum() / total_count
            overall_stats[std_col] = np.sqrt(combined_variance)
        else:
            overall_stats[std_col] = np.nan

    overall_stats['trackid'] = 'OVERALL'
    return pd.Series(overall_stats)


def main():
    """Main function to compute and save HDF5 statistics"""
    print("=" * 80)
    print("HDF5 Dataset Statistics Computation")
    print("=" * 80)

    # Get HDF5 directory from paths
    paths = get_paths()
    hdf5_dir = paths['hdf5']

    print(f"\nHDF5 directory: {hdf5_dir}")

    # Find all HDF5 files
    hdf5_files = sorted(hdf5_dir.glob('*.hdf5'))

    if not hdf5_files:
        print(f"Error: No HDF5 files found in {hdf5_dir}")
        return

    print(f"Found {len(hdf5_files)} HDF5 files")

    # Collect statistics for all files
    all_stats = []

    print("\nProcessing HDF5 files...")
    for hdf5_file in tqdm(hdf5_files, desc="Computing statistics", unit="file"):
        trackid = hdf5_file.stem

        # Process file
        file_stats = process_hdf5_file(hdf5_file)

        # Flatten statistics into a single row
        row_data = {'trackid': trackid}
        for dataset_name, stats in file_stats.items():
            for stat_name, stat_value in stats.items():
                col_name = f'{dataset_name}_{stat_name}'
                row_data[col_name] = stat_value

        all_stats.append(row_data)

    # Create DataFrame
    print("\nCreating statistics dataframe...")
    df = pd.DataFrame(all_stats)

    # Compute overall statistics
    print("Computing overall statistics...")
    overall_row = compute_overall_statistics(df)

    # Append overall row
    df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)

    # Sort columns: trackid first, then alphabetically
    cols = ['trackid'] + sorted([c for c in df.columns if c != 'trackid'])
    df = df[cols]

    # Save to CSV
    output_path = hdf5_dir / 'hdf5_dataset_statistics.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\n✓ Statistics saved to: {output_path}")
    print(f"✓ Total rows: {len(df)} (including OVERALL)")
    print(f"✓ Total columns: {len(df.columns)}")

    # Display summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Individual tracks: {len(df) - 1}")
    print(f"Datasets analyzed: {len([c for c in df.columns if c.endswith('_count')])}")
    print("\nFirst few columns of OVERALL statistics:")

    # Show a sample of overall statistics
    overall_series = df[df['trackid'] == 'OVERALL'].iloc[0]
    sample_cols = [c for c in cols[:20] if c != 'trackid']  # Show first 19 columns
    for col in sample_cols:
        value = overall_series[col]
        if pd.notna(value):
            print(f"  {col}: {value:.4f}" if isinstance(value, float) else f"  {col}: {value}")

    print("\n✓ Done!")


if __name__ == '__main__':
    main()
