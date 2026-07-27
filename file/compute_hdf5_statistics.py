#!/usr/bin/env python
"""
Compute per-track statistics for 1D datasets in HDF5 files.

Iterates through all HDF5 files, calculates statistics for each 1D dataset,
and saves them to a CSV log file. Multi-dimensional datasets are skipped.

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
from scipy import stats
import pandas as pd
from tqdm import tqdm
from tools import get_paths


def compute_dataset_statistics(dataset, crop=0):
    """
    Compute statistics for a single dataset.

    Parameters:
    -----------
    dataset : h5py.Dataset
        HDF5 dataset to analyze
    crop : int
        Number of datapoints to remove from each end before computing statistics

    Returns:
    --------
    dict
        Dictionary containing statistics: min, max, mean, std, count
    """
    try:
        # Read dataset into memory
        data = dataset[:]

        # Crop ends if requested (e.g. to remove transient effects)
        if crop > 0:
            if len(data) <= 2 * crop:
                raise ValueError(f"Dataset length {len(data)} is too short to crop {crop} points from each end")
            data = data[crop:-crop]

        # Flatten if multi-dimensional
        data_flat = data.flatten()

        # Remove NaN and inf values
        data_clean = data_flat[np.isfinite(data_flat)]

        if len(data_clean) == 0:
            # Handle empty or all-NaN datasets
            return {
                'count': 0,
                'min': np.nan,
                'max': np.nan,
                'mean': np.nan,
                'std': np.nan,
                'se': np.nan,
                'med': np.nan,
                'q25': np.nan,
                'q75': np.nan,
                'q90': np.nan,
                'iqr': np.nan,
                'skew': np.nan
            }

        # Compute statistics
        result = {
            'count': int(len(data_clean)),
            'min': float(np.min(data_clean)),
            'max': float(np.max(data_clean)),
            'mean': float(np.mean(data_clean)),
            'std': float(np.std(data_clean)),
            'se': float(stats.sem(data_clean)),
            'med': float(np.median(data_clean)),
            'q25': float(np.percentile(data_clean, 25)),
            'q75': float(np.percentile(data_clean, 75)),
            'q90': float(np.percentile(data_clean, 90)),
            'iqr': float(stats.iqr(data_clean)),
            'skew': float(stats.skew(data_clean))
        }

        return result

    except Exception as e:
        print(f"Warning: Error computing statistics for dataset: {e}")
        return {
            'count': 0,
            'min': np.nan,
            'max': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'se': np.nan,
            'med': np.nan,
            'q25': np.nan,
            'q75': np.nan,
            'q90': np.nan,
            'iqr': np.nan,
            'skew': np.nan
        }


def process_hdf5_file(hdf5_path):
    """
    Process a single HDF5 file and extract statistics for 1D datasets only.

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
                    # Skip multi-dimensional datasets (images, timeseries images)
                    if obj.ndim != 1:
                        return
                    # Compute statistics for this dataset; crop AMPM transient ends
                    crop = 500 if name.startswith('AMPM/') else 0
                    stats = compute_dataset_statistics(obj, crop=crop)
                    # Use full path as key (e.g., "AMPM/Photodiode1Bits")
                    file_stats[name] = stats

            f.visititems(collect_stats)

    except Exception as e:
        print(f"Error processing {hdf5_path}: {e}")

    return file_stats



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

    # Sort columns: trackid first, KH/ group before AMPM/, datasets alphabetically within each
    # group, stat suffixes in definition order within each dataset
    _stat_order = ['count', 'min', 'max', 'mean', 'std', 'se', 'med', 'q25', 'q75', 'q90', 'iqr', 'skew']

    def _col_key(col):
        group = 0 if col.startswith('KH/') else (1 if col.startswith('AMPM/') else 2)
        last_under = col.rfind('_')
        if last_under >= 0:
            dataset, stat = col[:last_under], col[last_under + 1:]
            stat_idx = _stat_order.index(stat) if stat in _stat_order else len(_stat_order)
        else:
            dataset, stat_idx = col, len(_stat_order)
        return (group, dataset, stat_idx)

    cols = ['trackid'] + sorted([c for c in df.columns if c != 'trackid'], key=_col_key)
    df = df[cols]

    # Save to CSV
    output_path = hdf5_dir / 'hdf5_dataset_statistics_extended.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\n✓ Statistics saved to: {output_path}")
    print(f"✓ Tracks processed: {len(df)}")
    print(f"✓ Total columns: {len(df.columns)}")

    # Display summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Tracks: {len(df)}")
    print(f"Datasets analyzed: {len([c for c in df.columns if c.endswith('_count')])}")

    print("\n✓ Done!")


if __name__ == '__main__':
    main()
