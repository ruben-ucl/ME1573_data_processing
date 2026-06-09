#!/usr/bin/env python
"""
Scatter plot of HDF5 statistics merged with logbook data.

This script reads the HDF5 dataset statistics CSV, merges it with
the logbook by trackid, filters the data, and plots a scatter plot.
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

import matplotlib.pyplot as plt
import pandas as pd
from tools import get_paths, get_logbook, filter_logbook_tracks

# =============================================================================
# CONFIGURATION - Edit these values as needed
# =============================================================================

# Path to statistics CSV (relative to hdf5 directory)
STATS_FILENAME = 'hdf5_dataset_statistics.csv'

# Filter dictionary for subset selection
# See tools.filter_logbook_tracks for available filters
FILTERS = {
    'material': 'AlSi10Mg',
    'layer': 1,
    'laser_mode': 'cw',
    'base_type': 'powder',
    'beamtime': 3
}

# Column names for scatter plot
X_COLUMN = 'LED [J/m]'              # Column from logbook
Y_COLUMN = 'AMPM/Photodiode1Bits_mean'  # Column from stats CSV

# Axis labels
X_LABEL = 'LED [J/m]'
Y_LABEL = 'Photodiode 1 Mean [bits]'

# Plot title
PLOT_TITLE = 'Photodiode Signal vs LED'

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    """Main function to create scatter plot."""
    print("=" * 60)
    print("Scatter Plot: HDF5 Statistics vs Logbook Data")
    print("=" * 60)

    # Get paths
    paths = get_paths()
    hdf5_dir = paths['hdf5']
    stats_path = hdf5_dir / STATS_FILENAME

    # Read statistics CSV
    print(f"\nReading statistics from: {stats_path}")
    if not stats_path.exists():
        print(f"Error: Statistics file not found: {stats_path}")
        print("Run file/compute_hdf5_statistics.py first.")
        return

    stats_df = pd.read_csv(stats_path, encoding='utf-8')

    # Remove OVERALL row if present
    stats_df = stats_df[stats_df['trackid'] != 'OVERALL']
    print(f"  Loaded {len(stats_df)} tracks from statistics")

    # Read logbook
    logbook = get_logbook()
    print(f"  Loaded {len(logbook)} rows from logbook")

    # Merge dataframes on trackid
    print("\nMerging statistics with logbook...")
    merged_df = pd.merge(
        stats_df,
        logbook,
        on='trackid',
        how='inner'
    )
    print(f"  Merged dataset: {len(merged_df)} rows")

    # Apply filters
    print(f"\nApplying filters: {FILTERS}")
    filtered_df, active_filters = filter_logbook_tracks(merged_df, FILTERS)
    print(f"  Active filters: {active_filters}")
    print(f"  Filtered dataset: {len(filtered_df)} rows")

    if len(filtered_df) == 0:
        print("\nError: No data remaining after filtering.")
        return

    # Check columns exist
    if X_COLUMN not in filtered_df.columns:
        print(f"\nError: X column '{X_COLUMN}' not found.")
        print(f"Available columns: {list(filtered_df.columns)}")
        return

    if Y_COLUMN not in filtered_df.columns:
        print(f"\nError: Y column '{Y_COLUMN}' not found.")
        print(f"Available columns: {list(filtered_df.columns)}")
        return

    # Extract data
    x_data = filtered_df[X_COLUMN]
    y_data = filtered_df[Y_COLUMN]

    # Remove NaN values
    valid_mask = x_data.notna() & y_data.notna()
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]
    print(f"  Valid data points: {len(x_data)}")

    # Create scatter plot
    print("\nCreating scatter plot...")
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(x_data, y_data, alpha=0.7, edgecolors='black', linewidth=0.5)

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(PLOT_TITLE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nDone!")


if __name__ == '__main__':
    main()
