#!/usr/bin/env python
"""
Minimal test for batch cross-correlation to verify fix works.
Tests with just 2 files to keep execution time short.
"""

import os
import sys
from pathlib import Path

# Silence joblib/loky CPU count warning on Windows
os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 4))

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from tools import get_paths, get_logbook, filter_logbook_tracks
from vis.timeseries.batch_cross_correlation import BatchCrossCorrelator
from vis.timeseries.config import DatasetConfig, ProcessingConfig

def main():
    print("=" * 80)
    print("MINIMAL BATCH CROSS-CORRELATION TEST")
    print("=" * 80)

    # Get paths and logbook
    paths = get_paths()
    hdf5_dir = Path(paths['hdf5'])
    logbook = get_logbook()

    # Filter for keyhole regime (includes all keyhole variants)
    filters = {'material': 'AlSi10Mg', 'layer': 1, 'base_type': 'powder', 'regime': 'keyhole'}
    filtered_logbook, _ = filter_logbook_tracks(logbook, filters)

    # Get just first 2 trackids
    trackids = filtered_logbook['trackid'].head(2).tolist()
    print(f"\nTesting with {len(trackids)} files: {trackids}")

    # Find corresponding HDF5 files
    hdf5_files = []
    for trackid in trackids:
        hdf5_path = hdf5_dir / f"{trackid}.hdf5"
        if hdf5_path.exists():
            hdf5_files.append(hdf5_path)
        else:
            print(f"WARNING: File not found: {hdf5_path}")

    print(f"Found {len(hdf5_files)} HDF5 files")

    if len(hdf5_files) == 0:
        print("ERROR: No HDF5 files found")
        sys.exit(1)

    # Define datasets
    datasets = [
        DatasetConfig(group='AMPM', name='Photodiode1Bits', label='PD1',
                     time_group='AMPM', time_name='Time', time_units='s'),
        DatasetConfig(group='AMPM', name='Photodiode2Bits', label='PD2',
                     time_group='AMPM', time_name='Time', time_units='s'),
        DatasetConfig(group='KH', name='max_depth', label='KH_depth',
                     time_group='KH', time_name='time', time_units='s'),
        DatasetConfig(group='KH', name='area', label='KH_area',
                     time_group='KH', time_name='time', time_units='s'),
    ]

    # Processing config (minimal for speed)
    processing_config = ProcessingConfig(
        apply_normalization=True,
        normalization_method='minmax',
        remove_outliers=False,  # Disable for speed
        apply_detrend=False
    )

    # Initialize correlator
    print("\n" + "=" * 80)
    print("INITIALIZING BATCH CORRELATOR")
    print("=" * 80)
    correlator = BatchCrossCorrelator(
        hdf5_files=hdf5_files,
        datasets=datasets,
        processing_config=processing_config,
        verbose=True
    )

    # Load and concatenate
    print("\n" + "=" * 80)
    print("LOADING AND CONCATENATING DATA")
    print("=" * 80)
    correlator.load_and_concatenate_data()

    # Calculate statistics
    print("\n" + "=" * 80)
    print("CALCULATING STATISTICS")
    print("=" * 80)
    correlator.calculate_statistics()
    print(f"✓ Statistics calculated for {len(correlator.statistics)} signals")

    # Calculate correlations (this is where the bug was)
    print("\n" + "=" * 80)
    print("CALCULATING CORRELATIONS (TESTING FIX)")
    print("=" * 80)
    try:
        correlator.calculate_correlations()
        print(f"✓ SUCCESS! Correlations calculated for {len(correlator.correlations)} signal pairs")
    except TypeError as e:
        print(f"✗ FAILED: {e}")
        sys.exit(1)

    # Generate report
    output_dir = Path('test_batch_minimal_output')
    correlator.generate_report(output_dir=str(output_dir))

    print("\n" + "=" * 80)
    print("TEST COMPLETE - FIX VERIFIED")
    print("=" * 80)
    print(f"Results saved to: {output_dir.absolute()}")

if __name__ == '__main__':
    main()
