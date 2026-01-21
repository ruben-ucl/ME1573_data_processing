"""
Extract labels from HDF5 timeseries data for CWT images.

This script:
1. Scans a directory of CWT images
2. Parses filenames to extract trackid, window_start_ms, window_end_ms
3. Loads corresponding HDF5 timeseries data
4. Extracts single value (max/mean) from the time slice
5. Writes to label CSV

Usage:
    python ml/extract_labels_from_hdf5.py \
        --data_dir "path/to/cwt/images" \
        --hdf5_dir "path/to/hdf5/files" \
        --label_columns "keyhole_depth,melt_pool_width" \
        --aggregation max \
        --output_csv "labels.csv"
"""

import os
import sys
from pathlib import Path
import re
import argparse
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

# Set UTF-8 encoding for Windows compatibility
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools import get_paths


def parse_image_filename(filename):
    """
    Parse CWT image filename to extract metadata.

    Expected format: trackid_start-end_ms.png
    Example: "240213_13-23-24_0.2-1.2ms.png"

    Returns:
        dict: {'trackid': str, 'window_start_ms': float, 'window_end_ms': float}
        None if parsing fails
    """
    # Remove extension
    name = Path(filename).stem

    # Pattern: trackid_start-endms
    # trackid can contain hyphens, so we look for the last occurrence of _
    # followed by decimal-decimal pattern and ms
    pattern = r'^(.+)_([\d.]+)-([\d.]+)ms$'
    match = re.match(pattern, name)

    if not match:
        return None

    trackid = match.group(1)
    window_start = float(match.group(2))
    window_end = float(match.group(3))

    return {
        'trackid': trackid,
        'window_start_ms': window_start,
        'window_end_ms': window_end
    }


def find_hdf5_file(trackid, hdf5_dir):
    """
    Find the HDF5 file corresponding to a trackid.

    Args:
        trackid: Track identifier
        hdf5_dir: Directory containing HDF5 files

    Returns:
        Path to HDF5 file or None if not found
    """
    hdf5_path = Path(hdf5_dir)

    # Try exact match first
    exact_match = hdf5_path / f"{trackid}.h5"
    if exact_match.exists():
        return exact_match

    # Try with different extensions
    for ext in ['.hdf5', '.h5']:
        candidate = hdf5_path / f"{trackid}{ext}"
        if candidate.exists():
            return candidate

    # Search for files containing the trackid
    for h5_file in hdf5_path.glob('*.h5'):
        if trackid in h5_file.stem:
            return h5_file

    for h5_file in hdf5_path.glob('*.hdf5'):
        if trackid in h5_file.stem:
            return h5_file

    return None


def extract_label_from_hdf5(h5_file, window_start_ms, window_end_ms,
                             label_column, aggregation='max'):
    """
    Extract a single label value from HDF5 timeseries data.

    Args:
        h5_file: Path to HDF5 file
        window_start_ms: Window start time in milliseconds
        window_end_ms: Window end time in milliseconds
        label_column: Name of the dataset/column to extract from
        aggregation: 'max' or 'mean'

    Returns:
        float: Extracted label value or np.nan if extraction fails
    """
    try:
        with h5py.File(h5_file, 'r') as f:
            # Check if label_column exists
            if label_column not in f:
                print(f"Warning: Column '{label_column}' not found in {h5_file}")
                return np.nan

            # Load data column
            data = f[label_column][:]

            # Determine which time array to use based on the data group
            t = None
            t_ms = None

            if label_column.startswith('KH/'):
                # Keyhole data uses KH/time (in seconds)
                if 'KH/time' in f:
                    t = f['KH/time'][:]
                    t_ms = t * 1000  # Convert seconds to milliseconds
                else:
                    print(f"Warning: KH/time not found for {label_column}")
                    return np.nan
            elif label_column.startswith('AMPM/'):
                # AMPM data uses AMPM/Time
                if 'AMPM/Time' in f:
                    t = f['AMPM/Time'][:]
                    t_ms = t * 1000  # Convert seconds to milliseconds
                else:
                    print(f"Warning: AMPM/Time not found for {label_column}")
                    return np.nan
            else:
                # Try common time array names
                if 't' in f:
                    t = f['t'][:]
                    t_ms = t * 1000  # Convert to milliseconds
                elif 'time' in f:
                    t = f['time'][:]
                    t_ms = t * 1000  # Assume seconds
                else:
                    print(f"Warning: No time array found for {label_column}")
                    return np.nan

            # Check if data and time arrays have same length
            if len(data) != len(t):
                print(f"Warning: Data length ({len(data)}) != time length ({len(t)}) for {label_column}")
                return np.nan

            # Find indices for time window
            start_idx = np.argmin(np.abs(t_ms - window_start_ms))
            end_idx = np.argmin(np.abs(t_ms - window_end_ms))

            # Extract window slice
            if start_idx >= end_idx:
                print(f"Warning: Invalid time window {window_start_ms}-{window_end_ms}ms")
                return np.nan

            data_slice = data[start_idx:end_idx]

            if len(data_slice) == 0:
                print(f"Warning: Empty data slice for window {window_start_ms}-{window_end_ms}ms")
                return np.nan

            # Apply aggregation
            if aggregation == 'max':
                return float(np.max(data_slice))
            elif aggregation == 'mean':
                return float(np.mean(data_slice))
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

    except Exception as e:
        print(f"Error extracting label from {h5_file}: {e}")
        return np.nan


def scan_images_and_extract_labels(data_dir, hdf5_dir, label_columns,
                                   aggregation='max', verbose=False):
    """
    Scan directory of CWT images and extract labels from HDF5 files.

    Args:
        data_dir: Directory containing CWT images
        hdf5_dir: Directory containing HDF5 files
        label_columns: List of column names to extract
        aggregation: 'max' or 'mean'
        verbose: Print detailed progress

    Returns:
        pd.DataFrame: Label data with columns [image_filename, trackid,
                      window_start_ms, window_end_ms, <label_columns>]
    """
    data_path = Path(data_dir)
    hdf5_path = Path(hdf5_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 directory not found: {hdf5_dir}")

    # Find all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(data_path.rglob(ext))

    if verbose:
        print(f"Found {len(image_files)} images in {data_dir}")

    # Parse filenames and build label dataframe
    records = []

    for img_file in tqdm(image_files, desc="Extracting labels"):
        # Parse filename
        metadata = parse_image_filename(img_file.name)
        if metadata is None:
            if verbose:
                print(f"Warning: Could not parse filename: {img_file.name}")
            continue

        # Find corresponding HDF5 file
        h5_file = find_hdf5_file(metadata['trackid'], hdf5_path)
        if h5_file is None:
            if verbose:
                print(f"Warning: HDF5 file not found for trackid: {metadata['trackid']}")
            continue

        # Extract labels for each column
        record = {
            'image_filename': img_file.name,
            'trackid': metadata['trackid'],
            'window_start_ms': metadata['window_start_ms'],
            'window_end_ms': metadata['window_end_ms']
        }

        for label_col in label_columns:
            label_value = extract_label_from_hdf5(
                h5_file,
                metadata['window_start_ms'],
                metadata['window_end_ms'],
                label_col,
                aggregation
            )
            record[label_col] = label_value

        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    if verbose:
        print(f"\nExtracted {len(df)} labels")
        print(f"Columns: {list(df.columns)}")
        print(f"\nSample data:")
        print(df.head())

        # Check for missing values
        for col in label_columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                print(f"Warning: {n_missing} missing values in column '{col}'")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Extract labels from HDF5 timeseries data for CWT images'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing CWT images')
    parser.add_argument('--hdf5_dir', type=str, required=True,
                       help='Directory containing HDF5 timeseries files')
    parser.add_argument('--label_columns', type=str, required=True,
                       help='Comma-separated list of HDF5 columns to extract (e.g., "depth,width")')
    parser.add_argument('--aggregation', type=str, default='max',
                       choices=['max', 'mean'],
                       help='Aggregation method for time window (default: max)')
    parser.add_argument('--output_csv', type=str, required=True,
                       help='Output CSV file path')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress information')

    args = parser.parse_args()

    # Parse label columns
    label_columns = [col.strip() for col in args.label_columns.split(',')]

    print(f"Extracting labels from HDF5 files...")
    print(f"  Data directory: {args.data_dir}")
    print(f"  HDF5 directory: {args.hdf5_dir}")
    print(f"  Label columns: {label_columns}")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  Output: {args.output_csv}")

    # Extract labels
    df = scan_images_and_extract_labels(
        args.data_dir,
        args.hdf5_dir,
        label_columns,
        args.aggregation,
        args.verbose
    )

    # Save to CSV
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\n✓ Saved {len(df)} labels to {output_path}")

    # Print statistics
    print("\nLabel statistics:")
    for col in label_columns:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Min: {df[col].min():.4f}")
            print(f"  Max: {df[col].max():.4f}")
            print(f"  Mean: {df[col].mean():.4f}")
            print(f"  Std: {df[col].std():.4f}")
            print(f"  Missing: {df[col].isna().sum()}")


if __name__ == '__main__':
    main()
