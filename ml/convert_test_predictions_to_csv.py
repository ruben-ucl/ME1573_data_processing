#!/usr/bin/env python3
"""Convert test predictions PKL to CSV with detailed window information"""

import argparse
import pickle
import pandas as pd
import re
from pathlib import Path

def extract_window_info(filepath):
    """
    Extract track ID, window index, and time information from filename.

    Example filename: 0105_06_0.2-1.2ms.png
    Returns: track_id='0105_06', window_start=0.2, window_end=1.2
    """
    filename = Path(filepath).name

    # Remove extension
    filename_no_ext = filename.replace('.png', '')

    # Pattern: trackid_window_info
    # Example: 0105_06_0.2-1.2ms
    parts = filename_no_ext.split('_')

    if len(parts) >= 3:
        # Track ID is first two parts (e.g., "0105_06")
        track_id = f"{parts[0]}_{parts[1]}"

        # Window info is remaining parts joined (e.g., "0.2-1.2ms")
        window_str = '_'.join(parts[2:])

        # Extract time range: "0.2-1.2ms" -> start=0.2, end=1.2
        time_match = re.search(r'([\d.]+)-([\d.]+)ms', window_str)
        if time_match:
            window_start = float(time_match.group(1))
            window_end = float(time_match.group(2))
        else:
            window_start = None
            window_end = None

        return track_id, window_start, window_end
    else:
        # Fallback if pattern doesn't match
        return filename_no_ext, None, None

def convert_pkl_to_csv(pkl_path, output_csv_path):
    """Convert test predictions PKL to CSV with detailed columns."""

    print(f"Loading: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Extract data
    y_true = data['y_true']
    y_pred = data['y_pred']
    y_proba = data['y_proba']
    test_files = data['test_files']
    best_threshold = data.get('best_threshold', 0.5)

    print(f"  Loaded {len(test_files)} test samples")
    print(f"  Best threshold: {best_threshold:.4f}")

    # Build DataFrame
    records = []
    for i, filepath in enumerate(test_files):
        track_id, window_start, window_end = extract_window_info(filepath)

        # Calculate window index for this track (count samples with same track_id up to this point)
        window_index = sum(1 for j in range(i+1) if extract_window_info(test_files[j])[0] == track_id) - 1

        record = {
            'trackid': track_id,
            'window_index': window_index,
            'window_start_ms': window_start,
            'window_end_ms': window_end,
            'actual_label': int(y_true[i]),
            'predicted_label': int(y_pred[i]),
            'predicted_probability': float(y_proba[i]),
            'filename': Path(filepath).name
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Sort by track ID and window start time
    df = df.sort_values(['trackid', 'window_start_ms']).reset_index(drop=True)

    # Save to CSV
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"\n✅ Saved to: {output_csv_path}")

    # Print summary statistics
    print(f"\n📊 Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Unique tracks: {df['trackid'].nunique()}")
    print(f"  Actual labels distribution:")
    print(f"    Class 0 (No porosity): {(df['actual_label']==0).sum()}")
    print(f"    Class 1 (Porosity): {(df['actual_label']==1).sum()}")
    print(f"  Prediction accuracy: {(df['actual_label']==df['predicted_label']).mean():.4f}")

    # Show first few rows
    print(f"\n📋 First 10 rows:")
    print(df.head(10).to_string(index=False))

    return df

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Convert test predictions PKL to CSV with detailed window information',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_test_predictions_to_csv.py --version v206
  python convert_test_predictions_to_csv.py --version 206
  python convert_test_predictions_to_csv.py -v v208
        """
    )
    parser.add_argument(
        '--version', '-v',
        required=True,
        help='Model version to process (e.g., v206 or 206)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Custom output CSV path (optional, defaults to test_evaluation folder)'
    )

    args = parser.parse_args()

    # Normalize version (add 'v' prefix if not present)
    version = args.version if args.version.startswith('v') else f'v{args.version}'

    # Set paths
    base_dir = Path('D:/ME1573_data_processing/ml/outputs/cwt')
    pkl_path = base_dir / version / 'test_evaluation' / f'test_predictions_{version}.pkl'

    # Check if PKL file exists
    if not pkl_path.exists():
        print(f"❌ Error: PKL file not found: {pkl_path}")
        print(f"\nPlease check that:")
        print(f"  1. Version {version} exists")
        print(f"  2. Test evaluation has been run for this version")
        exit(1)

    # Set output path
    if args.output:
        output_csv = Path(args.output)
    else:
        output_csv = base_dir / version / 'test_evaluation' / f'test_predictions_{version}_detailed.csv'

    # Convert
    df = convert_pkl_to_csv(pkl_path, output_csv)

    print(f"\n" + "="*60)
    print(f"Conversion complete!")
    print(f"Output file: {output_csv}")
