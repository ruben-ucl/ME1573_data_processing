"""
Migrate existing folder-based binary labels to unified CSV format.

This script scans directories with folder-based labels (e.g., good/, bad/)
and creates a unified label CSV matching the new schema.

Usage:
    python ml/migrate_binary_labels_to_csv.py \
        --data_dir "path/to/labeled/cwt/images" \
        --label_column_name "has_keyhole" \
        --folder_mapping "keyhole=1,no_keyhole=0" \
        --output_csv "binary_labels.csv"
"""

import os
import sys
from pathlib import Path
import re
import argparse
import pandas as pd
from tqdm import tqdm

# Set UTF-8 encoding for Windows compatibility
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_image_filename(filename):
    """
    Parse CWT image filename to extract metadata.

    Expected format: trackid_start-endms.png
    Example: "240213_13-23-24_0.2-1.2ms.png"

    Returns:
        dict: {'trackid': str, 'window_start_ms': float, 'window_end_ms': float}
        None if parsing fails
    """
    # Remove extension
    name = Path(filename).stem

    # Pattern: trackid_start-endms
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


def scan_labeled_folders(data_dir, folder_mapping, verbose=False):
    """
    Scan folder-based labeled data and extract labels.

    Args:
        data_dir: Root directory containing label folders
        folder_mapping: Dict mapping folder names to label values
        verbose: Print detailed progress

    Returns:
        pd.DataFrame: Label data with columns [image_filename, trackid,
                      window_start_ms, window_end_ms, label_value]
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    records = []

    # Scan each mapped folder
    for folder_name, label_value in folder_mapping.items():
        folder_path = data_path / folder_name

        if not folder_path.exists():
            if verbose:
                print(f"Warning: Folder '{folder_name}' not found in {data_dir}")
            continue

        # Find all images in this folder
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(folder_path.rglob(ext))

        if verbose:
            print(f"Found {len(image_files)} images in {folder_name}/ (label={label_value})")

        # Process each image
        for img_file in tqdm(image_files, desc=f"Processing {folder_name}", disable=not verbose):
            # Parse filename
            metadata = parse_image_filename(img_file.name)
            if metadata is None:
                if verbose:
                    print(f"Warning: Could not parse filename: {img_file.name}")
                continue

            record = {
                'image_filename': img_file.name,
                'trackid': metadata['trackid'],
                'window_start_ms': metadata['window_start_ms'],
                'window_end_ms': metadata['window_end_ms'],
                'label_value': label_value
            }

            records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    if verbose and len(df) > 0:
        print(f"\nMigrated {len(df)} labels")
        print(f"\nLabel distribution:")
        print(df['label_value'].value_counts().sort_index())

    return df


def parse_folder_mapping(mapping_str):
    """
    Parse folder mapping string.

    Args:
        mapping_str: String like "keyhole=1,no_keyhole=0,uncertain=0.5"

    Returns:
        dict: Mapping of folder names to label values
    """
    mapping = {}

    for pair in mapping_str.split(','):
        if '=' not in pair:
            raise ValueError(f"Invalid mapping format: {pair}")

        folder, value = pair.split('=', 1)
        folder = folder.strip()

        # Try to parse as number
        try:
            value = float(value.strip())
        except ValueError:
            # Keep as string for categorical labels
            value = value.strip()

        mapping[folder] = value

    return mapping


def main():
    parser = argparse.ArgumentParser(
        description='Migrate folder-based binary labels to unified CSV format'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Root directory containing label folders')
    parser.add_argument('--label_column_name', type=str, required=True,
                       help='Name for the label column (e.g., "has_keyhole")')
    parser.add_argument('--folder_mapping', type=str, required=True,
                       help='Folder to label mapping (e.g., "keyhole=1,no_keyhole=0")')
    parser.add_argument('--output_csv', type=str, required=True,
                       help='Output CSV file path')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress information')

    args = parser.parse_args()

    # Parse folder mapping
    try:
        folder_mapping = parse_folder_mapping(args.folder_mapping)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"Migrating folder-based labels to CSV...")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Label column: {args.label_column_name}")
    print(f"  Folder mapping: {folder_mapping}")
    print(f"  Output: {args.output_csv}")

    # Scan and migrate labels
    df = scan_labeled_folders(
        args.data_dir,
        folder_mapping,
        args.verbose
    )

    if len(df) == 0:
        print("\nWarning: No labeled images found!")
        return 1

    # Rename label column
    df = df.rename(columns={'label_value': args.label_column_name})

    # Save to CSV
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\n✓ Saved {len(df)} labels to {output_path}")

    # Print statistics
    print(f"\nLabel statistics for '{args.label_column_name}':")
    print(df[args.label_column_name].value_counts().sort_index())

    print("\nSample data:")
    print(df.head(10))

    return 0


if __name__ == '__main__':
    sys.exit(main())
