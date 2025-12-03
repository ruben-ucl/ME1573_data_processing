"""
Export Keyhole Perimeters to CSV

Standalone tool to convert HDF5 keyhole binary images to CSV format
compatible with RayTracer.py.

Output CSV format:
    X,Y,Slice
    123.4,45.6,1
    125.2,46.1,1
    ...

Usage:
    python export_keyhole_to_csv.py input.hdf5 output.csv
    python export_keyhole_to_csv.py input.hdf5 output.csv --opening-width 150 --smoothing 0.7

Author: Claude
Date: 2025-11-20
"""

import argparse
from pathlib import Path
import sys

# Add sim directory to path for imports when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from hdf5_keyhole_reader import extract_all_perimeters


def main():
    parser = argparse.ArgumentParser(
        description='Export keyhole perimeters from HDF5 to CSV for RayTracer'
    )
    parser.add_argument('input_hdf5', help='Input HDF5 file path')
    parser.add_argument('output_csv', help='Output CSV file path')
    parser.add_argument('--dataset', default='keyhole_bin',
                       help='HDF5 dataset name (default: keyhole_bin)')
    parser.add_argument('--opening-width', type=float, default=120.0,
                       help='Opening width in microns (default: 120.0)')
    parser.add_argument('--smoothing', type=float, default=0.5,
                       help='Smoothing factor (default: 0.5)')
    parser.add_argument('--resolution', type=float, default=4.3,
                       help='Resolution in μm/pixel (default: 4.3)')
    parser.add_argument('--frame-start', type=int, default=None,
                       help='Start frame index (optional)')
    parser.add_argument('--frame-end', type=int, default=None,
                       help='End frame index (optional)')

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input_hdf5)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine frame range
    frame_range = None
    if args.frame_start is not None or args.frame_end is not None:
        start = args.frame_start if args.frame_start is not None else 0
        end = args.frame_end if args.frame_end is not None else 999999  # Large number
        frame_range = (start, end)

    # Extract perimeters
    print(f"Extracting keyhole perimeters...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Opening width: {args.opening_width:.1f} μm")
    print(f"  Smoothing: {args.smoothing:.2f}")
    print(f"  Resolution: {args.resolution:.2f} μm/pixel")

    if frame_range:
        print(f"  Frame range: {frame_range[0]} to {frame_range[1]}")

    try:
        df = extract_all_perimeters(
            str(input_path),
            dataset_name=args.dataset,
            opening_width_um=args.opening_width,
            smoothing_factor=args.smoothing,
            resolution_um_per_pixel=args.resolution,
            frame_range=frame_range
        )

        # Save to CSV
        df.to_csv(output_path, index=False)

        print(f"\n✓ Success!")
        print(f"  Extracted {len(df):,} points from {df['Slice'].nunique()} frames")
        print(f"  X range: {df['X'].min():.1f} to {df['X'].max():.1f} μm")
        print(f"  Y range: {df['Y'].min():.1f} to {df['Y'].max():.1f} μm")
        print(f"  Saved to: {output_path}")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
