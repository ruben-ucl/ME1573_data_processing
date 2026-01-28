"""
Script to regenerate track prediction visualizations for all final model versions.

Reads the test_results_log.csv and runs visualize_track_predictions.py for each
version that has been tested.

Usage:
    python ml/regenerate_track_visualizations.py
    python ml/regenerate_track_visualizations.py --classifier_type cwt_image
    python ml/regenerate_track_visualizations.py --dry-run  # Show what would be run
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path
import argparse

# Set UTF-8 encoding for subprocess output
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')


def find_test_results_log(classifier_type='cwt_image'):
    """Find the test results log file for the given classifier type."""
    base_dir = Path(__file__).parent

    if classifier_type == 'cwt_image':
        log_path = base_dir / 'logs' / 'cwt' / 'test_results_log.csv'
    elif classifier_type == 'pd_signal':
        log_path = base_dir / 'logs' / 'pd_raw' / 'test_results_log.csv'
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    if not log_path.exists():
        raise FileNotFoundError(f"Test results log not found: {log_path}")

    return log_path


def get_final_model_versions(log_path):
    """Extract all version numbers from the test results log."""
    df = pd.read_csv(log_path, encoding='utf-8')

    # Get unique versions
    versions = df['version'].unique().tolist()

    # Sort versions (assuming format like v206, v207, etc.)
    versions.sort()

    return versions


def run_visualization(version, classifier_type='cwt_image', dry_run=False):
    """Run the visualize_track_predictions.py script for a specific version."""
    base_dir = Path(__file__).parent
    script_path = base_dir / 'visualize_track_predictions.py'

    cmd = [
        sys.executable,
        str(script_path),
        '--version', version,
        '--classifier_type', classifier_type
    ]

    print(f"\n{'='*70}")
    print(f"🔄 Processing {version} ({classifier_type})")
    print(f"{'='*70}")

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=str(base_dir.parent)  # Run from project root
        )

        if result.returncode == 0:
            print(f"✅ Successfully generated visualizations for {version}")
            # Show key output lines
            for line in result.stdout.split('\n'):
                if '✅' in line or 'Output directory' in line or 'Confusion matrix' in line:
                    print(f"   {line}")
            return True
        else:
            print(f"❌ Failed to generate visualizations for {version}")
            print(f"   Error: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Exception while processing {version}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate track prediction visualizations for all final models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regenerate all CWT image model visualizations
  python ml/regenerate_track_visualizations.py

  # Regenerate PD signal model visualizations
  python ml/regenerate_track_visualizations.py --classifier_type pd_signal

  # Dry run to see what would be executed
  python ml/regenerate_track_visualizations.py --dry-run
        """
    )

    parser.add_argument('--classifier_type', type=str,
                       choices=['cwt_image', 'pd_signal'],
                       default='cwt_image',
                       help='Type of classifier (default: cwt_image)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be executed without running')
    parser.add_argument('--versions', type=str, nargs='+',
                       help='Specific versions to process (e.g., v206 v207)')

    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"📊 Track Visualization Regenerator")
    print(f"{'='*70}")
    print(f"Classifier Type: {args.classifier_type}")
    if args.dry_run:
        print("Mode: DRY RUN (no actual execution)")
    print(f"{'='*70}\n")

    try:
        # Find test results log
        log_path = find_test_results_log(args.classifier_type)
        print(f"📂 Test results log: {log_path}\n")

        # Get versions to process
        if args.versions:
            versions = args.versions
            print(f"🎯 Processing specified versions: {', '.join(versions)}\n")
        else:
            versions = get_final_model_versions(log_path)
            print(f"🔍 Found {len(versions)} final model versions: {', '.join(versions)}\n")

        # Process each version
        success_count = 0
        failed_count = 0

        for i, version in enumerate(versions, 1):
            print(f"\n[{i}/{len(versions)}] Processing {version}...")

            success = run_visualization(version, args.classifier_type, args.dry_run)

            if success:
                success_count += 1
            else:
                failed_count += 1

        # Summary
        print(f"\n{'='*70}")
        print(f"📊 Summary")
        print(f"{'='*70}")
        print(f"✅ Successful: {success_count}/{len(versions)}")
        if failed_count > 0:
            print(f"❌ Failed: {failed_count}/{len(versions)}")
        print(f"{'='*70}\n")

        if failed_count > 0:
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
