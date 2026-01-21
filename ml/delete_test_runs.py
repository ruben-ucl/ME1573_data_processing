#!/usr/bin/env python3
"""
Delete Test Runs Utility

Identifies and deletes test runs from experiment logs and their associated
output folders. Test runs are identified by:
- k_folds <= 2 AND epochs <= 2
- OR source == 'manual' with very short training times (< 5 minutes)

Usage:
    python delete_test_runs.py cwt
    python delete_test_runs.py pd
"""

import argparse
import pandas as pd
import shutil
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    get_cwt_experiment_log_path,
    get_pd_experiment_log_path,
    CWT_OUTPUTS_DIR,
    PD_OUTPUTS_DIR
)


def identify_test_runs(df, classifier_type):
    """
    Identify test runs in the experiment log.

    Test run criteria:
    - k_folds <= 2 AND epochs <= 2 (strict test mode)

    Args:
        df: DataFrame with experiment log
        classifier_type: 'cwt_image' or 'pd_signal'

    Returns:
        DataFrame with test runs only
    """
    # Primary criterion: k_folds <= 2 AND epochs <= 2
    test_mask = (df['k_folds'] <= 2) & (df['epochs'] <= 2)

    test_runs = df[test_mask].copy()

    return test_runs


def delete_output_folders(versions, classifier_type, dry_run=False):
    """
    Delete output folders for the given versions.

    Args:
        versions: List of version strings (e.g., ['v001', 'v002'])
        classifier_type: 'cwt_image' or 'pd_signal'
        dry_run: If True, only print what would be deleted

    Returns:
        tuple: (deleted_count, failed_count)
    """
    if classifier_type == 'cwt_image':
        output_dir = CWT_OUTPUTS_DIR
    else:
        output_dir = PD_OUTPUTS_DIR

    deleted_count = 0
    failed_count = 0

    for version in versions:
        version_dir = output_dir / version

        if version_dir.exists():
            if dry_run:
                print(f"  [DRY RUN] Would delete: {version_dir}")
                deleted_count += 1
            else:
                try:
                    shutil.rmtree(version_dir)
                    print(f"  ✓ Deleted: {version_dir}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  ✗ Failed to delete {version_dir}: {e}")
                    failed_count += 1
        else:
            print(f"  ⊘ Not found: {version_dir}")

    return deleted_count, failed_count


def delete_log_folders(versions, classifier_type, dry_run=False):
    """
    Delete log folders for the given versions.

    Args:
        versions: List of version strings (e.g., ['v001', 'v002'])
        classifier_type: 'cwt_image' or 'pd_signal'
        dry_run: If True, only print what would be deleted

    Returns:
        tuple: (deleted_count, failed_count)
    """
    if classifier_type == 'cwt_image':
        log_base = Path("ml/logs/cwt")
    else:
        log_base = Path("ml/logs/pd")

    deleted_count = 0
    failed_count = 0

    for version in versions:
        version_dir = log_base / version

        if version_dir.exists():
            if dry_run:
                print(f"  [DRY RUN] Would delete: {version_dir}")
                deleted_count += 1
            else:
                try:
                    shutil.rmtree(version_dir)
                    print(f"  ✓ Deleted: {version_dir}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  ✗ Failed to delete {version_dir}: {e}")
                    failed_count += 1
        else:
            print(f"  ⊘ Not found: {version_dir}")

    return deleted_count, failed_count


def main():
    parser = argparse.ArgumentParser(
        description='Delete test runs from experiment logs and clean up associated files'
    )
    parser.add_argument(
        'classifier',
        choices=['cwt', 'pd'],
        help='Classifier type: cwt or pd'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    parser.add_argument(
        '--keep-logs',
        action='store_true',
        help='Delete from experiment log but keep output/log folders'
    )

    args = parser.parse_args()

    # Map argument to classifier type
    classifier_type = 'cwt_image' if args.classifier == 'cwt' else 'pd_signal'

    # Get experiment log path
    if classifier_type == 'cwt_image':
        log_path = get_cwt_experiment_log_path()
    else:
        log_path = get_pd_experiment_log_path()

    print("="*70)
    print(f"DELETE TEST RUNS - {classifier_type.upper()}")
    print("="*70)

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be deleted\n")

    # Load experiment log
    if not log_path.exists():
        print(f"\n❌ Experiment log not found: {log_path}")
        return 1

    print(f"\nLoading experiment log: {log_path}")
    df = pd.read_csv(log_path)
    print(f"Total experiments in log: {len(df)}")

    # Identify test runs
    test_runs = identify_test_runs(df, classifier_type)

    print(f"\n{'='*70}")
    print(f"IDENTIFIED TEST RUNS")
    print(f"{'='*70}")
    print(f"Test runs found: {len(test_runs)}")

    if len(test_runs) == 0:
        print("\n✅ No test runs found. Nothing to delete.")
        return 0

    # Show summary of test runs
    print(f"\nTest run criteria: k_folds <= 2 AND epochs <= 2")
    print(f"\nTest runs to be deleted:")
    for idx, row in test_runs.iterrows():
        version = row['version']
        k_folds = row['k_folds']
        epochs = row['epochs']
        timestamp = row['timestamp']
        accuracy = row.get('mean_val_accuracy', 'N/A')
        print(f"  - {version}: k_folds={k_folds}, epochs={epochs}, accuracy={accuracy}, timestamp={timestamp}")

    # Get versions to delete
    versions = test_runs['version'].tolist()

    # Confirmation prompt (skip in dry-run mode)
    if not args.dry_run:
        print(f"\n{'='*70}")
        print(f"⚠️  WARNING: This will permanently delete:")
        print(f"  - {len(test_runs)} rows from experiment log")
        if not args.keep_logs:
            print(f"  - Output folders for {len(versions)} versions")
            print(f"  - Log folders for {len(versions)} versions")
        print(f"{'='*70}")
        response = input("\nProceed with deletion? (yes/no): ").strip().lower()
        if response != 'yes':
            print("\n❌ Deletion cancelled.")
            return 0

    # Delete output and log folders
    if not args.keep_logs:
        print(f"\n{'='*70}")
        print(f"DELETING OUTPUT FOLDERS")
        print(f"{'='*70}")
        deleted_outputs, failed_outputs = delete_output_folders(versions, classifier_type, args.dry_run)

        print(f"\n{'='*70}")
        print(f"DELETING LOG FOLDERS")
        print(f"{'='*70}")
        deleted_logs, failed_logs = delete_log_folders(versions, classifier_type, args.dry_run)
    else:
        deleted_outputs = deleted_logs = 0
        failed_outputs = failed_logs = 0
        print(f"\n⊘ Skipping folder deletion (--keep-logs enabled)")

    # Delete from experiment log
    print(f"\n{'='*70}")
    print(f"UPDATING EXPERIMENT LOG")
    print(f"{'='*70}")

    if args.dry_run:
        print(f"[DRY RUN] Would delete {len(test_runs)} rows from experiment log")
    else:
        # Create backup
        backup_path = log_path.parent / f"{log_path.stem}_backup_before_delete{log_path.suffix}"
        shutil.copy2(log_path, backup_path)
        print(f"✓ Backup created: {backup_path}")

        # Remove test runs from dataframe
        df_cleaned = df[~df['version'].isin(versions)]

        # Save cleaned log
        df_cleaned.to_csv(log_path, index=False, encoding='utf-8')
        print(f"✓ Deleted {len(test_runs)} rows from experiment log")
        print(f"  Before: {len(df)} rows")
        print(f"  After: {len(df_cleaned)} rows")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    if args.dry_run:
        print(f"[DRY RUN] Would have deleted:")
    else:
        print(f"Successfully deleted:")

    print(f"  - Experiment log rows: {len(test_runs)}")
    if not args.keep_logs:
        print(f"  - Output folders: {deleted_outputs}")
        print(f"  - Log folders: {deleted_logs}")
        if failed_outputs > 0 or failed_logs > 0:
            print(f"  - Failed deletions: {failed_outputs + failed_logs}")

    if args.dry_run:
        print(f"\n💡 Run without --dry-run to actually delete files")
    else:
        print(f"\n✅ Cleanup completed successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
