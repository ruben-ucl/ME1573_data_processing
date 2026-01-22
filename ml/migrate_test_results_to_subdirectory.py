#!/usr/bin/env python3
"""
Migration Script: Move Test Results to test_evaluation/ Subdirectory

This script retroactively moves test evaluation outputs from version directories
into a test_evaluation/ subdirectory to separate them from training outputs.

Usage:
    # Dry run (preview changes)
    python ml/migrate_test_results_to_subdirectory.py --dry-run

    # Execute migration
    python ml/migrate_test_results_to_subdirectory.py

    # Specific classifier only
    python ml/migrate_test_results_to_subdirectory.py --classifier cwt
"""

import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import argparse
import shutil
from pathlib import Path
import json

# Define test output file patterns
TEST_FILE_PATTERNS = [
    'threshold_optimization_v*.csv',
    'threshold_optimization_v*.png',
    'test_predictions_v*.pkl',
    'confusion_matrix_v*.png',
    'classification_report_v*.txt',
    'comprehensive_evaluation_v*.json',
    'pv_map_test_set_v*.png',
]

TEST_SUBDIRECTORIES = [
    'track_predictions',
    'gradcam_analysis_v*',
]


def migrate_version_directory(version_dir, dry_run=False):
    """
    Migrate test results from a single version directory.

    Args:
        version_dir: Path to version directory (e.g., ml/outputs/cwt/v001/)
        dry_run: If True, only print what would be moved without moving

    Returns:
        dict: Statistics (files_moved, dirs_moved, skipped, errors)
    """
    stats = {
        'files_moved': 0,
        'dirs_moved': 0,
        'skipped': 0,
        'errors': 0
    }

    version_dir = Path(version_dir)
    test_eval_dir = version_dir / 'test_evaluation'

    # Create test_evaluation subdirectory
    if not dry_run and not test_eval_dir.exists():
        test_eval_dir.mkdir(exist_ok=True)
        print(f"  📁 Created: {test_eval_dir.relative_to(version_dir.parent.parent)}/")
    elif test_eval_dir.exists():
        print(f"  ✓ {test_eval_dir.relative_to(version_dir.parent.parent)}/ already exists")

    # Move test files
    for pattern in TEST_FILE_PATTERNS:
        matching_files = list(version_dir.glob(pattern))
        for src_file in matching_files:
            dest_file = test_eval_dir / src_file.name

            # Skip if already in destination
            if dest_file.exists():
                print(f"  ⊘ Skip (exists): {src_file.name}")
                stats['skipped'] += 1
                continue

            # Move file
            if dry_run:
                print(f"  [DRY RUN] Would move: {src_file.name} → test_evaluation/")
            else:
                try:
                    shutil.move(str(src_file), str(dest_file))
                    print(f"  ✓ Moved: {src_file.name} → test_evaluation/")
                    stats['files_moved'] += 1
                except Exception as e:
                    print(f"  ✗ Error moving {src_file.name}: {e}")
                    stats['errors'] += 1

    # Move test subdirectories
    for pattern in TEST_SUBDIRECTORIES:
        matching_dirs = [d for d in version_dir.glob(pattern) if d.is_dir()]
        for src_dir in matching_dirs:
            dest_dir = test_eval_dir / src_dir.name

            # Skip if already in destination
            if dest_dir.exists():
                print(f"  ⊘ Skip (exists): {src_dir.name}/")
                stats['skipped'] += 1
                continue

            # Move directory
            if dry_run:
                print(f"  [DRY RUN] Would move: {src_dir.name}/ → test_evaluation/")
            else:
                try:
                    shutil.move(str(src_dir), str(dest_dir))
                    print(f"  ✓ Moved: {src_dir.name}/ → test_evaluation/")
                    stats['dirs_moved'] += 1
                except Exception as e:
                    print(f"  ✗ Error moving {src_dir.name}/: {e}")
                    stats['errors'] += 1

    # Update paths in comprehensive_evaluation.json if it exists
    comp_eval_file = test_eval_dir / f'comprehensive_evaluation_{version_dir.name}.json'
    if comp_eval_file.exists() and not dry_run:
        try:
            with open(comp_eval_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Update output_files paths to reflect new location
            if 'output_files' in data:
                updated = False
                for key, path_str in data['output_files'].items():
                    path = Path(path_str)
                    # If path doesn't contain test_evaluation, update it
                    if 'test_evaluation' not in path.parts:
                        # Reconstruct path with test_evaluation
                        new_path = version_dir / 'test_evaluation' / path.name
                        data['output_files'][key] = str(new_path)
                        updated = True

                if updated:
                    with open(comp_eval_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                    print(f"  ✓ Updated paths in: comprehensive_evaluation_{version_dir.name}.json")
        except Exception as e:
            print(f"  ⚠ Could not update comprehensive_evaluation JSON: {e}")

    return stats


def migrate_classifier(classifier_type, dry_run=False):
    """
    Migrate all versions for a classifier type.

    Args:
        classifier_type: 'cwt' or 'pd_raw'
        dry_run: If True, only preview without moving

    Returns:
        dict: Total statistics
    """
    # Get outputs directory
    outputs_dir = Path(__file__).parent / 'outputs' / classifier_type

    if not outputs_dir.exists():
        print(f"⚠ Output directory not found: {outputs_dir}")
        return {'files_moved': 0, 'dirs_moved': 0, 'skipped': 0, 'errors': 0}

    # Get all version directories (v001, v002, ...)
    version_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith('v')])

    if not version_dirs:
        print(f"No version directories found in {outputs_dir}")
        return {'files_moved': 0, 'dirs_moved': 0, 'skipped': 0, 'errors': 0}

    print(f"\n{'='*70}")
    print(f"Migrating {classifier_type.upper()} classifier test results")
    print(f"Found {len(version_dirs)} version directories")
    print(f"{'='*70}\n")

    total_stats = {
        'files_moved': 0,
        'dirs_moved': 0,
        'skipped': 0,
        'errors': 0
    }

    for version_dir in version_dirs:
        print(f"\n📦 Processing: {version_dir.name}")

        stats = migrate_version_directory(version_dir, dry_run=dry_run)

        # Update totals
        for key in total_stats:
            total_stats[key] += stats[key]

    return total_stats


def main():
    parser = argparse.ArgumentParser(
        description='Migrate test results to test_evaluation/ subdirectory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes without moving files
  python ml/migrate_test_results_to_subdirectory.py --dry-run

  # Execute migration for all classifiers
  python ml/migrate_test_results_to_subdirectory.py

  # Migrate only CWT classifier
  python ml/migrate_test_results_to_subdirectory.py --classifier cwt
        """
    )

    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without actually moving files')
    parser.add_argument('--classifier', choices=['cwt', 'pd_raw', 'all'], default='all',
                       help='Classifier type to migrate (default: all)')

    args = parser.parse_args()

    if args.dry_run:
        print("\n" + "="*70)
        print("🔍 DRY RUN MODE - No files will be moved")
        print("="*70)

    # Determine which classifiers to migrate
    classifiers = ['cwt', 'pd_raw'] if args.classifier == 'all' else [args.classifier]

    grand_total = {
        'files_moved': 0,
        'dirs_moved': 0,
        'skipped': 0,
        'errors': 0
    }

    # Migrate each classifier
    for classifier in classifiers:
        stats = migrate_classifier(classifier, dry_run=args.dry_run)
        for key in grand_total:
            grand_total[key] += stats[key]

    # Print summary
    print("\n" + "="*70)
    print("📊 MIGRATION SUMMARY")
    print("="*70)
    print(f"Files moved:       {grand_total['files_moved']}")
    print(f"Directories moved: {grand_total['dirs_moved']}")
    print(f"Skipped (exist):   {grand_total['skipped']}")
    print(f"Errors:            {grand_total['errors']}")
    print("="*70)

    if args.dry_run:
        print("\n✓ Dry run complete. Run without --dry-run to execute migration.")
    else:
        print("\n✓ Migration complete!")

    return 0


if __name__ == '__main__':
    exit(main())
