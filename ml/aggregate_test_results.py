#!/usr/bin/env python3
"""
Test Results Aggregation Script

This script creates a centralized log of all test evaluation results by:
1. Reading the experiment log to find final model trainer runs
2. Loading test metrics from comprehensive_evaluation.json files
3. Joining test metrics with training configuration
4. Creating/appending to a centralized test_results_log.csv

Usage:
    # Aggregate all new test results
    python ml/aggregate_test_results.py

    # Specific classifier
    python ml/aggregate_test_results.py --classifier cwt

    # Verbose output with progress
    python ml/aggregate_test_results.py --verbose

    # Force re-scan all versions (ignore existing log)
    python ml/aggregate_test_results.py --force
"""

import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import argparse
import pandas as pd
import json
from pathlib import Path
import numpy as np

def load_comprehensive_evaluation(version_dir, version):
    """
    Load comprehensive_evaluation.json from version directory.

    Tries both new (test_evaluation/) and old (root) locations.

    Args:
        version_dir: Path to version directory
        version: Version string (e.g., 'v001')

    Returns:
        dict or None: Evaluation data if found, None otherwise
    """
    # Try new location first
    new_location = version_dir / 'test_evaluation' / f'comprehensive_evaluation_{version}.json'
    if new_location.exists():
        try:
            with open(new_location, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"    ⚠ Error reading {new_location.name}: {e}")
            return None

    # Fallback to old location
    old_location = version_dir / f'comprehensive_evaluation_{version}.json'
    if old_location.exists():
        try:
            with open(old_location, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"    ⚠ Error reading {old_location.name}: {e}")
            return None

    return None


def parse_channel_from_path(cwt_data_dir):
    r"""
    Parse the actual channel name from cwt_data_dir path.

    Path pattern: F:\...\CWT_labelled_windows\{SIGNAL}\{WAVELET}\...
    Examples:
        PD1\cmor1_5-1_0 → PD1_cmor1.5-1.0
        PD2\mexh → PD2_mexh
        PD1\cmor2_5-0_5 → PD1_cmor2.5-0.5

    Args:
        cwt_data_dir: Path string from experiment log

    Returns:
        str: Channel name (e.g., "PD1_cmor1.5-1.0") or None if parsing fails
    """
    if not cwt_data_dir or pd.isna(cwt_data_dir):
        return None

    try:
        from pathlib import Path
        path_parts = Path(cwt_data_dir).parts

        # Find 'CWT_labelled_windows' in path
        try:
            cwt_idx = path_parts.index('CWT_labelled_windows')
        except ValueError:
            return None

        # Signal source is the next part (e.g., PD1, PD2)
        if cwt_idx + 1 >= len(path_parts):
            return None
        signal = path_parts[cwt_idx + 1]

        # Wavelet is the part after signal (e.g., cmor1_5-1_0, mexh)
        if cwt_idx + 2 >= len(path_parts):
            return None
        wavelet_raw = path_parts[cwt_idx + 2]

        # Normalize wavelet name: replace underscores with dots in version numbers
        # cmor1_5-1_0 → cmor1.5-1.0
        # cmor2_5-0_5 → cmor2.5-0.5
        # fbsp1-1_5-1_0 → fbsp1-1.5-1.0
        # mexh → mexh (no change)
        import re
        # Replace underscores with dots when they separate digits
        wavelet_normalized = re.sub(r'(\d)_(\d)', r'\1.\2', wavelet_raw)

        return f"{signal}_{wavelet_normalized}"

    except Exception as e:
        return None


def extract_test_metrics(eval_data):
    """
    Extract test metrics from comprehensive_evaluation data.

    Args:
        eval_data: Dictionary from comprehensive_evaluation.json

    Returns:
        dict: Extracted metrics
    """
    metrics = {}

    # Test set info
    metrics['test_samples'] = eval_data.get('test_samples', None)
    class_dist = eval_data.get('class_distribution', {})
    metrics['class_0_count'] = class_dist.get('0', class_dist.get(0, None))
    metrics['class_1_count'] = class_dist.get('1', class_dist.get(1, None))

    # Best threshold and metrics
    metrics['best_threshold'] = eval_data.get('best_threshold', None)
    best_metrics = eval_data.get('best_metrics', {})
    metrics['test_accuracy'] = best_metrics.get('accuracy', None)
    metrics['test_precision'] = best_metrics.get('precision', None)
    metrics['test_recall'] = best_metrics.get('recall', None)
    metrics['test_f1_score'] = best_metrics.get('f1_score', None)
    metrics['balanced_score'] = best_metrics.get('balanced_score', None)

    # AUC score
    metrics['auc_score'] = eval_data.get('auc_score', None)

    # Confusion matrix (unpack into tn, fp, fn, tp)
    cm = eval_data.get('confusion_matrix', None)
    if cm and isinstance(cm, list) and len(cm) == 2:
        metrics['tn'] = cm[0][0] if len(cm[0]) >= 1 else None
        metrics['fp'] = cm[0][1] if len(cm[0]) >= 2 else None
        metrics['fn'] = cm[1][0] if len(cm[1]) >= 1 else None
        metrics['tp'] = cm[1][1] if len(cm[1]) >= 2 else None
    else:
        metrics['tn'] = None
        metrics['fp'] = None
        metrics['fn'] = None
        metrics['tp'] = None

    return metrics


def aggregate_test_results(classifier_type, force=False, verbose=False):
    """
    Aggregate test results for a classifier type.

    Args:
        classifier_type: 'cwt' or 'pd_raw'
        force: If True, re-scan all versions ignoring existing log
        verbose: If True, print detailed progress

    Returns:
        int: Number of new versions added
    """
    # Get paths
    ml_dir = Path(__file__).parent
    outputs_dir = ml_dir / 'outputs' / classifier_type
    logs_dir = ml_dir / 'logs' / classifier_type

    # Get experiment log path
    if classifier_type == 'cwt':
        exp_log_path = logs_dir / 'cwt_experiment_log.csv'
    else:
        exp_log_path = logs_dir / 'experiment_log.csv'

    test_log_path = logs_dir / 'test_results_log.csv'

    # Check if experiment log exists
    if not exp_log_path.exists():
        print(f"⚠ Experiment log not found: {exp_log_path}")
        return 0

    # Load experiment log
    print(f"\n📖 Loading experiment log: {exp_log_path.name}")
    try:
        exp_log = pd.read_csv(exp_log_path, encoding='utf-8')
    except Exception as e:
        print(f"✗ Error loading experiment log: {e}")
        return 0

    # Filter for final model trainer runs
    # Source column contains values like: 'final_model_from_v115', 'final_model_trainer', etc.
    if 'source' in exp_log.columns:
        # Match any source that starts with 'final_model'
        final_runs = exp_log[exp_log['source'].str.startswith('final_model', na=False)].copy()
    else:
        print("⚠ 'source' column not found in experiment log - cannot filter for final_model runs")
        print("  Processing all rows...")
        final_runs = exp_log.copy()

    print(f"   Found {len(final_runs)} final model runs")

    if len(final_runs) == 0:
        print("✓ No final model trainer runs to process")
        return 0

    # Load or create test results log
    if test_log_path.exists() and not force:
        print(f"📊 Loading existing test results log: {test_log_path.name}")
        try:
            test_log = pd.read_csv(test_log_path, encoding='utf-8')
            existing_versions = set(test_log['version'].values)
            print(f"   Existing log has {len(test_log)} entries")
        except Exception as e:
            print(f"⚠ Error loading existing log, creating new one: {e}")
            test_log = pd.DataFrame()
            existing_versions = set()
    else:
        if force:
            print(f"🔄 Force mode: Creating new test results log")
        else:
            print(f"📝 Creating new test results log: {test_log_path.name}")
        test_log = pd.DataFrame()
        existing_versions = set()

    # Process each final model trainer run
    new_entries = []
    versions_added = 0
    versions_skipped = 0
    versions_missing = 0

    print(f"\n🔍 Processing final model trainer runs...\n")

    for idx, row in final_runs.iterrows():
        version = row['version']

        # Skip if already in log (unless force mode)
        if not force and version in existing_versions:
            if verbose:
                print(f"  ⊘ {version}: Already in log (skipping)")
            versions_skipped += 1
            continue

        # Look for comprehensive_evaluation.json
        version_dir = outputs_dir / version
        if not version_dir.exists():
            if verbose:
                print(f"  ✗ {version}: Version directory not found")
            versions_missing += 1
            continue

        eval_data = load_comprehensive_evaluation(version_dir, version)

        if eval_data is None:
            if verbose:
                print(f"  ⊘ {version}: No comprehensive_evaluation.json found")
            versions_missing += 1
            continue

        # Extract test metrics
        test_metrics = extract_test_metrics(eval_data)

        # Combine with training config from experiment log
        combined_row = row.to_dict()
        combined_row.update(test_metrics)

        # Parse actual channel from cwt_data_dir path
        actual_channel = parse_channel_from_path(row.get('cwt_data_dir'))
        combined_row['actual_channel'] = actual_channel

        # Add path to comprehensive_evaluation file for reference
        if (version_dir / 'test_evaluation' / f'comprehensive_evaluation_{version}.json').exists():
            combined_row['comprehensive_evaluation_path'] = str(version_dir / 'test_evaluation' / f'comprehensive_evaluation_{version}.json')
        else:
            combined_row['comprehensive_evaluation_path'] = str(version_dir / f'comprehensive_evaluation_{version}.json')

        new_entries.append(combined_row)
        versions_added += 1

        if verbose:
            test_acc = test_metrics.get('test_accuracy')
            acc_str = f"{test_acc:.4f}" if test_acc is not None else "N/A"
            channel_str = actual_channel if actual_channel else "N/A"
            print(f"  ✓ {version}: Added ({channel_str}, test_acc={acc_str})")
        else:
            # Non-verbose: print progress every 10 versions
            if versions_added % 10 == 0:
                print(f"  Processed {versions_added} versions...")

    # Append new entries to test log
    if new_entries:
        new_df = pd.DataFrame(new_entries)
        test_log = pd.concat([test_log, new_df], ignore_index=True)

        # Sort by version
        test_log = test_log.sort_values('version').reset_index(drop=True)

        # Save test results log
        logs_dir.mkdir(exist_ok=True, parents=True)
        test_log.to_csv(test_log_path, index=False, encoding='utf-8')
        print(f"\n💾 Saved test results log: {test_log_path}")
        print(f"   Total entries: {len(test_log)}")

    # Print summary
    print(f"\n📊 SUMMARY FOR {classifier_type.upper()}")
    print(f"{'='*50}")
    print(f"Versions added:      {versions_added}")
    print(f"Versions skipped:    {versions_skipped}")
    print(f"Versions missing:    {versions_missing}")
    print(f"Total in log:        {len(test_log)}")
    print(f"{'='*50}")

    return versions_added


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate test results into centralized log',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate all new test results
  python ml/aggregate_test_results.py

  # Specific classifier with verbose output
  python ml/aggregate_test_results.py --classifier cwt --verbose

  # Force re-scan all versions
  python ml/aggregate_test_results.py --force
        """
    )

    parser.add_argument('--classifier', choices=['cwt', 'pd_raw', 'all'], default='all',
                       help='Classifier type to aggregate (default: all)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-scan all versions, ignoring existing log')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress for each version')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("📊 TEST RESULTS AGGREGATION")
    print("="*70)

    # Determine which classifiers to process
    classifiers = ['cwt', 'pd_raw'] if args.classifier == 'all' else [args.classifier]

    total_added = 0

    # Process each classifier
    for classifier in classifiers:
        added = aggregate_test_results(classifier, force=args.force, verbose=args.verbose)
        total_added += added

    # Final summary
    print(f"\n{'='*70}")
    print(f"✓ Aggregation complete!")
    print(f"  Total versions added: {total_added}")
    print(f"{'='*70}\n")

    return 0


if __name__ == '__main__':
    exit(main())
