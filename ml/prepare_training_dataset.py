#!/usr/bin/env python3
"""
Prepare Training Dataset Utility

Comprehensive dataset preparation for ML training that:
1. Filters tracks based on logbook criteria (material, layer, laser mode, etc.)
2. Removes low-signal images based on mean intensity threshold
3. Intelligently downsamples majority class to balance the dataset
4. Preserves variety across multiple dimensions:
   - Track ID (experimental run continuity)
   - Laser power (process parameter)
   - Scan speed (process parameter)
   - Signal quality (mean intensity)
   - Temporal coverage (window index within track)

Usage:
    # Basic balancing with default settings
    python ml/prepare_training_dataset.py --dry-run

    # With signal filtering and custom ratio
    python ml/prepare_training_dataset.py --signal-threshold 10 --target-ratio 1.5

    # With logbook-based filtering (AlSi10Mg, Layer 1, CW laser)
    python ml/prepare_training_dataset.py --material AlSi10Mg --layer 1 --laser-mode cw

    # Combined filtering and balancing
    python ml/prepare_training_dataset.py --material AlSi10Mg --layer 1 --signal-threshold 10 --target-ratio 1.5
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from PIL import Image
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import get_logbook, filter_logbook_tracks, define_collumn_labels
from ml.config import get_default_cwt_data_dir


def compute_signal_metrics(image_path):
    """
    Compute signal quality metrics for a CWT image.

    Args:
        image_path: Path to image file

    Returns:
        dict with mean, std, variance
    """
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)

        return {
            'mean': float(np.mean(img_array)),
            'std': float(np.std(img_array)),
            'variance': float(np.var(img_array))
        }
    except Exception as e:
        print(f"Warning: Could not read {image_path}: {e}")
        return {'mean': 0.0, 'std': 0.0, 'variance': 0.0}


def parse_filename(filename):
    """
    Extract track_id and window_index from CWT image filename.

    Expected format: {track_id}_window_{index}.png

    Args:
        filename: Image filename

    Returns:
        dict with track_id and window_index
    """
    try:
        stem = Path(filename).stem
        parts = stem.split('_window_')

        if len(parts) == 2:
            track_id = parts[0]
            window_index = int(parts[1])
            return {'track_id': track_id, 'window_index': window_index}
        else:
            # Fallback: try to extract track_id at least
            return {'track_id': stem.split('_')[0], 'window_index': 0}
    except Exception as e:
        print(f"Warning: Could not parse filename {filename}: {e}")
        return {'track_id': 'unknown', 'window_index': 0}


def join_with_logbook(df, logbook):
    """
    Join dataset with logbook to get process parameters.

    Args:
        df: DataFrame with 'track_id' column
        logbook: Logbook DataFrame from get_logbook()

    Returns:
        DataFrame with laser_power and scan_speed columns added
    """
    # Logbook uses 'trackid' (not 'track_id') and column names from define_collumn_labels():
    # 'Avg. power [W]' for laser power
    # 'Scan speed [mm/s]' for scan speed

    if 'trackid' not in logbook.columns:
        print("Warning: 'trackid' not found in logbook. Using default values.")
        df['laser_power'] = 0.0
        df['scan_speed'] = 0.0
        return df

    # Create a mapping with correct column names
    logbook_subset = logbook[['trackid', 'Avg. power [W]', 'Scan speed [mm/s]']].copy()
    logbook_subset = logbook_subset.rename(columns={
        'trackid': 'track_id',
        'Avg. power [W]': 'laser_power',
        'Scan speed [mm/s]': 'scan_speed'
    })

    # Merge
    df_merged = df.merge(logbook_subset, on='track_id', how='left')

    # Fill missing values with defaults
    df_merged['laser_power'] = df_merged['laser_power'].fillna(0.0)
    df_merged['scan_speed'] = df_merged['scan_speed'].fillna(0.0)

    return df_merged


def apply_logbook_filters(df, logbook, filters_dict):
    """
    Filter dataset based on logbook criteria.

    Args:
        df: DataFrame with 'track_id' column
        logbook: Logbook DataFrame from get_logbook()
        filters_dict: Dictionary of filter conditions (see filter_logbook_tracks)

    Returns:
        tuple: (filtered_df, active_filters)
            - filtered_df: DataFrame with only allowed tracks
            - active_filters: List of active filter names
    """
    if not filters_dict or len(filters_dict) == 0:
        return df, []

    # Apply filters to logbook
    filtered_logbook, active_filters = filter_logbook_tracks(logbook, filters_dict)
    allowed_trackids = set(filtered_logbook['trackid'].unique())

    print(f"\nLogbook filters applied: {active_filters}")
    print(f"Allowed trackids: {len(allowed_trackids)}")

    # Filter dataset to only include allowed trackids
    initial_count = len(df)
    df_filtered = df[df['track_id'].isin(allowed_trackids)].copy()
    filtered_count = len(df_filtered)

    print(f"Samples before logbook filtering: {initial_count}")
    print(f"Samples after logbook filtering: {filtered_count}")
    print(f"Removed: {initial_count - filtered_count} ({(initial_count - filtered_count)/initial_count*100:.1f}%)")

    return df_filtered, active_filters


def stratified_downsample(df, target_count, dimensions):
    """
    Perform stratified downsampling to preserve variety across dimensions.

    Args:
        df: DataFrame to downsample
        target_count: Target number of samples
        dimensions: List of column names to stratify by

    Returns:
        Downsampled DataFrame
    """
    if len(df) <= target_count:
        return df

    # Create bins for continuous variables
    df_binned = df.copy()

    # Bin signal intensity into low/medium/high
    if 'mean' in dimensions:
        df_binned['mean_bin'] = pd.qcut(df['mean'], q=3, labels=['low', 'medium', 'high'], duplicates='drop')
        dimensions = [d if d != 'mean' else 'mean_bin' for d in dimensions]

    # Bin window_index if it has many unique values
    if 'window_index' in dimensions and df['window_index'].nunique() > 10:
        df_binned['window_bin'] = pd.qcut(df['window_index'], q=5, labels=False, duplicates='drop')
        dimensions = [d if d != 'window_index' else 'window_bin' for d in dimensions]

    # Create stratification groups
    df_binned['stratum'] = df_binned[dimensions].astype(str).agg('_'.join, axis=1)

    # Count samples per stratum
    stratum_counts = df_binned['stratum'].value_counts()
    total_samples = len(df_binned)

    # Calculate proportional allocation
    sampled_indices = []

    for stratum, count in stratum_counts.items():
        stratum_df = df_binned[df_binned['stratum'] == stratum]

        # Proportional sample size
        n_samples = int(np.ceil(target_count * (count / total_samples)))
        n_samples = min(n_samples, len(stratum_df))  # Can't sample more than available

        # Random sample from stratum
        sampled = stratum_df.sample(n=n_samples, random_state=42)
        sampled_indices.extend(sampled.index.tolist())

    # If we have too many samples (due to rounding), randomly remove excess
    if len(sampled_indices) > target_count:
        np.random.seed(42)
        sampled_indices = np.random.choice(sampled_indices, size=target_count, replace=False).tolist()

    # If we have too few samples, add more from largest strata
    elif len(sampled_indices) < target_count:
        remaining_indices = df_binned.index.difference(sampled_indices).tolist()
        needed = target_count - len(sampled_indices)
        np.random.seed(42)
        additional = np.random.choice(remaining_indices, size=min(needed, len(remaining_indices)), replace=False).tolist()
        sampled_indices.extend(additional)

    return df.loc[sampled_indices]


def adaptive_block_temporal_split(df, gap_size=5, random_seed=42,
                                  short_threshold=10, medium_threshold=15, long_threshold=20):
    """
    Split dataset into train/val using adaptive block sizes based on track length.
    Guarantees zero temporal leakage by inserting gaps between blocks.

    Adaptive block sizes (corrected for fencepost problem):
      - Very short tracks (<short_threshold): B_train=1, B_val=1 (extract what we can)
      - Short tracks (short_threshold to medium_threshold-1): B_train=4, B_val=1
      - Medium tracks (medium_threshold to long_threshold-1): B_train=8, B_val=2
      - Long tracks (long_threshold+): B_train=12, B_val=3

    Args:
        df: DataFrame with 'track_id' and 'window_index' columns
        gap_size: Number of windows to skip between blocks (default=5 for zero overlap)
        random_seed: Random seed for reproducibility
        short_threshold: Minimum windows for short tracks (default=10)
        medium_threshold: Minimum windows for medium tracks (default=15)
        long_threshold: Minimum windows for long tracks (default=20)

    Returns:
        tuple: (train_df, val_df, split_stats)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    train_indices = []
    val_indices = []
    discarded_indices = []
    track_stats = []

    for track_id in sorted(df['track_id'].unique()):
        track_df = df[df['track_id'] == track_id].sort_values('window_index').copy()
        track_idx = track_df.index.tolist()
        track_len = len(track_idx)

        # Handle very short tracks (< 7 windows) - not enough for proper split with gaps
        # Assign entire track to train or val randomly
        if track_len < 7:
            if np.random.random() < 0.5:
                train_indices.extend(track_idx)
                track_stats.append({
                    'trackid': track_id,
                    'length': track_len,
                    'B_train': 0,  # Special marker for whole-track assignment
                    'total': track_len,
                    'train': track_len,
                    'val': 0,
                    'discarded': 0,
                    'discard_pct': 0,
                    'started_with': 'train (whole track)'
                })
            else:
                val_indices.extend(track_idx)
                track_stats.append({
                    'trackid': track_id,
                    'length': track_len,
                    'B_train': 0,  # Special marker for whole-track assignment
                    'total': track_len,
                    'train': 0,
                    'val': track_len,
                    'discarded': 0,
                    'discard_pct': 0,
                    'started_with': 'val (whole track)'
                })
            continue

        # Select appropriate block size based on track length thresholds
        # Minimum windows = B_train + gap + B_val + gap (full cycle)
        if track_len < short_threshold:
            # Very short tracks: use B_train = B_val = 1
            # Minimum: 1 + 5 + 1 + 5 = 12 windows for full cycle, 7 for one split
            B_train = 1
            B_val = 1
        elif track_len < medium_threshold:
            B_train = 4
            B_val = 1
        elif track_len < long_threshold:
            B_train = 8
            B_val = 2
        else:
            B_train = 12
            B_val = 3

        # Randomly choose starting block type (50/50 to preserve aggregate ratio)
        start_with_train = np.random.random() < 0.5

        idx = 0
        is_train_block = start_with_train
        track_train = 0
        track_val = 0
        track_discard = 0

        while idx < len(track_idx):
            # Determine current block size
            block_size = B_train if is_train_block else B_val

            # Extract block (or remainder if insufficient)
            block_end = min(idx + block_size, len(track_idx))
            block_indices = track_idx[idx:block_end]

            # Assign block
            if is_train_block:
                train_indices.extend(block_indices)
                track_train += len(block_indices)
            else:
                val_indices.extend(block_indices)
                track_val += len(block_indices)

            idx = block_end

            # Add gap ONLY if there's another block coming (fencepost correction)
            remaining_windows = len(track_idx) - idx
            next_block_size = B_val if is_train_block else B_train

            if remaining_windows >= next_block_size:
                # There's room for at least one more block, add gap
                gap_end = min(idx + gap_size, len(track_idx))
                gap_indices = track_idx[idx:gap_end]
                discarded_indices.extend(gap_indices)
                track_discard += len(gap_indices)
                idx = gap_end
            else:
                # Final block - no gap needed
                break

            # Toggle block type
            is_train_block = not is_train_block

        # Track statistics
        track_stats.append({
            'trackid': track_id,
            'length': track_len,
            'B_train': B_train,
            'total': len(track_idx),
            'train': track_train,
            'val': track_val,
            'discarded': track_discard,
            'discard_pct': track_discard / len(track_idx) * 100 if len(track_idx) > 0 else 0,
            'started_with': 'train' if start_with_train else 'val'
        })

    # Create split DataFrames
    train_df = df.loc[train_indices].copy()
    val_df = df.loc[val_indices].copy()

    # Overall statistics
    total_windows = len(df)
    retained_windows = len(train_indices) + len(val_indices)

    split_stats = {
        'total_windows': total_windows,
        'train_windows': len(train_indices),
        'val_windows': len(val_indices),
        'discarded_windows': len(discarded_indices),
        'retained_windows': retained_windows,
        'train_pct': len(train_indices) / retained_windows * 100 if retained_windows > 0 else 0,
        'val_pct': len(val_indices) / retained_windows * 100 if retained_windows > 0 else 0,
        'discard_pct': len(discarded_indices) / total_windows * 100 if total_windows > 0 else 0,
        'retention_pct': retained_windows / total_windows * 100 if total_windows > 0 else 0,
        'gap_size': gap_size,
        'num_tracks_total': df['track_id'].nunique(),
        'num_tracks_used': len(track_stats),
        'num_tracks_excluded': df['track_id'].nunique() - len(track_stats),
        'track_stats': track_stats
    }

    return train_df, val_df, split_stats


def find_best_temporal_split(df, target_val_ratio=0.2, target_class_ratio=None,
                             n_seeds=100, label_column='has_porosity', gap_size=5,
                             short_threshold=10, medium_threshold=15, long_threshold=20):
    """
    Try multiple random seeds and select the split with best class balance.

    Args:
        df: Full dataset DataFrame
        target_val_ratio: Desired validation ratio (default=0.2 for 20%)
        target_class_ratio: Desired positive class ratio (None = match dataset distribution)
        n_seeds: Number of random seeds to try (default=100)
        label_column: Column containing class labels (default='has_porosity')
        gap_size: Gap size between blocks (default=5)
        short_threshold: Minimum windows for short tracks (default=10)
        medium_threshold: Minimum windows for medium tracks (default=15)
        long_threshold: Minimum windows for long tracks (default=20)

    Returns:
        tuple: (best_seed, best_train_df, best_val_df, metrics_df)
    """
    # Calculate target class ratio from full dataset if not specified
    if target_class_ratio is None:
        target_class_ratio = df[label_column].mean()

    print(f"\nTrying {n_seeds} random seeds to find best class balance...")
    print(f"  Target validation ratio: {target_val_ratio*100:.1f}%")
    print(f"  Target class 1 ratio: {target_class_ratio*100:.1f}%")

    best_seed = None
    best_score = float('inf')
    best_train_df = None
    best_val_df = None
    all_metrics = []

    for seed in range(n_seeds):
        if seed % 20 == 0:
            print(f"  Testing seed {seed}/{n_seeds}...", end='\r')

        # Apply temporal splitting with this seed
        train_df, val_df, split_stats = adaptive_block_temporal_split(
            df, gap_size=gap_size, random_seed=seed,
            short_threshold=short_threshold, medium_threshold=medium_threshold, long_threshold=long_threshold
        )

        # Skip if split failed (no data retained)
        if len(train_df) == 0 or len(val_df) == 0:
            continue

        # Calculate class distributions
        train_class_ratio = train_df[label_column].mean()
        val_class_ratio = val_df[label_column].mean()

        # Calculate train/val split ratio
        total_retained = len(train_df) + len(val_df)
        actual_val_ratio = len(val_df) / total_retained if total_retained > 0 else 0

        # Calculate balance score (lower is better)
        # Penalize deviation from target class ratio AND target val ratio
        class_deviation = abs(train_class_ratio - target_class_ratio) + abs(val_class_ratio - target_class_ratio)
        split_deviation = abs(actual_val_ratio - target_val_ratio)

        # Weight class balance higher than split ratio
        score = class_deviation + 0.3 * split_deviation

        # Track metrics
        metrics = {
            'seed': seed,
            'score': score,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'val_ratio': actual_val_ratio * 100,
            'train_class_0': (train_df[label_column] == 0).sum(),
            'train_class_1': (train_df[label_column] == 1).sum(),
            'train_class_1_pct': train_class_ratio * 100,
            'val_class_0': (val_df[label_column] == 0).sum(),
            'val_class_1': (val_df[label_column] == 1).sum(),
            'val_class_1_pct': val_class_ratio * 100,
        }
        all_metrics.append(metrics)

        # Update best if this is better
        if score < best_score:
            best_score = score
            best_seed = seed
            best_train_df = train_df.copy()
            best_val_df = val_df.copy()

    print(f"  Testing seed {n_seeds}/{n_seeds}... Done!")

    # Create summary DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    # Calculate actual statistics for best split
    best_train_class_0 = (best_train_df[label_column] == 0).sum()
    best_train_class_1 = (best_train_df[label_column] == 1).sum()
    best_val_class_0 = (best_val_df[label_column] == 0).sum()
    best_val_class_1 = (best_val_df[label_column] == 1).sum()

    best_train_total = len(best_train_df)
    best_val_total = len(best_val_df)
    best_total = best_train_total + best_val_total

    best_train_pct = best_train_total / best_total * 100
    best_val_pct = best_val_total / best_total * 100

    best_train_class_1_pct = best_train_class_1 / best_train_total * 100
    best_val_class_1_pct = best_val_class_1 / best_val_total * 100

    # Get the split stats to access discard info
    _, _, temp_split_stats = adaptive_block_temporal_split(
        df, gap_size=gap_size, random_seed=best_seed,
        short_threshold=short_threshold, medium_threshold=medium_threshold, long_threshold=long_threshold
    )

    print(f"\n✓ Best seed found: {best_seed} (score={best_score:.4f})")
    print(f"\n{'='*85}")
    print("SPLIT COMPARISON: TARGET vs ACTUAL")
    print(f"{'='*85}")
    print(f"\n{'Metric':<35} {'Target':<15} {'Actual':<15} {'Deviation':<15}")
    print(f"{'-'*85}")

    # Data retention
    print(f"{'Data Retention:':<35}")
    total_windows = temp_split_stats['total_windows']
    retained_windows = temp_split_stats['retained_windows']
    discarded_windows = total_windows - retained_windows
    discard_pct = discarded_windows / total_windows * 100 if total_windows > 0 else 0

    print(f"  {'Total windows available':<33} {'-':<15} {total_windows:<15}")
    print(f"  {'Windows retained (train+val)':<33} {'-':<15} {retained_windows:<15}")
    print(f"  {'Windows discarded (gaps)':<33} {'-':<15} {discarded_windows:<15}")
    print(f"  {'Retention %':<33} {'-':<15} {temp_split_stats['retention_pct']:<15.1f}")
    print(f"  {'Discard %':<33} {'-':<15} {discard_pct:<15.1f}")
    tracks_used = temp_split_stats['num_tracks_used']
    tracks_total = temp_split_stats['num_tracks_total']
    print(f"  {'Tracks used / total':<33} {'-':<15} {f'{tracks_used}/{tracks_total}':<15}")

    # Train/Val split ratio
    target_train_pct = (1 - target_val_ratio) * 100
    target_val_pct = target_val_ratio * 100
    train_split_dev = abs(best_train_pct - target_train_pct)
    val_split_dev = abs(best_val_pct - target_val_pct)

    print(f"\n{'Train/Val Split:':<35}")
    print(f"  {'Train samples %':<33} {target_train_pct:<15.1f} {best_train_pct:<15.1f} {train_split_dev:<15.1f}")
    print(f"  {'Val samples %':<33} {target_val_pct:<15.1f} {best_val_pct:<15.1f} {val_split_dev:<15.1f}")

    # Class distribution
    target_class_1_pct = target_class_ratio * 100
    train_class_dev = abs(best_train_class_1_pct - target_class_1_pct)
    val_class_dev = abs(best_val_class_1_pct - target_class_1_pct)

    print(f"\n{'Class Distribution (% class 1):':<35}")
    print(f"  {'Train set':<33} {target_class_1_pct:<15.1f} {best_train_class_1_pct:<15.1f} {train_class_dev:<15.1f}")
    print(f"  {'Val set':<33} {target_class_1_pct:<15.1f} {best_val_class_1_pct:<15.1f} {val_class_dev:<15.1f}")

    # Absolute counts
    print(f"\n{'Absolute Counts:':<35}")
    print(f"  {'Train total':<33} {'-':<15} {best_train_total:<15}")
    print(f"    {'Class 0':<31} {'-':<15} {best_train_class_0:<15}")
    print(f"    {'Class 1':<31} {'-':<15} {best_train_class_1:<15}")
    print(f"  {'Val total':<33} {'-':<15} {best_val_total:<15}")
    print(f"    {'Class 0':<31} {'-':<15} {best_val_class_0:<15}")
    print(f"    {'Class 1':<31} {'-':<15} {best_val_class_1:<15}")

    print(f"\n{'Recommendations:':<35}")
    if best_val_class_1_pct < 40:
        print(f"  ⚠️  Val set has low class 1 representation ({best_val_class_1_pct:.1f}%)")
        print(f"     Consider using --augment_to_balance during training")
    elif best_val_class_1_pct > 60:
        print(f"  ⚠️  Val set has high class 1 representation ({best_val_class_1_pct:.1f}%)")
        print(f"     Validation may not reflect real-world distribution")
    else:
        print(f"  ✓ Val set has good class balance ({best_val_class_1_pct:.1f}%)")

    if best_train_class_1_pct < 30:
        print(f"  ⚠️  Train set has low class 1 ({best_train_class_1_pct:.1f}%)")
        print(f"     Use --augment_to_balance to balance training data")
    elif abs(best_train_class_1_pct - best_val_class_1_pct) > 15:
        print(f"  ⚠️  Large train/val class distribution mismatch ({abs(best_train_class_1_pct - best_val_class_1_pct):.1f}% difference)")
        print(f"     Consider increasing --n-seeds for better optimization")
    else:
        print(f"  ✓ Train/val class distributions are well matched")

    print(f"{'='*80}\n")

    return best_seed, best_train_df, best_val_df, metrics_df


def save_temporal_split_report(split_stats, best_seed, train_df, val_df,
                               label_column, output_path):
    """Save detailed temporal split statistics report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ADAPTIVE TEMPORAL SPLIT REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Gap size: {split_stats['gap_size']} windows (zero temporal leakage)\n")
        f.write(f"  Best random seed: {best_seed}\n")
        f.write(f"  Adaptive block sizes:\n")
        f.write(f"    - Short tracks (10-19 windows): B_train=4, B_val=1\n")
        f.write(f"    - Medium tracks (20-29 windows): B_train=8, B_val=2\n")
        f.write(f"    - Long tracks (30+ windows): B_train=12, B_val=3\n\n")

        f.write("Overall Statistics:\n")
        f.write(f"  Total windows in dataset: {split_stats['total_windows']}\n")
        f.write(f"  Retained windows: {split_stats['retained_windows']} ({split_stats['retention_pct']:.1f}%)\n")
        f.write(f"  Discarded windows (gaps): {split_stats['discarded_windows']} ({split_stats['discard_pct']:.1f}%)\n\n")

        f.write(f"  Training windows: {split_stats['train_windows']} ({split_stats['train_pct']:.1f}%)\n")
        f.write(f"  Validation windows: {split_stats['val_windows']} ({split_stats['val_pct']:.1f}%)\n\n")

        f.write(f"Track Statistics:\n")
        f.write(f"  Total tracks in dataset: {split_stats['num_tracks_total']}\n")
        f.write(f"  Tracks used: {split_stats['num_tracks_used']} ({split_stats['num_tracks_used']/split_stats['num_tracks_total']*100:.1f}%)\n")
        f.write(f"  Tracks excluded (< 10 windows): {split_stats['num_tracks_excluded']}\n\n")

        # Class distribution
        train_class_0 = (train_df[label_column] == 0).sum()
        train_class_1 = (train_df[label_column] == 1).sum()
        val_class_0 = (val_df[label_column] == 0).sum()
        val_class_1 = (val_df[label_column] == 1).sum()

        f.write(f"Class Distribution:\n")
        f.write(f"  Training set:\n")
        f.write(f"    Class 0: {train_class_0} ({train_class_0/len(train_df)*100:.1f}%)\n")
        f.write(f"    Class 1: {train_class_1} ({train_class_1/len(train_df)*100:.1f}%)\n")
        f.write(f"    Ratio: {train_class_0/train_class_1:.2f}:1\n\n")
        f.write(f"  Validation set:\n")
        f.write(f"    Class 0: {val_class_0} ({val_class_0/len(val_df)*100:.1f}%)\n")
        f.write(f"    Class 1: {val_class_1} ({val_class_1/len(val_df)*100:.1f}%)\n")
        f.write(f"    Ratio: {val_class_0/val_class_1:.2f}:1\n\n")

        f.write(f"Per-Track Statistics ({len(split_stats['track_stats'])} tracks):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Track ID':<15} {'Len':<6} {'B_tr':<6} {'Train':<8} {'Val':<8} {'Disc':<8} {'Disc%':<8} {'Start'}\n")
        f.write("-" * 80 + "\n")

        for track in split_stats['track_stats'][:50]:  # Show first 50 tracks
            f.write(f"{track['trackid']:<15} {track['length']:<6} {track['B_train']:<6} "
                   f"{track['train']:<8} {track['val']:<8} {track['discarded']:<8} "
                   f"{track['discard_pct']:<8.1f} {track['started_with']}\n")

        if len(split_stats['track_stats']) > 50:
            f.write(f"... ({len(split_stats['track_stats']) - 50} more tracks)\n")

        f.write("\nTemporal Leakage Guarantee:\n")
        f.write(f"  Minimum separation: {split_stats['gap_size']} windows\n")
        f.write(f"  Window overlap: 0.8 ms (80% with 1.0ms window, 0.2ms step)\n")
        f.write(f"  Required separation for zero overlap: 5 windows\n")
        f.write(f"  Status: {'✓ ZERO LEAKAGE GUARANTEED' if split_stats['gap_size'] >= 5 else '✗ LEAKAGE POSSIBLE'}\n")


def save_seed_analysis(metrics_df, output_path):
    """Save per-seed metrics to CSV for analysis."""
    metrics_df.to_csv(output_path, index=False, encoding='utf-8')


def generate_statistics_report(df_original, df_filtered, df_balanced, dimensions):
    """
    Generate detailed statistics report showing distribution changes.

    Args:
        df_original: Original dataset
        df_filtered: After signal filtering
        df_balanced: After downsampling
        dimensions: List of dimension columns

    Returns:
        String report
    """
    report = []
    report.append("="*80)
    report.append("DATASET BALANCING STATISTICS REPORT")
    report.append("="*80)

    # Overall statistics
    report.append("\n### OVERALL STATISTICS ###\n")

    for df, stage, label in [
        (df_original, 'Original', 'ORIGINAL'),
        (df_filtered, 'Filtered', 'AFTER SIGNAL FILTERING'),
        (df_balanced, 'Balanced', 'AFTER DOWNSAMPLING')
    ]:
        class_counts = df['has_porosity'].value_counts()
        class_0 = class_counts.get(0, 0)
        class_1 = class_counts.get(1, 0)
        ratio = class_0 / class_1 if class_1 > 0 else 0

        report.append(f"{label}:")
        report.append(f"  Total samples: {len(df)}")
        report.append(f"  Class 0 (no porosity): {class_0}")
        report.append(f"  Class 1 (has porosity): {class_1}")
        report.append(f"  Ratio (0:1): {ratio:.2f}:1")
        report.append("")

    # Dimension-wise statistics
    report.append("\n### DISTRIBUTION ACROSS DIMENSIONS ###\n")

    for dim in dimensions:
        if dim not in df_balanced.columns:
            continue

        report.append(f"\n{dim.upper()}:")

        # Original distribution
        orig_dist = df_original[df_original['has_porosity'] == 0][dim].value_counts().sort_index()

        # Balanced distribution
        bal_dist = df_balanced[df_balanced['has_porosity'] == 0][dim].value_counts().sort_index()

        # Compare
        all_values = sorted(set(orig_dist.index) | set(bal_dist.index))

        for val in all_values[:10]:  # Show first 10 values
            orig_count = orig_dist.get(val, 0)
            bal_count = bal_dist.get(val, 0)
            retention = (bal_count / orig_count * 100) if orig_count > 0 else 0

            report.append(f"  {val}: {orig_count} → {bal_count} ({retention:.1f}% retained)")

        if len(all_values) > 10:
            report.append(f"  ... ({len(all_values) - 10} more values)")

    # Signal quality distribution
    report.append("\n### SIGNAL QUALITY DISTRIBUTION (CLASS 0) ###\n")

    for df, label in [
        (df_original, 'Original'),
        (df_filtered, 'Filtered'),
        (df_balanced, 'Balanced')
    ]:
        class_0_df = df[df['has_porosity'] == 0]

        if len(class_0_df) > 0 and 'mean' in class_0_df.columns:
            report.append(f"{label}:")
            report.append(f"  Mean intensity: {class_0_df['mean'].mean():.2f} ± {class_0_df['mean'].std():.2f}")
            report.append(f"  Min: {class_0_df['mean'].min():.2f}, Max: {class_0_df['mean'].max():.2f}")
            report.append(f"  Samples with mean < 10: {(class_0_df['mean'] < 10).sum()} ({(class_0_df['mean'] < 10).mean()*100:.1f}%)")
            report.append("")

    report.append("="*80)

    return "\n".join(report)


def list_available_datasets():
    """List all prepared datasets from dataset_definitions directories."""
    print("=" * 80)
    print("AVAILABLE PREPARED DATASETS")
    print("=" * 80)

    found_datasets = False

    # Search for dataset_definitions in CWT data directories
    # Navigate up from deep CWT directory structure to find the root
    search_paths = []
    try:
        from config import get_default_cwt_data_dir
        cwt_data_path = Path(get_default_cwt_data_dir())

        # Look for the directory containing "dataset_definitions"
        current = cwt_data_path
        for _ in range(10):  # Safety limit
            if current.exists():
                dataset_defs = current / 'dataset_definitions'
                if dataset_defs.exists():
                    search_paths.append(dataset_defs)
                    break
            current = current.parent
            if current == current.parent:  # Reached root
                break

        # If not found via search, try the common pattern: go up to "CWT_labelled_windows" level
        if not search_paths:
            current = cwt_data_path
            while current.name not in ['', '/'] and not current.name.startswith('CWT'):
                current = current.parent
            if current.name.startswith('CWT'):
                dataset_defs = current / 'dataset_definitions'
                if dataset_defs.exists():
                    search_paths.append(dataset_defs)
    except:
        pass

    for datasets_dir in search_paths:
        if not datasets_dir.exists():
            continue

        for dataset_dir in sorted(datasets_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            config_file = dataset_dir / 'dataset_config.json'
            if not config_file.exists():
                continue

            found_datasets = True
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            print(f"\n{dataset_dir.name}:")
            print(f"  Location: {dataset_dir.parent}")
            print(f"  Created: {config['created_date']}")
            print(f"  Mode: {config['statistics']['mode']}")

            if config['statistics']['mode'] == 'k_fold_cv':
                print(f"  K-folds: {config['preparation_params']['k_folds']}")
                print(f"  Avg train samples: {config['statistics'].get('avg_train_samples', 'N/A')}")
                print(f"  Avg val samples: {config['statistics'].get('avg_val_samples', 'N/A')}")
            else:
                print(f"  Train samples: {config['statistics'].get('train_samples', 'N/A')}")
                print(f"  Test samples: {config['statistics'].get('test_samples', 'N/A')}")

            print(f"  Temporal split: {config['preparation_params']['temporal_split']}")
            if 'retention_pct' in config['statistics']:
                print(f"  Retention: {config['statistics']['retention_pct']:.1f}%")

    if not found_datasets:
        print("\nNo prepared datasets found.")
        print("Create datasets with: python ml/prepare_training_dataset.py --temporal-split --dataset-name <name>")


def score_test_holdout_split(df, train_tracks, test_tracks, target_test_fraction=0.2):
    """
    Calculate balance score for a test holdout split.

    Score is based on:
    - Class balance deviation from 50/50 in test set
    - Split fraction deviation from target

    Args:
        df (pd.DataFrame): Label dataframe with 'trackid' and 'has_porosity' columns
        train_tracks (list): List of train track IDs
        test_tracks (list): List of test track IDs
        target_test_fraction (float): Target test fraction

    Returns:
        dict: Metrics including score, class percentages, counts
    """
    # Get test set data
    test_df = df[df['trackid'].isin(test_tracks)]

    # Calculate class balance
    test_class_counts = test_df['has_porosity'].value_counts()
    test_class_0 = test_class_counts.get(0, 0)
    test_class_1 = test_class_counts.get(1, 0)
    total_test = len(test_df)

    if total_test > 0:
        test_class_1_pct = (test_class_1 / total_test) * 100
    else:
        test_class_1_pct = 0

    # Calculate split fraction
    total_tracks = len(train_tracks) + len(test_tracks)
    actual_test_fraction = len(test_tracks) / total_tracks if total_tracks > 0 else 0

    # Calculate deviations
    class_deviation = abs(test_class_1_pct - 50.0)  # Distance from 50/50 balance
    split_deviation = abs(actual_test_fraction - target_test_fraction) * 100  # As percentage

    # Combined score (lower is better)
    score = class_deviation + 0.3 * split_deviation

    return {
        'score': score,
        'test_tracks_count': len(test_tracks),
        'test_samples_count': total_test,
        'test_class_0': test_class_0,
        'test_class_1': test_class_1,
        'test_class_1_pct': test_class_1_pct,
        'actual_test_fraction': actual_test_fraction,
        'class_deviation': class_deviation,
        'split_deviation': split_deviation
    }


def generate_test_holdout_candidates(df, logbook, test_fraction=0.2, n_candidates=100, top_n=5, max_overlap=0.95):
    """
    Generate multiple test holdout split candidates and rank by balance score.

    Tests different random seeds to create variations in which tracks go to test set,
    then ranks them by class balance quality. Filters out near-duplicate candidates
    (>95% track overlap) to show distinct options.

    Args:
        df (pd.DataFrame): Label dataframe with 'trackid' and 'has_porosity' columns
        logbook (pd.DataFrame): Logbook with track metadata
        test_fraction (float): Target fraction of tracks for test set
        n_candidates (int): Number of random seeds to test (default: 100)
        top_n (int): Number of top-ranked candidates to return (default: 5)
        max_overlap (float): Maximum allowed track overlap fraction (default: 0.95)

    Returns:
        list: List of candidate dictionaries with metrics, sorted by score (best first)
    """
    print(f"\nGenerating {n_candidates} test holdout candidates...")

    all_candidates = []

    for seed in range(n_candidates):
        # Generate split with this seed
        train_tracks, test_tracks = select_test_tracks_preserve_diversity(
            df, logbook, test_fraction=test_fraction, seed=seed
        )

        # Calculate balance score
        metrics = score_test_holdout_split(df, train_tracks, test_tracks, target_test_fraction=test_fraction)

        # Store candidate
        all_candidates.append({
            'seed': seed,
            'train_tracks': train_tracks,
            'test_tracks': test_tracks,
            'metrics': metrics
        })

    # Sort by score (lower is better)
    all_candidates.sort(key=lambda x: x['metrics']['score'])

    # Filter for diversity: skip candidates with >max_overlap track overlap
    selected_candidates = []

    for candidate in all_candidates:
        # First candidate is always selected
        if len(selected_candidates) == 0:
            selected_candidates.append(candidate)
            continue

        # Check overlap with all previously selected candidates
        candidate_tracks = set(candidate['test_tracks'])
        is_diverse = True

        for selected in selected_candidates:
            selected_tracks = set(selected['test_tracks'])
            overlap = len(candidate_tracks & selected_tracks)
            overlap_fraction = overlap / len(candidate_tracks) if len(candidate_tracks) > 0 else 0

            if overlap_fraction > max_overlap:
                is_diverse = False
                break

        # Add if diverse enough
        if is_diverse:
            selected_candidates.append(candidate)

        # Stop if we have enough candidates
        if len(selected_candidates) >= top_n:
            break

    print(f"Selected {len(selected_candidates)} unique candidates from {len(all_candidates)} total (removed duplicates with >{max_overlap*100:.0f}% overlap)")

    return selected_candidates


def generate_test_holdout_candidates_random_search(df, logbook, test_fraction=0.2,
                                                     n_candidates=500, top_n=5,
                                                     max_overlap=0.5):
    """
    Generate multiple diverse test holdout options using random search.

    For each candidate:
    1. Run random search with different seed
    2. Find best combination for that seed (5-8 tracks, ~20% windows)
    3. Score and collect

    Then filter duplicates (>50% overlap) and return top N.

    Args:
        df (pd.DataFrame): Label dataframe with 'trackid' and 'has_porosity'
        logbook (pd.DataFrame): Logbook with track metadata
        test_fraction (float): Target fraction for test set (default: 0.2)
        n_candidates (int): Number of random seeds to test (default: 500)
        top_n (int): Number of top-ranked candidates to return (default: 5)
        max_overlap (float): Maximum allowed track overlap fraction (default: 0.5)

    Returns:
        list: List of candidate dictionaries with metrics, sorted by score (best first)
    """
    print(f"\nGenerating {n_candidates} test holdout candidates via random search...")

    all_candidates = []

    for seed in range(n_candidates):
        # Run random search with this seed
        train_tracks, test_tracks = select_test_tracks_random_search(
            df, logbook,
            test_fraction=test_fraction,
            seed=seed,
            n_attempts=1000,  # Each seed tries 1000 random combinations
            window_tolerance=0.02
        )

        # Calculate final score
        metrics = score_test_holdout_split(df, train_tracks, test_tracks, test_fraction)

        all_candidates.append({
            'seed': seed,
            'train_tracks': train_tracks,
            'test_tracks': test_tracks,
            'metrics': metrics
        })

        if (seed + 1) % 50 == 0:
            print(f"  Processed {seed + 1}/{n_candidates} seeds...")

    # Sort by score
    all_candidates.sort(key=lambda x: x['metrics']['score'])

    # Filter duplicates (>50% overlap)
    selected_candidates = []

    for candidate in all_candidates:
        if len(selected_candidates) == 0:
            selected_candidates.append(candidate)
            continue

        candidate_tracks = set(candidate['test_tracks'])
        is_diverse = True

        for selected in selected_candidates:
            selected_tracks = set(selected['test_tracks'])
            overlap = len(candidate_tracks & selected_tracks)
            overlap_fraction = overlap / len(candidate_tracks) if len(candidate_tracks) > 0 else 0

            if overlap_fraction > max_overlap:
                is_diverse = False
                break

        if is_diverse:
            selected_candidates.append(candidate)

        if len(selected_candidates) >= top_n:
            break

    print(f"Selected {len(selected_candidates)} unique candidates from {len(all_candidates)} total (removed duplicates with >{max_overlap*100:.0f}% overlap)")

    return selected_candidates


def select_test_tracks_preserve_diversity(df, logbook, test_fraction=0.2, min_samples_per_class=10, seed=None):
    """
    Select test tracks preserving training diversity and ensuring test diversity.

    Priority constraints:
    1. Training must have all unique parameter combinations
    2. Test must have both classes well represented
    3. Test should have multiple regimes
    4. Prefer duplicates for test set

    Args:
        df (pd.DataFrame): Label dataframe with 'trackid' and 'has_porosity' columns
        logbook (pd.DataFrame): Logbook with track metadata
        test_fraction (float): Target fraction of tracks for test set (default: 0.2)
        min_samples_per_class (int): Minimum samples per class in test set (default: 10)
        seed (int, optional): Random seed for shuffling track order in multi-track groups

    Returns:
        tuple: (train_tracks, test_tracks) - lists of track IDs
    """
    # Get column mapping dictionary
    col_dict = define_collumn_labels()

    # Check which columns are missing and need to be added from logbook
    required_cols = ['laser_power', 'scan_speed', 'base_type', 'material', 'laser_mode', 'regime']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        # Map missing script-friendly names to Excel headers
        script_to_excel = {}
        for col in missing_cols:
            if col in col_dict:
                script_to_excel[col] = col_dict[col][0]
            else:
                print(f"Warning: '{col}' not found in col_dict")

        # Select only missing columns from logbook (include trackid for merge)
        excel_cols = ['trackid'] + list(script_to_excel.values())

        # Check which Excel columns actually exist in logbook
        available_excel_cols = [c for c in excel_cols if c in logbook.columns]
        missing_excel_cols = [c for c in excel_cols if c not in logbook.columns]

        if missing_excel_cols:
            print(f"Warning: Columns not found in logbook: {missing_excel_cols}")

        if len(available_excel_cols) > 1:  # More than just 'trackid'
            logbook_subset = logbook[available_excel_cols].copy()

            # Rename to script-friendly names
            excel_to_script = {v: k for k, v in script_to_excel.items() if v in available_excel_cols}
            logbook_subset.rename(columns=excel_to_script, inplace=True)

            # Merge with df to add missing parameters
            df_with_params = df.merge(logbook_subset, on='trackid', how='left')
        else:
            # No valid columns to merge
            df_with_params = df
    else:
        # All columns already exist, use df directly
        df_with_params = df

    param_cols = ['laser_power', 'scan_speed', 'base_type', 'material', 'laser_mode']

    # Group tracks by parameters
    track_param_groups = {}
    for track_id in df['trackid'].unique():
        track_data = df_with_params[df_with_params['trackid'] == track_id].iloc[0]
        param_tuple = tuple(track_data[col] for col in param_cols)

        if param_tuple not in track_param_groups:
            track_param_groups[param_tuple] = []
        track_param_groups[param_tuple].append(track_id)

    # Step 1: Assign tracks ensuring training diversity
    train_tracks = []
    test_candidates = []

    for param_combo, track_ids in track_param_groups.items():
        if len(track_ids) == 1:
            # Single track - must go to training
            train_tracks.extend(track_ids)
        else:
            # Multiple tracks - keep majority in training
            # Shuffle order if seed provided to create different splits
            if seed is not None:
                import random
                rng = random.Random(seed)
                track_ids_copy = track_ids.copy()
                rng.shuffle(track_ids_copy)
            else:
                track_ids_copy = track_ids

            n_train = max(1, int(len(track_ids_copy) * (1 - test_fraction)))
            train_tracks.extend(track_ids_copy[:n_train])
            test_candidates.extend(track_ids_copy[n_train:])

    # Step 2: Select test tracks ensuring diversity
    test_tracks = []
    remaining_candidates = test_candidates.copy()

    # 2a: Ensure both classes represented
    for label in [0, 1]:
        label_tracks = df[
            (df['trackid'].isin(remaining_candidates)) &
            (df['has_porosity'] == label)
        ]['trackid'].unique()

        if len(label_tracks) > 0:
            n_take = min(len(label_tracks), max(2, len(test_candidates) // 10))
            test_tracks.extend(label_tracks[:n_take])
            remaining_candidates = [t for t in remaining_candidates if t not in label_tracks[:n_take]]

    # 2b: Ensure regime diversity
    if 'regime' in df_with_params.columns:
        for regime in df_with_params['regime'].dropna().unique():
            regime_tracks = df_with_params[
                (df_with_params['trackid'].isin(remaining_candidates)) &
                (df_with_params['regime'] == regime)
            ]['trackid'].unique()

            if len(regime_tracks) > 0 and regime_tracks[0] not in test_tracks:
                test_tracks.append(regime_tracks[0])
                if regime_tracks[0] in remaining_candidates:
                    remaining_candidates.remove(regime_tracks[0])

    # 2c: Fill to target test fraction
    target_test_count = int(len(df['trackid'].unique()) * test_fraction)
    n_remaining = max(0, target_test_count - len(test_tracks))

    if n_remaining > 0 and len(remaining_candidates) > 0:
        test_tracks.extend(remaining_candidates[:n_remaining])

    return train_tracks, test_tracks


def select_test_tracks_random_search(df, logbook, test_fraction=0.2, seed=42,
                                      n_attempts=1000, window_tolerance=0.02):
    """
    Select test tracks using random search with multi-objective scoring.

    Algorithm:
    1. Generate n_attempts random track selections (5-8 tracks each)
    2. Filter by window count (within tolerance of target)
    3. Score each by class balance + regime diversity
    4. Return best scoring combination

    Args:
        df (pd.DataFrame): Label dataframe with 'trackid' and 'has_porosity'
        logbook (pd.DataFrame): Logbook with track metadata
        test_fraction (float): Target fraction for test set (default: 0.2)
        seed (int): Random seed for reproducibility
        n_attempts (int): Number of random combinations to try (default: 1000)
        window_tolerance (float): Acceptable deviation from target windows (default: 0.02 = ±2%)

    Returns:
        tuple: (train_tracks, test_tracks)
    """
    import random

    # Setup
    all_tracks = df['trackid'].unique()
    total_windows = len(df)
    target_windows = int(total_windows * test_fraction)
    min_windows = int(total_windows * (test_fraction - window_tolerance))
    max_windows = int(total_windows * (test_fraction + window_tolerance))

    # Get regime info
    col_dict = define_collumn_labels()
    regime_col = col_dict['regime'][0]
    logbook_regime = logbook[['trackid', regime_col]].copy()
    logbook_regime.rename(columns={regime_col: 'regime'}, inplace=True)
    df_with_regime = df.merge(logbook_regime, on='trackid', how='left')
    total_regimes = df_with_regime['regime'].nunique()

    # Random search
    rng = random.Random(seed)
    valid_combinations = []

    for attempt in range(n_attempts):
        # Randomly select N tracks, where N is between 5-8
        n_tracks = rng.randint(5, 8)
        test_tracks = rng.sample(list(all_tracks), k=n_tracks)

        # Check window count
        test_windows = len(df[df['trackid'].isin(test_tracks)])

        if min_windows <= test_windows <= max_windows:
            # Calculate scores
            test_df = df[df['trackid'].isin(test_tracks)]

            # Class balance
            class_counts = test_df['has_porosity'].value_counts()
            class_1_pct = (class_counts.get(1, 0) / len(test_df)) * 100
            class_deviation = abs(class_1_pct - 50.0)

            # Regime diversity
            test_regimes = df_with_regime[
                df_with_regime['trackid'].isin(test_tracks)
            ]['regime'].nunique()
            regime_penalty = (total_regimes - test_regimes) * 10  # Penalty per missing regime

            # Window fraction deviation
            actual_fraction = test_windows / total_windows
            split_deviation = abs(actual_fraction - test_fraction) * 100

            # Composite score (lower is better)
            score = class_deviation + regime_penalty + 0.3 * split_deviation

            valid_combinations.append({
                'test_tracks': test_tracks,
                'train_tracks': [t for t in all_tracks if t not in test_tracks],
                'score': score,
                'class_deviation': class_deviation,
                'regime_penalty': regime_penalty,
                'split_deviation': split_deviation,
                'test_windows': test_windows,
                'test_regimes': test_regimes,
                'class_1_pct': class_1_pct
            })

    if not valid_combinations:
        raise ValueError(
            f"No valid combinations found in {n_attempts} attempts. "
            f"Try increasing n_attempts or window_tolerance."
        )

    # Sort by score and return best
    valid_combinations.sort(key=lambda x: x['score'])
    best = valid_combinations[0]

    print(f"Random search: {len(valid_combinations)}/{n_attempts} valid combinations")
    print(f"Best score: {best['score']:.2f} "
          f"(class_dev={best['class_deviation']:.1f}, "
          f"regime_penalty={best['regime_penalty']:.0f}, "
          f"regimes={best['test_regimes']}/{total_regimes})")

    return best['train_tracks'], best['test_tracks']


def validate_test_set_diversity(test_df, df_with_params, min_samples_per_class=10):
    """
    Validate test set has sufficient diversity.

    Args:
        test_df (pd.DataFrame): Test set dataframe with 'has_porosity' column
        df_with_params (pd.DataFrame): Full dataframe with parameters
        min_samples_per_class (int): Minimum samples per class required

    Returns:
        tuple: (is_valid, warnings) - validation status and list of warning messages
    """
    warnings = []

    # Check 1: Both classes
    class_counts = test_df['has_porosity'].value_counts()
    if len(class_counts) < 2:
        warnings.append("CRITICAL: Test set missing one class!")
        return False, warnings

    # Check 2: Minimum samples per class
    for label, count in class_counts.items():
        if count < min_samples_per_class:
            warnings.append(f"Warning: Class {label} has only {count} samples (< {min_samples_per_class})")

    # Check 3: Regime diversity
    test_with_params = test_df.merge(df_with_params[['trackid', 'regime']], on='trackid', how='left')
    if 'regime' in test_with_params.columns:
        regime_counts = test_with_params['regime'].value_counts()
        if len(regime_counts) < 2:
            warnings.append("Warning: Test set has limited regime diversity")

    is_valid = not any("CRITICAL" in w for w in warnings)
    return is_valid, warnings


def visualize_k_fold_splits(df, folds, output_path=None, dataset_name=None):
    """
    Visualize k-fold window assignments as horizontal bars.

    Shows how windows are distributed across train/val/discard for each fold,
    making overlap patterns immediately visible.

    Args:
        df (pd.DataFrame): Full dataset DataFrame
        folds (list): List of fold dictionaries from generate_k_fold_temporal_splits()
        output_path (Path): Optional path to save figure
        dataset_name (str): Optional dataset name for title

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    k_folds = len(folds)

    # Sort dataframe by track and window for consistent ordering
    df_sorted = df.sort_values(['trackid', 'window_n']).reset_index(drop=False)

    # Create mapping from original index to sorted position
    idx_to_pos = {orig_idx: pos for pos, orig_idx in enumerate(df_sorted['index'])}

    # Create figure
    fig, ax = plt.subplots(figsize=(16, k_folds * 0.8 + 2))

    colors = {
        'train': '#1f77b4',      # Blue
        'val': '#ff7f0e',        # Orange
        'discard': '#7f7f7f'     # Gray
    }

    # For each fold, create assignment array
    for fold_idx, fold in enumerate(folds):
        # Initialize all as discarded
        assignments = ['discard'] * len(df_sorted)

        # Mark training windows
        for orig_idx in fold['train_indices']:
            if orig_idx in idx_to_pos:
                assignments[idx_to_pos[orig_idx]] = 'train'

        # Mark validation windows
        for orig_idx in fold['val_indices']:
            if orig_idx in idx_to_pos:
                assignments[idx_to_pos[orig_idx]] = 'val'

        # Plot as horizontal bar
        y_pos = k_folds - fold_idx - 1  # Reverse so Fold 0 is at top

        # Create segments for continuous colors
        current_color = assignments[0]
        start_x = 0

        for x in range(1, len(assignments)):
            if assignments[x] != current_color:
                # Draw segment
                ax.barh(y_pos, x - start_x, left=start_x, height=0.8,
                       color=colors[current_color], edgecolor='none')
                start_x = x
                current_color = assignments[x]

        # Draw final segment
        ax.barh(y_pos, len(assignments) - start_x, left=start_x, height=0.8,
               color=colors[current_color], edgecolor='none')

    # Formatting
    ax.set_yticks(range(k_folds))
    ax.set_yticklabels([f'Fold {i}' for i in range(k_folds)])
    ax.set_xlabel('Window Index (sorted by trackid, window_n)')

    # Title with dataset name if provided
    if dataset_name:
        ax.set_title(f'{dataset_name}: K-Fold Window Assignments ({k_folds} folds)')
    else:
        ax.set_title(f'K-Fold Window Assignment Visualization ({k_folds} folds)')

    ax.set_xlim(0, len(df_sorted))

    # Legend
    legend_elements = [
        mpatches.Patch(color=colors['train'], label='Training'),
        mpatches.Patch(color=colors['val'], label='Validation'),
        mpatches.Patch(color=colors['discard'], label='Discarded (gaps)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Add vertical lines to show track boundaries (if not too many)
    if df_sorted['trackid'].nunique() < 50:
        track_changes = df_sorted['trackid'].ne(df_sorted['trackid'].shift())
        for boundary_idx in df_sorted[track_changes].index[1:]:  # Skip first
            ax.axvline(boundary_idx, color='black', linewidth=0.5, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Visualization saved to: {output_path}")

    return fig


def visualize_k_fold_per_track(df, folds, track_id, output_path=None):
    """
    Show window assignments for a specific track across all folds.

    Useful for debugging overlap - should show different windows in val across folds.

    Args:
        df (pd.DataFrame): Full dataset DataFrame
        folds (list): List of fold dictionaries
        track_id: Track ID to visualize
        output_path (Path): Optional path to save figure

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    k_folds = len(folds)

    track_df = df[df['trackid'] == track_id].sort_values('window_n')

    if len(track_df) == 0:
        print(f"Warning: Track {track_id} not found in dataset")
        return None

    fig, ax = plt.subplots(figsize=(14, k_folds * 0.6 + 1))

    colors = {'train': 'blue', 'val': 'orange', 'discard': 'gray'}

    for fold_idx, fold in enumerate(folds):
        y_pos = k_folds - fold_idx - 1

        for idx, row in track_df.iterrows():
            x = row['window_n']

            if idx in fold['train_indices']:
                color = colors['train']
            elif idx in fold['val_indices']:
                color = colors['val']
            else:
                color = colors['discard']

            ax.scatter(x, y_pos, color=color, s=50, marker='s')

    ax.set_yticks(range(k_folds))
    ax.set_yticklabels([f'Fold {i}' for i in range(k_folds)])
    ax.set_xlabel('Window Index')
    ax.set_title(f'Track {track_id}: Window Assignments Across Folds')
    ax.grid(axis='x', alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(color=colors['train'], label='Training'),
        mpatches.Patch(color=colors['val'], label='Validation'),
        mpatches.Patch(color=colors['discard'], label='Discarded (gaps)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Per-track visualization saved to: {output_path}")

    return fig


def calculate_validation_overlap(candidate_val_indices, existing_folds):
    """
    Calculate average validation overlap between candidate and existing folds.

    Args:
        candidate_val_indices (list): Validation indices for candidate fold
        existing_folds (list): List of previously created fold dictionaries

    Returns:
        float: Average overlap percentage (0.0 = no overlap, 1.0 = complete overlap)
    """
    if not existing_folds:
        return 0.0  # First fold, no comparison needed

    candidate_val_set = set(candidate_val_indices)
    if len(candidate_val_set) == 0:
        return 0.0

    overlap_scores = []
    for existing_fold in existing_folds:
        existing_val_set = set(existing_fold['val_indices'])
        overlap = candidate_val_set & existing_val_set
        overlap_pct = len(overlap) / len(candidate_val_set)
        overlap_scores.append(overlap_pct)

    return sum(overlap_scores) / len(overlap_scores)


def generate_k_fold_temporal_splits(df, k_folds, base_seed=42, label_column='has_porosity', dataset_name=None, output_dir=None):
    """
    Generate k independent temporal splits with different random seeds.

    Each fold gets its own zero-leakage train/val split using a different random seed,
    allowing k-fold CV while maintaining temporal independence in each fold.

    Args:
        df (pd.DataFrame): Full dataset with 'trackid' and 'window_n' columns
        k_folds (int): Number of folds to generate
        base_seed (int): Base random seed (default: 42)
        label_column (str): Column containing class labels (default: 'has_porosity')
        dataset_name (str): Name of dataset variant for visualization title
        output_dir (Path): Directory to save visualization

    Returns:
        list: List of k dictionaries, each containing:
            - 'fold_num': Fold number (0 to k-1)
            - 'seed': Random seed used for this fold
            - 'train_indices': Indices for training set
            - 'val_indices': Indices for validation set
            - 'train_samples': Number of training samples
            - 'val_samples': Number of validation samples
    """
    print(f"\nGenerating {k_folds} folds with similarity-based optimization...")
    print(f"Testing 100 candidate seeds per fold")
    print(f"Primary criterion: Minimize validation overlap")
    print(f"Constraint: Class balance within \u00b15% of 50/50 (45-55%)\n")

    folds = []

    for fold_num in range(k_folds):
        print(f"\nFold {fold_num + 1}/{k_folds}: Testing seeds...")

        # Test 100 candidate seeds
        candidate_seeds = list(range(base_seed + fold_num * 1000, base_seed + fold_num * 1000 + 100))

        # Collect all candidates with their metrics
        candidates = []
        for test_seed in candidate_seeds:
            # Apply temporal splitting with this test seed
            train_df, val_df, split_stats = adaptive_block_temporal_split(
                df, gap_size=5, random_seed=test_seed
            )

            # Calculate class balance
            val_class_1_pct = val_df[label_column].mean()

            # Calculate validation overlap with existing folds
            val_indices = val_df.index.tolist()
            overlap_score = calculate_validation_overlap(val_indices, folds)

            candidates.append({
                'seed': test_seed,
                'train_df': train_df,
                'val_df': val_df,
                'val_indices': val_indices,
                'val_class_1_pct': val_class_1_pct,
                'overlap_score': overlap_score
            })

        # For fold 0: optimize for class balance only
        if fold_num == 0:
            # Sort by how close to 50/50 balance
            candidates.sort(key=lambda x: abs(x['val_class_1_pct'] - 0.5))
            selected = candidates[0]
            print(f"  Fold 0: Optimizing for class balance")
            print(f"  Selected seed: {selected['seed']}")
            print(f"  Val class balance: {selected['val_class_1_pct']*100:.1f}%")

        # For subsequent folds: minimize overlap with balance constraint
        else:
            # Sort by overlap (ascending - lower is better)
            candidates.sort(key=lambda x: x['overlap_score'])

            # Try balance tolerances: ±5%, ±10%, ±15%, ±20%
            tolerances = [0.05, 0.10, 0.15, 0.20]
            selected = None
            used_tolerance = None

            for tolerance in tolerances:
                min_balance = 0.5 - tolerance
                max_balance = 0.5 + tolerance

                # Find first candidate that meets balance constraint
                for candidate in candidates:
                    if min_balance <= candidate['val_class_1_pct'] <= max_balance:
                        selected = candidate
                        used_tolerance = tolerance
                        break

                if selected:
                    break

            # Fallback: if no candidate meets even ±20%, just take lowest overlap
            if not selected:
                selected = candidates[0]
                used_tolerance = None
                print(f"  WARNING: No seed met balance constraints, using most dissimilar")

            print(f"  Selected seed: {selected['seed']}")
            print(f"  Avg overlap with previous folds: {selected['overlap_score']*100:.1f}%")
            print(f"  Val class balance: {selected['val_class_1_pct']*100:.1f}%")
            if used_tolerance:
                print(f"  Balance tolerance: \u00b1{used_tolerance*100:.0f}%")

        # Extract final results
        train_indices = selected['train_df'].index.tolist()
        val_indices = selected['val_indices']
        train_class_1_pct = selected['train_df'][label_column].mean() * 100
        val_class_1_pct = selected['val_class_1_pct'] * 100

        fold_info = {
            'fold_num': fold_num,
            'seed': selected['seed'],
            'train_indices': train_indices,
            'val_indices': val_indices,
            'train_samples': len(train_indices),
            'val_samples': len(val_indices)
        }

        folds.append(fold_info)

        print(f"  Train: {len(train_indices)} samples (class 1: {train_class_1_pct:.1f}%)")
        print(f"  Val: {len(val_indices)} samples (class 1: {val_class_1_pct:.1f}%)")

    # Calculate cross-fold validation set overlap statistics
    print(f"\n{'='*60}")
    print("CROSS-FOLD VALIDATION SET OVERLAP ANALYSIS")
    print(f"{'='*60}")

    for fold_num, fold in enumerate(folds):
        # Get validation indices for this fold
        this_val_set = set(fold['val_indices'])

        # Count overlaps with each other fold individually
        overlap_details = []
        total_overlap_indices = set()

        for other_fold_num, other_fold in enumerate(folds):
            if other_fold_num == fold_num:
                continue

            other_val_set = set(other_fold['val_indices'])
            overlap = this_val_set & other_val_set

            if len(overlap) > 0:
                overlap_pct = len(overlap) / len(this_val_set) * 100
                overlap_details.append(f"fold {other_fold_num}: {len(overlap)} ({overlap_pct:.1f}%)")
                total_overlap_indices.update(overlap)

        # Calculate total percentage that appear in at least one other fold's validation set
        total_overlap_pct = len(total_overlap_indices) / len(this_val_set) * 100 if len(this_val_set) > 0 else 0

        print(f"\nFold {fold_num} ({len(this_val_set)} val samples):")
        print(f"  {len(total_overlap_indices)} samples ({total_overlap_pct:.1f}%) appear in other folds' validation sets")
        if overlap_details:
            print(f"  Breakdown: {', '.join(overlap_details)}")
        else:
            print(f"  No overlap with any other fold's validation set")

    # Calculate overall statistics
    all_val_indices = set()
    for fold in folds:
        all_val_indices.update(fold['val_indices'])

    total_unique_val = len(all_val_indices)
    total_val_samples = sum(len(fold['val_indices']) for fold in folds)
    redundancy = (total_val_samples - total_unique_val) / total_val_samples * 100 if total_val_samples > 0 else 0

    print(f"\nOverall:")
    print(f"  Total validation samples across all folds: {total_val_samples}")
    print(f"  Unique validation samples: {total_unique_val}")
    print(f"  Redundancy: {redundancy:.1f}% of validation samples are duplicated across folds")

    # Generate visualization of fold assignments
    print(f"\n{'='*60}")
    print("GENERATING FOLD VISUALIZATION")
    print(f"{'='*60}")

    try:
        import matplotlib.pyplot as plt
        from pathlib import Path

        # Save visualization to dataset directory
        if output_dir is None:
            output_dir = Path('ml/datasets')
        output_dir.mkdir(parents=True, exist_ok=True)

        viz_filename = f"{dataset_name}_folds_visualization.png" if dataset_name else 'k_fold_visualization.png'
        output_path = output_dir / viz_filename

        fig = visualize_k_fold_splits(df, folds, output_path, dataset_name=dataset_name)
        plt.close(fig)
        print(f"Successfully generated visualization: {output_path}")
    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}")
        import traceback
        traceback.print_exc()

    return folds


def prepare_dataset_variant(df_filtered, args, logbook, active_filters, label_path, image_dir):
    """
    Prepare and save a dataset variant (k-fold CV or train/test split).

    Args:
        df_filtered (pd.DataFrame): Filtered dataset after Step 1
        args: Command-line arguments
        logbook (pd.DataFrame): Logbook with track metadata
        active_filters (list): List of active filter names for documentation
        label_path (Path): Path to the label file (used to determine output location)

    Returns:
        int: 0 on success, 1 on failure
    """
    from datetime import datetime

    print(f"\n{'='*80}")
    print(f"DATASET VARIANT PREPARATION: {args.dataset_name}")
    print(f"{'='*80}")

    # Create output directory structure in label file's parent directory
    datasets_root = label_path.parent / 'dataset_definitions'
    dataset_dir = datasets_root / args.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any legacy files from previous runs
    import glob

    # Remove old individual fold CSV files
    legacy_fold_csvs = glob.glob(str(dataset_dir / "fold_*_*.csv"))
    if legacy_fold_csvs:
        print(f"\nCleaning up {len(legacy_fold_csvs)} legacy fold CSV files...")
        for legacy_file in legacy_fold_csvs:
            Path(legacy_file).unlink()


    # Prepare dataset configuration metadata
    config = {
        'dataset_name': args.dataset_name,
        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_dir': str(image_dir),  # Store the image directory for classifier to use
        'label_file': 'trainval.csv',  # Always trainval.csv now
        'split_type': 'temporal' if (hasattr(args, 'temporal_split') and args.temporal_split) else 'stratified',
        'preparation_params': {
            'temporal_split': args.temporal_split if hasattr(args, 'temporal_split') else False,
            'signal_threshold': args.signal_threshold,
            'filters': active_filters,
            'k_folds': args.k_folds if args.k_folds > 1 else None,
            'test_holdout': args.test_holdout if args.test_holdout else None,
            'test_selection_strategy': args.test_selection_strategy if args.test_holdout else None,
        },
        'statistics': {}
    }

    # Mode 1: K-Fold CV (for hyperparameter tuning)
    if args.k_folds > 1:
        print(f"\nMode: K-FOLD CROSS-VALIDATION")
        print(f"K-folds: {args.k_folds}")
        print(f"Each fold gets independent temporal split with zero leakage")

        # Generate k-fold splits
        folds = generate_k_fold_temporal_splits(
            df_filtered, args.k_folds,
            dataset_name=args.dataset_name,
            output_dir=dataset_dir
        )

        # Create fold assignment column
        # Each row gets a string like "T0,V1,T2" meaning train in fold 0, val in fold 1, train in fold 2
        fold_column_name = f"{args.dataset_name}_folds"
        fold_assignments = {}

        for idx in df_filtered.index:
            assignments = []
            for fold in folds:
                fold_num = fold['fold_num']
                if idx in fold['train_indices']:
                    assignments.append(f"T{fold_num}")
                elif idx in fold['val_indices']:
                    assignments.append(f"V{fold_num}")
                else:
                    assignments.append(f"D{fold_num}")  # Discarded (gap)
            fold_assignments[idx] = ','.join(assignments)

        # Add fold assignments to dataframe
        df_with_folds = df_filtered.copy()
        df_with_folds[fold_column_name] = df_with_folds.index.map(fold_assignments)

        # Save trainval.csv with fold assignments
        output_columns = ['filename', 'has_porosity', fold_column_name]
        trainval_csv = dataset_dir / "trainval.csv"
        df_with_folds[output_columns].to_csv(trainval_csv, index=False, encoding='utf-8')

        print(f"\n✓ Saved: {trainval_csv.name} ({len(df_with_folds)} samples)")
        print(f"  Fold assignment column: '{fold_column_name}'")

        # Show example assignments
        print(f"\n  Example fold assignments:")
        for i, (idx, row) in enumerate(df_with_folds[output_columns].head(3).iterrows()):
            print(f"    {row['filename']}: {row[fold_column_name]}")

        # Print fold statistics
        for fold in folds:
            fold_num = fold['fold_num']
            print(f"  Fold {fold_num}: {fold['train_samples']} train, {fold['val_samples']} val")

        # Calculate statistics
        avg_train_samples = sum(f['train_samples'] for f in folds) / len(folds)
        avg_val_samples = sum(f['val_samples'] for f in folds) / len(folds)
        total_retention = sum(f['train_samples'] + f['val_samples'] for f in folds) / (len(df_filtered) * args.k_folds)

        config['statistics']['mode'] = 'k_fold_cv'
        config['statistics']['k_folds'] = args.k_folds
        config['statistics']['avg_train_samples'] = int(avg_train_samples)
        config['statistics']['avg_val_samples'] = int(avg_val_samples)
        config['statistics']['total_samples'] = len(df_with_folds)
        config['statistics']['retention_pct'] = total_retention * 100
        config['fold_column'] = fold_column_name  # Column containing fold assignments

        print(f"\nK-Fold Statistics:")
        print(f"  Average train samples per fold: {avg_train_samples:.0f}")
        print(f"  Average val samples per fold: {avg_val_samples:.0f}")
        print(f"  Average retention: {total_retention*100:.1f}%")

    # Mode 2: Train/Test Split (for final evaluation)
    elif args.test_holdout:
        print(f"\nMode: TRAIN/TEST SPLIT WITH HOLDOUT")
        print(f"Test fraction: {args.test_holdout*100:.0f}%")
        print(f"Test selection strategy: {args.test_selection_strategy}")

        # Select test tracks with diversity preservation
        if args.test_selection_strategy == 'preserve_diversity':
            # Check if user wants to choose from multiple candidates
            if args.test_holdout_candidates:
                # Generate and rank multiple candidates
                candidates = generate_test_holdout_candidates(
                    df_filtered, logbook,
                    test_fraction=args.test_holdout,
                    n_candidates=500,  # Test more seeds to find all unique splits
                    top_n=args.test_holdout_candidates
                )

                # Generate P-V maps for each candidate
                print(f"\nGenerating P-V maps for {len(candidates)} candidates...")
                from tools import generate_pv_map
                import matplotlib.pyplot as plt
                from pathlib import Path

                # Create temporary directory for P-V maps
                pv_map_dir = Path("temp_pv_maps")
                pv_map_dir.mkdir(exist_ok=True)

                # Get all unique trackids from dataset
                all_trackids = df_filtered['trackid'].unique().tolist()

                # Generate P-V map for each candidate
                pv_map_paths = []
                for i, candidate in enumerate(candidates, 1):
                    pv_map_path = pv_map_dir / f"option_{i}_seed_{candidate['seed']}.png"

                    # Generate P-V map with test tracks highlighted
                    fig, ax = generate_pv_map(
                        trackids=all_trackids,
                        output_path=str(pv_map_path),
                        highlight_trackids=candidate['test_tracks'],
                        figsize=(6, 5),
                        dpi=150,
                        font_size=10,
                        show_background_points=True,
                        show_led_contours=False
                    )
                    plt.close(fig)

                    pv_map_paths.append(pv_map_path)
                    print(f"  Generated: {pv_map_path.name}")

                # Display candidates in table format
                print(f"\nTop {len(candidates)} test holdout options (ranked by class balance):")
                print("-" * 95)
                print(f"{'#':<4} {'Seed':<8} {'Test Tracks':<13} {'Test Samples':<14} {'Class 0':<10} {'Class 1':<10} {'Score':<8}")
                print("-" * 95)

                for i, candidate in enumerate(candidates, 1):
                    m = candidate['metrics']
                    print(f"{i:<4} {candidate['seed']:<8} {m['test_tracks_count']:<13} {m['test_samples_count']:<14} "
                          f"{m['test_class_0']:<10} {m['test_class_1']:<10} {m['score']:<8.2f}")

                print("-" * 95)
                print(f"\nP-V maps showing test track distribution (red circles) saved to: {pv_map_dir.absolute()}")
                print("Review the P-V maps to see how test tracks are distributed across the processing space.")

                # Get user selection
                while True:
                    try:
                        user_input = input(f"\nSelect option 1-{len(candidates)} (or press Enter for best): ").strip()
                        if user_input == "":
                            selection = 0
                            print(f"Using best option (seed={candidates[0]['seed']})")
                            break
                        selection = int(user_input) - 1
                        if 0 <= selection < len(candidates):
                            print(f"Using option {selection + 1} (seed={candidates[selection]['seed']})")
                            break
                        else:
                            print(f"Please enter a number between 1 and {len(candidates)}")
                    except ValueError:
                        print("Invalid input. Please enter a number or press Enter.")

                # Use selected candidate
                selected = candidates[selection]
                train_tracks = selected['train_tracks']
                test_tracks = selected['test_tracks']

                # Clean up P-V map directory
                print(f"\nCleaning up temporary P-V maps...")
                for pv_path in pv_map_paths:
                    if pv_path.exists():
                        pv_path.unlink()
                pv_map_dir.rmdir()
                print("Temporary files removed.")

            elif args.test_holdout_seed is not None:
                # Use specific seed for deterministic behavior
                train_tracks, test_tracks = select_test_tracks_preserve_diversity(
                    df_filtered, logbook, test_fraction=args.test_holdout, seed=args.test_holdout_seed
                )
                print(f"Using seed: {args.test_holdout_seed}")
            else:
                # Default: no seed (original deterministic behavior)
                train_tracks, test_tracks = select_test_tracks_preserve_diversity(
                    df_filtered, logbook, test_fraction=args.test_holdout
                )
        elif args.test_selection_strategy == 'random_search':
            # Random search with multi-objective scoring
            if args.test_holdout_candidates:
                # Generate and rank multiple candidates
                candidates = generate_test_holdout_candidates_random_search(
                    df_filtered, logbook,
                    test_fraction=args.test_holdout,
                    n_candidates=500,
                    top_n=args.test_holdout_candidates
                )

                # Generate P-V maps for each candidate
                print(f"\nGenerating P-V maps for {len(candidates)} candidates...")
                from tools import generate_pv_map
                import matplotlib.pyplot as plt
                from pathlib import Path

                # Create temporary directory for P-V maps
                pv_map_dir = Path("temp_pv_maps")
                pv_map_dir.mkdir(exist_ok=True)

                # Get all unique trackids from dataset
                all_trackids = df_filtered['trackid'].unique().tolist()

                # Generate P-V map for each candidate
                pv_map_paths = []
                for i, candidate in enumerate(candidates, 1):
                    pv_map_path = pv_map_dir / f"option_{i}_seed_{candidate['seed']}.png"

                    # Generate P-V map with test tracks highlighted
                    fig, ax = generate_pv_map(
                        trackids=all_trackids,
                        output_path=str(pv_map_path),
                        highlight_trackids=candidate['test_tracks'],
                        figsize=(6, 5),
                        dpi=150,
                        font_size=10,
                        show_background_points=True,
                        show_led_contours=False
                    )
                    plt.close(fig)

                    pv_map_paths.append(pv_map_path)
                    print(f"  Generated: {pv_map_path.name}")

                # Display candidates in table format
                print(f"\nTop {len(candidates)} test holdout options (ranked by score):")
                print("-" * 95)
                print(f"{'#':<4} {'Seed':<8} {'Test Tracks':<13} {'Test Samples':<14} {'Class 0':<10} {'Class 1':<10} {'Score':<8}")
                print("-" * 95)

                for i, candidate in enumerate(candidates, 1):
                    m = candidate['metrics']
                    print(f"{i:<4} {candidate['seed']:<8} {m['test_tracks_count']:<13} {m['test_samples_count']:<14} "
                          f"{m['test_class_0']:<10} {m['test_class_1']:<10} {m['score']:<8.2f}")

                print("-" * 95)
                print(f"\nP-V maps showing test track distribution (red circles) saved to: {pv_map_dir.absolute()}")
                print("Review the P-V maps to see how test tracks are distributed across the processing space.")

                # Get user selection
                while True:
                    try:
                        user_input = input(f"\nSelect option 1-{len(candidates)} (or press Enter for best): ").strip()
                        if user_input == "":
                            selection = 0
                            print(f"Using best option (seed={candidates[0]['seed']})")
                            break
                        selection = int(user_input) - 1
                        if 0 <= selection < len(candidates):
                            print(f"Using option {selection + 1} (seed={candidates[selection]['seed']})")
                            break
                        else:
                            print(f"Please enter a number between 1 and {len(candidates)}")
                    except ValueError:
                        print("Invalid input. Please enter a number or press Enter.")

                # Use selected candidate
                selected = candidates[selection]
                train_tracks = selected['train_tracks']
                test_tracks = selected['test_tracks']

                # Clean up P-V map directory
                print(f"\nCleaning up temporary P-V maps...")
                for pv_path in pv_map_paths:
                    if pv_path.exists():
                        pv_path.unlink()
                pv_map_dir.rmdir()
                print("Temporary files removed.")

            elif args.test_holdout_seed is not None:
                # Use specific seed for deterministic behavior
                train_tracks, test_tracks = select_test_tracks_random_search(
                    df_filtered, logbook, test_fraction=args.test_holdout, seed=args.test_holdout_seed
                )
                print(f"Using seed: {args.test_holdout_seed}")
            else:
                # Default: use seed 42
                train_tracks, test_tracks = select_test_tracks_random_search(
                    df_filtered, logbook, test_fraction=args.test_holdout, seed=42
                )

        else:  # random
            all_tracks = df_filtered['trackid'].unique()
            np.random.seed(42)
            np.random.shuffle(all_tracks)
            n_test = int(len(all_tracks) * args.test_holdout)
            test_tracks = all_tracks[:n_test].tolist()
            train_tracks = all_tracks[n_test:].tolist()

        # Split dataframe by tracks
        df_train_tracks = df_filtered[df_filtered['trackid'].isin(train_tracks)]
        df_test = df_filtered[df_filtered['trackid'].isin(test_tracks)]

        print(f"\nTrack split:")
        print(f"  Train tracks: {len(train_tracks)}")
        print(f"  Test tracks: {len(test_tracks)}")
        print(f"  Train windows: {len(df_train_tracks)}")
        print(f"  Test windows: {len(df_test)}")

        # Validate test set diversity
        col_dict = define_collumn_labels()
        regime_col = col_dict['regime'][0]  # 'Melting regime'
        logbook_regime = logbook[['trackid', regime_col]].copy()
        logbook_regime.rename(columns={regime_col: 'regime'}, inplace=True)
        df_with_params = df_filtered.merge(logbook_regime, on='trackid', how='left')
        is_valid, warnings = validate_test_set_diversity(df_test, df_with_params)

        if warnings:
            print(f"\nTest set diversity validation:")
            for warning in warnings:
                print(f"  {warning}")

        if not is_valid:
            print("\n❌ Test set validation failed - cannot proceed")
            return 1

        # Check if temporal or stratified splitting for training data
        if args.temporal_split:
            # TEMPORAL: Apply temporal split to training tracks only
            print(f"\nApplying temporal split to training tracks...")
            best_seed, df_train, df_val, metrics_df = find_best_temporal_split(
                df_train_tracks,
                target_val_ratio=0.2,
                target_class_ratio=None,
                n_seeds=10,
                label_column='has_porosity',
                gap_size=5
            )

            print(f"  Best seed: {best_seed}")
            print(f"  Train: {len(df_train)} windows ({len(df_train[df_train['has_porosity']==1])} porosity)")
            print(f"  Val: {len(df_val)} windows ({len(df_val[df_val['has_porosity']==1])} porosity)")

            # Create fold assignments (single fold: T0/V0)
            df_train['temp_fold'] = 'T0'
            df_val['temp_fold'] = 'V0'

            # Combine into trainval.csv with fold column
            df_trainval = pd.concat([df_train, df_val], ignore_index=True)
            fold_column_name = f"{args.dataset_name}_folds"
            df_trainval[fold_column_name] = df_trainval['temp_fold']
            df_trainval.drop('temp_fold', axis=1, inplace=True)

            # Save trainval.csv with fold column
            trainval_csv = dataset_dir / "trainval.csv"
            output_columns = ['filename', 'has_porosity', fold_column_name]
            df_trainval[output_columns].to_csv(trainval_csv, index=False, encoding='utf-8')

            print(f"\n✓ Saved: {trainval_csv.name} ({len(df_trainval)} samples)")
            print(f"  Fold column: {fold_column_name} (T0={len(df_train)}, V0={len(df_val)})")

            # Update config
            config['statistics']['train_samples'] = len(df_train)
            config['statistics']['val_samples'] = len(df_val)
            config['fold_column'] = fold_column_name

        else:
            # STRATIFIED: No temporal split, save all training data for runtime k-fold
            print(f"\nNo temporal split - saving combined training data for stratified k-fold CV")

            # Save trainval.csv without fold column
            trainval_csv = dataset_dir / "trainval.csv"
            output_columns = ['filename', 'has_porosity']
            df_train_tracks[output_columns].to_csv(trainval_csv, index=False, encoding='utf-8')

            print(f"\n✓ Saved: {trainval_csv.name} ({len(df_train_tracks)} samples)")
            print(f"  (No fold assignments - stratified k-fold will be applied during training)")

            # Update config
            config['statistics']['trainval_samples'] = len(df_train_tracks)

        # Save test.csv (common to both modes)
        test_csv = dataset_dir / "test.csv"
        df_test[['filename', 'has_porosity']].to_csv(test_csv, index=False, encoding='utf-8')
        print(f"✓ Saved: {test_csv.name} ({len(df_test)} samples)")

        # Create test exclusion list for use with classifier
        test_exclusion_file = dataset_dir / "test_exclusion_list.txt"
        with open(test_exclusion_file, 'w', encoding='utf-8') as f:
            for filename in df_test['filename']:
                f.write(f"{filename}\n")
        print(f"✓ Saved: {test_exclusion_file.name} ({len(df_test)} files to exclude)")

        # Calculate statistics (already updated in temporal/stratified branches above)
        config['statistics']['mode'] = 'train_test_split'
        config['statistics']['test_samples'] = len(df_test)

        # Calculate retention
        if args.temporal_split:
            total_samples = len(df_train) + len(df_val) + len(df_test)
        else:
            total_samples = len(df_train_tracks) + len(df_test)
        config['statistics']['retention_pct'] = (total_samples / len(df_filtered)) * 100

    # Save dataset configuration
    config_file = dataset_dir / 'dataset_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Saved dataset config: {config_file}")

    # Generate P-V map if test holdout was created
    if args.test_holdout:
        print(f"\n🗺️  Generating P-V map for dataset...")
        try:
            from tools import generate_pv_map, get_logbook

            # Extract unique track IDs from test set
            test_trackids = df_test['track_id'].unique().tolist()

            # Get all track IDs from trainval set
            if args.temporal_split:
                trainval_trackids = pd.concat([df_train['track_id'], df_val['track_id']]).unique().tolist()
            else:
                trainval_trackids = df_train_tracks['track_id'].unique().tolist()

            # All tracks in the dataset
            all_dataset_trackids = sorted(set(test_trackids + trainval_trackids))

            # Get all AlSi10Mg CW L1 powder tracks for background
            logbook = get_logbook()
            AlSi10Mg = logbook['Substrate material'] == 'AlSi10Mg'
            L1 = logbook['Layer'] == 1
            cw = logbook['Point jump delay [us]'] == 0
            powder = logbook['Powder material'] != 'None'
            background_trackids = logbook[AlSi10Mg & L1 & cw & powder]['trackid'].unique().tolist()

            # Generate P-V map
            pv_map_path = dataset_dir / 'pv_map_dataset_distribution.png'
            fig, ax = generate_pv_map(
                trackids=all_dataset_trackids,  # All tracks in dataset
                output_path=pv_map_path,
                highlight_trackids=test_trackids,  # Test set highlighted with red ring
                figsize=(4, 3.2),
                dpi=300,
                font_size=8,
                show_background_points=True,  # Show all AlSi10Mg tracks in grey
                show_led_contours=False
            )

            print(f"✓ Saved P-V map: {pv_map_path.name}")
            print(f"   Total tracks in dataset: {len(all_dataset_trackids)}")
            print(f"   Test set tracks (highlighted): {len(test_trackids)}")
            print(f"   Trainval tracks: {len(trainval_trackids)}")

        except Exception as e:
            print(f"Warning: Could not generate P-V map: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ Dataset variant '{args.dataset_name}' prepared successfully!")
    print(f"   Location: {dataset_dir}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Prepare training dataset: filter tracks, remove low-signal images, and balance classes'
    )

    # File paths
    parser.add_argument(
        '--label-file',
        type=Path,
        help='Path to label CSV file (if not provided, uses default CWT data directory)'
    )
    parser.add_argument(
        '--image-dir',
        type=Path,
        help='Path to image directory (if not provided, derives from label file location)'
    )

    # Logbook-based filtering
    parser.add_argument(
        '--material',
        type=str,
        help='Filter by substrate material (e.g., AlSi10Mg, Al7A77, Ti64)'
    )
    parser.add_argument(
        '--layer',
        type=int,
        help='Filter by layer number (e.g., 1, 2)'
    )
    parser.add_argument(
        '--laser-mode',
        type=str,
        choices=['cw', 'pwm'],
        help='Filter by laser mode: cw (continuous wave) or pwm (pulsed)'
    )
    parser.add_argument(
        '--base-type',
        type=str,
        choices=['powder', 'welding'],
        help='Filter by base type: powder or welding (substrate only)'
    )
    parser.add_argument(
        '--substrate-no',
        type=str,
        help='Filter by substrate number (e.g., 514, 515)'
    )
    parser.add_argument(
        '--regime',
        type=str,
        choices=['conduction', 'keyhole', 'not_cond'],
        help='Filter by melting regime'
    )

    # Signal-based filtering
    parser.add_argument(
        '--signal-threshold',
        type=float,
        default=10.0,
        help='Minimum mean intensity threshold for filtering (default: 10.0)'
    )

    # Class balancing
    parser.add_argument(
        '--target-ratio',
        type=float,
        default=1.5,
        help='Target class imbalance ratio (class_0:class_1) for stratified downsampling (default: 1.5). '
             'Only used when --no-balancing is NOT set.'
    )
    parser.add_argument(
        '--no-balancing',
        action='store_true',
        help='Skip stratified downsampling (prevents random data removal to balance classes). '
             'Note: Does not affect smart seed selection when using --temporal-split.'
    )

    # Temporal splitting (prevents train/val leakage from window overlap)
    parser.add_argument(
        '--temporal-split',
        action='store_true',
        help='Use adaptive temporal block splitting to prevent train/val data leakage. '
             'Uses smart seed selection to optimize class balance without removing data. '
             'Incompatible with stratified downsampling (forces --no-balancing).'
    )
    parser.add_argument(
        '--n-seeds',
        type=int,
        default=100,
        help='Number of random seeds to try for best class balance (default: 100)'
    )
    parser.add_argument(
        '--stratified',
        action='store_true',
        help='Use stratified splitting to preserve original class distribution. '
             'Default is to target 50:50 class balance in validation set.'
    )
    parser.add_argument(
        '--gap-size',
        type=int,
        default=5,
        help='Gap size between blocks in windows (default: 5 for zero temporal overlap)'
    )

    # Output options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without creating output files'
    )
    parser.add_argument(
        '--output-type',
        choices=['filtered_csv', 'exclusion_list', 'both'],
        default='both',
        help='Type of output to generate (default: both)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for results (default: same as label file)'
    )

    # Dataset variant options (for separating dataset preparation from training)
    parser.add_argument(
        '--dataset-name',
        type=str,
        help='Name for this dataset variant (required when using --k-folds or --test-holdout)'
    )
    parser.add_argument(
        '--k-folds',
        type=int,
        default=1,
        help='Number of folds for k-fold CV dataset preparation (default: 1). '
             'Each fold gets independent temporal split with different seed.'
    )
    parser.add_argument(
        '--test-holdout',
        type=float,
        help='Fraction of tracks to hold out for test set (e.g., 0.2 for 20%%). '
             'Mutually exclusive with --k-folds.'
    )
    parser.add_argument(
        '--test-selection-strategy',
        type=str,
        choices=['random', 'preserve_diversity', 'random_search'],
        default='preserve_diversity',
        help='Strategy for selecting test tracks (default: preserve_diversity). '
             'preserve_diversity ensures training set retains all parameter combinations. '
             'random_search uses multi-objective optimization (class balance + regime diversity).'
    )
    parser.add_argument(
        '--test-holdout-seed',
        type=int,
        help='Random seed for test holdout selection. Allows deterministic/reproducible splits.'
    )
    parser.add_argument(
        '--test-holdout-candidates',
        type=int,
        help='Number of ranked test holdout options to show for selection (e.g., 5). '
             'If not specified, automatically selects the best option.'
    )
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List all prepared datasets and exit'
    )

    args = parser.parse_args()

    # Handle --list-datasets command (early exit)
    if args.list_datasets:
        list_available_datasets()
        return 0

    # Validate dataset variant options
    if args.dataset_name and not (args.k_folds > 1 or args.test_holdout):
        print("ERROR: --dataset-name requires either --k-folds or --test-holdout")
        return 1

    if (args.k_folds > 1 or args.test_holdout) and not args.dataset_name:
        print("ERROR: Dataset variant mode requires --dataset-name")
        return 1

    if args.k_folds > 1 and args.test_holdout:
        print("ERROR: Cannot use both --k-folds and --test-holdout simultaneously")
        print("  Use --k-folds for hyperparameter tuning (no test holdout)")
        print("  Use --test-holdout for final evaluation (with test set)")
        return 1

    print("="*80)
    print("PREPARE TRAINING DATASET")
    print("="*80)

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be created\n")

    # Determine label file path
    if args.label_file:
        label_path = args.label_file
    else:
        # get_default_cwt_data_dir() returns the full path to per_window directory
        # Label file is in the grandparent directory (CWT_labelled_windows)
        cwt_data_dir = Path(get_default_cwt_data_dir())
        label_path = cwt_data_dir.parent.parent.parent.parent.parent.parent / '1.0ms-window_0.2ms_offset_data_labels.csv'

    if not label_path.exists():
        print(f"❌ Label file not found: {label_path}")
        return 1

    print(f"Label file: {label_path}")

    # Determine image directory
    if args.image_dir:
        image_dir = args.image_dir
    else:
        # Derive from label file location
        image_dir = label_path.parent / 'PD1' / 'cmor1_5-1_0' / '1.0_ms' / '1000-50000_Hz_256_steps' / 'grey' / 'per_window'

    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return 1

    print(f"Image directory: {image_dir}")
    print(f"Signal threshold: {args.signal_threshold}")
    print(f"Target ratio: {args.target_ratio}:1")

    # Load labels
    print("\nLoading labels...")
    df = pd.read_csv(label_path, encoding='utf-8')
    print(f"Total samples: {len(df)}")

    # Parse filenames and use existing columns if available
    print("\nParsing filenames...")

    # Check if we have trackid and window_n columns already
    if 'trackid' in df.columns and 'window_n' in df.columns:
        df['track_id'] = df['trackid']
        df['window_index'] = df['window_n']
        print("  Using existing trackid and window_n columns")
    else:
        # Parse from filename
        filename_col = 'image_filename' if 'image_filename' in df.columns else 'filename'
        filename_info = df[filename_col].apply(parse_filename)
        df['track_id'] = filename_info.apply(lambda x: x['track_id'])
        df['window_index'] = filename_info.apply(lambda x: x['window_index'])
        print("  Parsed track_id and window_index from filenames")

    # Standardize filename column name
    if 'image_filename' in df.columns:
        df['filename'] = df['image_filename']

    # Join with logbook and apply logbook-based filters
    print("\nJoining with logbook...")
    try:
        logbook = get_logbook()
        df = join_with_logbook(df, logbook)
        print(f"✓ Joined with logbook data")

        # Build filters dict from command-line arguments
        filters_dict = {}
        if args.material:
            filters_dict['material'] = args.material
        if args.layer:
            filters_dict['layer'] = args.layer
        if args.laser_mode:
            filters_dict['laser_mode'] = args.laser_mode
        if args.base_type:
            filters_dict['base_type'] = args.base_type
        if args.substrate_no:
            filters_dict['substrate_no'] = args.substrate_no
        if args.regime:
            filters_dict['regime'] = args.regime

        # Apply logbook filters if any specified
        if filters_dict:
            print(f"\n{'='*80}")
            print("STEP 0: LOGBOOK-BASED FILTERING")
            print(f"{'='*80}")
            df, active_filters = apply_logbook_filters(df, logbook, filters_dict)
        else:
            active_filters = []
            print("\nNo logbook filters specified, using all tracks")

    except Exception as e:
        print(f"⚠️  Could not load logbook: {e}")
        print("  Using default values for laser_power and scan_speed")
        df['laser_power'] = 0.0
        df['scan_speed'] = 0.0
        active_filters = []

    # Compute signal metrics
    print("\nComputing signal metrics...")
    signal_metrics = []

    for i, row in df.iterrows():
        if i % 500 == 0:
            print(f"  Processing image {i+1}/{len(df)}...", end='\r')

        img_path = image_dir / row['filename']
        metrics = compute_signal_metrics(img_path)
        signal_metrics.append(metrics)

    print(f"  Processing image {len(df)}/{len(df)}... Done!")

    df['mean'] = [m['mean'] for m in signal_metrics]
    df['std'] = [m['std'] for m in signal_metrics]
    df['variance'] = [m['variance'] for m in signal_metrics]

    # Store original for reporting
    df_original = df.copy()

    # Step 1: Signal-based filtering
    print(f"\n{'='*80}")
    print("STEP 1: SIGNAL-BASED FILTERING")
    print(f"{'='*80}")

    print(f"\nFiltering images with mean intensity < {args.signal_threshold}...")

    class_0_before = len(df[df['has_porosity'] == 0])
    class_1_before = len(df[df['has_porosity'] == 1])

    df_filtered = df[df['mean'] >= args.signal_threshold].copy()

    class_0_after = len(df_filtered[df_filtered['has_porosity'] == 0])
    class_1_after = len(df_filtered[df_filtered['has_porosity'] == 1])

    class_0_removed = class_0_before - class_0_after
    class_1_removed = class_1_before - class_1_after

    print(f"  Class 0: {class_0_before} → {class_0_after} (removed {class_0_removed}, {class_0_removed/class_0_before*100:.1f}%)")
    print(f"  Class 1: {class_1_before} → {class_1_after} (removed {class_1_removed}, {class_1_removed/class_1_before*100:.1f}%)")

    ratio_after_filtering = class_0_after / class_1_after if class_1_after > 0 else 0
    print(f"  New ratio: {ratio_after_filtering:.2f}:1")

    # Class Balancing: Two Different Mechanisms
    # 1. Smart Seed Selection (non-destructive):
    #    - Tries N random seeds for temporal block assignment
    #    - Selects seed with best train/val class balance
    #    - Compatible with temporal splits (no data removal)
    #    - Always active when --temporal-split is used
    #
    # 2. Stratified Downsampling (destructive):
    #    - Randomly removes majority class samples to target ratio
    #    - Breaks temporal continuity (creates unpredictable gaps)
    #    - Incompatible with temporal splits
    #    - Controlled by --no-balancing flag (disabled when True)

    # Check for incompatible options
    if args.temporal_split and not args.no_balancing:
        print(f"\n⚠️  Warning: --temporal-split is incompatible with stratified downsampling")
        print(f"  Stratified downsampling randomly removes majority class samples")
        print(f"  This breaks temporal continuity required for zero-leakage splits")
        print(f"  Temporal split will use smart seed selection instead (trying {args.n_seeds} seeds)")
        print(f"  Use --augment_to_balance during training for additional balancing")
        print(f"  Forcing --no-balancing for temporal split")
        args.no_balancing = True

    # Check for dataset variant mode (k-fold CV or train/test split)
    if args.dataset_name:
        # Load logbook for diversity-preserving test selection
        logbook = get_logbook()

        # Call dataset variant preparation function
        result = prepare_dataset_variant(df_filtered, args, logbook, active_filters, label_path, image_dir)
        return result

    # Step 2: Temporal splitting OR stratified downsampling
    if args.temporal_split:
        print(f"\n{'='*80}")
        print("STEP 2: ADAPTIVE TEMPORAL SPLITTING")
        print(f"{'='*80}")

        print(f"\nThis will split data into train/val sets with zero temporal leakage.")
        print(f"Trying {args.n_seeds} random seeds to optimize class balance...")

        # Determine target class ratio
        if args.stratified:
            # Stratified: preserve original class distribution
            target_class_ratio = class_1_after / (class_0_after + class_1_after)
            print(f"  Using stratified splitting: target {target_class_ratio*100:.1f}% class 1 (matches dataset)")
        else:
            # Default: aim for 50% class balance in validation set (better for evaluation)
            # Training set can be rebalanced with augmentation during training
            target_class_ratio = 0.5  # 50% class 1 in validation
            print(f"  Using balanced splitting: target 50% class balance in validation set")
            print(f"  (Training set will be balanced with --augment_to_balance during training)")

        # Run multi-seed temporal splitting
        best_seed, train_df, val_df, metrics_df = find_best_temporal_split(
            df_filtered,
            target_val_ratio=0.2,
            target_class_ratio=target_class_ratio,
            n_seeds=args.n_seeds,
            label_column='has_porosity',
            gap_size=args.gap_size
        )

        # Get final split stats using the best seed
        # Note: Using default thresholds (20/25/30) - modify here if needed
        _, _, split_stats = adaptive_block_temporal_split(
            df_filtered, gap_size=args.gap_size, random_seed=best_seed
        )

        # For reporting, use the split datasets
        df_balanced = df_filtered  # Not used in temporal split mode
        dimensions = []  # Not used in temporal split mode

    # Step 2 (alternative): Stratified downsampling (skip if --no-balancing specified)
    elif args.no_balancing:
        print(f"\n⊘ Skipping class balancing (--no-balancing enabled)")
        df_balanced = df_filtered.copy()
        final_class_0 = class_0_after
        final_class_1 = class_1_after
        final_ratio = ratio_after_filtering
        train_df = None
        val_df = None
    else:
        print(f"\n{'='*80}")
        print("STEP 2: STRATIFIED DOWNSAMPLING")
        print(f"{'='*80}")

        # Calculate target count for class 0
        target_class_0_count = int(class_1_after * args.target_ratio)

        print(f"\nTarget class 0 count: {target_class_0_count} (ratio {args.target_ratio}:1)")

        # Separate classes
        df_class_0 = df_filtered[df_filtered['has_porosity'] == 0].copy()
        df_class_1 = df_filtered[df_filtered['has_porosity'] == 1].copy()

        # Downsample class 0
        print(f"Downsampling class 0 from {len(df_class_0)} to {target_class_0_count}...")

        dimensions = ['track_id', 'laser_power', 'scan_speed', 'mean', 'window_index']

        df_class_0_downsampled = stratified_downsample(df_class_0, target_class_0_count, dimensions)

        print(f"  ✓ Downsampled to {len(df_class_0_downsampled)} samples")

        # Combine classes
        df_balanced = pd.concat([df_class_0_downsampled, df_class_1], ignore_index=True)

        final_class_0 = len(df_balanced[df_balanced['has_porosity'] == 0])
        final_class_1 = len(df_balanced[df_balanced['has_porosity'] == 1])
        final_ratio = final_class_0 / final_class_1 if final_class_1 > 0 else 0

        print(f"\nFinal dataset:")
        print(f"  Total samples: {len(df_balanced)}")
        print(f"  Class 0: {final_class_0}")
        print(f"  Class 1: {final_class_1}")
        print(f"  Ratio: {final_ratio:.2f}:1")

        train_df = None
        val_df = None

    # Generate statistics report (skip for temporal split mode)
    if not args.temporal_split:
        print(f"\n{'='*80}")
        print("GENERATING STATISTICS REPORT")
        print(f"{'='*80}")

        report = generate_statistics_report(df_original, df_filtered, df_balanced, dimensions)
        print("\n" + report)

    # Save outputs
    if not args.dry_run:
        print(f"\n{'='*80}")
        print("SAVING OUTPUTS")
        print(f"{'='*80}")

        output_dir = args.output_dir if args.output_dir else label_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build filename suffix from active parameters
        suffix_parts = []
        if active_filters:
            suffix_parts.append('_'.join(active_filters))
        suffix_parts.append(f"threshold{args.signal_threshold}")

        if args.temporal_split:
            suffix_parts.append(f"temporal_split")
            suffix = '_'.join(suffix_parts)

            # Save temporal split outputs
            # Only save essential columns
            output_columns = ['filename', 'has_porosity']

            train_csv = output_dir / f"{label_path.stem}_prepared_{suffix}_train.csv"
            val_csv = output_dir / f"{label_path.stem}_prepared_{suffix}_val.csv"

            train_df[output_columns].to_csv(train_csv, index=False, encoding='utf-8')
            val_df[output_columns].to_csv(val_csv, index=False, encoding='utf-8')

            print(f"✓ Saved train CSV: {train_csv} ({len(train_df)} samples)")
            print(f"✓ Saved val CSV: {val_csv} ({len(val_df)} samples)")

            # Save seed analysis
            seed_analysis_csv = output_dir / f"{label_path.stem}_prepared_{suffix}_seed_analysis.csv"
            save_seed_analysis(metrics_df, seed_analysis_csv)
            print(f"✓ Saved seed analysis: {seed_analysis_csv}")

            # Save best seed
            best_seed_file = output_dir / f"{label_path.stem}_prepared_{suffix}_best_seed.txt"
            with open(best_seed_file, 'w', encoding='utf-8') as f:
                f.write(f"{best_seed}\n")
            print(f"✓ Saved best seed: {best_seed_file} (seed={best_seed})")

            # Save temporal split report
            split_report_file = output_dir / f"{label_path.stem}_prepared_{suffix}_report.txt"
            save_temporal_split_report(split_stats, best_seed, train_df, val_df,
                                      'has_porosity', split_report_file)
            print(f"✓ Saved temporal split report: {split_report_file}")

        else:
            # Standard balanced or filtered output
            if not args.no_balancing:
                suffix_parts.append(f"ratio{args.target_ratio}")
            suffix = '_'.join(suffix_parts)

            # Save filtered CSV
            if args.output_type in ['filtered_csv', 'both']:
                output_csv = output_dir / f"{label_path.stem}_prepared_{suffix}.csv"

                # Only save essential columns
                output_columns = ['filename', 'has_porosity']
                df_balanced[output_columns].to_csv(output_csv, index=False, encoding='utf-8')

                print(f"✓ Saved filtered CSV: {output_csv}")

            # Save exclusion list
            if args.output_type in ['exclusion_list', 'both']:
                excluded_filenames = set(df_original['filename']) - set(df_balanced['filename'])
                exclusion_file = output_dir / f"exclusion_list_{suffix}.txt"

                with open(exclusion_file, 'w', encoding='utf-8') as f:
                    for filename in sorted(excluded_filenames):
                        f.write(f"{filename}\n")

                print(f"✓ Saved exclusion list: {exclusion_file}")
                print(f"  Excluded {len(excluded_filenames)} files")

            # Save statistics report
            report_file = output_dir / f"preparation_report_{suffix}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)

            print(f"✓ Saved statistics report: {report_file}")

        print(f"\n✅ Dataset preparation completed successfully!")
    else:
        print(f"\n💡 Run without --dry-run to save output files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
