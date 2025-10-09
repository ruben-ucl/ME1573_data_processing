#!/usr/bin/env python3
"""
Generate Test Holdout Script

Creates a test holdout file from the experiment logbook with one trackid 
from each melting regime, restricted to AlSi10Mg material/powder and duty cycle 1.0.

The holdout approach ensures entire image sequences from selected trackids are held out,
providing a more realistic test set that better reflects model performance on new tracks.

Usage:
    # Generate test holdout file with automatic directory detection from config.py
    python ml/generate_test_holdout.py --verbose
    
    # Generate with custom data directories for accurate image counts
    python ml/generate_test_holdout.py --verbose --data_dirs "/path/to/cwt_data" "/path/to/more/data"
    
    # Use with final model trainer
    python ml/final_model_trainer.py --classifier_type cwt_image --test_holdout_file ml/test_holdout_trackids.txt
    
    # Custom output location and target count
    python ml/generate_test_holdout.py --output custom_holdout.txt --target_count 150 --verbose

Output files:
    - test_holdout_trackids.txt: Main holdout file with trackids and detailed statistics
    - test_holdout_trackids.json: Machine-readable statistics and metadata

Example Terminal Output:
    ======================================================================
    TEST HOLDOUT IMAGE COUNT ANALYSIS
    ======================================================================

    Total images in test holdout: 1,234

    Per-trackid breakdown:

      Conduction: 456 images (37.0% of holdout)
        • track_001: 123 images (10.0%)
        • track_005: 333 images (27.0%)

      Keyhole: 544 images (44.1% of holdout)
        • track_023: 544 images (44.1%)

      Transition: 234 images (19.0% of holdout)
        • track_012: 234 images (19.0%)
    ======================================================================

    Scanning entire dataset for regime distribution...
      Total images in dataset: 10,456
      Total trackids found: 50
      Trackids matched to regimes: 48

      Dataset regime distribution:
        Conduction: 22 trackids, 4,567 images (43.7%)
        Keyhole: 15 trackids, 3,234 images (30.9%)
        Transition: 11 trackids, 2,655 images (25.4%)

    ======================================================================
    📊 HOLDOUT SUMMARY
    ======================================================================
      Selected trackids: 3/50 (6.0%)

      Total dataset:      10,456 images (100.0%)
      Training set:       9,222 images (88.2%)
      Test holdout set:   1,234 images (11.8%)

    ======================================================================
    🎯 REGIME DISTRIBUTION COMPARISON
    ======================================================================

    Regime               Dataset                   Holdout                   Representation
    -------------------- ------------------------- ------------------------- ---------------
    Conduction           4,567 (43.7%)             456 (37.0%)               ⚠ 0.85x
    Keyhole              3,234 (30.9%)             544 (44.1%)               ⚡ 1.43x
    Transition           2,655 (25.4%)             234 (19.0%)               ⚠ 0.75x

      Legend:
        ✓ Well represented (0.8x - 1.2x of dataset proportion)
        ⚠ Under-represented (<0.8x)
        ⚡ Over-represented (>1.2x)
    ======================================================================

Author: Claude Code Assistant
"""

# Ensure UTF-8 encoding for all I/O operations
import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path to import tools
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_logbook

# Import config for data paths
from config import get_default_cwt_data_dir, CWT_DATA_DIR_DICT, PD_DATA_DIR

def filter_logbook_for_test_holdout():
    """
    Filter logbook for test holdout candidates.
    Based on filter_logbook() from dataset_labeller.py but with modifications.
    
    Returns:
        pd.DataFrame: Filtered logbook with AlSi10Mg, CW, L1, powder, duty cycle 1.0
    """
    log = get_logbook()
    
    # Filters for welding or powder melting
    welding = log['Powder material'] == 'None'
    powder = np.invert(welding)
    
    # Filters for CW laser mode (continuous wave, not pulsed)
    cw = log['Point jump delay [us]'] == 0
    
    # Filter for Layer 1 tracks only
    L1 = log['Layer'] == 1
    
    # Filter by material - both substrate and powder should be AlSi10Mg
    AlSi10Mg_substrate = log['Substrate material'] == 'AlSi10Mg'
    AlSi10Mg_powder = log['Powder material'] == 'AlSi10Mg'
    
    # Filter by duty cycle = 1.0 (if column exists)
    if 'Duty cycle' in log.columns:
        duty_cycle_1 = log['Duty cycle'] == 1.0
        print(f"Found Duty cycle column. Values: {sorted(log['Duty cycle'].dropna().unique())}")
    elif 'duty_cycle' in log.columns:
        duty_cycle_1 = log['duty_cycle'] == 1.0
        print(f"Found duty_cycle column. Values: {sorted(log['duty_cycle'].dropna().unique())}")
    else:
        # If no duty cycle column, assume all are 1.0 (CW mode implies duty cycle 1.0)
        print("No duty cycle column found, assuming all CW tracks have duty cycle 1.0")
        duty_cycle_1 = pd.Series([True] * len(log), index=log.index)
    
    # Apply filters
    log_filtered = log[AlSi10Mg_substrate & AlSi10Mg_powder & cw & L1 & powder & duty_cycle_1]
    
    return log_filtered

def select_test_holdout_trackids(log_filtered, target_percentage=15.0, tolerance=5.0, random_seed=None, data_directories=None, verbose=True):
    """
    Select trackids for test holdout targeting 15% of total images (±5%).

    Strategy:
    1. One random trackid from each melting regime (ensures stratification)
    2. Iteratively add random trackids to approach 15% target
    3. Stop when within 10-20% range

    Args:
        log_filtered: Filtered logbook DataFrame
        target_percentage: Target percentage of images for test set (default: 15%)
        tolerance: Acceptable deviation from target (default: ±5%)
        random_seed: Random seed for reproducibility (None = truly random)
        data_directories: Directories to scan for actual image counts
        verbose: Print detailed output

    Returns:
        tuple: (selected_trackids, regime_trackids dict)
    """
    if 'Melting regime' not in log_filtered.columns:
        raise ValueError("Melting regime column not found in logbook")

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
        if verbose:
            print(f"\nUsing random seed: {random_seed}")
    else:
        if verbose:
            print(f"\nUsing random selection (no seed)")

    # Get actual image counts per trackid by scanning directories
    trackid_image_counts = {}
    if data_directories:
        for data_dir in data_directories:
            data_path = Path(data_dir)
            if data_path.exists():
                for class_dir in ['0', '1']:
                    class_path = data_path / class_dir
                    if class_path.exists():
                        for img_file in class_path.glob('*.png'):
                            parts = img_file.stem.split('_')
                            if len(parts) >= 2:
                                trackid = f"{parts[0]}_{parts[1]}"
                                trackid_image_counts[trackid] = trackid_image_counts.get(trackid, 0) + 1

    total_images = sum(trackid_image_counts.values()) if trackid_image_counts else 0
    target_count = int(total_images * target_percentage / 100) if total_images > 0 else 0
    min_acceptable = int(total_images * (target_percentage - tolerance) / 100)
    max_acceptable = int(total_images * (target_percentage + tolerance) / 100)

    if verbose and total_images > 0:
        print(f"\nTotal images in dataset: {total_images}")
        print(f"Target test set size: {target_percentage}% = {target_count} images")
        print(f"Acceptable range: {target_percentage-tolerance}%-{target_percentage+tolerance}% = {min_acceptable}-{max_acceptable} images")

    # Group by melting regime
    regime_counts = log_filtered['Melting regime'].value_counts()
    regime_trackids = {}
    selected_trackids = []
    current_count = 0

    if verbose:
        print(f"\nMelting regime distribution in filtered data:")
        for regime, count in regime_counts.items():
            print(f"  {regime}: {count} trackids")

    if verbose:
        print(f"\nPhase 1: Selecting one random trackid from each regime...")

    # Phase 1: Select ONE RANDOM trackid from each regime (stratification)
    for regime in regime_counts.index:
        regime_data = log_filtered[log_filtered['Melting regime'] == regime]
        available_trackids = regime_data['trackid'].tolist()

        # Fully random selection
        selected_trackid = np.random.choice(available_trackids)

        selected_trackids.append(selected_trackid)
        regime_trackids[regime] = [selected_trackid]

        # Add to count if we have image data
        if selected_trackid in trackid_image_counts:
            current_count += trackid_image_counts[selected_trackid]

        if verbose:
            img_count = trackid_image_counts.get(selected_trackid, '?')
            print(f"  {regime}: {selected_trackid} ({img_count} images)")

    # Phase 2: Iteratively add random trackids to approach target
    if total_images > 0 and target_count > 0:
        if verbose:
            print(f"\nPhase 2: Adding trackids to approach {target_percentage}% target...")
            print(f"Current: {current_count} images ({current_count/total_images*100:.1f}%)")

        iteration = 0
        max_iterations = 100  # Safety limit

        # Continue while we're below minimum acceptable
        while current_count < min_acceptable and iteration < max_iterations:
            iteration += 1

            # Get available trackids (not yet selected)
            available = [
                tid for tid in log_filtered['trackid']
                if tid not in selected_trackids and tid in trackid_image_counts
            ]

            if not available:
                if verbose:
                    print(f"  No more trackids available to select")
                break

            # Randomly select next trackid
            next_trackid = np.random.choice(available)
            next_count = trackid_image_counts[next_trackid]

            # Check if adding this would exceed upper bound
            if current_count + next_count > max_acceptable:
                if verbose:
                    print(f"  Skipping {next_trackid} ({next_count} images) - would exceed upper bound")
                # Try to find a smaller trackid
                smaller_candidates = [t for t in available if trackid_image_counts[t] <= (max_acceptable - current_count)]
                if smaller_candidates:
                    next_trackid = np.random.choice(smaller_candidates)
                    next_count = trackid_image_counts[next_trackid]
                else:
                    if verbose:
                        print(f"  No trackids small enough to stay within bounds")
                    break

            # Add this trackid
            selected_trackids.append(next_trackid)
            current_count += next_count

            # Update regime mapping
            regime = log_filtered[log_filtered['trackid'] == next_trackid]['Melting regime'].iloc[0]
            if regime not in regime_trackids:
                regime_trackids[regime] = []
            regime_trackids[regime].append(next_trackid)

            if verbose:
                pct = current_count / total_images * 100
                print(f"  Added {next_trackid}: {next_count} images → Total: {current_count} ({pct:.1f}%)")

            # Stop if we're within acceptable range
            if min_acceptable <= current_count <= max_acceptable:
                if verbose:
                    print(f"  ✓ Reached acceptable range!")
                break

    final_percentage = (current_count / total_images * 100) if total_images > 0 else 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"SELECTION SUMMARY:")
        print(f"  Selected: {len(selected_trackids)} trackids")
        print(f"  Images: {current_count} / {total_images} ({final_percentage:.1f}%)")
        print(f"  Target: {target_percentage}% ± {tolerance}% (range: {target_percentage-tolerance}%-{target_percentage+tolerance}%)")

        if total_images > 0:
            if min_acceptable <= current_count <= max_acceptable:
                print(f"  Status: ✓ Within acceptable range")
            elif current_count < min_acceptable:
                print(f"  Status: ⚠ Below minimum (limited by trackid granularity)")
            else:
                print(f"  Status: ⚠ Above maximum")
        print(f"{'='*60}")

    return selected_trackids, regime_trackids

def count_actual_images(selected_trackids, regime_trackids, data_directories=None, verbose=True):
    """
    Count actual images for each trackid in the data directories.
    
    Args:
        selected_trackids: List of selected trackids
        regime_trackids: Dict mapping regimes to trackids
        data_directories: List of directories to search for images
        verbose: Print detailed information
        
    Returns:
        dict: Statistics about image counts per regime and total
    """
    if data_directories is None:
        # Use data directories from config.py for automatic detection
        data_directories = []
        
        # Add all CWT data directories (including default channel)
        for wavelet, directory in CWT_DATA_DIR_DICT.items():
            if directory and Path(directory).exists():
                data_directories.append(directory)
        
        # Also check PD data directory (for pd_signal classifier)
        if PD_DATA_DIR and Path(PD_DATA_DIR).exists():
            data_directories.append(PD_DATA_DIR)
            
        # If no config directories exist, fall back to common patterns
        if not data_directories:
            common_patterns = [
                "E:/AlSi10Mg single layer ffc/CWT_labelled_windows",
                "./data/cwt",
                "./cwt_data", 
                "../data/cwt"
            ]
            for pattern in common_patterns:
                if Path(pattern).exists():
                    data_directories.append(pattern)
        
        if verbose:
            print(f"Searching for images in {len(data_directories)} directories:")
            for i, dir_path in enumerate(data_directories, 1):
                exists = "✓" if Path(dir_path).exists() else "✗"
                print(f"  {i}. {exists} {dir_path}")
    
    total_images = 0
    regime_image_counts = {}
    trackid_image_counts = {}
    
    for trackid in selected_trackids:
        image_count = 0
        
        # Search in common subdirectory structures (0/ and 1/ for binary classification)
        for data_dir in data_directories:
            data_path = Path(data_dir)
            if data_path.exists():
                for class_dir in ['0', '1']:
                    class_path = data_path / class_dir
                    if class_path.exists():
                        # Count both PNG (CWT) and TIFF (PD) files matching trackid pattern
                        for file_extension in ['*.png', '*.tiff']:
                            pattern = f"{trackid}{file_extension}"
                            matching_files = list(class_path.glob(pattern))
                            image_count += len(matching_files)
        
        trackid_image_counts[trackid] = image_count
        total_images += image_count
    
    # Group by regime
    for regime, trackids in regime_trackids.items():
        regime_total = sum(trackid_image_counts.get(tid, 0) for tid in trackids)
        regime_image_counts[regime] = {
            'trackids': trackids,
            'image_count': regime_total,
            'trackid_details': {tid: trackid_image_counts.get(tid, 0) for tid in trackids}
        }
    
    if total_images > 0:
        print(f"\n{'='*70}")
        print("TEST HOLDOUT IMAGE COUNT ANALYSIS")
        print(f"{'='*70}")
        print(f"\nTotal images in test holdout: {total_images:,}")
        print(f"\nPer-trackid breakdown:")
        for regime, info in sorted(regime_image_counts.items()):
            regime_pct = (info['image_count'] / total_images * 100) if total_images > 0 else 0
            print(f"\n  {regime}: {info['image_count']:,} images ({regime_pct:.1f}% of holdout)")
            for tid, count in sorted(info['trackid_details'].items()):
                tid_pct = (count / total_images * 100) if total_images > 0 else 0
                print(f"    • {tid}: {count:,} images ({tid_pct:.1f}%)")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("⚠️  WARNING: Image Count Analysis")
        print(f"{'='*70}")
        print(f"Could not find images in searched directories:")
        for dir_path in data_directories:
            exists_mark = "✓" if Path(dir_path).exists() else "✗"
            print(f"  {exists_mark} {dir_path}")
        print(f"\nImage counts will be estimated. Actual counts will be determined during training.")
        print(f"{'='*70}")
    
    return {
        'total_images': total_images,
        'regime_image_counts': regime_image_counts,
        'trackid_image_counts': trackid_image_counts,
        'data_directories_searched': data_directories
    }

def analyze_dataset_regime_distribution(data_directories, log_filtered, verbose=True):
    """
    Analyze regime distribution across the entire dataset by scanning files and matching to logbook.

    Args:
        data_directories: List of directories to search
        log_filtered: Filtered logbook DataFrame with trackid and regime info
        verbose: Print information

    Returns:
        dict: Statistics about regime distribution in full dataset
    """
    if verbose:
        print(f"\nScanning entire dataset for regime distribution...")

    # Extract trackids from filenames and count images per trackid
    trackid_image_counts = {}
    total_images = 0

    for data_dir in data_directories:
        data_path = Path(data_dir)
        if data_path.exists():
            for class_dir in ['0', '1']:
                class_path = data_path / class_dir
                if class_path.exists():
                    # Find all image files
                    for file_pattern in ['*.png', '*.tiff']:
                        for img_file in class_path.glob(file_pattern):
                            # Extract trackid using helper function
                            from data_utils import extract_trackid_from_filename
                            trackid = extract_trackid_from_filename(img_file.name)

                            if trackid:
                                trackid_image_counts[trackid] = trackid_image_counts.get(trackid, 0) + 1
                                total_images += 1

    # Match trackids to regimes from logbook
    regime_distribution = {}
    trackids_without_regime = []

    for trackid, img_count in trackid_image_counts.items():
        # Look up regime in filtered logbook
        matching_rows = log_filtered[log_filtered['trackid'] == trackid]

        if len(matching_rows) > 0:
            regime = matching_rows.iloc[0]['Melting regime']

            if regime not in regime_distribution:
                regime_distribution[regime] = {
                    'trackids': [],
                    'image_count': 0
                }

            regime_distribution[regime]['trackids'].append(trackid)
            regime_distribution[regime]['image_count'] += img_count
        else:
            # Trackid not in filtered logbook (might be from different material/conditions)
            trackids_without_regime.append(trackid)

    if verbose:
        print(f"  Total images in dataset: {total_images:,}")
        print(f"  Total trackids found: {len(trackid_image_counts)}")
        print(f"  Trackids matched to regimes: {len(trackid_image_counts) - len(trackids_without_regime)}")

        if trackids_without_regime:
            print(f"  Trackids without regime info: {len(trackids_without_regime)} (excluded from analysis)")

        print(f"\n  Dataset regime distribution:")
        for regime, info in sorted(regime_distribution.items()):
            regime_pct = (info['image_count'] / total_images * 100) if total_images > 0 else 0
            print(f"    {regime}: {len(info['trackids'])} trackids, {info['image_count']:,} images ({regime_pct:.1f}%)")

    return {
        'total_images': total_images,
        'total_trackids': len(trackid_image_counts),
        'regime_distribution': regime_distribution,
        'trackids_without_regime': trackids_without_regime
    }

def create_test_holdout_file(selected_trackids, regime_trackids, log_filtered, output_file, data_directories=None, random_seed=None, verbose=True):
    """
    Create test holdout exclusion file with detailed statistics.

    Args:
        selected_trackids: List of trackids to hold out
        regime_trackids: Dict mapping regimes to trackids
        log_filtered: Filtered logbook DataFrame
        output_file: Path to output file
        verbose: Print information
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count actual images if possible
    image_stats = count_actual_images(selected_trackids, regime_trackids, data_directories=data_directories, verbose=verbose)

    # Get the directories that were actually used (either provided or auto-detected)
    actual_data_dirs = image_stats['data_directories_searched']

    # Analyze full dataset regime distribution
    dataset_stats = None
    total_dataset_images = 0
    if actual_data_dirs and image_stats['total_images'] > 0:
        dataset_stats = analyze_dataset_regime_distribution(actual_data_dirs, log_filtered, verbose=verbose)
        total_dataset_images = dataset_stats['total_images']
        image_stats['total_dataset_images'] = total_dataset_images
        image_stats['dataset_regime_distribution'] = dataset_stats['regime_distribution']
    
    # Calculate total available trackids and estimated total images
    total_available_trackids = len(log_filtered)
    selected_count = len(selected_trackids)
    trackid_percentage = (selected_count / total_available_trackids) * 100
    
    # Estimate total dataset size (if actual counts not available)
    estimated_total_images = total_available_trackids * 30  # Rough estimate: 30 images per trackid
    estimated_holdout_images = selected_count * 30
    estimated_image_percentage = (estimated_holdout_images / estimated_total_images) * 100
    
    # Use actual counts if available
    if image_stats['total_images'] > 0:
        actual_holdout_images = image_stats['total_images']
        # For total dataset, we'd need to scan all trackids - use estimate for now
        actual_image_percentage = "TBD during training"  # Will be calculated when full dataset is scanned
    else:
        actual_holdout_images = estimated_holdout_images
        actual_image_percentage = f"~{estimated_image_percentage:.1f}%"
    
    # Create the holdout file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Test Holdout Trackids - Generated from Experiment Logbook\n")
        f.write("# This file contains trackids selected for test set holdout\n")
        f.write("# All images matching trackid*.png will be excluded from training\n")
        f.write("#\n")
        f.write("# HOLDOUT STATISTICS:\n")
        f.write(f"# Total available trackids (filtered): {total_available_trackids}\n")
        f.write(f"# Selected trackids for holdout: {selected_count} ({trackid_percentage:.1f}%)\n")
        f.write(f"# Holdout images: {actual_holdout_images}\n")
        f.write(f"# Estimated holdout percentage: {actual_image_percentage}\n")
        f.write("#\n")
        f.write("# SELECTION CRITERIA:\n")
        f.write("# - Material: AlSi10Mg (substrate and powder)\n") 
        f.write("# - Laser mode: Continuous wave (CW)\n")
        f.write("# - Layer: 1\n")
        f.write("# - Duty cycle: 1.0\n")
        f.write("#\n")
        f.write("# SELECTION METHOD:\n")
        f.write(f"# - Random seed: {random_seed if random_seed is not None else 'None (truly random)'}\n")
        f.write("# - One representative trackid from each melting regime\n")
        f.write("# - Additional trackids from largest regime if needed for target count\n")
        f.write("#\n")
        f.write("# DATA DIRECTORIES SEARCHED:\n")
        for i, dir_path in enumerate(image_stats['data_directories_searched'], 1):
            exists = "✓" if Path(dir_path).exists() else "✗"
            f.write(f"# {i}. {exists} {dir_path}\n")
        f.write("#\n")
        f.write("# REGIME BREAKDOWN:\n")
        
        for regime, info in image_stats['regime_image_counts'].items():
            f.write(f"# {regime}: {len(info['trackids'])} trackids, {info['image_count']} images\n")
            for tid in info['trackids']:
                img_count = info['trackid_details'][tid]
                f.write(f"#   {tid}: {img_count} images\n")
        
        f.write("#\n")
        f.write("# TRACKID LIST (one per line):\n")
        f.write("\n")
        
        for trackid in sorted(selected_trackids):
            f.write(f"{trackid}\n")
    
    # Also create a detailed statistics file
    stats_file = output_path.with_suffix('.json')

    # Calculate detailed image statistics
    image_statistics = {
        'holdout_images': actual_holdout_images,
        'estimated_total_images': estimated_total_images,
        'estimated_percentage': estimated_image_percentage
    }

    if total_dataset_images > 0:
        holdout_pct = (actual_holdout_images / total_dataset_images * 100)
        train_images = total_dataset_images - actual_holdout_images
        image_statistics.update({
            'total_dataset_images': total_dataset_images,
            'training_images': train_images,
            'holdout_percentage': holdout_pct,
            'training_percentage': 100.0 - holdout_pct
        })

    stats_data = {
        'generation_timestamp': pd.Timestamp.now().isoformat(),
        'random_seed': random_seed,
        'selection_criteria': {
            'material': 'AlSi10Mg',
            'laser_mode': 'CW',
            'layer': 1,
            'duty_cycle': 1.0
        },
        'trackid_statistics': {
            'total_available': total_available_trackids,
            'selected_count': selected_count,
            'percentage_of_trackids': trackid_percentage
        },
        'image_statistics': image_statistics,
        'holdout_regime_breakdown': image_stats['regime_image_counts'],
        'selected_trackids': selected_trackids,
        'data_directories_searched': image_stats['data_directories_searched']
    }

    # Add dataset regime distribution if available
    if dataset_stats:
        stats_data['dataset_regime_distribution'] = dataset_stats['regime_distribution']
        stats_data['dataset_statistics'] = {
            'total_trackids': dataset_stats['total_trackids'],
            'trackids_without_regime': len(dataset_stats['trackids_without_regime'])
        }

        # Calculate representation ratios for each regime
        representation_analysis = {}
        for regime in dataset_stats['regime_distribution'].keys():
            dataset_pct = (dataset_stats['regime_distribution'][regime]['image_count'] / total_dataset_images * 100) if total_dataset_images > 0 else 0

            if regime in image_stats['regime_image_counts']:
                holdout_pct = (image_stats['regime_image_counts'][regime]['image_count'] / actual_holdout_images * 100) if actual_holdout_images > 0 else 0
                representation = holdout_pct / dataset_pct if dataset_pct > 0 else 0
            else:
                holdout_pct = 0.0
                representation = 0.0

            representation_analysis[regime] = {
                'dataset_percentage': dataset_pct,
                'holdout_percentage': holdout_pct,
                'representation_ratio': representation,
                'status': 'well_represented' if 0.8 <= representation <= 1.2 else ('under_represented' if representation < 0.8 else 'over_represented')
            }

        stats_data['representation_analysis'] = representation_analysis
    
    import json
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n{'='*70}")
        print("✅ TEST HOLDOUT FILES CREATED")
        print(f"{'='*70}")
        print(f"  Holdout file: {output_path}")
        print(f"  Statistics file: {stats_file}")

        print(f"\n{'='*70}")
        print("📊 HOLDOUT SUMMARY")
        print(f"{'='*70}")
        print(f"  Selected trackids: {selected_count}/{total_available_trackids} ({trackid_percentage:.1f}%)")

        if total_dataset_images > 0:
            holdout_pct = (actual_holdout_images / total_dataset_images * 100)
            train_images = total_dataset_images - actual_holdout_images
            train_pct = (train_images / total_dataset_images * 100)

            print(f"\n  Total dataset:      {total_dataset_images:,} images (100.0%)")
            print(f"  Training set:       {train_images:,} images ({train_pct:.1f}%)")
            print(f"  Test holdout set:   {actual_holdout_images:,} images ({holdout_pct:.1f}%)")
        else:
            print(f"\n  Holdout images: {actual_holdout_images}")
            print(f"  Estimated holdout percentage: {actual_image_percentage}")

        print(f"\n{'='*70}")
        print("🎯 REGIME DISTRIBUTION COMPARISON")
        print(f"{'='*70}")

        if dataset_stats and 'regime_image_counts' in image_stats:
            # Header
            print(f"\n{'Regime':<20} {'Dataset':<25} {'Holdout':<25} {'Representation':<15}")
            print(f"{'-'*20} {'-'*25} {'-'*25} {'-'*15}")

            # Get all regimes from both dataset and holdout
            all_regimes = set(dataset_stats['regime_distribution'].keys()) | set(image_stats['regime_image_counts'].keys())

            for regime in sorted(all_regimes):
                # Dataset stats
                if regime in dataset_stats['regime_distribution']:
                    dataset_info = dataset_stats['regime_distribution'][regime]
                    dataset_count = dataset_info['image_count']
                    dataset_pct = (dataset_count / total_dataset_images * 100) if total_dataset_images > 0 else 0
                    dataset_str = f"{dataset_count:,} ({dataset_pct:.1f}%)"
                else:
                    dataset_count = 0
                    dataset_pct = 0.0
                    dataset_str = "0 (0.0%)"

                # Holdout stats
                if regime in image_stats['regime_image_counts']:
                    holdout_info = image_stats['regime_image_counts'][regime]
                    holdout_count = holdout_info['image_count']
                    holdout_pct = (holdout_count / actual_holdout_images * 100) if actual_holdout_images > 0 else 0
                    holdout_str = f"{holdout_count:,} ({holdout_pct:.1f}%)"
                else:
                    holdout_count = 0
                    holdout_pct = 0.0
                    holdout_str = "0 (0.0%)"

                # Calculate representation ratio
                if dataset_pct > 0:
                    representation = holdout_pct / dataset_pct
                    if 0.8 <= representation <= 1.2:
                        rep_str = f"✓ {representation:.2f}x"  # Well represented
                    elif representation < 0.8:
                        rep_str = f"⚠ {representation:.2f}x"  # Under-represented
                    else:
                        rep_str = f"⚡ {representation:.2f}x"  # Over-represented
                else:
                    rep_str = "N/A"

                print(f"{regime:<20} {dataset_str:<25} {holdout_str:<25} {rep_str:<15}")

            print(f"\n  Legend:")
            print(f"    ✓ Well represented (0.8x - 1.2x of dataset proportion)")
            print(f"    ⚠ Under-represented (<0.8x)")
            print(f"    ⚡ Over-represented (>1.2x)")
        else:
            # Fallback to simple output if dataset stats not available
            for regime, info in image_stats['regime_image_counts'].items():
                regime_pct = (info['image_count'] / actual_holdout_images * 100) if actual_holdout_images > 0 else 0
                if total_dataset_images > 0:
                    regime_dataset_pct = (info['image_count'] / total_dataset_images * 100)
                    print(f"  {regime}:")
                    print(f"    {len(info['trackids'])} trackids, {info['image_count']:,} images")
                    print(f"    {regime_pct:.1f}% of holdout, {regime_dataset_pct:.1f}% of total dataset")
                else:
                    print(f"  {regime}: {len(info['trackids'])} trackids, {info['image_count']:,} images ({regime_pct:.1f}% of holdout)")

        print(f"{'='*70}")
    
    return stats_data

def main():
    parser = argparse.ArgumentParser(description='Generate test holdout file from experiment logbook')
    parser.add_argument('--output', type=str, default='ml/test_holdout_trackids.txt',
                       help='Output file path for test holdout trackids (default: ml/test_holdout_trackids.txt)')
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Random seed for reproducible trackid selection (default: None = truly random)')
    parser.add_argument('--data_dirs', type=str, nargs='*',
                       help='Data directories to search for actual image counts (optional)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information')
    
    args = parser.parse_args()
    
    print("Generating test holdout from experiment logbook...")
    print("="*50)
    
    try:
        # Step 1: Filter logbook
        if args.verbose:
            print("Step 1: Filtering logbook...")
        log_filtered = filter_logbook_for_test_holdout()
        
        if len(log_filtered) == 0:
            print("❌ No trackids found matching the filter criteria")
            return False
        
        print(f"Found {len(log_filtered)} trackids matching criteria:")
        print(f"  Material: AlSi10Mg (substrate and powder)")
        print(f"  Laser mode: Continuous wave (CW)")
        print(f"  Layer: 1")
        print(f"  Duty cycle: 1.0")
        
        # Step 2: Select trackids
        if args.verbose:
            print(f"\nStep 2: Selecting representative trackids...")
        selected_trackids, regime_trackids = select_test_holdout_trackids(
            log_filtered,
            target_percentage=15.0,
            tolerance=5.0,
            random_seed=args.random_seed,
            data_directories=args.data_dirs,
            verbose=args.verbose
        )
        
        # Step 3: Create output file
        if args.verbose:
            print(f"\nStep 3: Creating test holdout file with statistics...")
        stats_data = create_test_holdout_file(selected_trackids, regime_trackids, log_filtered, args.output, data_directories=args.data_dirs, random_seed=args.random_seed, verbose=args.verbose)
        
        # Summary
        print(f"\n{'='*70}")
        print("✅ TEST HOLDOUT GENERATION COMPLETE!")
        print(f"{'='*70}")
        if not args.verbose:  # If verbose, detailed stats already printed
            print(f"\nSelected trackids by melting regime:")
            for regime, trackids in regime_trackids.items():
                print(f"  {regime}: {trackids}")
            print(f"\nFor detailed statistics, run with --verbose flag")
        print(f"{'='*70}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating test holdout: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)