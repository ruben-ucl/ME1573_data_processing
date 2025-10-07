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

def select_test_holdout_trackids(log_filtered, target_count=100, verbose=True):
    """
    Select one trackid from each melting regime for test holdout.
    
    Args:
        log_filtered: Filtered logbook DataFrame
        target_count: Target number of test images (~100)
        verbose: Print detailed information
        
    Returns:
        list: Selected trackids for test holdout
    """
    if 'Melting regime' not in log_filtered.columns:
        raise ValueError("Melting regime column not found in logbook")
    
    # Get melting regime counts
    regime_counts = log_filtered['Melting regime'].value_counts()
    if verbose:
        print(f"\nMelting regime distribution in filtered data:")
        for regime, count in regime_counts.items():
            print(f"  {regime}: {count} trackids")
    
    # Select one trackid from each regime
    selected_trackids = []
    regime_trackids = {}
    
    for regime in regime_counts.index:
        regime_data = log_filtered[log_filtered['Melting regime'] == regime]
        # Select first trackid from each regime (could be randomized)
        selected_trackid = regime_data['trackid'].iloc[0]
        selected_trackids.append(selected_trackid)
        regime_trackids[regime] = [selected_trackid]
        
        if verbose:
            print(f"  Selected from {regime}: {selected_trackid}")
    
    # If we need more trackids to reach target_count, add more from largest regime
    if len(selected_trackids) < target_count // 20:  # Rough estimate: ~20 images per trackid
        largest_regime = regime_counts.index[0]  # Already sorted by count
        largest_regime_data = log_filtered[log_filtered['Melting regime'] == largest_regime]
        
        # Add more trackids from largest regime
        remaining_trackids = largest_regime_data['trackid'].tolist()
        additional_needed = max(0, (target_count // 20) - len(selected_trackids))
        
        for trackid in remaining_trackids[1:additional_needed+1]:  # Skip first (already selected)
            if trackid not in selected_trackids:
                selected_trackids.append(trackid)
                regime_trackids[largest_regime].append(trackid)
        
        if verbose and additional_needed > 0:
            print(f"\nAdded {len(regime_trackids[largest_regime])-1} additional trackids from {largest_regime} regime")
    
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
    
    if verbose and total_images > 0:
        print(f"\nActual image count analysis:")
        for regime, info in regime_image_counts.items():
            print(f"  {regime}: {info['image_count']} images from {len(info['trackids'])} trackids")
            if verbose:
                for tid, count in info['trackid_details'].items():
                    print(f"    {tid}: {count} images")
    elif verbose:
        print(f"\nWarning: Could not find images in common directories: {data_directories}")
        print(f"Image counts will be estimated. Actual counts will be determined during training.")
    
    return {
        'total_images': total_images,
        'regime_image_counts': regime_image_counts,
        'trackid_image_counts': trackid_image_counts,
        'data_directories_searched': data_directories
    }

def create_test_holdout_file(selected_trackids, regime_trackids, log_filtered, output_file, data_directories=None, verbose=True):
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
    with open(output_path, 'w') as f:
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
    stats_data = {
        'generation_timestamp': pd.Timestamp.now().isoformat(),
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
        'image_statistics': {
            'holdout_images': actual_holdout_images,
            'estimated_total_images': estimated_total_images,
            'estimated_percentage': estimated_image_percentage
        },
        'regime_breakdown': image_stats['regime_image_counts'],
        'selected_trackids': selected_trackids,
        'data_directories_searched': image_stats['data_directories_searched']
    }
    
    import json
    with open(stats_file, 'w') as f:
        json.dump(stats_data, f, indent=2)
    
    if verbose:
        print(f"\n✅ Test holdout files created:")
        print(f"  Holdout file: {output_path}")
        print(f"  Statistics file: {stats_file}")
        print(f"\n📊 Holdout Summary:")
        print(f"  Selected trackids: {selected_count}/{total_available_trackids} ({trackid_percentage:.1f}%)")
        print(f"  Holdout images: {actual_holdout_images}")
        print(f"  Estimated holdout percentage: {actual_image_percentage}")
        
        print(f"\n🎯 Regime Distribution:")
        for regime, info in image_stats['regime_image_counts'].items():
            print(f"  {regime}: {len(info['trackids'])} trackids, {info['image_count']} images")
    
    return stats_data

def main():
    parser = argparse.ArgumentParser(description='Generate test holdout file from experiment logbook')
    parser.add_argument('--output', type=str, default='ml/test_holdout_trackids.txt',
                       help='Output file path for test holdout trackids (default: ml/test_holdout_trackids.txt)')
    parser.add_argument('--target_count', type=int, default=100,
                       help='Target number of test images (default: 100)')
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
            log_filtered, target_count=args.target_count, verbose=args.verbose
        )
        
        # Step 3: Create output file
        if args.verbose:
            print(f"\nStep 3: Creating test holdout file with statistics...")
        stats_data = create_test_holdout_file(selected_trackids, regime_trackids, log_filtered, args.output, data_directories=args.data_dirs, verbose=args.verbose)
        
        # Summary
        print(f"\n✅ Test holdout generation complete!")
        if not args.verbose:  # If verbose, this was already printed in create_test_holdout_file
            print(f"Selected trackids by melting regime:")
            for regime, trackids in regime_trackids.items():
                print(f"  {regime}: {trackids}")
        
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