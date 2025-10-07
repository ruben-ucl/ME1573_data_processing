#!/usr/bin/env python3
"""
Migration Script: Bundle tuning_progress.json into run_info.json

This script migrates legacy tuning_progress.json files into the run_info.json
files to reduce directory clutter and consolidate related information.

Usage:
    python migrate_progress_files.py [--cleanup]
    
    --cleanup: Remove legacy tuning_progress.json files after migration
"""

import json
import argparse
from pathlib import Path
import datetime

def migrate_progress_files(cleanup=False):
    """Migrate legacy tuning_progress.json files into run_info.json files."""
    
    ml_root = Path(__file__).parent
    
    # Directories to check for hyperopt results
    hyperopt_dirs = [
        ml_root / 'logs' / 'pd_raw' / 'hyperopt_results',
        ml_root / 'logs' / 'cwt' / 'hyperopt_results'
    ]
    
    migrated_count = 0
    removed_count = 0
    
    for hyperopt_dir in hyperopt_dirs:
        if not hyperopt_dir.exists():
            continue
            
        print(f"Processing {hyperopt_dir}...")
        
        for run_dir in hyperopt_dir.iterdir():
            if not run_dir.is_dir():
                continue
                
            progress_file = run_dir / 'tuning_progress.json'
            run_info_file = run_dir / 'run_info.json'
            
            if not progress_file.exists():
                continue
                
            # Load progress data
            try:
                with open(progress_file) as f:
                    progress_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not read {progress_file}: {e}")
                continue
            
            # Load or create run_info
            if run_info_file.exists():
                try:
                    with open(run_info_file) as f:
                        run_info = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not read {run_info_file}: {e}")
                    continue
            else:
                # Create minimal run_info for legacy runs
                run_info = {
                    'run_id': run_dir.name,
                    'mode': 'unknown',
                    'start_time': 'legacy_run',
                    'total_configs_executed': progress_data.get('current', 0),
                    'config_files': {},
                    'version_mapping': {},
                    'is_legacy': True
                }
            
            # Add progress section if not exists
            if 'progress' not in run_info:
                run_info['progress'] = {
                    'completed_configs': progress_data.get('completed_configs', []),
                    'current': progress_data.get('current', 0),
                    'total': progress_data.get('total', 0),
                    'last_updated': progress_data.get('timestamp', datetime.datetime.now().isoformat())
                }
                
                # Save updated run_info
                try:
                    with open(run_info_file, 'w') as f:
                        json.dump(run_info, f, indent=2)
                    print(f"  Migrated: {run_dir.name}")
                    migrated_count += 1
                    
                    # Remove legacy file if cleanup requested
                    if cleanup:
                        progress_file.unlink()
                        removed_count += 1
                        print(f"    Removed: tuning_progress.json")
                        
                except Exception as e:
                    print(f"Warning: Could not save {run_info_file}: {e}")
            else:
                print(f"  Skipped: {run_dir.name} (already has progress section)")
                
                # Still remove legacy file if cleanup requested and progress exists
                if cleanup:
                    progress_file.unlink()
                    removed_count += 1
                    print(f"    Removed: tuning_progress.json")
    
    print(f"\nMigration complete:")
    print(f"  Migrated: {migrated_count} runs")
    if cleanup:
        print(f"  Removed: {removed_count} legacy files")

def main():
    parser = argparse.ArgumentParser(description='Migrate legacy tuning_progress.json files into run_info.json')
    parser.add_argument('--cleanup', action='store_true', 
                       help='Remove legacy tuning_progress.json files after migration')
    
    args = parser.parse_args()
    
    print("Migrating legacy tuning_progress.json files...")
    migrate_progress_files(cleanup=args.cleanup)

if __name__ == '__main__':
    main()