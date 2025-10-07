#!/usr/bin/env python3
"""
Migrate Experiment Logs to List Format

This script updates existing experiment logs to use consistent list format 
for architecture parameters like dense_dropout.

Changes:
- dense_dropout: 0.5 → "[0.5]"
- dense_dropout: "[0.3, 0.2]" → unchanged (already list string)
- Handles both PD and CWT experiment logs
"""

import pandas as pd
import ast
import json
from pathlib import Path
import argparse
import shutil
from datetime import datetime

# Local imports
from config import (
    get_pd_experiment_log_path, get_cwt_experiment_log_path,
    PD_LOGS_DIR, CWT_LOGS_DIR
)


def migrate_dense_dropout_to_list(value):
    """Convert dense_dropout values to list format."""
    if pd.isna(value) or value is None:
        return "[0.5]"  # Default
    
    if isinstance(value, str):
        # Check if already list format
        if value.startswith('[') and value.endswith(']'):
            return value  # Already list string
        else:
            # Single value string - convert to list
            try:
                float_val = float(value)
                return f"[{float_val}]"
            except ValueError:
                return "[0.5]"  # Fallback
    
    elif isinstance(value, (int, float)):
        # Single numeric value
        return f"[{value}]"
    
    elif isinstance(value, list):
        # Already a list - convert to string
        return str(value)
    
    else:
        return "[0.5]"  # Fallback


def backup_log_file(log_path):
    """Create a timestamped backup of the log file."""
    if not log_path.exists():
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = log_path.with_suffix(f'.backup_{timestamp}.csv')
    shutil.copy2(log_path, backup_path)
    return backup_path


def migrate_log_file(log_path, log_type='pd'):
    """Migrate a single experiment log file."""
    if not log_path.exists():
        print(f"⚠️  Log file does not exist: {log_path}")
        return False
    
    print(f"📄 Migrating {log_type.upper()} experiment log: {log_path}")
    
    # Create backup
    backup_path = backup_log_file(log_path)
    print(f"💾 Backup created: {backup_path}")
    
    # Load the log
    try:
        df = pd.read_csv(log_path, encoding='utf-8')
        print(f"   Loaded {len(df)} experiment records")
    except Exception as e:
        print(f"❌ Error loading log: {e}")
        return False
    
    # Track changes
    changes_made = 0
    
    # Process dense_dropout column
    if 'dense_dropout' in df.columns:
        print("   Processing dense_dropout column...")
        
        # Show some examples before migration
        print("   Examples before migration:")
        for i, val in enumerate(df['dense_dropout'].head(3)):
            print(f"     {i+1}. {val} (type: {type(val).__name__})")
        
        # Migrate values
        original_values = df['dense_dropout'].copy()
        df['dense_dropout'] = df['dense_dropout'].apply(migrate_dense_dropout_to_list)
        
        # Count changes
        changes_made = sum(original_values != df['dense_dropout'])
        print(f"   ✅ Modified {changes_made}/{len(df)} dense_dropout values")
        
        # Show examples after migration
        print("   Examples after migration:")
        for i, val in enumerate(df['dense_dropout'].head(3)):
            print(f"     {i+1}. {val}")
    
    else:
        print("   No dense_dropout column found")
    
    # Handle PD-specific dropout_rates column (combined format)
    if log_type == 'pd' and 'dropout_rates' in df.columns:
        print("   Processing PD dropout_rates column...")
        
        # PD stores [conv_dropout, dense_dropout] combined
        # Need to ensure dense_dropout part is in list format
        def fix_pd_dropout_rates(value):
            if pd.isna(value) or value is None:
                return "[0.2, [0.3, 0.2]]"  # Default PD format
            
            try:
                if isinstance(value, str) and value.startswith('['):
                    # Parse the list
                    parsed = ast.literal_eval(value)
                    if len(parsed) >= 2:
                        conv_dropout = parsed[0]  # Should be single float
                        dense_dropout = parsed[1]  # Might be single value or list
                        
                        # Ensure dense_dropout is a list
                        if not isinstance(dense_dropout, list):
                            dense_dropout = [dense_dropout]
                        
                        return str([conv_dropout, dense_dropout])
                    else:
                        return value  # Keep as is if can't parse properly
                else:
                    return value  # Keep as is
            except:
                return value  # Keep as is if parsing fails
        
        original_dropout_rates = df['dropout_rates'].copy()
        df['dropout_rates'] = df['dropout_rates'].apply(fix_pd_dropout_rates)
        
        dropout_changes = sum(original_dropout_rates != df['dropout_rates'])
        print(f"   ✅ Modified {dropout_changes}/{len(df)} dropout_rates values")
        changes_made += dropout_changes
    
    # Save the migrated file
    if changes_made > 0:
        try:
            df.to_csv(log_path, index=False, encoding='utf-8')
            print(f"✅ Migration completed: {changes_made} total changes saved")
        except Exception as e:
            print(f"❌ Error saving migrated log: {e}")
            return False
    else:
        print("   No changes needed - log already in correct format")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Migrate experiment logs to list format')
    parser.add_argument('--log_type', choices=['pd', 'cwt', 'both'], default='both',
                       help='Which logs to migrate (default: both)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be changed without making changes')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    print("🔄 EXPERIMENT LOG MIGRATION")
    print("=" * 40)
    print("Converting architecture parameters to consistent list format")
    print(f"Target: dense_dropout single values → list format")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No changes will be made")
    
    print()
    
    success_count = 0
    total_count = 0
    
    # Migrate PD logs
    if args.log_type in ['pd', 'both']:
        pd_log_path = get_pd_experiment_log_path()
        total_count += 1
        
        if args.dry_run:
            if pd_log_path.exists():
                df = pd.read_csv(pd_log_path, encoding='utf-8')
                print(f"🔍 PD Log Analysis: {len(df)} records")
                if 'dense_dropout' in df.columns:
                    non_list_count = sum(~df['dense_dropout'].astype(str).str.startswith('['))
                    print(f"   Would migrate {non_list_count} dense_dropout values")
                else:
                    print("   No dense_dropout column")
            else:
                print(f"🔍 PD Log: Does not exist ({pd_log_path})")
        else:
            if migrate_log_file(pd_log_path, 'pd'):
                success_count += 1
        print()
    
    # Migrate CWT logs  
    if args.log_type in ['cwt', 'both']:
        cwt_log_path = get_cwt_experiment_log_path()
        total_count += 1
        
        if args.dry_run:
            if cwt_log_path.exists():
                df = pd.read_csv(cwt_log_path, encoding='utf-8')
                print(f"🔍 CWT Log Analysis: {len(df)} records")
                if 'dense_dropout' in df.columns:
                    non_list_count = sum(~df['dense_dropout'].astype(str).str.startswith('['))
                    print(f"   Would migrate {non_list_count} dense_dropout values")
                else:
                    print("   No dense_dropout column")
            else:
                print(f"🔍 CWT Log: Does not exist ({cwt_log_path})")
        else:
            if migrate_log_file(cwt_log_path, 'cwt'):
                success_count += 1
        print()
    
    # Summary
    if args.dry_run:
        print("✅ Dry run completed - use without --dry_run to apply changes")
    else:
        print(f"✅ Migration completed: {success_count}/{total_count} logs migrated successfully")
        print("\n📋 Next Steps:")
        print("1. Test hyperparameter tuner with --mode quick to verify deduplication works")
        print("2. Check that new experiments are correctly deduplicated")
        print("3. Backup files can be removed once migration is verified")


if __name__ == "__main__":
    main()