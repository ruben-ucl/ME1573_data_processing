#!/usr/bin/env python3
"""
Backfill Config Paths Script

This script reads all existing config files and automatically adds the file paths
and config numbers to the experiment log where they are missing by comparing
hyperparameter values.

Author: AI Assistant
"""

import json
import os
import sys
from glob import glob
from pathlib import Path
import pandas as pd


def load_config_file(config_path):
    """Load and return config file contents."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {config_path}: {e}")
        return None


def extract_comparable_params(config):
    """Extract parameters that can be compared between config file and log."""
    return {
        'learning_rate': config.get('learning_rate'),
        'batch_size': config.get('batch_size'),
        'epochs': config.get('epochs'),
        'conv_filters': str(config.get('conv_filters', [])),
        'dense_units': str(config.get('dense_units', [])),
        'l2_regularization': config.get('l2_regularization'),
        'conv_dropout': config.get('conv_dropout'),
        'dense_dropout': str(config.get('dense_dropout', [])),
        'use_batch_norm': config.get('use_batch_norm'),
        'optimizer': config.get('optimizer', 'Adam'),
        'early_stopping_patience': config.get('early_stopping_patience'),
        'lr_reduction_patience': config.get('lr_reduction_patience'),
        'use_class_weights': config.get('use_class_weights'),
        'augment_fraction': config.get('augment_fraction'),
        'time_shift_range': config.get('time_shift_range'),
        'stretch_probability': config.get('stretch_probability'),
        'noise_probability': config.get('noise_probability'),
        'amplitude_scale_probability': config.get('amplitude_scale_probability'),
    }


def extract_log_params(row):
    """Extract comparable parameters from experiment log row."""
    # Parse dropout rates
    dropout_str = str(row['dropout_rates'])
    if '[' in dropout_str and ',' in dropout_str:
        # Format like "[0.2, [0.3, 0.2]]"
        parts = dropout_str.split(',')
        conv_dropout = float(parts[0].strip('[]'))
        dense_dropout = str([float(x.strip('[] ')) for x in parts[1:]])
    else:
        conv_dropout = 0.2  # default
        dense_dropout = str([0.3, 0.2])  # default
    
    return {
        'learning_rate': float(row['learning_rate']),
        'batch_size': int(row['batch_size']),
        'epochs': int(row['epochs']),
        'conv_filters': str(row['conv_filters']),
        'dense_units': str(row['dense_units']),
        'l2_regularization': float(row['l2_reg']),
        'conv_dropout': conv_dropout,
        'dense_dropout': dense_dropout,
        'use_batch_norm': str(row['batch_norm']).upper() == 'TRUE',
        'optimizer': str(row['optimizer']),
        'early_stopping_patience': int(row['early_stopping_patience']),
        'lr_reduction_patience': int(row['lr_reduction_patience']),
        'use_class_weights': str(row['class_weights']).upper() == 'TRUE',
        'augment_fraction': float(row.get('augment_fraction', 0.5)),
        'time_shift_range': int(row.get('time_shift_range', 5)),
        'stretch_probability': float(row.get('stretch_probability', 0.3)),
        'noise_probability': float(row.get('noise_probability', 0.5)),
        'amplitude_scale_probability': float(row.get('amplitude_scale_probability', 0.5)),
    }


def params_match(config_params, log_params, tolerance=1e-6):
    """Check if config and log parameters match within tolerance."""
    for key in config_params:
        if key not in log_params:
            continue
            
        config_val = config_params[key]
        log_val = log_params[key]
        
        # Handle None values - if config value is None, skip comparison (legacy configs)
        if config_val is None:
            continue
        if log_val is None:
            continue
        
        # Handle string comparisons
        if isinstance(config_val, str) and isinstance(log_val, str):
            if config_val.strip() != log_val.strip():
                return False
        # Handle boolean comparisons
        elif isinstance(config_val, bool) or isinstance(log_val, bool):
            if bool(config_val) != bool(log_val):
                return False
        # Handle numeric comparisons
        elif isinstance(config_val, (int, float)) and isinstance(log_val, (int, float)):
            if abs(float(config_val) - float(log_val)) > tolerance:
                return False
        # Handle other types
        else:
            if str(config_val) != str(log_val):
                return False
                
    return True


def main():
    print("🔍 Backfilling missing config file paths in experiment log...")
    
    # Load experiment log
    log_path = "ml/logs/experiment_log.csv"
    if not Path(log_path).exists():
        print(f"❌ Experiment log not found: {log_path}")
        return False
        
    df = pd.read_csv(log_path)
    print(f"📊 Loaded experiment log with {len(df)} entries")
    
    # Find all config files
    config_pattern = "ml/logs/hyperopt_results/*/config_*.json"
    config_files = glob(config_pattern)
    print(f"📁 Found {len(config_files)} config files")
    
    if not config_files:
        print("❌ No config files found")
        return False
    
    # Load all configs with metadata
    configs_data = []
    for config_path in config_files:
        config = load_config_file(config_path)
        if config is None:
            continue
            
        # Extract run info from path
        path_parts = Path(config_path).parts
        run_id = path_parts[-2]  # e.g., "smart_run_2025-08-21_16-57-50"
        config_filename = path_parts[-1]  # e.g., "config_018.json"
        config_number = int(config_filename.split('_')[1].split('.')[0])
        
        configs_data.append({
            'file_path': str(Path(config_path).resolve()),
            'run_id': run_id,
            'config_number': config_number,
            'params': extract_comparable_params(config)
        })
    
    print(f"📋 Successfully loaded {len(configs_data)} config files")
    
    # Process each log entry that's missing config info
    matches_found = 0
    rows_updated = []
    
    for idx, row in df.iterrows():
        # Skip if already has config file path
        config_file_val = row.get('config_file')
        if pd.notna(config_file_val) and str(config_file_val).strip() and str(config_file_val) != 'nan':
            continue
        
        # Skip non-hyperopt entries
        if row.get('source') != 'hyperopt':
            continue
            
        version = row['version']
        print(f"🔍 Processing {version}...")
        
        try:
            log_params = extract_log_params(row)
        except Exception as e:
            print(f"⚠️  Could not extract params from {version}: {e}")
            continue
        
        # Find matching config file
        best_match = None
        for config_data in configs_data:
            if params_match(config_data['params'], log_params):
                best_match = config_data
                break
        
        # Debug output for first few mismatches
        if not best_match and version in ['v001', 'v002']:
            print(f"🔍 Debug {version}:")
            print(f"   Log params: {log_params}")
            # Find the config from quick_run that should match
            matching_config = None
            for config_data in configs_data:
                if config_data['run_id'] == 'quick_run_2025-08-19_14-31-59' and config_data['config_number'] == int(version[1:]):
                    matching_config = config_data
                    break
            if matching_config:
                print(f"   Expected config: {matching_config['params']}")
                print(f"   Match check: {params_match(matching_config['params'], log_params)}")
            else:
                print(f"   No config found for run quick_run_2025-08-19_14-31-59 config {int(version[1:])}")
        
        if best_match:
            # Update the row
            df.at[idx, 'config_file'] = best_match['file_path']
            df.at[idx, 'hyperopt_run_id'] = best_match['run_id']
            df.at[idx, 'config_number_in_run'] = best_match['config_number']
            
            matches_found += 1
            rows_updated.append(version)
            print(f"✅ Matched {version} -> {best_match['file_path']}")
        else:
            print(f"❌ No config file match found for {version}")
    
    print(f"\n📈 Summary:")
    print(f"   Config matches found: {matches_found}")
    print(f"   Rows updated: {rows_updated}")
    
    if matches_found > 0:
        # Create backup
        backup_path = log_path.replace('.csv', '_backup_before_backfill.csv')
        df_original = pd.read_csv(log_path)
        df_original.to_csv(backup_path, index=False)
        print(f"💾 Created backup: {backup_path}")
        
        # Save updated log
        df.to_csv(log_path, index=False)
        print(f"✅ Updated experiment log saved to: {log_path}")
        
        # Show sample of changes
        if rows_updated:
            print(f"\n📋 Updated versions: {', '.join(rows_updated[:10])}")
            if len(rows_updated) > 10:
                print(f"   ... and {len(rows_updated) - 10} more")
    else:
        print("ℹ️  No updates needed - all entries either have config files or no matches found")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)