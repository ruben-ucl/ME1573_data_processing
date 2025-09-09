#!/usr/bin/env python3
"""
Hyperparameter Optimization Script for PD Signal Classifier

This script systematically tests different hyperparameter combinations 
using a grid search approach. It runs the main training script with 
different configurations and tracks all results.

Author: AI Assistant
"""

# Ensure UTF-8 encoding for all I/O operations
import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import argparse
import ast
import datetime
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    get_data_dir, get_timing_database_path, get_experiment_log_path,
    get_config_template, HYPEROPT_RESULTS_DIR, normalize_path, format_version,
    convert_numpy_types, get_next_version_from_log
)

class HyperparameterTuner:
    """Manages hyperparameter optimization experiments."""
    
    def __init__(self, base_config=None, output_root=None, verbose=False, concise=False):
        self.base_config = base_config or get_config_template()
        # Use centralized configuration for output directory
        self.base_output_root = Path(output_root) if output_root else HYPEROPT_RESULTS_DIR
        self.results = []
        self.failed_configs = []
        
        # Store verbosity settings
        self.verbose = verbose
        self.concise = concise
        
        # Create base output directory
        self.base_output_root.mkdir(parents=True, exist_ok=True)
        
        # Will be set when run_optimization is called
        self.output_root = None
        self.progress_file = None
        self.results_file = None
        
        # Timing and progress tracking
        self.total_configs = 0
        
        # Time estimation
        self.timing_db_file = str(get_timing_database_path())
        self.timing_data = self.load_timing_database()
        
        # Cache for experiment log to avoid repeated file reads
        self._experiment_log_cache = None
        self._experiment_log_last_modified = 0
        
        if not self.concise:
            print(f"Hyperparameter Tuner initialized")
            print(f"Base output directory: {self.base_output_root.as_posix()}")
    
    def _safe_parse_list(self, value, default=None):
        """Safely parse a string representation of a list using ast.literal_eval."""
        if default is None:
            default = []
        
        if pd.isna(value) or value is None:
            return default
        
        try:
            # Handle string representation of lists
            if isinstance(value, str):
                return ast.literal_eval(value)
            elif isinstance(value, (list, tuple)):
                return list(value)
            else:
                return default
        except (ValueError, SyntaxError, TypeError):
            return default
    
    def _config_signature(self, config):
        """Create a unique signature for a configuration to enable deduplication."""
        # Key hyperparameters that define a unique experiment
        # Based on all parameters in search space and experiment log
        key_params = [
            # Core training parameters
            'learning_rate',
            'batch_size', 
            'epochs',
            'k_folds',
            # Architecture parameters
            'conv_filters',
            'dense_units',
            # Regularization parameters  
            'conv_dropout',
            'dense_dropout',
            'l2_regularization',
            'early_stopping_patience',
            'use_class_weights',
            # Data augmentation parameters (critical for deduplication)
            'augment_fraction',
            'time_shift_range',
            'stretch_probability',
            'stretch_scale',
            'noise_probability',
            'noise_std',
            'amplitude_scale_probability',
            'amplitude_scale'
        ]
        
        signature_parts = []
        for param in key_params:
            value = config.get(param)
            # Normalize the value for consistent comparison
            if isinstance(value, list):
                # Handle nested lists properly for dense_dropout
                if param == 'dense_dropout' and len(value) > 0 and isinstance(value[0], (int, float)):
                    # This is a list of numbers, keep as is
                    signature_parts.append(str(value))
                elif param == 'conv_filters' or param == 'dense_units':
                    # These should be sorted as integers
                    try:
                        signature_parts.append(str(sorted([int(x) for x in value])))
                    except (ValueError, TypeError):
                        signature_parts.append(str(value))
                else:
                    signature_parts.append(str(value))
            else:
                signature_parts.append(str(value))
        
        return '|'.join(signature_parts)
    
    def _load_previous_configs(self):
        """Load configurations from previous experiments to avoid duplication."""
        log_file = get_experiment_log_path()
        
        if not Path(log_file).exists():
            return set()
        
        try:
            df = pd.read_csv(log_file, encoding='utf-8')
            previous_signatures = set()
            
            for _, row in df.iterrows():
                # Parse dropout rates from combined format [conv_dropout, dense_dropout]
                dropout_rates = self._safe_parse_list(row.get('dropout_rates'), [])
                conv_dropout = dropout_rates[0] if len(dropout_rates) > 0 else 0.2
                dense_dropout = dropout_rates[1] if len(dropout_rates) > 1 else [0.3, 0.2]
                
                # Reconstruct config from log entry
                config = {
                    # Core training parameters
                    'learning_rate': row.get('learning_rate'),
                    'batch_size': int(row.get('batch_size', 0)) if pd.notna(row.get('batch_size')) else None,
                    'epochs': int(row.get('epochs', 0)) if pd.notna(row.get('epochs')) else None,
                    'k_folds': int(row.get('k_folds', 0)) if pd.notna(row.get('k_folds')) else None,
                    # Architecture parameters
                    'conv_filters': self._safe_parse_list(row.get('conv_filters')),
                    'dense_units': self._safe_parse_list(row.get('dense_units')),
                    # Regularization parameters
                    'conv_dropout': conv_dropout,
                    'dense_dropout': dense_dropout,
                    'l2_regularization': row.get('l2_reg'),
                    'early_stopping_patience': int(row.get('early_stopping_patience', 0)) if pd.notna(row.get('early_stopping_patience')) else None,
                    'use_class_weights': row.get('class_weights'),
                    # Data augmentation parameters
                    'augment_fraction': row.get('augment_fraction'),
                    'time_shift_range': int(row.get('time_shift_range', 0)) if pd.notna(row.get('time_shift_range')) else None,
                    'stretch_probability': row.get('stretch_probability'),
                    'stretch_scale': row.get('stretch_scale'),
                    'noise_probability': row.get('noise_probability'),
                    'noise_std': row.get('noise_std'),
                    'amplitude_scale_probability': row.get('amplitude_scale_probability'),
                    'amplitude_scale': row.get('amplitude_scale')
                }
                
                signature = self._config_signature(config)
                previous_signatures.add(signature)
                
            return previous_signatures
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load previous configs for deduplication: {e}")
            return set()
    
    def _deduplicate_configs(self, configs, skip_deduplication=False):
        """Remove configurations that have been tried before."""
        if skip_deduplication:
            return configs
        
        previous_signatures = self._load_previous_configs()
        
        if not previous_signatures:
            if self.verbose:
                print("No previous experiments found - using all configurations")
            return configs
        
        original_count = len(configs)
        unique_configs = []
        skipped_configs = []
        
        for config in configs:
            signature = self._config_signature(config)
            if signature not in previous_signatures:
                unique_configs.append(config)
            else:
                skipped_configs.append(config)
        
        skipped_count = len(skipped_configs)
        if skipped_count > 0:
            print(f"Smart deduplication: Skipped {skipped_count}/{original_count} configurations (already tested)")
            if self.verbose and skipped_configs:
                print("Skipped configurations:")
                for i, config in enumerate(skipped_configs[:5], 1):  # Show first 5
                    print(f"  {i}. LR={config.get('learning_rate')}, BS={config.get('batch_size')}, Drop={config.get('conv_dropout')}")
                if len(skipped_configs) > 5:
                    print(f"  ... and {len(skipped_configs) - 5} more")
        else:
            print("Smart deduplication: All configurations are new")
        
        return unique_configs
    
    def _get_duplicate_indices(self, configs):
        """Get indices of configurations that have been tried before."""
        previous_signatures = self._load_previous_configs()
        
        if not previous_signatures:
            return set()
        
        duplicate_indices = set()
        
        for i, config in enumerate(configs):
            signature = self._config_signature(config)
            if signature in previous_signatures:
                duplicate_indices.add(i)
        
        return duplicate_indices
    
    def _find_best_previous_config(self):
        """Find the best performing configuration from previous experiments."""
        log_file = get_experiment_log_path()
        
        if not Path(log_file).exists():
            return None
        
        try:
            df = pd.read_csv(log_file, encoding='utf-8')
            if df.empty or 'mean_val_accuracy' not in df.columns:
                return None
            
            # Find the row with highest mean validation accuracy
            best_idx = df['mean_val_accuracy'].idxmax()
            best_row = df.iloc[best_idx]
            
            # Parse dropout rates from combined format [conv_dropout, dense_dropout]
            dropout_rates = self._safe_parse_list(best_row.get('dropout_rates'), [])
            conv_dropout = dropout_rates[0] if len(dropout_rates) > 0 else 0.2
            dense_dropout = dropout_rates[1] if len(dropout_rates) > 1 else [0.3, 0.2]
            
            # Extract configuration from the best row
            best_config = {
                'learning_rate': best_row.get('learning_rate'),
                'batch_size': int(best_row.get('batch_size', 0)) if pd.notna(best_row.get('batch_size')) else None,
                'conv_dropout': conv_dropout,
                'dense_dropout': dense_dropout,
                'l2_regularization': best_row.get('l2_reg'),
                'conv_filters': self._safe_parse_list(best_row.get('conv_filters')),
                'dense_units': self._safe_parse_list(best_row.get('dense_units')),
                'early_stopping_patience': int(best_row.get('early_stopping_patience', 0)) if pd.notna(best_row.get('early_stopping_patience')) else None,
                'use_class_weights': best_row.get('class_weights'),
                'mean_val_accuracy': best_row.get('mean_val_accuracy'),
                'version': best_row.get('version')
            }
            
            if not self.concise:
                print(f"Best previous config found: {best_config.get('version', 'unknown')} with accuracy {best_config['mean_val_accuracy']:.4f}")
            
            return best_config
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not find best previous config: {e}")
            return None
    
    def _get_neighboring_values(self, param_name, current_value, search_space, radius=1):
        """Get neighboring values around the current best value within search radius."""
        if param_name not in search_space:
            return []
        
        value_list = search_space[param_name]
        
        # Handle special cases for complex data types
        if isinstance(current_value, list):
            # For list values, find exact match
            try:
                current_idx = None
                for i, val in enumerate(value_list):
                    if isinstance(val, list) and val == current_value:
                        current_idx = i
                        break
                
                if current_idx is None:
                    if self.verbose:
                        print(f"  Warning: Current {param_name} value {current_value} not found in search space")
                    return []
                    
            except (ValueError, TypeError):
                if self.verbose:
                    print(f"  Warning: Could not find {param_name} value {current_value} in search space")
                return []
        else:
            # For scalar values, find exact match
            try:
                current_idx = value_list.index(current_value)
            except (ValueError, TypeError):
                if self.verbose:
                    print(f"  Warning: Current {param_name} value {current_value} not found in search space")
                return []
        
        # Get neighboring values within radius
        neighbors = []
        for offset in range(-radius, radius + 1):
            if offset == 0:  # Skip the current value
                continue
            idx = current_idx + offset
            if 0 <= idx < len(value_list):
                neighbors.append(value_list[idx])
            elif self.verbose:
                direction = "higher" if offset > 0 else "lower"
                print(f"  Note: No {direction} value available for {param_name} (reached boundary)")
        
        return neighbors
    
    def define_search_space(self):
        """
        Define hyperparameter search space based on ML best practices.
        
        Values are chosen based on:
        - Learning rates: Log scale from 1e-4 to 1e-2
        - Batch sizes: Powers of 2, typical range
        - Dropout: Standard values that work well (separated for better smart search)
        - Architecture: Conservative variations
        - L2 reg: Log scale, including no regularization
        """
        
        return {
            # Tier 1: Highest Impact (test thoroughly)
            'learning_rate': [0.0001, 0.0005, 0.001, 0.002, 0.005],
            'batch_size': [8, 16, 32],
            
            # Tier 2: High Impact (test key values) - separated for better smart search
            'conv_dropout': [0.1, 0.2, 0.3],
            'dense_dropout': [[0.2, 0.1], [0.2, 0.2], [0.3, 0.2], [0.3, 0.3], [0.4, 0.3]],
            'l2_regularization': [0.0, 0.0001, 0.001, 0.01],
            
            # Tier 3: Architecture variations (test a few key combinations)
            'conv_filters': [
                [16, 32],           # Simplest model
                [32, 64],           # Simple model
                [16, 32, 64],       # Default model  
                [32, 64, 128],      # More complex model
            ],
            'dense_units': [
                [64],               # Single dense layer
                [128],              # Large single dense layer
                [128, 64],          # Default
                [256, 128],         # Larger capacity
            ],
            
            # Tier 4: Training control (test a few values)
            'early_stopping_patience': [8, 12],
            'use_class_weights': [True, False],
            
            # Tier 5: Augmentation parameters (for smart-augmentation mode)
            'augment_fraction': [0.0, 0.25, 0.5, 0.75, 1.0],
            'time_shift_range': [2, 5, 10],
            'stretch_probability': [0.0, 0.25, 0.5, 0.75, 1.0],
            'stretch_scale': [0.005, 0.1, 0.15, 0.2],
            'noise_probability': [0.0, 0.25, 0.5, 0.75, 1.0],
            'noise_std': [0.01, 0.02, 0.03, 0.05],
            'amplitude_scale_probability': [0.0, 0.25, 0.5, 0.75, 1.0],
            'amplitude_scale': [0.005, 0.1, 0.15, 0.2],
        }
    
    def generate_test_configs(self):
        """Generate minimal configurations for testing (2 configs, 2 folds, 2 epochs)."""
        configs = []
        base = self.base_config.copy()
        
        # Override base settings for fast testing
        base['k_folds'] = 2
        base['epochs'] = 2
        base['early_stopping_patience'] = 1  # Reduced for quick testing
        
        if not self.concise:
            print("Generating test configurations (2 configs, 2 folds, 2 epochs each)...")
        
        # Test config 1: Default settings with reduced training
        config1 = base.copy()
        config1['learning_rate'] = 0.001
        config1['batch_size'] = 16
        configs.append(config1)
        
        # Test config 2: Higher learning rate
        config2 = base.copy()
        config2['learning_rate'] = 0.005
        config2['batch_size'] = 16
        configs.append(config2)
        
        if not self.concise:
            print(f"Generated {len(configs)} test configurations")
        return configs
    
    def generate_quick_configs(self):
        """Generate a smaller set of configurations for quick testing (~15-20 configs)."""
        configs = []
        base = self.base_config.copy()
        
        # 1. Learning rate sweep (5 configs)
        if self.verbose:
            print("Generating learning rate configurations...")
        for lr in [0.0005, 0.001, 0.002, 0.005, 0.0001]:
            config = base.copy()
            config['learning_rate'] = lr
            configs.append(config)
        
        # 2. Batch size variations (3 configs)
        if self.verbose:
            print("Generating batch size configurations...")
        for bs in [8, 16, 32]:
            config = base.copy()
            config['learning_rate'] = 0.001  # Use default
            config['batch_size'] = bs
            configs.append(config)
        
        # 3. Key dropout combinations (4 configs)
        if self.verbose:
            print("Generating dropout configurations...")
        dropout_combos = [
            (0.1, [0.2, 0.1]),
            (0.2, [0.3, 0.2]), 
            (0.3, [0.4, 0.3]),
            (0.2, [0.2, 0.2])
        ]
        for conv_drop, dense_drop in dropout_combos:
            config = base.copy()
            config['learning_rate'] = 0.001
            config['batch_size'] = 16
            config['conv_dropout'] = conv_drop
            config['dense_dropout'] = dense_drop
            configs.append(config)
        
        # 4. Regularization comparison (3 configs)
        if self.verbose:
            print("Generating regularization configurations...")
        for l2_reg in [0.0, 0.001, 0.01]:
            config = base.copy()
            config['learning_rate'] = 0.001
            config['batch_size'] = 16
            config['l2_regularization'] = l2_reg
            configs.append(config)
        
        # 5. Class weighting comparison (2 configs) 
        if self.verbose:
            print("Generating class weighting configurations...")
        for use_weights in [True, False]:
            config = base.copy()
            config['learning_rate'] = 0.001
            config['batch_size'] = 16
            config['use_class_weights'] = use_weights
            configs.append(config)
        
        if not self.concise:
            print(f"Generated {len(configs)} quick configurations")
        return configs
    
    def generate_full_configs(self):
        """Generate full hyperparameter grid (warning: can be large!)."""
        return self._generate_configs_from_space(self.define_search_space())
    
    def generate_smart_configs(self, search_radius=1, grid_search=False, ignore_params=None, max_grid_size=100, mode='all'):
        """
        Generate smart configurations using focused parameter optimization modes.
        
        Args:
            search_radius: Number of neighboring values to test (±1 or ±2)
            grid_search: If True, do full grid search within smart space; if False, use OFAT
            ignore_params: List of parameters to ignore (use best previous value)
            max_grid_size: Maximum number of configs before limiting parameters
            mode: Parameter mode to focus on ('all', 'training', 'architecture', 'regularization', 'augmentation')
        """
        if ignore_params is None:
            ignore_params = []
        
        configs = []
        
        # Try to find the best previous configuration
        best_config = self._find_best_previous_config()
        search_space = self.define_search_space()
        
        if best_config is None:
            if not self.concise:
                print(f"No previous experiments found - using default smart search ({mode} mode)")
            return self._generate_default_smart_configs(mode=mode)
        
        mode_str = "grid search" if grid_search else "OFAT"
        if not self.concise:
            print(f"\nAdaptive Smart Mode - {mode.title()} Focus ({mode_str}): Building search around best config {best_config.get('version', 'unknown')}")
            print(f"Best accuracy: {best_config['mean_val_accuracy']:.4f}")
            print(f"Search radius: ±{search_radius}")
            if ignore_params:
                print(f"Ignoring parameters: {', '.join(ignore_params)}")
        
        # Use the best config as the base for all new configurations
        base_config = self.base_config.copy()
        
        # Define parameter groups
        parameter_groups = self._get_parameter_groups()
        
        # Update base config with best known values for all parameters
        all_params = set()
        for group_params in parameter_groups.values():
            all_params.update(group_params)
        
        for param in all_params:
            if best_config.get(param) is not None:
                base_config[param] = best_config[param]
        
        # Select parameters to optimize based on mode
        if mode == 'all':
            param_priority = self._get_all_params_priority()
        else:
            param_priority = parameter_groups.get(mode, [])
            if not param_priority:
                raise ValueError(f"Unknown mode: {mode}. Valid modes: {list(parameter_groups.keys()) + ['all']}")
        
        # Remove ignored parameters
        param_priority = [p for p in param_priority if p not in ignore_params]
        
        if not self.concise:
            print(f"Focusing on {len(param_priority)} parameters: {', '.join(param_priority)}")
        
        # Build search space around best values
        smart_search_space = {}
        for param_name in param_priority:
            if param_name not in search_space:
                continue
                
            current_value = best_config.get(param_name)
            if current_value is None:
                continue
            
            # Get neighboring values within search radius
            neighbors = self._get_neighboring_values(param_name, current_value, search_space, search_radius)
            
            if neighbors:
                smart_search_space[param_name] = [current_value] + neighbors
                if self.verbose:
                    print(f"Exploring {param_name}: current={current_value}, testing {len(smart_search_space[param_name])} values")
        
        if not smart_search_space:
            if not self.concise:
                print("Warning: No neighboring values found - using default smart search")
            return self._generate_default_smart_configs(mode=mode)
        
        if grid_search:
            # Calculate grid size before generating
            grid_size = 1
            for param, values in smart_search_space.items():
                grid_size *= len(values)
            
            if grid_size > max_grid_size:
                if not self.concise:
                    print(f"Warning: Smart grid search would generate {grid_size:,} configurations!")
                    print(f"Limiting to most important parameters for grid search...")
                
                # Limit to top parameters within the selected mode
                limited_params = param_priority[:min(4, len(param_priority))]
                limited_space = {}
                for param in limited_params:
                    if param in smart_search_space:
                        limited_space[param] = smart_search_space[param]
                
                # For remaining parameters, just use the current best value (no variation)
                for param in smart_search_space:
                    if param not in limited_space:
                        current_value = best_config.get(param)
                        if current_value is not None:
                            base_config[param] = current_value
                
                configs = self._generate_grid_configs_from_space(limited_space, base_config)
                
                if not self.concise:
                    print(f"Limited grid search to {len(configs)} configurations using params: {list(limited_space.keys())}")
            else:
                # Grid size is reasonable, proceed normally
                configs = self._generate_grid_configs_from_space(smart_search_space, base_config)
        else:
            # Generate OFAT configurations around the best point
            for param_name in param_priority:
                if param_name not in smart_search_space:
                    continue
                
                current_value = best_config.get(param_name)
                neighbors = [v for v in smart_search_space[param_name] if v != current_value]
                
                for neighbor_value in neighbors:
                    config = base_config.copy()
                    config[param_name] = neighbor_value
                    configs.append(config)
        
        if not self.concise:
            print(f"Generated {len(configs)} adaptive smart configurations for {mode} mode")
            
        return configs
    
    def _get_parameter_groups(self):
        """Define parameter groups for focused optimization modes."""
        return {
            'training': [
                'learning_rate',           # Most critical for convergence
                'batch_size',              # Affects gradient quality and memory
                'epochs',                  # Training duration
                'early_stopping_patience'  # Training control
            ],
            'architecture': [
                'conv_filters',    # CNN layer sizes
                'dense_units'      # Dense layer sizes
            ],
            'regularization': [
                'conv_dropout',        # Dropout after conv layers
                'dense_dropout',       # Dropout after dense layers
                'l2_regularization',   # Weight decay
                'use_class_weights'    # Class imbalance handling
            ],
            'augmentation': [
                'augment_fraction',                # How much training data to augment
                'time_shift_range',                # Time series shifting
                'stretch_probability',             # Time warping probability
                'stretch_scale',                   # Time warping magnitude
                'noise_probability',               # Noise injection probability
                'noise_std',                       # Noise magnitude
                'amplitude_scale_probability',     # Amplitude scaling probability
                'amplitude_scale'                  # Amplitude scaling magnitude
            ]
        }
    
    def _get_all_params_priority(self):
        """Get all parameters in priority order for 'all' mode."""
        return [
            # Training dynamics (highest priority)
            'learning_rate',
            'batch_size', 
            
            # Regularization (high priority)
            'conv_dropout',
            'dense_dropout',
            'l2_regularization',
            
            # Architecture (medium priority) 
            'conv_filters',
            'dense_units',
            
            # Training control (lower priority)
            'early_stopping_patience',
            'use_class_weights',
            
            # Augmentation (specialized focus)
            'augment_fraction',
            'time_shift_range',
            'stretch_probability',
            'stretch_scale', 
            'noise_probability',
            'noise_std',
            'amplitude_scale_probability',
            'amplitude_scale'
        ]
    
    def _generate_default_smart_configs(self, mode='all'):
        """Fallback to original smart config generation when no previous results exist."""
        configs = []
        base = self.base_config.copy()
        
        # Use a subset of the original smart config logic
        search_space = self.define_search_space()
        
        # Test key parameters with reasonable defaults
        if self.verbose:
            print("Generating default smart configurations...")
        
        # Learning rate sweep
        for lr in search_space['learning_rate'][:3]:  # Test first 3 values
            config = base.copy()
            config['learning_rate'] = lr
            configs.append(config)
        
        # Batch size variations
        for bs in search_space['batch_size']:
            config = base.copy()
            config['batch_size'] = bs
            configs.append(config)
            
        # Key dropout combinations (limit to 2 for efficiency)
        for conv_drop in search_space['conv_dropout'][:2]:
            for dense_drop in search_space['dense_dropout'][:2]:
                config = base.copy()
                config['conv_dropout'] = conv_drop
                config['dense_dropout'] = dense_drop
                configs.append(config)
        
        # L2 regularization
        for l2_reg in search_space['l2_regularization'][:3]:
            config = base.copy()
            config['l2_regularization'] = l2_reg
            configs.append(config)
        
        # Class weighting
        for use_weights in search_space['use_class_weights']:
            config = base.copy()
            config['use_class_weights'] = use_weights
            configs.append(config)
            
        if not self.concise:
            print(f"Generated {len(configs)} default smart configurations")
        return configs
    
    def generate_medium_configs(self):
        """
        Generate medium-sized configuration set using systematic OFAT approach.
        This is the original 'smart' mode logic - balanced thoroughness with efficiency.
        Tests ~25-30 configurations across all key hyperparameters.
        """
        configs = []
        base = self.base_config.copy()
        
        # 1. Learning rate sweep with default other params
        if self.verbose:
            print("Generating learning rate configurations...")
        for lr in [0.0001, 0.0005, 0.001, 0.002, 0.005]:
            config = base.copy()
            config['learning_rate'] = lr
            configs.append(config)
        
        # 2. Batch size variations with best LR from above (we'll assume 0.001 is good)
        if self.verbose:
            print("Generating batch size configurations...")
        for bs in [8, 16, 32]:
            config = base.copy()
            config['learning_rate'] = 0.001  # Use reasonable default
            config['batch_size'] = bs
            configs.append(config)
        
        # 3. Dropout variations
        if self.verbose:
            print("Generating dropout configurations...")
        for conv_drop in [0.1, 0.2, 0.3]:
            for dense_drop in [[0.2, 0.1], [0.3, 0.2], [0.4, 0.3]]:
                config = base.copy()
                config['learning_rate'] = 0.001
                config['batch_size'] = 16
                config['conv_dropout'] = conv_drop
                config['dense_dropout'] = dense_drop
                configs.append(config)
        
        # 4. Architecture variations
        if self.verbose:
            print("Generating architecture configurations...")
        architectures = [
            ([16, 32], [64]),
            ([16, 32, 64], [128, 64]),  # Default
            ([32, 64, 128], [256, 128])
        ]
        for conv_filters, dense_units in architectures:
            config = base.copy()
            config['learning_rate'] = 0.001
            config['batch_size'] = 16
            config['conv_filters'] = conv_filters
            config['dense_units'] = dense_units
            configs.append(config)
        
        # 5. Regularization sweep
        if self.verbose:
            print("Generating regularization configurations...")
        for l2_reg in [0.0, 0.0001, 0.001, 0.01]:
            config = base.copy()
            config['learning_rate'] = 0.001
            config['batch_size'] = 16
            config['l2_regularization'] = l2_reg
            configs.append(config)
        
        # 6. Class weighting comparison
        if self.verbose:
            print("Generating class weighting configurations...")
        for use_weights in [True, False]:
            config = base.copy()
            config['learning_rate'] = 0.001
            config['batch_size'] = 16
            config['use_class_weights'] = use_weights
            configs.append(config)
        
        if not self.concise:
            print(f"Generated {len(configs)} medium configurations")
        return configs
    
    def generate_augmentation_configs(self, use_grid_search=False):
        """Generate configurations focused on optimizing data augmentation parameters.
        
        Args:
            use_grid_search (bool): If True, use full grid search (many configs).
                                  If False, use OFAT approach (fewer configs, default).
        """
        configs = []
        base = self.base_config.copy()
        
        # Fix model architecture and training parameters at good defaults
        # Use parameters from v023 (the best performing config from backfill)
        base['learning_rate'] = 0.001
        base['batch_size'] = 16
        base['conv_filters'] = [32, 64, 128]
        base['dense_units'] = [128, 64]
        base['conv_dropout'] = 0.2
        base['dense_dropout'] = [0.3, 0.2]
        base['l2_regularization'] = 0.001
        base['use_batch_norm'] = True
        base['epochs'] = 50
        base['k_folds'] = 5
        base['early_stopping_patience'] = 10
        base['lr_reduction_patience'] = 5
        base['lr_reduction_factor'] = 0.5
        base['use_class_weights'] = True
        
        # Define augmentation parameter search space
        augmentation_space = {
            'time_shift_range': [2, 5, 10],
            'stretch_probability': [0.0, 0.25, 0.75, 1.0],
            'stretch_scale': [0.1, 0.2],
            'noise_probability': [0.0, 0.25, 0.75, 1.0],
            'noise_std': [0.01, 0.02, 0.03, 0.05],
            'amplitude_scale_probability': [0.0, 0.25, 0.75, 1.0],
            'amplitude_scale': [0.1, 0.2],
            'augment_fraction': [0.5, 1.0]
        }
        
        if use_grid_search:
            # Grid search mode: test all combinations
            if not self.concise:
                print("Generating grid search augmentation optimization configurations...")
                print("Fixed parameters: architecture, learning rate, batch size, regularization")
                print("Varying: augment_fraction, time_shift_range, stretch_probability,")
                print("         stretch_scale, noise_probability, noise_std,")
                print("         amplitude_scale_probability, amplitude_scale")
            
            configs = self._generate_grid_configs_from_space(augmentation_space, base)
            
            if not self.concise:
                print(f"Generated {len(configs)} grid search augmentation configurations")
        else:
            # OFAT mode: test each parameter individually (default)
            if not self.concise:
                print("Generating OFAT augmentation optimization configurations...")
                print("Fixed parameters: architecture, learning rate, batch size, regularization")
                print("Strategy: One Factor At a Time (OFAT) parameter impact analysis")
            
            configs = self._generate_augmentation_ofat(base, augmentation_space)
            
            if not self.concise:
                print(f"Generated {len(configs)} OFAT augmentation configurations")
        
        return configs
    
    def _generate_augmentation_ofat(self, base, augmentation_space):
        """Generate OFAT (One Factor At a Time) augmentation configurations.
        
        Tests each parameter individually while keeping others at baseline values.
        Much faster than grid search with ~20-25 configs instead of 2000+.
        """
        configs = []
        
        # Baseline values for OFAT (different from search space extremes)
        baseline_values = {
            'augment_fraction': 0.5,
            'time_shift_range': 5,
            'stretch_probability': 0.3,
            'stretch_scale': 0.1,
            'noise_probability': 0.5,
            'noise_std': 0.02,
            'amplitude_scale_probability': 0.5,
            'amplitude_scale': 0.1
        }
        
        # Baseline configuration (no augmentation)
        baseline = base.copy()
        baseline['augment_fraction'] = 0.0
        configs.append(baseline)
        
        # Test each parameter at extreme values while keeping others at baseline
        for param, test_values in augmentation_space.items():
            for value in test_values:
                # Skip baseline value for augment_fraction (already added as no-aug baseline)
                if param == 'augment_fraction' and value == 0.0:
                    continue
                    
                config = base.copy()
                # Set all parameters to baseline values
                config.update(baseline_values)
                # Vary only the current parameter
                config[param] = value
                configs.append(config)
        
        return configs
    
    def generate_anova_improvement_configs(self):
        """
        Generate targeted configurations to improve ANOVA analysis quality.
        
        Based on comprehensive analysis, focuses on parameters with high importance
        but high uncertainty, and adds balanced sampling for better statistical power.
        
        Returns ~80 configurations across 2 phases:
        - Phase 1 (50 configs): High-impact parameter sweeps  
        - Phase 2 (30 configs): Interaction effects and architecture balance
        """
        if not self.concise:
            print("Generating ANOVA improvement configurations...")
            print("Targeting parameters with high importance but high uncertainty")
        
        configs = []
        base = self.base_config.copy()
        
        # Use best known architecture as baseline
        best_config = self._find_best_previous_config()
        if best_config:
            # Use best values for stable parameters
            for param in ['epochs', 'k_folds', 'dropout_rates_max', 'conv_filters_progression']:
                if best_config.get(param) is not None:
                    base[param] = best_config[param]
        
        # Phase 1: High-impact parameter sweeps (50 configs)
        if not self.concise:
            print("Phase 1: High-impact parameter systematic sweeps...")
        
        # Learning rate: 25 configs - highest importance (33.9%) but high uncertainty
        learning_rates = [0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005]
        for lr in learning_rates:
            for batch_size in [16, 32]:  # Two common batch sizes
                config = base.copy()
                config['learning_rate'] = lr
                config['batch_size'] = batch_size
                configs.append(config)
                
        # Additional batch size sweep: 10 configs - second highest importance (20.5%)
        batch_sizes = [8, 12, 20, 24, 48, 64]
        for bs in batch_sizes:
            config = base.copy()
            config['batch_size'] = bs
            # Use mid-range learning rate for consistency
            config['learning_rate'] = 0.002
            configs.append(config)
        
        # Augment fraction sweep: 9 configs - third highest importance (8.8%) but very high uncertainty
        augment_fractions = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
        for af in augment_fractions:
            config = base.copy()
            config['augment_fraction'] = af
            configs.append(config)
        
        # Dropout and regularization sweep: 6 configs - moderate importance but need better coverage
        dropout_configs = [
            {'dropout_rates_mean': 0.1, 'l2_reg': 1e-5},
            {'dropout_rates_mean': 0.2, 'l2_reg': 1e-4}, 
            {'dropout_rates_mean': 0.3, 'l2_reg': 1e-3},
            {'dropout_rates_mean': 0.4, 'l2_reg': 1e-4},
            {'dropout_rates_mean': 0.15, 'l2_reg': 5e-5},
            {'dropout_rates_mean': 0.25, 'l2_reg': 5e-4}
        ]
        for dropout_config in dropout_configs:
            config = base.copy()
            config.update(dropout_config)
            configs.append(config)
        
        phase1_count = len(configs)
        if not self.concise:
            print(f"Phase 1 generated: {phase1_count} configs")
            print("Phase 2: Interaction effects and architecture balance...")
        
        # Phase 2: Interaction effects and architecture balance (30 configs)
        
        # Learning rate × batch size factorial: 12 configs
        lr_bs_factorial = [
            (0.001, 16), (0.001, 32), (0.001, 48),
            (0.002, 16), (0.002, 32), (0.002, 48), 
            (0.003, 16), (0.003, 32), (0.003, 48),
            (0.004, 16), (0.004, 32), (0.004, 48)
        ]
        for lr, bs in lr_bs_factorial:
            config = base.copy()
            config['learning_rate'] = lr
            config['batch_size'] = bs
            configs.append(config)
        
        # Learning rate × augment fraction interaction: 9 configs
        lr_af_factorial = [
            (0.001, 0.2), (0.001, 0.5), (0.001, 0.8),
            (0.002, 0.2), (0.002, 0.5), (0.002, 0.8),
            (0.003, 0.2), (0.003, 0.5), (0.003, 0.8)
        ]
        for lr, af in lr_af_factorial:
            config = base.copy()
            config['learning_rate'] = lr
            config['augment_fraction'] = af
            configs.append(config)
        
        # Architecture balance: 9 configs
        # Current analysis shows imbalance: [32,64,128] has 64 experiments vs others have ~10
        undersampled_architectures = [
            # Dense architectures
            {'dense_units': [64]},
            {'dense_units': [128]}, 
            {'dense_units': [256]},
            {'dense_units': [32, 64]},
            {'dense_units': [256, 128]},
            # Conv architectures  
            {'conv_filters': [16, 32]},
            {'conv_filters': [32, 64]},
            {'conv_filters': [64, 128, 256]},
            {'conv_filters': [16, 32, 64]}
        ]
        for arch_config in undersampled_architectures:
            config = base.copy()
            config.update(arch_config)
            configs.append(config)
        
        total_configs = len(configs)
        phase2_count = total_configs - phase1_count
        
        if not self.concise:
            print(f"Phase 2 generated: {phase2_count} configs")
            print(f"Total ANOVA improvement configurations: {total_configs}")
            print(f"Expected ANOVA benefits:")
            print(f"  - Reduce learning_rate uncertainty from ±0.251 to ~±0.15")
            print(f"  - Reduce batch_size uncertainty from ±0.332 to ~±0.20") 
            print(f"  - Reduce augment_fraction uncertainty from ±0.315 to ~±0.18")
            print(f"  - Better capture interaction effects between top parameters")
            print(f"  - Balance architecture sampling for improved statistical power")
        
        return configs
    
    
    def _generate_configs_from_space(self, param_space):
        """Generate all combinations from parameter space."""
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        configs = []
        for combination in product(*param_values):
            config = self.base_config.copy()
            for name, value in zip(param_names, combination):
                config[name] = value
            configs.append(config)
        
        return configs
    
    def _generate_grid_configs_from_space(self, param_space, base_config):
        """Generate all combinations from parameter space with custom base config."""
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        configs = []
        for combination in product(*param_values):
            config = base_config.copy()
            for name, value in zip(param_names, combination):
                config[name] = value
            configs.append(config)
        
        return configs
    
    def get_image_dimensions(self, config):
        """Get actual image dimensions used in training."""
        # For photodiode signal data, images are resized to (img_width, 2)
        # This matches the cv2.resize(img, (img_width, 2)) in PD_signal_classifier_v3.py:100
        
        img_width = config['img_width']
        img_height = 2  # Fixed for this specific application
        
        # Future enhancement: Could detect from data_dir or add img_height to config
        # For now, hardcode for photodiode signal processing
        
        return img_width, img_height
    
    def load_timing_database(self):
        """Load historical timing data for adaptive estimation."""
        if Path(self.timing_db_file).exists():
            try:
                with open(self.timing_db_file, 'r') as f:
                    data = json.load(f)
                if self.verbose:
                    print(f"Loaded timing database with {len(data.get('timing_records', []))} records")
                return data
            except Exception as e:
                if self.verbose:
                    print(f"Failed to load timing database: {e}")
                    print("Starting fresh")
        
        # Default timing database structure
        return {
            'timing_records': [],
            'base_time_per_complexity': 0.001,  # Base time in minutes per complexity unit
            'last_updated': None
        }
    
    def calculate_model_complexity(self, config):
        """Calculate a complexity score for the model configuration."""
        # Count total parameters (roughly)
        conv_params = 0
        input_channels = 1  # Starting with 1 channel (grayscale)
        
        for filters in config['conv_filters']:
            # Conv2D params = (kernel_size * kernel_size * input_channels + 1) * output_channels
            # Assume 3x3 kernels
            conv_params += (3 * 3 * input_channels + 1) * filters
            input_channels = filters
        
        # Dense parameters
        dense_params = 0
        # Assume flattened conv output feeds into first dense layer
        # Rough estimate: last conv layer * 10 (for spatial dimensions after pooling)
        prev_units = config['conv_filters'][-1] * 10 if config['conv_filters'] else 100
        
        for units in config['dense_units']:
            dense_params += (prev_units + 1) * units
            prev_units = units
        
        # Final output layer (binary classification)
        dense_params += (prev_units + 1) * 2
        
        total_params = conv_params + dense_params
        
        # Estimate actual epochs considering early stopping
        estimated_epochs = self.estimate_actual_epochs(config)
        
        # Complexity score = parameters * estimated actual epochs (not configured epochs)
        complexity = total_params * estimated_epochs
        
        return complexity
    
    def estimate_actual_epochs(self, config):
        """
        Estimate actual epochs considering early stopping based on historical data.
        
        Uses historical patterns to predict how many epochs will actually run
        before early stopping kicks in.
        """
        configured_epochs = config['epochs']
        
        # For very short training (test mode), assume full epochs
        if configured_epochs <= 5:
            return configured_epochs
            
        # Analyze historical early stopping patterns
        early_stopping_ratios = []
        
        for record in self.timing_data['timing_records']:
            record_config = record.get('config_summary', {})
            record_epochs = record_config.get('epochs', 50)
            
            # Skip if this is also a short training run
            if record_epochs <= 5:
                continue
                
            # Try to extract actual epochs from timing and complexity
            if 'actual_time' in record and 'time_per_complexity' in record:
                # Rough back-calculation of actual epochs
                # This is an approximation based on the relationship:
                # actual_time ≈ time_per_complexity * (params * actual_epochs) * data_factor
                
                total_params_estimate = 50000  # Rough estimate for typical model size
                data_factor = record.get('data_factor', 1.0)
                
                if record['time_per_complexity'] > 0 and data_factor > 0:
                    implied_complexity = record['actual_time'] / (record['time_per_complexity'] * data_factor)
                    implied_epochs = implied_complexity / total_params_estimate
                    
                    # Only use reasonable ratios (between 0.1 and 1.0)
                    if 0.1 <= implied_epochs <= record_epochs:
                        ratio = implied_epochs / record_epochs
                        early_stopping_ratios.append(ratio)
        
        # Calculate early stopping factor
        if early_stopping_ratios:
            # Use median to avoid outliers
            early_stopping_ratios.sort()
            n = len(early_stopping_ratios)
            if n >= 10:  # Need sufficient data
                median_ratio = early_stopping_ratios[n // 2]
                # Conservative approach: don't go below 0.3, but allow full training if data supports it
                early_stopping_factor = max(0.3, median_ratio)
            else:
                # Not enough data, use conservative default
                early_stopping_factor = 0.6  # Assume ~60% of epochs on average
        else:
            # No historical data, use reasonable default based on early stopping settings
            patience = config.get('early_stopping_patience', 10)
            # Heuristic: with patience=10, typical training runs ~40-80% of max epochs
            # Allow higher ratios for less aggressive early stopping
            if patience <= 5:
                early_stopping_factor = 0.5  # More aggressive stopping
            elif patience >= 15:
                early_stopping_factor = 0.9  # Less aggressive stopping - often trains to completion
            else:
                early_stopping_factor = 0.6  # Default moderate stopping
        
        estimated_epochs = configured_epochs * early_stopping_factor
        
        # Ensure reasonable bounds
        estimated_epochs = max(5, min(configured_epochs, estimated_epochs))
        
        return estimated_epochs
    
    def estimate_data_size_factor(self, config):
        """Estimate relative data loading/processing time based on configuration."""
        # Factors that affect data processing time:
        # - Batch size (larger = more efficient per sample, but more memory)
        # - Image dimensions (affects preprocessing time)
        # - K-folds (affects total data passes)
        
        batch_efficiency = 1.0 / (config['batch_size'] / 16.0)  # Normalize to batch_size=16
        
        # Image size calculation: get actual dimensions used in training
        img_width, img_height = self.get_image_dimensions(config)
        total_pixels = img_width * img_height
        baseline_pixels = 100 * 2  # Baseline: 100 width × 2 height = 200 pixels
        
        # Linear scaling with total pixels (not quadratic, since it's just resizing)
        image_factor = total_pixels / baseline_pixels
        
        fold_factor = config['k_folds'] / 5.0  # Normalize to 5 folds
        
        return batch_efficiency * image_factor * fold_factor
    
    def get_adaptive_time_estimate(self, config):
        """Get time estimate based on historical data and model complexity."""
        complexity = self.calculate_model_complexity(config)
        data_factor = self.estimate_data_size_factor(config)
        
        # Base estimate using learned time per complexity unit
        base_time = self.timing_data['base_time_per_complexity'] * complexity * data_factor
        
        # If we have recent timing records, adjust based on them
        recent_records = self.timing_data['timing_records'][-10:]  # Last 10 records
        if recent_records:
            # Calculate average actual vs predicted ratio
            ratios = []
            for record in recent_records:
                predicted = record['predicted_time']
                actual = record['actual_time']
                if predicted > 0:
                    ratios.append(actual / predicted)
            
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                base_time *= avg_ratio
        
        return max(base_time, 0.5)  # Minimum 0.5 minutes per epoch
    
    def update_timing_database(self, config, actual_time_minutes):
        """Update the timing database with actual training results."""
        complexity = self.calculate_model_complexity(config)
        data_factor = self.estimate_data_size_factor(config)
        
        # Calculate actual time per complexity unit
        actual_time_per_complexity = actual_time_minutes / (complexity * data_factor)
        
        # Store the record
        record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'config_summary': {
                'conv_filters': config['conv_filters'],
                'dense_units': config['dense_units'],
                'epochs': config['epochs'],
                'estimated_epochs': self.estimate_actual_epochs(config),
                'k_folds': config['k_folds'],
                'batch_size': config['batch_size'],
                'img_width': config['img_width'],
                'img_height': self.get_image_dimensions(config)[1],
                'total_pixels': config['img_width'] * self.get_image_dimensions(config)[1],
                'early_stopping_patience': config.get('early_stopping_patience', 10)
            },
            'complexity': complexity,
            'data_factor': data_factor,
            'predicted_time': self.get_adaptive_time_estimate(config),
            'actual_time': actual_time_minutes,
            'time_per_complexity': actual_time_per_complexity
        }
        
        self.timing_data['timing_records'].append(record)
        
        # Simple moving average update (much simpler than adaptive learning)
        alpha = 0.2  # Fixed learning rate
        self.timing_data['base_time_per_complexity'] = (
            (1 - alpha) * self.timing_data['base_time_per_complexity'] + 
            alpha * actual_time_per_complexity
        )
        self.timing_data['last_updated'] = datetime.datetime.now().isoformat()
        
        # Keep only last 50 records (reduced from 100)
        self.timing_data['timing_records'] = self.timing_data['timing_records'][-50:]
        
        # Save to disk
        with open(self.timing_db_file, 'w') as f:
            json.dump(self.timing_data, f, indent=2)
        
        if self.verbose:
            print(f"Updated timing database: {actual_time_minutes:.1f}min actual vs {record['predicted_time']:.1f}min predicted")
    
    def save_config_to_file(self, config, config_number_in_run):
        """Save configuration to JSON file with run-scoped sequential numbering."""
        config_file = Path(self.output_root) / f"config_{config_number_in_run:03d}.json"
        
        # Add metadata to config for better traceability
        enhanced_config = config.copy()
        enhanced_config['_metadata'] = {
            'config_number_in_run': config_number_in_run,
            'run_id': self.run_info['run_id'],
            'generated_at': datetime.datetime.now().isoformat()
        }
        
        # Convert NumPy types before saving
        clean_config = convert_numpy_types(enhanced_config)
        with open(config_file, 'w') as f:
            json.dump(clean_config, f, indent=2)
        
        # Update run info tracking
        self.run_info['config_files'][config_number_in_run] = str(config_file)
        self.run_info['total_configs_executed'] = max(self.run_info['total_configs_executed'], config_number_in_run)
        
        # Save updated run info
        run_info_file = self.output_root / "run_info.json"
        with open(run_info_file, 'w') as f:
            json.dump(self.run_info, f, indent=2)
        
        return str(config_file)
    
    def run_training_experiment_enhanced(self, config, config_number_in_run, config_file_path, version, verbose=False, concise=False):
        """Run training experiment with enhanced traceability."""
        try:
            # Start timing
            start_time = time.time()
            
            # Run training script (from ml directory)
            return self._execute_training_subprocess_enhanced(config, config_number_in_run, config_file_path, version, start_time, verbose, concise)
            
        except FileNotFoundError as e:
            print(f"Error: Training script not found - {e}")
            self.failed_configs.append({
                'config_number_in_run': config_number_in_run,
                'config': config,
                'error': f'Training script not found: {e}'
            })
            return None
        except Exception as e:
            print(f"Unexpected error in config {config_number_in_run}: {e}")
            self.failed_configs.append({
                'config_number_in_run': config_number_in_run,
                'config': config,
                'error': f'Unexpected error: {e}'
            })
            return None
    
    def run_training_experiment(self, config, config_id, verbose=False, concise=False):
        """Run training experiment with given configuration."""
        try:
            # Save config to file
            config_file = self.save_config_to_file(config, config_id)
            # Ensure absolute path for cross-directory compatibility
            config_file = str(Path(config_file).resolve())
            
            # Start timing
            start_time = time.time()
            
            # Run training script (from ml directory)
            return self._execute_training_subprocess(config, config_id, config_file, start_time, verbose, concise)
            
        except FileNotFoundError as e:
            print(f"Configuration {config_id} failed: Required file not found - {e}")
            self.failed_configs.append({
                'config_id': config_id,
                'config': config,
                'error': f'File not found: {e}'
            })
            return None
        except PermissionError as e:
            print(f"Configuration {config_id} failed: Permission denied - {e}")
            self.failed_configs.append({
                'config_id': config_id,
                'config': config,
                'error': f'Permission error: {e}'
            })
            return None
        except Exception as e:
            print(f"Configuration {config_id} failed with unexpected error: {e}")
            self.failed_configs.append({
                'config_id': config_id,
                'config': config,
                'error': f'Unexpected error: {e}'
            })
            return None
    
    def _execute_training_subprocess(self, config, config_id, config_file, start_time, verbose, concise):
        """Execute the training subprocess with proper error handling."""
        cmd = [
            sys.executable, 
            str(Path(__file__).parent / 'PD_signal_classifier_v3.py'), 
            '--config', config_file
        ]
        
        # Add concise flag if needed
        if concise:
            cmd.append('--concise')
            
        # Add source flag to identify hyperparameter optimization runs
        cmd.extend(['--source', 'hyperopt'])
        
        # Add progress tracking info for ETA calculation
        cmd.extend(['--current_config', str(config_id)])
        cmd.extend(['--total_configs', str(self.total_configs)])
        
        # Use sophisticated time estimation instead of simple averaging
        estimated_total_time = self.get_adaptive_time_estimate(config)
        estimated_fold_time = estimated_total_time / config.get('k_folds', 5)
        cmd.extend(['--estimated_fold_time', str(estimated_fold_time)])
        
        try:
            # Ensure UTF-8 environment for subprocess
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSFSENCODING'] = '0'  # Force UTF-8 on Windows
            
            if verbose:
                # Show output in real-time
                result = subprocess.run(cmd, timeout=7200, cwd=str(Path(__file__).parent), env=env)
                # For verbose mode, we can't easily extract results, so return a dummy result
                if result.returncode == 0:
                    print(f"Configuration {config_id} completed successfully")
                    return {
                        'config_id': config_id,
                        'timestamp': datetime.datetime.now().isoformat(),
                        **config,
                        'mean_val_accuracy': 0.0,  # Will need manual checking
                        'note': 'Verbose mode - check experiment logs for results'
                    }
                else:
                    print(f"Configuration {config_id} failed (exit code {result.returncode})")
                    print("Error: Check terminal output above for details")
                    self.failed_configs.append({
                        'config_id': config_id,
                        'config': config,
                        'error': f'Process failed with exit code {result.returncode} in verbose mode'
                    })
                    return None
            elif concise:
                # Show concise output in real-time but also capture for extraction
                import subprocess
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                         text=True, bufsize=1, universal_newlines=True,
                                         encoding='utf-8', errors='replace',
                                         env=env, cwd=str(Path(__file__).parent))
                
                output_lines = []
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(output.rstrip())  # Show in real-time
                        output_lines.append(output)
                        
                full_output = ''.join(output_lines)
                result = type('Result', (), {
                    'returncode': process.returncode, 
                    'stdout': full_output,
                    'stderr': ''  # Concise mode combines stdout/stderr, so stderr is empty
                })()
            else:
                # Capture output for result extraction
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=7200,  # 2 hour timeout per experiment
                    encoding='utf-8',
                    errors='replace',  # Replace problematic characters instead of crashing
                    env=env,
                    cwd=str(Path(__file__).parent)
                )
            
            if result.returncode == 0:
                # Calculate run duration
                end_time = time.time()
                duration_minutes = (end_time - start_time) / 60
                completion_time = datetime.datetime.now().strftime("%H:%M:%S")
                
                # Extract results first to get accuracy
                extracted_result = self.extract_results_from_output(result.stdout, config, config_id)
                val_acc = extracted_result.get('mean_val_accuracy', 0.0) if extracted_result else 0.0
                
                # Update timing database with actual results
                self.update_timing_database(config, duration_minutes)
                
                if concise:
                    print(f"\nConfiguration {config_id}/{self.total_configs} - COMPLETED | "
                          f"Val Acc: {val_acc:.4f} | Completed: {completion_time} | Duration: {duration_minutes:.1f}m")
                else:
                    print(f"Configuration {config_id} completed successfully")
                    print(f"Validation Accuracy: {val_acc:.4f}")
                    print(f"Completion time: {completion_time}, Duration: {duration_minutes:.1f} minutes")
                    
                return extracted_result
            else:
                print(f"Configuration {config_id} failed")
                # Provide meaningful error message, handling empty stderr
                error_msg = result.stderr.strip() if result.stderr and result.stderr.strip() else "No error details available"
                if concise and not error_msg.strip():
                    # In concise mode, stderr might be empty because output is combined
                    # Check the end of stdout for error information
                    stdout_lines = result.stdout.strip().split('\n') if result.stdout else []
                    last_few_lines = '\n'.join(stdout_lines[-5:]) if len(stdout_lines) > 0 else "No output captured"
                    error_msg = f"Process failed (exit code {result.returncode}). Last output:\n{last_few_lines}"
                else:
                    error_msg = f"Process failed (exit code {result.returncode}). {error_msg}"
                
                print(f"Error: {error_msg}")
                self.failed_configs.append({
                    'config_id': config_id,
                    'config': config,
                    'error': error_msg
                })
                return None
                
        except subprocess.TimeoutExpired:
            print(f"Configuration {config_id} timed out")
            self.failed_configs.append({
                'config_id': config_id,
                'config': config,
                'error': 'Timeout after 2 hours'
            })
            return None
        except Exception as e:
            print(f"Configuration {config_id} crashed: {e}")
            self.failed_configs.append({
                'config_id': config_id,
                'config': config,
                'error': str(e)
            })
            return None
    
    def _execute_training_subprocess_enhanced(self, config, config_number_in_run, config_file_path, version, start_time, verbose, concise):
        """Execute the training subprocess with enhanced traceability."""
        cmd = [
            sys.executable, 
            str(Path(__file__).parent / 'PD_signal_classifier_v3.py'), 
            '--config', config_file_path,
            '--source', 'hyperopt',
            '--hyperopt_run_id', self.run_info['run_id'],
            '--config_file', config_file_path,
            '--config_number_in_run', str(config_number_in_run)
        ]
        
        # Add concise flag if needed
        if concise:
            cmd.append('--concise')
        
        try:
            # Set up environment for subprocess
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSFSENCODING'] = '0'
            
            if verbose:
                # Run with real-time output for verbose mode
                result = subprocess.run(
                    cmd, 
                    timeout=7200,  # 2 hour timeout
                    encoding='utf-8',
                    errors='replace',
                    env=env,
                    cwd=str(Path(__file__).parent)
                )
                
                if result.returncode == 0:
                    if not self.concise:
                        print(f"Config {config_number_in_run} completed successfully")
                    
                    # Extract results from experiment log (most reliable)
                    experiment_result = self.extract_from_experiment_log_enhanced(version, config_number_in_run, config_file_path)
                    if experiment_result:
                        return experiment_result
                    else:
                        return {
                            'config_number_in_run': config_number_in_run,
                            'timestamp': datetime.datetime.now().isoformat(),
                            'version': version,
                            'config_file': config_file_path,
                            'note': 'Verbose mode - check experiment logs for results'
                        }
                else:
                    print(f"Config {config_number_in_run} failed (exit code {result.returncode})")
                    print("Error: Check terminal output above for details")
                    self.failed_configs.append({
                        'config_number_in_run': config_number_in_run,
                        'config': config,
                        'error': f'Process failed with exit code {result.returncode} in verbose mode'
                    })
                    return None
            else:
                # Capture output for non-verbose mode  
                result = subprocess.run(
                    cmd, 
                    capture_output=True,
                    text=True,
                    timeout=7200,  # 2 hour timeout
                    encoding='utf-8',
                    errors='replace',
                    env=env,
                    cwd=str(Path(__file__).parent)
                )
                
                if result.returncode == 0:
                    if not self.concise:
                        print(f"Config {config_number_in_run} completed successfully")
                    
                    # Extract results from experiment log (most reliable)  
                    experiment_result = self.extract_from_experiment_log_enhanced(version, config_number_in_run, config_file_path)
                    if experiment_result:
                        return experiment_result
                    else:
                        # Fallback to parsing output
                        return self.extract_results_from_output_enhanced(result.stdout, config, config_number_in_run, version, config_file_path)
                else:
                    print(f"Config {config_number_in_run} failed:")
                    print(f"Exit code: {result.returncode}")
                    if result.stderr:
                        print(f"Error output: {result.stderr}")
                    
                    self.failed_configs.append({
                        'config_number_in_run': config_number_in_run,
                        'config': config,
                        'error': f'Exit code {result.returncode}: {result.stderr}'
                    })
                    return None
                    
        except subprocess.TimeoutExpired:
            print(f"Config {config_number_in_run} timed out")
            self.failed_configs.append({
                'config_number_in_run': config_number_in_run,
                'config': config,
                'error': 'Timeout after 2 hours'
            })
            return None
        except Exception as e:
            print(f"Config {config_number_in_run} crashed: {e}")
            self.failed_configs.append({
                'config_number_in_run': config_number_in_run,
                'config': config,
                'error': str(e)
            })
            return None
    
    def extract_from_experiment_log_enhanced(self, version, config_number_in_run, config_file_path):
        """Extract results from experiment log with enhanced traceability."""
        try:
            df = self._get_experiment_log_dataframe()
            if df is None or df.empty:
                return None
            
            # Find entry by version (most reliable identifier)
            matching_entries = df[df['version'] == version]
            
            if matching_entries.empty:
                return None
            
            latest_entry = matching_entries.iloc[-1]
            
            # Extract the key results with enhanced traceability
            result = {
                'config_number_in_run': config_number_in_run,
                'timestamp': datetime.datetime.now().isoformat(),
                'version': version,
                'config_file': config_file_path,
                'hyperopt_run_id': self.run_info['run_id'],
                'mean_val_accuracy': latest_entry.get('mean_val_accuracy', 0.0),
                'std_val_accuracy': latest_entry.get('std_val_accuracy', 0.0),
                'best_val_accuracy': latest_entry.get('best_fold_accuracy', 0.0),
                'training_time_minutes': latest_entry.get('total_training_time_minutes', 0.0),
                'learning_rate': latest_entry.get('learning_rate', 0.0),
                'batch_size': latest_entry.get('batch_size', 0),
                'epochs': latest_entry.get('epochs', 0),
                'k_folds': latest_entry.get('k_folds', 0),
                'conv_filters': str(latest_entry.get('conv_filters', '[]')),
                'dense_units': str(latest_entry.get('dense_units', '[]')),
                'conv_dropout': str(latest_entry.get('dropout_rates', '[0.0]')).split(',')[0].strip('['),
                'dense_dropout': str(latest_entry.get('dropout_rates', '[0.0, [0.0]]')),
                'l2_regularization': latest_entry.get('l2_reg', 0.0)
            }
            
            return result
            
        except Exception as e:
            print(f"Failed to extract enhanced results from experiment log: {e}")
            return None
    
    def extract_results_from_output_enhanced(self, output, config, config_number_in_run, version, config_file_path):
        """Extract key results from training output with enhanced traceability (fallback method)."""
        result = {
            'config_number_in_run': config_number_in_run,
            'timestamp': datetime.datetime.now().isoformat(),
            'version': version,
            'config_file': config_file_path,
            'hyperopt_run_id': self.run_info['run_id'],
        }
        
        # Try to extract key metrics from output (basic patterns)
        try:
            import re
            
            # Look for validation accuracy
            val_acc_match = re.search(r'val_accuracy[:\s]+(\d+\.\d+)', output, re.IGNORECASE)
            if val_acc_match:
                result['mean_val_accuracy'] = float(val_acc_match.group(1))
            
            # Look for training time 
            time_match = re.search(r'training.*time[:\s]+(\d+\.\d+)', output, re.IGNORECASE)
            if time_match:
                result['training_time_minutes'] = float(time_match.group(1))
            
            # Add config parameters
            result.update({
                'learning_rate': config.get('learning_rate', 0.0),
                'batch_size': config.get('batch_size', 0),
                'epochs': config.get('epochs', 0),
                'k_folds': config.get('k_folds', 0),
                'conv_filters': str(config.get('conv_filters', [])),
                'dense_units': str(config.get('dense_units', [])),
                'conv_dropout': config.get('conv_dropout', 0.0),
                'dense_dropout': str(config.get('dense_dropout', [])),
                'l2_regularization': config.get('l2_regularization', 0.0)
            })
            
        except Exception as e:
            print(f"Warning: Could not parse training output: {e}")
            
        return result
    
    def extract_results_from_output(self, output, config, config_id):
        """Extract key results from training output and experiment log."""
        # Try to read from the experiment log CSV first (most reliable)
        result = self.extract_from_experiment_log(config_id)
        if result:
            return result
        
        # Fallback: parse training output (less reliable)
        
        result = {
            'config_id': config_id,
            'timestamp': datetime.datetime.now().isoformat(),
        }
        
        # Add all config parameters
        result.update(config)
        
        # Try to extract validation accuracy and fold times from output
        lines = output.split('\n')
        fold_times = []  # Track individual fold times
        
        for line in lines:
            # Extract fold completion times (format: "| 45.2m")
            if 'Fold' in line and 'completed' in line and 'm |' in line:
                try:
                    # Extract timing from "...| 45.2m" or "...| 45.2m | ETA: ..."
                    parts = line.split('|')
                    for part in parts:
                        part = part.strip()
                        if part.endswith('m'):
                            time_str = part.replace('m', '')
                            try:
                                fold_time = float(time_str)
                                fold_times.append(fold_time)
                                break
                            except ValueError:
                                pass
                except (ValueError, IndexError, AttributeError):
                    pass
            
            elif 'Mean Validation Accuracy:' in line:
                try:
                    # Extract "Mean Validation Accuracy: 0.8532 ± 0.0234"
                    parts = line.split(':')[1].strip().split('±')
                    mean_acc = float(parts[0].strip())
                    std_acc = float(parts[1].strip()) if len(parts) > 1 else 0.0
                    result['mean_val_accuracy'] = mean_acc
                    result['std_val_accuracy'] = std_acc
                except (ValueError, IndexError, AttributeError):
                    pass
            
            elif 'Best Fold Accuracy:' in line:
                try:
                    # Extract "Best Fold Accuracy: 0.8732 (Fold 3)"
                    acc_part = line.split(':')[1].strip().split('(')[0].strip()
                    result['best_val_accuracy'] = float(acc_part)
                except (ValueError, IndexError, AttributeError):
                    pass
            
            elif 'Training Time:' in line:
                try:
                    # Extract "Training Time: 45.2 minutes"
                    time_part = line.split(':')[1].strip().split()[0]
                    result['training_time_minutes'] = float(time_part)
                except (ValueError, IndexError, AttributeError):
                    pass
        
        # Note: fold_times tracking removed - using sophisticated prediction instead
        
        return result
    
    def analyze_configuration_space(self, configs):
        """Analyze the configuration space and return summary statistics."""
        if not configs:
            return {}
        
        analysis = {
            'total_configs': len(configs),
            'learning_rates': sorted(set(config['learning_rate'] for config in configs)),
            'batch_sizes': sorted(set(config['batch_size'] for config in configs)),
            'epochs': sorted(set(config['epochs'] for config in configs)),
            'k_folds': sorted(set(config['k_folds'] for config in configs)),
            'conv_dropouts': sorted(set(config['conv_dropout'] for config in configs)),
            'dense_dropouts': sorted(list(set(tuple(config['dense_dropout']) for config in configs))),
            'l2_regs': sorted(set(config['l2_regularization'] for config in configs)),
            'architectures': []
        }
        
        # Analyze unique architectures
        arch_set = set()
        for config in configs:
            arch = (tuple(config['conv_filters']), tuple(config['dense_units']))
            arch_set.add(arch)
        analysis['architectures'] = sorted(list(arch_set))
        
        # Calculate total training time estimate using adaptive system
        total_time_minutes = 0
        for config in configs:
            # Use adaptive time estimation if available, otherwise fall back to heuristic
            adaptive_time = self.get_adaptive_time_estimate(config)
            total_time_minutes += adaptive_time
        
        analysis['estimated_total_time_hours'] = total_time_minutes / 60
        
        return analysis
    
    def display_start_screen(self, mode, configs, resume=False, completed_ids=None, skip_from_deduplication=None, configs_to_run_count=None):
        """Display comprehensive start screen with configuration analysis."""
        analysis = self.analyze_configuration_space(configs)
        
        # Use provided counts if available, otherwise calculate
        if completed_ids is None:
            completed_ids = set()
        if skip_from_deduplication is None:
            skip_from_deduplication = set()
            
        # Calculate which configs will actually run for better preview
        configs_to_execute = []
        execution_order = 0
        for i, config in enumerate(configs, 1):
            config_index = i - 1  # Convert to 0-based index for deduplication check
            if config_index not in skip_from_deduplication and i not in completed_ids:
                execution_order += 1
                configs_to_execute.append((execution_order, i, config))
        
        # Use accurate count if provided
        if configs_to_run_count is not None:
            remaining_configs = configs_to_run_count
        else:
            remaining_configs = analysis['total_configs'] - len(completed_ids)
        
        print("\n" + "="*100)
        print("HYPERPARAMETER OPTIMIZATION - CONFIGURATION SUMMARY")
        print("="*100)
        
        print(f"\nOPTIMIZATION DETAILS:")
        print(f"   Mode: {mode.upper()}")
        print(f"   Total Configurations: {analysis['total_configs']}")
        
        # Show deduplication and completion info more accurately
        if skip_from_deduplication:
            print(f"   Skipped by Deduplication: {len(skip_from_deduplication)}")
        if resume and completed_ids:
            completed_also_duplicates = sum(1 for i in completed_ids if (i-1) in skip_from_deduplication)
            if completed_also_duplicates > 0:
                print(f"   Already Completed (included in dedup): {len(completed_ids)}")
            else:
                print(f"   Already Completed: {len(completed_ids)}")
        
        print(f"   Configurations to Run: {remaining_configs}")
        print(f"   Resume Mode: {'Yes' if resume else 'No'}")
        print(f"   Output Directory: {self.output_root}")
        
        print(f"\nTIME ESTIMATES:")
        if resume and completed_ids:
            # Calculate time for remaining configs only
            avg_time_per_config = analysis['estimated_total_time_hours'] / analysis['total_configs']
            remaining_time_hours = avg_time_per_config * remaining_configs
            print(f"   Estimated Remaining Time: {remaining_time_hours:.1f} hours")
            print(f"   Average per Config: {avg_time_per_config*60:.1f} minutes")
            print(f"   (Original total estimate was {analysis['estimated_total_time_hours']:.1f} hours for {analysis['total_configs']} configs)")
        else:
            print(f"   Estimated Total Time: {analysis['estimated_total_time_hours']:.1f} hours")
            print(f"   Average per Config: {analysis['estimated_total_time_hours']*60/analysis['total_configs']:.1f} minutes")
        
        # Show timing database status
        num_records = len(self.timing_data['timing_records'])
        if num_records > 0:
            print(f"   Timing Database: {num_records} historical records (adaptive estimation)")
        else:
            print(f"   Timing Database: No historical data (using conservative estimates)")
        
        print(f"\nHYPERPARAMETER RANGES:")
        print(f"   Learning Rates: {analysis['learning_rates']}")
        print(f"   Batch Sizes: {analysis['batch_sizes']}")
        print(f"   Epochs per Config: {analysis['epochs']}")
        print(f"   K-Folds per Config: {analysis['k_folds']}")
        print(f"   Conv Dropout Rates: {analysis['conv_dropouts']}")
        print(f"   Dense Dropout Rates: {[list(dropout) for dropout in analysis['dense_dropouts']]}")
        print(f"   L2 Regularization: {analysis['l2_regs']}")
        
        print(f"\nNETWORK ARCHITECTURES:")
        for i, (conv_arch, dense_arch) in enumerate(analysis['architectures'], 1):
            print(f"   Architecture {i}: Conv{list(conv_arch)} + Dense{list(dense_arch)}")
        
        # Show configurations that will actually be executed
        num_to_show = min(3, len(configs_to_execute))
        if len(configs_to_execute) <= 3:
            print(f"\nCONFIGURATIONS TO EXECUTE:")
        else:
            print(f"\nCONFIGURATIONS TO EXECUTE (first {num_to_show}):")
            
        for execution_order, original_index, config in configs_to_execute[:num_to_show]:
            print(f"   Config {execution_order} (Original #{original_index}):")
            print(f"      LR: {config['learning_rate']}, BS: {config['batch_size']}, "
                  f"Epochs: {config['epochs']}, Folds: {config['k_folds']}")
            print(f"      Conv: {config['conv_filters']}, Dense: {config['dense_units']}")
            dense_dropout_str = str(config['dense_dropout']).replace(' ', '') if isinstance(config['dense_dropout'], list) else config['dense_dropout']
            print(f"      Dropout: {config['conv_dropout']}/{dense_dropout_str}, "
                  f"L2: {config['l2_regularization']}")
        
        if len(configs_to_execute) > num_to_show:
            print(f"   ... and {len(configs_to_execute) - num_to_show} more configurations to execute")
        
        if skip_from_deduplication or completed_ids:
            skipped_total = len(skip_from_deduplication) + len(completed_ids)
            print(f"   (Skipping {skipped_total} configs: {len(skip_from_deduplication)} duplicates + {len(completed_ids)} completed)")
        
        print("\n" + "="*100)
        
        return analysis
    
    def create_run_directory(self, mode, resume=False):
        """Create a unique directory for this optimization run with enhanced tracking."""
        if resume:
            # Find the most recent directory for this mode
            pattern = f"{mode}_run_*"
            existing_dirs = [d for d in os.listdir(self.base_output_root) 
                           if (self.base_output_root / d).is_dir() and d.startswith(f"{mode}_run_")]
            
            if existing_dirs:
                # Sort by creation time and use the most recent
                existing_dirs.sort(key=lambda x: (self.base_output_root / x).stat().st_ctime, reverse=True)
                run_dir = existing_dirs[0]
                print(f"Resuming from existing directory: {run_dir}")
                # Load existing run info
                run_info_file = self.base_output_root / run_dir / "run_info.json"
                if run_info_file.exists():
                    with open(run_info_file) as f:
                        self.run_info = json.load(f)
                else:
                    # Create run info for legacy runs
                    self.run_info = {
                        'run_id': run_dir,
                        'mode': mode,
                        'start_time': 'legacy_run',
                        'total_configs_executed': 0,
                        'is_legacy': True
                    }
            else:
                print(f"No existing {mode} run found to resume from. Creating new run.")
                resume = False
        
        if not resume:
            # Create new run directory with timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = f"{mode}_run_{timestamp}"
            # Initialize run info for new runs
            self.run_info = {
                'run_id': run_dir,
                'mode': mode,
                'start_time': datetime.datetime.now().isoformat(),
                'total_configs_executed': 0,
                'config_files': {},  # Maps config_number -> config_file_path
                'version_mapping': {}  # Maps version -> config_number
            }
        
        self.output_root = self.base_output_root / run_dir
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Setup progress tracking files
        self.progress_file = self.output_root / "tuning_progress.json"
        self.results_file = self.output_root / "hyperparameter_results.csv"
        
        # Save run info
        run_info_file = self.output_root / "run_info.json"
        with open(run_info_file, 'w') as f:
            json.dump(self.run_info, f, indent=2)
        
        print(f"Run directory: {self.output_root.as_posix()}")
        print(f"Run ID: {self.run_info['run_id']}")
        return str(self.output_root)
    
    def ensure_experiment_log_columns(self):
        """Ensure experiment log has the new columns for enhanced traceability."""
        log_file = get_experiment_log_path()
        new_columns = ['hyperopt_run_id', 'config_file', 'config_number_in_run']
        
        if not Path(log_file).exists():
            return  # Will be created with proper schema later
        
        try:
            df = pd.read_csv(log_file, encoding='utf-8')
            columns_added = False
            
            for col in new_columns:
                if col not in df.columns:
                    if col == 'hyperopt_run_id':
                        df[col] = 'legacy_run'  # Default for existing entries
                    elif col == 'config_file':
                        df[col] = ''  # Will be populated retroactively
                    elif col == 'config_number_in_run':
                        df[col] = -1  # -1 indicates legacy/unknown
                    columns_added = True
            
            if columns_added:
                # Save updated DataFrame
                df.to_csv(log_file, index=False, encoding='utf-8')
                print(f"Enhanced experiment log with new traceability columns")
        
        except Exception as e:
            print(f"Warning: Could not update experiment log schema: {e}")
    
    def get_user_confirmation(self, analysis, resume=False, completed_ids=None, configs_to_run_count=None):
        """Get user confirmation to proceed with optimization."""
        while True:
            print(f"\nCONFIRMATION REQUIRED:")
            
            if resume and completed_ids:
                # Use accurate count if provided
                if configs_to_run_count is not None:
                    remaining_configs = configs_to_run_count
                else:
                    remaining_configs = analysis['total_configs'] - len(completed_ids)
                avg_time_per_config = analysis['estimated_total_time_hours'] / analysis['total_configs']
                remaining_time_hours = avg_time_per_config * remaining_configs
                print(f"   This will run {remaining_configs} remaining configurations")
                print(f"   Estimated remaining time: {remaining_time_hours:.1f} hours")
            else:
                # Use accurate count if provided
                if configs_to_run_count is not None:
                    configs_to_run = configs_to_run_count
                    avg_time_per_config = analysis['estimated_total_time_hours'] / analysis['total_configs']
                    estimated_time_hours = avg_time_per_config * configs_to_run
                else:
                    configs_to_run = analysis['total_configs']
                    estimated_time_hours = analysis['estimated_total_time_hours']
                
                print(f"   This will run {configs_to_run} configurations")
                print(f"   Estimated time: {estimated_time_hours:.1f} hours")
                
            print(f"   All results will be saved to: {Path(self.output_root).as_posix()}")
            
            response = input("\n   Continue with hyperparameter optimization? [y/n]: ").strip().lower()
            
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no', '']:
                print("\nHyperparameter optimization cancelled.")
                return False
            else:
                print("   Please enter 'y' for yes or 'n' for no.")
    
    def run_optimization(self, mode='smart', max_configs=None, resume=False, verbose=False, concise=False, skip_deduplication=False, search_radius=1, grid_search=False, ignore_params=None, max_grid_size=100):
        """
        Run hyperparameter optimization.
        
        Args:
            mode: Optimization mode options:
                - test: 2 configs for testing (2 folds, 2 epochs)
                - quick: ~15-20 configs, basic parameter exploration
                - medium: ~25-30 configs, systematic OFAT approach
                - smart: Adaptive search around best previous results (all parameters)
                - smart-training: Focus on learning dynamics (learning_rate, batch_size, epochs, early_stopping_patience)
                - smart-architecture: Focus on model structure (conv_filters, dense_units)
                - smart-regularization: Focus on overfitting control (dropout, l2_reg, class_weights)
                - smart-augmentation: Focus on data augmentation parameters
                - full: Exhaustive grid search (hundreds of configs)
                - augmentation: Specialized augmentation parameter optimization
                - anova: ~80 configs to improve ANOVA analysis quality (focus on high-uncertainty parameters)
            max_configs: Maximum number of configs to test (None for all)
            resume: Whether to resume from previous run
            skip_deduplication: If True, don't skip previously tested configurations
            search_radius: For smart mode, number of neighboring values to test around best config (±1 or ±2)
        """
        # Create run-specific directory first
        self.create_run_directory(mode, resume)
        
        # Ensure experiment log has new columns
        self.ensure_experiment_log_columns()
        
        # Generate configurations
        if mode == 'smart':
            configs = self.generate_smart_configs(search_radius=search_radius, grid_search=grid_search, ignore_params=ignore_params or [], max_grid_size=max_grid_size, mode='all')
        elif mode in ['smart-training', 'smart-architecture', 'smart-regularization', 'smart-augmentation']:
            # New focused smart modes
            focus_mode = mode.split('-')[1]  # Extract 'training', 'architecture', etc.
            configs = self.generate_smart_configs(search_radius=search_radius, grid_search=grid_search, ignore_params=ignore_params or [], max_grid_size=max_grid_size, mode=focus_mode)
        elif mode == 'augmentation':
            # Augmentation mode supports grid_search parameter
            configs = self.generate_augmentation_configs(use_grid_search=grid_search)
        else:
            config_generators = {
                'test': self.generate_test_configs,
                'quick': self.generate_quick_configs,
                'medium': self.generate_medium_configs,
                'full': self.generate_full_configs,
                'anova': self.generate_anova_improvement_configs
            }
            
            if mode not in config_generators:
                available_modes = list(config_generators.keys()) + ['smart', 'smart-training', 'smart-architecture', 'smart-regularization', 'smart-augmentation', 'augmentation']
                raise ValueError(f"Unknown mode: {mode}. Available modes: {available_modes}")
            
            configs = config_generators[mode]()
        
        # Determine which configs to skip from deduplication
        skip_from_deduplication = set()
        if not skip_deduplication:
            skip_from_deduplication = self._get_duplicate_indices(configs)
            if skip_from_deduplication and not self.concise:
                print(f"Smart deduplication: Will skip {len(skip_from_deduplication)}/{len(configs)} configurations (already tested)")
        
        # Load previous progress if resuming
        completed_ids = set()
        if resume and self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                completed_ids = set(progress.get('completed_configs', []))
        
        # Calculate available configs (accounting for mixed indexing)
        available_configs = 0
        for i in range(len(configs)):
            config_index_0based = i  # 0-based for deduplication 
            config_index_1based = i + 1  # 1-based for resume progress
            
            if config_index_0based not in skip_from_deduplication and config_index_1based not in completed_ids:
                available_configs += 1
        
        # Apply max_configs limit after determining skips
        if max_configs and max_configs < available_configs:
            if not self.concise:
                print(f"Limiting to first {max_configs} configurations (from {available_configs} available)")
            
            # Keep only the first max_configs available configs
            kept_count = 0
            additional_skips = set()  # Store 0-based indices to skip due to max_configs
            
            for i in range(len(configs)):
                config_index_0based = i
                config_index_1based = i + 1
                
                # If this config would normally run
                if config_index_0based not in skip_from_deduplication and config_index_1based not in completed_ids:
                    kept_count += 1
                    if kept_count > max_configs:
                        additional_skips.add(config_index_0based)
            
            # Add additional skips to deduplication skips
            skip_from_deduplication.update(additional_skips)
        
        # Calculate actual configs that will run after all filtering
        configs_to_run_count = 0
        for i in range(len(configs)):
            config_index_0based = i
            config_index_1based = i + 1
            if config_index_0based not in skip_from_deduplication and config_index_1based not in completed_ids:
                configs_to_run_count += 1
        
        # Display start screen and get confirmation (with accurate counts)
        analysis = self.display_start_screen(mode, configs, resume, completed_ids, 
                                           skip_from_deduplication, configs_to_run_count)
        
        if not self.get_user_confirmation(analysis, resume, completed_ids, configs_to_run_count):
            return
        
        if not self.concise:
            print(f"\nStarting Hyperparameter Optimization")
            print(f"Mode: {mode}")
        
        # Store total configs for concise mode
        self.total_configs = len(configs)
        
        if completed_ids and not self.concise:
            print(f"Resuming: {len(completed_ids)} configs already completed")
        
        # Run experiments with sequential numbering within run
        start_time = time.time()
        config_number_in_run = 0  # Sequential counter for executed configs in this run
        
        for i, config in enumerate(configs, 1):
            # Skip configs that were already tested or completed
            config_index = i - 1  # Convert to 0-based index for deduplication check
            if config_index in skip_from_deduplication or i in completed_ids:
                if not self.concise:
                    skip_reason = "already completed" if (i in completed_ids) else "already tested"
                    print(f"Skipping configuration {i} ({skip_reason})")
                continue
            
            # Increment sequential counter for configs actually being run in this run
            config_number_in_run += 1
            
            # Format dense dropout for clean display
            dense_dropout_str = str(config['dense_dropout']).replace(' ', '') if isinstance(config['dense_dropout'], list) else config['dense_dropout']
            
            # Get the actual version that will be used (accounts for previous experiments)
            current_version_num = get_next_version_from_log()
            actual_version = format_version(current_version_num)
            
            print(f"\n{'='*90}")
            print(f"Config {config_number_in_run}/{configs_to_run_count} | Version: {actual_version}")
            print(f"Parameters: LR={config['learning_rate']} | BS={config['batch_size']} | Dropout={config['conv_dropout']}/{dense_dropout_str}")
            print(f"Run: {self.run_info['run_id']} | Config File: config_{config_number_in_run:03d}.json")
            print(f"{'='*90}")
            
            # Save config with sequential numbering and run the experiment
            config_file_path = self.save_config_to_file(config, config_number_in_run)
            
            # Update run info with version mapping
            self.run_info['version_mapping'][actual_version] = config_number_in_run
            
            result = self.run_training_experiment_enhanced(config, config_number_in_run, config_file_path, actual_version, verbose=verbose, concise=concise)
            
            if result:
                # Enhanced result with traceability info
                result['config_id'] = i  # Keep original for resume compatibility
                result['config_number_in_run'] = config_number_in_run
                result['config_file'] = config_file_path
                result['hyperopt_run_id'] = self.run_info['run_id']
                result['execution_order'] = config_number_in_run  # Same as config_number_in_run now
                self.results.append(result)
                self.save_intermediate_results()
                # Only mark as completed if successful (use original index for resume tracking)
                completed_ids.add(i)
                self.save_progress(list(completed_ids), i, len(configs))
            else:
                # Config failed, don't mark as completed but still save progress
                self.save_progress(list(completed_ids), i, len(configs))
            
            # Print progress
            elapsed = time.time() - start_time
            avg_time = elapsed / config_number_in_run  # Average time per config actually run
            remaining = configs_to_run_count - config_number_in_run
            eta = avg_time * remaining / 3600  # in hours
            
            if not self.concise:
                print(f"\nProgress: {config_number_in_run}/{configs_to_run_count} ({config_number_in_run/configs_to_run_count*100:.1f}%)")
                print(f"Average time per config (all folds): {avg_time/60:.1f} minutes")
                print(f"Estimated time remaining: {eta:.1f} hours")
        
        # Calculate total execution time
        total_time_minutes = (time.time() - start_time) / 60
        
        # Final results
        self.analyze_and_save_final_results(total_time_minutes)
        
        print(f"\nHyperparameter optimization complete!")
        print(f"Successful runs: {len(self.results)}")
        print(f"Failed runs: {len(self.failed_configs)}")
        print(f"Total execution time: {total_time_minutes:.1f} minutes ({total_time_minutes/60:.1f} hours)")
        print(f"Results saved to: {self.results_file.as_posix()}")
    
    def save_progress(self, completed_configs, current, total):
        """Save progress to resume later."""
        progress = {
            'completed_configs': completed_configs,
            'current': current,
            'total': total,
            'timestamp': datetime.datetime.now().isoformat()
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def _format_dropout_value(self, dropout_value):
        """Format dropout value for display, handling various data types."""
        if isinstance(dropout_value, str):
            try:
                dropout_value = ast.literal_eval(dropout_value)
            except (ValueError, SyntaxError, TypeError):
                # If parsing fails, use the string as-is
                pass
        return str(dropout_value).replace(' ', '')
    
    def save_intermediate_results(self):
        """Save current results to CSV."""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(self.results_file, index=False, encoding='utf-8')
    
    def _get_experiment_log_dataframe(self):
        """Get experiment log DataFrame with caching to avoid repeated file reads."""
        log_file = str(get_experiment_log_path())
        
        if not Path(log_file).exists():
            return None
        
        # Check if cache is valid by comparing file modification time
        current_modified = Path(log_file).stat().st_mtime
        if (self._experiment_log_cache is None or 
            current_modified > self._experiment_log_last_modified):
            # Cache is invalid, reload
            try:
                self._experiment_log_cache = pd.read_csv(log_file, encoding='utf-8')
                self._experiment_log_last_modified = current_modified
            except Exception as e:
                print(f"Failed to read experiment log: {e}")
                return None
        
        return self._experiment_log_cache
    
    def extract_from_experiment_log(self, config_id):
        """Extract results from the experiment log CSV (most reliable source)."""
        try:
            df = self._get_experiment_log_dataframe()
            if df is None or df.empty:
                return None
            
            # Filter for hyperopt source entries (recent runs from hyperparameter optimization)
            hyperopt_entries = df[df['source'] == 'hyperopt'].copy()
            
            if hyperopt_entries.empty:
                return None
            
            # Find the entry for the specific config_id by matching version
            target_version = format_version(config_id)
            matching_entries = hyperopt_entries[hyperopt_entries['version'] == target_version]
            
            if matching_entries.empty:
                # Fallback to the most recent entry if version doesn't match
                latest_entry = hyperopt_entries.iloc[-1]
            else:
                latest_entry = matching_entries.iloc[-1]
            
            # Extract the key results
            result = {
                'config_id': config_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'version': latest_entry.get('version', format_version(config_id)),
                'mean_val_accuracy': latest_entry.get('mean_val_accuracy', 0.0),
                'std_val_accuracy': latest_entry.get('std_val_accuracy', 0.0),
                'best_val_accuracy': latest_entry.get('best_fold_accuracy', 0.0),
                'training_time_minutes': latest_entry.get('total_training_time_minutes', 0.0),
                'learning_rate': latest_entry.get('learning_rate', 0.0),
                'batch_size': latest_entry.get('batch_size', 0),
                'epochs': latest_entry.get('epochs', 0),
                'k_folds': latest_entry.get('k_folds', 0),
                'conv_filters': self._safe_parse_list(latest_entry.get('conv_filters', '[]')),
                'dense_units': self._safe_parse_list(latest_entry.get('dense_units', '[]')),
                'conv_dropout': latest_entry.get('conv_dropout', 0.0),
                'dense_dropout': self._safe_parse_list(latest_entry.get('dropout_rates', '[]')),
                'l2_regularization': latest_entry.get('l2_reg', 0.0),
            }
            
            return result
            
        except Exception as e:
            print(f"Failed to extract from experiment log: {e}")
            return None
    
    def analyze_and_save_final_results(self, total_time_minutes=None):
        """Analyze results and save final report."""
        if not self.results:
            print("No successful results to analyze")
            return
        
        df = pd.DataFrame(self.results)
        
        # Sort by best fold performance (not mean performance)
        if 'best_val_accuracy' in df.columns:
            df = df.sort_values('best_val_accuracy', ascending=False)
        elif 'mean_val_accuracy' in df.columns:
            df = df.sort_values('mean_val_accuracy', ascending=False)
        
        # Save full results
        df.to_csv(self.results_file, index=False, encoding='utf-8')
        
        # Save top 10 results
        top_10_file = self.output_root / "top_10_results.csv"
        df.head(10).to_csv(top_10_file, index=False, encoding='utf-8')
        
        # Print summary
        print(f"\nRESULTS SUMMARY")
        print("="*100)
        if total_time_minutes:
            print(f"Total Execution Time: {total_time_minutes:.1f} minutes ({total_time_minutes/60:.1f} hours)")
            print(f"Average Time per Config: {total_time_minutes/len(self.results):.1f} minutes")
            print("="*50)
        if 'mean_val_accuracy' in df.columns:
            best_result = df.iloc[0]
            print(f"Best Accuracy: {best_result['mean_val_accuracy']:.4f}")
            print(f"   Config ID: {best_result['config_id']}")
            print(f"   Learning Rate: {best_result['learning_rate']}")
            print(f"   Batch Size: {best_result['batch_size']}")
            best_dense_dropout_str = self._format_dropout_value(best_result['dense_dropout'])
            print(f"   Dropout: {best_result['conv_dropout']}/{best_dense_dropout_str}")
            
            print(f"\nTop 5 Results:")
            for i, (_, row) in enumerate(df.head(5).iterrows()):
                # Get reliable data from experiment log for this config
                config_id = row['config_id']
                log_result = self.extract_from_experiment_log(config_id)
                
                if log_result:
                    # Use data from experiment log (more reliable)
                    accuracy = log_result['mean_val_accuracy']
                    learning_rate = log_result['learning_rate']
                    batch_size = log_result['batch_size']
                    version = log_result.get('version', format_version(config_id))
                else:
                    # Fallback to DataFrame data if log extraction fails
                    accuracy = row['mean_val_accuracy']
                    learning_rate = row['learning_rate']
                    batch_size = row['batch_size']
                    version = format_version(config_id)
                
                print(f"   {i+1}. Config {config_id} ({version}): {accuracy:.4f} "
                      f"(LR={learning_rate}, BS={batch_size})")
        
        # Save failed configs if any
        if self.failed_configs:
            failed_file = self.output_root / "failed_configs.json"
            with open(failed_file, 'w') as f:
                clean_failed_configs = convert_numpy_types(self.failed_configs)
                json.dump(clean_failed_configs, f, indent=2)
            print(f"\n{len(self.failed_configs)} configs failed - see {failed_file.as_posix()}")

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for PD Signal Classifier')
    parser.add_argument('--mode', choices=['test', 'quick', 'medium', 'smart', 'smart-training', 'smart-architecture', 'smart-regularization', 'smart-augmentation', 'full', 'augmentation', 'anova'], default='smart',
                       help='Optimization mode (default: smart). Smart modes: smart=all parameters, smart-training=learning dynamics, smart-architecture=model structure, smart-regularization=overfitting control, smart-augmentation=data augmentation, anova=ANOVA analysis improvement')
    parser.add_argument('--max_configs', type=int, default=None,
                       help='Maximum number of configurations to test')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (defaults to config HYPEROPT_RESULTS_DIR)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous run')
    parser.add_argument('--verbose', action='store_true',
                       help='Show training output in real-time')
    parser.add_argument('--concise', action='store_true',
                       help='Show concise progress updates (one line per epoch)')
    parser.add_argument('--skip-deduplication', action='store_true',
                       help='Skip smart deduplication - allow retesting previous configurations')
    parser.add_argument('--search-radius', type=int, choices=[1, 2], default=1,
                       help='For smart mode: search radius around best config (±1 or ±2 values per parameter)')
    parser.add_argument('--grid-search', action='store_true',
                       help='For smart/augmentation modes: do full grid search instead of OFAT')
    parser.add_argument('--ignore', nargs='+', default=[], metavar='PARAM',
                       help='For smart mode: parameters to ignore in search, using previous optimum value')
    parser.add_argument('--max-grid-size', type=int, default=100, metavar='N',
                       help='Maximum grid search size before parameter limiting kicks in')
    
    args = parser.parse_args()
    
    # Initialize tuner (output_root will be set when run_optimization is called)
    tuner = HyperparameterTuner(output_root=args.output_dir, verbose=args.verbose, concise=args.concise)
    
    # Run optimization
    tuner.run_optimization(
        mode=args.mode,
        max_configs=args.max_configs,
        resume=args.resume,
        verbose=args.verbose,
        concise=args.concise,
        skip_deduplication=args.skip_deduplication,
        search_radius=args.search_radius,
        grid_search=args.grid_search,
        ignore_params=args.ignore,
        max_grid_size=args.max_grid_size
    )

if __name__ == "__main__":
    main()