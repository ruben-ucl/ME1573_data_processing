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
from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    get_data_dir, get_pd_timing_database_path, get_pd_experiment_log_path,
    get_pd_config_template, PD_HYPEROPT_RESULTS_DIR, normalize_path, format_version,
    convert_numpy_types, get_next_version_from_log,
    # CWT-specific imports
    get_default_cwt_data_dir, get_cwt_experiment_log_path, get_cwt_config_template,
    get_cwt_timing_database_path, CWT_LOGS_DIR, CWT_HYPEROPT_RESULTS_DIR,
    # Consolidated functions
    extract_experiment_result
)

# ========================================================================================
# CLASSIFIER STRATEGY INTERFACE
# ========================================================================================

class ClassifierStrategy(ABC):
    """Abstract base class for classifier-specific hyperparameter optimization strategies."""
    
    @abstractmethod
    def get_parameter_space(self, categories=None, tiers=None):
        """Return the hyperparameter search space for this classifier.
        
        Args:
            categories (list, optional): Filter by parameter categories
            tiers (list, optional): Filter by priority tiers
        """
        pass
    
    @abstractmethod
    def get_config_template(self):
        """Return the default configuration template for this classifier."""
        pass
        
    @abstractmethod
    def get_experiment_log_path(self):
        """Return the path to the experiment log for this classifier."""
        pass
    
    @abstractmethod
    def parse_experiment_log_row(self, row, tuner):
        """Parse a row from the experiment log into a config dict."""
        pass
    
    @abstractmethod
    def get_execution_command(self, config_file_path):
        """Return the command to execute the classifier with given config."""
        pass
    
    @abstractmethod
    def get_config_signature_params(self):
        """Return the list of parameters to include in config signature for deduplication."""
        pass
    
    @abstractmethod
    def normalize_config(self, config):
        """Normalize configuration parameters for this classifier."""
        pass
    
    @abstractmethod  
    def get_timing_database_path(self):
        """Return the path to the timing database for this classifier."""
        pass
    
    @abstractmethod
    def get_quick_grid_parameters(self):
        """Return 3-level parameter grids for the top 3 highest impact parameters."""
        pass

# ========================================================================================
# PD SIGNAL CLASSIFIER STRATEGY
# ========================================================================================

class PDSignalStrategy(ClassifierStrategy):
    """Strategy for PD Signal Classifier optimization."""
    
    def get_parameter_space(self, categories=None, tiers=None):
        """Get hyperparameter search space from registry."""
        from hyperparameter_registry import get_search_space
        return get_search_space('pd_signal', categories=categories, tiers=tiers)
    
    def get_config_template(self):
        return get_pd_config_template()
    
    def get_experiment_log_path(self):
        return get_pd_experiment_log_path()
        
    def parse_experiment_log_row(self, row, tuner):
        """Parse a row from the PD signal experiment log."""
        # Parse dropout rates from combined format [conv_dropout, dense_dropout]
        dropout_rates = tuner._safe_parse_list(row.get('dropout_rates'), [])
        conv_dropout = dropout_rates[0] if len(dropout_rates) > 0 else 0.2
        dense_dropout = dropout_rates[1] if len(dropout_rates) > 1 else [0.3, 0.2]
        
        return {
            # Core training parameters
            'learning_rate': row.get('learning_rate'),
            'batch_size': int(row.get('batch_size', 0)) if pd.notna(row.get('batch_size')) else None,
            'epochs': int(row.get('epochs', 0)) if pd.notna(row.get('epochs')) else None,
            'k_folds': int(row.get('k_folds', 0)) if pd.notna(row.get('k_folds')) else None,
            # Architecture parameters
            'conv_filters': tuner._safe_parse_list(row.get('conv_filters')),
            'dense_units': tuner._safe_parse_list(row.get('dense_units')),
            # Regularization parameters
            'conv_dropout': conv_dropout,
            'dense_dropout': dense_dropout,
            'l2_regularization': row.get('l2_reg'),
            'early_stopping_patience': int(row.get('early_stopping_patience', 0)) if pd.notna(row.get('early_stopping_patience')) else None,
            'use_class_weights': row.get('class_weights'),
            # Data augmentation parameters
            'augment_fraction': row.get('augment_fraction'),
            'time_shift_probability': int(row.get('time_shift_probability', 0)),
            'time_shift_range': int(row.get('time_shift_range', 0)) if pd.notna(row.get('time_shift_range')) else None,
            'stretch_probability': row.get('stretch_probability'),
            'stretch_scale': row.get('stretch_scale'),
            'noise_probability': row.get('noise_probability'),
            'noise_std': row.get('noise_std'),
            'amplitude_scale_probability': row.get('amplitude_scale_probability'),
            'amplitude_scale': row.get('amplitude_scale')
        }
    
    def get_execution_command(self, config_file_path):
        return [
            sys.executable, 
            str(Path(__file__).parent / 'PD_signal_classifier_v3.py'), 
            '--config', str(config_file_path),
            '--source', 'hyperopt',
        ]
    
    def get_config_signature_params(self):
        """Return parameters for config signature generation."""
        return [
            # Core training parameters
            'learning_rate', 'batch_size', 'epochs', 'k_folds',
            # Architecture parameters
            'conv_filters', 'dense_units',
            # Regularization parameters  
            'conv_dropout', 'dense_dropout', 'l2_regularization', 'early_stopping_patience', 'use_class_weights',
            # Data augmentation parameters 
            'augment_fraction', 'time_shift_probability', 'time_shift_range', 'stretch_probability', 'stretch_scale',
            'noise_probability', 'noise_std', 'amplitude_scale_probability', 'amplitude_scale'
        ]
    
    def normalize_config(self, config):
        """Normalize configuration for PD signal classifier."""
        # PD signal classifier expects these exact parameter names - no transformation needed
        return config
    
    def get_timing_database_path(self):
        return get_pd_timing_database_path()
    
    def get_quick_grid_parameters(self):
        """Return 3-level grids for top 3 impact parameters: learning_rate, batch_size, dropout_rates."""
        return {
            'learning_rate': [0.0005, 0.001, 0.002],          # Conservative, default, aggressive
            'batch_size': [8, 16, 32],                        # Small, medium, large
            'dropout_rates': [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]  # Light, medium, heavy regularization
        }

# ========================================================================================
# CWT IMAGE CLASSIFIER STRATEGY  
# ========================================================================================

class CWTImageStrategy(ClassifierStrategy):
    """Strategy for CWT Image Classifier optimization."""
    
    def __init__(self, multi_channel=False):
        self.multi_channel = multi_channel
    
    def get_parameter_space(self, categories=None, tiers=None):
        """Get hyperparameter search space from registry."""
        from hyperparameter_registry import get_search_space
        return get_search_space('cwt_image', categories=categories, tiers=tiers)
    
    def get_config_template(self):
        return get_cwt_config_template(multi_channel=self.multi_channel)
    
    def get_experiment_log_path(self):
        return get_cwt_experiment_log_path()
        
    def parse_experiment_log_row(self, row, tuner):
        """Parse a row from the CWT experiment log."""
        return {
            # Core training parameters
            'learning_rate': row.get('learning_rate'),
            'batch_size': int(row.get('batch_size', 0)) if pd.notna(row.get('batch_size')) else None,
            'epochs': int(row.get('epochs', 0)) if pd.notna(row.get('epochs')) else None,
            'k_folds': int(row.get('k_folds', 0)) if pd.notna(row.get('k_folds')) else None,
            # Architecture parameters
            'conv_filters': tuner._safe_parse_list(row.get('conv_filters')),
            'dense_units': tuner._safe_parse_list(row.get('dense_units')),
            # Regularization parameters (CWT uses separate columns)
            'conv_dropout': row.get('conv_dropout'),
            'dense_dropout': tuner._safe_parse_list(row.get('dense_dropout'), [0.5]),  # Always expect list format
            'l2_regularization': row.get('l2_regularization'),
            'early_stopping_patience': int(row.get('early_stopping_patience', 0)) if pd.notna(row.get('early_stopping_patience')) else None,
            'use_class_weights': row.get('use_class_weights'),
            # CWT-specific parameters
            'img_width': int(row.get('img_width', 0)) if pd.notna(row.get('img_width')) else None,
            'img_height': int(row.get('img_height', 0)) if pd.notna(row.get('img_height')) else None,
            'img_channels': int(row.get('img_channels', 0)) if pd.notna(row.get('img_channels')) else None,
            # CWT-suitable augmentation parameters
            'augment_fraction': row.get('augment_fraction'),
            'time_shift_probability': row.get('time_shift_probability'),
            'time_shift_range': row.get('time_shift_range'),
            'noise_probability': row.get('noise_probability'),
            'noise_std': row.get('noise_std'),
            'brightness_probability': row.get('brightness_probability'),
            'brightness_range': row.get('brightness_range'),
            'contrast_probability': row.get('contrast_probability'),
            'contrast_range': row.get('contrast_range')
        }
    
    def get_execution_command(self, config_file_path):
        return [
            sys.executable,
            str(Path(__file__).parent / 'CWT_image_classifier_v3.py'),
            '--config', str(config_file_path)
        ]
    
    def get_config_signature_params(self):
        """Return parameters for config signature generation."""
        return [
            # Core training parameters
            'learning_rate', 'batch_size', 'epochs', 'k_folds',
            # Architecture parameters
            'conv_filters', 'dense_units',
            # Regularization parameters
            'conv_dropout', 'dense_dropout', 'l2_regularization', 'early_stopping_patience', 'use_class_weights',
            # Image-specific parameters
            'img_width', 'img_height', 'img_channels',
            # CWT-suitable augmentation parameters (critical for deduplication)
            'augment_fraction', 'time_shift_range', 'noise_std',
            'brightness_range', 'contrast_range'
        ]
    
    def normalize_config(self, config):
        """Normalize configuration for CWT image classifier with multi-channel support."""
        normalized = config.copy()
        
        # Handle multi-channel data directory configuration
        from config import resolve_cwt_data_channels
        try:
            # If config has multi-channel data, ensure consistency
            if 'cwt_data_channels' in normalized and normalized['cwt_data_channels'] is not None:
                channels_dict, channel_labels, channel_paths = resolve_cwt_data_channels(normalized)
                # Ensure img_channels matches directory count
                normalized['img_channels'] = len(channel_paths)
                # Keep cwt_data_dir for backward compatibility (use first path)
                normalized['cwt_data_dir'] = channel_paths[0]
            elif 'cwt_data_dir' in normalized:
                # Single-channel mode - ensure consistency
                normalized['img_channels'] = 1
                # Clear multi-channel config if present
                normalized['cwt_data_channels'] = None
        except:
            # Fallback for any configuration issues
            pass
        
        # Convert data_dir to cwt_data_dir if present (hyperopt compatibility)
        if 'data_dir' in normalized:
            normalized['cwt_data_dir'] = normalized.pop('data_dir')
            
        # Ensure dense_dropout is a single value, not list
        if 'dense_dropout' in normalized and isinstance(normalized['dense_dropout'], list):
            normalized['dense_dropout'] = normalized['dense_dropout'][0]  # Use first value
            
        return normalized
    
    def get_timing_database_path(self):
        return get_cwt_timing_database_path()
    
    def get_quick_grid_parameters(self):
        """Return 3-level grids for top 3 impact parameters: learning_rate, batch_size, dense_dropout."""
        return {
            'learning_rate': [0.0005, 0.001, 0.002],     # Conservative, default, aggressive
            'batch_size': [16, 32, 64],                  # Small, medium, large (CWT needs larger batches)
            'dense_dropout': [0.2, 0.4, 0.6]            # Light, medium, heavy regularization (single value for CWT)
        }

# ========================================================================================
# STRATEGY FACTORY
# ========================================================================================

def get_classifier_strategy(classifier_type, multi_channel=False):
    """Factory function to create classifier strategies."""
    strategies = {
        'pd_signal': PDSignalStrategy(),
        'cwt_image': CWTImageStrategy(multi_channel=multi_channel)
    }
    
    if classifier_type not in strategies:
        raise ValueError(f"Unknown classifier type: {classifier_type}. Available: {list(strategies.keys())}")
    
    return strategies[classifier_type]

class HyperparameterTuner:
    """Manages hyperparameter optimization experiments with classifier-agnostic optimization."""
    
    def __init__(self, classifier_type='pd_signal', base_config=None, output_root=None, verbose=False, concise=False, multi_channel=False, base_version=None):
        # Initialize classifier strategy
        self.classifier_type = classifier_type
        self.multi_channel = multi_channel
        self.verbose = verbose
        self.concise = concise
        self.strategy = get_classifier_strategy(classifier_type, multi_channel=multi_channel)
        
        # Determine base configuration
        if base_config:
            # Explicit base_config provided
            self.base_config = base_config
        elif base_version:
            # Load config from specific version
            version_config = self._find_config_by_version(base_version)
            if version_config:
                # Merge version config with template to ensure all required fields are present
                template_config = self.strategy.get_config_template()
                template_config.update(version_config)
                self.base_config = template_config
                if verbose:
                    accuracy = version_config.get('mean_val_accuracy')
                    accuracy_str = f"{accuracy:.4f}" if accuracy is not None else "unknown"
                    print(f"Using base configuration from {version_config.get('version', f'v{base_version}')} (accuracy: {accuracy_str})")
            else:
                if verbose:
                    print(f"Warning: Could not find version {base_version}, using default template")
                self.base_config = self.strategy.get_config_template()
        else:
            # Use default template
            self.base_config = self.strategy.get_config_template()
        
        # Use centralized configuration for output directory
        if output_root:
            self.base_output_root = Path(output_root)
        elif classifier_type == 'cwt_image':
            self.base_output_root = CWT_HYPEROPT_RESULTS_DIR
        else:
            self.base_output_root = PD_HYPEROPT_RESULTS_DIR
        
        # Initialise progress tracking lists
        self.results = []
        self.failed_configs = []
        
        # Create base output directory
        self.base_output_root.mkdir(parents=True, exist_ok=True)
        
        # Ensure classifier-specific directories exist
        if self.classifier_type == 'cwt_image':
            from config import ensure_cwt_directories
            ensure_cwt_directories()
        else:
            from config import ensure_directories
            ensure_directories()
        
        # Will be set when run_optimization is called
        self.output_root = None
        self.results_file = None
        self.run_info_file = None
        
        # Timing and progress tracking
        self.total_configs = 0
        
        # Simple time estimation (replaces complex timing system)
        from simple_timing_estimator import create_simple_estimator
        self.timing_estimator = create_simple_estimator(classifier_type)
        
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
        # Use strategy to get classifier-specific parameters for signature
        key_params = self.strategy.get_config_signature_params()
        
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
        log_file = self.strategy.get_experiment_log_path()
        
        if not Path(log_file).exists():
            return set()
        
        try:
            df = pd.read_csv(log_file, encoding='utf-8')
            previous_signatures = set()
            
            for _, row in df.iterrows():
                # Use strategy to parse experiment log row
                config = self.strategy.parse_experiment_log_row(row, self)
                
                signature = self._config_signature(config)
                previous_signatures.add(signature)
                
            return previous_signatures
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load previous configs for deduplication: {e}")
            return set()
    
    def _deduplicate_configs(self, configs, deduplication=True):
        """Remove configurations that have been tried before."""
        if not deduplication:
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
        
        return {i for i, config in enumerate(configs) 
                if self._config_signature(config) in previous_signatures}
    
    def _find_config_by_version(self, version_number):
        """Find configuration from previous experiments by version number.
        
        Args:
            version_number (int): Version number to find (e.g., 115 for v115)
            
        Returns:
            dict: Configuration dictionary or None if not found
        """
        log_file = self.strategy.get_experiment_log_path()
        
        if not Path(log_file).exists():
            return None
        
        try:
            df = pd.read_csv(log_file, encoding='utf-8')
            if df.empty:
                return None
            
            # Look for version in format "v115" or just "115"
            version_patterns = [f'v{version_number:03d}', f'v{version_number}', str(version_number)]
            
            matching_row = None
            for pattern in version_patterns:
                matches = df[df['version'].astype(str) == pattern]
                if not matches.empty:
                    matching_row = matches.iloc[0]  # Take first match
                    break
            
            if matching_row is None:
                return None
            
            return self._extract_config_from_row(matching_row)
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not find config for version {version_number}: {e}")
            return None
    
    def _extract_config_from_row(self, row):
        """Extract configuration dictionary from a dataframe row.

        Uses centralized template management to ensure all parameters have valid defaults,
        preventing None values that cause comparison errors.
        """
        # Start with template to ensure all parameters have valid defaults
        config = self.strategy.get_config_template()

        # Parse experiment log row using strategy-specific parsing
        row_config = self.strategy.parse_experiment_log_row(row, self)

        # Update template with values from experiment log (only non-None values)
        for key, value in row_config.items():
            if value is not None:
                config[key] = value

        # Add metadata fields
        config['mean_val_accuracy'] = row.get('mean_val_accuracy')
        config['version'] = row.get('version')

        return config
    
    def _find_best_previous_config(self):
        """Find the best performing configuration from previous experiments."""
        log_file = self.strategy.get_experiment_log_path()
        
        if not Path(log_file).exists():
            return None
        
        try:
            df = pd.read_csv(log_file, encoding='utf-8')
            if df.empty or 'mean_val_accuracy' not in df.columns:
                return None
            
            # Find the row with highest mean validation accuracy
            best_idx = df['mean_val_accuracy'].idxmax()
            best_row = df.iloc[best_idx]
            
            config = self._extract_config_from_row(best_row)
            
            # Brief mention when found (detailed display happens in smart mode)
            if self.verbose:
                print(f"Best previous config found: {config.get('version', 'unknown')} with accuracy {config['mean_val_accuracy']:.4f}")
            
            return config
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not find best previous config: {e}")
            return None
    
    def _get_neighboring_values(self, param_name, current_value, search_space, radius=1):
        """Get neighboring values around the current best value within search radius."""
        
        if self.verbose:
            print(f'\nSmart search called for {param_name} = {current_value}')
        
        if param_name not in search_space:
            return []
        
        value_list = search_space[param_name]
        
        if self.verbose:
            print(f'Available values: {value_list}')
        
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
        
        if self.verbose:
            print(f'Found neighbours: {neighbors}')
        
        return neighbors
    
    def define_search_space(self, categories=None, tiers=None):
        """
        Define hyperparameter search space using classifier strategy.
        
        Args:
            categories (list, optional): Filter by parameter categories
            tiers (list, optional): Filter by priority tiers
        
        Returns:
            dict: Classifier-specific parameter space
        """
        return self.strategy.get_parameter_space(categories=categories, tiers=tiers)
    
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
    
    
    
    
    def generate_smart_configs(self, search_radius=1, grid_search=False, ignore_params=None, max_grid_size=100, categories=None, priority_tiers=None, best_config='auto'):
        """
        Generate smart configurations using registry-based parameter filtering.
        
        Args:
            search_radius: Number of neighboring values to test (±1 or ±2)
            grid_search: If True, do full grid search within smart space; if False, use OFAT
            ignore_params: List of parameters to ignore (use best previous value)
            max_grid_size: Maximum number of configs before limiting parameters
            categories: List of parameter categories to include (training, regularization, architecture, training_control, augmentation)
            priority_tiers: List of priority tiers to include (1-5, where 1 is highest priority)
            best_config: Best configuration to search around ('auto' to find automatically)
        """
        if ignore_params is None:
            ignore_params = []
        
        configs = []
        
        # Try to find the best previous configuration
        if best_config == 'auto':
            best_config = self._find_best_previous_config()
        
        if best_config is None:
            return [self.base_config]
        
        # Get filtered search space from registry based on categories and priority tiers
        search_space = self.define_search_space(categories=categories, tiers=priority_tiers)
        
        if not search_space:
            if not self.concise:
                print("No parameters match the specified categories/priority filters. Using base config.")
            return [self.base_config]
        
        # Use the best config as the base for all new configurations
        base_config = self.base_config.copy()
        
        # Update base config with best known values for all parameters in search space
        for param in search_space.keys():
            if best_config.get(param) is not None:
                base_config[param] = best_config[param]
        
        # Get parameter list from search space (registry handles priority ordering)
        param_priority = list(search_space.keys())
        
        # Remove ignored parameters
        param_priority = [p for p in param_priority if p not in ignore_params]
        
        if not param_priority:
            if not self.concise:
                print("All parameters were ignored. Using base config.")
            return [self.base_config]
        
        if not self.concise:
            filter_info = []
            if categories:
                filter_info.append(f"categories: {categories}")
            if priority_tiers:
                filter_info.append(f"tiers: {priority_tiers}")
            filter_desc = f" ({', '.join(filter_info)})" if filter_info else ""
            print(f"Focusing on {len(param_priority)} parameters{filter_desc}: {', '.join(param_priority)}\n")
        
        # Build search space around best values
        smart_search_space = {}
        for param_name in param_priority:
            if param_name not in search_space:
                if self.verbose:
                    print(f'\n{param_name} not in search space')
                continue
                
            current_value = best_config.get(param_name)
            if current_value is None:
                if self.verbose:
                    print(f'\n{param_name} no current value found, starting from 0.0')
                current_value = 0.0
            
            # Get neighboring values within search radius
            neighbors = self._get_neighboring_values(param_name, current_value, search_space, search_radius)
            
            if neighbors:
                smart_search_space[param_name] = [current_value] + neighbors
                if self.verbose:
                    print(f"Exploring {param_name}: current={current_value}, testing {len(smart_search_space[param_name])} values")
        
        if not smart_search_space:
            if not self.concise:
                print("Warning: No neighboring values found - update the search space")
            return set()
        
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
    
    
    
    
    def generate_channel_ablation_configs(self, study_name=None):
        """
        Generate configurations for channel ablation study (CWT only).
        
        Tests:
        - All channels together
        - Each channel individually  
        - Each pair of channels (for 3+ channels)
        
        Args:
            study_name: Name prefix for the study (auto-generated if None)
            
        Returns:
            list: List of configurations for ablation study
        """
        if self.classifier_type != 'cwt_image':
            raise ValueError("Channel ablation is only supported for CWT image classifier")
        
        # Get base configuration
        base_config = self.base_config
        
        # Check if multi-channel configuration exists
        from config import resolve_cwt_data_channels
        try:
            channels_dict, channel_labels, channel_paths = resolve_cwt_data_channels(base_config)
        except Exception as e:
            error_msg = ("Channel ablation requires multi-channel data. Use --multi-channel flag to enable.\n"
                       "Multi-channel paths are configured in config.py (CWT_DATA_DIR_DICT).\n"
                       f"Error: {e}")
            raise ValueError(error_msg)
        
        if len(channel_labels) < 2:
            raise ValueError(f"Need at least 2 channels for ablation study. Found {len(channel_labels)}: {channel_labels}")
        
        # Auto-generate study name if not provided
        if study_name is None:
            study_name = f"ablation_{len(channel_labels)}ch"
        
        if not self.concise:
            print(f"Generating channel ablation study: {study_name}")
            print(f"Channels: {channel_labels}")
        
        configs = []
        
        # 1. All channels together (baseline)
        config_all = base_config.copy()
        config_all['ablation_study'] = f"{study_name}_all_channels"
        config_all['ablation_channels'] = channel_labels
        configs.append(config_all)
        
        # 2. Each channel individually
        for label in channel_labels:
            config_single = base_config.copy()
            config_single['cwt_data_channels'] = {label: channels_dict[label]}
            config_single['img_channels'] = 1
            config_single['ablation_study'] = f"{study_name}_only_{label}"
            config_single['ablation_channels'] = [label]
            configs.append(config_single)
        
        # 3. Each pair of channels (for 3+ channels)
        if len(channel_labels) >= 3:
            from itertools import combinations
            for pair in combinations(channel_labels, 2):
                config_pair = base_config.copy()
                config_pair['cwt_data_channels'] = {label: channels_dict[label] for label in pair}
                config_pair['img_channels'] = 2
                config_pair['ablation_study'] = f"{study_name}_pair_{'_'.join(pair)}"
                config_pair['ablation_channels'] = list(pair)
                configs.append(config_pair)
        
        if not self.concise:
            print(f"Generated {len(configs)} ablation configurations:")
            print(f"  - 1 baseline (all channels)")
            print(f"  - {len(channel_labels)} individual channels")
            if len(channel_labels) >= 3:
                pair_count = len(list(combinations(channel_labels, 2)))
                print(f"  - {pair_count} channel pairs")
        
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
        """Get actual image dimensions used in training (classifier-specific)."""
        img_width = config['img_width']
        
        if self.classifier_type == 'cwt_image':
            # CWT images have configurable height (default 256) and width (default 100)
            img_height = config.get('img_height', 256)
        else:
            # PD signal data: images are resized to (img_width, 2)
            # This matches cv2.resize(img, (img_width, 2)) in PD_signal_classifier_v3.py:100
            img_height = 2
        
        return img_width, img_height
    
    # Removed old complex timing database loading - using SimpleTimingEstimator now
    
    def calculate_model_complexity(self, config):
        """Calculate a complexity score for the model configuration."""
        # Count total parameters (roughly)
        conv_params = 0
        input_channels = config.get('img_channels', 1)  # CWT: 1 channel, PD: 1 channel
        
        # Get kernel size (CWT has configurable kernel size, PD assumes 3x3)
        if self.classifier_type == 'cwt_image':
            kernel_size = config.get('conv_kernel_size', [3, 3])
            kernel_h, kernel_w = kernel_size if len(kernel_size) == 2 else (kernel_size[0], kernel_size[0])
        else:
            kernel_h, kernel_w = 3, 3  # PD uses fixed 3x3 kernels
        
        for filters in config['conv_filters']:
            # Conv2D params = (kernel_h * kernel_w * input_channels + 1) * output_channels
            conv_params += (kernel_h * kernel_w * input_channels + 1) * filters
            input_channels = filters
        
        # Dense parameters
        dense_params = 0
        # Estimate flattened conv output size that feeds into first dense layer
        if config['conv_filters']:
            last_conv_filters = config['conv_filters'][-1]
            
            if self.classifier_type == 'cwt_image':
                # CWT: More complex spatial reduction due to pooling layers
                img_width, img_height = self.get_image_dimensions(config)
                pool_layers = config.get('pool_layers', [2, 5])
                pool_size = config.get('pool_size', [2, 2])
                
                # Estimate spatial dimensions after pooling (rough approximation)
                # Each pooling reduces dimensions by pool_size factor
                spatial_reduction = (pool_size[0] * pool_size[1]) ** len(pool_layers)
                remaining_spatial = max(1, (img_width * img_height) // spatial_reduction)
                prev_units = last_conv_filters * remaining_spatial
            else:
                # PD: Simple spatial reduction assumption
                prev_units = last_conv_filters * 10
        else:
            prev_units = 100  # Fallback
        
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
        
        # Classifier-specific baseline
        if self.classifier_type == 'cwt_image':
            baseline_pixels = 100 * 256  # CWT baseline: 100 × 256 = 25,600 pixels
        else:
            baseline_pixels = 100 * 2    # PD baseline: 100 × 2 = 200 pixels
        
        # Linear scaling with total pixels (not quadratic, since it's just resizing)
        image_factor = total_pixels / baseline_pixels
        
        fold_factor = config['k_folds'] / 5.0  # Normalize to 5 folds
        
        return batch_efficiency * image_factor * fold_factor
    
    def get_power_law_time_estimate(self, complexity, data_factor, estimated_epochs, config):
        """
        Get time estimate using power law relationship fitted to historical data.
        
        Args:
            complexity: Model complexity (parameter count)
            data_factor: Data processing factor
            estimated_epochs: Estimated actual epochs
            config: Configuration dictionary
            
        Returns:
            float: Estimated training time in minutes
        """
        import numpy as np
        from scipy.stats import linregress
        
        records = self.timing_data['timing_records']
        
        if len(records) < 3:
            # Fallback to conservative estimate if insufficient data
            return self.get_conservative_time_estimate(complexity, data_factor, estimated_epochs)
        
        # Extract data for power law fitting
        complexities = np.array([r['complexity'] for r in records])
        actual_times = np.array([r['actual_time'] for r in records])
        data_factors = np.array([r['data_factor'] for r in records])
        
        # Normalize by data factor to isolate complexity effect
        normalized_times = actual_times / data_factors
        
        # Fit power law using log-log regression: log(time) = log(a) + b*log(complexity)
        try:
            log_complexity = np.log(complexities)
            log_time = np.log(np.maximum(normalized_times, 0.01))  # Avoid log(0)
            
            slope, intercept, r_value, p_value, std_err = linregress(log_complexity, log_time)
            
            # Power law parameters: time = scaling_factor * complexity^power_coefficient
            power_coefficient = slope
            scaling_factor = np.exp(intercept)
            
            # Store power law parameters for future use
            if not hasattr(self.timing_data, 'power_law_params'):
                self.timing_data['power_law_params'] = {}
            
            self.timing_data['power_law_params'] = {
                'power_coefficient': power_coefficient,
                'scaling_factor': scaling_factor,
                'r_squared': r_value**2,
                'sample_size': len(records),
                'last_fitted': datetime.datetime.now().isoformat()
            }
            
            # Calculate base prediction using power law
            base_time = scaling_factor * (complexity ** power_coefficient) * data_factor
            
            # Apply recent adjustment based on prediction accuracy
            recent_records = records[-5:]  # Use fewer records for power law adjustment
            if len(recent_records) >= 2:
                ratios = []
                for record in recent_records:
                    # Recalculate power law prediction for this record
                    record_complexity = record['complexity']
                    record_data_factor = record['data_factor']
                    power_prediction = scaling_factor * (record_complexity ** power_coefficient) * record_data_factor
                    
                    if power_prediction > 0:
                        ratios.append(record['actual_time'] / power_prediction)
                
                if ratios:
                    # Apply median adjustment to reduce outlier impact
                    ratios.sort()
                    median_ratio = ratios[len(ratios)//2]
                    base_time *= median_ratio
            
            if self.verbose:
                interpretation = "Sub-linear" if power_coefficient < 0.9 else "Super-linear" if power_coefficient > 1.1 else "Linear"
                print(f"Power law timing model: complexity^{power_coefficient:.3f} ({interpretation}), R²={r_value**2:.3f}")
            
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as e:
            # Fallback to linear model if power law fitting fails
            if self.verbose:
                print(f"Power law fitting failed ({e}), using linear fallback")
            return self.get_adaptive_time_estimate(config)
        
        return max(base_time, 0.5)  # Minimum 0.5 minutes
    
    def get_adaptive_time_estimate(self, config):
        """Get time estimate based on historical data (fallback when power law unavailable)."""
        # IMPORTANT: This should NOT use the old broken complexity calculation
        # Instead, use conservative estimate based on architectural heuristics
        
        # Simplified data factor for fallback
        batch_efficiency = 16.0 / config['batch_size']  # Normalize to batch_size=16
        fold_factor = config['k_folds'] / 5.0  # Normalize to 5 folds
        data_factor = batch_efficiency * fold_factor
        
        estimated_epochs = self.estimate_actual_epochs(config)
        
        # If we have timing records with REAL complexity, try to estimate from them
        recent_records = self.timing_data['timing_records'][-10:]  # Last 10 records
        if recent_records:
            # Use actual recorded complexity values (which should be real Keras counts)
            similar_records = []
            for record in recent_records:
                # Look for records with similar architecture characteristics
                if (record['classifier_type'] == self.classifier_type and 
                    record['complexity'] > 0 and record['data_factor'] > 0):
                    similar_records.append(record)
            
            if similar_records:
                # Average the time per complexity from real records
                time_per_complexity_values = []
                for record in similar_records:
                    time_per_complexity = record['actual_time'] / (record['complexity'] * record['data_factor'])
                    time_per_complexity_values.append(time_per_complexity)
                
                avg_time_per_complexity = sum(time_per_complexity_values) / len(time_per_complexity_values)
                
                # For this config, we need to estimate complexity without the broken calculation
                # Use a simple heuristic based on architecture size until we get real complexity
                estimated_complexity = self._estimate_complexity_heuristic(config)
                base_time = avg_time_per_complexity * estimated_complexity * data_factor
                return max(base_time, 0.5)
        
        # Fallback to conservative estimate if no historical data
        estimated_complexity = self._estimate_complexity_heuristic(config)
        return self.get_conservative_time_estimate(estimated_complexity, data_factor, estimated_epochs)
    
    def _estimate_complexity_heuristic(self, config):
        """Simple heuristic for complexity estimation when no real complexity available."""
        if self.classifier_type == 'cwt_image':
            # CWT models are typically 8-12M parameters for standard architectures
            conv_layers = len(config['conv_filters'])
            dense_size = sum(config['dense_units']) if config['dense_units'] else 128
            # Rough heuristic: more layers and dense units = more parameters
            estimated_complexity = (conv_layers * 500000) + (dense_size * 30000)
            return max(estimated_complexity, 8000000)  # Minimum 8M for CWT
        else:
            # PD models are typically 0.5-2M parameters  
            conv_layers = len(config['conv_filters'])
            dense_size = sum(config['dense_units']) if config['dense_units'] else 128
            estimated_complexity = (conv_layers * 50000) + (dense_size * 5000)
            return max(estimated_complexity, 500000)  # Minimum 500K for PD
    
    def get_conservative_time_estimate(self, complexity, data_factor, estimated_epochs):
        """Conservative time estimate when no historical data is available."""
        # Conservative estimate: ~1-5 seconds per thousand parameters per epoch
        base_time_per_epoch = (complexity / 1000.0) * 0.05  # 3 seconds per 1K params per epoch
        base_time = base_time_per_epoch * estimated_epochs * data_factor
        
        return max(base_time, 1.0)  # Minimum 1 minute
    
    def update_timing_database(self, config, actual_time_minutes, model_complexity=None):
        """Update the timing database with actual training results."""
        # Use real model complexity if available, otherwise fall back to calculation
        if model_complexity and model_complexity > 0:
            complexity = model_complexity
        else:
            complexity = self.calculate_model_complexity(config)
            
        # Simplified data factor - just batch size and k-folds scaling
        batch_efficiency = 16.0 / config['batch_size']  # Normalize to batch_size=16
        fold_factor = config['k_folds'] / 5.0  # Normalize to 5 folds
        data_factor = batch_efficiency * fold_factor
        
        # Calculate actual time per complexity unit
        actual_time_per_complexity = actual_time_minutes / (complexity * data_factor)
        
        # Get predicted time using appropriate model  
        estimated_epochs = self.estimate_actual_epochs(config)
        if len(self.timing_data['timing_records']) >= 2:
            predicted_time = self.get_power_law_time_estimate(complexity, data_factor, estimated_epochs, config)
        else:
            predicted_time = self.get_conservative_time_estimate(complexity, data_factor, estimated_epochs)
        
        # Clean, minimal record structure - only essential data for power law modeling
        record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'complexity': complexity,  # Real Keras parameter count
            'data_factor': data_factor,  # Simplified: batch size and k-fold scaling only  
            'actual_time': actual_time_minutes,  # Total training time
            'predicted_time': predicted_time,  # What we predicted
            'epochs_trained': estimated_epochs,  # Estimated actual epochs (for reference)
            'classifier_type': self.classifier_type,  # Track PD vs CWT
            # Minimal config fingerprint for debugging
            'config_key': {
                'arch': f"{config['conv_filters']}-{config['dense_units']}",
                'lr': config['learning_rate'],
                'bs': config['batch_size'],
                'kf': config['k_folds']
            }
        }
        
        self.timing_data['timing_records'].append(record)
        
        # Update metadata
        self.timing_data['last_updated'] = datetime.datetime.now().isoformat()
        
        # Keep only last 30 records per classifier type (more manageable)
        self.timing_data['timing_records'] = self.timing_data['timing_records'][-30:]
        
        # Save to disk
        with open(self.timing_db_file, 'w') as f:
            json.dump(self.timing_data, f, indent=2)
        
        if self.verbose:
            error_pct = abs(actual_time_minutes - predicted_time) / actual_time_minutes * 100 if actual_time_minutes > 0 else 0
            print(f"Updated timing database: {actual_time_minutes:.1f}min actual vs {predicted_time:.1f}min predicted ({error_pct:.1f}% error)")
    
    def save_config_to_file(self, config, config_number_in_run):
        """Save configuration to JSON file with run-scoped sequential numbering."""
        config_file = Path(self.output_root) / f"config_{config_number_in_run:03d}.json"
        
        # Normalize config using strategy
        normalized_config = self.strategy.normalize_config(config)
        
        # Add metadata to config for better traceability
        enhanced_config = normalized_config.copy()
        enhanced_config['_metadata'] = {
            'config_number_in_run': config_number_in_run,
            'run_id': self.run_info['run_id'],
            'classifier_type': self.classifier_type,
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
    
    def run_training_experiment(self, config, config_number_in_run, config_file_path, version, verbose=False, concise=False):
        """Run training experiment"""
        try:
            # Start timing
            start_time = time.time()
            
            # Run training script (from ml directory)
            return self._execute_training_subprocess(config, config_number_in_run, config_file_path, version, start_time, verbose, concise)
            
        except FileNotFoundError as e:
            print(f"Error: Training script not found - {e}")
            self.failed_configs.append({
                'config_number_in_run': config_number_in_run,
                'config': config,
                'error': f'Training script not found: {e}'
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
            print(f"Unexpected error in config {config_number_in_run}: {e}")
            self.failed_configs.append({
                'config_number_in_run': config_number_in_run,
                'config': config,
                'error': f'Unexpected error: {e}'
            })
            return None
    
    def _execute_training_subprocess(self, config, config_number_in_run, config_file_path, version, start_time, verbose, concise):
        """Execute the training subprocess"""
        cmd = self.strategy.get_execution_command(config_file_path)
        
        # Add hyperopt-specific parameters for both PD and CWT classifiers
        if self.classifier_type in ['pd_signal', 'cwt_image']:
            cmd.extend([
                '--source', 'hyperopt',
                '--hyperopt_run_id', self.run_info['run_id'],
                '--config_file', config_file_path,
                '--config_number_in_run', str(config_number_in_run)
            ])
        
        # Add concise flag if needed
        if concise:
            cmd.append('--concise')
        
        try:
            # Set up environment for subprocess
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSFSENCODING'] = '0'
            
            if verbose or concise:
                # Run with real-time output for verbose or concise mode
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
                    
                    # Extract results from experiment log using consolidated function
                    classifier_type = 'pd_signal' if isinstance(self.strategy, PDSignalStrategy) else 'cwt_image'
                    experiment_result = extract_experiment_result(
                        classifier_type=classifier_type,
                        version=version,
                        config_number_in_run=config_number_in_run,
                        config_file_path=config_file_path,
                        run_id=self.run_info['run_id']
                    )
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
                # Run in silent mode (default case)
                result = subprocess.run(
                    cmd, 
                    timeout=7200,  # 2 hour timeout
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    env=env,
                    cwd=str(Path(__file__).parent)
                )
                
                if result.returncode == 0:
                    print(f"Config {config_number_in_run} completed successfully")
                    
                    # Extract results from experiment log using consolidated function
                    classifier_type = 'pd_signal' if isinstance(self.strategy, PDSignalStrategy) else 'cwt_image'
                    experiment_result = extract_experiment_result(
                        classifier_type=classifier_type,
                        version=version,
                        config_number_in_run=config_number_in_run,
                        config_file_path=config_file_path,
                        run_id=self.run_info['run_id']
                    )
                    if experiment_result:
                        return experiment_result
                    else:
                        # Fallback: try to extract results from captured output
                        if result.stdout:
                            extracted = self.extract_results_from_output(result.stdout, config, config_number_in_run, version, config_file_path)
                            if extracted and 'mean_val_accuracy' in extracted:
                                return extracted
                        
                        return {
                            'config_number_in_run': config_number_in_run,
                            'timestamp': datetime.datetime.now().isoformat(),
                            'version': version,
                            'config_file': config_file_path,
                            'note': 'Silent mode - check experiment logs for results'
                        }
                else:
                    print(f"Config {config_number_in_run} failed (exit code {result.returncode})")
                    if result.stderr:
                        print(f"Error output: {result.stderr[:500]}...")
                    self.failed_configs.append({
                        'config_number_in_run': config_number_in_run,
                        'config': config,
                        'error': f'Process failed with exit code {result.returncode}',
                        'stderr': result.stderr[:1000] if result.stderr else None
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
    
    # Using consolidated extract_experiment_result function from config.py
    
    def extract_results_from_output(self, output, config, config_number_in_run, version, config_file_path):
        """Extract key results from training output (fallback method)."""
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
            val_acc_match = re.search(r'Mean Val Acc[:\s]+(\d+\.\d+)', output, re.IGNORECASE)
            if val_acc_match:
                result['mean_val_accuracy'] = float(val_acc_match.group(1))
            
            # Look for training time 
            time_match = re.search(r'(\d+\.\d+)\s*minutes', output, re.IGNORECASE)
            if time_match:
                result['training_time_minutes'] = float(time_match.group(1))
            
            # Look for model complexity (Total Parameters: X,XXX,XXX)
            params_match = re.search(r'Total Parameters:\s*([\d,]+)', output, re.IGNORECASE)
            if params_match:
                # Remove commas and convert to int
                param_str = params_match.group(1).replace(',', '')
                result['model_complexity'] = int(param_str)
            
            # Look for precision, recall, F1-score in concise output (pattern: "P: 0.85 | R: 0.90 | F1: 0.87")
            metrics_match = re.search(r'P:\s*(\d+\.\d+)\s*\|\s*R:\s*(\d+\.\d+)\s*\|\s*F1:\s*(\d+\.\d+)', output, re.IGNORECASE)
            if metrics_match:
                result['mean_precision'] = float(metrics_match.group(1))
                result['mean_recall'] = float(metrics_match.group(2))
                result['mean_f1_score'] = float(metrics_match.group(3))
            else:
                # Try individual patterns as fallback
                precision_match = re.search(r'precision[:\s]+(\d+\.\d+)', output, re.IGNORECASE)
                if precision_match:
                    result['mean_precision'] = float(precision_match.group(1))
                
                recall_match = re.search(r'recall[:\s]+(\d+\.\d+)', output, re.IGNORECASE)
                if recall_match:
                    result['mean_recall'] = float(recall_match.group(1))
                
                f1_match = re.search(r'f1[:\s]+(\d+\.\d+)', output, re.IGNORECASE)
                if f1_match:
                    result['mean_f1_score'] = float(f1_match.group(1))
            
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
            'dense_dropouts': self._analyze_dense_dropouts(configs),
            'l2_regs': sorted(set(config['l2_regularization'] for config in configs)),
            'architectures': []
        }
        
        # Analyze unique architectures
        arch_set = set()
        for config in configs:
            arch = (tuple(config['conv_filters']), tuple(config['dense_units']))
            arch_set.add(arch)
        analysis['architectures'] = sorted(list(arch_set))
        
        # Calculate total training time estimate using simple timing system
        total_time_minutes = 0
        for config in configs:
            time_estimate = self.timing_estimator.estimate_time(config)
            total_time_minutes += time_estimate
        
        analysis['estimated_total_time_hours'] = total_time_minutes / 60
        
        return analysis
    
    def _analyze_dense_dropouts(self, configs):
        """Analyze dense dropout values - now consistently stored as lists/tuples."""
        dropout_values = set()
        
        for config in configs:
            dropout = config.get('dense_dropout')
            if dropout is not None:
                if isinstance(dropout, (list, tuple)):
                    # Standard format: list of values
                    dropout_values.add(tuple(dropout))
                else:
                    # Legacy single value - convert to tuple for consistency
                    dropout_values.add((dropout,))
        
        return sorted(list(dropout_values))
    
    def _format_dense_dropouts(self, dense_dropouts):
        """Format dense dropout values for display - all values are now tuples."""
        return [list(dropout) for dropout in dense_dropouts]
    
    def display_start_screen(self, mode, configs, resume=False, completed_ids=None, skip_from_deduplication=None, skip_from_max_configs=None, configs_to_run_count=None, smart_config_info=None):
        """Display comprehensive start screen with configuration analysis."""
        analysis = self.analyze_configuration_space(configs)
        
        # Use provided counts if available, otherwise calculate
        if completed_ids is None:
            completed_ids = set()
        if skip_from_deduplication is None:
            skip_from_deduplication = set()
        if skip_from_max_configs is None:
            skip_from_max_configs = set()
            
        # Calculate which configs will actually run for better preview
        configs_to_execute = []
        execution_order = 0
        for i, config in enumerate(configs, 1):
            config_index = i - 1  # Convert to 0-based index for skip check
            if (config_index not in skip_from_deduplication and 
                config_index not in skip_from_max_configs and 
                i not in completed_ids):
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
        
        # Handle smart mode display with special formatting
        if smart_config_info:
            focus_mode = smart_config_info['focus_mode']
            mode_str = "grid search" if smart_config_info['grid_search'] else "OFAT"
            print(f"   Mode: SMART ({focus_mode.upper()})")
            print(f"   Best config: {smart_config_info['best_config'].get('version', 'unknown')} (accuracy: {smart_config_info['best_config']['mean_val_accuracy']:.4f})")
            print(f"   Search radius: ±{smart_config_info['search_radius']}")
            if smart_config_info['ignore_params']:
                print(f"   Ignoring parameters: {', '.join(smart_config_info['ignore_params'])}")
        else:
            print(f"   Mode: {mode.upper()}")
            
        print(f"   Total Configurations: {analysis['total_configs']}")
        
        # Show deduplication and completion info more accurately
        if skip_from_deduplication:
            print(f"   Skipped by Deduplication: {len(skip_from_deduplication)}")
        if skip_from_max_configs:
            print(f"   Skipped by Max Configs Limit: {len(skip_from_max_configs)}")
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
        
        # Calculate time estimate for configs that will actually run
        if configs_to_run_count is not None and configs_to_run_count > 0:
            # Calculate time only for configs that will run
            total_time_for_running_configs = 0
            configs_counted = 0
            
            for i, config in enumerate(configs):
                config_index = i  # 0-based index
                if (config_index not in skip_from_deduplication and 
                    config_index not in skip_from_max_configs and 
                    (i + 1) not in completed_ids):
                    time_estimate = self.timing_estimator.estimate_time(config)
                    total_time_for_running_configs += time_estimate
                    configs_counted += 1
                    if configs_counted >= configs_to_run_count:
                        break
            
            actual_time_hours = total_time_for_running_configs / 60
            avg_time_per_config = total_time_for_running_configs / max(configs_counted, 1)
            
            # Store for use in confirmation section
            self._last_calculated_time_hours = actual_time_hours
            
            if resume and completed_ids:
                print(f"   Estimated Remaining Time: {actual_time_hours:.1f} hours")
                print(f"   Average per Config: {avg_time_per_config:.1f} minutes")
                print(f"   (Original total estimate was {analysis['estimated_total_time_hours']:.1f} hours for {analysis['total_configs']} configs)")
            else:
                print(f"   Estimated Total Time: {actual_time_hours:.1f} hours")
                print(f"   Average per Config: {avg_time_per_config:.1f} minutes")
        else:
            # Fallback to original calculation
            print(f"   Estimated Total Time: {analysis['estimated_total_time_hours']:.1f} hours")
            print(f"   Average per Config: {analysis['estimated_total_time_hours']*60/analysis['total_configs']:.1f} minutes")
        
        # Show simple timing database status
        num_records = len(self.timing_estimator.records)
        if num_records >= 3:
            stats = self.timing_estimator.get_stats()
            print(f"   Timing Database: {stats}")
        elif num_records > 0:
            print(f"   Timing Database: {num_records} records (insufficient for power law)")
        else:
            print(f"   Timing Database: No historical data (using conservative estimates)")
        
        print(f"\nHYPERPARAMETER RANGES (most significant):")
        print(f"   Learning Rates: {analysis['learning_rates']}")
        print(f"   Batch Sizes: {analysis['batch_sizes']}")
        print(f"   Epochs per Config: {analysis['epochs']}")
        print(f"   K-Folds per Config: {analysis['k_folds']}")
        print(f"   Conv Dropout Rates: {analysis['conv_dropouts']}")
        print(f"   Dense Dropout Rates: {self._format_dense_dropouts(analysis['dense_dropouts'])}")
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
        
        if skip_from_deduplication or skip_from_max_configs or completed_ids:
            skipped_total = len(skip_from_deduplication) + len(skip_from_max_configs) + len(completed_ids)
            skip_parts = []
            if skip_from_deduplication:
                skip_parts.append(f"{len(skip_from_deduplication)} duplicates")
            if skip_from_max_configs:
                skip_parts.append(f"{len(skip_from_max_configs)} by max_configs limit")
            if completed_ids:
                skip_parts.append(f"{len(completed_ids)} completed")
            print(f"   (Skipping {skipped_total} configs: {' + '.join(skip_parts)})")
        
        print("\n" + "="*100)
        
        return analysis
    
    def setup_run_directory(self, mode, resume=False):
        """Setup run directory paths and load info (but don't create directory yet)"""
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
                    
                    # Ensure progress structure exists (for legacy run_info files)
                    if 'progress' not in self.run_info:
                        self.run_info['progress'] = {
                            'completed_configs': [],
                            'current': 0,
                            'total': 0,
                            'last_updated': datetime.datetime.now().isoformat()
                        }
                else:
                    # Create run info for legacy runs
                    self.run_info = {
                        'run_id': run_dir,
                        'mode': mode,
                        'start_time': 'unknown',
                        'total_configs_executed': 0,
                        'is_legacy': True
                    }
            else:
                print(f"No existing {mode} run found to resume from. Creating new run.")
                resume = False
        
        if not resume:
            # Create new run directory name with timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = f"{mode}_run_{timestamp}"
            # Initialize run info for new runs
            self.run_info = {
                'run_id': run_dir,
                'mode': mode,
                'start_time': datetime.datetime.now().isoformat(),
                'total_configs_executed': 0,
                'config_files': {},  # Maps config_number -> config_file_path
                'version_mapping': {},  # Maps version -> config_number
                
                # Progress tracking (replaces tuning_progress.json)
                'progress': {
                    'completed_configs': [],
                    'current': 0,
                    'total': 0,
                    'last_updated': datetime.datetime.now().isoformat()
                }
            }
        
        # Setup paths (but don't create directory yet)
        self.output_root = self.base_output_root / run_dir
        self.results_file = self.output_root / "hyperparameter_results.csv"
        self.run_info_file = self.output_root / "run_info.json"
        
        return str(self.output_root)
    
    def _determine_configs_to_skip(self, configs, deduplication, resume, max_configs):
        """
        Determine which configurations to skip based on deduplication, resume, and limits.
        
        Returns:
            tuple: (skip_duplicates, completed_ids, skip_max_configs, configs_to_run_count)
                - skip_duplicates: set of 0-based indices to skip due to deduplication
                - completed_ids: set of 1-based indices already completed (for resume)  
                - skip_max_configs: set of 0-based indices to skip due to max_configs limit
                - configs_to_run_count: number of configs that will actually run
        """
        total_configs = len(configs)
        
        # Get duplicates (0-based indices)
        skip_duplicates = set()
        if deduplication:
            skip_duplicates = self._get_duplicate_indices(configs)
            if skip_duplicates and not self.concise:
                print(f"Smart deduplication: Will skip {len(skip_duplicates)}/{total_configs} configurations (already tested)")
        
        # Get completed configs from resume (1-based indices)
        completed_ids = set()
        if resume:
            # Try to load from run_info.json first (new format)
            if self.run_info_file.exists():
                with open(self.run_info_file, 'r') as f:
                    run_info = json.load(f)
                    completed_ids = set(run_info.get('progress', {}).get('completed_configs', []))
        
        # Get indices of configs to run using direct filtering
        available_indices = [
            i for i in range(total_configs) 
            if i not in skip_duplicates and (i + 1) not in completed_ids
        ]
        
        # Apply max_configs limit with separate tracking
        skip_max_configs = set()
        if max_configs and max_configs < len(available_indices):
            if not self.concise:
                print(f"Limiting to first {max_configs} configurations (from {len(available_indices)} available)")
            
            # Mark excess configs as skipped due to max_configs (separate from deduplication)
            excess_indices = available_indices[max_configs:]
            skip_max_configs.update(excess_indices)
        
        # Calculate final count
        configs_to_run_count = min(len(available_indices), max_configs or len(available_indices))
        
        return skip_duplicates, completed_ids, skip_max_configs, configs_to_run_count
    
    def create_actual_directory(self):
        """Actually create the directory and files after user confirmation"""
        # Create the directory
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Save run info
        run_info_file = self.output_root / "run_info.json"
        with open(run_info_file, 'w') as f:
            json.dump(self.run_info, f, indent=2)
        
        print(f"Run directory: {self.output_root.as_posix()}")
        print(f"Run ID: {self.run_info['run_id']}")
    
    def ensure_experiment_log_columns(self):
        """Ensure experiment log has the new columns for enhanced traceability."""
        log_file = self.strategy.get_experiment_log_path()
        new_columns = ['hyperopt_run_id', 'config_file', 'config_number_in_run']
        
        if not Path(log_file).exists():
            return  # Will be created with proper schema later
        
        try:
            df = pd.read_csv(log_file, encoding='utf-8')
            columns_added = False
            
            for col in new_columns:
                if col not in df.columns:
                    if col == 'hyperopt_run_id':
                        df[col] = 'unknown'  # Default for existing entries
                    elif col == 'config_file':
                        df[col] = ''  # Will be populated retroactively
                    elif col == 'config_number_in_run':
                        df[col] = -1  # -1 indicates unknown
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
                    # Recalculate time based on actual configs to run (already calculated above)
                    if hasattr(self, '_last_calculated_time_hours'):
                        estimated_time_hours = self._last_calculated_time_hours
                    else:
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
    
    def run_optimization(self, mode='smart', max_configs=None, resume=False, verbose=False, concise=False, deduplication=True, search_radius=1, grid_search=False, ignore_params=None, max_grid_size=100, categories=None, priority_tiers=None):
        """
        Run hyperparameter optimization.
        
        Args:
            mode: Optimization mode options:
                - test: 2 configs for testing (2 folds, 2 epochs)
                - smart: Registry-based adaptive search around best previous results
                - channel-ablation: Multi-channel analysis study (CWT only)
            max_configs: Maximum number of configs to test (None for all)
            resume: Whether to resume from previous run
            deduplication: If True, skip previously tested configurations (default: True)
            search_radius: For smart mode, number of neighboring values to test around best config (±1 or ±2)
            grid_search: If True, do full grid search; if False, use OFAT (default)
            ignore_params: List of parameters to ignore in smart mode
            max_grid_size: Maximum grid size before parameter limiting
            categories: List of parameter categories to include (training, regularization, architecture, training_control, augmentation)
            priority_tiers: List of priority tiers to include (1-5, where 1 is highest priority)
        """
        # Setup run directory paths (but don't create directory yet)
        self.setup_run_directory(mode, resume)
        
        # Ensure experiment log has new columns
        self.ensure_experiment_log_columns()
        
        # For smart mode, store config info for later display in OPTIMIZATION DETAILS
        smart_config_info = None
        if mode == 'smart':
            best_config = self._find_best_previous_config()
            print()
            print('Best config:')
            print(best_config)
            print()
            if best_config:
                # Determine focus mode description
                if categories:
                    focus_mode = f"categories: {', '.join(categories)}"
                elif priority_tiers:
                    focus_mode = f"priority tiers: {', '.join(map(str, priority_tiers))}"
                else:
                    focus_mode = "all parameters"
                
                smart_config_info = {
                    'best_config': best_config,
                    'categories': categories,
                    'priority_tiers': priority_tiers,
                    'search_radius': search_radius,
                    'ignore_params': ignore_params,
                    'grid_search': grid_search,
                    'focus_mode': focus_mode
                }
            else:
                print(f"\nNo previous experiments found - using default base config as origin")
        
        # Generate configurations
        if mode == 'smart':
            configs = self.generate_smart_configs(
                search_radius=search_radius, 
                grid_search=grid_search, 
                ignore_params=ignore_params or [], 
                max_grid_size=max_grid_size, 
                categories=categories,
                priority_tiers=priority_tiers,
                best_config=best_config
            )
        elif mode == 'channel-ablation':
            # Channel ablation mode - CWT only
            if self.classifier_type != 'cwt_image':
                raise ValueError("Channel ablation mode is only supported for CWT image classifier")
            configs = self.generate_channel_ablation_configs()
        elif mode == 'test':
            configs = self.generate_test_configs()
        else:
            available_modes = ['smart', 'channel-ablation', 'test']
            raise ValueError(f"Unknown mode: {mode}. Available modes: {available_modes}")
        
        # Determine which configs to skip and which to run
        skip_from_deduplication, completed_ids, skip_from_max_configs, configs_to_run_count = self._determine_configs_to_skip(
            configs, deduplication, resume, max_configs)
        
        # Display start screen and get confirmation (with accurate counts)
        analysis = self.display_start_screen(mode, configs, resume, completed_ids, 
                                           skip_from_deduplication, skip_from_max_configs, configs_to_run_count, 
                                           smart_config_info=smart_config_info)
        
        if not self.get_user_confirmation(analysis, resume, completed_ids, configs_to_run_count):
            return
        
        # Now create the actual directory after user confirmation
        self.create_actual_directory()
        
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
            # Skip configs that were already tested, completed, or beyond max_configs limit
            config_index = i - 1  # Convert to 0-based index for skip checks
            if (config_index in skip_from_deduplication or 
                config_index in skip_from_max_configs or 
                i in completed_ids):
                if not self.concise:
                    if i in completed_ids:
                        skip_reason = "already completed"
                    elif config_index in skip_from_deduplication:
                        skip_reason = "already tested in a previous run"
                    else:  # config_index in skip_from_max_configs
                        skip_reason = "beyond max_configs limit"
                    print(f"Skipping configuration {i} ({skip_reason})")
                continue
            
            # Increment sequential counter for configs actually being run in this run
            config_number_in_run += 1
            
            # Format dense dropout for clean display
            dense_dropout_str = str(config['dense_dropout']).replace(' ', '') if isinstance(config['dense_dropout'], list) else config['dense_dropout']
            
            # Get the actual version that will be used (accounts for previous experiments)
            current_version_num = get_next_version_from_log(classifier_type=self.classifier_type)
            actual_version = format_version(current_version_num)
            
            print(f"\n{'='*90}")
            print(f"Config {config_number_in_run}/{configs_to_run_count} | Version: {actual_version}")
            print(f"Parameters: LR={config['learning_rate']} | BS={config['batch_size']} | Dropout={config['conv_dropout']}/{dense_dropout_str}")
            print(f"Run: {self.run_info['run_id']} | Config File: config_{i:03d}.json")
            print(f"{'='*90}")
            
            # Save config with original config numbering and run the experiment
            config_file_path = self.save_config_to_file(config, i)
            
            # Update run info with version mapping
            self.run_info['version_mapping'][actual_version] = config_number_in_run
            
            result = self.run_training_experiment(config, config_number_in_run, config_file_path, actual_version, verbose=verbose, concise=concise)
            
            if result:
                # Enhanced result with traceability info
                result['config_id'] = i  # Keep original for resume compatibility
                result['config_number_in_run'] = config_number_in_run
                result['config_file'] = config_file_path
                result['hyperopt_run_id'] = self.run_info['run_id']
                result['execution_order'] = config_number_in_run  # Same as config_number_in_run now
                self.results.append(result)
                self.save_intermediate_results()
                
                # Update timing database with actual training time
                if result.get('training_time_minutes', 0) > 0 and result.get('model_complexity', 0) > 0:
                    self.timing_estimator.record_actual_time(
                        config, result['training_time_minutes'], result['model_complexity'])
                
                # Concise completion message
                if self.concise:
                    actual_time_min = result.get('training_time_minutes', 0)
                    print(f"✅ Config {config_number_in_run}/{configs_to_run_count} completed ({actual_time_min:.1f} min)")
                
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
            
            # Concise progress message with ETA
            if self.concise and remaining > 0:
                print(f"📊 Progress: {config_number_in_run}/{configs_to_run_count} | ETA: {eta:.1f}h ({remaining} configs remaining)")
            
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
        """Save progress to run_info file (replaces separate tuning_progress.json)."""
        # Update progress in run_info
        self.run_info['progress'] = {
            'completed_configs': completed_configs,
            'current': current,
            'total': total,
            'last_updated': datetime.datetime.now().isoformat()
        }
        
        # Save updated run_info
        with open(self.run_info_file, 'w') as f:
            json.dump(self.run_info, f, indent=2)
    
    
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
        log_file = str(self.strategy.get_experiment_log_path())
        
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
            
            # Display precision, recall, F1-score if available
            if 'mean_precision' in best_result and best_result['mean_precision'] > 0:
                print(f"   Precision: {best_result['mean_precision']:.4f}")
            if 'mean_recall' in best_result and best_result['mean_recall'] > 0:
                print(f"   Recall: {best_result['mean_recall']:.4f}")
            if 'mean_f1_score' in best_result and best_result['mean_f1_score'] > 0:
                print(f"   F1 Score: {best_result['mean_f1_score']:.4f}")
                
            print(f"   Config ID: {best_result['config_id']}")
            print(f"   Learning Rate: {best_result['learning_rate']}")
            print(f"   Batch Size: {best_result['batch_size']}")
            best_dense_dropout_str = self._format_dropout_value(best_result['dense_dropout'])
            print(f"   Dropout: {best_result['conv_dropout']}/{best_dense_dropout_str}")
            
            print(f"\nTop 5 Results:")
            for i, (_, row) in enumerate(df.head(5).iterrows()):
                # Read from DataFrame 
                config_id = row['config_id']
                accuracy = row['mean_val_accuracy']
                learning_rate = row['learning_rate']
                batch_size = row['batch_size']
                version = format_version(config_id)
                
                # Include precision, recall, F1-score if available
                metrics_str = ""
                if 'mean_precision' in row and row['mean_precision'] > 0:
                    metrics_str += f", P={row['mean_precision']:.3f}"
                if 'mean_recall' in row and row['mean_recall'] > 0:
                    metrics_str += f", R={row['mean_recall']:.3f}"
                if 'mean_f1_score' in row and row['mean_f1_score'] > 0:
                    metrics_str += f", F1={row['mean_f1_score']:.3f}"
                
                print(f"   {i+1}. Config {config_id} ({version}): {accuracy:.4f} "
                      f"(LR={learning_rate}, BS={batch_size}{metrics_str})")
        
        # Save failed configs if any
        if self.failed_configs:
            failed_file = self.output_root / "failed_configs.json"
            with open(failed_file, 'w') as f:
                clean_failed_configs = convert_numpy_types(self.failed_configs)
                json.dump(clean_failed_configs, f, indent=2)
            print(f"\n{len(self.failed_configs)} configs failed - see {failed_file.as_posix()}")

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for ML Classifiers')
    parser.add_argument('--classifier', choices=['pd_signal', 'cwt_image'], default='pd_signal',
                       help='Type of classifier to optimize (default: pd_signal)')
    parser.add_argument('--multi-channel', action='store_true',
                       help='Use multi-channel configuration for CWT classifier (uses hardcoded paths from config.py)')
    parser.add_argument('--base', type=int, default=None,
                       help='Use configuration from specific version as base (e.g., --base 115 for v115)')
    parser.add_argument('--mode', choices=['test', 'smart', 'channel-ablation'], default='smart',
                       help='Optimization mode (default: smart). Modes: test=quick validation, smart=registry-based adaptive search, channel-ablation=multi-channel analysis study (CWT only)')
    parser.add_argument('--max_configs', type=int, default=None,
                       help='Maximum number of configurations to test')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (defaults to config PD_HYPEROPT_RESULTS_DIR)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous run')
    parser.add_argument('--verbose', action='store_true',
                       help='Show training output in real-time')
    parser.add_argument('--concise', action='store_true',
                       help='Show concise progress updates (one line per epoch)')
    parser.add_argument('--skip_deduplication', action='store_true',
                       help='Disable smart deduplication - allow retesting previous configurations')
    parser.add_argument('--search_radius', type=int, choices=[1, 2], default=1,
                       help='For smart mode: search radius around best config (±1 or ±2 values per parameter)')
    parser.add_argument('--grid_search', action='store_true',
                       help='For smart mode: do full grid search instead of OFAT')
    parser.add_argument('--ignore', nargs='+', default=[], metavar='PARAM',
                       help='For smart mode: parameters to ignore in search, using previous optimum value')
    parser.add_argument('--max_grid_size', type=int, default=100, metavar='N',
                       help='Maximum grid search size before parameter limiting kicks in')
    parser.add_argument('--category', nargs='+', choices=['training', 'regularization', 'architecture', 'training_control', 'augmentation', 'fixed'], default=None, metavar='CAT',
                       help='For smart mode: parameter categories to include (default: all non-fixed categories)')
    parser.add_argument('--priority', nargs='+', type=int, choices=[1, 2, 3, 4, 5], default=None, metavar='TIER',
                       help='For smart mode: priority tiers to include (1=highest, 5=lowest, default: all tiers)')
    
    # DoE-specific arguments
    parser.add_argument('--doe_design', type=str, choices=['factorial', 'response_surface', 'lhs', 'from_file'], default='factorial',
                       help='DoE design type for --mode doe (default: factorial)')
    parser.add_argument('--doe_file', type=str,
                       help='Path to CSV file containing DoE experiments (for --doe_design from_file)')
    parser.add_argument('--doe_phase', type=int, choices=[1, 2, 3],
                       help='DoE phase (1=screening, 2=optimization, 3=validation)')
    parser.add_argument('--doe_factors', type=str, nargs='*',
                       help='Specific factors to include in DoE (default: auto-select based on analysis)')
    parser.add_argument('--auto_analyze', action='store_true',
                       help='Automatically run comprehensive analysis after DoE completion')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.multi_channel and args.classifier != 'cwt_image':
        parser.error("--multi-channel can only be used with --classifier cwt_image")
    
    # Channel-ablation mode requires multi-channel configuration
    if args.mode == 'channel-ablation':
        if args.classifier != 'cwt_image':
            parser.error("--mode channel-ablation can only be used with --classifier cwt_image")
        args.multi_channel = True  # Automatically enable multi-channel mode
    
    # Initialize tuner with classifier type (output_root will be set when run_optimization is called)
    tuner = HyperparameterTuner(
        classifier_type=args.classifier,
        output_root=args.output_dir, 
        verbose=args.verbose, 
        concise=args.concise,
        multi_channel=args.multi_channel,
        base_version=args.base
    )
    
    # Set DoE-specific parameters as instance variables for access in DoE mode
    if args.mode == 'doe':
        tuner._doe_design = args.doe_design
        tuner._doe_phase = args.doe_phase
        tuner._doe_factors = args.doe_factors
        tuner._doe_file = args.doe_file
        tuner._auto_analyze = args.auto_analyze
    
    # Run optimization
    tuner.run_optimization(
        mode=args.mode,
        max_configs=args.max_configs,
        resume=args.resume,
        verbose=args.verbose,
        concise=args.concise,
        deduplication=not args.skip_deduplication,  # Default to True unless --skip_deduplication is used
        search_radius=args.search_radius,
        grid_search=args.grid_search,
        ignore_params=args.ignore,
        max_grid_size=args.max_grid_size,
        categories=args.category,
        priority_tiers=args.priority
    )
    
    # Run comprehensive analysis after DoE completion if requested
    if args.mode == 'doe' and getattr(args, 'auto_analyze', False):
        print("\n" + "="*60)
        print("🔬 RUNNING AUTOMATIC POST-DOE COMPREHENSIVE ANALYSIS")
        print("="*60)
        
        try:
            from comprehensive_hyperopt_analyzer import ComprehensiveHyperoptAnalyzer
            
            # Determine analysis mode based on classifier type
            analysis_mode = 'cwt' if args.classifier == 'cwt_image' else 'pd'
            
            # Initialize and run comprehensive analysis
            analyzer = ComprehensiveHyperoptAnalyzer(
                mode=analysis_mode,
                verbose=args.verbose
            )
            
            # Run complete analysis including interaction effects
            results = analyzer.run_complete_analysis()
            
            print(f"\n✅ Comprehensive analysis completed!")
            print(f"📊 Analysis results saved to: {analyzer.output_dir}")
            print(f"📈 Key findings:")
            
            if analyzer.anova_results:
                # Show top parameters
                sorted_params = sorted(analyzer.anova_results.items(), 
                                     key=lambda x: x[1]['normalized_importance'], reverse=True)
                top_param, top_result = sorted_params[0]
                print(f"   • Most important parameter: {top_param} ({top_result['normalized_importance']*100:.1f}%)")
                
                # Show interaction effects if any
                if hasattr(analyzer, 'interaction_results') and analyzer.interaction_results:
                    significant_interactions = sum(1 for result in analyzer.interaction_results.values() 
                                                 if result['significant'])
                    print(f"   • Significant interactions found: {significant_interactions}")
                    
                    if significant_interactions > 0:
                        print("   • Recommended next phase: Response Surface Methodology (RSM)")
                    else:
                        print("   • Recommended next phase: Factorial optimization on main effects")
                        
                print(f"   • DoE recommendations available in analysis report")
                
        except Exception as e:
            print(f"⚠️  Warning: Automatic analysis failed: {e}")
            print("   You can run analysis manually with:")
            print(f"   python ml/comprehensive_hyperopt_analyzer.py --mode {analysis_mode} --verbose")

if __name__ == "__main__":
    main()