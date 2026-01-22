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
            # Data augmentation parameters (NEW simplified system)
            'augment_probability', 'augment_strength', 'augment_methods'
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

    def __init__(self, multi_channel=False, channel=None):
        self.multi_channel = multi_channel
        self.channel = channel

    def get_parameter_space(self, categories=None, tiers=None):
        """Get hyperparameter search space from registry."""
        from hyperparameter_registry import get_search_space
        return get_search_space('cwt_image', categories=categories, tiers=tiers)

    def get_config_template(self):
        return get_cwt_config_template(multi_channel=self.multi_channel, channel=self.channel)
    
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
            # CWT-suitable augmentation parameters (NEW system)
            'augment_probability': row.get('augment_probability'),
            'augment_strength': row.get('augment_strength'),
            'augment_methods': tuner._safe_parse_list(row.get('augment_methods')),
            'augment_to_balance': row.get('augment_to_balance'),
            # CWT-suitable augmentation parameters (OLD system - for backward compatibility)
            'augment_fraction': row.get('augment_fraction'),
            'time_shift_probability': row.get('time_shift_probability'),
            'time_shift_range': row.get('time_shift_range'),
            'noise_probability': row.get('noise_probability'),
            'noise_std': row.get('noise_std'),
            'brightness_probability': row.get('brightness_probability'),
            'brightness_range': row.get('brightness_range'),
            'contrast_probability': row.get('contrast_probability'),
            'contrast_range': row.get('contrast_range'),
            # CWT dataset-level parameters
            'cwt_mode': row.get('cwt_mode', 'full'),
            'coi_masking': row.get('coi_masking', False),
            'normalization_mode': row.get('normalization_mode', 'global')
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
            # Data augmentation parameters (NEW simplified system)
            'augment_probability', 'augment_strength', 'augment_methods', 'augment_to_balance'
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

def get_classifier_strategy(classifier_type, multi_channel=False, channel=None):
    """Factory function to create classifier strategies."""
    strategies = {
        'pd_signal': PDSignalStrategy(),
        'cwt_image': CWTImageStrategy(multi_channel=multi_channel, channel=channel)
    }
    
    if classifier_type not in strategies:
        raise ValueError(f"Unknown classifier type: {classifier_type}. Available: {list(strategies.keys())}")
    
    return strategies[classifier_type]

class HyperparameterTuner:
    """Manages hyperparameter optimization experiments with classifier-agnostic optimization."""
    
    def __init__(self, classifier_type='pd_signal', base_config=None, output_root=None, verbose=False, concise=False, multi_channel=False, channel=None, base_version=None, label_file=None, dataset_variant=None):
        # Initialize classifier strategy
        self.classifier_type = classifier_type
        self.multi_channel = multi_channel
        self.channel = channel
        self.verbose = verbose
        self.concise = concise
        self.label_file = label_file  # Prepared dataset label file
        self.dataset_variant = dataset_variant  # Pre-prepared dataset variant for k-fold CV
        self.base_version = base_version  # Store for later use in smart mode
        self.strategy = get_classifier_strategy(classifier_type, multi_channel=multi_channel, channel=channel)
        
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

        # Override channel configuration if multi-channel mode is explicitly enabled
        # This allows using single-channel base configs with --multi-channel flag
        if multi_channel and classifier_type == 'cwt_image':
            from config import CWT_DATA_DIR_DICT
            self.base_config['cwt_data_channels'] = CWT_DATA_DIR_DICT
            self.base_config['img_channels'] = len(CWT_DATA_DIR_DICT)
            self.base_config['cwt_data_dir'] = None  # Clear single-channel path
            if verbose:
                print(f"Multi-channel mode enabled: overriding base config to use {len(CWT_DATA_DIR_DICT)} channels")

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
        base['run_gradcam'] = False  # Disable Grad-CAM for test mode
        
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
    
    def generate_smart_configs(self, search_radius=1, grid_search=False, include_seed=False, ignore_params=None, max_grid_size=100, categories=None, priority_tiers=None, best_config='auto'):
        """
        Generate smart configurations using registry-based parameter filtering.

        Args:
            search_radius: Number of neighboring values to test (±1 or ±2)
            grid_search: If True, do full grid search within smart space; if False, use OFAT
            include_seed: If True, include seed/base config in search (default: False)
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
            # Include seed config first - deduplication will remove it if already tested
            # (unless --include_seed is specified, which adds it back later)
            seed_config = base_config.copy()
            configs.append(seed_config)

            for param_name in param_priority:
                if param_name not in smart_search_space:
                    continue

                current_value = best_config.get(param_name)
                # Get all neighboring values (including current) - let deduplication handle duplicates
                neighbors = smart_search_space[param_name]

                for neighbor_value in neighbors:
                    config = base_config.copy()
                    config[param_name] = neighbor_value
                    configs.append(config)

        if not self.concise:
            print(f"Generated {len(configs)} adaptive smart configurations")

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

    def _constrain_space_to_radius(self, base_config, search_radius):
        """
        Constrain parameter search space to a radius around base config.

        Similar to smart mode's neighbor generation, but returns a parameter space dict
        for DoE methods to use.

        Args:
            base_config: Base configuration to center search around
            search_radius: 1 (±1 level) or 2 (±2 levels)

        Returns:
            dict: Constrained parameter space with format {param_name: {'levels': [...]}}
        """
        from hyperparameter_registry import get_parameter_info, HYPERPARAMETER_REGISTRY

        constrained_space = {}

        for param_name, param_info in HYPERPARAMETER_REGISTRY.items():
            # Get classifier-specific parameter info
            param_info = get_parameter_info(param_name, self.classifier_type)
            if param_info is None:
                continue

            # Skip fixed parameters
            if param_info.get('category') == 'fixed':
                continue

            # Get current value from base config
            current_value = base_config.get(param_name)
            if current_value is None:
                # If not in base config, use full search space
                if 'search_space' in param_info:
                    constrained_space[param_name] = {
                        'type': param_info['type'],
                        'levels': param_info['search_space']
                    }
                continue

            # Get search space for this parameter
            search_space = param_info.get('search_space', [])
            if not search_space:
                continue

            # Find neighbors based on parameter type
            if param_info['type'] in ['categorical', 'discrete', 'continuous']:
                # For list-based search spaces, find index and get neighbors
                try:
                    current_index = search_space.index(current_value)

                    # Get neighbor indices
                    neighbor_indices = [current_index]
                    for offset in range(1, search_radius + 1):
                        if current_index - offset >= 0:
                            neighbor_indices.append(current_index - offset)
                        if current_index + offset < len(search_space):
                            neighbor_indices.append(current_index + offset)

                    # Get neighbor values
                    neighbor_values = [search_space[i] for i in sorted(neighbor_indices)]

                    constrained_space[param_name] = {
                        'type': param_info['type'],
                        'levels': neighbor_values
                    }

                except (ValueError, IndexError):
                    # Current value not in search space or error - use full space
                    constrained_space[param_name] = {
                        'type': param_info['type'],
                        'levels': search_space
                    }
            else:
                # Unknown type - use full search space
                constrained_space[param_name] = {
                    'type': param_info['type'],
                    'levels': search_space
                }

        return constrained_space

    def generate_doe_configs(self, doe_design='factorial', doe_factors=None,
                            search_radius=None, max_configs=None):
        """
        Generate configurations using Design of Experiments methods.

        Supports three design types:
        - factorial: Fractional factorial for screening (32-64 experiments)
        - response_surface: Central Composite Design for optimization (~20 experiments)
        - lhs: Latin Hypercube Sampling for exploration (user-specified N)

        Compatible with --base and --search_radius:
        - If base_config exists: constrains search to radius around base
        - Otherwise: uses full parameter space from registry

        Args:
            doe_design: 'factorial', 'response_surface', or 'lhs'
            doe_factors: Specific parameters to include (None = auto-select)
            search_radius: 1 (±1 level) or 2 (±2 levels), None = full space
            max_configs: For LHS design, number of experiments to generate (default 50)

        Returns:
            list: List of configuration dictionaries
        """
        from generate_doe_experiments import DoEGenerator
        from hyperparameter_registry import get_search_space

        if not self.concise:
            print(f"Generating DoE configurations: design={doe_design}")
            if self.base_config and search_radius:
                print(f"Using base config with search_radius={search_radius}")

        # Determine parameter space
        if self.base_config and search_radius:
            # Constrain to radius around base config
            constrained_space = self._constrain_space_to_radius(self.base_config, search_radius)
            parameter_space = {}

            # Convert to DoEGenerator format
            for param_name, param_data in constrained_space.items():
                parameter_space[param_name] = {
                    'type': param_data['type'],
                    'levels': param_data['levels'],
                    'range': (param_data['levels'][0], param_data['levels'][-1])
                        if param_data['type'] in ['continuous', 'discrete'] else None
                }

            base_for_doe = self.base_config
        else:
            # Use full search space from registry
            # DoEGenerator expects specific format - we'll override its internal space
            parameter_space = None  # Will use DoEGenerator's default
            base_for_doe = self.base_config if self.base_config else {}

        # Initialize DoE generator
        mode = 'cwt' if self.classifier_type == 'cwt_image' else 'pd'
        doe_gen = DoEGenerator(mode=mode, output_dir=None, verbose=(not self.concise))

        # Override parameter space if constrained
        if parameter_space:
            doe_gen.parameter_space = parameter_space

        # Filter factors if specified
        if doe_factors:
            # Validate factors exist in parameter space
            valid_factors = [f for f in doe_factors if f in doe_gen.parameter_space]
            if len(valid_factors) < len(doe_factors):
                invalid = set(doe_factors) - set(valid_factors)
                print(f"Warning: Ignoring invalid factors: {invalid}")
            factors = valid_factors if valid_factors else None
        else:
            factors = None

        # Generate experiments based on design type
        if doe_design == 'factorial':
            df = doe_gen.generate_factorial_design(
                factors=factors,
                center_points=4,
                randomize=True
            )
        elif doe_design == 'response_surface':
            df = doe_gen.generate_response_surface_design(
                factors=factors,
                alpha=None,  # Auto-calculate for rotatability
                center_points=6
            )
        elif doe_design == 'lhs':
            # For LHS, use max_configs to specify number of experiments (default 50)
            n_experiments = max_configs if max_configs else 50
            df = doe_gen.generate_lhs_design(
                n_experiments=n_experiments,
                factors=factors
            )
        else:
            raise ValueError(f"Unknown DoE design: {doe_design}")

        # Convert DataFrame to config dictionaries
        configs = []
        for i, row in df.iterrows():
            config = base_for_doe.copy()

            # Update with DoE values (skip metadata columns)
            for col, value in row.items():
                if col not in ['experiment_id', 'design_type']:
                    # Handle string representations of lists
                    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                        try:
                            config[col] = eval(value)
                        except:
                            config[col] = value
                    else:
                        config[col] = value

            # Add DoE metadata
            config['doe_design'] = doe_design
            config['doe_experiment_id'] = row.get('experiment_id', f'doe_{i+1:03d}')

            configs.append(config)

        if not self.concise:
            print(f"Generated {len(configs)} DoE configurations")
            if doe_factors:
                print(f"Factors: {doe_factors}")

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

        return total_params

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

        # Add dataset variant if specified (for k-fold CV datasets)
        if self.dataset_variant:
            cmd.extend(['--dataset_variant', str(self.dataset_variant)])

        # Add label file if specified (for prepared datasets)
        if self.label_file:
            cmd.extend(['--label_file', str(self.label_file)])
            cmd.extend(['--label_column', 'has_porosity'])  # Standard column for binary classification

        # Add label arguments if specified (for CSV-based labeling)
        if 'label_file' in config and config['label_file']:
            cmd.extend(['--label_file', str(config['label_file'])])
        if 'label_column' in config and config['label_column']:
            cmd.extend(['--label_column', str(config['label_column'])])
        if 'label_type' in config and config['label_type']:
            cmd.extend(['--label_type', str(config['label_type'])])

        # Add concise flag if needed
        if concise:
            cmd.append('--concise')
        
        try:
            # Set up environment for subprocess
            env = os.environ.copy()

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

        from hyperparameter_registry import HYPERPARAMETER_REGISTRY, get_parameter_info

        analysis = {
            'total_configs': len(configs),
            'parameters': {}  # Will store all non-fixed parameters
        }

        # Collect all non-fixed parameters that vary across configs
        for param_name in HYPERPARAMETER_REGISTRY:
            param_info = get_parameter_info(param_name, self.classifier_type)
            if param_info is None:
                continue

            # Skip fixed parameters
            if param_info.get('category') == 'fixed':
                continue

            # Check if this parameter exists in configs and varies
            if param_name in configs[0]:
                values = set()
                for config in configs:
                    value = config.get(param_name)
                    if value is not None:
                        # Handle list/tuple values
                        if isinstance(value, (list, tuple)):
                            values.add(tuple(value))
                        else:
                            values.add(value)

                # Only include if there are multiple values or at least one value
                if values:
                    analysis['parameters'][param_name] = sorted(list(values))

        # Calculate total training time estimate
        total_time_minutes = 0
        for config in configs:
            # Calculate real model complexity for accurate estimation
            complexity = self.calculate_model_complexity(config)
            time_estimate = self.timing_estimator.estimate_time(config, real_complexity=complexity)
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
                    # Calculate real model complexity for accurate estimation
                    complexity = self.calculate_model_complexity(config)
                    time_estimate = self.timing_estimator.estimate_time(config, real_complexity=complexity)
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
        
        print(f"\nHYPERPARAMETER RANGES (all non-fixed parameters):")

        # Group parameters by category for better organization
        from hyperparameter_registry import get_parameter_info

        params_by_category = {}
        for param_name, param_values in sorted(analysis['parameters'].items()):
            param_info = get_parameter_info(param_name, self.classifier_type)
            if param_info:
                category = param_info.get('category', 'other')
                if category not in params_by_category:
                    params_by_category[category] = []
                params_by_category[category].append((param_name, param_values))

        # Display by category
        category_order = ['training', 'regularization', 'architecture', 'augmentation', 'training_control']
        for category in category_order:
            if category in params_by_category:
                category_title = category.replace('_', ' ').title()
                print(f"\n   {category_title}:")
                for param_name, param_values in params_by_category[category]:
                    # Format parameter name nicely
                    param_display = param_name.replace('_', ' ').title()

                    # Format values based on type
                    if all(isinstance(v, (list, tuple)) for v in param_values):
                        # List values (e.g., architectures)
                        values_str = ', '.join([str(list(v)) for v in param_values])
                    else:
                        values_str = str(param_values)

                    print(f"      {param_display}: {values_str}")

        # Show any remaining categories not in the standard order
        for category, params in params_by_category.items():
            if category not in category_order:
                category_title = category.replace('_', ' ').title()
                print(f"\n   {category_title}:")
                for param_name, param_values in params:
                    param_display = param_name.replace('_', ' ').title()
                    if all(isinstance(v, (list, tuple)) for v in param_values):
                        values_str = ', '.join([str(list(v)) for v in param_values])
                    else:
                        values_str = str(param_values)
                    print(f"      {param_display}: {values_str}")
        
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
    
    def run_optimization(self, mode='smart', max_configs=None, resume=False, verbose=False, concise=False, deduplication=True, search_radius=1, grid_search=False, include_seed=False, ignore_params=None, max_grid_size=100, categories=None, priority_tiers=None):
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
            include_seed: If True, include seed/base config in smart mode search (default: False)
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
            # Determine base config for smart mode: --base takes precedence over auto-best
            if self.base_version is not None:
                # User explicitly specified --base, use that config
                best_config = self.base_config.copy()
                best_config['version'] = f'v{self.base_version:03d}'  # Add version for display
                if self.verbose:
                    print(f"Using --base {self.base_version} as origin for smart search")
            else:
                # Auto-find best config by accuracy
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
                include_seed=include_seed,
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
        elif mode == 'doe':
            # Design of Experiments mode - statistically efficient parameter exploration
            configs = self.generate_doe_configs(
                doe_design=getattr(self, '_doe_design', 'factorial'),
                doe_factors=getattr(self, '_doe_factors', None),
                search_radius=search_radius,
                max_configs=max_configs
            )
        elif mode == 'test':
            configs = self.generate_test_configs()
        else:
            available_modes = ['smart', 'doe', 'channel-ablation', 'test']
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

            # Special case: if --include_seed was specified in smart mode, never skip the seed config (index 0)
            is_seed_config = (mode == 'smart' and config_index == 0)
            skip_seed_override = is_seed_config and include_seed

            # Check if this config should be skipped
            should_skip = (config_index in skip_from_deduplication or
                          config_index in skip_from_max_configs or
                          i in completed_ids)

            # Apply skip decision (unless overridden by --include_seed for seed config)
            if should_skip and not skip_seed_override:
                if not self.concise:
                    if i in completed_ids:
                        skip_reason = "already completed"
                    elif config_index in skip_from_deduplication:
                        skip_reason = "already tested in a previous run"
                    else:  # config_index in skip_from_max_configs
                        skip_reason = "beyond max_configs limit"
                    print(f"Skipping configuration {i} ({skip_reason})")
                continue

            # If we reach here, config will be executed
            if skip_seed_override and not self.concise:
                print(f"Config {i}: Re-running seed config (--include_seed specified)")
            
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
    parser = argparse.ArgumentParser(
        description='Hyperparameter Optimization for ML Classifiers',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Core arguments
    parser.add_argument('--classifier', choices=['pd_signal', 'cwt_image'], default='pd_signal',
                       help='Type of classifier to optimize. Options: pd_signal, cwt_image (default: pd_signal)')
    parser.add_argument('--multi-channel', action='store_true',
                       help='Use multi-channel configuration for CWT classifier (uses hardcoded paths from config.py)')
    parser.add_argument('--channel', type=str, default=None, metavar='CHANNEL',
                       help='For CWT classifier: specify single channel to use (e.g., "PD1_cmor1.5-1.0", "PD1_mexh"). Mutually exclusive with --multi-channel. Available channels defined in config.py CWT_DATA_DIR_DICT. If not specified, uses default_channel from config.py.')
    parser.add_argument('--base', type=int, default=None, metavar='VERSION',
                       help='Use configuration from specific version as base (e.g., --base 115 for v115)')
    parser.add_argument('--mode', choices=['test', 'smart', 'doe', 'channel-ablation'], default='smart',
                       help='Optimization mode. Options: test (quick validation with 2 configs), smart (registry-based adaptive search), doe (Design of Experiments for efficient parameter space exploration), channel-ablation (multi-channel analysis, CWT only). Default: smart')
    parser.add_argument('--max_configs', type=int, default=None, metavar='N',
                       help='Maximum number of configurations to test (default: no limit)')
    parser.add_argument('--output_dir', type=str, default=None, metavar='DIR',
                       help='Output directory for results (default: from config PD_HYPEROPT_RESULTS_DIR or CWT_HYPEROPT_RESULTS_DIR)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous run in the same output directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Show training output in real-time (detailed mode)')
    parser.add_argument('--concise', action='store_true',
                       help='Show concise progress updates (one line per epoch)')
    parser.add_argument('--skip_deduplication', action='store_true',
                       help='Disable smart deduplication - allow retesting previous configurations')

    # Smart mode arguments
    parser.add_argument('--search_radius', type=int, choices=[1, 2], default=1, metavar='R',
                       help='For smart mode: search radius around best config. Options: 1 (±1 value per parameter), 2 (±2 values per parameter). Default: 1')
    parser.add_argument('--grid_search', action='store_true',
                       help='For smart mode: do full grid search instead of OFAT (One Factor At a Time)')
    parser.add_argument('--include_seed', action='store_true',
                       help='For smart mode: include seed/base config in search (re-run the base/best config even if already tested)')
    parser.add_argument('--ignore', nargs='+', default=[], metavar='PARAM',
                       help='For smart mode: parameters to ignore in search, using previous optimum value (e.g., --ignore batch_size learning_rate)')
    parser.add_argument('--max_grid_size', type=int, default=100, metavar='N',
                       help='Maximum grid search size before parameter limiting kicks in (default: 100)')
    parser.add_argument('--category', nargs='+', choices=['training', 'regularization', 'architecture', 'training_control', 'augmentation', 'fixed'], default=None, metavar='CAT',
                       help='For smart mode: parameter categories to include. Options: training, regularization, architecture, training_control, augmentation, fixed. Default: all non-fixed categories. Example: --category training regularization')
    parser.add_argument('--priority', nargs='+', type=int, choices=[1, 2, 3, 4, 5], default=None, metavar='TIER',
                       help='For smart mode: priority tiers to include. Options: 1 (highest), 2, 3, 4, 5 (lowest). Default: all tiers. Example: --priority 1 2')
    # Label file argument (for prepared datasets)
    parser.add_argument('--label-file', type=str,
                       help='Path to prepared label CSV file from prepare_training_dataset.py '
                            '(e.g., 1.0ms-window_prepared_AlSi10Mg_L1_threshold10_ratio1.5.csv)')

    # Dataset variant argument (for k-fold CV datasets)
    parser.add_argument('--dataset-variant', type=str,
                       help='Name of pre-prepared dataset variant for k-fold CV '
                            '(e.g., baseline_5fold). Mutually exclusive with --label-file.')

    # DoE-specific arguments
    parser.add_argument('--doe_design', type=str, choices=['factorial', 'response_surface', 'lhs'], default='factorial', metavar='DESIGN',
                       help='DoE design type for --mode doe. Options: factorial, response_surface, lhs. Default: factorial')
    parser.add_argument('--doe_factors', type=str, nargs='*', metavar='FACTOR',
                       help='Specific factors to include in DoE (default: auto-select based on analysis). Example: --doe_factors learning_rate batch_size')
    parser.add_argument('--auto_analyze', action='store_true',
                       help='Automatically run comprehensive analysis after DoE completion')
    
    args = parser.parse_args()

    # Validate arguments
    if args.multi_channel and args.classifier != 'cwt_image':
        parser.error("--multi-channel can only be used with --classifier cwt_image")

    if args.channel and args.classifier != 'cwt_image':
        parser.error("--channel can only be used with --classifier cwt_image")

    if args.channel and args.multi_channel:
        parser.error("--channel and --multi-channel are mutually exclusive")

    # Validate channel name if specified
    if args.channel:
        from config import CWT_DATA_DIR_DICT
        if args.channel not in CWT_DATA_DIR_DICT:
            available_channels = ', '.join(CWT_DATA_DIR_DICT.keys())
            parser.error(f"Unknown channel '{args.channel}'. Available channels: {available_channels}")

    # Channel-ablation mode requires multi-channel configuration
    if args.mode == 'channel-ablation':
        if args.classifier != 'cwt_image':
            parser.error("--mode channel-ablation can only be used with --classifier cwt_image")
        args.multi_channel = True  # Automatically enable multi-channel mode
    
    # Validate mutually exclusive arguments
    if hasattr(args, 'label_file') and args.label_file and hasattr(args, 'dataset_variant') and args.dataset_variant:
        parser.error("--label-file and --dataset-variant are mutually exclusive. Use one or the other.")

    # Initialize tuner with classifier type (output_root will be set when run_optimization is called)
    tuner = HyperparameterTuner(
        classifier_type=args.classifier,
        output_root=args.output_dir,
        verbose=args.verbose,
        concise=args.concise,
        multi_channel=args.multi_channel,
        channel=args.channel,
        base_version=args.base,
        label_file=args.label_file if hasattr(args, 'label_file') else None,
        dataset_variant=args.dataset_variant if hasattr(args, 'dataset_variant') else None
    )
    
    # Set DoE-specific parameters as instance variables for access in DoE mode
    if args.mode == 'doe':
        tuner._doe_design = args.doe_design
        tuner._doe_factors = args.doe_factors
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
        include_seed=args.include_seed,
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
