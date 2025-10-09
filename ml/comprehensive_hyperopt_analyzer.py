#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Optimization Analysis Tool

This script combines the best features from multiple analysis tools to provide
a complete analysis of hyperparameter optimization results with robust data
validation and comprehensive visualizations.

Key Features:
- OFAT (One-Factor-At-A-Time) sensitivity analysis 
- Full ANOVA decomposition with variance explained
- Parameter importance ranking
- Optimization timeline analysis
- Performance distributions
- Top configuration comparison
- Robust data validation

Author: AI Assistant (combining best features from existing tools)
"""

# Ensure UTF-8 encoding for all I/O operations
import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import argparse
import ast
import json
import warnings
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import griddata
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

# Local imports - will be conditionally aliased based on mode
import config

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class ComprehensiveHyperoptAnalyzer:
    """
    Comprehensive analysis tool for hyperparameter optimization results.
    Combines OFAT analysis, ANOVA decomposition, and visualization tools.
    """
    
    def __init__(self, mode='pd', log_path=None, output_dir=None, target_metric='mean_val_accuracy', 
                 min_experiments=3, verbose=False):
        # Validate mode
        if mode not in ['pd', 'cwt']:
            raise ValueError(f"Mode must be 'pd' or 'cwt', got: {mode}")
        
        self.mode = mode
        
        # Set up conditional imports based on mode
        if mode == 'cwt':
            # CWT mode - use CWT-specific functions
            get_experiment_log_path = config.get_cwt_experiment_log_path
            OUTPUTS_DIR = config.CWT_LOGS_DIR
        else:
            # PD mode - use PD-specific functions  
            get_experiment_log_path = config.get_pd_experiment_log_path
            OUTPUTS_DIR = config.PD_LOGS_DIR
        
        # Set paths using the aliased functions
        self.log_path = Path(log_path) if log_path else get_experiment_log_path()
        self.output_dir = Path(output_dir) if output_dir else OUTPUTS_DIR / 'comprehensive_analysis'
        
        self.target_metric = target_metric
        self.min_experiments = min_experiments
        self.verbose = verbose
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.df = None
        self.X_processed = None
        self.y = None
        self.parameter_names = []
        self.encoders = {}
        
        # Analysis results
        self.ofat_results = {}
        self.anova_results = {}
        self.interaction_results = {}
        self.sensitivity_indices = {}
        self.performance_stats = {}
        
        # Load and validate data
        self.load_and_validate_data()
        
    def load_and_validate_data(self):
        """Load experiment data with comprehensive validation."""
        if not self.log_path.exists():
            raise FileNotFoundError(f"Experiment log not found: {self.log_path}")
        
        if self.verbose:
            print(f"Loading experiment data from: {self.log_path}")
        
        # Load data with error handling for malformed CSV lines
        try:
            self.df = pd.read_csv(self.log_path, encoding='utf-8')
        except pd.errors.ParserError as e:
            print(f"CSV parsing error: {e}")
            print("Attempting to load with error handling...")
            # Try with error handling for malformed lines
            try:
                self.df = pd.read_csv(self.log_path, encoding='utf-8', error_bad_lines=False, warn_bad_lines=True)
            except TypeError:
                # For newer pandas versions
                self.df = pd.read_csv(self.log_path, encoding='utf-8', on_bad_lines='skip')
            print(f"Successfully loaded with {len(self.df)} valid lines")
        initial_count = len(self.df)
        
        # Data validation and cleaning
        if self.verbose:
            print(f"Initial experiments loaded: {initial_count}")
        
        # Check for target metric
        if self.target_metric not in self.df.columns:
            available_metrics = [col for col in self.df.columns if 'accuracy' in col.lower() or 'loss' in col.lower()]
            if available_metrics:
                self.target_metric = available_metrics[0]
                print(f"Warning: Target metric not found, using {self.target_metric}")
            else:
                raise ValueError(f"Target metric {self.target_metric} not found in data")
        
        # Filter valid experiments
        self.df = self.df[self.df[self.target_metric].notna()].copy()
        self.df = self.df.reset_index(drop=True)
        
        # Remove experiments with invalid metric values
        initial_valid = len(self.df)
        self.df = self.df[(self.df[self.target_metric] >= 0) & (self.df[self.target_metric] <= 1)].copy()
        self.df = self.df.reset_index(drop=True)
        
        if len(self.df) < self.min_experiments * 2:
            raise ValueError(f"Insufficient valid experiments: {len(self.df)} (need at least {self.min_experiments * 2})")
        
        # Remove experiments with suspicious patterns (potential outliers)
        q1 = self.df[self.target_metric].quantile(0.25)
        q3 = self.df[self.target_metric].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_count = len(self.df[(self.df[self.target_metric] < lower_bound) | 
                                   (self.df[self.target_metric] > upper_bound)])
        
        # Only remove extreme outliers (beyond 3 IQR)
        extreme_lower = q1 - 3 * iqr
        extreme_upper = q3 + 3 * iqr
        self.df = self.df[(self.df[self.target_metric] >= extreme_lower) & 
                         (self.df[self.target_metric] <= extreme_upper)].copy()
        
        # Reset index after filtering
        self.df = self.df.reset_index(drop=True)
        
        if self.verbose:
            print(f"Experiments after validation: {len(self.df)}")
            print(f"Removed {initial_count - len(self.df)} invalid/extreme experiments")
            print(f"Detected {outlier_count} potential outliers (not removed)")
            print(f"Target metric range: {self.df[self.target_metric].min():.4f} - {self.df[self.target_metric].max():.4f}")
        
        # Identify available parameters
        self.identify_analysis_parameters()
        
        # Preprocess parameters (must happen after data validation)
        self.preprocess_parameters()
        
        if self.verbose:
            print(f"Final dataset: {len(self.df)} experiments, {len(self.parameter_names)} parameters")
            print(f"X_processed shape: {self.X_processed.shape}, y shape: {self.y.shape}")
    
    def identify_analysis_parameters(self):
        """Identify parameters suitable for analysis."""
        # Core ML parameters (common to both PD and CWT)
        candidate_params = [
            'learning_rate', 'batch_size', 'epochs', 'k_folds',
            'conv_filters', 'dense_units', 'dropout_rates', 'l2_reg', 'l2_regularization',
            'batch_norm', 'use_batch_norm', 'optimizer', 'early_stopping_patience',
            'lr_reduction_patience', 'class_weights', 'use_class_weights',
            'conv_dropout', 'dense_dropout', 'lr_reduction_factor'
        ]
        
        # Add mode-specific parameters
        if self.mode == 'cwt':
            # CWT-specific parameters
            candidate_params.extend([
                'img_width', 'img_height', 'img_channels',
                'conv_kernel_size', 'pool_size', 'pool_layers',
                # CWT augmentation parameters
                'augment_fraction', 'time_shift_probability', 'time_shift_range',
                'noise_probability', 'noise_std', 'brightness_probability', 'brightness_range',
                'contrast_probability', 'contrast_range',
                # CWT analysis parameters
                'run_gradcam', 'gradcam_layer', 'save_gradcam_images', 'gradcam_threshold'
            ])
        else:
            # PD-specific parameters
            candidate_params.extend([
                'img_width',  # PD also has img_width
                # PD augmentation parameters
                'augment_fraction', 'time_shift_range', 'stretch_probability', 
                'stretch_scale', 'noise_probability', 'noise_std',
                'amplitude_scale_probability', 'amplitude_scale'
            ])
        
        # Find available parameters with sufficient variation
        available_params = []
        for param in candidate_params:
            if param in self.df.columns:
                # Check for variation (not all same value)
                unique_values = self.df[param].dropna().nunique()
                if unique_values > 1:  # Has variation
                    available_params.append(param)
                elif self.verbose:
                    print(f"Skipping {param}: no variation ({unique_values} unique values)")
        
        self.parameter_names = available_params
        
        if len(self.parameter_names) < 2:
            raise ValueError("Need at least 2 varying parameters for meaningful analysis")
        
        if self.verbose:
            print(f"Parameters for analysis: {self.parameter_names}")
    
    def preprocess_parameters(self):
        """Convert parameters to numerical format for analysis."""
        X = pd.DataFrame()
        
        for param in self.parameter_names:
            values = self.df[param].copy()
            
            # Handle different parameter types
            if param in ['learning_rate', 'l2_reg', 'l2_regularization', 'augment_fraction', 
                         'conv_dropout', 'dense_dropout', 'noise_std', 'brightness_range', 'contrast_range',
                         'time_shift_range', 'width_shift_range', 'lr_reduction_factor',
                         'time_shift_probability', 'noise_probability', 'brightness_probability', 'contrast_probability',
                         'stretch_probability', 'amplitude_scale_probability', 'stretch_scale', 'amplitude_scale']:
                # Numeric parameters (float)
                X[param] = pd.to_numeric(values, errors='coerce')
                
            elif param in ['batch_size', 'epochs', 'k_folds', 'early_stopping_patience', 'lr_reduction_patience',
                           'img_width', 'img_height', 'img_channels']:
                # Integer parameters
                X[param] = pd.to_numeric(values, errors='coerce')
                
            elif param in ['batch_norm', 'use_batch_norm', 'class_weights', 'use_class_weights']:
                # Boolean parameters
                bool_map = {True: 1, False: 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0}
                X[param] = values.map(bool_map).fillna(0)
                
            elif param == 'optimizer':
                # Categorical parameters
                le = LabelEncoder()
                values_str = values.fillna('unknown').astype(str).str.lower()
                X[param] = le.fit_transform(values_str)
                self.encoders[param] = le
                
            elif param in ['conv_filters', 'dense_units', 'dropout_rates', 'conv_kernel_size', 'pool_size', 'pool_layers']:
                # List parameters - convert to meaningful features
                features = self._process_list_parameter(values, param)
                X = pd.concat([X, features], axis=1)
                continue
                
            else:
                # Try numeric first, then categorical
                try:
                    X[param] = pd.to_numeric(values, errors='coerce')
                except:
                    le = LabelEncoder()
                    values_str = values.fillna('unknown').astype(str)
                    X[param] = le.fit_transform(values_str)
                    self.encoders[param] = le
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Update parameter names and store processed data
        self.parameter_names = list(X.columns)
        self.X_processed = X.values
        self.y = self.df[self.target_metric].values
        
        if self.verbose:
            print(f"Preprocessed features: {X.shape}")
            print(f"Features: {list(X.columns)}")
    
    def _process_list_parameter(self, values, param_name):
        """Convert list parameters to numerical features."""
        processed_df = pd.DataFrame()
        
        if param_name == 'conv_filters':
            features = {f'{param_name}_num_layers': [], f'{param_name}_total_filters': [], 
                       f'{param_name}_max_filters': [], f'{param_name}_progression': []}
            
            for val in values:
                try:
                    if isinstance(val, str):
                        filters = ast.literal_eval(val)
                    else:
                        filters = val if isinstance(val, list) else [16, 32, 64]
                    
                    features[f'{param_name}_num_layers'].append(len(filters))
                    features[f'{param_name}_total_filters'].append(sum(filters))
                    features[f'{param_name}_max_filters'].append(max(filters))
                    features[f'{param_name}_progression'].append(filters[-1] / filters[0] if len(filters) > 1 else 1.0)
                except:
                    # Defaults
                    features[f'{param_name}_num_layers'].append(3)
                    features[f'{param_name}_total_filters'].append(112)
                    features[f'{param_name}_max_filters'].append(64)
                    features[f'{param_name}_progression'].append(4.0)
            
            for feature_name, feature_values in features.items():
                processed_df[feature_name] = feature_values
                
        elif param_name == 'dense_units':
            features = {f'{param_name}_num_layers': [], f'{param_name}_total_units': [], 
                       f'{param_name}_max_units': []}
            
            for val in values:
                try:
                    if isinstance(val, str):
                        units = ast.literal_eval(val)
                    else:
                        units = val if isinstance(val, list) else [128, 64]
                    
                    features[f'{param_name}_num_layers'].append(len(units))
                    features[f'{param_name}_total_units'].append(sum(units))
                    features[f'{param_name}_max_units'].append(max(units))
                except:
                    features[f'{param_name}_num_layers'].append(2)
                    features[f'{param_name}_total_units'].append(192)
                    features[f'{param_name}_max_units'].append(128)
            
            for feature_name, feature_values in features.items():
                processed_df[feature_name] = feature_values
                
        elif param_name == 'dropout_rates':
            features = {f'{param_name}_first': [], f'{param_name}_mean': [], f'{param_name}_max': []}
            
            for val in values:
                try:
                    if isinstance(val, str):
                        rates = ast.literal_eval(val)
                        flat_rates = []
                        for rate in rates:
                            if isinstance(rate, list):
                                flat_rates.extend(rate)
                            else:
                                flat_rates.append(rate)
                        rates = [float(r) for r in flat_rates]
                    else:
                        rates = val if isinstance(val, list) else [0.2]
                    
                    features[f'{param_name}_first'].append(rates[0])
                    features[f'{param_name}_mean'].append(np.mean(rates))
                    features[f'{param_name}_max'].append(max(rates))
                except:
                    features[f'{param_name}_first'].append(0.2)
                    features[f'{param_name}_mean'].append(0.25)
                    features[f'{param_name}_max'].append(0.3)
            
            for feature_name, feature_values in features.items():
                processed_df[feature_name] = feature_values
                
        elif param_name == 'conv_kernel_size':
            features = {f'{param_name}_width': [], f'{param_name}_height': []}
            
            for val in values:
                try:
                    if isinstance(val, str):
                        kernel = ast.literal_eval(val)
                    else:
                        kernel = val if isinstance(val, list) else [3, 3]
                    
                    features[f'{param_name}_width'].append(kernel[0] if len(kernel) > 0 else 3)
                    features[f'{param_name}_height'].append(kernel[1] if len(kernel) > 1 else kernel[0] if len(kernel) > 0 else 3)
                except:
                    features[f'{param_name}_width'].append(3)
                    features[f'{param_name}_height'].append(3)
            
            for feature_name, feature_values in features.items():
                processed_df[feature_name] = feature_values
                
        elif param_name == 'pool_size':
            features = {f'{param_name}_width': [], f'{param_name}_height': []}
            
            for val in values:
                try:
                    if isinstance(val, str):
                        pool = ast.literal_eval(val)
                    else:
                        pool = val if isinstance(val, list) else [2, 2]
                    
                    features[f'{param_name}_width'].append(pool[0] if len(pool) > 0 else 2)
                    features[f'{param_name}_height'].append(pool[1] if len(pool) > 1 else pool[0] if len(pool) > 0 else 2)
                except:
                    features[f'{param_name}_width'].append(2)
                    features[f'{param_name}_height'].append(2)
            
            for feature_name, feature_values in features.items():
                processed_df[feature_name] = feature_values
                
        elif param_name == 'pool_layers':
            features = {f'{param_name}_count': [], f'{param_name}_first': [], f'{param_name}_last': []}
            
            for val in values:
                try:
                    if isinstance(val, str):
                        layers = ast.literal_eval(val)
                    else:
                        layers = val if isinstance(val, list) else [2, 5]
                    
                    features[f'{param_name}_count'].append(len(layers))
                    features[f'{param_name}_first'].append(layers[0] if len(layers) > 0 else 2)
                    features[f'{param_name}_last'].append(layers[-1] if len(layers) > 0 else 5)
                except:
                    features[f'{param_name}_count'].append(2)
                    features[f'{param_name}_first'].append(2)
                    features[f'{param_name}_last'].append(5)
            
            for feature_name, feature_values in features.items():
                processed_df[feature_name] = feature_values
        
        return processed_df
    
    def perform_ofat_analysis(self):
        """Perform One-Factor-At-A-Time sensitivity analysis."""
        if self.verbose:
            print("Performing OFAT sensitivity analysis...")
        
        # Find parameter variations where only one parameter changes
        variations = defaultdict(list)
        
        # Create normalized representations for comparison
        normalized_data = {}
        for i, param in enumerate(self.parameter_names):
            param_values = self.X_processed[:, i]
            normalized_data[param] = [str(val) for val in param_values]
        
        # Group experiments by parameter combinations
        for target_param in self.parameter_names:
            # Group by all other parameters (fixed)
            other_params = [p for p in self.parameter_names if p != target_param]
            
            if not other_params:
                continue
                
            experiment_groups = defaultdict(list)
            
            for idx in range(len(self.df)):
                # Create key from all other parameters
                other_values = tuple(normalized_data[p][idx] for p in other_params)
                experiment_groups[other_values].append(idx)
            
            # Find groups with sufficient variation in target parameter
            for group_key, indices in experiment_groups.items():
                if len(indices) >= self.min_experiments:
                    target_values = [normalized_data[target_param][idx] for idx in indices]
                    unique_targets = set(target_values)
                    
                    if len(unique_targets) >= self.min_experiments:
                        # This is a valid OFAT group
                        performance_values = [self.y[idx] for idx in indices]
                        target_numeric = [self.X_processed[idx, self.parameter_names.index(target_param)] for idx in indices]
                        
                        variations[target_param].append({
                            'group_key': group_key,
                            'indices': indices,
                            'target_values': target_numeric,
                            'performance': performance_values,
                            'performance_range': max(performance_values) - min(performance_values),
                            'n_experiments': len(indices)
                        })
        
        self.ofat_results = dict(variations)
        
        if self.verbose:
            total_groups = sum(len(groups) for groups in self.ofat_results.values())
            print(f"Found {total_groups} OFAT parameter groups across {len(self.ofat_results)} parameters")
    
    def perform_anova_analysis(self, n_bootstrap=500):
        """Perform comprehensive ANOVA decomposition with data validation."""
        if self.verbose:
            print("Performing ANOVA variance decomposition...")
        
        # Validate data quality for ANOVA
        if len(self.X_processed) < 20:
            print("Warning: Limited data for robust ANOVA analysis")
        
        # Main effects analysis using multiple methods
        main_effects = {}
        total_effects = {}
        
        # Method 1: Random Forest feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.X_processed, self.y)
        rf_importance = dict(zip(self.parameter_names, rf.feature_importances_))
        
        # Method 2: Statistical F-test
        try:
            f_stats, p_values = f_regression(self.X_processed, self.y)
            f_importance = {}
            for i, param in enumerate(self.parameter_names):
                # Convert F-statistic to approximate variance explained
                f_importance[param] = f_stats[i] / (f_stats[i] + len(self.y) - len(self.parameter_names))
        except:
            f_importance = {param: 0 for param in self.parameter_names}
        
        # Method 3: Bootstrap sensitivity analysis
        bootstrap_importance = {param: [] for param in self.parameter_names}
        
        for iteration in range(min(n_bootstrap, 200)):  # Limit for performance
            # Bootstrap sample
            indices = np.random.choice(len(self.X_processed), size=len(self.X_processed), replace=True)
            X_boot = self.X_processed[indices]
            y_boot = self.y[indices]
            
            # Fit surrogate model
            model = RandomForestRegressor(n_estimators=50, random_state=iteration)
            model.fit(X_boot, y_boot)
            
            # Calculate total variance
            y_pred_full = model.predict(X_boot)
            total_var = np.var(y_pred_full)
            
            if total_var > 1e-10:  # Avoid division by zero
                # Calculate main effects
                X_mean = np.mean(X_boot, axis=0)
                
                for i, param in enumerate(self.parameter_names):
                    # Create dataset with only parameter i varying
                    X_varied = np.tile(X_mean, (100, 1))
                    param_range = np.linspace(np.min(X_boot[:, i]), np.max(X_boot[:, i]), 100)
                    X_varied[:, i] = param_range
                    
                    y_pred_varied = model.predict(X_varied)
                    param_var = np.var(y_pred_varied)
                    
                    # Main effect as proportion of total variance
                    main_effect = param_var / total_var if total_var > 0 else 0
                    bootstrap_importance[param].append(min(main_effect, 1.0))  # Cap at 1.0
        
        # Combine methods with weights
        combined_importance = {}
        for param in self.parameter_names:
            rf_score = rf_importance.get(param, 0)
            f_score = f_importance.get(param, 0)
            
            bootstrap_scores = bootstrap_importance[param]
            bootstrap_score = np.mean(bootstrap_scores) if bootstrap_scores else 0
            bootstrap_std = np.std(bootstrap_scores) if len(bootstrap_scores) > 1 else 0
            
            # Weighted combination (RF: 40%, Bootstrap: 40%, F-test: 20%)
            combined_score = 0.4 * rf_score + 0.4 * bootstrap_score + 0.2 * f_score
            
            combined_importance[param] = {
                'importance': combined_score,
                'rf_importance': rf_score,
                'f_importance': f_score,
                'bootstrap_mean': bootstrap_score,
                'bootstrap_std': bootstrap_std,
                'bootstrap_samples': len(bootstrap_scores)
            }
        
        # Normalize to sum to 1
        total_importance = sum(result['importance'] for result in combined_importance.values())
        if total_importance > 0:
            for param in combined_importance:
                combined_importance[param]['normalized_importance'] = (
                    combined_importance[param]['importance'] / total_importance
                )
        
        self.anova_results = combined_importance
        
        # Calculate performance statistics
        self.performance_stats = {
            'mean_performance': np.mean(self.y),
            'std_performance': np.std(self.y),
            'min_performance': np.min(self.y),
            'max_performance': np.max(self.y),
            'performance_range': np.max(self.y) - np.min(self.y),
            'n_experiments': len(self.y)
        }
        
        if self.verbose:
            print("ANOVA analysis completed!")
            print(f"Performance range: {self.performance_stats['min_performance']:.4f} - {self.performance_stats['max_performance']:.4f}")
    
    def perform_interaction_analysis(self, max_interactions=10):
        """Perform pairwise interaction effects analysis."""
        if self.verbose:
            print("Performing interaction effects analysis...")
        
        if len(self.parameter_names) < 2:
            print("Need at least 2 parameters for interaction analysis")
            return
        
        # Focus on top parameters to limit computational cost
        if hasattr(self, 'anova_results') and self.anova_results:
            # Sort parameters by importance
            sorted_params = sorted(self.anova_results.items(), 
                                 key=lambda x: x[1]['normalized_importance'], reverse=True)
            top_params = [param for param, _ in sorted_params[:min(6, len(sorted_params))]]
        else:
            # Use first few parameters if ANOVA not available
            top_params = self.parameter_names[:min(6, len(self.parameter_names))]
        
        interaction_effects = {}
        
        # Analyze all pairwise combinations of top parameters
        param_pairs = list(combinations(top_params, 2))[:max_interactions]
        
        if self.verbose:
            print(f"Analyzing {len(param_pairs)} parameter interactions...")
        
        # Use Random Forest for interaction detection
        from sklearn.ensemble import RandomForestRegressor
        
        for param1, param2 in param_pairs:
            try:
                # Get parameter indices
                idx1 = self.parameter_names.index(param1)
                idx2 = self.parameter_names.index(param2)
                
                # Extract the two parameters and target
                X_pair = self.X_processed[:, [idx1, idx2]]
                y = self.y
                
                if len(np.unique(X_pair[:, 0])) < 2 or len(np.unique(X_pair[:, 1])) < 2:
                    continue  # Skip if no variation
                
                # Fit model with interaction term
                # Create interaction feature
                X_interaction = np.column_stack([
                    X_pair[:, 0],  # param1
                    X_pair[:, 1],  # param2
                    X_pair[:, 0] * X_pair[:, 1]  # interaction term
                ])
                
                # Fit full model (with interaction)
                rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_full.fit(X_interaction, y)
                full_score = rf_full.score(X_interaction, y)
                
                # Fit additive model (no interaction)
                rf_additive = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_additive.fit(X_pair, y)
                additive_score = rf_additive.score(X_pair, y)
                
                # Interaction strength = improvement from adding interaction term
                interaction_strength = max(0, full_score - additive_score)
                
                # Statistical significance test using permutation
                n_permutations = 50
                permuted_improvements = []
                
                for _ in range(n_permutations):
                    # Permute target to break any real relationship
                    y_perm = np.random.permutation(y)
                    
                    rf_full_perm = RandomForestRegressor(n_estimators=50, random_state=42)
                    rf_additive_perm = RandomForestRegressor(n_estimators=50, random_state=42)
                    
                    rf_full_perm.fit(X_interaction, y_perm)
                    rf_additive_perm.fit(X_pair, y_perm)
                    
                    full_perm = rf_full_perm.score(X_interaction, y_perm)
                    additive_perm = rf_additive_perm.score(X_pair, y_perm)
                    
                    permuted_improvements.append(max(0, full_perm - additive_perm))
                
                # Calculate p-value
                p_value = np.mean(np.array(permuted_improvements) >= interaction_strength)
                
                # Calculate correlation between parameters
                param_correlation = np.corrcoef(X_pair[:, 0], X_pair[:, 1])[0, 1]
                
                interaction_effects[f"{param1}_x_{param2}"] = {
                    'param1': param1,
                    'param2': param2,
                    'interaction_strength': interaction_strength,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'param_correlation': param_correlation,
                    'full_r2': full_score,
                    'additive_r2': additive_score,
                    'sample_size': len(y)
                }
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not analyze interaction {param1} x {param2}: {e}")
                continue
        
        # Sort interactions by strength
        self.interaction_results = dict(sorted(interaction_effects.items(), 
                                             key=lambda x: x[1]['interaction_strength'], reverse=True))
        
        if self.verbose:
            print(f"Interaction analysis completed! Found {len(self.interaction_results)} interactions")
            significant_interactions = sum(1 for result in self.interaction_results.values() if result['significant'])
            print(f"Significant interactions (p < 0.05): {significant_interactions}")
    
    def plot_ofat_analysis(self):
        """Create OFAT sensitivity plots."""
        if not self.ofat_results:
            return
        
        if self.verbose:
            print("Creating OFAT sensitivity plots...")
        
        # Calculate plot dimensions
        n_params = len(self.ofat_results)
        if n_params == 0:
            return
        
        cols = min(3, n_params)
        rows = (n_params + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        
        for param, groups in self.ofat_results.items():
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            
            # Combine all data points from all groups
            all_x = []
            all_y = []
            
            for group in groups:
                all_x.extend(group['target_values'])
                all_y.extend(group['performance'])
            
            if len(all_x) > 0:
                # Scatter plot
                ax.scatter(all_x, all_y, alpha=0.6, s=40)
                
                # Calculate correlation for numerical parameters
                if len(all_x) >= 3:
                    try:
                        correlation = np.corrcoef(all_x, all_y)[0, 1]
                        if abs(correlation) > 0.1:  # Only show meaningful correlations
                            ax.text(0.05, 0.95, f'r = {correlation:.3f}', 
                                   transform=ax.transAxes, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except:
                        pass
                
                ax.set_xlabel(param.replace('_', ' ').title())
                ax.set_ylabel(self.target_metric.replace('_', ' ').title())
                ax.set_title(f'OFAT: {param} ({len(all_x)} points)')
                ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'ofat_sensitivity_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"OFAT plots saved: {plot_path}")
    
    def plot_architecture_analysis(self):
        """Create dedicated architecture parameter analysis plots."""
        if self.verbose:
            print("Creating architecture analysis plots...")
        
        # Check if we have architecture parameters
        has_conv = 'conv_filters' in self.df.columns
        has_dense = 'dense_units' in self.df.columns
        
        if not (has_conv or has_dense):
            if self.verbose:
                print("No architecture parameters found, skipping architecture analysis")
            return
        
        # Determine subplot layout
        n_plots = sum([has_conv, has_dense])
        if n_plots == 0:
            return
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 6*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Conv Architecture Analysis
        if has_conv:
            ax = axes[plot_idx]
            plot_idx += 1
            
            # Group by conv architecture
            conv_data = []
            for _, row in self.df.iterrows():
                try:
                    conv_arch = str(ast.literal_eval(row['conv_filters']) if isinstance(row['conv_filters'], str) else row['conv_filters'])
                    performance = row[self.target_metric]
                    conv_data.append({'architecture': conv_arch, 'performance': performance})
                except:
                    continue
            
            if conv_data:
                conv_df = pd.DataFrame(conv_data)
                
                # Group and calculate statistics
                arch_stats = conv_df.groupby('architecture')['performance'].agg(['mean', 'std', 'count']).reset_index()
                arch_stats = arch_stats.sort_values('mean', ascending=False)
                
                # Choose plot type based on data
                if len(arch_stats) <= 8:  # Bar plot for few architectures
                    bars = ax.bar(range(len(arch_stats)), arch_stats['mean'], 
                                 yerr=arch_stats['std'], capsize=5, alpha=0.7)
                    ax.set_xticks(range(len(arch_stats)))
                    ax.set_xticklabels([arch[:20] + '...' if len(arch) > 20 else arch 
                                       for arch in arch_stats['architecture']], rotation=45)
                    
                    # Color bars by performance
                    max_perf = arch_stats['mean'].max()
                    min_perf = arch_stats['mean'].min()
                    if max_perf > min_perf:
                        for bar, perf in zip(bars, arch_stats['mean']):
                            bar.set_color(plt.cm.viridis((perf - min_perf) / (max_perf - min_perf)))
                    
                    # Add experiment counts
                    for i, (_, row) in enumerate(arch_stats.iterrows()):
                        ax.text(i, row['mean'] + (row['std'] if not np.isnan(row['std']) else 0) + 0.01, 
                               f"n={row['count']}", ha='center', fontsize=8)
                else:
                    # Box plot for many architectures
                    arch_order = arch_stats['architecture'].tolist()
                    conv_df['architecture'] = pd.Categorical(conv_df['architecture'], categories=arch_order)
                    conv_df = conv_df.sort_values('architecture')
                    
                    sns.boxplot(data=conv_df, x='architecture', y='performance', ax=ax)
                    ax.set_xticklabels([arch[:15] + '...' if len(arch) > 15 else arch 
                                       for arch in arch_order], rotation=45)
                
                ax.set_xlabel('Conv Filter Architecture')
                ax.set_ylabel(self.target_metric.replace('_', ' ').title())
                ax.set_title(f'Convolutional Architecture Performance ({len(conv_data)} experiments)')
                ax.grid(True, alpha=0.3)
        
        # Dense Architecture Analysis  
        if has_dense:
            ax = axes[plot_idx]
            
            # Group by dense architecture
            dense_data = []
            for _, row in self.df.iterrows():
                try:
                    dense_arch = str(ast.literal_eval(row['dense_units']) if isinstance(row['dense_units'], str) else row['dense_units'])
                    performance = row[self.target_metric]
                    dense_data.append({'architecture': dense_arch, 'performance': performance})
                except:
                    continue
            
            if dense_data:
                dense_df = pd.DataFrame(dense_data)
                
                # Group and calculate statistics
                arch_stats = dense_df.groupby('architecture')['performance'].agg(['mean', 'std', 'count']).reset_index()
                arch_stats = arch_stats.sort_values('mean', ascending=False)
                
                # Choose plot type based on data
                if len(arch_stats) <= 8:  # Bar plot for few architectures
                    bars = ax.bar(range(len(arch_stats)), arch_stats['mean'], 
                                 yerr=arch_stats['std'], capsize=5, alpha=0.7, color='orange')
                    ax.set_xticks(range(len(arch_stats)))
                    ax.set_xticklabels([arch[:20] + '...' if len(arch) > 20 else arch 
                                       for arch in arch_stats['architecture']], rotation=45)
                    
                    # Color bars by performance
                    max_perf = arch_stats['mean'].max()
                    min_perf = arch_stats['mean'].min()
                    if max_perf > min_perf:
                        for bar, perf in zip(bars, arch_stats['mean']):
                            bar.set_color(plt.cm.plasma((perf - min_perf) / (max_perf - min_perf)))
                    
                    # Add experiment counts
                    for i, (_, row) in enumerate(arch_stats.iterrows()):
                        ax.text(i, row['mean'] + (row['std'] if not np.isnan(row['std']) else 0) + 0.01, 
                               f"n={row['count']}", ha='center', fontsize=8)
                else:
                    # Box plot for many architectures
                    arch_order = arch_stats['architecture'].tolist()
                    dense_df['architecture'] = pd.Categorical(dense_df['architecture'], categories=arch_order)
                    dense_df = dense_df.sort_values('architecture')
                    
                    sns.boxplot(data=dense_df, x='architecture', y='performance', ax=ax)
                    ax.set_xticklabels([arch[:15] + '...' if len(arch) > 15 else arch 
                                       for arch in arch_order], rotation=45)
                
                ax.set_xlabel('Dense Units Architecture')
                ax.set_ylabel(self.target_metric.replace('_', ' ').title())
                ax.set_title(f'Dense Layer Architecture Performance ({len(dense_data)} experiments)')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'architecture_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Architecture plots saved: {plot_path}")
    
    def plot_anova_analysis(self):
        """Create ANOVA decomposition plots."""
        if not self.anova_results:
            return
        
        if self.verbose:
            print("Creating ANOVA analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Parameter importance ranking
        params = list(self.anova_results.keys())
        importance_values = [self.anova_results[p]['normalized_importance'] for p in params]
        importance_errors = [self.anova_results[p]['bootstrap_std'] for p in params]
        
        # Sort by importance
        sorted_data = sorted(zip(params, importance_values, importance_errors), 
                            key=lambda x: x[1], reverse=True)
        sorted_params, sorted_importance, sorted_errors = zip(*sorted_data)
        
        y_pos = np.arange(len(sorted_params))
        bars = axes[0, 0].barh(y_pos, sorted_importance, xerr=sorted_errors, capsize=3)
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels([p.replace('_', ' ').title() for p in sorted_params])
        axes[0, 0].set_xlabel('Normalized Importance')
        axes[0, 0].set_title('Parameter Importance Ranking')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Color bars by importance
        max_importance = max(sorted_importance)
        for bar, imp in zip(bars, sorted_importance):
            bar.set_color(plt.cm.viridis(imp / max_importance))
        
        # 2. Method comparison
        methods = ['RF Importance', 'Bootstrap Mean', 'F-test Score']
        method_keys = ['rf_importance', 'bootstrap_mean', 'f_importance']
        
        top_5_params = sorted_params[:5]
        x = np.arange(len(top_5_params))
        width = 0.25
        
        for i, (method, key) in enumerate(zip(methods, method_keys)):
            values = [self.anova_results[p][key] for p in top_5_params]
            axes[0, 1].bar(x + i*width, values, width, label=method, alpha=0.8)
        
        axes[0, 1].set_xlabel('Top Parameters')
        axes[0, 1].set_ylabel('Importance Score')
        axes[0, 1].set_title('Method Comparison (Top 5 Parameters)')
        axes[0, 1].set_xticks(x + width)
        axes[0, 1].set_xticklabels([p.replace('_', ' ')[:10] for p in top_5_params], rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Performance distribution
        axes[1, 0].hist(self.y, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(self.performance_stats['mean_performance'], color='red', 
                          linestyle='--', linewidth=2, 
                          label=f'Mean: {self.performance_stats["mean_performance"]:.4f}')
        axes[1, 0].set_xlabel(self.target_metric.replace('_', ' ').title())
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Performance Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Cumulative importance
        cumsum = np.cumsum(sorted_importance)
        axes[1, 1].plot(range(1, len(cumsum) + 1), cumsum, 'o-', linewidth=2, markersize=6)
        axes[1, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
        axes[1, 1].set_xlabel('Number of Top Parameters')
        axes[1, 1].set_ylabel('Cumulative Importance')
        axes[1, 1].set_title('Cumulative Parameter Importance')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'anova_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"ANOVA plots saved: {plot_path}")
    
    def plot_interaction_analysis(self):
        """Create interaction effects plots."""
        if not self.interaction_results:
            return
        
        if self.verbose:
            print("Creating interaction effects plots...")
        
        # Filter significant interactions for plotting
        significant_interactions = {k: v for k, v in self.interaction_results.items() if v['significant']}
        all_interactions = self.interaction_results
        
        if not all_interactions:
            if self.verbose:
                print("No interactions to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Interaction strength ranking
        interactions = list(all_interactions.keys())
        strengths = [all_interactions[k]['interaction_strength'] for k in interactions]
        p_values = [all_interactions[k]['p_value'] for k in interactions]
        
        # Color by significance
        colors = ['red' if p < 0.05 else 'blue' for p in p_values]
        
        y_pos = np.arange(len(interactions))
        bars = axes[0, 0].barh(y_pos, strengths, color=colors, alpha=0.7)
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels([k.replace('_x_', ' × ') for k in interactions])
        axes[0, 0].set_xlabel('Interaction Strength (R² improvement)')
        axes[0, 0].set_title('Parameter Interaction Effects')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add significance indicators
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            if p_val < 0.05:
                axes[0, 0].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                               f'p={p_val:.3f}', va='center', fontsize=8)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Significant (p < 0.05)'),
                          Patch(facecolor='blue', alpha=0.7, label='Not significant')]
        axes[0, 0].legend(handles=legend_elements, loc='lower right')
        
        # 2. Interaction strength vs significance
        axes[0, 1].scatter(strengths, [-np.log10(p) for p in p_values], 
                          c=colors, alpha=0.7, s=60)
        axes[0, 1].axhline(y=-np.log10(0.05), color='red', linestyle='--', 
                          alpha=0.7, label='p = 0.05 threshold')
        axes[0, 1].set_xlabel('Interaction Strength')
        axes[0, 1].set_ylabel('-log₁₀(p-value)')
        axes[0, 1].set_title('Interaction Strength vs Statistical Significance')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Add labels for significant interactions
        for i, (strength, p_val, interaction) in enumerate(zip(strengths, p_values, interactions)):
            if p_val < 0.05 and strength > np.median(strengths):
                axes[0, 1].annotate(interaction.replace('_x_', ' × '), 
                                  (strength, -np.log10(p_val)), 
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=8, alpha=0.8)
        
        # 3. Parameter correlation heatmap
        if len(all_interactions) > 0:
            # Extract unique parameters involved in interactions
            all_params = set()
            for result in all_interactions.values():
                all_params.add(result['param1'])
                all_params.add(result['param2'])
            all_params = sorted(list(all_params))
            
            # Create correlation matrix
            n_params = len(all_params)
            if n_params > 1:
                corr_matrix = np.eye(n_params)
                
                for result in all_interactions.values():
                    p1_idx = all_params.index(result['param1'])
                    p2_idx = all_params.index(result['param2'])
                    corr = result['param_correlation']
                    corr_matrix[p1_idx, p2_idx] = corr
                    corr_matrix[p2_idx, p1_idx] = corr
                
                im = axes[1, 0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[1, 0].set_xticks(range(n_params))
                axes[1, 0].set_yticks(range(n_params))
                axes[1, 0].set_xticklabels([p.replace('_', ' ') for p in all_params], rotation=45)
                axes[1, 0].set_yticklabels([p.replace('_', ' ') for p in all_params])
                axes[1, 0].set_title('Parameter Correlation Matrix')
                
                # Add correlation values to cells
                for i in range(n_params):
                    for j in range(n_params):
                        text = axes[1, 0].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                             ha="center", va="center", 
                                             color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                                             fontsize=8)
                
                plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
        
        # 4. Sample size distribution for interactions
        sample_sizes = [all_interactions[k]['sample_size'] for k in interactions]
        axes[1, 1].hist(sample_sizes, bins=min(10, len(set(sample_sizes))), alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=np.mean(sample_sizes), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(sample_sizes):.0f}')
        axes[1, 1].set_xlabel('Sample Size')
        axes[1, 1].set_ylabel('Number of Interactions')
        axes[1, 1].set_title('Sample Size Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'interaction_effects_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Interaction plots saved: {plot_path}")

    def plot_pairwise_surfaces(self, top_n=4):
        """Create pairwise surface plots for the top N most significant parameters."""
        if not self.anova_results:
            print("Warning: No ANOVA results available for surface plots")
            return
        
        if self.verbose:
            print("Creating pairwise parameter surface plots...")
        
        # Get top N parameters by importance
        sorted_params = sorted(self.anova_results.items(), 
                             key=lambda x: x[1]['normalized_importance'], reverse=True)
        top_params = [param for param, _ in sorted_params[:top_n]]
        
        if len(top_params) < 2:
            print(f"Warning: Only {len(top_params)} parameters available, need at least 2 for surface plots")
            return
        
        # Get parameter data for top parameters from processed data
        param_data = {}
        for param in top_params:
            if param in self.parameter_names:
                # Get data from processed feature matrix
                param_idx = self.parameter_names.index(param)
                param_data[param] = self.X_processed[:, param_idx]
            elif param in self.df.columns:
                # Fallback to original dataframe for unprocessed parameters
                param_data[param] = self.df[param].values
            else:
                # Handle processed parameter names (from feature engineering)
                processed_names = [col for col in self.df.columns if param in col]
                if processed_names:
                    param_data[param] = self.df[processed_names[0]].values
                else:
                    print(f"Warning: Parameter {param} not found in data")
                    continue
        
        # Filter to parameters we actually have data for
        available_params = list(param_data.keys())
        if len(available_params) < 2:
            print(f"Warning: Only {len(available_params)} parameters have data available")
            return
        
        # Create pairwise combinations
        pairs = list(combinations(available_params, 2))
        n_pairs = len(pairs)
        
        if n_pairs == 0:
            return
        
        # Set up subplot grid
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        # Ensure axes is always 2D array for consistent indexing
        if n_pairs == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        target_values = self.df[self.target_metric].values
        
        for idx, (param1, param2) in enumerate(pairs):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            x_data = param_data[param1]
            y_data = param_data[param2]
            
            # Create surface plot using griddata interpolation
            
            # Create grid for interpolation
            x_min, x_max = x_data.min(), x_data.max()
            y_min, y_max = y_data.min(), y_data.max()
            
            # Add small margins to avoid edge effects
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            
            xi = np.linspace(x_min - x_margin, x_max + x_margin, 50)
            yi = np.linspace(y_min - y_margin, y_max + y_margin, 50)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            
            # Interpolate surface
            try:
                zi = griddata((x_data, y_data), target_values, (xi_grid, yi_grid), method='linear') # 'linear', 'nearest' or 'cubic'
                
                # Create contour plot with surface
                # levels = np.linspace(target_values.min(), target_values.max(), 20)
                levels = np.linspace(0.5, 1.0, 11)
                contour = ax.contourf(xi_grid, yi_grid, zi, levels=levels, cmap='viridis', alpha=0.8)
                
                # Add contour lines
                contour_lines = ax.contour(xi_grid, yi_grid, zi, levels=10, colors='white', alpha=0.6, linewidths=0.5)
                
                # Overlay actual data points
                scatter = ax.scatter(x_data, y_data, c=target_values, cmap='viridis', 
                                   s=30, edgecolors='white', linewidths=0.5, alpha=0.9)
                
                # Add colorbar
                cbar = plt.colorbar(contour, ax=ax, shrink=0.8, ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], extend='min')
                cbar.set_label(self.target_metric.replace('_', ' ').title(), rotation=270, labelpad=15)
                
                # Find and mark best point
                best_idx = np.argmax(target_values)
                ax.scatter(x_data[best_idx], y_data[best_idx], c='red', s=100, 
                          marker='*', edgecolors='white', linewidths=1, zorder=10,
                          label=f'Best: {target_values[best_idx]:.4f}')
                
                # Formatting
                ax.set_xlabel(param1.replace('_', ' ').title())
                ax.set_ylabel(param2.replace('_', ' ').title())
                ax.set_title(f'{param1} vs {param2}\nSurface Plot')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                # Fallback to simple scatter plot if interpolation fails
                print(f"Warning: Surface interpolation failed for {param1} vs {param2}: {e}")
                scatter = ax.scatter(x_data, y_data, c=target_values, cmap='viridis', s=50)
                ax.set_xlabel(param1.replace('_', ' ').title())
                ax.set_ylabel(param2.replace('_', ' ').title())
                ax.set_title(f'{param1} vs {param2}\nScatter Plot')
                plt.colorbar(scatter, ax=ax, shrink=0.8)
        
        # Hide empty subplots
        if n_pairs < n_rows * n_cols:
            for idx in range(n_pairs, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'pairwise_surface_plots_top{top_n}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Pairwise surface plots saved: {plot_path}")
            print(f"Generated {n_pairs} surface plots for parameters: {', '.join(available_params)}")
    
    def _analyze_parameter_coverage(self):
        """Analyze current parameter coverage and provide DoE recommendations."""
        coverage_analysis = {
            'low_sample_params': [],
            'balanced_params': [],
            'recommended_sample_size': 0,
            'current_coverage_score': 0.0
        }
        
        # Analyze each parameter's coverage
        for param in self.parameter_names:
            param_idx = self.parameter_names.index(param)
            unique_values = len(np.unique(self.X_processed[:, param_idx]))
            
            if unique_values < 3:
                coverage_analysis['low_sample_params'].append((param, unique_values))
            elif unique_values >= 5:
                coverage_analysis['balanced_params'].append((param, unique_values))
        
        # Calculate recommended sample size based on DoE best practices
        n_params = len(self.parameter_names)
        
        # Base sample size calculation
        if hasattr(self, 'interaction_results') and self.interaction_results:
            significant_interactions = sum(1 for result in self.interaction_results.values() if result['significant'])
            if significant_interactions > 0:
                # RSM design: 2^k + 2k + center points
                base_size = 2**min(n_params, 5) + 2*n_params + 5  # Limit factorial for large k
                coverage_analysis['recommended_sample_size'] = min(base_size, 200)  # Cap at reasonable size
            else:
                # Fractional factorial: 2^(k-1) with center points
                base_size = 2**(min(n_params-1, 6)) + 5
                coverage_analysis['recommended_sample_size'] = min(base_size, 150)
        else:
            # LHS: 5-10 times the number of parameters
            coverage_analysis['recommended_sample_size'] = min(10 * n_params, 100)
        
        # Calculate current coverage score
        well_sampled = len(coverage_analysis['balanced_params'])
        total_params = len(self.parameter_names)
        coverage_analysis['current_coverage_score'] = well_sampled / total_params if total_params > 0 else 0.0
        
        return coverage_analysis
    
    def plot_optimization_timeline(self):
        """Plot optimization progress over time."""
        if 'timestamp' not in self.df.columns:
            return
        
        if self.verbose:
            print("Creating optimization timeline...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Sort by timestamp
        df_sorted = self.df.copy()
        try:
            # Try multiple datetime formats
            df_sorted['datetime'] = pd.to_datetime(df_sorted['timestamp'], format='mixed')
            df_sorted = df_sorted.sort_values('datetime')
            
            # Running best
            df_sorted['running_best'] = df_sorted[self.target_metric].cummax()
            
            # Plot 1: Performance over time
            ax1.scatter(df_sorted['datetime'], df_sorted[self.target_metric], 
                       alpha=0.6, s=30, label='Individual experiments')
            ax1.plot(df_sorted['datetime'], df_sorted['running_best'], 
                    color='red', linewidth=2, label='Running best')
            ax1.set_ylabel(self.target_metric.replace('_', ' ').title())
            ax1.set_title('Optimization Progress Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Experiment frequency over time
            df_sorted['hour'] = df_sorted['datetime'].dt.floor('H')
            hourly_counts = df_sorted.groupby('hour').size()
            ax2.bar(hourly_counts.index, hourly_counts.values, 
                   width=pd.Timedelta(hours=0.8), alpha=0.7, color='orange')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Experiments per Hour')
            ax2.set_title('Experiment Frequency')
            ax2.grid(True, alpha=0.3)
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not create timeline plot: {e}")
            return
        
        plt.tight_layout()
        plot_path = self.output_dir / 'optimization_timeline.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Timeline plot saved: {plot_path}")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        if self.verbose:
            print("Generating comprehensive report...")
        
        report_lines = []
        report_lines.append("# Comprehensive Hyperparameter Optimization Analysis Report\n\n")
        report_lines.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append(f"**Target Metric:** {self.target_metric}\n")
        report_lines.append(f"**Total Valid Experiments:** {len(self.df)}\n")
        report_lines.append(f"**Parameters Analyzed:** {len(self.parameter_names)}\n\n")
        
        # Performance overview
        report_lines.append("## Performance Overview\n\n")
        stats = self.performance_stats
        report_lines.append(f"- **Best Performance:** {stats['max_performance']:.4f}\n")
        report_lines.append(f"- **Worst Performance:** {stats['min_performance']:.4f}\n")
        report_lines.append(f"- **Mean Performance:** {stats['mean_performance']:.4f} ± {stats['std_performance']:.4f}\n")
        report_lines.append(f"- **Performance Range:** {stats['performance_range']:.4f}\n\n")
        
        # ANOVA results
        report_lines.append("## Parameter Importance Analysis\n\n")
        
        # Sort by importance
        sorted_params = sorted(self.anova_results.items(), 
                             key=lambda x: x[1]['normalized_importance'], reverse=True)
        
        report_lines.append("### Main Effects (Normalized Importance)\n\n")
        total_explained = 0
        for i, (param, results) in enumerate(sorted_params, 1):
            importance = results['normalized_importance']
            bootstrap_std = results['bootstrap_std']
            total_explained += importance
            
            report_lines.append(f"{i}. **{param}**: {importance:.3f} ± {bootstrap_std:.3f} ({importance*100:.1f}%)\n")
        
        report_lines.append(f"\n**Total Explained by Top Parameters:** {total_explained:.3f} ({total_explained*100:.1f}%)\n\n")
        
        # Method reliability
        report_lines.append("### Method Agreement Analysis\n\n")
        for param, results in sorted_params[:5]:  # Top 5
            rf_score = results['rf_importance']
            f_score = results['f_importance']  
            bootstrap_score = results['bootstrap_mean']
            
            # Calculate method agreement (correlation)
            scores = [rf_score, f_score, bootstrap_score]
            method_std = np.std(scores)
            agreement = "High" if method_std < 0.1 else "Medium" if method_std < 0.2 else "Low"
            
            report_lines.append(f"- **{param}**: Method agreement = {agreement} (std = {method_std:.3f})\n")
        
        # Architecture Analysis
        has_conv = 'conv_filters' in self.df.columns
        has_dense = 'dense_units' in self.df.columns
        
        if has_conv or has_dense:
            report_lines.append("\n## Architecture Analysis\n\n")
            
            if has_conv:
                # Analyze conv architectures
                conv_data = []
                for _, row in self.df.iterrows():
                    try:
                        conv_arch = str(ast.literal_eval(row['conv_filters']) if isinstance(row['conv_filters'], str) else row['conv_filters'])
                        performance = row[self.target_metric]
                        conv_data.append({'architecture': conv_arch, 'performance': performance})
                    except:
                        continue
                
                if conv_data:
                    conv_df = pd.DataFrame(conv_data)
                    arch_stats = conv_df.groupby('architecture')['performance'].agg(['mean', 'count']).reset_index()
                    best_conv = arch_stats.loc[arch_stats['mean'].idxmax()]
                    
                    report_lines.append(f"### Convolutional Architectures\n\n")
                    report_lines.append(f"- **Architectures tested:** {len(arch_stats)}\n")
                    report_lines.append(f"- **Best conv architecture:** {best_conv['architecture']} (accuracy: {best_conv['mean']:.4f}, n={best_conv['count']})\n")
            
            if has_dense:
                # Analyze dense architectures
                dense_data = []
                for _, row in self.df.iterrows():
                    try:
                        dense_arch = str(ast.literal_eval(row['dense_units']) if isinstance(row['dense_units'], str) else row['dense_units'])
                        performance = row[self.target_metric]
                        dense_data.append({'architecture': dense_arch, 'performance': performance})
                    except:
                        continue
                
                if dense_data:
                    dense_df = pd.DataFrame(dense_data)
                    arch_stats = dense_df.groupby('architecture')['performance'].agg(['mean', 'count']).reset_index()
                    best_dense = arch_stats.loc[arch_stats['mean'].idxmax()]
                    
                    report_lines.append(f"### Dense Layer Architectures\n\n")
                    report_lines.append(f"- **Architectures tested:** {len(arch_stats)}\n")
                    report_lines.append(f"- **Best dense architecture:** {best_dense['architecture']} (accuracy: {best_dense['mean']:.4f}, n={best_dense['count']})\n")

        # OFAT results
        if self.ofat_results:
            report_lines.append("\n## OFAT (One-Factor-At-A-Time) Analysis\n\n")
            report_lines.append(f"**Parameter Sweeps Found:** {sum(len(groups) for groups in self.ofat_results.values())}\n\n")
            
            for param, groups in self.ofat_results.items():
                if groups:
                    total_experiments = sum(group['n_experiments'] for group in groups)
                    avg_range = np.mean([group['performance_range'] for group in groups])
                    report_lines.append(f"- **{param}**: {len(groups)} sweep(s), {total_experiments} experiments, avg range = {avg_range:.4f}\n")
        
        # Interaction effects results
        if hasattr(self, 'interaction_results') and self.interaction_results:
            report_lines.append("\n## Interaction Effects Analysis\n\n")
            
            significant_interactions = {k: v for k, v in self.interaction_results.items() if v['significant']}
            total_interactions = len(self.interaction_results)
            significant_count = len(significant_interactions)
            
            report_lines.append(f"**Total Interactions Analyzed:** {total_interactions}\n")
            report_lines.append(f"**Significant Interactions (p < 0.05):** {significant_count}\n\n")
            
            if significant_interactions:
                report_lines.append("### Significant Parameter Interactions\n\n")
                
                # Sort by interaction strength
                sorted_interactions = sorted(significant_interactions.items(), 
                                           key=lambda x: x[1]['interaction_strength'], reverse=True)
                
                for i, (interaction, results) in enumerate(sorted_interactions[:5], 1):  # Top 5
                    param1, param2 = results['param1'], results['param2']
                    strength = results['interaction_strength']
                    p_value = results['p_value']
                    correlation = results['param_correlation']
                    
                    report_lines.append(f"{i}. **{param1} × {param2}**:\n")
                    report_lines.append(f"   - Interaction strength: {strength:.4f}\n")
                    report_lines.append(f"   - Statistical significance: p = {p_value:.4f}\n")
                    report_lines.append(f"   - Parameter correlation: {correlation:.3f}\n\n")
                
                # Analysis of interaction patterns
                high_strength_interactions = [k for k, v in significant_interactions.items() 
                                            if v['interaction_strength'] > np.median([r['interaction_strength'] for r in significant_interactions.values()])]
                
                if high_strength_interactions:
                    report_lines.append("### Key Findings\n\n")
                    report_lines.append(f"- **{len(high_strength_interactions)} interactions show strong effects** (above median strength)\n")
                    
                    # Identify most frequently interacting parameters
                    param_interaction_count = defaultdict(int)
                    for interaction_name in high_strength_interactions:
                        result = significant_interactions[interaction_name]
                        param_interaction_count[result['param1']] += 1
                        param_interaction_count[result['param2']] += 1
                    
                    if param_interaction_count:
                        most_interactive = max(param_interaction_count.items(), key=lambda x: x[1])
                        report_lines.append(f"- **{most_interactive[0]}** appears in {most_interactive[1]} significant interactions\n")
                    
                    # Check for highly correlated parameters
                    high_corr_pairs = [(k, v) for k, v in significant_interactions.items() 
                                     if abs(v['param_correlation']) > 0.7]
                    if high_corr_pairs:
                        report_lines.append(f"- **{len(high_corr_pairs)} parameter pairs show high correlation** (|r| > 0.7), indicating potential confounding\n")
            else:
                report_lines.append("*No statistically significant interactions detected.*\n")
                report_lines.append("This suggests parameters act mostly independently.\n")
        
        # Data quality assessment
        report_lines.append("\n## Data Quality Assessment\n\n")
        report_lines.append(f"- **Experiments after validation:** {len(self.df)}\n")
        report_lines.append(f"- **Performance coefficient of variation:** {stats['std_performance']/stats['mean_performance']:.3f}\n")
        
        # Check for sufficient parameter coverage
        param_coverage = len(self.parameter_names) / max(len(self.parameter_names), 10)  # Assume 10 ideal params
        coverage_level = "Excellent" if param_coverage > 0.8 else "Good" if param_coverage > 0.6 else "Limited"
        report_lines.append(f"- **Parameter coverage:** {coverage_level} ({len(self.parameter_names)} parameters)\n")
        
        # Recommendations
        report_lines.append("\n## Recommendations\n\n")
        
        top_param = sorted_params[0][0]
        top_importance = sorted_params[0][1]['normalized_importance']
        
        report_lines.append(f"1. **Focus on {top_param}**: Highest impact parameter ({top_importance*100:.1f}% of variance)\n")
        
        if total_explained > 0.8:
            report_lines.append("2. **Good parameter coverage**: Current parameters explain most variation\n")
        else:
            report_lines.append("2. **Consider additional parameters**: Current set explains limited variation\n")
        
        if len(self.ofat_results) > 0:
            report_lines.append("3. **OFAT analysis available**: Use parameter sweep insights for targeted optimization\n")
        
        # DoE recommendations for improved sampling
        report_lines.append("\n### Design of Experiments (DoE) Recommendations\n\n")
        
        # Analyze current parameter coverage
        param_coverage_analysis = self._analyze_parameter_coverage()
        
        if hasattr(self, 'interaction_results') and self.interaction_results:
            significant_interactions = {k: v for k, v in self.interaction_results.items() if v['significant']}
            if significant_interactions:
                report_lines.append("**Recommended Strategy: Response Surface Methodology (RSM)**\n")
                report_lines.append("- Significant interactions detected, requiring systematic exploration\n")
                report_lines.append("- Use Central Composite Design (CCD) or Box-Behnken Design\n")
                top_interaction_params = set()
                for result in list(significant_interactions.values())[:3]:  # Top 3 interactions
                    top_interaction_params.add(result['param1'])
                    top_interaction_params.add(result['param2'])
                report_lines.append(f"- Focus on: {', '.join(sorted(top_interaction_params))}\n")
            else:
                report_lines.append("**Recommended Strategy: Orthogonal Factorial Design**\n")
                report_lines.append("- No significant interactions, main effects dominate\n")
                report_lines.append("- Use fractional factorial design for efficiency\n")
        else:
            report_lines.append("**Recommended Strategy: Latin Hypercube Sampling (LHS)**\n")
            report_lines.append("- Broad parameter space exploration needed\n")
        
        # Identify parameters with insufficient samples
        low_sample_params = param_coverage_analysis['low_sample_params']
        if low_sample_params:
            report_lines.append(f"\n**Priority Parameters for Additional Sampling:**\n")
            for param, count in low_sample_params:
                report_lines.append(f"- **{param}**: Only {count} unique values (recommend ≥5 levels)\n")
        
        # Architecture-specific recommendations
        has_architecture_params = any(param in ['conv_filters', 'dense_units'] for param in self.parameter_names)
        if has_architecture_params:
            report_lines.append(f"\n**Architecture Sampling Strategy:**\n")
            report_lines.append("- Use progressive complexity: start simple, increase systematically\n")
            report_lines.append("- Ensure balanced representation across network sizes\n")
            report_lines.append("- Consider computational constraints in design\n")
        
        report_lines.append(f"\n**Target Sample Size:** {param_coverage_analysis['recommended_sample_size']} experiments\n")
        report_lines.append("- Based on parameter count and interaction complexity\n")
        report_lines.append("- Ensures adequate power for ANOVA (≥5 replicates per factor level)\n")
        
        # Save report
        report_path = self.output_dir / 'comprehensive_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        
        if self.verbose:
            print(f"Report saved: {report_path}")
        
        return report_lines
    
    def run_complete_analysis(self):
        """Run all analysis components and generate outputs."""
        print("Starting Comprehensive Hyperparameter Analysis...")
        print(f"Target metric: {self.target_metric}")
        print(f"Data quality: {len(self.df)} experiments, {len(self.parameter_names)} parameters")
        print()
        
        # Perform analyses
        self.perform_ofat_analysis()
        self.perform_anova_analysis()
        self.perform_interaction_analysis()
        
        # Generate visualizations
        self.plot_ofat_analysis()
        self.plot_architecture_analysis()
        self.plot_anova_analysis()
        self.plot_interaction_analysis()
        self.plot_pairwise_surfaces(top_n=4)
        self.plot_optimization_timeline()
        
        # Generate report
        self.generate_comprehensive_report()
        
        print(f"\n✅ Analysis complete! Results saved to: {self.output_dir}")
        
        # Print key findings
        if self.anova_results:
            print("\n📊 Key Findings:")
            sorted_params = sorted(self.anova_results.items(), 
                                 key=lambda x: x[1]['normalized_importance'], reverse=True)
            top_param, top_result = sorted_params[0]
            print(f"   Most important parameter: {top_param} ({top_result['normalized_importance']*100:.1f}%)")
            
            performance_range = self.performance_stats['performance_range']
            print(f"   Performance range: {performance_range:.4f}")
            
            if self.ofat_results:
                total_ofat_groups = sum(len(groups) for groups in self.ofat_results.values())
                print(f"   OFAT parameter sweeps found: {total_ofat_groups}")
        
        return {
            'ofat_results': self.ofat_results,
            'anova_results': self.anova_results,
            'performance_stats': self.performance_stats,
            'output_dir': self.output_dir
        }


def main():
    parser = argparse.ArgumentParser(description='Comprehensive hyperparameter optimization analysis')
    parser.add_argument('--mode', type=str, choices=['pd', 'cwt'], default='pd',
                       help='Analysis mode: pd (PD signal) or cwt (CWT image) (default: pd)')
    parser.add_argument('--log_path', type=str, 
                       help='Path to experiment log CSV file')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for analysis results')
    parser.add_argument('--target_metric', type=str, default='mean_val_accuracy',
                       help='Target metric to analyze (default: mean_val_accuracy)')
    parser.add_argument('--min_experiments', type=int, default=3,
                       help='Minimum experiments per parameter variation (default: 3)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    try:
        # Initialize comprehensive analyzer
        analyzer = ComprehensiveHyperoptAnalyzer(
            mode=args.mode,
            log_path=args.log_path,
            output_dir=args.output_dir,
            target_metric=args.target_metric,
            min_experiments=args.min_experiments,
            verbose=args.verbose
        )
        
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        print("\n🎉 Comprehensive hyperparameter analysis completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())