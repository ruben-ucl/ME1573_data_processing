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
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

# Local imports
from config import get_experiment_log_path, OUTPUTS_DIR

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class ComprehensiveHyperoptAnalyzer:
    """
    Comprehensive analysis tool for hyperparameter optimization results.
    Combines OFAT analysis, ANOVA decomposition, and visualization tools.
    """
    
    def __init__(self, log_path=None, output_dir=None, target_metric='mean_val_accuracy', 
                 min_experiments=3, verbose=False):
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
        
        # Load data
        self.df = pd.read_csv(self.log_path, encoding='utf-8')
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
        # Core ML parameters
        candidate_params = [
            'learning_rate', 'batch_size', 'epochs', 'k_folds',
            'conv_filters', 'dense_units', 'dropout_rates', 'l2_reg', 'l2_regularization',
            'batch_norm', 'use_batch_norm', 'optimizer', 'early_stopping_patience',
            'lr_reduction_patience', 'class_weights', 'use_class_weights',
            # Augmentation parameters
            'augment_fraction', 'time_shift_range', 'stretch_probability', 
            'stretch_scale', 'noise_probability', 'noise_std',
            'amplitude_scale_probability', 'amplitude_scale',
            'rotation_range', 'width_shift_range', 'height_shift_range'
        ]
        
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
            if param in ['learning_rate', 'l2_reg', 'l2_regularization', 'augment_fraction']:
                # Numeric parameters
                X[param] = pd.to_numeric(values, errors='coerce')
                
            elif param in ['batch_size', 'epochs', 'k_folds', 'early_stopping_patience', 'lr_reduction_patience']:
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
                
            elif param in ['conv_filters', 'dense_units', 'dropout_rates']:
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
        
        # Generate visualizations
        self.plot_ofat_analysis()
        self.plot_architecture_analysis()
        self.plot_anova_analysis()
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