#!/usr/bin/env python3
"""
Parameter Sensitivity Analysis Script

This script analyzes the hyperparameter optimization experiment log to identify
sets of experiments where only one parameter is varied while all others are fixed.
For such parameter sets with 3+ instances, it plots the impact on key performance
metrics, primarily validation accuracy.

Author: AI Assistant
"""

# Ensure UTF-8 encoding for all I/O operations
import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import ast
from collections import defaultdict
import argparse
from itertools import combinations

# Local imports
from config import get_pd_experiment_log_path, PD_OUTPUTS_DIR

class ParameterSensitivityAnalyzer:
    """Analyzes parameter sensitivity from experiment logs."""
    
    def __init__(self, log_path=None, output_dir=None, verbose=False):
        self.log_path = Path(log_path) if log_path else get_pd_experiment_log_path()
        self.output_dir = Path(output_dir) if output_dir else PD_OUTPUTS_DIR / 'parameter_analysis'
        self.verbose = verbose
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load experiment data
        self.df = None
        self.load_experiment_data()
        
        # Parameters to analyze (excluding metadata columns)
        self.analysis_params = [
            'learning_rate', 'batch_size', 'epochs', 'k_folds',
            'conv_filters', 'dense_units', 'dropout_rates', 'l2_reg',
            'batch_norm', 'optimizer', 'early_stopping_patience',
            'lr_reduction_patience', 'class_weights', 'augment_fraction',
            'time_shift_range', 'stretch_probability', 'noise_probability',
            'amplitude_scale_probability'
        ]
        
    def load_experiment_data(self):
        """Load and preprocess experiment log data."""
        if not self.log_path.exists():
            raise FileNotFoundError(f"Experiment log not found: {self.log_path}")
            
        if self.verbose:
            print(f"Loading experiment data from: {self.log_path}")
            
        self.df = pd.read_csv(self.log_path, encoding='utf-8')
        
        # Filter out experiments without validation results
        self.df = self.df[self.df['mean_val_accuracy'].notna()].copy()
        
        if self.verbose:
            print(f"Loaded {len(self.df)} experiments with validation results")
            
    def normalize_parameter_values(self, param_name, values):
        """
        Normalize parameter values for comparison.
        Handles lists, strings, and numeric values.
        """
        normalized = []
        for val in values:
            if pd.isna(val):
                normalized.append(None)
                continue
                
            # Handle string representations of lists/dicts
            if isinstance(val, str) and (val.startswith('[') or val.startswith('{')):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        # Sort lists safely - convert all elements to string for comparison
                        try:
                            sorted_list = sorted(parsed)
                        except TypeError:
                            # Mixed types in list, sort as strings
                            sorted_list = sorted([str(x) for x in parsed])
                        normalized.append(str(sorted_list))
                    else:
                        normalized.append(str(parsed))
                except (ValueError, SyntaxError):
                    normalized.append(str(val))
            else:
                normalized.append(str(val))
                
        return normalized
        
    def find_single_parameter_variations(self, min_experiments=3):
        """
        Find sets of experiments where only one parameter varies while all others are fixed.
        
        Returns:
            dict: {param_name: [list of experiment groups]}
        """
        if self.verbose:
            print(f"Finding single parameter variations (min {min_experiments} experiments)...")
            
        variations = defaultdict(list)
        
        for param in self.analysis_params:
            if param not in self.df.columns:
                continue
                
            # Group by all other parameters
            other_params = [p for p in self.analysis_params if p != param and p in self.df.columns]
            
            # Normalize all parameter values for grouping
            group_cols = []
            for other_param in other_params:
                normalized = self.normalize_parameter_values(other_param, self.df[other_param])
                self.df[f'{other_param}_normalized'] = normalized
                group_cols.append(f'{other_param}_normalized')
            
            # Group by all other parameters (fixed)
            groups = self.df.groupby(group_cols)
            
            for group_key, group_df in groups:
                if len(group_df) >= min_experiments:
                    # Check if the target parameter actually varies
                    param_values = self.normalize_parameter_values(param, group_df[param])
                    unique_values = set([v for v in param_values if v is not None])
                    
                    if len(unique_values) >= min_experiments:
                        variations[param].append({
                            'group_key': group_key,
                            'experiments': group_df.copy(),
                            'param_values': param_values,
                            'unique_values': sorted(unique_values)
                        })
            
            # Clean up temporary columns
            for col in group_cols:
                if col in self.df.columns:
                    self.df.drop(col, axis=1, inplace=True)
                    
        if self.verbose:
            for param, groups in variations.items():
                print(f"  {param}: {len(groups)} parameter sweep(s)")
                
        return dict(variations)
    
    def plot_parameter_sensitivity(self, variations, metrics=['mean_val_accuracy']):
        """
        Plot parameter sensitivity for identified variations.
        
        Args:
            variations: Dictionary from find_single_parameter_variations()
            metrics: List of metrics to plot
        """
        if self.verbose:
            print("Generating parameter sensitivity plots...")
            
        for param_name, param_groups in variations.items():
            if not param_groups:
                continue
                
            # Create figure for this parameter
            n_groups = len(param_groups)
            n_metrics = len(metrics)
            
            fig, axes = plt.subplots(n_groups, n_metrics, 
                                   figsize=(6*n_metrics, 4*n_groups),
                                   squeeze=False)
            
            if n_groups == 1:
                axes = [axes]
                
            fig.suptitle(f'Parameter Sensitivity Analysis: {param_name}', 
                        fontsize=16, fontweight='bold')
            
            for group_idx, group_data in enumerate(param_groups):
                experiments = group_data['experiments']
                unique_values = group_data['unique_values']
                
                for metric_idx, metric in enumerate(metrics):
                    ax = axes[group_idx][metric_idx]
                    
                    if metric not in experiments.columns:
                        ax.text(0.5, 0.5, f'Metric {metric}\nnot available', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'Group {group_idx+1}: {metric}')
                        continue
                    
                    # Prepare data for plotting
                    plot_data = []
                    for _, exp in experiments.iterrows():
                        param_val = self.normalize_parameter_values(param_name, [exp[param_name]])[0]
                        if param_val is not None and exp[metric] is not np.nan:
                            plot_data.append({
                                'param_value': param_val,
                                'metric_value': exp[metric],
                                'version': exp['version']
                            })
                    
                    if not plot_data:
                        ax.text(0.5, 0.5, 'No valid data', 
                               ha='center', va='center', transform=ax.transAxes)
                        continue
                    
                    plot_df = pd.DataFrame(plot_data)
                    
                    # Handle different parameter types for plotting
                    try:
                        # Try numeric plotting first
                        plot_df['param_numeric'] = pd.to_numeric(plot_df['param_value'])
                        plot_df = plot_df.sort_values('param_numeric')
                        
                        # Scatter plot with trend line
                        ax.scatter(plot_df['param_numeric'], plot_df['metric_value'], 
                                 alpha=0.7, s=60)
                        
                        # Add trend line if enough points
                        if len(plot_df) >= 3:
                            z = np.polyfit(plot_df['param_numeric'], plot_df['metric_value'], 1)
                            p = np.poly1d(z)
                            x_trend = np.linspace(plot_df['param_numeric'].min(), 
                                                plot_df['param_numeric'].max(), 100)
                            ax.plot(x_trend, p(x_trend), 'r--', alpha=0.8)
                        
                        ax.set_xlabel(f'{param_name}')
                        
                    except (ValueError, TypeError):
                        # Categorical parameter - use box plot or bar plot
                        if len(plot_df) > len(plot_df['param_value'].unique()) * 2:
                            # Box plot for multiple values per category
                            sns.boxplot(data=plot_df, x='param_value', y='metric_value', ax=ax)
                        else:
                            # Bar plot for single values per category
                            mean_values = plot_df.groupby('param_value')['metric_value'].mean()
                            ax.bar(range(len(mean_values)), mean_values.values)
                            ax.set_xticks(range(len(mean_values)))
                            ax.set_xticklabels(mean_values.index, rotation=45)
                        
                        ax.set_xlabel(f'{param_name}')
                    
                    ax.set_ylabel(metric.replace('_', ' ').title())
                    ax.set_title(f'Group {group_idx+1}: {metric}')
                    ax.grid(True, alpha=0.3)
                    
                    # Add annotations for experiment versions
                    for _, row in plot_df.iterrows():
                        try:
                            if isinstance(row['param_value'], (int, float)):
                                x_pos = row['param_value']
                            else:
                                # For categorical, approximate position
                                unique_vals = sorted(plot_df['param_value'].unique())
                                x_pos = unique_vals.index(row['param_value'])
                        except:
                            continue
                            
                        ax.annotate(row['version'], 
                                  (x_pos, row['metric_value']),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=8, alpha=0.7)
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = self.output_dir / f'sensitivity_{param_name.lower()}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"  Saved: {plot_filename}")
            
            plt.close()
    
    def generate_summary_report(self, variations):
        """Generate a summary report of parameter sensitivity findings."""
        if self.verbose:
            print("Generating summary report...")
            
        report_lines = []
        report_lines.append("# Parameter Sensitivity Analysis Report\n\n")
        report_lines.append(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append(f"**Total Experiments Analyzed:** {len(self.df)}\n")
        report_lines.append(f"**Output Directory:** {self.output_dir}\n\n")
        
        report_lines.append("## Summary\n\n")
        total_variations = sum(len(groups) for groups in variations.values())
        report_lines.append(f"Found **{total_variations} parameter variations** across **{len(variations)} parameters**:\n\n")
        
        for param_name, param_groups in variations.items():
            if not param_groups:
                continue
                
            report_lines.append(f"### {param_name}\n\n")
            report_lines.append(f"- **Number of parameter sweeps:** {len(param_groups)}\n\n")
            
            for group_idx, group_data in enumerate(param_groups):
                experiments = group_data['experiments']
                unique_values = group_data['unique_values']
                
                report_lines.append(f"**Sweep {group_idx+1}:**\n")
                report_lines.append(f"- Values tested: {unique_values}\n")
                report_lines.append(f"- Number of experiments: {len(experiments)}\n")
                
                # Performance summary
                if 'mean_val_accuracy' in experiments.columns:
                    acc_values = experiments['mean_val_accuracy'].dropna()
                    if len(acc_values) > 0:
                        report_lines.append(f"- Validation accuracy range: {acc_values.min():.4f} - {acc_values.max():.4f}\n")
                        report_lines.append(f"- Best accuracy: {acc_values.max():.4f} (version {experiments.loc[acc_values.idxmax(), 'version']})\n")
                
                report_lines.append(f"\n")
            
            report_lines.append(f"\n")
        
        # Save report
        report_path = self.output_dir / 'sensitivity_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
            
        if self.verbose:
            print(f"Report saved: {report_path}")
    
    def run_analysis(self, min_experiments=3, metrics=['mean_val_accuracy']):
        """Run complete parameter sensitivity analysis."""
        print("Starting Parameter Sensitivity Analysis...")
        print(f"Minimum experiments per variation: {min_experiments}")
        print(f"Metrics to analyze: {metrics}")
        print()
        
        # Find parameter variations
        variations = self.find_single_parameter_variations(min_experiments)
        
        if not any(variations.values()):
            print("No parameter variations found with the specified criteria.")
            return
        
        # Generate plots
        self.plot_parameter_sensitivity(variations, metrics)
        
        # Generate summary report
        self.generate_summary_report(variations)
        
        print(f"\\nAnalysis complete! Results saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze parameter sensitivity from experiment logs')
    parser.add_argument('--log_path', type=str, 
                       help='Path to experiment log CSV file')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for analysis results')
    parser.add_argument('--min_experiments', type=int, default=3,
                       help='Minimum number of experiments per parameter variation (default: 3)')
    parser.add_argument('--metrics', nargs='+', 
                       default=['mean_val_accuracy', 'mean_val_loss', 'best_fold_accuracy'],
                       help='Metrics to analyze (default: mean_val_accuracy mean_val_loss best_fold_accuracy)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = ParameterSensitivityAnalyzer(
            log_path=args.log_path,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        # Run analysis
        analyzer.run_analysis(
            min_experiments=args.min_experiments,
            metrics=args.metrics
        )
        
        print("\\n✅ Parameter sensitivity analysis completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()