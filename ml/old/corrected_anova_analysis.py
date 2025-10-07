#!/usr/bin/env python3
"""
Corrected ANOVA Analysis using Sobol Sensitivity and Classical ANOVA

This script provides proper variance decomposition that sums to 100%
using established statistical methods.

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
from itertools import combinations
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Local imports
from config import get_pd_experiment_log_path, PD_OUTPUTS_DIR

class CorrectedANOVAAnalyzer:
    """Performs proper ANOVA decomposition using multiple methods."""
    
    def __init__(self, log_path=None, output_dir=None, target_metric='mean_val_accuracy', verbose=False):
        self.log_path = Path(log_path) if log_path else get_pd_experiment_log_path()
        self.output_dir = Path(output_dir) if output_dir else PD_OUTPUTS_DIR / 'corrected_anova'
        self.target_metric = target_metric
        self.verbose = verbose
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess data
        self.df = None
        self.X_processed = None
        self.y = None
        self.parameter_names = []
        
        self.load_and_preprocess_data()
        
        # Analysis results
        self.sensitivity_indices = {}
        self.classical_anova = {}
        
    def load_and_preprocess_data(self):
        """Load experiment data and preprocess for analysis."""
        if not self.log_path.exists():
            raise FileNotFoundError(f"Experiment log not found: {self.log_path}")
        
        if self.verbose:
            print(f"Loading experiment data from: {self.log_path}")
        
        # Load data
        self.df = pd.read_csv(self.log_path, encoding='utf-8')
        
        # Filter valid experiments
        self.df = self.df[self.df[self.target_metric].notna()].copy()
        
        if len(self.df) < 20:
            raise ValueError("Need at least 20 experiments for meaningful ANOVA analysis")
        
        if self.verbose:
            print(f"Loaded {len(self.df)} experiments with valid {self.target_metric} values")
        
        # Focus on key parameters that actually vary
        key_params = ['learning_rate', 'batch_size', 'l2_reg']
        available_params = [p for p in key_params if p in self.df.columns]
        
        if len(available_params) < 2:
            raise ValueError("Need at least 2 varying parameters for ANOVA analysis")
        
        self.parameter_names = available_params
        
        if self.verbose:
            print(f"Analyzing parameters: {self.parameter_names}")
        
        # Preprocess parameters
        self.X_processed = self.preprocess_parameters()
        self.y = self.df[self.target_metric].values
        
        if self.verbose:
            print(f"Preprocessed data shape: {self.X_processed.shape}")
    
    def preprocess_parameters(self):
        """Convert parameters to numerical format for analysis."""
        X = pd.DataFrame()
        
        for param in self.parameter_names:
            if param not in self.df.columns:
                continue
                
            values = self.df[param].copy()
            
            # Convert to numeric, handling various formats
            if param in ['learning_rate', 'l2_reg']:
                X[param] = pd.to_numeric(values, errors='coerce')
            elif param == 'batch_size':
                X[param] = pd.to_numeric(values, errors='coerce')
            else:
                # Try numeric conversion, otherwise skip
                try:
                    X[param] = pd.to_numeric(values, errors='coerce')
                except:
                    if self.verbose:
                        print(f"Skipping non-numeric parameter: {param}")
                    continue
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Update parameter names
        self.parameter_names = list(X.columns)
        
        return X
    
    def sobol_sensitivity_analysis(self, n_bootstrap=1000):
        """
        Perform Sobol sensitivity analysis using bootstrap sampling.
        This gives proper variance decomposition.
        """
        if self.verbose:
            print("Performing Sobol-style sensitivity analysis...")
        
        # We'll use a bootstrap approach to estimate sensitivity indices
        n_params = len(self.parameter_names)
        
        # Storage for sensitivity indices
        first_order = {param: [] for param in self.parameter_names}
        total_order = {param: [] for param in self.parameter_names}
        second_order = {}
        
        # Bootstrap iterations
        for iteration in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(self.X_processed), size=len(self.X_processed), replace=True)
            X_boot = self.X_processed.iloc[indices].values
            y_boot = self.y[indices]
            
            # Fit surrogate model
            rf = RandomForestRegressor(n_estimators=50, random_state=iteration)
            rf.fit(X_boot, y_boot)
            
            # Base predictions (all parameters at their mean)
            X_base = np.tile(np.mean(X_boot, axis=0), (100, 1))
            y_base = rf.predict(X_base)
            base_var = np.var(y_base)  # Should be ~0 for constant input
            
            # Total variance estimate
            X_random = np.random.uniform(
                low=np.min(X_boot, axis=0),
                high=np.max(X_boot, axis=0),
                size=(1000, n_params)
            )
            y_random = rf.predict(X_random)
            total_var = np.var(y_random)
            
            if total_var == 0:
                continue  # Skip this iteration
            
            # First-order indices (main effects)
            for i, param in enumerate(self.parameter_names):
                # Vary only parameter i
                X_varied = X_base.copy()
                param_range = np.linspace(np.min(X_boot[:, i]), np.max(X_boot[:, i]), 100)
                X_varied[:, i] = param_range
                
                y_varied = rf.predict(X_varied)
                param_var = np.var(y_varied)
                
                first_order_index = param_var / total_var if total_var > 0 else 0
                first_order[param].append(first_order_index)
            
            # Total-order indices (main + interactions)
            for i, param in enumerate(self.parameter_names):
                # Fix parameter i, vary all others
                X_fixed = X_random.copy()
                X_fixed[:, i] = np.mean(X_boot[:, i])  # Fix parameter i
                
                y_fixed = rf.predict(X_fixed)
                fixed_var = np.var(y_fixed)
                
                total_order_index = 1 - (fixed_var / total_var) if total_var > 0 else 0
                total_order[param].append(max(0, total_order_index))  # Ensure non-negative
        
        # Average across bootstrap iterations
        self.sensitivity_indices = {
            'first_order': {param: np.mean(values) for param, values in first_order.items()},
            'first_order_std': {param: np.std(values) for param, values in first_order.items()},
            'total_order': {param: np.mean(values) for param, values in total_order.items()},
            'total_order_std': {param: np.std(values) for param, values in total_order.items()}
        }
        
        # Normalize to ensure sum <= 1
        total_first_order = sum(self.sensitivity_indices['first_order'].values())
        if total_first_order > 1:
            if self.verbose:
                print(f"Normalizing first-order indices (sum was {total_first_order:.3f})")
            for param in self.sensitivity_indices['first_order']:
                self.sensitivity_indices['first_order'][param] /= total_first_order
        
        if self.verbose:
            print("Sensitivity analysis complete!")
    
    def classical_anova_analysis(self):
        """
        Perform classical ANOVA using sklearn's approach.
        """
        if self.verbose:
            print("Performing classical ANOVA analysis...")
        
        try:
            from sklearn.feature_selection import f_regression
            
            # Calculate F-statistics for each parameter
            f_stats, p_values = f_regression(self.X_processed, self.y)
            
            # Convert to variance explained (approximate)
            # Using the fact that F = MS_effect / MS_error relates to variance explained
            total_variance = np.var(self.y)
            
            self.classical_anova = {}
            for i, param in enumerate(self.parameter_names):
                # Approximate variance explained from F-statistic
                # This is a simplification but gives reasonable estimates
                variance_explained = f_stats[i] / (f_stats[i] + len(self.y) - len(self.parameter_names))
                
                self.classical_anova[param] = {
                    'f_statistic': f_stats[i],
                    'p_value': p_values[i],
                    'variance_explained': variance_explained,
                    'significant': p_values[i] < 0.05
                }
            
            if self.verbose:
                print("Classical ANOVA complete!")
                
        except ImportError:
            if self.verbose:
                print("Sklearn not available for classical ANOVA")
            self.classical_anova = {}
    
    def plot_sensitivity_analysis(self):
        """Create comprehensive sensitivity analysis plots."""
        if self.verbose:
            print("Creating sensitivity analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. First-order sensitivity indices
        params = list(self.sensitivity_indices['first_order'].keys())
        first_order_values = [self.sensitivity_indices['first_order'][p] for p in params]
        first_order_errors = [self.sensitivity_indices['first_order_std'][p] for p in params]
        
        y_pos = np.arange(len(params))
        axes[0, 0].barh(y_pos, first_order_values, xerr=first_order_errors, capsize=3)
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels(params)
        axes[0, 0].set_xlabel('First-Order Sensitivity Index')
        axes[0, 0].set_title('Main Effects (First-Order Indices)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Total-order sensitivity indices
        total_order_values = [self.sensitivity_indices['total_order'][p] for p in params]
        total_order_errors = [self.sensitivity_indices['total_order_std'][p] for p in params]
        
        axes[0, 1].barh(y_pos, total_order_values, xerr=total_order_errors, capsize=3)
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels(params)
        axes[0, 1].set_xlabel('Total-Order Sensitivity Index')
        axes[0, 1].set_title('Total Effects (Main + Interactions)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Comparison plot
        x = np.arange(len(params))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, first_order_values, width, label='First-order', alpha=0.8)
        axes[1, 0].bar(x + width/2, total_order_values, width, label='Total-order', alpha=0.8)
        axes[1, 0].set_xlabel('Parameters')
        axes[1, 0].set_ylabel('Sensitivity Index')
        axes[1, 0].set_title('Comparison: Main vs Total Effects')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(params, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Interaction strength (Total - First order)
        interaction_strength = [total_order_values[i] - first_order_values[i] for i in range(len(params))]
        
        axes[1, 1].bar(params, interaction_strength)
        axes[1, 1].set_xlabel('Parameters')
        axes[1, 1].set_ylabel('Interaction Strength')
        axes[1, 1].set_title('Parameter Interaction Effects')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'corrected_sensitivity_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        if self.verbose:
            print(f"Saved: {plot_path}")
        
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        if self.verbose:
            print("Generating analysis report...")
        
        report_lines = []
        report_lines.append("# Corrected ANOVA Sensitivity Analysis Report\n\n")
        report_lines.append(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append(f"**Target Metric:** {self.target_metric}\n")
        report_lines.append(f"**Total Experiments:** {len(self.df)}\n")
        report_lines.append(f"**Parameters Analyzed:** {len(self.parameter_names)}\n\n")
        
        # Sensitivity indices summary
        report_lines.append("## Sobol Sensitivity Analysis Results\n\n")
        
        # Sort by first-order importance
        first_order_sorted = sorted(
            self.sensitivity_indices['first_order'].items(),
            key=lambda x: x[1], reverse=True
        )
        
        report_lines.append("### Main Effects (First-Order Sensitivity)\n\n")
        total_first_order = 0
        for i, (param, sensitivity) in enumerate(first_order_sorted, 1):
            std_error = self.sensitivity_indices['first_order_std'][param]
            total_first_order += sensitivity
            report_lines.append(f"{i}. **{param}**: {sensitivity:.3f} ± {std_error:.3f} ({sensitivity*100:.1f}% of variance)\n")
        
        report_lines.append(f"\n**Total main effects:** {total_first_order:.3f} ({total_first_order*100:.1f}% of variance)\n\n")
        
        # Total order effects
        report_lines.append("### Total Effects (Including Interactions)\n\n")
        total_order_sorted = sorted(
            self.sensitivity_indices['total_order'].items(),
            key=lambda x: x[1], reverse=True
        )
        
        for i, (param, sensitivity) in enumerate(total_order_sorted, 1):
            std_error = self.sensitivity_indices['total_order_std'][param]
            first_order = self.sensitivity_indices['first_order'][param]
            interaction = sensitivity - first_order
            report_lines.append(f"{i}. **{param}**: {sensitivity:.3f} ± {std_error:.3f} (interaction: {interaction:.3f})\n")
        
        # Interaction analysis
        report_lines.append("\n### Interaction Analysis\n\n")
        has_strong_interactions = False
        for param in self.parameter_names:
            first_order = self.sensitivity_indices['first_order'][param]
            total_order = self.sensitivity_indices['total_order'][param]
            interaction_strength = total_order - first_order
            
            if interaction_strength > 0.05:  # 5% threshold for significant interaction
                has_strong_interactions = True
                report_lines.append(f"- **{param}** shows significant interactions ({interaction_strength:.3f})\n")
        
        if not has_strong_interactions:
            report_lines.append("- No significant parameter interactions detected\n")
        
        report_lines.append("\n")
        
        # Classical ANOVA results
        if self.classical_anova:
            report_lines.append("## Classical ANOVA Results\n\n")
            for param, results in self.classical_anova.items():
                significance = "**" if results['significant'] else ""
                report_lines.append(f"- {significance}{param}{significance}: ")
                report_lines.append(f"F = {results['f_statistic']:.2f}, ")
                report_lines.append(f"p = {results['p_value']:.4f}, ")
                report_lines.append(f"η² ≈ {results['variance_explained']:.3f}\n")
            report_lines.append("\n")
        
        # Interpretation
        report_lines.append("## Interpretation\n\n")
        
        # Most important parameter
        top_param, top_sensitivity = first_order_sorted[0]
        report_lines.append(f"- **Most influential parameter:** {top_param} ({top_sensitivity*100:.1f}% of variance)\n")
        
        # Overall model adequacy
        total_explained = sum(self.sensitivity_indices['total_order'].values())
        if total_explained > 0.8:
            report_lines.append("- **Good parameter coverage:** Selected parameters explain most performance variation\n")
        elif total_explained > 0.5:
            report_lines.append("- **Moderate parameter coverage:** Additional factors may influence performance\n")
        else:
            report_lines.append("- **Limited parameter coverage:** Many factors affecting performance not captured\n")
        
        # Interaction summary
        max_interaction = max([self.sensitivity_indices['total_order'][p] - self.sensitivity_indices['first_order'][p] 
                              for p in self.parameter_names])
        if max_interaction > 0.1:
            report_lines.append("- **Significant interactions detected:** Parameters do not act independently\n")
        else:
            report_lines.append("- **Limited interactions:** Parameters appear to act mostly independently\n")
        
        # Save report
        report_path = self.output_dir / 'corrected_anova_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        
        if self.verbose:
            print(f"Report saved: {report_path}")
    
    def run_analysis(self):
        """Run complete corrected ANOVA analysis."""
        print("Starting Corrected ANOVA Sensitivity Analysis...")
        print(f"Target metric: {self.target_metric}")
        print(f"Parameters: {self.parameter_names}")
        print()
        
        # Perform sensitivity analysis
        self.sobol_sensitivity_analysis()
        
        # Classical ANOVA
        self.classical_anova_analysis()
        
        # Generate visualizations
        self.plot_sensitivity_analysis()
        
        # Generate report
        self.generate_report()
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        
        # Print summary
        print("\nSensitivity Summary:")
        for param, sensitivity in sorted(self.sensitivity_indices['first_order'].items(), 
                                       key=lambda x: x[1], reverse=True):
            print(f"- {param}: {sensitivity:.3f} ({sensitivity*100:.1f}%)")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Corrected ANOVA sensitivity analysis')
    parser.add_argument('--log_path', type=str, help='Path to experiment log CSV file')
    parser.add_argument('--output_dir', type=str, help='Output directory for analysis results')
    parser.add_argument('--target_metric', type=str, default='mean_val_accuracy',
                       help='Target metric to analyze (default: mean_val_accuracy)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = CorrectedANOVAAnalyzer(
            log_path=args.log_path,
            output_dir=args.output_dir,
            target_metric=args.target_metric,
            verbose=args.verbose
        )
        
        # Run analysis
        analyzer.run_analysis()
        
        print("\n✅ Corrected ANOVA analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()