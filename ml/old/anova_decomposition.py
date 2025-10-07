#!/usr/bin/env python3
"""
ANOVA Decomposition for Hyperparameter Sensitivity Analysis

This script performs Analysis of Variance (ANOVA) decomposition to understand
the contribution of individual parameters and their interactions to model performance.

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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Local imports
from config import get_pd_experiment_log_path, PD_OUTPUTS_DIR

class ANOVADecomposer:
    """Performs ANOVA decomposition for hyperparameter sensitivity analysis."""
    
    def __init__(self, log_path=None, output_dir=None, target_metric='mean_val_accuracy', verbose=False):
        self.log_path = Path(log_path) if log_path else get_pd_experiment_log_path()
        self.output_dir = Path(output_dir) if output_dir else PD_OUTPUTS_DIR / 'anova_analysis'
        self.target_metric = target_metric
        self.verbose = verbose
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess data
        self.df = None
        self.X_processed = None
        self.y = None
        self.parameter_names = []
        self.encoders = {}
        self.scaler = StandardScaler()
        
        self.load_and_preprocess_data()
        
        # ANOVA decomposition results
        self.anova_components = {}
        self.variance_explained = {}
        
    def load_and_preprocess_data(self):
        """Load experiment data and preprocess for ANOVA analysis."""
        if not self.log_path.exists():
            raise FileNotFoundError(f"Experiment log not found: {self.log_path}")
        
        if self.verbose:
            print(f"Loading experiment data from: {self.log_path}")
        
        # Load data
        self.df = pd.read_csv(self.log_path, encoding='utf-8')
        
        # Filter valid experiments
        self.df = self.df[self.df[self.target_metric].notna()].copy()
        
        if self.verbose:
            print(f"Loaded {len(self.df)} experiments with valid {self.target_metric} values")
        
        # Define parameters for analysis
        self.parameter_names = [
            'learning_rate', 'batch_size', 'epochs', 'k_folds',
            'conv_filters', 'dense_units', 'dropout_rates', 'l2_reg',
            'batch_norm', 'optimizer', 'early_stopping_patience',
            'lr_reduction_patience', 'class_weights', 'augment_fraction'
        ]
        
        # Filter available parameters
        available_params = [p for p in self.parameter_names if p in self.df.columns]
        self.parameter_names = available_params
        
        if self.verbose:
            print(f"Available parameters for analysis: {self.parameter_names}")
        
        # Preprocess parameters
        self.X_processed = self.preprocess_parameters()
        self.y = self.df[self.target_metric].values
        
        if self.verbose:
            print(f"Preprocessed data shape: {self.X_processed.shape}")
    
    def preprocess_parameters(self):
        """Convert parameters to numerical format suitable for ANOVA."""
        X = pd.DataFrame()
        
        for param in self.parameter_names:
            if param not in self.df.columns:
                continue
                
            values = self.df[param].copy()
            
            # Handle different parameter types
            if param in ['learning_rate', 'l2_reg', 'augment_fraction']:
                # Numeric parameters - use directly
                X[param] = pd.to_numeric(values, errors='coerce')
                
            elif param in ['batch_size', 'epochs', 'k_folds', 'early_stopping_patience', 'lr_reduction_patience']:
                # Integer parameters - use directly
                X[param] = pd.to_numeric(values, errors='coerce')
                
            elif param in ['batch_norm', 'class_weights']:
                # Boolean parameters - convert to 0/1
                X[param] = values.map({True: 1, False: 0, 'True': 1, 'False': 0})
                
            elif param in ['optimizer']:
                # Categorical parameters - label encode
                le = LabelEncoder()
                # Fill NaN with 'Unknown'
                values = values.fillna('Unknown').astype(str)
                X[param] = le.fit_transform(values)
                self.encoders[param] = le
                
            elif param in ['conv_filters', 'dense_units', 'dropout_rates']:
                # List parameters - convert to meaningful numeric features
                X = pd.concat([X, self.process_list_parameter(values, param)], axis=1)
                continue
                
            else:
                # Try to convert to numeric, otherwise label encode
                try:
                    X[param] = pd.to_numeric(values, errors='coerce')
                except:
                    le = LabelEncoder()
                    values = values.fillna('Unknown').astype(str)
                    X[param] = le.fit_transform(values)
                    self.encoders[param] = le
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Update parameter names to match processed columns
        self.parameter_names = list(X.columns)
        
        return X.values
    
    def process_list_parameter(self, values, param_name):
        """Process list-type parameters into numerical features."""
        import ast
        
        processed_df = pd.DataFrame()
        
        if param_name == 'conv_filters':
            # Extract meaningful features from conv filter lists
            features = {f'{param_name}_num_layers': [], f'{param_name}_total_filters': [], 
                       f'{param_name}_max_filters': [], f'{param_name}_filter_progression': []}
            
            for val in values:
                try:
                    if isinstance(val, str):
                        filters = ast.literal_eval(val)
                    else:
                        filters = val if isinstance(val, list) else [32, 64]  # default
                    
                    features[f'{param_name}_num_layers'].append(len(filters))
                    features[f'{param_name}_total_filters'].append(sum(filters))
                    features[f'{param_name}_max_filters'].append(max(filters))
                    
                    # Filter progression (ratio of last to first)
                    progression = filters[-1] / filters[0] if len(filters) > 1 else 1.0
                    features[f'{param_name}_filter_progression'].append(progression)
                    
                except:
                    # Default values for failed parsing
                    features[f'{param_name}_num_layers'].append(3)
                    features[f'{param_name}_total_filters'].append(112)  # 16+32+64
                    features[f'{param_name}_max_filters'].append(64)
                    features[f'{param_name}_filter_progression'].append(4.0)
            
            for feature_name, feature_values in features.items():
                processed_df[feature_name] = feature_values
                
        elif param_name == 'dense_units':
            # Extract features from dense unit lists
            features = {f'{param_name}_num_layers': [], f'{param_name}_total_units': [], 
                       f'{param_name}_max_units': []}
            
            for val in values:
                try:
                    if isinstance(val, str):
                        units = ast.literal_eval(val)
                    else:
                        units = val if isinstance(val, list) else [128, 64]  # default
                    
                    features[f'{param_name}_num_layers'].append(len(units))
                    features[f'{param_name}_total_units'].append(sum(units))
                    features[f'{param_name}_max_units'].append(max(units))
                    
                except:
                    # Default values
                    features[f'{param_name}_num_layers'].append(2)
                    features[f'{param_name}_total_units'].append(192)  # 128+64
                    features[f'{param_name}_max_units'].append(128)
            
            for feature_name, feature_values in features.items():
                processed_df[feature_name] = feature_values
                
        elif param_name == 'dropout_rates':
            # Extract dropout features
            features = {f'{param_name}_first': [], f'{param_name}_mean': [], f'{param_name}_max': []}
            
            for val in values:
                try:
                    if isinstance(val, str):
                        # Handle complex dropout rate formats like "[0.2, [0.3, 0.2]]"
                        rates = ast.literal_eval(val)
                        # Flatten nested lists
                        flat_rates = []
                        for rate in rates:
                            if isinstance(rate, list):
                                flat_rates.extend(rate)
                            else:
                                flat_rates.append(rate)
                        rates = [float(r) for r in flat_rates]
                    else:
                        rates = val if isinstance(val, list) else [0.2]  # default
                    
                    features[f'{param_name}_first'].append(rates[0])
                    features[f'{param_name}_mean'].append(np.mean(rates))
                    features[f'{param_name}_max'].append(max(rates))
                    
                except:
                    # Default values
                    features[f'{param_name}_first'].append(0.2)
                    features[f'{param_name}_mean'].append(0.25)
                    features[f'{param_name}_max'].append(0.3)
            
            for feature_name, feature_values in features.items():
                processed_df[feature_name] = feature_values
        
        return processed_df
    
    def compute_main_effects(self):
        """Compute main effects for each parameter."""
        if self.verbose:
            print("Computing main effects...")
            
        main_effects = {}
        
        for i, param in enumerate(self.parameter_names):
            # Create dataset with all parameters at their mean except the current one
            X_mean = np.tile(np.mean(self.X_processed, axis=0), (self.X_processed.shape[0], 1))
            X_varied = X_mean.copy()
            X_varied[:, i] = self.X_processed[:, i]
            
            # Fit model to predict performance from this single parameter variation
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(self.X_processed, self.y)
            
            # Predict with only this parameter varying
            y_pred_varied = model.predict(X_varied)
            y_pred_mean = model.predict(X_mean)
            
            # Main effect is the difference from mean
            main_effect = y_pred_varied - np.mean(y_pred_mean)
            main_effects[param] = main_effect
            
            if self.verbose:
                effect_magnitude = np.var(main_effect)
                print(f"  {param}: variance = {effect_magnitude:.6f}")
        
        return main_effects
    
    def compute_interaction_effects(self, max_order=2):
        """Compute interaction effects between parameters."""
        if self.verbose:
            print(f"Computing interaction effects (up to order {max_order})...")
        
        interaction_effects = {}
        
        # Two-way interactions
        if max_order >= 2:
            for i, j in combinations(range(len(self.parameter_names)), 2):
                param_i, param_j = self.parameter_names[i], self.parameter_names[j]
                
                # Create dataset with all parameters at mean except i and j
                X_mean = np.tile(np.mean(self.X_processed, axis=0), (self.X_processed.shape[0], 1))
                X_varied = X_mean.copy()
                X_varied[:, i] = self.X_processed[:, i]
                X_varied[:, j] = self.X_processed[:, j]
                
                # Fit model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(self.X_processed, self.y)
                
                # Compute interaction effect
                y_pred_varied = model.predict(X_varied)
                y_pred_mean = model.predict(X_mean)
                
                # Individual effects
                X_i_only = X_mean.copy()
                X_i_only[:, i] = self.X_processed[:, i]
                y_pred_i = model.predict(X_i_only)
                
                X_j_only = X_mean.copy()
                X_j_only[:, j] = self.X_processed[:, j]
                y_pred_j = model.predict(X_j_only)
                
                # Interaction = Combined - Individual effects
                interaction = (y_pred_varied - np.mean(y_pred_mean)) - \
                            ((y_pred_i - np.mean(y_pred_mean)) + (y_pred_j - np.mean(y_pred_mean)))
                
                interaction_effects[f"{param_i}_x_{param_j}"] = interaction
                
                if self.verbose:
                    effect_magnitude = np.var(interaction)
                    print(f"  {param_i} × {param_j}: variance = {effect_magnitude:.6f}")
        
        return interaction_effects
    
    def perform_anova_decomposition(self, max_interaction_order=2):
        """Perform complete ANOVA decomposition."""
        print("Starting ANOVA Decomposition...")
        
        # Global mean
        f0 = np.mean(self.y)
        
        # Main effects
        main_effects = self.compute_main_effects()
        
        # Interaction effects
        interaction_effects = self.compute_interaction_effects(max_interaction_order)
        
        # Store results
        self.anova_components = {
            'global_mean': f0,
            'main_effects': main_effects,
            'interaction_effects': interaction_effects
        }
        
        # Compute variance explained by each component
        self.compute_variance_explained()
        
        if self.verbose:
            print("ANOVA decomposition complete!")
    
    def compute_variance_explained(self):
        """Compute variance explained by each ANOVA component."""
        total_variance = np.var(self.y)
        
        # Main effects variance
        main_variances = {}
        total_main_variance = 0
        for param, effect in self.anova_components['main_effects'].items():
            var = np.var(effect)
            main_variances[param] = var / total_variance
            total_main_variance += var
        
        # Interaction effects variance
        interaction_variances = {}
        total_interaction_variance = 0
        for param_pair, effect in self.anova_components['interaction_effects'].items():
            var = np.var(effect)
            interaction_variances[param_pair] = var / total_variance
            total_interaction_variance += var
        
        # Residual variance
        explained_variance = total_main_variance + total_interaction_variance
        residual_variance = max(0, total_variance - explained_variance)
        
        self.variance_explained = {
            'main_effects': main_variances,
            'interaction_effects': interaction_variances,
            'total_main': total_main_variance / total_variance,
            'total_interaction': total_interaction_variance / total_variance,
            'residual': residual_variance / total_variance,
            'total_explained': explained_variance / total_variance
        }
    
    def plot_variance_decomposition(self):
        """Create visualization of variance decomposition."""
        if self.verbose:
            print("Creating variance decomposition plots...")
        
        # Main effects plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Main effects bar plot
        main_effects_df = pd.DataFrame({
            'Parameter': list(self.variance_explained['main_effects'].keys()),
            'Variance_Explained': list(self.variance_explained['main_effects'].values())
        })
        main_effects_df = main_effects_df.sort_values('Variance_Explained', ascending=True)
        
        axes[0, 0].barh(main_effects_df['Parameter'], main_effects_df['Variance_Explained'])
        axes[0, 0].set_xlabel('Proportion of Variance Explained')
        axes[0, 0].set_title('Main Effects: Parameter Importance')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Interaction effects plot (top 10)
        interaction_df = pd.DataFrame({
            'Interaction': list(self.variance_explained['interaction_effects'].keys()),
            'Variance_Explained': list(self.variance_explained['interaction_effects'].values())
        })
        interaction_df = interaction_df.sort_values('Variance_Explained', ascending=False).head(10)
        
        axes[0, 1].barh(interaction_df['Interaction'], interaction_df['Variance_Explained'])
        axes[0, 1].set_xlabel('Proportion of Variance Explained')
        axes[0, 1].set_title('Top 10 Interaction Effects')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Overall variance breakdown pie chart
        breakdown_data = [
            self.variance_explained['total_main'],
            self.variance_explained['total_interaction'], 
            self.variance_explained['residual']
        ]
        breakdown_labels = ['Main Effects', 'Interactions', 'Residual']
        
        axes[1, 0].pie(breakdown_data, labels=breakdown_labels, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Overall Variance Decomposition')
        
        # 4. Cumulative importance plot
        all_effects = pd.concat([
            main_effects_df[['Parameter', 'Variance_Explained']].rename(columns={'Parameter': 'Effect'}),
            interaction_df[['Interaction', 'Variance_Explained']].rename(columns={'Interaction': 'Effect'})
        ]).sort_values('Variance_Explained', ascending=False)
        
        cumsum = np.cumsum(all_effects['Variance_Explained'])
        axes[1, 1].plot(range(1, len(cumsum) + 1), cumsum, 'o-')
        axes[1, 1].set_xlabel('Number of Effects')
        axes[1, 1].set_ylabel('Cumulative Variance Explained')
        axes[1, 1].set_title('Cumulative Importance of Effects')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'anova_variance_decomposition.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        if self.verbose:
            print(f"Saved: {plot_path}")
        
        plt.close()
    
    def generate_anova_report(self):
        """Generate comprehensive ANOVA analysis report."""
        if self.verbose:
            print("Generating ANOVA report...")
        
        report_lines = []
        report_lines.append("# ANOVA Decomposition Report\n\n")
        report_lines.append(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append(f"**Target Metric:** {self.target_metric}\n")
        report_lines.append(f"**Total Experiments:** {len(self.df)}\n")
        report_lines.append(f"**Parameters Analyzed:** {len(self.parameter_names)}\n\n")
        
        # Overall summary
        report_lines.append("## Variance Decomposition Summary\n\n")
        report_lines.append(f"- **Main Effects:** {self.variance_explained['total_main']:.1%} of total variance\n")
        report_lines.append(f"- **Interaction Effects:** {self.variance_explained['total_interaction']:.1%} of total variance\n")
        report_lines.append(f"- **Residual (Unexplained):** {self.variance_explained['residual']:.1%} of total variance\n")
        report_lines.append(f"- **Total Explained:** {self.variance_explained['total_explained']:.1%} of total variance\n\n")
        
        # Main effects ranking
        report_lines.append("## Main Effects Ranking\n\n")
        main_effects_sorted = sorted(self.variance_explained['main_effects'].items(), 
                                   key=lambda x: x[1], reverse=True)
        
        for i, (param, variance) in enumerate(main_effects_sorted, 1):
            report_lines.append(f"{i}. **{param}**: {variance:.3%} of variance\n")
        
        report_lines.append("\n")
        
        # Top interaction effects
        report_lines.append("## Top Interaction Effects\n\n")
        interaction_effects_sorted = sorted(self.variance_explained['interaction_effects'].items(), 
                                          key=lambda x: x[1], reverse=True)
        
        for i, (param_pair, variance) in enumerate(interaction_effects_sorted[:10], 1):
            report_lines.append(f"{i}. **{param_pair}**: {variance:.3%} of variance\n")
        
        report_lines.append("\n")
        
        # Interpretation
        report_lines.append("## Interpretation\n\n")
        
        # Most important parameter
        top_param, top_variance = main_effects_sorted[0]
        report_lines.append(f"- **Most influential parameter:** {top_param} ({top_variance:.1%} of variance)\n")
        
        # Interaction importance
        if self.variance_explained['total_interaction'] > 0.1:
            report_lines.append(f"- **Significant interactions detected:** {self.variance_explained['total_interaction']:.1%} of variance from parameter interactions\n")
        else:
            report_lines.append("- **Limited interaction effects:** Parameters appear to act mostly independently\n")
        
        # Model adequacy
        if self.variance_explained['total_explained'] > 0.8:
            report_lines.append("- **Good model fit:** Parameters explain most of the performance variation\n")
        elif self.variance_explained['total_explained'] > 0.6:
            report_lines.append("- **Moderate model fit:** Additional factors may influence performance\n")
        else:
            report_lines.append("- **Limited model fit:** Consider additional parameters or non-linear relationships\n")
        
        # Save report
        report_path = self.output_dir / 'anova_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        
        if self.verbose:
            print(f"Report saved: {report_path}")
    
    def run_analysis(self, max_interaction_order=2):
        """Run complete ANOVA decomposition analysis."""
        print("Starting ANOVA Decomposition Analysis...")
        print(f"Target metric: {self.target_metric}")
        print(f"Max interaction order: {max_interaction_order}")
        print()
        
        # Perform decomposition
        self.perform_anova_decomposition(max_interaction_order)
        
        # Generate visualizations
        self.plot_variance_decomposition()
        
        # Generate report
        self.generate_anova_report()
        
        print(f"\nANOVA analysis complete! Results saved to: {self.output_dir}")
        
        # Print quick summary
        print("\nQuick Summary:")
        print(f"- Main effects explain {self.variance_explained['total_main']:.1%} of variance")
        print(f"- Interactions explain {self.variance_explained['total_interaction']:.1%} of variance")
        print(f"- Total explained: {self.variance_explained['total_explained']:.1%}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ANOVA decomposition for hyperparameter sensitivity')
    parser.add_argument('--log_path', type=str, help='Path to experiment log CSV file')
    parser.add_argument('--output_dir', type=str, help='Output directory for analysis results')
    parser.add_argument('--target_metric', type=str, default='mean_val_accuracy',
                       help='Target metric to analyze (default: mean_val_accuracy)')
    parser.add_argument('--max_interaction_order', type=int, default=2,
                       help='Maximum order of interactions to compute (default: 2)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = ANOVADecomposer(
            log_path=args.log_path,
            output_dir=args.output_dir,
            target_metric=args.target_metric,
            verbose=args.verbose
        )
        
        # Run analysis
        analyzer.run_analysis(max_interaction_order=args.max_interaction_order)
        
        print("\n✅ ANOVA decomposition completed successfully!")
        
    except Exception as e:
        print(f"\n❌ ANOVA analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()