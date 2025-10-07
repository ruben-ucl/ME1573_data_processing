#!/usr/bin/env python3
"""
Hyperparameter Optimization Results Visualizer

This script creates comprehensive visualizations of hyperparameter optimization results
to help understand parameter importance, relationships, and optimization trends.

Features:
- Parameter importance analysis
- Performance correlation matrices  
- 2D parameter interaction plots
- Optimization timeline
- Best configuration comparison
- Statistical summaries

Author: AI Assistant
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import ast

# Set up matplotlib for better display
plt.style.use('default')
sns.set_palette("husl")

class HyperoptVisualizer:
    """Visualizes hyperparameter optimization results."""
    
    def __init__(self, experiment_log_path, output_dir, verbose=False):
        self.experiment_log_path = Path(experiment_log_path)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_hyperopt_data(self):
        """Load and preprocess hyperopt data from experiment log."""
        if not self.experiment_log_path.exists():
            raise FileNotFoundError(f"Experiment log not found: {self.experiment_log_path}")
        
        df = pd.read_csv(self.experiment_log_path, encoding='utf-8')
        
        # Filter to hyperopt runs only
        hyperopt_df = df[df['source'] == 'hyperopt'].copy()
        
        if len(hyperopt_df) == 0:
            raise ValueError("No hyperopt entries found in experiment log")
        
        # Parse list columns
        for col in ['conv_filters', 'dense_units']:
            if col in hyperopt_df.columns:
                hyperopt_df[col] = hyperopt_df[col].apply(self._safe_parse_list)
        
        # Parse dropout rates
        if 'dropout_rates' in hyperopt_df.columns:
            hyperopt_df['conv_dropout'] = hyperopt_df['dropout_rates'].apply(
                lambda x: self._extract_conv_dropout(x)
            )
            hyperopt_df['dense_dropout_avg'] = hyperopt_df['dropout_rates'].apply(
                lambda x: self._extract_dense_dropout_avg(x)
            )
        
        # Convert architecture to string representations
        hyperopt_df['conv_arch'] = hyperopt_df['conv_filters'].apply(str)
        hyperopt_df['dense_arch'] = hyperopt_df['dense_units'].apply(str)
        
        # Add derived metrics
        hyperopt_df['accuracy_stability'] = hyperopt_df['mean_val_accuracy'] / (hyperopt_df['std_val_accuracy'] + 1e-8)
        hyperopt_df['efficiency'] = hyperopt_df['mean_val_accuracy'] / (hyperopt_df['total_training_time_minutes'] + 1e-8)
        
        if self.verbose:
            print(f"Loaded {len(hyperopt_df)} hyperopt experiments")
            print(f"Date range: {hyperopt_df['timestamp'].min()} to {hyperopt_df['timestamp'].max()}")
        
        return hyperopt_df
    
    def _safe_parse_list(self, value):
        """Safely parse list from string representation."""
        if isinstance(value, list):
            return value
        try:
            return ast.literal_eval(str(value))
        except:
            return str(value)
    
    def _extract_conv_dropout(self, dropout_str):
        """Extract conv dropout from dropout_rates string."""
        try:
            dropout_list = ast.literal_eval(str(dropout_str))
            return float(dropout_list[0]) if isinstance(dropout_list, list) else float(dropout_str)
        except:
            return 0.2  # default
    
    def _extract_dense_dropout_avg(self, dropout_str):
        """Extract average dense dropout from dropout_rates string."""
        try:
            dropout_list = ast.literal_eval(str(dropout_str))
            if isinstance(dropout_list, list) and len(dropout_list) > 1:
                dense_dropout = dropout_list[1]
                if isinstance(dense_dropout, list):
                    return np.mean(dense_dropout)
                else:
                    return float(dense_dropout)
            return 0.2  # default
        except:
            return 0.2  # default
    
    def plot_parameter_importance(self, df):
        """Create parameter importance plot based on correlation with accuracy."""
        # Select numeric parameters
        param_cols = ['learning_rate', 'batch_size', 'conv_dropout', 'dense_dropout_avg', 
                     'l2_reg', 'total_training_time_minutes']
        param_cols = [col for col in param_cols if col in df.columns]
        
        # Calculate correlations with accuracy
        correlations = {}
        for col in param_cols:
            if df[col].dtype in ['float64', 'int64']:
                corr = df[col].corr(df['mean_val_accuracy'])
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
        
        if not correlations:
            print("Warning: No numeric correlations found")
            return
        
        # Create importance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        params = list(correlations.keys())
        importance = list(correlations.values())
        
        bars = ax.barh(params, importance)
        ax.set_xlabel('Absolute Correlation with Validation Accuracy')
        ax.set_title('Hyperparameter Importance Analysis')
        ax.grid(True, alpha=0.3)
        
        # Color bars by importance
        max_importance = max(importance)
        for bar, imp in zip(bars, importance):
            bar.set_color(plt.cm.viridis(imp / max_importance))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print("Parameter importance analysis saved")
    
    def plot_optimization_timeline(self, df):
        """Plot optimization progress over time."""
        # Convert timestamp to datetime
        df_sorted = df.copy()
        df_sorted['datetime'] = pd.to_datetime(df_sorted['timestamp'])
        df_sorted = df_sorted.sort_values('datetime')
        
        # Calculate running best
        df_sorted['running_best'] = df_sorted['mean_val_accuracy'].cummax()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Accuracy over time
        ax1.scatter(df_sorted['datetime'], df_sorted['mean_val_accuracy'], 
                   alpha=0.7, s=50, label='Individual experiments')
        ax1.plot(df_sorted['datetime'], df_sorted['running_best'], 
                color='red', linewidth=2, label='Running best')
        ax1.set_ylabel('Validation Accuracy')
        ax1.set_title('Hyperparameter Optimization Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training time distribution
        ax2.bar(range(len(df_sorted)), df_sorted['total_training_time_minutes'], 
               alpha=0.7, color='orange')
        ax2.set_xlabel('Experiment Order')
        ax2.set_ylabel('Training Time (minutes)')
        ax2.set_title('Training Time per Experiment')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'optimization_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print("Optimization timeline saved")
    
    def plot_parameter_relationships(self, df):
        """Create correlation matrix and pairwise plots."""
        # Select numeric parameters
        param_cols = ['learning_rate', 'batch_size', 'conv_dropout', 'dense_dropout_avg', 
                     'l2_reg', 'mean_val_accuracy', 'std_val_accuracy', 'total_training_time_minutes']
        param_cols = [col for col in param_cols if col in df.columns and df[col].dtype in ['float64', 'int64']]
        
        if len(param_cols) < 3:
            print("Warning: Not enough numeric parameters for correlation analysis")
            return
        
        # Create correlation matrix
        corr_matrix = df[param_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax, fmt='.3f')
        ax.set_title('Hyperparameter Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print("Parameter correlation matrix saved")
    
    def plot_performance_distributions(self, df):
        """Plot distributions of key performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy distribution
        axes[0,0].hist(df['mean_val_accuracy'], bins=15, alpha=0.7, color='blue', edgecolor='black')
        axes[0,0].axvline(df['mean_val_accuracy'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {df["mean_val_accuracy"].mean():.4f}')
        axes[0,0].set_xlabel('Mean Validation Accuracy')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Accuracy Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Training time distribution
        axes[0,1].hist(df['total_training_time_minutes'], bins=15, alpha=0.7, color='orange', edgecolor='black')
        axes[0,1].axvline(df['total_training_time_minutes'].mean(), color='red', linestyle='--',
                         label=f'Mean: {df["total_training_time_minutes"].mean():.1f}m')
        axes[0,1].set_xlabel('Training Time (minutes)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Training Time Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Accuracy vs Training Time
        axes[1,0].scatter(df['total_training_time_minutes'], df['mean_val_accuracy'], 
                         alpha=0.7, s=60)
        axes[1,0].set_xlabel('Training Time (minutes)')
        axes[1,0].set_ylabel('Mean Validation Accuracy')
        axes[1,0].set_title('Accuracy vs Training Time')
        axes[1,0].grid(True, alpha=0.3)
        
        # Stability analysis (accuracy vs std)
        axes[1,1].scatter(df['std_val_accuracy'], df['mean_val_accuracy'], 
                         alpha=0.7, s=60, c=df['total_training_time_minutes'], 
                         cmap='viridis')
        axes[1,1].set_xlabel('Standard Deviation (Accuracy)')
        axes[1,1].set_ylabel('Mean Validation Accuracy')
        axes[1,1].set_title('Accuracy vs Stability')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print("Performance distributions saved")
    
    def plot_top_configurations(self, df, top_n=5):
        """Plot comparison of top N configurations."""
        # Get top configurations by accuracy
        top_configs = df.nlargest(top_n, 'mean_val_accuracy')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Accuracy comparison
        x_pos = range(len(top_configs))
        axes[0,0].bar(x_pos, top_configs['mean_val_accuracy'], 
                     yerr=top_configs['std_val_accuracy'], capsize=5, alpha=0.7)
        axes[0,0].set_xlabel('Configuration Rank')
        axes[0,0].set_ylabel('Validation Accuracy')
        axes[0,0].set_title(f'Top {top_n} Configurations by Accuracy')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels([f"{row['version']}" for _, row in top_configs.iterrows()])
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Learning rate comparison
        axes[0,1].bar(x_pos, top_configs['learning_rate'], alpha=0.7, color='green')
        axes[0,1].set_xlabel('Configuration Rank')
        axes[0,1].set_ylabel('Learning Rate')
        axes[0,1].set_title('Learning Rates of Top Configurations')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels([f"{row['version']}" for _, row in top_configs.iterrows()])
        axes[0,1].set_yscale('log')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Batch size comparison
        axes[1,0].bar(x_pos, top_configs['batch_size'], alpha=0.7, color='purple')
        axes[1,0].set_xlabel('Configuration Rank')
        axes[1,0].set_ylabel('Batch Size')
        axes[1,0].set_title('Batch Sizes of Top Configurations')
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels([f"{row['version']}" for _, row in top_configs.iterrows()])
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Training time vs Accuracy for top configs
        axes[1,1].scatter(top_configs['total_training_time_minutes'], 
                         top_configs['mean_val_accuracy'], 
                         s=100, alpha=0.7, c=range(len(top_configs)), cmap='viridis')
        axes[1,1].set_xlabel('Training Time (minutes)')
        axes[1,1].set_ylabel('Mean Validation Accuracy')
        axes[1,1].set_title('Efficiency of Top Configurations')
        
        # Add labels for each point
        for i, (_, row) in enumerate(top_configs.iterrows()):
            axes[1,1].annotate(row['version'], 
                              (row['total_training_time_minutes'], row['mean_val_accuracy']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_configurations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Top {top_n} configurations comparison saved")
    
    def plot_learning_rate_analysis(self, df):
        """Detailed analysis of learning rate impact."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Learning rate vs Accuracy
        lr_values = df['learning_rate'].unique()
        lr_performance = []
        
        for lr in sorted(lr_values):
            lr_experiments = df[df['learning_rate'] == lr]
            lr_performance.append({
                'lr': lr,
                'mean_acc': lr_experiments['mean_val_accuracy'].mean(),
                'std_acc': lr_experiments['mean_val_accuracy'].std(),
                'count': len(lr_experiments),
                'best_acc': lr_experiments['mean_val_accuracy'].max()
            })
        
        lr_df = pd.DataFrame(lr_performance)
        
        # Bar plot with error bars
        x_pos = range(len(lr_df))
        axes[0].bar(x_pos, lr_df['mean_acc'], yerr=lr_df['std_acc'], 
                   capsize=5, alpha=0.7)
        axes[0].set_xlabel('Learning Rate')
        axes[0].set_ylabel('Mean Validation Accuracy')
        axes[0].set_title('Learning Rate Impact Analysis')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([f"{lr:.4f}" for lr in lr_df['lr']], rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Add count labels
        for i, (_, row) in enumerate(lr_df.iterrows()):
            axes[0].text(i, row['mean_acc'] + row['std_acc'] + 0.01, 
                        f"n={row['count']}", ha='center', fontsize=8)
        
        # Plot 2: Learning rate vs Training time
        axes[1].scatter(df['learning_rate'], df['total_training_time_minutes'], 
                       alpha=0.7, s=60, c=df['mean_val_accuracy'], cmap='viridis')
        axes[1].set_xlabel('Learning Rate')
        axes[1].set_ylabel('Training Time (minutes)')
        axes[1].set_title('Learning Rate vs Training Time')
        axes[1].set_xscale('log')
        axes[1].grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
        cbar.set_label('Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_rate_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print("Learning rate analysis saved")
    
    def plot_architecture_comparison(self, df):
        """Compare different architecture configurations."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Conv architecture comparison
        conv_performance = df.groupby('conv_arch').agg({
            'mean_val_accuracy': ['mean', 'std', 'count'],
            'total_training_time_minutes': 'mean'
        }).round(4)
        
        conv_performance.columns = ['mean_acc', 'std_acc', 'count', 'avg_time']
        conv_performance = conv_performance.reset_index()
        
        x_pos = range(len(conv_performance))
        axes[0].bar(x_pos, conv_performance['mean_acc'], 
                   yerr=conv_performance['std_acc'], capsize=5, alpha=0.7)
        axes[0].set_xlabel('Conv Architecture')
        axes[0].set_ylabel('Mean Validation Accuracy')
        axes[0].set_title('Conv Filter Architecture Comparison')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(conv_performance['conv_arch'], rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Add count labels
        for i, (_, row) in enumerate(conv_performance.iterrows()):
            axes[0].text(i, row['mean_acc'] + row['std_acc'] + 0.01, 
                        f"n={row['count']}", ha='center', fontsize=8)
        
        # Plot 2: Dense architecture comparison
        dense_performance = df.groupby('dense_arch').agg({
            'mean_val_accuracy': ['mean', 'std', 'count'],
            'total_training_time_minutes': 'mean'
        }).round(4)
        
        dense_performance.columns = ['mean_acc', 'std_acc', 'count', 'avg_time']
        dense_performance = dense_performance.reset_index()
        
        x_pos = range(len(dense_performance))
        axes[1].bar(x_pos, dense_performance['mean_acc'], 
                   yerr=dense_performance['std_acc'], capsize=5, alpha=0.7, color='green')
        axes[1].set_xlabel('Dense Architecture')
        axes[1].set_ylabel('Mean Validation Accuracy')
        axes[1].set_title('Dense Layer Architecture Comparison')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(dense_performance['dense_arch'], rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Add count labels
        for i, (_, row) in enumerate(dense_performance.iterrows()):
            axes[1].text(i, row['mean_acc'] + row['std_acc'] + 0.01, 
                        f"n={row['count']}", ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'architecture_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print("Architecture comparison saved")
    
    def generate_summary_report(self, df):
        """Generate a comprehensive text summary report."""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("HYPERPARAMETER OPTIMIZATION SUMMARY REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total experiments: {len(df)}")
        report_lines.append("")
        
        # Overall statistics
        report_lines.append("PERFORMANCE OVERVIEW:")
        report_lines.append(f"  Best accuracy: {df['mean_val_accuracy'].max():.4f} (version {df.loc[df['mean_val_accuracy'].idxmax(), 'version']})")
        report_lines.append(f"  Worst accuracy: {df['mean_val_accuracy'].min():.4f} (version {df.loc[df['mean_val_accuracy'].idxmin(), 'version']})")
        report_lines.append(f"  Mean accuracy: {df['mean_val_accuracy'].mean():.4f} ± {df['mean_val_accuracy'].std():.4f}")
        report_lines.append(f"  Total training time: {df['total_training_time_minutes'].sum():.1f} minutes")
        report_lines.append(f"  Average time per experiment: {df['total_training_time_minutes'].mean():.1f} minutes")
        report_lines.append("")
        
        # Parameter analysis
        report_lines.append("PARAMETER ANALYSIS:")
        
        # Learning rate
        lr_stats = df.groupby('learning_rate')['mean_val_accuracy'].agg(['mean', 'std', 'count'])
        report_lines.append("  Learning Rate Performance:")
        for lr, stats in lr_stats.iterrows():
            report_lines.append(f"    {lr:.4f}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
        
        # Batch size  
        batch_stats = df.groupby('batch_size')['mean_val_accuracy'].agg(['mean', 'std', 'count'])
        report_lines.append("  Batch Size Performance:")
        for batch, stats in batch_stats.iterrows():
            report_lines.append(f"    {batch}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
        
        report_lines.append("")
        
        # Top 5 configurations
        top_5 = df.nlargest(5, 'mean_val_accuracy')
        report_lines.append("TOP 5 CONFIGURATIONS:")
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            report_lines.append(f"  #{i}. {row['version']} - Accuracy: {row['mean_val_accuracy']:.4f}")
            report_lines.append(f"      LR: {row['learning_rate']}, Batch: {row['batch_size']}, Time: {row['total_training_time_minutes']:.1f}m")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        # Save report
        report_file = self.output_dir / 'hyperopt_summary_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        if self.verbose:
            print("Summary report saved")
        
        return report_lines
    
    def create_all_visualizations(self):
        """Create all visualization plots and reports."""
        print(f"Creating hyperopt visualizations...")
        
        # Load data
        df = self.load_hyperopt_data()
        
        # Generate all plots
        self.plot_parameter_importance(df)
        self.plot_optimization_timeline(df)
        self.plot_parameter_relationships(df) 
        self.plot_performance_distributions(df)
        self.plot_top_configurations(df)
        
        # Generate summary report
        report_lines = self.generate_summary_report(df)
        
        print(f"\n✅ Visualizations completed!")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Files created:")
        print(f"     - parameter_importance.png")
        print(f"     - optimization_timeline.png") 
        print(f"     - parameter_correlations.png")
        print(f"     - performance_distributions.png")
        print(f"     - architecture_comparison.png")
        print(f"     - top_configurations.png")
        print(f"     - hyperopt_summary_report.txt")
        
        # Display key insights
        print(f"\n📊 KEY INSIGHTS:")
        best_config = df.loc[df['mean_val_accuracy'].idxmax()]
        print(f"   Best configuration: {best_config['version']} ({best_config['mean_val_accuracy']:.4f} accuracy)")
        print(f"   Most efficient: {df.loc[df['efficiency'].idxmax(), 'version']} (highest accuracy/time ratio)")
        print(f"   Optimization range: {df['mean_val_accuracy'].min():.4f} - {df['mean_val_accuracy'].max():.4f}")
        
        return df

def main():
    parser = argparse.ArgumentParser(description='Visualize hyperparameter optimization results')
    parser.add_argument('--experiment_log', type=str, 
                       default='ml/logs/experiment_log.csv',
                       help='Path to experiment log CSV file')
    parser.add_argument('--output_dir', type=str,
                       default='ml/hyperopt_analysis',
                       help='Output directory for visualizations')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    try:
        # Create visualizer
        visualizer = HyperoptVisualizer(
            experiment_log_path=args.experiment_log,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        # Generate all visualizations
        visualizer.create_all_visualizations()
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()