#!/usr/bin/env python3
"""
DoE Experiment Generator for Neural Network Hyperparameter Optimization

Generates experimental designs following best practices for statistical analysis.
Supports factorial designs, response surface methodology, and Latin hypercube sampling.
"""

import argparse
import json
import warnings
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Local imports
from config import get_pd_experiment_log_path, get_cwt_experiment_log_path, PD_OUTPUTS_DIR


class DoEGenerator:
    """Generate statistically sound experimental designs for hyperparameter optimization."""
    
    def __init__(self, mode='pd', output_dir=None, verbose=False):
        self.mode = mode
        self.verbose = verbose
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = PD_OUTPUTS_DIR / 'doe_experiments'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameter definitions based on mode
        if mode == 'cwt':
            self.parameter_space = self._get_cwt_parameter_space()
        else:
            self.parameter_space = self._get_pd_parameter_space()
        
        if self.verbose:
            print(f"DoE Generator initialized for {mode.upper()} mode")
            print(f"Output directory: {self.output_dir}")
    
    def _get_pd_parameter_space(self):
        """Define parameter space for PD signal classification."""
        return {
            'learning_rate': {
                'type': 'continuous',
                'levels': [0.0001, 0.0005, 0.001, 0.002, 0.005],
                'range': (0.0001, 0.01),
                'log_scale': True
            },
            'batch_size': {
                'type': 'discrete',
                'levels': [8, 16, 32, 64],
                'range': (8, 64)
            },
            'conv_filters': {
                'type': 'categorical',
                'levels': [
                    '[16, 32]',
                    '[16, 32, 64]', 
                    '[32, 64]',
                    '[32, 64, 128]',
                    '[16, 32, 64, 128]'
                ]
            },
            'dense_units': {
                'type': 'categorical',
                'levels': [
                    '[64]',
                    '[128]',
                    '[128, 64]',
                    '[256]',
                    '[256, 128]'
                ]
            },
            'conv_dropout': {
                'type': 'continuous',
                'levels': [0.0, 0.1, 0.2, 0.3, 0.5],
                'range': (0.0, 0.5)
            },
            'dense_dropout': {
                'type': 'categorical',
                'levels': [
                    '[0.0]',
                    '[0.2]',
                    '[0.3, 0.2]',
                    '[0.4, 0.3]',
                    '[0.5, 0.3]'
                ]
            },
            'l2_regularization': {
                'type': 'continuous',
                'levels': [0.0, 0.0001, 0.001, 0.01, 0.1],
                'range': (0.0, 0.1),
                'log_scale': True
            },
            'optimizer': {
                'type': 'categorical',
                'levels': ['adam', 'sgd', 'rmsprop', 'adamw']
            },
            'use_batch_norm': {
                'type': 'categorical',
                'levels': [True, False]
            }
        }
    
    def _get_cwt_parameter_space(self):
        """Define parameter space for CWT image classification."""
        return {
            'learning_rate': {
                'type': 'continuous',
                'levels': [0.0001, 0.0005, 0.001, 0.002, 0.005],
                'range': (0.0001, 0.01),
                'log_scale': True
            },
            'batch_size': {
                'type': 'discrete', 
                'levels': [16, 32, 64, 128],
                'range': (16, 128)
            },
            'conv_filters': {
                'type': 'categorical',
                'levels': [
                    '[16, 16, 32, 32]',
                    '[16, 16, 32, 32, 64]',
                    '[16, 16, 32, 32, 64, 64]',
                    '[32, 32, 64, 64, 128]',
                    '[32, 32, 64, 64, 128, 128]'
                ]
            },
            'dense_units': {
                'type': 'categorical',
                'levels': [
                    '[64]',
                    '[128]',
                    '[256]',
                    '[128, 64]',
                    '[256, 128]'
                ]
            },
            'conv_dropout': {
                'type': 'continuous',
                'levels': [0.0, 0.1, 0.2, 0.3, 0.5],
                'range': (0.0, 0.5)
            },
            'dense_dropout': {
                'type': 'categorical',
                'levels': [
                    '[0.0]',
                    '[0.2]', 
                    '[0.3]',
                    '[0.4]',
                    '[0.5]',
                    '[0.3, 0.2]',  # Multi-layer options
                    '[0.4, 0.3]'
                ]
            },
            'l2_regularization': {
                'type': 'continuous',
                'levels': [0.0, 0.0001, 0.001, 0.01],
                'range': (0.0, 0.01),
                'log_scale': True
            },
            'use_batch_norm': {
                'type': 'categorical',
                'levels': [True, False]
            },
            'pool_layers': {
                'type': 'categorical',
                'levels': [
                    '[2, 5]',
                    '[1, 3, 5]',
                    '[2, 4]',
                    '[1, 2, 4]'
                ]
            }
        }
    
    def generate_factorial_design(self, factors=None, resolution=None, center_points=4, 
                                 axial_points=True, randomize=True):
        """Generate fractional factorial design for screening experiments."""
        if factors is None:
            # Select top factors based on importance (placeholder logic)
            factors = list(self.parameter_space.keys())[:6]  # Top 6 factors
        
        if self.verbose:
            print(f"Generating factorial design for factors: {factors}")
        
        # Get factor levels
        factor_levels = {}
        for factor in factors:
            param_info = self.parameter_space[factor]
            if param_info['type'] == 'categorical':
                # For categorical, use discrete levels
                factor_levels[factor] = param_info['levels']
            else:
                # For continuous, use 3 levels (low, medium, high)
                levels = param_info['levels']
                if len(levels) >= 3:
                    factor_levels[factor] = [levels[0], levels[len(levels)//2], levels[-1]]
                else:
                    factor_levels[factor] = levels
        
        # Generate full factorial combinations
        factor_names = list(factor_levels.keys())
        level_combinations = list(product(*[factor_levels[f] for f in factor_names]))
        
        # For screening, use fractional factorial if too many combinations
        if len(level_combinations) > 64:  # Limit to reasonable size
            # Use every nth combination for fractional factorial
            step_size = max(1, len(level_combinations) // 32)
            level_combinations = level_combinations[::step_size]
            
            if self.verbose:
                print(f"Using fractional factorial: {len(level_combinations)} experiments")
        
        # Create experiment dataframe
        experiments = []
        for i, combination in enumerate(level_combinations):
            experiment = {'experiment_id': f'factorial_{i+1:03d}', 'design_type': 'factorial'}
            for factor, level in zip(factor_names, combination):
                experiment[factor] = level
            experiments.append(experiment)
        
        # Add center points for error estimation
        if center_points > 0:
            center_config = {}
            for factor in factor_names:
                param_info = self.parameter_space[factor]
                if param_info['type'] == 'categorical':
                    # Use middle level for categorical
                    levels = param_info['levels']
                    center_config[factor] = levels[len(levels)//2]
                else:
                    # Use middle level for continuous
                    levels = param_info['levels']
                    center_config[factor] = levels[len(levels)//2]
            
            for i in range(center_points):
                experiment = {'experiment_id': f'center_{i+1:03d}', 'design_type': 'center_point'}
                experiment.update(center_config)
                experiments.append(experiment)
        
        # Add default fixed parameters
        for experiment in experiments:
            experiment.update(self._get_default_config())
        
        # Randomize experiment order
        if randomize:
            np.random.shuffle(experiments)
        
        df = pd.DataFrame(experiments)
        
        if self.verbose:
            print(f"Generated {len(experiments)} experiments ({len(level_combinations)} factorial + {center_points} center points)")
        
        return df
    
    def generate_response_surface_design(self, factors=None, alpha=None, center_points=6):
        """Generate Central Composite Design for response surface methodology."""
        if factors is None:
            # Use top 3-4 factors for RSM
            factors = list(self.parameter_space.keys())[:3]
        
        if alpha is None:
            alpha = len(factors) ** 0.5  # Rotatable design
        
        if self.verbose:
            print(f"Generating CCD for factors: {factors} (α = {alpha:.2f})")
        
        # Get factor ranges for continuous variables
        factor_info = {}
        for factor in factors:
            param_info = self.parameter_space[factor]
            if param_info['type'] == 'categorical':
                factor_info[factor] = {'type': 'categorical', 'levels': param_info['levels']}
            else:
                levels = param_info['levels']
                factor_info[factor] = {
                    'type': 'continuous',
                    'low': levels[0],
                    'high': levels[-1],
                    'center': levels[len(levels)//2]
                }
        
        experiments = []
        exp_id = 1
        
        # Factorial points (2^k)
        for combination in product([-1, 1], repeat=len(factors)):
            experiment = {'experiment_id': f'ccd_fact_{exp_id:03d}', 'design_type': 'factorial'}
            for i, factor in enumerate(factors):
                info = factor_info[factor]
                if info['type'] == 'categorical':
                    # Use low/high levels for categorical
                    level_idx = 0 if combination[i] == -1 else -1
                    experiment[factor] = info['levels'][level_idx]
                else:
                    # Use coded values for continuous
                    if combination[i] == -1:
                        experiment[factor] = info['low']
                    else:
                        experiment[factor] = info['high']
            experiments.append(experiment)
            exp_id += 1
        
        # Axial points (2k)
        for i, factor in enumerate(factors):
            info = factor_info[factor]
            
            # Low axial point
            experiment_low = {'experiment_id': f'ccd_axial_{exp_id:03d}', 'design_type': 'axial'}
            for j, f in enumerate(factors):
                f_info = factor_info[f]
                if i == j:  # Axial factor
                    if info['type'] == 'categorical':
                        experiment_low[f] = info['levels'][0]  # First level
                    else:
                        # -α level
                        range_val = info['high'] - info['low']
                        experiment_low[f] = max(info['low'], info['center'] - alpha * range_val / 2)
                else:  # Center level for other factors
                    if f_info['type'] == 'categorical':
                        experiment_low[f] = f_info['levels'][len(f_info['levels'])//2]
                    else:
                        experiment_low[f] = f_info['center']
            experiments.append(experiment_low)
            exp_id += 1
            
            # High axial point
            experiment_high = {'experiment_id': f'ccd_axial_{exp_id:03d}', 'design_type': 'axial'}
            for j, f in enumerate(factors):
                f_info = factor_info[f]
                if i == j:  # Axial factor
                    if info['type'] == 'categorical':
                        experiment_high[f] = info['levels'][-1]  # Last level
                    else:
                        # +α level
                        range_val = info['high'] - info['low']
                        experiment_high[f] = min(info['high'], info['center'] + alpha * range_val / 2)
                else:  # Center level for other factors
                    if f_info['type'] == 'categorical':
                        experiment_high[f] = f_info['levels'][len(f_info['levels'])//2]
                    else:
                        experiment_high[f] = f_info['center']
            experiments.append(experiment_high)
            exp_id += 1
        
        # Center points
        for i in range(center_points):
            experiment = {'experiment_id': f'ccd_center_{i+1:03d}', 'design_type': 'center_point'}
            for factor in factors:
                info = factor_info[factor]
                if info['type'] == 'categorical':
                    experiment[factor] = info['levels'][len(info['levels'])//2]
                else:
                    experiment[factor] = info['center']
            experiments.append(experiment)
        
        # Add default fixed parameters
        for experiment in experiments:
            experiment.update(self._get_default_config())
        
        # Randomize
        np.random.shuffle(experiments)
        
        df = pd.DataFrame(experiments)
        
        if self.verbose:
            print(f"Generated {len(experiments)} CCD experiments")
        
        return df
    
    def generate_lhs_design(self, n_experiments=50, factors=None):
        """Generate Latin Hypercube Sampling design for space exploration."""
        if factors is None:
            factors = list(self.parameter_space.keys())
        
        if self.verbose:
            print(f"Generating LHS design: {n_experiments} experiments, {len(factors)} factors")
        
        experiments = []
        
        # Generate LHS samples
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=len(factors), scramble=True, seed=42)
        samples = sampler.random(n=n_experiments)
        
        for i, sample in enumerate(samples):
            experiment = {'experiment_id': f'lhs_{i+1:03d}', 'design_type': 'lhs'}
            
            for j, factor in enumerate(factors):
                param_info = self.parameter_space[factor]
                sample_val = sample[j]
                
                if param_info['type'] == 'categorical':
                    # Sample from categorical levels
                    level_idx = int(sample_val * len(param_info['levels']))
                    level_idx = min(level_idx, len(param_info['levels']) - 1)
                    experiment[factor] = param_info['levels'][level_idx]
                
                elif param_info['type'] == 'discrete':
                    # Sample from discrete range
                    min_val, max_val = param_info['range']
                    val = min_val + sample_val * (max_val - min_val)
                    # Round to nearest valid level
                    valid_levels = sorted(param_info['levels'])
                    closest_idx = np.argmin([abs(val - level) for level in valid_levels])
                    experiment[factor] = valid_levels[closest_idx]
                
                else:  # continuous
                    min_val, max_val = param_info['range']
                    if param_info.get('log_scale', False):
                        # Log-uniform sampling
                        log_min, log_max = np.log10(min_val), np.log10(max_val)
                        log_val = log_min + sample_val * (log_max - log_min)
                        experiment[factor] = 10 ** log_val
                    else:
                        # Linear sampling
                        experiment[factor] = min_val + sample_val * (max_val - min_val)
            
            experiments.append(experiment)
        
        # Add default fixed parameters
        for experiment in experiments:
            experiment.update(self._get_default_config())
        
        df = pd.DataFrame(experiments)
        
        if self.verbose:
            print(f"Generated {len(experiments)} LHS experiments")
        
        return df
    
    def _get_default_config(self):
        """Get default configuration for fixed parameters."""
        if self.mode == 'cwt':
            return {
                'img_width': 100,
                'img_height': 256,
                'img_channels': 1,
                'epochs': 30,  # Reduced for DoE experiments
                'k_folds': 5,
                'early_stopping_patience': 8,
                'lr_reduction_patience': 5,
                'lr_reduction_factor': 0.5,
                'use_class_weights': True,
                'augment_fraction': 0.0,  # Start without augmentation
                'run_gradcam': False      # Skip for speed
            }
        else:  # pd mode
            return {
                'img_width': 100,
                'epochs': 30,  # Reduced for DoE experiments
                'k_folds': 5,
                'early_stopping_patience': 8,
                'lr_reduction_patience': 5,
                'lr_reduction_factor': 0.5,
                'use_class_weights': True,
                'augment_fraction': 0.0   # Start without augmentation
            }
    
    def save_experiments(self, df, design_name, phase=None):
        """Save experimental design to files."""
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        if phase:
            filename = f'{design_name}_phase{phase}_{timestamp}'
        else:
            filename = f'{design_name}_{timestamp}'
        
        # Save as CSV
        csv_path = self.output_dir / f'{filename}.csv'
        df.to_csv(csv_path, index=False)
        
        # Save as JSON configs for direct use with hyperparameter tuner
        json_dir = self.output_dir / f'{filename}_configs'
        json_dir.mkdir(exist_ok=True)
        
        for i, row in df.iterrows():
            config = row.to_dict()
            # Remove metadata columns
            config = {k: v for k, v in config.items() 
                     if k not in ['experiment_id', 'design_type']}
            
            # Convert string representations back to proper types
            for param, value in config.items():
                if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                    try:
                        config[param] = eval(value)  # Convert string lists back to lists
                    except:
                        pass  # Keep as string if conversion fails
            
            config_path = json_dir / f"config_{i+1:03d}.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        # Save design summary
        summary = {
            'design_type': design_name,
            'phase': phase,
            'timestamp': timestamp,
            'mode': self.mode,
            'total_experiments': len(df),
            'design_summary': df['design_type'].value_counts().to_dict() if 'design_type' in df.columns else {},
            'factors_tested': [col for col in df.columns if col not in ['experiment_id', 'design_type']],
            'estimated_runtime_hours': len(df) * 2,  # Assume 2 hours per experiment
        }
        
        summary_path = self.output_dir / f'{filename}_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.verbose:
            print(f"Experiments saved:")
            print(f"  CSV: {csv_path}")
            print(f"  Configs: {json_dir}")
            print(f"  Summary: {summary_path}")
        
        return csv_path, json_dir, summary_path


def main():
    parser = argparse.ArgumentParser(description='Generate DoE experiments for hyperparameter optimization')
    parser.add_argument('--mode', type=str, choices=['pd', 'cwt'], default='pd',
                       help='Model type: pd (PD signal) or cwt (CWT image)')
    parser.add_argument('--design', type=str, 
                       choices=['factorial', 'response_surface', 'lhs'], 
                       default='factorial',
                       help='Experimental design type')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                       help='DoE phase (1=screening, 2=optimization, 3=validation)')
    parser.add_argument('--n_experiments', type=int, default=50,
                       help='Number of experiments (for LHS design)')
    parser.add_argument('--factors', type=str, nargs='*',
                       help='Specific factors to include (default: auto-select)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for experiments')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = DoEGenerator(mode=args.mode, output_dir=args.output_dir, verbose=args.verbose)
    
    # Generate experiments based on design type
    if args.design == 'factorial':
        df = generator.generate_factorial_design(factors=args.factors)
        design_name = 'fractional_factorial'
    elif args.design == 'response_surface':
        df = generator.generate_response_surface_design(factors=args.factors)
        design_name = 'central_composite'
    elif args.design == 'lhs':
        df = generator.generate_lhs_design(n_experiments=args.n_experiments, factors=args.factors)
        design_name = 'latin_hypercube'
    
    # Save experiments
    csv_path, json_dir, summary_path = generator.save_experiments(df, design_name, args.phase)
    
    print(f"\n✅ DoE experiment generation completed!")
    print(f"Design: {design_name}")
    print(f"Experiments: {len(df)}")
    print(f"Estimated runtime: {len(df) * 2:.0f} hours")
    print(f"Files saved to: {generator.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())