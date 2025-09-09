#!/usr/bin/env python3
"""
Final Model Training Script

This script trains the final production model using the best hyperparameters
from hyperparameter optimization. It supports two modes:

1. Auto mode (default): Uses the configuration with highest k-fold mean accuracy
2. Manual mode: Uses a user-specified version number

The final model is trained on the full dataset with train/val/test split
(no k-fold cross-validation for faster training).

Author: AI Assistant
"""

# Ensure UTF-8 encoding for all I/O operations
import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import logging
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from config import (
    get_experiment_log_path, format_version, get_next_version_from_log,
    convert_numpy_types, normalize_path
)
from data_utils import normalize_image, split_dual_branch_image, estimate_memory_usage_gb

class FinalModelTrainer:
    """Trains final production model using best hyperparameters."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO if verbose else logging.WARNING)


    def _check_memory_warning(self, file_paths, img_width):
        """Check if dataset size exceeds memory threshold and warn user."""
        total_gb = estimate_memory_usage_gb(len(file_paths), img_width)
        
        if total_gb > 32:
            print(f"\n⚠️  MEMORY WARNING")
            print(f"📊 Dataset size: {total_gb:.1f} GB")
            print(f"📁 Total files: {len(file_paths)}")
            print(f"⚠️  This dataset is very large and may cause memory issues.")
            print(f"💡 Consider reducing dataset size or running on a machine with more RAM.")
            
            response = input("Continue anyway? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("❌ Operation cancelled by user.")
                return False
        elif total_gb > 16:
            print(f"📊 Dataset size: {total_gb:.1f} GB - Large dataset detected")
            
        return True
        
    def get_best_config(self):
        """Get configuration with highest k-fold mean accuracy."""
        log_file = get_experiment_log_path()
        
        if not Path(log_file).exists():
            raise FileNotFoundError(f"Experiment log not found: {log_file}")
        
        df = pd.read_csv(log_file, encoding='utf-8')
        
        if 'mean_val_accuracy' not in df.columns:
            raise ValueError("No mean_val_accuracy column found in experiment log")
        
        if len(df) == 0:
            raise ValueError("Experiment log is empty")
        
        # Find best configuration
        best_idx = df['mean_val_accuracy'].idxmax()
        best_row = df.iloc[best_idx]
        
        if self.verbose:
            print(f"Best configuration found:")
            print(f"  Version: {best_row['version']}")
            print(f"  Mean validation accuracy: {best_row['mean_val_accuracy']:.4f}")
            print(f"  Standard deviation: {best_row['std_val_accuracy']:.4f}")
        
        return best_row
    
    def get_config_by_version(self, version_num):
        """Get configuration for specific version number."""
        log_file = get_experiment_log_path()
        
        if not Path(log_file).exists():
            raise FileNotFoundError(f"Experiment log not found: {log_file}")
        
        df = pd.read_csv(log_file, encoding='utf-8')
        
        # Format version consistently
        version_str = format_version(version_num)
        
        # Find matching version
        matching_rows = df[df['version'] == version_str]
        
        if len(matching_rows) == 0:
            available_versions = df['version'].tolist()
            raise ValueError(f"Version {version_str} not found in experiment log. Available versions: {available_versions}")
        
        if len(matching_rows) > 1:
            print(f"Warning: Multiple entries found for version {version_str}, using first one")
        
        selected_row = matching_rows.iloc[0]
        
        if self.verbose:
            print(f"Selected configuration:")
            print(f"  Version: {selected_row['version']}")
            print(f"  Mean validation accuracy: {selected_row['mean_val_accuracy']:.4f}")
            print(f"  Standard deviation: {selected_row['std_val_accuracy']:.4f}")
        
        return selected_row
    
    def extract_config_from_log_row(self, log_row):
        """Extract configuration parameters from experiment log row."""
        config = {
            'data_dir': log_row['data_dir'],
            'img_width': int(log_row['img_width']),
            'learning_rate': float(log_row['learning_rate']),
            'batch_size': int(log_row['batch_size']),
            'epochs': int(log_row['epochs']),
            'k_folds': 1,  # No k-fold for final training
            'conv_filters': self._safe_parse_list(log_row['conv_filters']),
            'dense_units': self._safe_parse_list(log_row['dense_units']),
            'conv_dropout': float(log_row.get('dropout_rates', '0.2').split(',')[0].strip('[]')),
            'dense_dropout': self._safe_parse_list(log_row['dropout_rates'].split(',')[1:] if ',' in str(log_row['dropout_rates']) else [log_row['dropout_rates']]),
            'l2_regularization': float(log_row['l2_reg']),
            'use_batch_norm': bool(log_row['batch_norm']),
            'optimizer': str(log_row['optimizer']),
            'early_stopping_patience': int(log_row['early_stopping_patience']),
            'lr_reduction_patience': int(log_row['lr_reduction_patience']),
            'use_class_weights': bool(log_row['class_weights']),
            'augment_fraction': float(log_row.get('augment_fraction', 0.5)),
            'time_shift_range': int(log_row.get('time_shift_range', 5)),
            'stretch_probability': float(log_row.get('stretch_probability', 0.3)),
            'noise_probability': float(log_row.get('noise_probability', 0.5)),
            'amplitude_scale_probability': float(log_row.get('amplitude_scale_probability', 0.5)),
        }
        
        return config
    
    def _safe_parse_list(self, value):
        """Safely parse list from string representation."""
        if isinstance(value, list):
            return value
        
        try:
            # Handle string representations of lists
            import ast
            return ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            # Fallback for simple comma-separated values
            if ',' in str(value):
                return [float(x.strip()) for x in str(value).strip('[]').split(',')]
            else:
                return [float(value)]
    
    def find_existing_config_file(self, source_version):
        """Find the existing config file for the source version using enhanced traceability."""
        log_file = get_experiment_log_path()
        
        if not Path(log_file).exists():
            raise FileNotFoundError(f"Experiment log not found: {log_file}")
        
        df = pd.read_csv(log_file, encoding='utf-8')
        
        # Find matching version
        matching_rows = df[df['version'] == source_version]
        
        if len(matching_rows) == 0:
            raise FileNotFoundError(f"Version {source_version} not found in experiment log")
        
        # Get the most recent entry for this version
        latest_entry = matching_rows.iloc[-1]
        
        # Try enhanced lookup first (new system)
        if 'config_file' in df.columns and pd.notna(latest_entry.get('config_file')) and latest_entry.get('config_file'):
            config_file_path = latest_entry['config_file']
            if Path(config_file_path).exists():
                if self.verbose:
                    print(f"Found config file via enhanced traceability: {config_file_path}")
                return Path(config_file_path)
        
        # Fallback to pattern matching (legacy system)
        if self.verbose:
            print(f"Enhanced traceability unavailable for {source_version}, falling back to pattern matching")
        
        from glob import glob
        config_pattern = f"ml/logs/hyperopt_results/*/config_{int(source_version[1:]):03d}.json"
        config_files = glob(config_pattern)
        
        if not config_files:
            # Try searching by run ID if available
            if 'hyperopt_run_id' in df.columns and pd.notna(latest_entry.get('hyperopt_run_id')):
                run_id = latest_entry['hyperopt_run_id']
                config_number = latest_entry.get('config_number_in_run', -1)
                if config_number > 0:
                    run_config_file = f"ml/logs/hyperopt_results/{run_id}/config_{config_number:03d}.json"
                    if Path(run_config_file).exists():
                        if self.verbose:
                            print(f"Found config file via run ID lookup: {run_config_file}")
                        return Path(run_config_file)
            
            raise FileNotFoundError(f"No config file found for version {source_version}")
        
        # Use the most recent config file if multiple exist
        config_file = sorted(config_files)[-1]
        if self.verbose:
            print(f"Found config file via pattern matching: {config_file}")
        return Path(config_file)
    
    def holdout_test_set(self, config, output_version_dir):
        """Hold out test set before training and save metadata to model output folder.
        
        Args:
            config: Configuration dictionary containing data_dir, img_width
            output_version_dir: Output directory for this model version
            
        Returns:
            tuple: (test_data_file, exclusion_file) both in output_version_dir
        """
        from glob import glob
        sys.path.append('ml')
        
        print(f"\n📊 Creating test set holdout via file exclusion...")
        
        # Get all file paths and labels
        data_dir = config['data_dir']
        img_width = config['img_width']
        file_paths, labels = [], []
        
        for class_label in ['0', '1']:  # Assuming binary classification
            class_path = Path(data_dir) / class_label
            if class_path.exists():
                class_files = glob(str(class_path / "*.tiff"))
                file_paths.extend(class_files)
                labels.extend([int(class_label)] * len(class_files))
        
        if not file_paths:
            raise ValueError(f"No TIFF files found in {data_dir}")
        
        file_paths = np.array(file_paths)
        labels = np.array(labels)
        
        if self.verbose:
            print(f"Full dataset: {len(labels)} files")
            print(f"Class distribution: {np.bincount(labels)}")
        
        # Hold out test set (20% of full data)
        trainval_files, test_files, trainval_labels, test_labels = train_test_split(
            file_paths, labels,
            test_size=0.2, 
            stratify=labels, 
            random_state=42
        )
        
        print(f"Data split:")
        print(f"  Training+Validation: {len(trainval_labels)} files ({len(trainval_labels)/len(labels)*100:.1f}%)")
        print(f"  Test (held out): {len(test_labels)} files ({len(test_labels)/len(labels)*100:.1f}%)")
        print(f"  Test class distribution: {np.bincount(test_labels)}")
        
        # Check memory usage and warn if necessary
        if not self._check_memory_warning(test_files, img_width):
            return None, None  # User cancelled
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_version_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create exclusion list (files to skip during training)
        exclusion_list = set(Path(f).name for f in test_files)
        exclusion_file = output_dir / 'test_set_exclusions.txt'
        with open(exclusion_file, 'w') as f:
            for filename in sorted(exclusion_list):
                f.write(f"{filename}\n")
        
        if self.verbose:
            print(f"Test set exclusion list saved to: {exclusion_file}")
            print(f"Excluded {len(exclusion_list)} files from training")
        
        # Load and save test data to pickle for the tester
        print(f"Loading test set images...")
        test_data_file = output_dir / 'test_data.pkl'
        
        # Load only test images
        pd1_test, pd2_test = [], []
        y_test_filtered = []
        
        for class_label in [0, 1]:
            class_files = test_files[test_labels == class_label]
            for file_path in class_files:
                try:
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
                    if img is not None:
                        # Convert from (2, 100) to (100, 2) format for proper signal extraction  
                        # Original: (2_photodiodes, 100_time_samples) -> Target: (100_time_samples, 2_photodiodes)
                        if img.shape == (2, img_width):
                            img = img.T  # Transpose (2, 100) -> (100, 2)
                        elif img.shape == (img_width, 2):
                            pass  # Already correct format
                        else:
                            raise ValueError(f"Unexpected image shape {img.shape}. Expected (2, {img_width}) or ({img_width}, 2)")
                        
                        # Verify final shape
                        assert img.shape == (img_width, 2), f"Expected ({img_width}, 2), got {img.shape}"
                        
                        # Use centralized normalization and splitting functions
                        img = normalize_image(img)
                        # Use same format as training data: (img_width, 1)
                        pd1_signal, pd2_signal = split_dual_branch_image(img, img_width)
                        
                        pd1_test.append(pd1_signal)
                        pd2_test.append(pd2_signal)
                        y_test_filtered.append(class_label)
                except Exception as e:
                    self.logger.warning(f"Could not load test image {file_path}: {e}")
                    if self.verbose:
                        print(f"Warning: Could not load test image {file_path}: {e}")
        
        # Create comprehensive test set metadata
        test_data = {
            'X_test': (np.array(pd1_test), np.array(pd2_test)),
            'y_test': np.array(y_test_filtered),
            'data_dir': data_dir,  # Original directory
            'img_width': img_width,
            'test_files': [str(f) for f in test_files],  # Save file list for reference
            'train_test_split_params': {
                'test_size': 0.2,
                'stratify': True,
                'random_state': 42
            }
        }
        
        with open(test_data_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        if self.verbose:
            print(f"Test data saved to: {test_data_file}")
            print(f"Test set contains {len(y_test_filtered)} images")
        
        return test_data_file, exclusion_file

    def train_final_model(self, config, source_version):
        """Train final model with test set holdout and evaluation."""
        
        print(f"\n{'='*80}")
        print(f"FINAL MODEL TRAINING WITH TEST EVALUATION")
        print(f"{'='*80}")
        print(f"Source Configuration: {source_version}")
        print(f"Training Mode: Test holdout (20%) + Train/Val split (64%/16% of original)")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"Batch Size: {config['batch_size']}")
        print(f"Architecture: Conv{config['conv_filters']} + Dense{config['dense_units']}")
        print(f"Dropout: {config['conv_dropout']}/{config['dense_dropout']}")
        print(f"{'='*80}")
        
        # Find existing config file
        try:
            config_file = self.find_existing_config_file(source_version)
            if self.verbose:
                print(f"Using existing config file: {config_file}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return False
        
        # Step 1: Hold out test set
        # First, determine the output version directory that will be created by training
        from config import get_next_version_from_log, format_version
        current_version = get_next_version_from_log()
        version_str = format_version(current_version)
        output_version_dir = f'ml/outputs/{version_str}'
        
        try:
            test_data_file, exclusion_file = self.holdout_test_set(config, output_version_dir)
            if test_data_file is None:  # User cancelled due to memory warning
                return False
        except Exception as e:
            self.logger.error(f"Failed to create test set holdout: {e}", exc_info=True)
            print(f"❌ Failed to create test set holdout: {e}")
            return False
        
        # Step 2: Train model on training+validation data
        print(f"\n🔧 Training model on training+validation data...")
        
        # Prepare command for training script (use exclusion list)
        cmd = [
            sys.executable, 'ml/PD_signal_classifier_v3.py',
            '--config', str(config_file),
            '--k_folds', '1',  # Override k_folds to 1 for final training
            '--source', f'final_model_from_{source_version}',
            '--config_file', str(config_file),  # For traceability reference
            '--exclude_files', str(exclusion_file)  # Skip test set files
        ]
        
        # Add verbosity if requested
        if self.verbose:
            cmd.append('--verbose')
        else:
            cmd.append('--concise')
        
        try:
            # Set up environment for subprocess
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSFSENCODING'] = '0'
            
            # Run training with real-time output
            result = subprocess.run(
                cmd, 
                timeout=7200,  # 2 hour timeout
                encoding='utf-8',
                errors='replace',
                env=env,
                cwd=str(Path(__file__).parent.parent)  # Project root
            )
            
            if result.returncode != 0:
                print(f"\n❌ Final model training failed with exit code {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"\n⏱️ Training timed out after 2 hours")
            return False
        except Exception as e:
            print(f"\n💥 Training failed with error: {e}")
            return False
        
        print(f"\n✅ Model training completed successfully!")
        
        # Step 3: Find the trained model and run test evaluation
        return self.run_test_evaluation(source_version, test_data_file)
    
    def run_test_evaluation(self, source_version, test_data_file):
        """Run test evaluation on the trained model."""
        from config import get_next_version_from_log, format_version
        
        print(f"\n🧪 Running test evaluation...")
        
        # Get the version that was just created
        current_version = get_next_version_from_log() - 1  # Just created version
        version_str = format_version(current_version)
        
        # Find the model file
        model_dir = Path(f'ml/outputs/{version_str}')
        model_files = list(model_dir.glob('models/*.h5')) + list(model_dir.glob('models/*.keras'))
        
        if not model_files:
            print(f"❌ No trained model found in {model_dir}/models/")
            return False
        
        # Use the most recent model file
        model_file = sorted(model_files)[-1]
        
        if self.verbose:
            print(f"Found trained model: {model_file}")
        
        # Prepare command for model tester
        cmd = [
            sys.executable, 'ml/model_tester.py',
            '--model', str(model_file),
            '--test_data', str(test_data_file),
            '--output_dir', str(model_dir),
            '--model_version', version_str
        ]
        
        if self.verbose:
            cmd.append('--verbose')
        
        try:
            # Run test evaluation
            result = subprocess.run(
                cmd,
                timeout=600,  # 10 minute timeout
                encoding='utf-8',
                errors='replace',
                cwd=str(Path(__file__).parent.parent)  # Project root
            )
            
            if result.returncode == 0:
                print(f"\n✅ Test evaluation completed successfully!")
                
                # Load and display test results
                test_results_file = model_dir / 'test_results.json'
                if test_results_file.exists():
                    import json
                    with open(test_results_file, 'r') as f:
                        test_results = json.load(f)
                    
                    test_acc = test_results['results']['test_accuracy']
                    test_f1 = test_results['results']['test_f1_score']
                    print(f"\n📊 FINAL TEST RESULTS:")
                    print(f"   Test Accuracy: {test_acc:.4f}")
                    print(f"   Test F1-Score: {test_f1:.4f}")
                    print(f"   Results saved to: {model_dir}")
                    
                    # Update experiment log with test results
                    try:
                        sys.path.append('ml')
                        from PD_signal_classifier_v3 import update_experiment_log_with_test_results
                        
                        update_experiment_log_with_test_results(version_str, test_results['results'])
                        print(f"   Test results added to experiment log")
                        
                    except Exception as e:
                        print(f"   Warning: Failed to update experiment log with test results: {e}")
                
                return True
            else:
                print(f"\n❌ Test evaluation failed with exit code {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"\n⏱️ Test evaluation timed out")
            return False
        except Exception as e:
            print(f"\n💥 Test evaluation failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Train final production model using best hyperparameters')
    parser.add_argument('--version', type=int, 
                       help='Version number to use (if not specified, uses best config)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed training output')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = FinalModelTrainer(verbose=args.verbose)
    
    try:
        if args.version is None:
            print("🔍 Finding best configuration from hyperparameter optimization results...")
            best_row = trainer.get_best_config()
            source_version = best_row['version']
            config = trainer.extract_config_from_log_row(best_row)
        else:
            print(f"🔍 Loading configuration for version {args.version}...")
            selected_row = trainer.get_config_by_version(args.version)
            source_version = selected_row['version']
            config = trainer.extract_config_from_log_row(selected_row)
        
        # Train final model
        success = trainer.train_final_model(config, source_version)
        
        if success:
            print("\n🎉 Final model training completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Final model training failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()