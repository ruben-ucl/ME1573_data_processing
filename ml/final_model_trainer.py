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
import pickle
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import cv2
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    get_pd_experiment_log_path, get_cwt_experiment_log_path, format_version, get_next_version_from_log,
    convert_numpy_types, normalize_path, PD_OUTPUTS_DIR, CWT_OUTPUTS_DIR,
    PD_HYPEROPT_RESULTS_DIR, CWT_HYPEROPT_RESULTS_DIR, ensure_directories, ensure_cwt_directories,
    load_dataset_variant_info
)
from gradcam_utils import generate_comprehensive_gradcam_analysis
from data_utils import (normalize_image, split_dual_branch_image, estimate_memory_usage_gb,
                        load_cwt_test_images, load_pd_test_images)

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU detected: {len(gpus)} device(s) — using GPU for evaluation")
else:
    print("No GPU detected — evaluation will run on CPU")

# ============================================================================
# COLOR SCHEME CONFIGURATION FOR TRACK PREDICTION VISUALIZATIONS
# ============================================================================
# Centralized color definitions for easy customization
COLOR_NO_POROSITY = '#3498db'      # Blue for no porosity (class 0)
COLOR_POROSITY = '#e74c3c'         # Red for porosity (class 1)
COLOR_SKIPPED_WINDOW = '#95a5a6'   # Grey for skipped first window
# ============================================================================

class FinalModelTrainer:
    """Trains final production model using best hyperparameters."""
    
    def __init__(self, classifier_type='pd_signal', verbose=False, k_folds_override=None, dataset_variant=None):
        self.classifier_type = classifier_type
        self.verbose = verbose
        self.k_folds_override = k_folds_override
        self.dataset_variant = dataset_variant
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO if verbose else logging.WARNING)


    def _check_memory_warning(self, file_paths, img_width, classifier_type=None):
        """Check if dataset size exceeds memory threshold and warn user."""
        if classifier_type == 'cwt_image':
            # CWT images are much larger (256x100 vs 100x2), estimate accordingly
            # Approximate factor for CWT vs PD memory usage
            total_gb = estimate_memory_usage_gb(len(file_paths), img_width) * 256 / 100
        else:
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
        if self.classifier_type == 'cwt_image':
            log_file = get_cwt_experiment_log_path()
        else:
            log_file = get_pd_experiment_log_path()
        
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
        if self.classifier_type == 'cwt_image':
            log_file = get_cwt_experiment_log_path()
        else:
            log_file = get_pd_experiment_log_path()
        
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
        # Use proper output directory based on classifier type
        if self.classifier_type == 'cwt_image':
            output_root = str(CWT_OUTPUTS_DIR)
        else:
            output_root = str(PD_OUTPUTS_DIR)

        # Build base config (model hyperparameters only)
        config = {
            'img_width': int(log_row['img_width']),
            'output_root': output_root,
            'learning_rate': float(log_row['learning_rate']),
            'batch_size': int(log_row['batch_size']),
            'epochs': int(log_row['epochs']),
            'k_folds': int(log_row.get('k_folds', 5)),  # Keep original k-fold CV, save best model
            'conv_filters': self._safe_parse_list(log_row['conv_filters']),
            'dense_units': self._safe_parse_list(log_row['dense_units']),
            # Handle dropout rates based on classifier type
            'conv_dropout': float(log_row.get('conv_dropout', 0.2)) if self.classifier_type == 'cwt_image' else float(log_row.get('dropout_rates', '0.2').split(',')[0].strip('[]')),
            'dense_dropout': self._safe_parse_list(log_row['dense_dropout']) if self.classifier_type == 'cwt_image' else self._safe_parse_list(log_row['dropout_rates'].split(',')[1:] if ',' in str(log_row['dropout_rates']) else [log_row['dropout_rates']]),
            'l2_regularization': float(log_row['l2_regularization']) if self.classifier_type == 'cwt_image' else float(log_row['l2_reg']),
            'use_batch_norm': bool(log_row['use_batch_norm']) if self.classifier_type == 'cwt_image' else bool(log_row['batch_norm']),
            'optimizer': str(log_row['optimizer']),
            'early_stopping_patience': int(log_row['early_stopping_patience']),
            'lr_reduction_patience': int(log_row['lr_reduction_patience']),
            'use_class_weights': bool(log_row['use_class_weights']) if self.classifier_type == 'cwt_image' else bool(log_row['class_weights']),
            'augment_fraction': float(log_row.get('augment_fraction', 0.5)),
            'time_shift_range': self._safe_convert_to_numeric(log_row.get('time_shift_range', 5), int, 5),
            'stretch_probability': float(log_row.get('stretch_probability', 0.3)),
            'noise_probability': float(log_row.get('noise_probability', 0.5)),
            'amplitude_scale_probability': float(log_row.get('amplitude_scale_probability', 0.5)),
        }

        # Add data directory ONLY if not using dataset variant
        # Dataset variant will provide data_dir from its own config
        if not self.dataset_variant:
            if self.classifier_type == 'cwt_image':
                config['cwt_data_dir'] = log_row['cwt_data_dir']
            else:
                config['data_dir'] = log_row['data_dir']

        # Add CWT-specific parameters
        if self.classifier_type == 'cwt_image':
            config.update({
                'img_height': int(log_row.get('img_height', 256)),
                'img_channels': int(log_row.get('img_channels', 1)),
            })

        return config
    
    def _safe_parse_list(self, value):
        """Safely parse list from string representation."""
        if isinstance(value, list):
            # If already a list, clean up any string elements that should be numeric
            cleaned = []
            for item in value:
                if isinstance(item, str):
                    # Remove extra brackets and whitespace
                    item = item.strip().strip('[]\'\"')
                    if item:
                        try:
                            cleaned.append(float(item))
                        except ValueError:
                            # Keep as string if not numeric
                            cleaned.append(item)
                else:
                    cleaned.append(item)
            return cleaned
        
        try:
            # Handle string representations of lists
            import ast
            return ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            # Fallback for simple comma-separated values
            if ',' in str(value):
                return [float(x.strip().strip('[]\'\"')) for x in str(value).strip('[]').split(',') if x.strip()]
            else:
                return [float(value)]
    
    def _safe_convert_to_numeric(self, value, target_type, default):
        """Safely convert value to numeric type with fallback."""
        try:
            # Handle boolean strings
            if isinstance(value, str):
                if value.lower() in ('true', 'false'):
                    # Convert boolean to numeric (True->1, False->0) then to target type
                    bool_val = value.lower() == 'true'
                    return target_type(1 if bool_val else 0)
                elif value.lower() in ('none', 'nan', ''):
                    return default
            
            # Try direct conversion
            return target_type(value)
        except (ValueError, TypeError):
            # Return default if conversion fails
            return default
    
    def find_existing_config_file(self, source_version):
        """Find the existing config file for the source version using enhanced traceability."""
        if self.classifier_type == 'cwt_image':
            log_file = get_cwt_experiment_log_path()
        else:
            log_file = get_pd_experiment_log_path()
        
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
                return self.load_and_correct_config(config_file_path)
        
        # Fallback to pattern matching (legacy system)
        if self.verbose:
            print(f"Enhanced traceability unavailable for {source_version}, falling back to pattern matching")
        
        from glob import glob
        try:
            # Extract version number safely
            version_str = source_version[1:] if source_version.startswith('v') else source_version
            version_num = int(float(version_str))  # Handle potential float strings
            
            # Use proper hyperopt results directory based on classifier type
            if self.classifier_type == 'cwt_image':
                hyperopt_dir = CWT_HYPEROPT_RESULTS_DIR
            else:
                hyperopt_dir = PD_HYPEROPT_RESULTS_DIR
            
            config_pattern = str(hyperopt_dir / f"*/config_{version_num:03d}.json")
        except (ValueError, IndexError) as e:
            print(f"Could not parse version number from {source_version}: {e}")
            return None
        config_files = glob(config_pattern)
        
        if not config_files:
            # Try searching by run ID if available
            if 'hyperopt_run_id' in df.columns and pd.notna(latest_entry.get('hyperopt_run_id')):
                run_id = latest_entry['hyperopt_run_id']
                config_number = latest_entry.get('config_number_in_run', -1)
                if config_number > 0:
                    try:
                        config_num = int(float(config_number))  # Safe conversion
                        run_config_file = str(hyperopt_dir / run_id / f"config_{config_num:03d}.json")
                        if Path(run_config_file).exists():
                            if self.verbose:
                                print(f"Found config file via run ID lookup: {run_config_file}")
                            return self.load_and_correct_config(run_config_file)
                    except (ValueError, TypeError):
                        pass  # Skip if conversion fails
            
            # No existing config file found, create one from log data
            if self.verbose:
                print(f"No config file found for {source_version}, creating from experiment log")
            return self._create_config_from_log_data(latest_entry, source_version)
        
        # Use the most recent config file if multiple exist
        config_file = sorted(config_files)[-1]
        if self.verbose:
            print(f"Found config file via pattern matching: {config_file}")
        return self.load_and_correct_config(config_file)
    
    def _create_config_from_log_data(self, log_row, source_version):
        """Create a temporary config file from experiment log data."""
        import json
        import tempfile
        
        # Extract config from log row
        config = self.extract_config_from_log_row(log_row)
        
        # Create temporary config file
        temp_dir = Path("ml/temp")
        temp_dir.mkdir(exist_ok=True)
        config_file = temp_dir / f"temp_config_{source_version}.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        if self.verbose:
            print(f"Created temporary config file: {config_file}")
        
        return config_file
    
    def load_and_correct_config(self, config_file_path):
        """Load config file and ensure it has correct paths for this classifier type."""
        import json

        # Load existing config
        with open(config_file_path, 'r') as f:
            config = json.load(f)

        # Ensure correct output_root based on classifier type
        if self.classifier_type == 'cwt_image':
            correct_output_root = str(CWT_OUTPUTS_DIR)
        else:
            correct_output_root = str(PD_OUTPUTS_DIR)

        # Update or add output_root
        config['output_root'] = correct_output_root

        # Remove data directory paths if using dataset variant
        # Dataset variant will provide the correct data_dir
        if self.dataset_variant:
            config.pop('cwt_data_dir', None)
            config.pop('cwt_data_channels', None)
            config.pop('data_dir', None)
            if self.verbose:
                print("Removed data_dir/cwt_data_channels from config - will use dataset variant's data_dir")

        # Ensure k_folds is at least 2 for sklearn compatibility, but prefer original value
        if config.get('k_folds', 5) < 2:
            config['k_folds'] = 5

        # Create corrected config file
        corrected_path = Path(config_file_path).parent / f"corrected_{Path(config_file_path).name}"
        with open(corrected_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        if self.verbose:
            print(f"Corrected config file created: {corrected_path}")
            print(f"Set output_root to: {correct_output_root}")

        return corrected_path
    
    def holdout_test_set(self, config, output_version_dir):
        """
        Create holdout test set by sampling files and creating exclusion list.
        For dataset variants, returns None to signal that classifier should handle exclusion.
        For random sampling, creates test set and exclusion list.

        Args:
            config: Configuration dictionary containing data_dir/cwt_data_dir, img_width, img_height
            output_version_dir: Output directory for this model version

        Returns:
            tuple: (test_data_file, exclusion_file) or (None, None) if using dataset variant
        """
        from glob import glob
        sys.path.append('ml')

        # If using dataset variant, don't create test set here - classifier handles it
        if self.dataset_variant:
            print(f"\n📊 Using dataset variant '{self.dataset_variant}' for test holdout")
            print("Test exclusion will be handled by classifier automatically")
            return None, None
        
        print(f"\n📊 Creating test set holdout via random file sampling...")
        
        # Get classifier-specific parameters
        if self.classifier_type == 'cwt_image':
            # Handle both single-channel (string) and multi-channel (dict) cwt_data_dir
            cwt_data = config.get('cwt_data_dir')
            if isinstance(cwt_data, dict):
                # Multi-channel: use first channel for file enumeration
                data_dir = list(cwt_data.values())[0]
                is_multi_channel = True
                channel_paths = list(cwt_data.values())
            else:
                data_dir = cwt_data
                is_multi_channel = False
                channel_paths = None

            img_width = config['img_width']
            img_height = config['img_height']
            img_channels = config['img_channels']
            file_pattern = "*.png"
        else:  # pd_signal
            data_dir = config['data_dir']
            is_multi_channel = False
            channel_paths = None
            img_width = config['img_width']
            img_height = None  # PD doesn't use height
            img_channels = None
            file_pattern = "*.tiff"

        # Get all file paths and labels
        file_paths, labels = [], []

        for class_label in ['0', '1']:  # Assuming binary classification
            class_path = Path(data_dir) / class_label
            if class_path.exists():
                class_files = glob(str(class_path / file_pattern))
                file_paths.extend(class_files)
                labels.extend([int(class_label)] * len(class_files))
        
        if not file_paths:
            raise ValueError(f"No {file_pattern} files found in {data_dir}")
        
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
        if not self._check_memory_warning(test_files, img_width, classifier_type=self.classifier_type):
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

        # Load test images based on classifier type
        if self.classifier_type == 'cwt_image':
            X_test, y_test_filtered, test_files_filtered = load_cwt_test_images(
                test_files, test_labels, img_width, img_height, img_channels,
                channel_paths=channel_paths if is_multi_channel else None,
                verbose=self.verbose
            )
        else:  # pd_signal
            X_test, y_test_filtered, test_files_filtered = load_pd_test_images(
                test_files, test_labels, img_width, verbose=self.verbose)
        
        # Create comprehensive test set metadata
        test_data = {
            'X_test': X_test,
            'y_test': np.array(y_test_filtered),
            'data_dir': data_dir,  # Original directory
            'classifier_type': self.classifier_type,
            'img_width': img_width,
            'img_height': img_height if self.classifier_type == 'cwt_image' else None,
            'img_channels': img_channels if self.classifier_type == 'cwt_image' else None,
            'test_files': test_files_filtered,  # Save file list for reference
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

    def _use_preexisting_holdout(self, config, output_version_dir):
        """
        Use preexisting test holdout file with trackids instead of random sampling.
        
        Args:
            config: Configuration dictionary
            output_version_dir: Output directory for this model version
            
        Returns:
            tuple: (test_data_file, exclusion_file) both in output_version_dir
        """
        from glob import glob
        
        print(f"\n📊 Using preexisting test holdout from: {self.test_holdout_file}")
        
        # Read trackids from holdout file
        holdout_trackids = set()
        try:
            with open(self.test_holdout_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        holdout_trackids.add(line)
        except FileNotFoundError:
            raise FileNotFoundError(f"Test holdout file not found: {self.test_holdout_file}")
        
        if not holdout_trackids:
            raise ValueError(f"No trackids found in holdout file: {self.test_holdout_file}")
        
        print(f"Loaded {len(holdout_trackids)} trackids for test holdout")
        if self.verbose:
            print(f"Holdout trackids: {sorted(holdout_trackids)}")
        
        # Get classifier-specific parameters
        if self.classifier_type == 'cwt_image':
            # Handle both single-channel (string) and multi-channel (dict) cwt_data_dir
            cwt_data = config.get('cwt_data_dir')
            if isinstance(cwt_data, dict):
                # Multi-channel: use first channel for file enumeration
                data_dir = list(cwt_data.values())[0]
                is_multi_channel = True
                channel_paths = list(cwt_data.values())
            else:
                data_dir = cwt_data
                is_multi_channel = False
                channel_paths = None

            img_width = config['img_width']
            img_height = config['img_height']
            img_channels = config['img_channels']
            file_pattern = "*.png"
        else:  # pd_signal
            data_dir = config['data_dir']
            is_multi_channel = False
            channel_paths = None
            img_width = config['img_width']
            img_height = None
            img_channels = None
            file_pattern = "*.tiff"

        # Find all files and separate by trackid
        all_files = []
        test_files = []
        train_files = []
        labels = []
        test_labels = []
        train_labels = []
        
        for class_label in ['0', '1']:
            class_path = Path(data_dir) / class_label
            if class_path.exists():
                class_files = glob(str(class_path / file_pattern))
                
                for file_path in class_files:
                    # Extract trackid using helper function
                    from data_utils import extract_trackid_from_filename
                    trackid = extract_trackid_from_filename(file_path)

                    if not trackid:
                        continue

                    all_files.append(file_path)
                    labels.append(int(class_label))

                    if trackid in holdout_trackids:
                        test_files.append(file_path)
                        test_labels.append(int(class_label))
                    else:
                        train_files.append(file_path)
                        train_labels.append(int(class_label))
        
        if not all_files:
            raise ValueError(f"No {file_pattern} files found in {data_dir}")
        
        if not test_files:
            raise ValueError(f"No files found matching holdout trackids in {data_dir}")
        
        # Convert to numpy arrays
        test_files = np.array(test_files)
        test_labels = np.array(test_labels)
        
        print(f"Data split based on trackid holdout:")
        print(f"  Training+Validation: {len(train_labels)} files ({len(train_labels)/len(labels)*100:.1f}%)")
        print(f"  Test (held out): {len(test_labels)} files ({len(test_labels)/len(labels)*100:.1f}%)")
        print(f"  Test class distribution: {np.bincount(test_labels)}")
        
        # Check memory usage and warn if necessary
        if not self._check_memory_warning(test_files, img_width, classifier_type=self.classifier_type):
            return None, None  # User cancelled
        
        # Create output directory
        output_dir = Path(output_version_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create exclusion list (files to skip during training)
        exclusion_list = set(Path(f).name for f in test_files)
        exclusion_file = output_dir / 'test_set_exclusions.txt'
        with open(exclusion_file, 'w') as f:
            f.write(f"# Test set exclusions generated from trackid holdout file: {self.test_holdout_file}\n")
            f.write(f"# Holdout method: trackid-based (representative sampling by melting regime)\n")
            f.write(f"# Holdout trackids: {', '.join(sorted(holdout_trackids))}\n")
            f.write("\n")
            for filename in sorted(exclusion_list):
                f.write(f"{filename}\n")
        
        if self.verbose:
            print(f"Exclusion list saved: {exclusion_file}")
            print(f"Files to exclude from training: {len(exclusion_list)}")
        
        # Load test images based on classifier type
        if self.classifier_type == 'cwt_image':
            X_test, y_test_filtered, test_files_filtered = load_cwt_test_images(
                test_files, test_labels, img_width, img_height, img_channels,
                channel_paths=channel_paths if is_multi_channel else None,
                verbose=self.verbose,
            )
        else:  # pd_signal
            X_test, y_test_filtered, test_files_filtered = load_pd_test_images(
                test_files, test_labels, img_width, verbose=self.verbose)

        # Create test data file
        test_data_file = output_dir / 'test_set_data.pkl'
        test_data = {
            'X_test': X_test,
            'y_test': np.array(y_test_filtered),
            'data_dir': data_dir,
            'classifier_type': self.classifier_type,
            'img_width': img_width,
            'img_height': img_height if self.classifier_type == 'cwt_image' else None,
            'img_channels': img_channels if self.classifier_type == 'cwt_image' else None,
            'test_files': test_files_filtered,
            'holdout_method': 'trackid_based',
            'holdout_trackids': list(holdout_trackids),
            'holdout_file_source': str(self.test_holdout_file)
        }
        
        with open(test_data_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        if self.verbose:
            print(f"Test data saved to: {test_data_file}")
            print(f"Test set contains {len(y_test_filtered)} images from {len(holdout_trackids)} trackids")
        
        return test_data_file, exclusion_file

    def evaluate_with_threshold_optimization(self, model, test_data, output_dir, version):
        """
        Comprehensive test evaluation with threshold optimization and Grad-CAM analysis.
        
        Args:
            model: Trained model to evaluate
            test_data: Dictionary containing test data and metadata
            output_dir: Directory to save evaluation results
            version: Version string for naming
            
        Returns:
            dict: Comprehensive evaluation results
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        from sklearn.metrics import confusion_matrix, classification_report
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print(f"\n🎯 COMPREHENSIVE TEST EVALUATION")
        print(f"{'='*60}")

        # Create test_evaluation subdirectory for all test outputs
        test_eval_dir = Path(output_dir) / 'test_evaluation'
        test_eval_dir.mkdir(exist_ok=True)

        X_test = test_data['X_test']
        y_test = test_data['y_test']
        classifier_type = test_data['classifier_type']
        
        # Get model predictions (probabilities)
        if classifier_type == 'cwt_image':
            y_proba = model.predict(X_test, verbose=0)
        else:  # pd_signal
            pd1_test, pd2_test = X_test
            y_proba = model.predict([pd1_test, pd2_test], verbose=0)
        
        y_proba_flat = y_proba.flatten()
        
        # Threshold optimization (0.2 to 0.8 in steps of 0.05)
        thresholds = np.arange(0.2, 0.85, 0.05)
        threshold_results = []
        
        print(f"🔍 Optimizing classification threshold...")
        for threshold in thresholds:
            y_pred = (y_proba_flat >= threshold).astype(int)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            threshold_results.append({
                'threshold': threshold,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
            })

        # Find best threshold by F1 score
        best_result = max(threshold_results, key=lambda x: x['f1_score'])
        best_threshold = best_result['threshold']
        
        print(f"✅ Best threshold: {best_threshold:.3f}")
        print(f"   Accuracy: {best_result['accuracy']:.4f}")
        print(f"   Precision: {best_result['precision']:.4f}")
        print(f"   Recall: {best_result['recall']:.4f}")
        print(f"   F1-Score: {best_result['f1_score']:.4f}")
        
        # Generate final predictions using best threshold
        y_pred_best = (y_proba_flat >= best_threshold).astype(int)
        
        # Calculate additional metrics
        try:
            auc_score = roc_auc_score(y_test, y_proba_flat)
        except ValueError:
            auc_score = None  # In case of single class
        
        # Save threshold optimization results
        threshold_df = pd.DataFrame(threshold_results)
        threshold_csv = test_eval_dir / f'threshold_optimization_{version}.csv'
        threshold_df.to_csv(threshold_csv, index=False, encoding='utf-8')

        # Save test predictions for standalone visualization script
        test_files = test_data.get('test_files', None)
        predictions_data = {
            'y_pred': y_pred_best,
            'y_proba': y_proba_flat,
            'y_true': y_test,
            'best_threshold': best_threshold,
            'test_files': test_files,
            'classifier_type': classifier_type
        }
        predictions_file = test_eval_dir / f'test_predictions_{version}.pkl'
        with open(predictions_file, 'wb') as f:
            pickle.dump(predictions_data, f)
        print(f"💾 Saved predictions to: {predictions_file}")

        # Plot threshold optimization — full A4 width, project conventions
        fig, axes = plt.subplots(2, 2, figsize=(6.30, 4.80))
        axes = axes.flatten()

        for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1_score']):
            axes[i].plot(threshold_df['threshold'], threshold_df[metric], 'b-', linewidth=1.5)
            axes[i].axvline(best_threshold, color='red', linestyle='--', linewidth=1.0,
                            label=f'Best: {best_threshold:.2f}')
            axes[i].set_xlabel('Threshold', fontsize=9)
            axes[i].set_ylabel(metric.replace('_', ' ').title(), fontsize=9)
            axes[i].tick_params(axis='both', labelsize=8)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(fontsize=8)
            axes[i].set_ylim(0, 1)

        plt.tight_layout()
        plot_stem = test_eval_dir / f'threshold_optimization_{version}'
        plt.savefig(str(plot_stem) + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(str(plot_stem) + '.pdf', bbox_inches='tight')
        plt.close()

        # Generate confusion matrix using consolidated function
        from visualize_track_predictions import generate_confusion_matrix

        # Determine class labels based on classifier type
        if classifier_type == 'pd_signal':
            class_labels = ['Conduct', 'Keyhole']
        elif classifier_type == 'cwt_image':
            class_labels = ['No Porosity', 'Porosity']
        else:
            class_labels = None

        generate_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred_best,
            output_dir=test_eval_dir.parent,  # Parent since function adds subdirectory
            version=version,
            threshold=best_threshold,
            test_files=test_data.get('test_files', None),
            class_labels=class_labels,
            subdir='test_evaluation'  # Explicit subdirectory for test evaluation
        )

        # Generate classification report
        class_report = classification_report(y_test, y_pred_best, output_dict=True)
        class_report_str = classification_report(y_test, y_pred_best)

        # Save classification report
        with open(test_eval_dir / f'classification_report_{version}.txt', 'w') as f:
            f.write(f"Classification Report - Threshold: {best_threshold:.3f}\n")
            f.write("="*50 + "\n")
            f.write(class_report_str)
            f.write(f"\nAUC Score: {auc_score:.4f}" if auc_score else "\nAUC Score: N/A (single class)")
        
        # Generate Grad-CAM analysis for CWT classifiers
        gradcam_results = None
        if classifier_type == 'cwt_image':
            test_files = test_data.get('test_files', None)

            # Extract channel labels if multi-channel
            channel_labels = None
            num_channels = X_test.shape[-1] if len(X_test.shape) == 4 else 1

            if num_channels > 1:
                # Try to get from test_data
                if 'channel_labels' in test_data:
                    channel_labels = test_data['channel_labels']
                else:
                    # Try to load from dataset_config.json (if using dataset variant)
                    try:
                        if hasattr(self, 'dataset_variant') and self.dataset_variant:
                            from config import load_dataset_variant_info
                            dataset_info = load_dataset_variant_info(self.dataset_variant)
                            dataset_dir = dataset_info['dataset_dir']
                            dataset_config_path = Path(dataset_dir) / 'dataset_config.json'

                            if dataset_config_path.exists():
                                with open(dataset_config_path, 'r', encoding='utf-8') as f:
                                    dataset_config = json.load(f)
                                data_dir = dataset_config.get('data_dir')
                                if isinstance(data_dir, dict):
                                    channel_labels = list(data_dir.keys())
                    except Exception as e:
                        print(f"   Note: Could not load channel labels from dataset_config: {e}")

                    # Fallback: try to load from config
                    if channel_labels is None:
                        try:
                            from config import resolve_cwt_data_channels
                            _, channel_labels, _ = resolve_cwt_data_channels(self.config)
                        except Exception as e:
                            print(f"   Note: Could not resolve channel labels from config: {e}")

                    # Final fallback: auto-generate labels
                    if channel_labels is None:
                        channel_labels = [f'Channel_{i+1}' for i in range(num_channels)]

            gradcam_results = generate_comprehensive_gradcam_analysis(
                model, X_test, y_test, y_pred_best, y_proba_flat,
                best_threshold, test_eval_dir, version, test_files,
                channel_labels=channel_labels
            )

        # Generate track-level prediction visualizations if we have filenames
        track_vis_results = None
        if 'test_files' in test_data:
            track_vis_results = self._generate_track_predictions_viz(
                test_data['test_files'], y_test, y_pred_best, test_eval_dir, version,
                y_proba=y_proba_flat,
            )

        # Generate P-V map showing test set track locations
        pv_map_results = None
        if 'test_files' in test_data:
            pv_map_results = self._generate_pv_map_for_test_set(
                test_data['test_files'], test_eval_dir, version
            )

        cm = confusion_matrix(y_test, y_pred_best)

        # Compile comprehensive results
        evaluation_results = {
            'version': version,
            'classifier_type': classifier_type,
            'test_samples': len(y_test),
            'class_distribution': dict(zip(*np.unique(y_test, return_counts=True))),
            'best_threshold': best_threshold,
            'best_metrics': best_result,
            'auc_score': auc_score,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'threshold_optimization': threshold_results,
            'gradcam_results': gradcam_results,
            'track_predictions': track_vis_results,
            'pv_map': pv_map_results,
            'output_files': {
                'threshold_csv': str(threshold_csv),
                'threshold_plot': str(test_eval_dir / f'threshold_optimization_{version}.png'),
                'confusion_matrix': str(test_eval_dir / f'confusion_matrix_{version}.png'),
                'classification_report': str(test_eval_dir / f'classification_report_{version}.txt')
            }
        }

        # Save comprehensive results as JSON
        # Convert numpy types to Python native types for JSON serialization
        evaluation_results_converted = convert_numpy_types(evaluation_results)
        results_json = test_eval_dir / f'comprehensive_evaluation_{version}.json'
        with open(results_json, 'w') as f:
            json.dump(evaluation_results_converted, f, indent=2, default=str)
        
        print(f"📊 Evaluation complete!")
        print(f"   Results saved to: {results_json}")
        if gradcam_results:
            print(f"   Grad-CAM analysis: {gradcam_results['total_images']} images analyzed")
        if track_vis_results:
            print(f"   Track predictions: {track_vis_results['figures_generated']} tracks visualized")
        if pv_map_results:
            print(f"   P-V map: {pv_map_results['unique_tracks']} tracks plotted")
        
        return evaluation_results

    def _generate_track_predictions_viz(self, test_files, y_true, y_pred, output_dir, version, vote_windows=False, y_proba=None):
        """Delegate to the shared track predictions visualizer."""
        from visualize_track_predictions import generate_track_predictions_viz
        return generate_track_predictions_viz(
            test_files=test_files,
            y_true=y_true,
            y_pred=y_pred,
            output_dir=output_dir,
            version=version,
            use_time_labels=True,
            vote_windows=vote_windows,
            y_proba=y_proba,
        )

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
        current_version = get_next_version_from_log(classifier_type=self.classifier_type)
        version_str = format_version(current_version)
        
        # Use proper output directory based on classifier type
        if self.classifier_type == 'cwt_image':
            ensure_cwt_directories()
            output_base_dir = CWT_OUTPUTS_DIR
        else:
            ensure_directories()
            output_base_dir = PD_OUTPUTS_DIR
            
        output_version_dir = str(output_base_dir / version_str)

        # Copy dataset config for traceability if using dataset variant
        if self.dataset_variant:
            try:
                import shutil
                dataset_info = load_dataset_variant_info(self.dataset_variant)
                dataset_config_src = dataset_info['dataset_dir'] / 'dataset_config.json'
                dataset_config_dst = Path(output_version_dir) / 'dataset_config.json'
                Path(output_version_dir).mkdir(parents=True, exist_ok=True)
                shutil.copy2(dataset_config_src, dataset_config_dst)
                if self.verbose:
                    print(f"Copied dataset config to: {dataset_config_dst}")
            except Exception as e:
                print(f"Warning: Could not copy dataset config: {e}")

        try:
            # Handle test holdout for random sampling mode
            test_data_file, exclusion_file = self.holdout_test_set(config, output_version_dir)
            if self.dataset_variant:
                # Dataset variant mode - no exclusion file needed (classifier handles it)
                test_data_file = None
                exclusion_file = None
        except Exception as e:
            self.logger.error(f"Failed to create test set holdout: {e}", exc_info=True)
            print(f"❌ Failed to create test set holdout: {e}")
            return False
        
        # Step 2: Train model on training+validation data
        print(f"\n🔧 Training model on training+validation data...")
        
        # Prepare command for training script (use exclusion list)
        script_name = 'CWT_image_classifier_v3.py' if self.classifier_type == 'cwt_image' else 'PD_signal_classifier_v3.py'
        cmd = [
            sys.executable, f'ml/{script_name}',
            '--config', str(config_file),
            '--source', f'final_model_from_{source_version}',
            '--config_file', str(config_file),  # For traceability reference
        ]
        
        # Handle k_folds override logic
        if self.k_folds_override is not None:
            # User explicitly specified k_folds - use it for any classifier type
            cmd.extend(['--k_folds', str(self.k_folds_override)])
        elif self.classifier_type != 'cwt_image':
            # For PD classifiers, default to k_folds=1 for final training
            cmd.extend(['--k_folds', '1'])
        # For CWT classifiers without override, use original k_folds from config (keep 5-fold CV)

        # Add test exclusion logic
        if self.dataset_variant:
            # Pass dataset variant - classifier will auto-load test exclusion
            cmd.extend(['--dataset_variant', str(self.dataset_variant)])
        elif exclusion_file:
            # Random sampling mode - pass explicit exclusion file
            cmd.extend(['--exclude_files', str(exclusion_file)])
        
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

        # Step 3: Run test evaluation if test data available
        if test_data_file is not None:
            # Random sampling mode - test data pre-loaded into pickle
            # Pass None to use just-trained model, not base config version
            return self.run_test_evaluation(None, test_data_file)
        elif self.dataset_variant:
            # Dataset variant mode — delegate to model_tester for full evaluation
            cmd = [sys.executable, 'ml/model_tester.py',
                   '--classifier_type', self.classifier_type,
                   '--dataset_variant', self.dataset_variant,
                   '--full',
                   '--version', str(current_version)]
            result = subprocess.run(cmd, encoding='utf-8', errors='replace',
                                    cwd=str(Path(__file__).parent.parent))
            return result.returncode == 0
        else:
            print(f"\n✅ Final model training completed successfully!")
            print(f"Note: No test evaluation configured")
            return True
    

    def _fix_legacy_test_data_order(self, test_data):
        """
        Fix file order mismatch in legacy pickled test data.
        
        Old test data has files in CSV order but images/labels in class-grouped order
        (all class 0 samples first, then all class 1 samples).
        
        This method re-loads and re-orders the data correctly.
        """
        import cv2
        import numpy as np
        from pathlib import Path
        
        if self.verbose:
            print("Detecting file order mismatch in legacy test data...")
        
        # Check if this is legacy data with mismatch
        # Legacy data will have all class 0 labels first
        y_test = test_data['y_test']
        test_files = test_data['test_files']
        
        # Count consecutive zeros at start
        consecutive_zeros = 0
        for label in y_test:
            if label == 0:
                consecutive_zeros += 1
            else:
                break
        
        # If more than 10 consecutive zeros and there are class 1 samples,
        # this is likely legacy class-grouped data
        has_class_1 = any(label == 1 for label in y_test)
        is_legacy = consecutive_zeros > 10 and has_class_1
        
        if not is_legacy:
            if self.verbose:
                print("  Data appears to be in correct order already")
            return test_data
        
        print(f"⚠️  Detected legacy file order mismatch ({consecutive_zeros} consecutive class 0 samples)")
        print("   Re-loading images in correct order to match filenames...")
        
        # Re-load images in the correct order (matching test_files order)
        X_test_reordered = []
        y_test_reordered = []
        files_reordered = []
        
        data_dir = test_data.get('data_dir', '')
        img_width = test_data['img_width']
        img_height = test_data.get('img_height', img_width)
        img_channels = test_data.get('img_channels', 1)
        classifier_type = test_data.get('classifier_type', 'cwt_image')
        
        for file_path in test_files:
            try:
                # Determine label from filename (need to load from CSV or infer)
                # For now, we'll load the image and match it with existing X_test
                img_path = Path(file_path) if Path(file_path).exists() else Path(data_dir) / Path(file_path).name
                
                if not img_path.exists():
                    print(f"Warning: Cannot find image {img_path}, skipping")
                    continue
                
                # Load image
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Cannot load image {img_path}, skipping")
                    continue
                
                # Resize and normalize
                img = cv2.resize(img, (img_width, img_height))
                if img_channels == 1:
                    img = np.expand_dims(img, axis=-1)
                img = img.astype(np.float32) / 255.0
                
                # Find matching image in original X_test to get label
                original_X = test_data['X_test']
                found_match = False
                for i, orig_img in enumerate(original_X):
                    if np.allclose(img, orig_img, atol=1e-6):
                        X_test_reordered.append(img)
                        y_test_reordered.append(y_test[i])
                        files_reordered.append(file_path)
                        found_match = True
                        break
                
                if not found_match:
                    print(f"Warning: Could not match image {img_path} with original data")
                    
            except Exception as e:
                print(f"Warning: Error processing {file_path}: {e}")
                continue
        
        import numpy as np
        test_data['X_test'] = np.array(X_test_reordered)
        test_data['y_test'] = np.array(y_test_reordered)
        test_data['test_files'] = files_reordered
        
        print(f"✅ Reordered {len(files_reordered)} samples to match filename order")
        
        return test_data


    def run_test_evaluation(self, source_version, test_data_file):
        """Run comprehensive test evaluation with threshold optimization and Grad-CAM analysis."""
        from config import get_next_version_from_log, format_version
        from tensorflow.keras.models import load_model

        print(f"\n🧪 Running comprehensive test evaluation...")

        # Use the specified version (from --version parameter or just-trained model)
        if source_version:
            # User specified a version explicitly (eval_only mode)
            version_str = format_version(source_version)
            print(f"Using specified model version: {version_str}")
        else:
            # Get the version that was just created (post-training evaluation)
            current_version = get_next_version_from_log(classifier_type=self.classifier_type) - 1
            version_str = format_version(current_version)
            print(f"Using just-trained model version: {version_str}")

        # Use proper output directory based on classifier type
        if self.classifier_type == 'cwt_image':
            output_base_dir = CWT_OUTPUTS_DIR
        else:
            output_base_dir = PD_OUTPUTS_DIR

        model_dir = output_base_dir / version_str

        # Check if model directory exists
        if not model_dir.exists():
            raise FileNotFoundError(
                f"❌ Model directory not found: {model_dir}\n"
                f"   Version {version_str} does not exist.\n"
                f"   Available versions: {sorted([d.name for d in output_base_dir.iterdir() if d.is_dir() and d.name.startswith('v')])}"
            )

        # Find the best model file
        if self.classifier_type == 'cwt_image':
            # For CWT, look for best_model from 5-fold CV
            model_files = list(model_dir.glob('best_model*.h5'))
            if not model_files:
                # Fallback to models directory
                model_files = list(model_dir.glob('models/*.h5')) + list(model_dir.glob('models/*.keras'))
        else:
            # For PD, look in models directory
            model_files = list(model_dir.glob('models/*.h5')) + list(model_dir.glob('models/*.keras'))

        if not model_files:
            raise FileNotFoundError(
                f"❌ No trained model found in {model_dir}\n"
                f"   Expected: best_model*.h5 or models/*.h5 or models/*.keras\n"
                f"   The model directory exists but contains no model files."
            )
            return False
        
        # Use the most recent model file
        model_file = sorted(model_files)[-1]
        
        if self.verbose:
            print(f"Loading trained model: {model_file}")
        
        try:
            # Load the trained model
            model = load_model(model_file)
            
            # Load test data
            with open(test_data_file, 'rb') as f:
                test_data = pickle.load(f)

            # Fix file order mismatch in old pickled data
            # Old data has files in CSV order but images/labels in class-grouped order
            if 'test_files' in test_data and 'y_test' in test_data:
                test_data = self._fix_legacy_test_data_order(test_data)

            # Run comprehensive evaluation
            evaluation_results = self.evaluate_with_threshold_optimization(
                model, test_data, model_dir, version_str
            )
            
            print(f"✅ Comprehensive test evaluation completed!")
            print(f"   Best threshold: {evaluation_results['best_threshold']:.3f}")
            print(f"   Test accuracy: {evaluation_results['best_metrics']['accuracy']:.4f}")
            print(f"   Test F1-score: {evaluation_results['best_metrics']['f1_score']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test evaluation failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def _generate_pv_map_for_test_set(self, test_files, output_dir, version):
        """
        Generate P-V map showing the test set track locations.

        Args:
            test_files: List of test file paths
            output_dir: Output directory for saving the figure
            version: Version string for naming the output file

        Returns:
            Dictionary with P-V map generation results
        """
        import re
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from tools import generate_pv_map

        print(f"\n🗺️  Generating P-V map for test set...")

        # Extract unique track IDs from test files
        test_trackids = set()
        for filepath in test_files:
            filename = Path(filepath).name
            # Extract track ID from filename (e.g., "0105_01_0.2-1.2ms.png" -> "0105_01")
            parts = filename.split('_')
            if len(parts) >= 2:
                track_id = f"{parts[0]}_{parts[1]}"
                test_trackids.add(track_id)

        test_trackids = sorted(list(test_trackids))

        if not test_trackids:
            print("Warning: Could not extract track IDs from test files")
            return None

        print(f"   Found {len(test_trackids)} unique tracks in test set")

        # Generate P-V map with test set highlighted
        output_path = Path(output_dir) / f'pv_map_test_set_{version}.png'

        try:
            # Get all track IDs from the same dataset for background
            # (AlSi10Mg, CW, Layer 1, powder)
            from tools import get_logbook
            logbook = get_logbook()
            AlSi10Mg = logbook['Substrate material'] == 'AlSi10Mg'
            L1 = logbook['Layer'] == 1
            cw = logbook['Point jump delay [us]'] == 0
            powder = logbook['Powder material'] != 'None'

            background_trackids = logbook[AlSi10Mg & L1 & cw & powder]['trackid'].unique().tolist()

            # Generate P-V map
            fig, ax = generate_pv_map(
                trackids=background_trackids,  # All possible tracks (shown in grey)
                output_path=output_path,
                highlight_trackids=test_trackids,  # Test set (highlighted with red ring)
                figsize=(4, 3.2),
                dpi=300,
                font_size=8,
                show_background_points=True,
                show_led_contours=False
            )

            results = {
                'unique_tracks': len(test_trackids),
                'track_ids': test_trackids,
                'output_file': str(output_path)
            }

            print(f"   P-V map saved to: {output_path}")

            return results

        except Exception as e:
            print(f"Warning: Could not generate P-V map: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None

    def _select_best_fold_model(self, version_dir, fold_models):
        """
        Select the best fold model based on validation F1-score from experiment summary.

        Args:
            version_dir: Path to version directory
            fold_models: List of fold model file paths

        Returns:
            Path to best fold model, or None if couldn't determine
        """
        import json

        try:
            # Try to load experiment summary JSON
            summary_json = version_dir / 'logs' / f'experiment_summary_{version_dir.name}.json'

            if not summary_json.exists():
                print(f"   No experiment summary found at {summary_json}")
                return None

            with open(summary_json, 'r') as f:
                summary = json.load(f)

            # Extract fold F1 scores from results
            if 'results' not in summary or 'fold_f1_scores' not in summary['results']:
                print("   No fold_f1_scores in experiment summary")
                return None

            fold_f1_scores = summary['results']['fold_f1_scores']

            # Find fold with highest F1 score (folds are 1-indexed)
            best_fold_idx = None
            best_f1 = -1

            for idx, f1_score in enumerate(fold_f1_scores):
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_fold_idx = idx + 1  # Convert to 1-indexed

            if best_fold_idx is None:
                print("   Could not find F1 scores in fold results")
                return None

            # Find the model file for the best fold
            best_model_pattern = f'best_model_fold_{best_fold_idx}'
            for model_path in fold_models:
                if best_model_pattern in model_path.name:
                    print(f"✅ Selected best performing fold: {best_fold_idx} (F1={best_f1:.4f})")
                    print(f"   Using model: {model_path.name}")
                    return model_path

            print(f"   Could not find model file for best fold {best_fold_idx}")
            return None

        except Exception as e:
            print(f"   Error selecting best fold: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None

def main():
    parser = argparse.ArgumentParser(description='Train final production model using best hyperparameters')
    parser.add_argument('--version', type=int,
                       help='Version number to use (for training: uses this config; for --eval_only: evaluates this model)')
    parser.add_argument('--classifier_type', type=str, choices=['pd_signal', 'cwt_image'], 
                       default='pd_signal', help='Type of classifier to train (default: pd_signal)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed training output')
    parser.add_argument('--test_vis', action='store_true',
                       help='Regenerate visualizations from existing test data (old --test behavior)')
    parser.add_argument('--eval_only', action='store_true',
                       help='Skip training and run evaluation on latest model using dataset variant test set')
    parser.add_argument('--k_folds', type=int,
                       help='Override number of k-folds for cross-validation')
    parser.add_argument('--dataset_variant', type=str,
                       help='Dataset variant name to use for test holdout (instead of random sampling)')

    args = parser.parse_args()

    # Initialize trainer
    trainer = FinalModelTrainer(classifier_type=args.classifier_type, verbose=args.verbose, k_folds_override=args.k_folds, dataset_variant=args.dataset_variant)
    
    try:
        if args.test_vis:
            cmd = [sys.executable, 'ml/model_tester.py',
                   '--classifier_type', args.classifier_type or 'cwt_image']
            if args.version:
                cmd += ['--version', str(args.version)]
            result = subprocess.run(cmd, encoding='utf-8', errors='replace',
                                    cwd=str(Path(__file__).parent.parent))
            sys.exit(result.returncode)

        elif args.eval_only:
            if not args.dataset_variant:
                print("❌ --eval_only requires --dataset_variant")
                sys.exit(1)
            cmd = [sys.executable, 'ml/model_tester.py',
                   '--classifier_type', args.classifier_type or 'cwt_image',
                   '--dataset_variant', args.dataset_variant,
                   '--full']
            if args.version:
                cmd += ['--version', str(args.version)]
            result = subprocess.run(cmd, encoding='utf-8', errors='replace',
                                    cwd=str(Path(__file__).parent.parent))
            sys.exit(result.returncode)

        else:
            # Training mode: normal operation
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