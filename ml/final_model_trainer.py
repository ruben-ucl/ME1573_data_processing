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

import cv2
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    get_pd_experiment_log_path, get_cwt_experiment_log_path, format_version, get_next_version_from_log,
    convert_numpy_types, normalize_path, PD_OUTPUTS_DIR, CWT_OUTPUTS_DIR, 
    PD_HYPEROPT_RESULTS_DIR, CWT_HYPEROPT_RESULTS_DIR, ensure_directories, ensure_cwt_directories
)
from data_utils import normalize_image, split_dual_branch_image, estimate_memory_usage_gb

class FinalModelTrainer:
    """Trains final production model using best hyperparameters."""
    
    def __init__(self, classifier_type='pd_signal', verbose=False, k_folds_override=None, test_holdout_file=None):
        self.classifier_type = classifier_type
        self.verbose = verbose
        self.k_folds_override = k_folds_override
        self.test_holdout_file = test_holdout_file
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
        
        # Set appropriate data directory key based on classifier type
        if self.classifier_type == 'cwt_image':
            data_dir_key = 'cwt_data_dir'
            data_dir_source = log_row['cwt_data_dir']  # CWT logs use 'cwt_data_dir' column
        else:
            data_dir_key = 'data_dir'
            data_dir_source = log_row['data_dir']  # PD logs use 'data_dir' column
        
        config = {
            data_dir_key: data_dir_source,
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
        Can use either random sampling or preexisting trackid-based holdout.
        Unified approach for both PD and CWT classifiers.
        
        Args:
            config: Configuration dictionary containing data_dir/cwt_data_dir, img_width, img_height
            output_version_dir: Output directory for this model version
            
        Returns:
            tuple: (test_data_file, exclusion_file) both in output_version_dir
        """
        from glob import glob
        sys.path.append('ml')
        
        # Check if using preexisting test holdout file
        if self.test_holdout_file:
            return self._use_preexisting_holdout(config, output_version_dir)
        
        print(f"\n📊 Creating test set holdout via random file sampling...")
        
        # Get classifier-specific parameters
        if self.classifier_type == 'cwt_image':
            data_dir = config['cwt_data_dir']
            img_width = config['img_width']
            img_height = config['img_height']
            img_channels = config['img_channels']
            file_pattern = "*.png"
        else:  # pd_signal
            data_dir = config['data_dir']
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
            X_test, y_test_filtered = self._load_cwt_test_images(test_files, test_labels, img_width, img_height, img_channels)
        else:  # pd_signal
            X_test, y_test_filtered = self._load_pd_test_images(test_files, test_labels, img_width)
        
        # Create comprehensive test set metadata
        test_data = {
            'X_test': X_test,
            'y_test': np.array(y_test_filtered),
            'data_dir': data_dir,  # Original directory
            'classifier_type': self.classifier_type,
            'img_width': img_width,
            'img_height': img_height if self.classifier_type == 'cwt_image' else None,
            'img_channels': img_channels if self.classifier_type == 'cwt_image' else None,
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
            data_dir = config['cwt_data_dir']
            img_width = config['img_width']
            img_height = config['img_height']
            img_channels = config['img_channels']
            file_pattern = "*.png"
        else:  # pd_signal
            data_dir = config['data_dir']
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
            X_test, y_test_filtered = self._load_cwt_test_images(test_files, test_labels, img_width, img_height, img_channels)
        else:  # pd_signal
            X_test, y_test_filtered = self._load_pd_test_images(test_files, test_labels, img_width)
        
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
            'test_files': [str(f) for f in test_files],
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

    def _load_cwt_test_images(self, test_files, test_labels, img_width, img_height, img_channels):
        """Load CWT test images from file paths."""
        X_test = []
        y_test_filtered = []
        
        for class_label in [0, 1]:
            class_files = test_files[test_labels == class_label]
            for file_path in class_files:
                try:
                    # Load CWT image (PNG format)
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Resize to expected dimensions
                        img = cv2.resize(img, (img_width, img_height))
                        
                        # Add channel dimension if needed
                        if img_channels == 1:
                            img = np.expand_dims(img, axis=-1)
                        
                        # Normalize to [0, 1]
                        img = img.astype(np.float32) / 255.0
                        
                        X_test.append(img)
                        y_test_filtered.append(class_label)
                except Exception as e:
                    self.logger.warning(f"Could not load CWT test image {file_path}: {e}")
                    if self.verbose:
                        print(f"Warning: Could not load CWT test image {file_path}: {e}")
        
        return np.array(X_test), y_test_filtered

    def _load_pd_test_images(self, test_files, test_labels, img_width):
        """Load PD test images from file paths."""
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
                    self.logger.warning(f"Could not load PD test image {file_path}: {e}")
                    if self.verbose:
                        print(f"Warning: Could not load PD test image {file_path}: {e}")
        
        return (np.array(pd1_test), np.array(pd2_test)), y_test_filtered

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
                'balanced_score': (acc + f1) / 2  # Compromise between accuracy and F1
            })
        
        # Find best threshold based on balanced score (accuracy + F1) / 2
        best_result = max(threshold_results, key=lambda x: x['balanced_score'])
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
        threshold_csv = Path(output_dir) / f'threshold_optimization_{version}.csv'
        threshold_df.to_csv(threshold_csv, index=False, encoding='utf-8')
        
        # Plot threshold optimization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1_score']):
            axes[i].plot(threshold_df['threshold'], threshold_df[metric], 'b-', linewidth=2)
            axes[i].axvline(best_threshold, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_threshold:.3f}')
            axes[i].set_xlabel('Classification Threshold')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'threshold_optimization_{version}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred_best)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Class 0', 'Class 1'], 
                   yticklabels=['Class 0', 'Class 1'])
        plt.title(f'Confusion Matrix (Threshold: {best_threshold:.3f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(Path(output_dir) / f'confusion_matrix_{version}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred_best, output_dict=True)
        class_report_str = classification_report(y_test, y_pred_best)
        
        # Save classification report
        with open(Path(output_dir) / f'classification_report_{version}.txt', 'w') as f:
            f.write(f"Classification Report - Threshold: {best_threshold:.3f}\n")
            f.write("="*50 + "\n")
            f.write(class_report_str)
            f.write(f"\nAUC Score: {auc_score:.4f}" if auc_score else "\nAUC Score: N/A (single class)")
        
        # Generate Grad-CAM analysis for CWT classifiers
        gradcam_results = None
        if classifier_type == 'cwt_image':
            gradcam_results = self._generate_comprehensive_gradcam(
                model, X_test, y_test, y_pred_best, y_proba_flat, 
                best_threshold, output_dir, version
            )
        
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
            'output_files': {
                'threshold_csv': str(threshold_csv),
                'threshold_plot': str(Path(output_dir) / f'threshold_optimization_{version}.png'),
                'confusion_matrix': str(Path(output_dir) / f'confusion_matrix_{version}.png'),
                'classification_report': str(Path(output_dir) / f'classification_report_{version}.txt')
            }
        }
        
        # Save comprehensive results as JSON
        results_json = Path(output_dir) / f'comprehensive_evaluation_{version}.json'
        with open(results_json, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        print(f"📊 Evaluation complete!")
        print(f"   Results saved to: {results_json}")
        if gradcam_results:
            print(f"   Grad-CAM analysis: {gradcam_results['total_images']} images analyzed")
        
        return evaluation_results

    def _generate_comprehensive_gradcam(self, model, X_test, y_test, y_pred, y_proba, threshold, output_dir, version):
        """Generate comprehensive Grad-CAM analysis with class-specific folders and averages."""
        import tensorflow as tf
        from tensorflow.keras.models import Model
        
        print(f"🔥 Generating comprehensive Grad-CAM analysis...")
        
        gradcam_dir = Path(output_dir) / f'gradcam_analysis_{version}'
        
        # Create class-specific directories
        class_dirs = {}
        for class_label in [0, 1]:
            class_dirs[class_label] = {
                'correct': gradcam_dir / f'class_{class_label}' / 'correct_predictions',
                'incorrect': gradcam_dir / f'class_{class_label}' / 'incorrect_predictions',
                'all': gradcam_dir / f'class_{class_label}' / 'all_samples'
            }
            for dir_path in class_dirs[class_label].values():
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create summary directory
        summary_dir = gradcam_dir / 'class_averages'
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the best layer for Grad-CAM (last convolutional layer)
        conv_layers = [layer for layer in model.layers if 'conv' in layer.name.lower()]
        if not conv_layers:
            print("Warning: No convolutional layers found for Grad-CAM")
            return None
            
        target_layer = conv_layers[-1]
        print(f"Using layer '{target_layer.name}' for Grad-CAM analysis")
        
        # Create Grad-CAM model
        gradcam_model = Model(inputs=model.input, outputs=[target_layer.output, model.output])
        
        # Store heatmaps for averaging
        class_heatmaps = {0: [], 1: []}
        gradcam_results = {
            'target_layer': target_layer.name,
            'threshold': threshold,
            'total_images': len(X_test),
            'class_analysis': {0: {'correct': 0, 'incorrect': 0}, 1: {'correct': 0, 'incorrect': 0}},
            'saved_images': 0
        }
        
        for i in range(len(X_test)):
            true_label = int(y_test[i])
            pred_label = int(y_pred[i])
            confidence = float(y_proba[i])
            is_correct = (true_label == pred_label)
            
            # Generate Grad-CAM heatmap
            heatmap = self._generate_gradcam_heatmap(
                gradcam_model, X_test[i:i+1], target_layer_idx=-2
            )
            
            if heatmap is not None:
                # Store for class averaging
                class_heatmaps[true_label].append(heatmap)
                
                # Save individual image
                prediction_type = 'correct' if is_correct else 'incorrect'
                filename = f'sample_{i:04d}_true_{true_label}_pred_{pred_label}_conf_{confidence:.3f}.png'
                
                # Save to class-specific directories
                self._save_gradcam_image(
                    X_test[i], heatmap, 
                    class_dirs[true_label][prediction_type] / filename,
                    title=f'True: {true_label}, Pred: {pred_label}, Conf: {confidence:.3f}'
                )
                
                self._save_gradcam_image(
                    X_test[i], heatmap, 
                    class_dirs[true_label]['all'] / filename,
                    title=f'True: {true_label}, Pred: {pred_label}, Conf: {confidence:.3f}'
                )
                
                gradcam_results['saved_images'] += 1
                gradcam_results['class_analysis'][true_label][prediction_type] += 1
        
        # Generate class average heatmaps
        for class_label in [0, 1]:
            if class_heatmaps[class_label]:
                avg_heatmap = np.mean(class_heatmaps[class_label], axis=0)
                
                # Create a representative image (mean of class images)
                class_indices = np.where(y_test == class_label)[0]
                representative_img = np.mean(X_test[class_indices], axis=0)
                
                self._save_gradcam_image(
                    representative_img, avg_heatmap,
                    summary_dir / f'class_{class_label}_average_gradcam.png',
                    title=f'Class {class_label} Average Grad-CAM (n={len(class_heatmaps[class_label])})'
                )
        
        # Generate difference heatmap (Class 1 - Class 0)
        if class_heatmaps[0] and class_heatmaps[1]:
            avg_heatmap_0 = np.mean(class_heatmaps[0], axis=0)
            avg_heatmap_1 = np.mean(class_heatmaps[1], axis=0)
            diff_heatmap = avg_heatmap_1 - avg_heatmap_0
            
            # Create difference visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(avg_heatmap_0, cmap='viridis')
            plt.title('Class 0 Average')
            plt.colorbar()
            
            plt.subplot(1, 3, 2)
            plt.imshow(avg_heatmap_1, cmap='viridis')
            plt.title('Class 1 Average')
            plt.colorbar()
            
            plt.subplot(1, 3, 3)
            plt.imshow(diff_heatmap, cmap='RdBu_r', vmin=-np.max(np.abs(diff_heatmap)), vmax=np.max(np.abs(diff_heatmap)))
            plt.title('Difference (Class 1 - Class 0)')
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(summary_dir / 'class_difference_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"   Grad-CAM images saved to: {gradcam_dir}")
        print(f"   Generated {gradcam_results['saved_images']} individual heatmaps")
        print(f"   Class 0: {gradcam_results['class_analysis'][0]} samples")
        print(f"   Class 1: {gradcam_results['class_analysis'][1]} samples")
        
        return gradcam_results

    def _generate_gradcam_heatmap(self, gradcam_model, img_array, target_layer_idx=-2):
        """Generate Grad-CAM heatmap for a single image."""
        import tensorflow as tf
        
        try:
            with tf.GradientTape() as tape:
                conv_outputs, predictions = gradcam_model(img_array)
                loss = predictions[:, 0]  # Binary classification
            
            # Calculate gradients
            grads = tape.gradient(loss, conv_outputs)
            
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the feature maps by the gradients
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            return heatmap.numpy()
            
        except Exception as e:
            print(f"Warning: Could not generate Grad-CAM for image: {e}")
            return None

    def _save_gradcam_image(self, original_img, heatmap, filepath, title=None):
        """Save Grad-CAM visualization combining original image and heatmap."""
        import matplotlib.pyplot as plt
        import cv2
        
        try:
            # Prepare original image for display
            if len(original_img.shape) == 3 and original_img.shape[-1] == 1:
                display_img = original_img.squeeze()
            else:
                display_img = original_img
            
            # Resize heatmap to match image dimensions
            heatmap_resized = cv2.resize(heatmap, (display_img.shape[1], display_img.shape[0]))
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(display_img, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Heatmap
            im1 = axes[1].imshow(heatmap_resized, cmap='viridis')
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1])
            
            # Overlay
            axes[2].imshow(display_img, cmap='gray', alpha=0.7)
            axes[2].imshow(heatmap_resized, cmap='viridis', alpha=0.3)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            if title:
                fig.suptitle(title, fontsize=12)
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not save Grad-CAM image to {filepath}: {e}")

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
        
        try:
            # Use unified holdout logic for both PD and CWT classifiers
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
        
        # Add exclusion logic for both PD and CWT classifiers
        cmd.extend(['--exclude_files', str(exclusion_file)])  # Skip test set files
        
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
        
        # Step 3: Find the trained model and run test evaluation (only for PD classifiers)
        if self.classifier_type == 'pd_signal' and test_data_file is not None:
            return self.run_test_evaluation(source_version, test_data_file)
        else:
            print(f"\n✅ Final model training completed successfully!")
            print(f"Note: Test evaluation not available for {self.classifier_type} classifier type")
            return True
    
    def run_test_evaluation(self, source_version, test_data_file):
        """Run comprehensive test evaluation with threshold optimization and Grad-CAM analysis."""
        from config import get_next_version_from_log, format_version
        from tensorflow.keras.models import load_model
        
        print(f"\n🧪 Running comprehensive test evaluation...")
        
        # Get the version that was just created
        current_version = get_next_version_from_log(classifier_type=self.classifier_type) - 1  # Just created version
        version_str = format_version(current_version)
        
        # Use proper output directory based on classifier type
        if self.classifier_type == 'cwt_image':
            output_base_dir = CWT_OUTPUTS_DIR
        else:
            output_base_dir = PD_OUTPUTS_DIR
        
        model_dir = output_base_dir / version_str
        
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
            print(f"❌ No trained model found in {model_dir}")
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

    def test_latest_model(self):
        """Test the latest existing model without training."""
        from config import get_next_version_from_log, format_version
        
        print(f"\n🔍 Finding latest trained model...")
        
        # Use proper output directory based on classifier type
        if self.classifier_type == 'cwt_image':
            output_base_dir = CWT_OUTPUTS_DIR
        else:
            output_base_dir = PD_OUTPUTS_DIR
        
        # Find the latest version directory
        version_dirs = [d for d in output_base_dir.iterdir() if d.is_dir() and d.name.startswith('v')]
        if not version_dirs:
            print(f"❌ No model versions found in {output_base_dir}")
            return False
        
        # Sort by version number and get the latest
        latest_version_dir = sorted(version_dirs, key=lambda x: int(x.name[1:]))[-1]
        version_str = latest_version_dir.name
        
        print(f"Found latest model version: {version_str}")
        
        # Find the model file
        model_files = list(latest_version_dir.glob('models/*.h5')) + list(latest_version_dir.glob('models/*.keras'))
        
        if not model_files:
            print(f"❌ No trained model found in {latest_version_dir}/models/")
            return False
        
        # Use the most recent model file
        model_file = sorted(model_files)[-1]
        
        # Check for existing test data
        test_data_file = latest_version_dir / 'test_data.pkl'
        
        if not test_data_file.exists():
            print(f"❌ No test data found in {latest_version_dir}")
            print("   Test data is required for evaluation. Please run full training first.")
            return False
        
        print(f"Using model: {model_file.name}")
        print(f"Using test data: {test_data_file.name}")
        
        # Prepare command for model tester
        cmd = [
            sys.executable, 'ml/model_tester.py',
            '--model', str(model_file),
            '--test_data', str(test_data_file),
            '--output_dir', str(latest_version_dir),
            '--model_version', version_str
        ]
        
        if self.verbose:
            cmd.append('--verbose')
        
        try:
            # Run test evaluation
            print(f"\n🧪 Running test evaluation...")
            result = subprocess.run(
                cmd,
                timeout=600,  # 10 minute timeout
                encoding='utf-8',
                errors='replace',
                cwd=str(Path(__file__).parent.parent)  # Project root
            )
            
            if result.returncode == 0:
                print(f"\n✅ Model evaluation completed successfully!")
                print(f"Results saved to: {latest_version_dir}")
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
    parser.add_argument('--classifier_type', type=str, choices=['pd_signal', 'cwt_image'], 
                       default='pd_signal', help='Type of classifier to train (default: pd_signal)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed training output')
    parser.add_argument('--test', action='store_true',
                       help='Skip training and just test the latest model')
    parser.add_argument('--k_folds', type=int,
                       help='Override number of k-folds for cross-validation')
    parser.add_argument('--test_holdout_file', type=str,
                       help='Path to preexisting test holdout file with trackids (instead of random sampling)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = FinalModelTrainer(classifier_type=args.classifier_type, verbose=args.verbose, k_folds_override=args.k_folds, test_holdout_file=args.test_holdout_file)
    
    try:
        if args.test:
            # Test mode: skip training and just test the latest model
            print("🧪 Test mode: Skipping training and testing latest model...")
            success = trainer.test_latest_model()
            
            if success:
                print("\n🎉 Model testing completed successfully!")
                sys.exit(0)
            else:
                print("\n❌ Model testing failed!")
                sys.exit(1)
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