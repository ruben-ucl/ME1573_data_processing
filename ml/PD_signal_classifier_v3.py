# Ensure UTF-8 encoding for all I/O operations
import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import argparse
import datetime
import json
import random
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, callbacks

from config import (
    get_data_dir, get_pd_experiment_log_path, load_config, format_version,
    PD_OUTPUTS_DIR, normalize_path, ensure_path_exists, convert_numpy_types,
    # Consolidated functions
    log_experiment_results, create_experiment_summary_files, save_fold_plots
)
from data_utils import normalize_image

# -------------------------
# Concise Progress Callback
# -------------------------
class ConciseProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback for concise progress updates."""
    
    def __init__(self, fold, total_folds, max_epochs):
        super().__init__()
        self.fold = fold
        self.total_folds = total_folds
        self.max_epochs = max_epochs
        self.best_val_acc = 0.0
        self.best_overall_acc = 0.0
        self.last_epoch = 0
        
    def set_best_overall(self, best_overall):
        """Set the best accuracy from previous folds."""
        self.best_overall_acc = best_overall
        
    def on_epoch_end(self, epoch, logs=None):
        """Print concise progress update."""
        logs = logs or {}
        val_acc = logs.get('val_accuracy', 0.0)
        current_epoch = epoch + 1
        
        # Track if this epoch set new records
        is_new_fold_best = val_acc > self.best_val_acc
        is_new_overall_best = val_acc > self.best_overall_acc
        
        # Update best accuracy for this fold
        if is_new_fold_best:
            self.best_val_acc = val_acc
            
        # Update best overall if this epoch is better
        if is_new_overall_best:
            self.best_overall_acc = val_acc
        
        # Only print every 5 epochs or if it's the last epoch or if accuracy improved
        should_print = (
            current_epoch % 5 == 0 or 
            current_epoch == self.max_epochs or 
            is_new_fold_best or 
            is_new_overall_best or
            current_epoch == 1
        )
        
        if should_print:
            # Basic progress info
            progress_line = f"Fold {self.fold}/{self.total_folds} | Epoch {current_epoch}/{self.max_epochs} | Val Acc: {val_acc:.4f}"
            
            # Add achievement symbols
            if is_new_overall_best:
                progress_line += " 🏆"  # Trophy for new overall best
            elif is_new_fold_best:
                progress_line += " ⭐"  # Star for new fold best
            
            print(progress_line, flush=True)

# -------------------------
# 1. Load images + labels
# -------------------------
def load_data(root_dir, img_width, verbose=False, exclude_files=None):
    """Load image data from directory structure."""
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Data directory not found: {root_dir}")
    
    if not root_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root_dir}")
    
    images, labels = [], []
    valid_classes = []
    
    # Load exclusion set if provided
    excluded_files = set()
    if exclude_files and Path(exclude_files).exists():
        with open(exclude_files, 'r') as f:
            excluded_files = set(line.strip() for line in f if line.strip())
        if verbose:
            print(f"Loaded {len(excluded_files)} files to exclude from training")
    
    try:
        dir_contents = os.listdir(root_dir)
    except PermissionError:
        raise PermissionError(f"Permission denied accessing directory: {root_dir}")
    
    for label_dir in dir_contents:
        if len(label_dir) > 1:
            continue
        class_path = Path(root_dir) / label_dir
        if not class_path.is_dir():
            continue
        
        valid_classes.append(label_dir)
        class_images = glob(str(class_path / "*.tiff"))
        
        if not class_images:
            print(f"Warning: No TIFF images found in class directory: {class_path}")
            continue
            
        for f in class_images:
            # Check if this file should be excluded
            filename = Path(f).name
            if filename in excluded_files:
                if verbose:
                    print(f"Skipping excluded file: {filename}")
                continue
                
            try:
                # Try to read as 16-bit first, fallback to 8-bit
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
                
                if img is None:
                    print(f"Warning: Could not read {f}")
                    continue
            except Exception as e:
                print(f"Error reading image {f}: {e}")
                continue
                
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
            
            # Use centralized normalization function
            img = normalize_image(img)
            if verbose:
                print(f"Image: {f}, shape: {img.shape}, normalized range: [{img.min():.6f}, {img.max():.6f}]")
            
            # Split into two separate signals for dual-branch model
            pd1_signal = img[:, 0].reshape(-1, 1)      # First column: (width, 1)
            pd2_signal = img[:, 1].reshape(-1, 1)      # Second column: (width, 1)
            
            images.append((pd1_signal, pd2_signal))    # Store as tuple
            labels.append(label_dir)
    if len(images) == 0:
        raise ValueError(f"No valid images found in {root_dir}")
    
    print(f'Loaded {len(images)} labelled datasets')
    
    # Convert to proper format for dual-branch model
    pd1_data = np.array([img[0] for img in images])  # (n_samples, width, 1)
    pd2_data = np.array([img[1] for img in images])  # (n_samples, width, 1)
    labels = np.array(labels)
    
    # Final normalization check and fix if needed
    pd1_max, pd2_max = pd1_data.max(), pd2_data.max()
    if pd1_max < 0.1 or pd2_max < 0.1:
        print(f"\nWARNING: Data appears under-normalized!")
        print(f"Current max values: PD1={pd1_max:.6f}, PD2={pd2_max:.6f}")
        print("Applying additional normalization...")
        
        # Renormalize to [0,1] range
        pd1_data = (pd1_data - pd1_data.min()) / (pd1_data.max() - pd1_data.min())
        pd2_data = (pd2_data - pd2_data.min()) / (pd2_data.max() - pd2_data.min())
        
        print(f"After renormalization: PD1=[{pd1_data.min():.6f}, {pd1_data.max():.6f}]")
        print(f"After renormalization: PD2=[{pd2_data.min():.6f}, {pd2_data.max():.6f}]")
    
    return (pd1_data, pd2_data), labels

# -------------------------
# 2. Augmentation function
# -------------------------
def augment_sample(pd1_signal, pd2_signal, time_shift_range=5, stretch_probability=0.3,
                   stretch_scale=0.1, noise_probability=0.5, noise_std=0.02,
                   amplitude_scale_probability=0.5, amplitude_scale=0.1):
    """Augment a single dual-signal sample with configurable parameters."""
    aug_pd1 = pd1_signal.copy()
    aug_pd2 = pd2_signal.copy()

    # Random horizontal shift (time shift) - apply same shift to both signals
    max_shift = int(time_shift_range)
    shift = np.random.randint(-max_shift, max_shift+1)
    aug_pd1 = np.roll(aug_pd1, shift, axis=0)
    aug_pd2 = np.roll(aug_pd2, shift, axis=0)

    # Random horizontal stretch/compression - apply same to both signals
    if random.random() < stretch_probability:
        # Convert scale parameter to min/max: stretch_scale=0.1 -> uniform(0.9, 1.1)
        scale_min = 1.0 - stretch_scale
        scale_max = 1.0 + stretch_scale
        scale_factor = random.uniform(scale_min, scale_max)
        new_width = int(aug_pd1.shape[0] * scale_factor)
        
        # Resize both signals
        aug_pd1_2d = cv2.resize(aug_pd1.squeeze(), (1, new_width), interpolation=cv2.INTER_LINEAR).reshape(-1, 1)
        aug_pd2_2d = cv2.resize(aug_pd2.squeeze(), (1, new_width), interpolation=cv2.INTER_LINEAR).reshape(-1, 1)
        
        # Pad/crop back to original width
        original_width = pd1_signal.shape[0]
        if new_width < original_width:
            pad_width = original_width - new_width
            aug_pd1 = np.pad(aug_pd1_2d, ((0, pad_width), (0, 0)), mode='constant')[:original_width, :]
            aug_pd2 = np.pad(aug_pd2_2d, ((0, pad_width), (0, 0)), mode='constant')[:original_width, :]
        elif new_width > original_width:
            aug_pd1 = aug_pd1_2d[:original_width, :]
            aug_pd2 = aug_pd2_2d[:original_width, :]
        else:
            aug_pd1 = aug_pd1_2d
            aug_pd2 = aug_pd2_2d

    # Add Gaussian noise
    if random.random() < noise_probability:
        noise_pd1 = np.random.normal(0, noise_std, aug_pd1.shape)
        noise_pd2 = np.random.normal(0, noise_std, aug_pd2.shape)
        aug_pd1 = np.clip(aug_pd1 + noise_pd1, 0.0, 1.0)
        aug_pd2 = np.clip(aug_pd2 + noise_pd2, 0.0, 1.0)

    # Amplitude scaling per signal
    if random.random() < amplitude_scale_probability:
        # Convert scale parameter to min/max: amplitude_scale=0.1 -> uniform(0.9, 1.1)
        scale_min = 1.0 - amplitude_scale
        scale_max = 1.0 + amplitude_scale
        scale_pd1 = random.uniform(scale_min, scale_max)
        aug_pd1 = np.clip(aug_pd1 * scale_pd1, 0.0, 1.0)
    
    if random.random() < amplitude_scale_probability:
        # Convert scale parameter to min/max: amplitude_scale=0.1 -> uniform(0.9, 1.1)
        scale_min = 1.0 - amplitude_scale
        scale_max = 1.0 + amplitude_scale
        scale_pd2 = random.uniform(scale_min, scale_max)
        aug_pd2 = np.clip(aug_pd2 * scale_pd2, 0.0, 1.0)

    return aug_pd1, aug_pd2

def augment_batch(X, y, augment_fraction=0.5, time_shift_range=5, stretch_probability=0.3,
                  stretch_scale=0.1, noise_probability=0.5, noise_std=0.02,
                  amplitude_scale_probability=0.5, amplitude_scale=0.1):
    """
    Apply augmentation to a batch with configurable parameters.
    X is tuple of (pd1_batch, pd2_batch).
    Only augments a fraction of samples to avoid duplicate processing.
    """
    pd1_batch, pd2_batch = X
    num_samples = len(pd1_batch)
    num_to_augment = int(num_samples * augment_fraction)
    
    if num_to_augment == 0:
        return X, y
    
    # Randomly select indices to augment
    indices_to_augment = np.random.choice(num_samples, num_to_augment, replace=False)
    
    # Create augmented versions of selected samples
    pd1_aug_samples = []
    pd2_aug_samples = []
    y_aug_samples = []
    
    for idx in indices_to_augment:
        aug_pd1, aug_pd2 = augment_sample(pd1_batch[idx], pd2_batch[idx], 
                                        time_shift_range, stretch_probability,
                                        stretch_scale, noise_probability, noise_std,
                                        amplitude_scale_probability, amplitude_scale)
        pd1_aug_samples.append(aug_pd1)
        pd2_aug_samples.append(aug_pd2)
        y_aug_samples.append(y[idx])
    
    # Combine original and augmented data
    pd1_combined = np.concatenate([pd1_batch, np.array(pd1_aug_samples)], axis=0)
    pd2_combined = np.concatenate([pd2_batch, np.array(pd2_aug_samples)], axis=0)
    y_combined = np.concatenate([y, np.array(y_aug_samples)], axis=0)
    
    return (pd1_combined, pd2_combined), y_combined

# -------------------------
# 3. Configuration and Model builder
# -------------------------

def build_dual_branch_model(input_shape, n_classes, config):
    """Build model with configurable hyperparameters."""
    conv_filters = config.get('conv_filters', [16, 32, 64])
    dense_units = config.get('dense_units', [128, 64])
    conv_dropout = config.get('conv_dropout', 0.2)
    dense_dropout = config.get('dense_dropout', [0.3, 0.2])
    l2_reg = config.get('l2_regularization', 0.001)
    use_batch_norm = config.get('use_batch_norm', True)
    learning_rate = config.get('learning_rate', 0.001)
    
    def branch_model(input_shape):
        inp = layers.Input(shape=input_shape)
        x = inp
        
        # Configurable conv layers
        for i, filters in enumerate(conv_filters):
            kernel_size = 5 if i == 0 else 3  # First layer uses larger kernel
            x = layers.Conv1D(filters, kernel_size=kernel_size, padding='same')(x)
            
            if use_batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            if i == 0:  # Only pool after first conv block
                x = layers.MaxPooling1D(pool_size=2)(x)
            
            if i > 0:  # Add dropout to later conv layers
                x = layers.Dropout(conv_dropout)(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        return inp, x

    # Build both branches
    input_pd1, branch1 = branch_model(input_shape)
    input_pd2, branch2 = branch_model(input_shape)

    # Merge features with configurable dense layers
    merged = layers.Concatenate()([branch1, branch2])
    
    # Configurable dense layers
    for i, units in enumerate(dense_units):
        merged = layers.Dense(units, activation='relu', 
                             kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None)(merged)
        if use_batch_norm:
            merged = layers.BatchNormalization()(merged)
        
        dropout_rate = dense_dropout[i] if isinstance(dense_dropout, list) else dense_dropout
        merged = layers.Dropout(dropout_rate)(merged)
    
    # Output layer
    output = layers.Dense(n_classes, activation='softmax')(merged)

    model = models.Model(inputs=[input_pd1, input_pd2], outputs=output)
    
    # Configurable optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# -------------------------
# 4. Experiment Tracking Functions
# -------------------------
def get_next_version_number(output_root):
    """Get the next version number based on completed experiments in the log."""
    # Use centralized version management from config
    from config import get_next_version_from_log
    return get_next_version_from_log(classifier_type='pd_signal')


def create_experiment_log_entry(version, hyperparams, data_info, results, output_root, source='manual', 
                               hyperopt_run_id=None, config_file=None, config_number_in_run=None, test_results=None):
    """Create or update the experiment tracking CSV with enhanced traceability."""
    # Use centralized experiment log path
    log_file = str(get_pd_experiment_log_path())
    
    # Create logs directory using centralized path (ensures correct location)
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive log entry
    log_entry = {
        # Version and timestamp
        'version': format_version(version),
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': source,  # 'manual' or 'hyperopt'
        
        # Data information
        'data_dir': data_info['data_dir'],
        'total_samples': int(data_info['total_samples']),
        'num_classes': int(data_info['num_classes']),
        'class_distribution': json.dumps(convert_numpy_types(data_info['class_distribution'])),
        'img_width': data_info['img_width'],
        
        # Model hyperparameters
        'learning_rate': hyperparams['learning_rate'],
        'batch_size': hyperparams['batch_size'],
        'epochs': hyperparams['epochs'],
        'k_folds': hyperparams['k_folds'],
        
        # Architecture parameters
        'conv_filters': json.dumps(convert_numpy_types(hyperparams['conv_filters'])),
        'dense_units': json.dumps(convert_numpy_types(hyperparams['dense_units'])),
        'dropout_rates': json.dumps(convert_numpy_types([hyperparams['conv_dropout'], hyperparams['dense_dropout']])),
        'l2_reg': hyperparams['l2_regularization'],
        'batch_norm': hyperparams.get('use_batch_norm', True),
        
        # Training parameters
        'optimizer': hyperparams.get('optimizer_type', 'adam'),
        'early_stopping_patience': hyperparams['early_stopping_patience'],
        'lr_reduction_patience': hyperparams['lr_reduction_patience'],
        'class_weights': hyperparams['use_class_weights'],
        
        # Augmentation parameters
        'augment_fraction': hyperparams.get('augment_fraction', 0.0),
        'time_shift_range': hyperparams.get('time_shift_range', 0.0),
        'stretch_probability': hyperparams.get('stretch_probability', 0.0),
        'stretch_scale': hyperparams.get('stretch_scale', 0.0),
        'noise_probability': hyperparams.get('noise_probability', 0.0),
        'noise_std': hyperparams.get('noise_std', 0.0),
        'amplitude_scale_probability': hyperparams.get('amplitude_scale_probability', 0.0),
        'amplitude_scale': hyperparams.get('amplitude_scale', 0.0),
        
        # Augmentation data metrics (meaningful relative to total_samples already in log)
        # augmented_samples_per_fold: Average NEW samples created per fold (not cumulative)
        # total_augmented_samples_created: Total NEW samples across all CV folds
        # augmentation_ratio: Ratio of augmented samples to training samples per fold
        'augmented_samples_per_fold': results.get('augmented_samples_per_fold', 0.0),
        'total_augmented_samples_created': results.get('total_augmented_samples_created', 0),
        'augmentation_ratio': results.get('augmentation_ratio', 0.0),
        
        # Results
        'mean_val_accuracy': results['mean_accuracy'],
        'std_val_accuracy': results['std_accuracy'],
        'mean_val_loss': results['mean_loss'],
        'std_val_loss': results['std_loss'],
        'best_fold_accuracy': results['best_accuracy'],
        'best_fold_number': results['best_fold'],
        'per_fold_accuracies': json.dumps(results['fold_accuracies']),
        'per_fold_losses': json.dumps(results['fold_losses']),
        
        # Training info
        'total_training_time_minutes': results.get('training_time_minutes', 0),
        'convergence_issues': results.get('convergence_issues', ''),
        'notes': results.get('notes', ''),
        
        # Enhanced traceability fields (new)
        'hyperopt_run_id': hyperopt_run_id if hyperopt_run_id else '',
        'config_file': config_file if config_file else '',
        'config_number_in_run': config_number_in_run if config_number_in_run is not None else -1,
        
        # Test results fields (new)
        'test_accuracy': test_results.get('test_accuracy') if test_results else None,
        'test_precision': test_results.get('test_precision') if test_results else None,
        'test_recall': test_results.get('test_recall') if test_results else None,
        'test_f1_score': test_results.get('test_f1_score') if test_results else None,
        'test_roc_auc': test_results.get('test_roc_auc') if test_results else None,
        'test_samples': test_results.get('test_samples') if test_results else None
    }
    
    # Create or append to CSV
    if Path(log_file).exists():
        df = pd.read_csv(log_file, encoding='utf-8')
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])
    
    df.to_csv(log_file, index=False, encoding='utf-8')
    return log_file


def update_experiment_log_with_test_results(version, test_results):
    """Update existing experiment log entry with test results."""
    log_file = str(get_pd_experiment_log_path())
    
    if not Path(log_file).exists():
        raise FileNotFoundError(f"Experiment log not found: {log_file}")
    
    # Read existing log
    df = pd.read_csv(log_file, encoding='utf-8')
    
    # Find the entry to update
    version_str = format_version(version) if isinstance(version, int) else str(version)
    matching_rows = df['version'] == version_str
    
    if not matching_rows.any():
        raise ValueError(f"Version {version_str} not found in experiment log")
    
    # Update the most recent entry for this version
    latest_idx = df[matching_rows].index[-1]
    
    # Add test results columns if they don't exist
    test_columns = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1_score', 'test_roc_auc', 'test_samples']
    for col in test_columns:
        if col not in df.columns:
            df[col] = None
    
    # Update test results
    df.loc[latest_idx, 'test_accuracy'] = test_results.get('test_accuracy')
    df.loc[latest_idx, 'test_precision'] = test_results.get('test_precision')
    df.loc[latest_idx, 'test_recall'] = test_results.get('test_recall')
    df.loc[latest_idx, 'test_f1_score'] = test_results.get('test_f1_score')
    df.loc[latest_idx, 'test_roc_auc'] = test_results.get('test_roc_auc')
    df.loc[latest_idx, 'test_samples'] = test_results.get('test_samples')
    
    # Save updated log
    df.to_csv(log_file, index=False, encoding='utf-8')
    return log_file

def setup_experiment_directory(version, output_root):
    """Create directory structure for this experiment version."""
    version_dir = Path(output_root) / format_version(version)
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['models', 'history', 'predictions', 'plots', 'logs']
    for subdir in subdirs:
        (version_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return str(version_dir)

def extract_hyperparameters(config, model=None):
    """Extract hyperparameters from configuration and model."""
    hyperparams = {
        # Training hyperparameters
        'learning_rate': config.get('learning_rate', 0.001),
        'batch_size': config.get('batch_size', 16),
        'epochs': config.get('epochs', 50),
        'k_folds': config.get('k_folds', 5),
        
        # Model architecture
        'conv_filters': config.get('conv_filters', [16, 32, 64]),
        'dense_units': config.get('dense_units', [128, 64]),
        'conv_dropout': config.get('conv_dropout', 0.2),
        'dense_dropout': config.get('dense_dropout', [0.3, 0.2]),
        'l2_regularization': config.get('l2_regularization', 0.001),
        'batch_normalization': config.get('use_batch_norm', True),
        
        # Optimizer settings
        'optimizer_type': 'Adam',
        'early_stopping_patience': config.get('early_stopping_patience', 10),
        'lr_reduction_patience': config.get('lr_reduction_patience', 5),
        'lr_reduction_factor': config.get('lr_reduction_factor', 0.5),
        'use_class_weights': config.get('use_class_weights', True),
        
        # Augmentation settings (read from config)
        'augment_fraction': config.get('augment_fraction', 0.5),
        'time_shift_range': config.get('time_shift_range', 5),
        'stretch_probability': config.get('stretch_probability', 0.3),
        'stretch_scale': config.get('stretch_scale', 0.1),
        'noise_probability': config.get('noise_probability', 0.5),
        'noise_std': config.get('noise_std', 0.02),
        'amplitude_scale_probability': config.get('amplitude_scale_probability', 0.5),
        'amplitude_scale': config.get('amplitude_scale', 0.1),
    }
    
    return hyperparams

# -------------------------
# 5. Diagnostic Functions
# -------------------------
def analyze_class_distribution(y, verbose=True):
    """Analyze and visualize class distribution in the dataset."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    if verbose:
        print("\n" + "="*70)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*70)
        print(f"Total samples: {total}")
        print(f"Number of classes: {len(unique)}")
        print("\nClass breakdown:")
    
    class_info = {}
    for cls, count in zip(unique, counts):
        percentage = count/total*100
        class_info[cls] = {'count': count, 'percentage': percentage}
        if verbose:
            print(f"  Class '{cls}': {count} samples ({percentage:.1f}%)")
    
    # Check for severe imbalance
    max_percentage = max([info['percentage'] for info in class_info.values()])
    min_percentage = min([info['percentage'] for info in class_info.values()])
    imbalance_ratio = max_percentage / min_percentage if min_percentage > 0 else float('inf')
    
    if verbose:
        print(f"\nImbalance Analysis:")
        print(f"  Majority class: {max_percentage:.1f}%")
        print(f"  Minority class: {min_percentage:.1f}%") 
        print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 5:
            print("  WARNING: Severe class imbalance detected!")
            print("  This is likely causing the model to predict only the majority class.")
        
        print("="*70)
    return class_info

def analyze_data_quality(X, y, verbose=True):
    """Analyze data quality and normalization."""
    pd1_data, pd2_data = X
    
    if verbose:
        print("\n" + "="*70)
        print("DATA QUALITY ANALYSIS")
        print("="*70)
        
        # Data shape analysis
        print(f"PD1 data shape: {pd1_data.shape}")
        print(f"PD2 data shape: {pd2_data.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Data range analysis
        print(f"\nData Range Analysis:")
        print(f"  PD1 range: [{pd1_data.min():.6f}, {pd1_data.max():.6f}]")
        print(f"  PD2 range: [{pd2_data.min():.6f}, {pd2_data.max():.6f}]")
        
        # Statistical analysis
        print(f"\nStatistical Analysis:")
        print(f"  PD1 mean±std: {pd1_data.mean():.6f}±{pd1_data.std():.6f}")
        print(f"  PD2 mean±std: {pd2_data.mean():.6f}±{pd2_data.std():.6f}")
    
    # Check for potential issues (always check, but only print warnings if verbose or critical)
    has_issues = False
    if pd1_data.max() > 1.0 or pd2_data.max() > 1.0:
        has_issues = True
        if verbose:
            print("  WARNING: Data values > 1.0 detected - normalization may be incorrect")
    
    if pd1_data.std() < 0.01 or pd2_data.std() < 0.01:
        has_issues = True
        if verbose:
            print("  WARNING: Very low variance detected - signals may be too similar")
    
    # Check for NaN or infinite values
    pd1_issues = np.isnan(pd1_data).sum() + np.isinf(pd1_data).sum()
    pd2_issues = np.isnan(pd2_data).sum() + np.isinf(pd2_data).sum()
    
    if pd1_issues > 0 or pd2_issues > 0:
        has_issues = True
        if verbose:
            print(f"  WARNING: Found {pd1_issues} NaN/inf in PD1, {pd2_issues} in PD2")
    
    if verbose:
        print("="*70)
    
    # Always print critical data issues even in concise mode
    if not verbose and has_issues:
        print("Data quality issues detected (run without --concise for details)")

def monitor_training_progress(model, X_val, y_val, fold):
    """Monitor model predictions during training to detect stuck behavior."""
    pd1_val, pd2_val = X_val
    predictions = model.predict([pd1_val, pd2_val], verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Check prediction diversity
    unique_preds, pred_counts = np.unique(predicted_classes, return_counts=True)
    
    print(f"\nFold {fold} Prediction Analysis:")
    print(f"  Unique predictions: {len(unique_preds)} out of {len(np.unique(np.argmax(y_val, axis=1)))} classes")
    
    for pred, count in zip(unique_preds, pred_counts):
        percentage = count/len(predicted_classes)*100
        print(f"    Predicted class {pred}: {count} samples ({percentage:.1f}%)")
    
    # Warning if model is stuck
    if len(unique_preds) == 1:
        print("  🚨 CRITICAL: Model is predicting only ONE class!")
        dominant_class = unique_preds[0]
        confidence = np.mean(np.max(predictions, axis=1))
        print(f"    Stuck on class: {dominant_class}")
        print(f"    Average confidence: {confidence:.3f}")

# -------------------------
# 5. Helper functions for training
# -------------------------
def setup_fold_data(pd1_data, pd2_data, y_categorical, y_enc, train_idx, val_idx, n_classes, concise=False):
    """Setup training and validation data for a fold."""
    # Split both signal types
    pd1_train, pd1_val = pd1_data[train_idx], pd1_data[val_idx]
    pd2_train, pd2_val = pd2_data[train_idx], pd2_data[val_idx]
    y_train, y_val = y_categorical[train_idx], y_categorical[val_idx]

    # Enhanced class balancing strategy
    unique_classes = np.unique(y_enc[train_idx])
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_enc[train_idx])
    class_weight_dict = dict(enumerate(class_weights))
    
    if not concise:
        print(f"Class weights: {class_weight_dict}")
        
        # Print class distribution for this fold
        train_unique, train_counts = np.unique(y_enc[train_idx], return_counts=True)
        val_unique, val_counts = np.unique(y_enc[val_idx], return_counts=True)
        
        print(f"Training set distribution:")
        for cls, count in zip(train_unique, train_counts):
            print(f"  Class {cls}: {count} samples ({count/len(train_idx)*100:.1f}%)")
        
        print(f"Validation set distribution:")
        for cls, count in zip(val_unique, val_counts):
            print(f"  Class {cls}: {count} samples ({count/len(val_idx)*100:.1f}%)")

    return (pd1_train, pd2_train), (pd1_val, pd2_val), y_train, y_val, class_weight_dict

def create_callbacks(config, experiment_dir, fold, concise, best_overall_accuracy, k, epochs):
    """Create training callbacks for a fold."""
    early_stopping_patience = config.get('early_stopping_patience', 10)
    lr_reduction_patience = config.get('lr_reduction_patience', 5)
    lr_reduction_factor = config.get('lr_reduction_factor', 0.5)
    
    callback_verbose = 0 if concise else 1
    
    cb = [
        callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=early_stopping_patience, 
            restore_best_weights=True, 
            verbose=callback_verbose,
            min_delta=0.001
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=lr_reduction_factor,
            patience=lr_reduction_patience, 
            min_lr=1e-6,
            verbose=callback_verbose
        ),
        callbacks.ModelCheckpoint(
            str(Path(experiment_dir) / 'models' / f'best_model_fold_{fold}.h5') if experiment_dir else f'best_model_fold_{fold}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=callback_verbose
        )
    ]
    
    # Add concise progress callback if needed
    if concise:
        concise_callback = ConciseProgressCallback(fold, k, epochs)
        concise_callback.set_best_overall(best_overall_accuracy)
        cb.append(concise_callback)
    
    return cb

def save_fold_results(model, X_val, y_val, history, fold, experiment_dir, concise):
    """Save fold results including predictions, history, and plots."""
    pd1_val, pd2_val = X_val
    
    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate([pd1_val, pd2_val], y_val, verbose=0)
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_filename = str(Path(experiment_dir) / 'history' / f'training_history_fold_{fold}.csv') if experiment_dir else f'training_history_fold_{fold}.csv'
    history_df.to_csv(history_filename, index=False, encoding='utf-8')
    if not concise:
        print(f"Training history saved: {history_filename}")
    
    # Generate predictions and save
    val_predictions = model.predict([pd1_val, pd2_val], verbose=0)
    val_pred_classes = np.argmax(val_predictions, axis=1)
    val_true_classes = np.argmax(y_val, axis=1)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_class': val_true_classes,
        'predicted_class': val_pred_classes,
        'confidence': np.max(val_predictions, axis=1)
    })
    pred_filename = str(Path(experiment_dir) / 'predictions' / f'predictions_fold_{fold}.csv') if experiment_dir else f'predictions_fold_{fold}.csv'
    predictions_df.to_csv(pred_filename, index=False, encoding='utf-8')
    if not concise:
        print(f"Predictions saved: {pred_filename}")
    
    # Print metrics if not in concise mode
    if not concise:
        from sklearn.metrics import classification_report, confusion_matrix
        print(f"\nFold {fold} Classification Report:")
        n_classes = len(np.unique(val_true_classes))
        print(classification_report(val_true_classes, val_pred_classes, 
                                   target_names=[f'Class_{i}' for i in range(n_classes)]))
        
        print(f"Fold {fold} Confusion Matrix:")
        print(confusion_matrix(val_true_classes, val_pred_classes))
        print("-" * 80)
    
    # Save plots if experiment_dir is provided
    if experiment_dir:
        _save_fold_plots(val_true_classes, val_pred_classes, history, fold, experiment_dir)
    
    return val_loss, val_accuracy

def _save_fold_plots(val_true_classes, val_pred_classes, history, fold, experiment_dir):
    """Save confusion matrix and training history plots."""
    from sklearn.metrics import confusion_matrix
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    # Save confusion matrix plot
    cm = confusion_matrix(val_true_classes, val_pred_classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    plt.colorbar(im)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    # Labels and title
    ax.set_title(f'Confusion Matrix - Fold {fold}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Conduct', 'Keyhole'])
    ax.set_yticklabels(['Conduct', 'Keyhole'])
    
    cm_filename = str(Path(experiment_dir) / 'plots' / f'confusion_matrix_fold_{fold}.png')
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save training history plot
    if history:
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Model Accuracy - Fold {fold}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss - Fold {fold}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        history_plot_filename = str(Path(experiment_dir) / 'plots' / f'training_history_fold_{fold}.png')
        plt.savefig(history_plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

# -------------------------
# 6. Training with K-fold CV + Augmentation
# -------------------------
def train_kfold(X, y, config, experiment_dir=None, concise=False, progress_info=None):
    import time
    start_time = time.time()
    
    # Extract parameters from config
    k = config.get('k_folds', 5)
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 16)
    early_stopping_patience = config.get('early_stopping_patience', 10)
    lr_reduction_patience = config.get('lr_reduction_patience', 5)
    lr_reduction_factor = config.get('lr_reduction_factor', 0.5)
    use_class_weights = config.get('use_class_weights', True)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    # Convert labels to categorical
    y_categorical = tf.keras.utils.to_categorical(y_enc, n_classes)

    # X is a tuple of (pd1_data, pd2_data)
    pd1_data, pd2_data = X
    
    # Handle different training modes
    if k == 1:
        # Single training run with train/val split (for final model training)
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(pd1_data))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, stratify=y_enc, random_state=42
        )
        fold_splits = [(train_idx, val_idx)]
        if not concise:
            print(f"Single training mode: {len(train_idx)} train, {len(val_idx)} validation samples")
    else:
        # K-fold cross-validation
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_splits = list(skf.split(pd1_data, y_enc))
        if not concise:
            print(f"K-fold cross-validation: {k} folds")
    
    # Track results across folds
    fold_results = {
        'fold': [],
        'val_loss': [],
        'val_accuracy': [],
        'best_epoch': []
    }
    
    # Track best overall accuracy
    best_overall_accuracy = 0.0
    
    # Track augmentation statistics across all folds
    total_augmented_samples_added = 0  # Total NEW samples created by augmentation
    fold_count = 0

    for fold, (train_idx, val_idx) in enumerate(fold_splits, 1):
        fold_start_time = time.time()
        
        if concise:
            if k == 1:
                print(f"\n+-- Final Model Training " + "-" * 56)
            else:
                print(f"\n+-- Fold {fold}/{k} " + "-" * 70)
        else:
            if k == 1:
                print(f"\n--- Final Model Training ---")
            else:
                print(f"\n--- Fold {fold} ---")
        
        # Setup fold data and class weights
        (pd1_train, pd2_train), (pd1_val, pd2_val), y_train, y_val, class_weight_dict = setup_fold_data(
            pd1_data, pd2_data, y_categorical, y_enc, train_idx, val_idx, n_classes, concise
        )
        
        if concise:
            print(f"| Class weights: {class_weight_dict}")

        # Build model
        model = build_dual_branch_model(input_shape=(pd1_data.shape[1], 1), n_classes=n_classes, config=config)

        # Create callbacks
        cb = create_callbacks(config, experiment_dir, fold, concise, best_overall_accuracy, k, epochs)

        # Apply augmentation to training set
        if not concise:
            print("Applying augmentation to training data...")
        
        (pd1_combined, pd2_combined), y_train_enc_combined = augment_batch(
            (pd1_train, pd2_train), y_enc[train_idx], 
            augment_fraction=config.get('augment_fraction', 0.5),
            time_shift_range=config.get('time_shift_range', 5),
            stretch_probability=config.get('stretch_probability', 0.3),
            stretch_scale=config.get('stretch_scale', 0.1),
            noise_probability=config.get('noise_probability', 0.5),
            noise_std=config.get('noise_std', 0.02),
            amplitude_scale_probability=config.get('amplitude_scale_probability', 0.5),
            amplitude_scale=config.get('amplitude_scale', 0.1)
        )
        y_combined = tf.keras.utils.to_categorical(y_train_enc_combined, n_classes)
        
        # Always report augmentation results
        original_size = len(pd1_train)
        final_size = len(pd1_combined)
        augmented_count = final_size - original_size
        print(f"Training fold {fold}: {final_size} samples (original: {original_size}, augmented: +{augmented_count})")
        
        # Accumulate augmentation statistics across folds
        total_augmented_samples_added += augmented_count  # Only count NEW samples
        fold_count += 1
        
        if not concise:
            print(f"\nStarting training for fold {fold}...")
        
        # Train the model
        fit_verbose = 0 if concise else 1
        history = model.fit([pd1_combined, pd2_combined], y_combined,
                           validation_data=([pd1_val, pd2_val], y_val),
                           batch_size=batch_size,
                           epochs=epochs,
                           class_weight=class_weight_dict,
                           callbacks=cb,
                           verbose=fit_verbose,
                           shuffle=True)
        
        # Monitor predictions after training
        if not concise:
            monitor_training_progress(model, (pd1_val, pd2_val), y_val, fold)
        
        # Save fold results and get validation metrics
        val_loss, val_accuracy = save_fold_results(
            model, (pd1_val, pd2_val), y_val, history, fold, experiment_dir, concise
        )
        
        if not concise:
            print(f"Fold {fold} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        # Store results
        fold_results['fold'].append(fold)
        fold_results['val_loss'].append(val_loss)
        fold_results['val_accuracy'].append(val_accuracy)
        fold_results['best_epoch'].append(len(history.history['loss']))  # or use early stopping info
        
        # Calculate fold timing
        fold_duration_minutes = (time.time() - fold_start_time) / 60
        
        # Track accuracy improvements
        is_new_best = val_accuracy > best_overall_accuracy
        old_best_overall = best_overall_accuracy
        
        # Update best accuracy
        if val_accuracy > best_overall_accuracy:
            best_overall_accuracy = val_accuracy
        
        # Print enhanced fold completion in concise mode
        if concise:
            print()  # Add blank line before fold completion
            print("-" * 80)  # Add separator line
            message_parts = [f"Fold {fold}/{k} completed | Val Acc: {val_accuracy:.4f}"]
            
            # Add improvement tag
            if is_new_best:
                if fold == 1:
                    message_parts.append("NEW BEST!")
                else:
                    delta = val_accuracy - old_best_overall
                    message_parts.append(f"NEW BEST! (+{delta:.4f})")
            
            # Add timing
            message_parts.append(f"{fold_duration_minutes:.1f}m")
            
            # Add ETA calculation for hyperparameter optimization
            if progress_info and 'total_configs' in progress_info:
                # Calculate remaining work across entire hyperopt run
                remaining_folds_this_config = k - fold
                remaining_configs = progress_info['total_configs'] - progress_info['current_config']
                total_remaining_folds = remaining_folds_this_config + (remaining_configs * k)
                
                # Use sophisticated time estimation from historical data
                estimated_fold_time = progress_info.get('estimated_fold_time', fold_duration_minutes)
                
                if total_remaining_folds > 0:
                    # Calculate total remaining time for entire hyperopt run
                    total_eta_minutes = total_remaining_folds * estimated_fold_time
                    eta_hours = total_eta_minutes / 60
                    
                    if eta_hours < 1:
                        message_parts.append(f"ETA: {total_eta_minutes:.0f}m")
                    else:
                        message_parts.append(f"ETA: {eta_hours:.1f}h")
            
            print(" | ".join(message_parts))
            print("-" * 80)  # Add separator line after fold completion
    
    # Calculate training time
    training_time_minutes = (time.time() - start_time) / 60
    
    # Save overall results summary
    results_df = pd.DataFrame(fold_results)
    summary_filename = str(Path(experiment_dir) / 'kfold_results_summary.csv') if experiment_dir else 'kfold_results_summary.csv'
    results_df.to_csv(summary_filename, index=False, encoding='utf-8')
    
    # Print summary statistics
    print("\n" + "="*80)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("="*80)
    mean_acc = np.mean(fold_results['val_accuracy'])
    std_acc = np.std(fold_results['val_accuracy'])
    mean_loss = np.mean(fold_results['val_loss'])
    std_loss = np.std(fold_results['val_loss'])
    best_acc = np.max(fold_results['val_accuracy'])
    best_fold = fold_results['fold'][np.argmax(fold_results['val_accuracy'])]
    
    # Calculate total augmented samples across all folds
    # Note: Each fold used ~50% augmentation, so total augmented ≈ original_training_samples * 0.5 * k_folds
    pd1_data, pd2_data = X
    original_total = len(pd1_data)
    training_samples_per_fold = int(original_total * (1 - (0.2 if k == 1 else 0.2)))  # Accounting for validation split
    estimated_augmented_per_fold = int(training_samples_per_fold * 0.5)  # 50% augmentation fraction
    total_training_samples_used = (training_samples_per_fold + estimated_augmented_per_fold) * k if k > 1 else (training_samples_per_fold + estimated_augmented_per_fold)
    
    print(f"DATASET SUMMARY:")
    print(f"  Original dataset: {original_total} samples")
    print(f"  Training per fold: ~{training_samples_per_fold} samples")
    print(f"  Augmented per fold: ~{estimated_augmented_per_fold} samples")
    print(f"  Total training samples used: ~{total_training_samples_used} samples")
    print(f"")
    print(f"PERFORMANCE SUMMARY:")
    print(f"  Mean Validation Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Mean Validation Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"  Best Fold Accuracy: {best_acc:.4f} (Fold {best_fold})") 
    print(f"  Training Time: {training_time_minutes:.1f} minutes")
    print(f"  Results saved to: {summary_filename}")
    print("="*80)
    
    # Calculate meaningful augmentation metrics
    avg_augmented_samples_per_fold = total_augmented_samples_added / fold_count if fold_count > 0 else 0.0
    # This represents: "On average, how many NEW samples were created per fold"
    
    # Calculate augmentation ratio (meaningful metric for comparison)
    # Assumes 80/20 train/val split - could be made configurable if needed
    training_samples_per_fold = len(pd1_data) * 0.8  # Approximate training size per fold
    augmentation_ratio = avg_augmented_samples_per_fold / training_samples_per_fold if training_samples_per_fold > 0 else 0.0
    
    # Prepare results for experiment logging
    experiment_results = {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_loss': mean_loss,
        'std_loss': std_loss,
        'best_accuracy': best_acc,
        'best_fold': best_fold,
        'fold_accuracies': fold_results['val_accuracy'],
        'fold_losses': fold_results['val_loss'],
        'training_time_minutes': training_time_minutes,
        
        # Augmentation statistics (relative to total_samples already in log)
        'augmented_samples_per_fold': avg_augmented_samples_per_fold,
        'total_augmented_samples_created': total_augmented_samples_added,
        'augmentation_ratio': augmentation_ratio
    }
    
    # Extract hyperparameters from config for logging
    hyperparams = extract_hyperparameters(config)
    experiment_results['hyperparams'] = hyperparams
    
    return results_df, experiment_results

# -------------------------
# 5. Run everything
# -------------------------
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PD Signal Classifier with configurable hyperparameters')
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--k_folds', type=int, help='Number of K-fold splits')
    parser.add_argument('--data_dir', type=str, help='Path to data directory')
    parser.add_argument('--output_root', type=str, help='Root output directory')
    parser.add_argument('--concise', action='store_true', help='Show concise progress updates on one line')
    parser.add_argument('--verbose', action='store_true', help='Show detailed diagnostic information')
    parser.add_argument('--source', type=str, default='manual', help='Source of the run (manual or hyperopt)')
    parser.add_argument('--current_config', type=int, default=1, help='Current configuration number (for progress tracking)')
    parser.add_argument('--total_configs', type=int, default=1, help='Total number of configurations (for progress tracking)')
    parser.add_argument('--estimated_fold_time', type=float, default=0, help='Sophisticated time estimate per fold (for ETA calculation)')
    
    # Enhanced traceability arguments (new)
    parser.add_argument('--hyperopt_run_id', type=str, help='Hyperopt run identifier for traceability')
    parser.add_argument('--config_file', type=str, help='Path to config file for traceability')
    parser.add_argument('--config_number_in_run', type=int, help='Config number within hyperopt run for traceability')
    
    # Final model training with test holdout
    parser.add_argument('--exclude_files', type=str, help='Path to file containing list of files to exclude from training')
    
    return parser.parse_args()

def main():
    """Main execution function that handles both manual and automated modes."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set TensorFlow logging level based on mode
    if args.verbose:
        # Keep all TensorFlow messages in verbose mode
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        tf.get_logger().setLevel('INFO')
    elif args.concise:
        # Suppress TensorFlow warnings and info messages in concise mode
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('WARNING')
    else:
        # Normal mode - show warnings but not info
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.get_logger().setLevel('WARNING')
    
    # Load configuration using consolidated function
    config = load_config(
        config_path=args.config,
        classifier_type='pd_signal',
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        k_folds=args.k_folds,
        data_dir=args.data_dir,
        output_root=args.output_root
    )
    
    if args.config:
        print(f"\nLoaded configuration from: {args.config}")
    
    # Extract key parameters
    data_dir = config['data_dir']
    img_width = config['img_width']
    output_root = config['output_root']
    
    # Setup experiment tracking
    version = get_next_version_number(output_root)
    experiment_dir = setup_experiment_directory(version, output_root)
    
    if not args.concise:
        print(f"\nStarting Experiment Version {version}")
        print(f"Output directory: {experiment_dir}")
        print("="*80)
        
        # Print configuration being used
        print("Configuration:")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Epochs: {config['epochs']}")
        print(f"  K-Folds: {config['k_folds']}")
        print(f"  Conv Filters: {config['conv_filters']}")
        print(f"  Dense Units: {config['dense_units']}")
        print(f"  Conv Dropout: {config['conv_dropout']}")
        print(f"  Dense Dropout: {config['dense_dropout']}")
        print(f"  L2 Regularization: {config['l2_regularization']}")
        print("="*80)
    
    # Load data with diagnostics
    if not args.concise:
        print("Loading data...")
    verbose_mode = args.verbose
    X, y = load_data(data_dir, img_width, verbose=verbose_mode, exclude_files=args.exclude_files)
    
    # Report original dataset size
    print(f"Original dataset loaded: {len(y)} samples")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Run comprehensive diagnostics with appropriate verbosity
    class_info = analyze_class_distribution(y, verbose=args.verbose)
    analyze_data_quality(X, y, verbose=args.verbose)
    
    # Prepare data info for logging
    pd1_data, pd2_data = X
    data_info = {
        'data_dir': data_dir,
        'total_samples': len(y),
        'num_classes': len(np.unique(y)),
        'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
        'img_width': img_width,
    }
    
    # Train with enhanced monitoring and experiment tracking
    if not args.concise:
        print(f"\nTraining Model Version {version}...")
    # Prepare progress info for ETA calculation
    progress_info = None
    if args.source == 'hyperopt':
        progress_info = {
            'current_config': args.current_config,
            'total_configs': args.total_configs,
            'estimated_fold_time': args.estimated_fold_time
        }
    
    results_df, experiment_results = train_kfold(X, y, config, experiment_dir=experiment_dir, concise=args.concise, progress_info=progress_info)
    
    # Log experiment to CSV
    if 'hyperparams' in experiment_results:
        log_file = create_experiment_log_entry(
            version=version,
            hyperparams=experiment_results['hyperparams'],
            data_info=data_info,
            results=experiment_results,
            output_root=output_root,
            source=args.source,
            hyperopt_run_id=args.hyperopt_run_id,
            config_file=args.config_file,
            config_number_in_run=args.config_number_in_run
        )
        print(f"\nExperiment logged to: {log_file}")
    
    # Save detailed experiment summary to logs folder
    if experiment_dir:
        experiment_summary = {
            'experiment_version': format_version(version),
            'timestamp': datetime.datetime.now().isoformat(),
            'configuration': config,
            'data_info': data_info,
            'results': experiment_results,
            'source': args.source if hasattr(args, 'source') else 'manual'
        }
        
        summary_filename = str(Path(experiment_dir) / 'logs' / f'experiment_summary_{format_version(version)}.json')
        with open(summary_filename, 'w') as f:
            json.dump(experiment_summary, f, indent=2, default=str)
        
        # Also save a human-readable summary
        readable_summary = f"""Experiment Version {format_version(version)} Summary
=====================================
Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source: {args.source if hasattr(args, 'source') else 'manual'}

Configuration:
- Learning Rate: {config['learning_rate']}
- Batch Size: {config['batch_size']}
- Epochs: {config['epochs']}
- K-Folds: {config['k_folds']}
- Conv Filters: {config['conv_filters']}
- Dense Units: {config['dense_units']}
- Conv Dropout: {config['conv_dropout']}
- Dense Dropout: {config['dense_dropout']}
- L2 Regularization: {config['l2_regularization']}

Data Information:
- Total Samples: {data_info['total_samples']}
- Number of Classes: {data_info['num_classes']}
- Class Distribution: {data_info['class_distribution']}

Results:
- Mean Validation Accuracy: {experiment_results['mean_accuracy']:.4f} ± {experiment_results['std_accuracy']:.4f}
- Best Fold Accuracy: {experiment_results['best_accuracy']:.4f} (Fold {experiment_results['best_fold']})
- Mean Validation Loss: {experiment_results['mean_loss']:.4f} ± {experiment_results['std_loss']:.4f}
- Training Time: {experiment_results.get('training_time_minutes', 0):.1f} minutes

Per-Fold Results:
{chr(10).join([f"Fold {i+1}: Accuracy={acc:.4f}, Loss={loss:.4f}" for i, (acc, loss) in enumerate(zip(experiment_results['fold_accuracies'], experiment_results['fold_losses']))])}
"""
        
        readable_summary_filename = str(Path(experiment_dir) / 'logs' / f'experiment_summary_{format_version(version)}.txt')
        with open(readable_summary_filename, 'w') as f:
            f.write(readable_summary)
    
    print(f"\nExperiment Version {version} Complete!")
    print(f"All outputs saved to: {experiment_dir}")
    print(f"Check experiment_log.csv for hyperparameter tracking")
    print("="*80)
    
    return experiment_results

if __name__ == "__main__":
    main()
