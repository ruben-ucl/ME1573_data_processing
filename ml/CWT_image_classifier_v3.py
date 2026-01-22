#!/usr/bin/env python3
"""
CWT Image Binary Classifier v3

A configurable CNN-based binary classifier for CWT image data with comprehensive
experiment logging, k-fold cross-validation, and Grad-CAM analysis.

Based on the architecture of PD_signal_classifier_v3.py but specialized for
CWT image classification tasks.

Author: AI Assistant (adapted from Dr Wei Li and Rubén Lambert-Garcia's original)
"""

# Ensure UTF-8 encoding for all I/O operations
import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import argparse
import datetime
import json
import random
import sys
import warnings
from glob import glob
from pathlib import Path
from scipy.stats import describe
from matplotlib import pyplot as plt

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

from config import (
    get_default_cwt_data_dir, get_cwt_experiment_log_path, get_cwt_config_template, 
    format_version, get_next_version_from_log, CWT_OUTPUTS_DIR, CWT_LOGS_DIR, ensure_cwt_directories,
    normalize_path, ensure_path_exists, convert_numpy_types,
    log_experiment_results, create_experiment_summary_files, save_fold_plots
)

from data_utils import normalize_image

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -------------------------
# Concise Progress Callback
# -------------------------
class ConciseProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback for concise progress updates during CWT image training."""
    
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
# 1. Load CWT images + labels
# -------------------------
def load_cwt_image_data(root_dirs, img_size, verbose=False, exclude_files=None):
    """
    Load CWT image data from single or multiple directory structures for multi-channel support.
    
    Expected structure (per directory):
    root_dir/
    ├── 0/  (class 0 images)
    │   ├── image1.png
    │   └── image2.png
    └── 1/  (class 1 images)
        ├── image3.png
        └── image4.png
    
    Args:
        root_dirs: str or list of str - Single directory or list of directories for multi-channel
        img_size: Tuple (width, height) for image resizing
        verbose: Print progress information
        exclude_files: Set of filenames to exclude
        
    Returns:
        tuple: (images_array, labels_array, class_counts, label_encoder)
        
    Images array shape: (N, H, W, C) where C = len(root_dirs)
    """
    # Handle backward compatibility: single string → list
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]
    
    num_channels = len(root_dirs)
    exclude_files = exclude_files or set()
    
    if verbose:
        print(f"Loading {num_channels}-channel CWT image data from {len(root_dirs)} directories")
    
    # Import validation from config to avoid duplication
    from config import validate_multichannel_structure
    
    # Validate all directories exist and have same structure
    validate_multichannel_structure(root_dirs, verbose)
    
    # Use first directory to discover images and classes
    reference_dir = Path(root_dirs[0])
    image_registry = build_image_registry(reference_dir, exclude_files)
    
    if verbose:
        total_images = sum(len(files) for files in image_registry.values())
        print(f"Found {len(image_registry)} classes with {total_images} total images")
    
    # Load images from all channels
    images, labels = [], []
    
    for class_label, image_files in image_registry.items():
        class_count = 0
        
        for img_filename in image_files:
            try:
                # Load corresponding image from each channel
                multichannel_image = load_multichannel_image(
                    root_dirs, class_label, img_filename, img_size
                )
                images.append(multichannel_image)
                labels.append(class_label)
                class_count += 1
                
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to load {img_filename} from all channels: {e}")
                continue
        
        if verbose:
            print(f"Class {class_label}: {class_count} images loaded")
    
    if not images:
        raise ValueError(f"No valid images found across {len(root_dirs)} directories")
    
    # Convert to arrays
    images_array = np.array(images)  # Shape: (N, H, W, C)
    labels_array = np.array(labels)
    
    # Create label encoder for consistency
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels_array)
    
    # Count samples per class
    unique_labels, counts = np.unique(labels_encoded, return_counts=True)
    class_counts = dict(zip(unique_labels, counts))
    
    if verbose:
        print(f"Final dataset: {images_array.shape}")
        print(f"Class distribution: {class_counts}")
        print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    return images_array, labels_encoded, class_counts, label_encoder

def build_image_registry(reference_dir, exclude_files):
    """
    Build registry of {class_label: [image_filenames]} from reference directory.
    
    Args:
        reference_dir: Path to reference directory
        exclude_files: Set of filenames to exclude
        
    Returns:
        dict: Dictionary mapping class labels to lists of image filenames
    """
    image_registry = {}
    
    # Find all class directories (should be numeric: 0, 1, etc.)
    class_dirs = [d for d in reference_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    class_dirs.sort(key=lambda x: int(x.name))  # Sort numerically
    
    if not class_dirs:
        raise ValueError(f"No class directories found in {reference_dir}")
    
    for class_dir in class_dirs:
        class_label = int(class_dir.name)
        
        # Find all image files in this class directory
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
            image_files.extend(class_dir.glob(ext))
        
        # Filter out excluded files and get just the filenames
        valid_filenames = []
        for img_path in sorted(image_files):
            if img_path.name not in exclude_files:
                valid_filenames.append(img_path.name)
        
        image_registry[class_label] = valid_filenames
    
    return image_registry

def load_multichannel_image(root_dirs, class_label, img_filename, img_size):
    """
    Load the same image file from multiple channel directories and stack.
    
    Args:
        root_dirs: List of root directory paths
        class_label: Class directory name (e.g., 0, 1)
        img_filename: Image filename (e.g., "image001.png")
        img_size: Target size (width, height)
        
    Returns:
        numpy.ndarray: Stacked multichannel image, shape (H, W, C)
    """
    channels = []
    
    for root_dir in root_dirs:
        img_path = Path(root_dir) / str(class_label) / img_filename
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found in channel: {img_path}")
            
        # Load single channel using existing function
        channel_image = load_and_preprocess_image(str(img_path), img_size)
        
        # Remove channel dimension to get (H, W)
        if len(channel_image.shape) == 3:
            channel_image = channel_image[:, :, 0]
            
        channels.append(channel_image)
    
    # Stack channels: (H, W, C)
    multichannel_image = np.stack(channels, axis=-1)
    return multichannel_image

def load_and_preprocess_image(img_path, img_size):
    """
    Load and preprocess a single CWT image.
    
    Args:
        img_path: Path to image file
        img_size: Tuple (width, height) for resizing
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    # Read image using OpenCV (handles various formats)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    # Convert to PIL for consistent resizing
    image = Image.fromarray(image)
    image = image.resize(img_size, Image.Resampling.LANCZOS)
    
    # Convert back to numpy array and add channel dimension
    image = np.array(image)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
    
    return image

def load_cwt_image_data_from_csv(channel_paths, label_file, img_size, verbose=False, exclude_files=None):
    """
    Load CWT image data from flat directory structure with CSV labels.

    This function supports dataset variants where:
    - All images are in a single directory (flat structure)
    - Labels are provided in a CSV file with columns: image_filename, has_porosity
    - Multiple channels can be loaded from different directories

    Args:
        channel_paths: list of str - Directories containing images (one per channel)
        label_file: str or Path - Path to CSV file with image_filename and has_porosity columns
        img_size: Tuple (width, height) for image resizing
        verbose: Print progress information
        exclude_files: Set of filenames to exclude

    Returns:
        tuple: (images_array, labels_array, class_counts, label_encoder)

    Images array shape: (N, H, W, C) where C = len(channel_paths)
    """
    import pandas as pd

    exclude_files = exclude_files or set()
    num_channels = len(channel_paths)

    if verbose:
        print(f"Loading {num_channels}-channel CWT image data from CSV labels")
        print(f"Root directories: {channel_paths}")

    # Load label CSV
    label_df = pd.read_csv(label_file, encoding='utf-8')
    if verbose:
        print(f"Loaded {len(label_df)} labels from {label_file}")

    # Support both 'filename' and 'image_filename' column names
    filename_col = 'filename' if 'filename' in label_df.columns else 'image_filename'

    # Filter out excluded files
    if exclude_files:
        label_df = label_df[~label_df[filename_col].isin(exclude_files)]
        if verbose:
            print(f"After exclusion: {len(label_df)} images remain")

    # Load images
    images, labels = [], []
    failed_count = 0

    for idx, row in label_df.iterrows():
        img_filename = row[filename_col]
        label = int(row['has_porosity'])

        try:
            # Load corresponding image from each channel
            channels = []
            for channel_path in channel_paths:
                img_path = Path(channel_path) / img_filename

                if not img_path.exists():
                    raise FileNotFoundError(f"Image not found: {img_path}")

                # Load and preprocess single channel
                channel_image = load_and_preprocess_image(str(img_path), img_size)

                # Remove channel dimension to get (H, W)
                if len(channel_image.shape) == 3:
                    channel_image = channel_image[:, :, 0]

                channels.append(channel_image)

            # Stack channels: (H, W, C)
            multichannel_image = np.stack(channels, axis=-1)
            images.append(multichannel_image)
            labels.append(label)

        except Exception as e:
            failed_count += 1
            if verbose:
                print(f"Warning: Failed to load {img_filename}: {e}")
            continue

    if verbose and failed_count > 0:
        print(f"Failed to load {failed_count} images")

    if not images:
        # Provide detailed debug info when no images are loaded
        print(f"\nDEBUG: Failed to load any images!")
        print(f"  Root directories checked: {channel_paths}")
        if len(label_df) > 0:
            sample_file = label_df.iloc[0][filename_col]
            print(f"  Sample filename from CSV: {sample_file}")
            if len(channel_paths) > 0:
                sample_path = Path(channel_paths[0]) / sample_file
                print(f"  Sample constructed path: {sample_path}")
                print(f"  Sample path exists: {sample_path.exists()}")
                if not sample_path.exists():
                    parent_dir = Path(channel_paths[0])
                    if parent_dir.exists():
                        print(f"  Parent directory exists: True")
                        files_in_parent = list(parent_dir.iterdir())[:5]
                        print(f"  First 5 files in parent: {[f.name for f in files_in_parent]}")
                    else:
                        print(f"  Parent directory exists: False")
        raise ValueError(f"No valid images could be loaded from {len(label_df)} labeled files")

    # Convert to arrays
    images_array = np.array(images)  # Shape: (N, H, W, C)
    labels_array = np.array(labels)

    # Create label encoder for consistency
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels_array)

    # Count samples per class
    unique_labels, counts = np.unique(labels_encoded, return_counts=True)
    class_counts = dict(zip(unique_labels, counts))

    if verbose:
        print(f"Final dataset: {images_array.shape}")
        print(f"Class distribution: {class_counts}")
        print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    return images_array, labels_encoded, class_counts, label_encoder

# -------------------------
# 2. Data Augmentation
# -------------------------
def apply_cwt_augmentation(X, config):
    """
    Apply CWT-suitable augmentation to image data.
    
    Args:
        X: Input images (N, H, W, C)
        config: Configuration dictionary with augmentation parameters
        
    Returns:
        Augmented images
    """
    X_aug = X.copy()
    
    # Apply time shift (width shift)
    if random.random() < config.get('time_shift_probability', 0.0):
        width_shift = config.get('time_shift_range', 0)
        if width_shift > 0:
            shift_pixels = int(X.shape[2] * width_shift)  # Width dimension
            if shift_pixels > 0:
                # Random shift for each image
                for i in range(X_aug.shape[0]):
                    shift = np.random.randint(-shift_pixels, shift_pixels + 1)
                    if shift != 0:
                        X_aug[i] = np.roll(X_aug[i], shift, axis=1)  # Shift along width (time)
    
    # Apply additive noise
    if random.random() < config.get('noise_probability', 0.0):
        noise_std = config.get('noise_std', 0.0)
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, X_aug.shape)
            X_aug = X_aug + noise
            X_aug = np.clip(X_aug, 0, 1)  # Keep in valid range
    
    # Apply brightness variation (amplitude scaling)
    if random.random() < config.get('brighness_probability', 0.0):
        brightness_range = config.get('brightness_range', 0.0)
        if brightness_range > 0:
            for i in range(X_aug.shape[0]):
                # Random brightness factor
                brightness_factor = 1.0 + np.random.uniform(-brightness_range, brightness_range)
                X_aug[i] = X_aug[i] * brightness_factor
                X_aug[i] = np.clip(X_aug[i], 0, 1)
    
    # Apply contrast variation
    if random.random() < config.get('contrast_probability', 0.0):
        contrast_range = config.get('contrast_range', 0.0)
        if contrast_range > 0:
            for i in range(X_aug.shape[0]):
                # Random contrast factor
                contrast_factor = 1.0 + np.random.uniform(-contrast_range, contrast_range)
                mean_val = np.mean(X_aug[i])
                X_aug[i] = mean_val + (X_aug[i] - mean_val) * contrast_factor
                X_aug[i] = np.clip(X_aug[i], 0, 1)
    
    return X_aug

# -------------------------
# 3. Model Architecture
# -------------------------
def create_cnn_model(input_shape, config, verbose=False):
    """
    Create CNN model for CWT image classification.
    
    Args:
        input_shape: Tuple (height, width, channels) for input images
        config: Configuration dictionary
        verbose: Print model summary
        
    Returns:
        tf.keras.Model: Compiled CNN model
    """
    model = models.Sequential()
    
    # Extract configuration
    conv_filters = config['conv_filters']
    kernel_size = tuple(config['conv_kernel_size'])
    pool_size = tuple(config['pool_size'])
    pool_layers = config.get('pool_layers', [])
    conv_dropout = config['conv_dropout']
    dense_units = config['dense_units']
    dense_dropout = config['dense_dropout']
    l2_reg = config['l2_regularization']
    use_batch_norm = config.get('use_batch_norm', True)
    
    # Validate pool_layers don't exceed conv_filters length
    num_conv_layers = len(conv_filters)
    valid_pool_layers = [i for i in pool_layers if i < num_conv_layers]
    if len(valid_pool_layers) != len(pool_layers) and verbose:
        invalid_layers = [i for i in pool_layers if i >= num_conv_layers]
        print(f"Warning: Removed invalid pool_layers {invalid_layers} for {num_conv_layers}-layer architecture")
    pool_layers = valid_pool_layers
    
    # Add convolutional layers
    for i, filters in enumerate(conv_filters):
        if i == 0:
            # First layer needs input shape
            model.add(layers.Conv2D(
                filters, kernel_size, activation='relu', 
                input_shape=input_shape,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
                name=f'conv2d_{i}'
            ))
        else:
            model.add(layers.Conv2D(
                filters, kernel_size, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
                name=f'conv2d_{i}'
            ))
        
        # Add batch normalization if requested
        if use_batch_norm:
            model.add(layers.BatchNormalization(name=f'batch_norm_{i}'))
        
        # Add dropout if specified
        if conv_dropout > 0:
            model.add(layers.Dropout(conv_dropout, name=f'conv_dropout_{i}'))
        
        # Add pooling after specified layers
        if i in pool_layers:
            model.add(layers.MaxPooling2D(pool_size, name=f'max_pool_{i}'))
    
    # Flatten before dense layers
    model.add(layers.Flatten(name='flatten'))
    
    # Add dense layers
    for i, units in enumerate(dense_units):
        model.add(layers.Dense(
            units, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
            name=f'dense_{i}'
        ))
        
        if use_batch_norm:
            model.add(layers.BatchNormalization(name=f'dense_batch_norm_{i}'))
        
        # Apply dropout (can be list or single value)
        if isinstance(dense_dropout, list):
            dropout_rate = dense_dropout[i] if i < len(dense_dropout) else dense_dropout[-1]
        else:
            dropout_rate = dense_dropout
        
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name=f'dense_dropout_{i}'))
    
    # Output layer for binary classification
    model.add(layers.Dense(1, activation='sigmoid', name='output'))
    
    # Compile model
    optimizer_name = config.get('optimizer', 'adam').lower()
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'])
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=config['learning_rate'])
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Calculate model complexity using Keras built-in method
    total_parameters = model.count_params()
    
    if verbose:
        print("\n" + "="*50)
        print("MODEL ARCHITECTURE")
        print("="*50)
        model.summary()
        print(f"Total Parameters: {total_parameters:,}")
        print("="*50)
    
    # Store complexity in the model for later retrieval
    model._model_complexity = total_parameters
    
    return model

# -------------------------
# 4. Training Functions
# -------------------------
def train_fold(fold, train_idx, val_idx, X, y, config, class_weights=None, best_overall_acc=0.0, concise=False, output_dir=None):
    """
    Train model on a single fold of the data.
    
    Args:
        fold: Current fold number (1-indexed)
        train_idx: Training indices for this fold
        val_idx: Validation indices for this fold
        X: Feature data (images)
        y: Target data (labels)
        config: Configuration dictionary
        class_weights: Class weight dictionary
        best_overall_acc: Best accuracy from previous folds
        concise: Use concise progress reporting
        
    Returns:
        dict: Training results for this fold
    """
    # Split data for this fold
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Normalize data
    X_train = normalize_image(X_train)
    X_val = normalize_image(X_val)
    
    # Create model
    input_shape = X_train.shape[1:]  # (height, width, channels)
    model = create_cnn_model(input_shape, config, verbose=(fold == 1 and not concise))
    
    # Setup callbacks
    callback_list = []
    
    # Progress callback
    if concise:
        progress_callback = ConciseProgressCallback(fold, config['k_folds'], config['epochs'])
        progress_callback.set_best_overall(best_overall_acc)
        callback_list.append(progress_callback)
    
    # Early stopping
    if config['early_stopping_patience'] > 0:
        callback_list.append(callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=0
        ))
    
    # Learning rate reduction
    if config['lr_reduction_patience'] > 0:
        callback_list.append(callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=config['lr_reduction_factor'],
            patience=config['lr_reduction_patience'],
            verbose=0,
            min_lr=1e-7
        ))
    
    # Model checkpoint to save best model per fold (like PD training does)
    if output_dir is not None:
        models_dir = output_dir / 'models'
        ensure_path_exists(models_dir)
        callback_list.append(callbacks.ModelCheckpoint(
            str(models_dir / f'best_model_fold_{fold}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        ))
    
    # Apply CWT-suitable augmentation if specified
    if config['augment_fraction'] > 0:
        # Determine how much data to augment
        num_original = len(X_train)
        num_augmented = int(num_original * config['augment_fraction'])
        
        if num_augmented > 0:
            # Select random subset for augmentation
            aug_indices = np.random.choice(num_original, num_augmented, replace=True)
            X_train_subset = X_train[aug_indices]
            y_train_subset = y_train[aug_indices]
            
            # Apply custom CWT augmentation
            X_train_aug = apply_cwt_augmentation(X_train_subset, config)
            
            # Combine original and augmented data
            X_train_combined = np.concatenate([X_train, X_train_aug], axis=0)
            y_train_combined = np.concatenate([y_train, y_train_subset], axis=0)
            
            # Shuffle combined dataset
            shuffle_indices = np.random.permutation(len(X_train_combined))
            X_train_final = X_train_combined[shuffle_indices]
            y_train_final = y_train_combined[shuffle_indices]
            
            if not concise:
                print(f"   Training with augmentation: {len(X_train_final)} samples ({num_original} original + {num_augmented} augmented)")
        else:
            X_train_final = X_train
            y_train_final = y_train
    else:
        X_train_final = X_train
        y_train_final = y_train
    
    # Train model with possibly augmented data
    history = model.fit(
        X_train_final, y_train_final,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callback_list,
        verbose=0 if concise else 1
    )
    
    # Get final metrics
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    
    # Calculate additional metrics
    y_pred_proba = model.predict(X_val, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Ensure y_val is also flattened and in correct format
    y_val_flat = y_val.flatten() if hasattr(y_val, 'flatten') else y_val
    
    precision = metrics.precision_score(y_val_flat, y_pred, average='binary', zero_division=0)
    recall = metrics.recall_score(y_val_flat, y_pred, average='binary', zero_division=0)
    f1 = metrics.f1_score(y_val_flat, y_pred, average='binary', zero_division=0)
    
    # Get best epoch from history
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    
    return {
        'model': model,
        'history': history,
        'fold': fold,
        'train_accuracy': train_accuracy,
        'train_loss': train_loss,
        'val_accuracy': val_accuracy,
        'val_loss': val_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'best_epoch': best_epoch,
        'total_epochs': len(history.history['loss']),
        'model_complexity': getattr(model, '_model_complexity', 0),  # Get the stored complexity
        'X_val': X_val,
        'y_val': y_val,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'y_val_flat': y_val_flat
    }

# -------------------------
# 5. Evaluation and Analysis
# -------------------------
def get_gradcam_layer_for_logging(config):
    """Get gradcam layer name for logging purposes."""
    gradcam_layer = config.get('gradcam_layer', 'auto')
    if gradcam_layer == 'auto':
        return 'auto'  # Will be resolved during actual analysis
    return gradcam_layer

def run_gradcam_analysis(model, X_sample, y_sample, config, output_dir, concise=False):
    """
    Run Grad-CAM analysis on sample images and save heatmap visualizations.
    
    Args:
        model: Trained CNN model
        X_sample: Sample images for analysis
        y_sample: Sample labels
        config: Configuration dictionary
        output_dir: Directory to save results
    """
    if not config.get('run_gradcam', False):
        return None
    
    if not concise:
        print("\nRunning Grad-CAM analysis...")
    
    # Get configuration parameters
    save_images = config.get('save_gradcam_images', True)  # Default to True
    gradcam_threshold = config.get('gradcam_threshold', 0.5)
    
    # Auto-detect last convolutional layer if needed
    gradcam_layer = config.get('gradcam_layer', 'auto')
    gradcam_layer_logged = gradcam_layer  # Keep original for logging
    if gradcam_layer == 'auto':
        # Find last convolutional layer
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                gradcam_layer = layer.name
                gradcam_layer_logged = f'auto({layer.name})'
                break
        
        if gradcam_layer == 'auto':
            if not concise:
                print("Warning: No convolutional layers found for Grad-CAM")
            return None
    
    if not concise:
        print(f"Using layer '{gradcam_layer}' for Grad-CAM analysis")
        if save_images:
            print(f"Saving Grad-CAM images with threshold {gradcam_threshold}")
    
    # Create grad-CAM model
    try:
        grad_model = tf.keras.Model(
            model.inputs,
            [model.get_layer(gradcam_layer).output, model.output]
        )
    except ValueError as e:
        print(f"Warning: Could not create Grad-CAM model: {e}")
        return None
    
    # Create gradcam subdirectory if saving images
    if save_images:
        gradcam_dir = Path(output_dir) / 'gradcam_images'
        gradcam_dir.mkdir(exist_ok=True)
    
    # Analyze a few sample images from each class
    gradcam_results = {}
    unique_labels = np.unique(y_sample)
    
    for label in unique_labels:
        label_indices = np.where(y_sample == label)[0]
        sample_indices = label_indices[:min(5, len(label_indices))]  # Max 5 samples per class
        
        heatmaps = []
        saved_images = []
        
        for i, idx in enumerate(sample_indices):
            # Get input image
            original_img = X_sample[idx]  # Original for visualization
            img_array = np.expand_dims(X_sample[idx], axis=0)
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                predicted_class = tf.argmax(predictions[0])
                class_output = predictions[:, predicted_class]
            
            # Compute gradients of predicted class wrt conv layer
            grads = tape.gradient(class_output, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight conv outputs with pooled gradients
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            heatmap_np = heatmap.numpy()
            heatmaps.append(heatmap_np)
            
            # Save individual Grad-CAM images if requested
            if save_images:
                try:
                    # Resize heatmap to match original image size
                    import cv2
                    heatmap_resized = cv2.resize(heatmap_np, (original_img.shape[1], original_img.shape[0]))
                    
                    # Create colormap version (0-255 range)
                    heatmap_colored = np.uint8(255 * heatmap_resized)
                    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
                    
                    # Prepare original image for overlay (convert to 3-channel if needed)
                    if len(original_img.shape) == 2:
                        original_3ch = cv2.cvtColor((original_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    else:
                        original_3ch = (original_img * 255).astype(np.uint8)
                        if original_3ch.shape[2] == 1:
                            original_3ch = cv2.cvtColor(original_3ch.squeeze(), cv2.COLOR_GRAY2BGR)
                    
                    # Create overlay with heatmap
                    overlay = cv2.addWeighted(original_3ch, 0.7, heatmap_colored, 0.3, 0)
                    
                    # Get prediction confidence
                    pred_confidence = float(tf.nn.softmax(predictions[0])[predicted_class])
                    actual_label = int(y_sample[idx])
                    predicted_label = int(predicted_class)
                    
                    # Save images only if confidence is above threshold
                    if pred_confidence >= gradcam_threshold:
                        # Save original image
                        orig_filename = f'class_{label}_sample_{i}_original.png'
                        cv2.imwrite(str(gradcam_dir / orig_filename), original_3ch)
                        
                        # Save heatmap
                        heatmap_filename = f'class_{label}_sample_{i}_heatmap.png'
                        cv2.imwrite(str(gradcam_dir / heatmap_filename), heatmap_colored)
                        
                        # Save overlay
                        overlay_filename = f'class_{label}_sample_{i}_overlay.png'
                        cv2.imwrite(str(gradcam_dir / overlay_filename), overlay)
                        
                        saved_images.append({
                            'sample_idx': int(idx),
                            'actual_label': actual_label,
                            'predicted_label': predicted_label,
                            'confidence': float(pred_confidence),
                            'files': {
                                'original': orig_filename,
                                'heatmap': heatmap_filename,
                                'overlay': overlay_filename
                            }
                        })
                    
                except Exception as e:
                    print(f"Warning: Failed to save Grad-CAM image for class {label}, sample {i}: {e}")
        
        if heatmaps:
            gradcam_results[f'class_{label}'] = {
                'mean_heatmap': np.mean(heatmaps, axis=0).tolist(),  # Convert to list for JSON serialization
                'sample_count': len(heatmaps),
                'saved_images': saved_images if save_images else []
            }
    
    # Save results summary with enhanced information
    gradcam_summary_path = Path(output_dir) / 'gradcam_summary.json'
    
    # Calculate summary statistics
    total_images_analyzed = sum(v['sample_count'] for v in gradcam_results.values())
    total_images_saved = sum(len(v['saved_images']) for v in gradcam_results.values()) if save_images else 0
    
    summary_data = {
        'gradcam_layer': gradcam_layer,
        'save_images_enabled': save_images,
        'confidence_threshold': gradcam_threshold,
        'classes_analyzed': list(gradcam_results.keys()),
        'total_images_analyzed': total_images_analyzed,
        'total_images_saved': total_images_saved,
        'samples_per_class': {k: v['sample_count'] for k, v in gradcam_results.items()},
        'saved_images_per_class': {k: len(v['saved_images']) for k, v in gradcam_results.items()} if save_images else {},
        'detailed_results': gradcam_results
    }
    
    with open(gradcam_summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    if save_images:
        print(f"Grad-CAM analysis completed. {total_images_saved}/{total_images_analyzed} images saved to {gradcam_dir}")
    else:
        print(f"Grad-CAM analysis completed. Summary saved to {gradcam_summary_path}")
    
    return gradcam_results

# -------------------------
# 6. Main Training Function
# -------------------------
def main():
    """Main training function with comprehensive experiment management."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CWT Image Binary Classifier v3')
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--k_folds', type=int, help='Number of cross-validation folds')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--concise', action='store_true', help='Concise progress reporting')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Hyperopt-specific arguments for experiment tracking
    parser.add_argument('--source', type=str, default='manual', help='Source of experiment: manual or hyperopt')
    parser.add_argument('--hyperopt_run_id', type=str, help='Hyperopt run ID for tracking')
    parser.add_argument('--config_file', type=str, help='Config file path used for this experiment')
    parser.add_argument('--config_number_in_run', type=str, help='Config number within hyperopt run')
    parser.add_argument('--exclude_files', type=str, help='Path to file containing list of filenames to exclude from training')
    parser.add_argument('--dataset_variant', type=str, help='Dataset variant name for CSV-based labeling with flat directory structure')

    # Label configuration arguments (for CSV-based labeling)
    parser.add_argument('--label_file', type=str, help='Path to CSV file containing labels for flat directory structure')
    parser.add_argument('--label_column', type=str, default='has_porosity', help='Column name in label file to use for classification')
    parser.add_argument('--label_type', type=str, default='binary', choices=['binary', 'continuous'], help='Type of labels: binary or continuous')

    args = parser.parse_args()
    
    # Ensure CWT directories exist
    ensure_cwt_directories()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Load configuration
    config = get_cwt_config_template()
    if args.config:
        try:
            with open(args.config, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            print(f"Error loading config file: {e}")
            return 1
    
    # Apply command line overrides
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config[key] = value
    
    start_time = datetime.datetime.now()
    
    # Generate version for this experiment using centralized version management
    # Use proper v001, v002 format like PD log
    if hasattr(args, 'source') and args.source == 'hyperopt':
        # For hyperopt runs, use centralized version system
        version_num = get_next_version_from_log(classifier_type='cwt_image')
        version = format_version(version_num)
    else:
        # For manual runs, also use centralized version system
        version_num = get_next_version_from_log(classifier_type='cwt_image')
        version = format_version(version_num)
    
    if not args.concise:
        print("="*80)
        print("CWT IMAGE BINARY CLASSIFIER v3")
        print("="*80)
        print(f"Experiment Version: {version}")
        print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration: {args.config if args.config else 'Default + CLI overrides'}")
        print("="*80)
    
    try:
        # Load data
        if not args.concise:
            print("Loading CWT image data...")
        
        # Load exclusion list if provided
        exclude_files = set()
        if args.exclude_files:
            try:
                with open(args.exclude_files, 'r') as f:
                    exclude_files = set(line.strip() for line in f if line.strip())
                if not args.concise:
                    print(f"Loaded {len(exclude_files)} files to exclude from training")
            except Exception as e:
                print(f"Warning: Could not load exclusion file {args.exclude_files}: {e}")
        
        img_size = (config['img_width'], config['img_height'])

        # Check if using CSV-based labeling (either dataset variant or direct label_file)
        if args.dataset_variant or config.get('label_file'):
            # Handle dataset variant configuration
            if args.dataset_variant:
                from config import load_dataset_variant_info

                dataset_info = load_dataset_variant_info(args.dataset_variant)
                dataset_dir = dataset_info['dataset_dir']
                dataset_config = dataset_info['config']

                # Override config data_dir with variant's data_dir
                config['cwt_data_dir'] = dataset_config['data_dir']

                # Also update cwt_data_channels if it exists (multi-channel configs)
                # to ensure resolve_cwt_data_channels returns the correct path
                if 'cwt_data_channels' in config and config['cwt_data_channels']:
                    # Update all channels to point to the dataset variant's path
                    for channel_key in config['cwt_data_channels'].keys():
                        config['cwt_data_channels'][channel_key] = dataset_config['data_dir']

                # DEBUG: Verify the override worked
                if args.verbose:
                    print(f"DEBUG: Overrode config['cwt_data_dir'] = {config['cwt_data_dir']}")
                    if 'cwt_data_channels' in config:
                        print(f"DEBUG: Overrode config['cwt_data_channels'] = {config['cwt_data_channels']}")

                # Get label file path
                label_file = dataset_dir / dataset_config['label_file']

                # Load test exclusion if exists
                test_exclusion_file = dataset_dir / 'test.csv'
                if test_exclusion_file.exists():
                    import pandas as pd
                    test_df = pd.read_csv(test_exclusion_file, encoding='utf-8')
                    # Support both 'filename' and 'image_filename' column names
                    filename_col = 'filename' if 'filename' in test_df.columns else 'image_filename'
                    test_files = set(test_df[filename_col].tolist())
                    exclude_files.update(test_files)
                    if args.verbose:
                        print(f"Excluding {len(test_files)} test files from training")

                if args.verbose:
                    print(f"Using dataset variant: {args.dataset_variant}")
                    print(f"Data directory: {config['cwt_data_dir']}")
                    print(f"Label file: {label_file}")
            else:
                # Using direct label_file without dataset variant
                label_file = Path(config['label_file'])
                if args.verbose:
                    print(f"Using direct label file: {label_file}")
                    print(f"Data directory: {config.get('cwt_data_dir', 'multi-channel')}")

            # Resolve data directories for multi-channel support
            from config import resolve_cwt_data_channels
            try:
                channels_dict, channel_labels, channel_paths = resolve_cwt_data_channels(config)
                if args.verbose:
                    print(f"Loading data with channels: {channel_labels}")
                    print(f"DEBUG: Channel paths resolved: {channel_paths}")
            except:
                # Fallback for backward compatibility
                channel_paths = [config['cwt_data_dir']]
                if args.verbose:
                    print(f"Loading single-channel data from: {config['cwt_data_dir']}")
                    print(f"DEBUG: Using fallback channel path: {channel_paths}")

            # Load using CSV-based flat directory loader
            X, y, class_counts, label_encoder = load_cwt_image_data_from_csv(
                channel_paths,
                label_file,
                img_size,
                verbose=args.verbose,
                exclude_files=exclude_files
            )
        else:
            # Standard class-based directory structure loading
            # Resolve data directories for multi-channel support
            from config import resolve_cwt_data_channels
            try:
                channels_dict, channel_labels, channel_paths = resolve_cwt_data_channels(config)
                if args.verbose:
                    print(f"Loading data with channels: {channel_labels}")
            except:
                # Fallback for backward compatibility
                channel_paths = [config['cwt_data_dir']]
                if args.verbose:
                    print(f"Loading single-channel data from: {config['cwt_data_dir']}")

            X, y, class_counts, label_encoder = load_cwt_image_data(
                channel_paths,
                img_size,
                verbose=args.verbose,
                exclude_files=exclude_files
            )
        
        if not args.concise:
            print(f"Dataset loaded: {X.shape} images, {len(np.unique(y))} classes")
            print(f"Class distribution: {class_counts}")
        
        # Setup cross-validation
        skf = StratifiedKFold(n_splits=config['k_folds'], shuffle=True, random_state=args.seed)
        
        # Calculate class weights if requested
        class_weights = None
        if config['use_class_weights']:
            weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            class_weights = dict(zip(np.unique(y), weights))
            if not args.concise:
                print(f"Using class weights: {class_weights}")
        
        # Train all folds
        if not args.concise:
            print(f"\nStarting {config['k_folds']}-fold cross-validation...")
        
        fold_results = []
        best_overall_acc = 0.0
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            if not args.concise:
                print(f"\n{'='*20} FOLD {fold}/{config['k_folds']} {'='*20}")
            
            # Train this fold
            output_dir = CWT_OUTPUTS_DIR / version if config.get('run_gradcam', False) else None
            if output_dir:
                ensure_path_exists(output_dir)
            
            fold_result = train_fold(
                fold, train_idx, val_idx, X, y, config, 
                class_weights=class_weights, 
                best_overall_acc=best_overall_acc,
                concise=args.concise,
                output_dir=output_dir
            )
            
            fold_results.append(fold_result)
            
            # Save training history per fold (like PD training does)
            if config.get('run_gradcam', False):  # Only save if we're creating output directory
                output_dir = CWT_OUTPUTS_DIR / version
                ensure_path_exists(output_dir)
                
                # Create history subdirectory
                history_dir = output_dir / 'history'
                ensure_path_exists(history_dir)
                
                # Save training history as CSV
                history_df = pd.DataFrame(fold_result['history'].history)
                history_filename = history_dir / f'training_history_fold_{fold}.csv'
                history_df.to_csv(history_filename, index=False, encoding='utf-8')
                
                # Save predictions CSV per fold
                predictions_dir = output_dir / 'predictions'
                ensure_path_exists(predictions_dir)
                
                # Create predictions DataFrame
                pred_data = {
                    'true_label': fold_result['y_val_flat'],
                    'predicted_label': fold_result['y_pred'],
                    'predicted_probability': fold_result['y_pred_proba'].flatten() if hasattr(fold_result['y_pred_proba'], 'flatten') else fold_result['y_pred_proba']
                }
                pred_df = pd.DataFrame(pred_data)
                pred_filename = predictions_dir / f'predictions_fold_{fold}.csv'
                pred_df.to_csv(pred_filename, index=False, encoding='utf-8')
                
                # Save confusion matrix and training history plots using consolidated function
                save_fold_plots(
                    classifier_type='cwt_image',
                    y_true=fold_result['y_val_flat'], 
                    y_pred=fold_result['y_pred'],
                    history=fold_result['history'],
                    fold=fold,
                    output_dir=output_dir,
                    concise=args.concise
                )
                
                if not args.concise:
                    print(f"Training history saved: {history_filename}")
                    print(f"Predictions saved: {pred_filename}")
            
            # Update best overall accuracy
            if fold_result['val_accuracy'] > best_overall_acc:
                best_overall_acc = fold_result['val_accuracy']
            
            if not args.concise:
                print(f"Fold {fold} Results:")
                print(f"  Validation Accuracy: {fold_result['val_accuracy']:.4f}")
                print(f"  Training Accuracy: {fold_result['train_accuracy']:.4f}")
                print(f"  Precision: {fold_result['precision']:.4f}")
                print(f"  Recall: {fold_result['recall']:.4f}")
                print(f"  F1 Score: {fold_result['f1_score']:.4f}")
            else:
                # Concise fold completion message
                print(f"Fold {fold}/{config['k_folds']} completed | Val Acc: {fold_result['val_accuracy']:.4f} | P: {fold_result['precision']:.3f} | R: {fold_result['recall']:.3f} | F1: {fold_result['f1_score']:.3f}\n")
        
        end_time = datetime.datetime.now()
        
        # Calculate final metrics
        val_accuracies = [r['val_accuracy'] for r in fold_results]
        mean_val_acc = np.mean(val_accuracies)
        std_val_acc = np.std(val_accuracies)
        
        # Calculate precision, recall, F1 metrics for concise display
        precisions = [r['precision'] for r in fold_results]
        recalls = [r['recall'] for r in fold_results]
        f1_scores = [r['f1_score'] for r in fold_results]
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1_score = np.mean(f1_scores)
        
        # Save k-fold results summary (like PD training does)
        if config.get('run_gradcam', False):  # Only save if we're creating output directory
            output_dir = CWT_OUTPUTS_DIR / version
            ensure_path_exists(output_dir)
            
            # Prepare fold results for CSV (exclude complex objects)
            results_for_csv = []
            for result in fold_results:
                csv_result = {
                    'fold': result['fold'],
                    'train_accuracy': result['train_accuracy'],
                    'train_loss': result['train_loss'],
                    'val_accuracy': result['val_accuracy'],
                    'val_loss': result['val_loss'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1_score': result['f1_score'],
                    'best_epoch': result['best_epoch'],
                    'total_epochs': result['total_epochs'],
                    'model_complexity': result['model_complexity']
                }
                results_for_csv.append(csv_result)
            
            # Save k-fold results summary
            results_df = pd.DataFrame(results_for_csv)
            summary_filename = output_dir / 'kfold_results_summary.csv'
            results_df.to_csv(summary_filename, index=False, encoding='utf-8')
            
            if not args.concise:
                print(f"K-fold results summary saved: {summary_filename}")
        
        # Print final results
        if args.concise:
            print(f"\n✅ Training completed | Mean Val Acc: {mean_val_acc:.4f} ± {std_val_acc:.4f} | Best: {best_overall_acc:.4f} | P: {mean_precision:.3f} | R: {mean_recall:.3f} | F1: {mean_f1_score:.3f}")
        else:
            print("\n" + "="*60)
            print("FINAL RESULTS")
            print("="*60)
            print(f"Mean Validation Accuracy: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
            print(f"Best Fold Accuracy: {best_overall_acc:.4f}")
            print(f"Training Duration: {(end_time - start_time).total_seconds()/60:.1f} minutes")
            print("="*60)
        
        # Get model complexity from first fold (should be same for all folds)
        model_complexity = fold_results[0].get('model_complexity', 0) if fold_results else 0
        
        # Log results using consolidated function
        log_entry = log_experiment_results(
            classifier_type='cwt_image',
            version=version,
            start_time=start_time, 
            end_time=end_time,
            config=config,
            fold_results=fold_results,
            X=X, 
            y=y, 
            class_counts=class_counts,
            model_complexity=model_complexity,
            source=getattr(args, 'source', 'manual'),
            hyperopt_run_id=getattr(args, 'hyperopt_run_id', None),
            config_file=getattr(args, 'config_file', None),
            config_number_in_run=getattr(args, 'config_number_in_run', None)
        )
        
        # Save experiment summary files using consolidated function
        if config.get('run_gradcam', False):
            output_dir = CWT_OUTPUTS_DIR / version
            ensure_path_exists(output_dir)
            
            # Prepare data info and experiment results for consolidated function
            training_time_minutes = (end_time - start_time).total_seconds() / 60
            best_fold_idx = np.argmax(val_accuracies)
            
            data_info = {
                'cwt_data_dir': config['cwt_data_dir'],
                'total_samples': len(X),
                'num_classes': len(np.unique(y)),
                'class_distribution': {str(k): str(v) for k, v in class_counts.items()},
                'img_width': config['img_width'],
                'img_height': config['img_height'],
                'imbalance_ratio': max(class_counts.values()) / min(class_counts.values()) if len(class_counts) > 1 else 1.0
            }
            
            experiment_results = {
                'mean_accuracy': mean_val_acc,
                'std_accuracy': std_val_acc,
                'mean_loss': np.mean([r['val_loss'] for r in fold_results]),
                'std_loss': np.std([r['val_loss'] for r in fold_results]),
                'mean_precision': mean_precision,
                'mean_recall': mean_recall,
                'mean_f1_score': mean_f1_score,
                'best_accuracy': best_overall_acc,
                'best_fold': best_fold_idx + 1,
                'fold_accuracies': [r['val_accuracy'] for r in fold_results],
                'fold_losses': [r['val_loss'] for r in fold_results],
                'fold_precisions': [r['precision'] for r in fold_results],
                'fold_recalls': [r['recall'] for r in fold_results],
                'fold_f1_scores': [r['f1_score'] for r in fold_results],
                'training_time_minutes': training_time_minutes,
            }
            
            # Create summary files using consolidated function
            summary_filename, readable_summary_filename = create_experiment_summary_files(
                classifier_type='cwt_image',
                output_dir=output_dir,
                version=version,
                config=config,
                data_info=data_info,
                experiment_results=experiment_results,
                source=getattr(args, 'source', 'manual'),
                start_time=start_time
            )
            
            if not args.concise:
                print(f"Experiment summary saved: {summary_filename}")
                print(f"Human-readable summary saved: {readable_summary_filename}")
        
        # Run Grad-CAM analysis on the best model
        if config.get('run_gradcam', False):
            best_fold_idx = np.argmax(val_accuracies)
            best_model = fold_results[best_fold_idx]['model']
            
            # Output directory already created above
            output_dir = CWT_OUTPUTS_DIR / version
            
            # Save best model
            best_model.save(output_dir / f"best_model_{version}.h5")
            
            # Run Grad-CAM on validation set from best fold
            gradcam_results = run_gradcam_analysis(
                best_model,
                fold_results[best_fold_idx]['X_val'],
                fold_results[best_fold_idx]['y_val'],
                config,
                output_dir,
                concise=args.concise
            )
            
            # Save configuration
            with open(output_dir / f"config_{version}.json", 'w') as f:
                json.dump(config, f, indent=2, default=str)
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())