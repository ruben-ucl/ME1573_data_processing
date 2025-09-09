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
from tensorflow.keras.utils import normalize
from PIL import Image

from config import (
    get_cwt_data_dir, get_cwt_experiment_log_path, get_cwt_config_template, 
    format_version, CWT_MODELS_DIR, CWT_LOGS_DIR, ensure_cwt_directories,
    normalize_path, ensure_path_exists, convert_numpy_types
)

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
def load_cwt_image_data(root_dir, img_size, verbose=False, exclude_files=None):
    """
    Load CWT image data from directory structure.
    
    Expected structure:
    root_dir/
    ├── 0/  (class 0 images)
    │   ├── image1.png
    │   └── image2.png
    └── 1/  (class 1 images)
        ├── image3.png
        └── image4.png
    
    Args:
        root_dir: Path to root directory containing class subdirectories
        img_size: Tuple (width, height) for image resizing
        verbose: Print progress information
        exclude_files: Set of filenames to exclude
        
    Returns:
        tuple: (images_array, labels_array, class_counts, label_encoder)
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"CWT data directory not found: {root_dir}")
    
    if not root_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root_dir}")
    
    images, labels = [], []
    valid_classes = []
    exclude_files = exclude_files or set()
    
    # Find all class directories (should be numeric: 0, 1, etc.)
    class_dirs = [d for d in root_path.iterdir() if d.is_dir() and d.name.isdigit()]
    class_dirs.sort(key=lambda x: int(x.name))  # Sort numerically
    
    if not class_dirs:
        raise ValueError(f"No class directories found in {root_dir}")
    
    if verbose:
        print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    
    # Load images from each class directory
    for class_dir in class_dirs:
        class_label = int(class_dir.name)
        valid_classes.append(class_label)
        
        # Find all image files in this class directory
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
            image_files.extend(class_dir.glob(ext))
        
        class_count = 0
        for img_path in sorted(image_files):
            if img_path.name in exclude_files:
                if verbose:
                    print(f"Excluding file: {img_path.name}")
                continue
                
            try:
                # Load and preprocess image
                image = load_and_preprocess_image(str(img_path), img_size)
                images.append(image)
                labels.append(class_label)
                class_count += 1
                
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to load {img_path}: {e}")
                continue
        
        if verbose:
            print(f"Class {class_label}: {class_count} images loaded")
    
    if not images:
        raise ValueError(f"No valid images found in {root_dir}")
    
    # Convert to arrays
    images_array = np.array(images)
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

# -------------------------
# 2. Data Augmentation
# -------------------------
def create_data_generator(config):
    """Create ImageDataGenerator for data augmentation."""
    if config['augment_fraction'] <= 0:
        return None
    
    return ImageDataGenerator(
        rotation_range=config.get('rotation_range', 0.0),
        width_shift_range=config.get('width_shift_range', 0.0),
        height_shift_range=config.get('height_shift_range', 0.0),
        horizontal_flip=config.get('horizontal_flip', False),
        vertical_flip=config.get('vertical_flip', False),
        fill_mode='constant',
        cval=0.0
    )

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
    use_batch_norm = config.get('use_batch_norm', False)
    
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
    
    if verbose:
        print("\n" + "="*50)
        print("MODEL ARCHITECTURE")
        print("="*50)
        model.summary()
        print("="*50)
    
    return model

# -------------------------
# 4. Training Functions
# -------------------------
def train_fold(fold, train_idx, val_idx, X, y, config, class_weights=None, best_overall_acc=0.0, concise=False):
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
    X_train = normalize(X_train, axis=1)
    X_val = normalize(X_val, axis=1)
    
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
    
    # Setup data augmentation
    data_generator = create_data_generator(config)
    
    # Train model
    if data_generator and config['augment_fraction'] > 0:
        # Calculate steps for augmented training
        augment_samples = int(len(X_train) * config['augment_fraction'])
        steps_per_epoch = max(1, augment_samples // config['batch_size'])
        
        history = model.fit(
            data_generator.flow(X_train, y_train, batch_size=config['batch_size']),
            steps_per_epoch=steps_per_epoch,
            epochs=config['epochs'],
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callback_list,
            verbose=0 if concise else 1
        )
    else:
        # Standard training without augmentation
        history = model.fit(
            X_train, y_train,
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
    
    precision = metrics.precision_score(y_val, y_pred, average='binary', zero_division=0)
    recall = metrics.recall_score(y_val, y_pred, average='binary', zero_division=0)
    f1 = metrics.f1_score(y_val, y_pred, average='binary', zero_division=0)
    
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
        'X_val': X_val,
        'y_val': y_val,
        'y_pred_proba': y_pred_proba
    }

# -------------------------
# 5. Evaluation and Analysis
# -------------------------
def run_gradcam_analysis(model, X_sample, y_sample, config, output_dir):
    """
    Run Grad-CAM analysis on sample images.
    
    Args:
        model: Trained CNN model
        X_sample: Sample images for analysis
        y_sample: Sample labels
        config: Configuration dictionary
        output_dir: Directory to save results
    """
    if not config.get('run_gradcam', False):
        return None
    
    print("\nRunning Grad-CAM analysis...")
    
    # Auto-detect last convolutional layer if needed
    gradcam_layer = config.get('gradcam_layer', 'auto')
    if gradcam_layer == 'auto':
        # Find last convolutional layer
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                gradcam_layer = layer.name
                break
        
        if gradcam_layer == 'auto':
            print("Warning: No convolutional layers found for Grad-CAM")
            return None
    
    print(f"Using layer '{gradcam_layer}' for Grad-CAM analysis")
    
    # Create grad-CAM model
    try:
        grad_model = tf.keras.Model(
            model.inputs,
            [model.get_layer(gradcam_layer).output, model.output]
        )
    except ValueError as e:
        print(f"Warning: Could not create Grad-CAM model: {e}")
        return None
    
    # Analyze a few sample images from each class
    gradcam_results = {}
    unique_labels = np.unique(y_sample)
    
    for label in unique_labels:
        label_indices = np.where(y_sample == label)[0]
        sample_indices = label_indices[:min(5, len(label_indices))]  # Max 5 samples per class
        
        heatmaps = []
        for idx in sample_indices:
            img_array = np.expand_dims(X_sample[idx], axis=0)
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                class_channel = predictions[:, 0] if len(predictions.shape) > 1 else predictions
            
            grads = tape.gradient(class_channel, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            heatmaps.append(heatmap.numpy())
        
        if heatmaps:
            gradcam_results[f'class_{label}'] = {
                'mean_heatmap': np.mean(heatmaps, axis=0),
                'sample_count': len(heatmaps)
            }
    
    # Save results summary
    gradcam_summary_path = Path(output_dir) / 'gradcam_summary.json'
    with open(gradcam_summary_path, 'w') as f:
        json.dump({
            'gradcam_layer': gradcam_layer,
            'classes_analyzed': list(gradcam_results.keys()),
            'samples_per_class': {k: v['sample_count'] for k, v in gradcam_results.items()}
        }, f, indent=2)
    
    print(f"Grad-CAM analysis completed. Results saved to {output_dir}")
    return gradcam_results

# -------------------------
# 6. Experiment Logging
# -------------------------
def log_experiment_results(config, fold_results, version, start_time, end_time):
    """
    Log experiment results to CWT-specific experiment log.
    
    Args:
        config: Configuration dictionary
        fold_results: List of fold result dictionaries
        version: Experiment version string
        start_time: Experiment start timestamp
        end_time: Experiment end timestamp
    """
    # Calculate aggregate metrics
    val_accuracies = [r['val_accuracy'] for r in fold_results]
    val_losses = [r['val_loss'] for r in fold_results]
    train_accuracies = [r['train_accuracy'] for r in fold_results]
    precisions = [r['precision'] for r in fold_results]
    recalls = [r['recall'] for r in fold_results]
    f1_scores = [r['f1_score'] for r in fold_results]
    
    # Prepare log entry with CWT-specific columns
    log_entry = {
        'version': version,
        'timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_minutes': (end_time - start_time).total_seconds() / 60,
        
        # Data configuration
        'cwt_data_dir': config['cwt_data_dir'],
        'img_width': config['img_width'],
        'img_height': config['img_height'],
        'img_channels': config['img_channels'],
        
        # Training configuration
        'k_folds': config['k_folds'],
        'epochs': config['epochs'],
        'batch_size': config['batch_size'],
        'learning_rate': config['learning_rate'],
        'optimizer': config.get('optimizer', 'adam'),
        
        # Architecture configuration
        'conv_filters': str(config['conv_filters']),
        'conv_kernel_size': str(config['conv_kernel_size']),
        'pool_size': str(config['pool_size']),
        'pool_layers': str(config.get('pool_layers', [])),
        'dense_units': str(config['dense_units']),
        'conv_dropout': config['conv_dropout'],
        'dense_dropout': config['dense_dropout'] if isinstance(config['dense_dropout'], (int, float)) else str(config['dense_dropout']),
        'l2_regularization': config['l2_regularization'],
        'use_batch_norm': config.get('use_batch_norm', False),
        'use_class_weights': config['use_class_weights'],
        
        # Training parameters
        'early_stopping_patience': config['early_stopping_patience'],
        'lr_reduction_patience': config['lr_reduction_patience'],
        'lr_reduction_factor': config['lr_reduction_factor'],
        
        # Augmentation parameters
        'augment_fraction': config['augment_fraction'],
        'rotation_range': config.get('rotation_range', 0.0),
        'width_shift_range': config.get('width_shift_range', 0.0),
        'height_shift_range': config.get('height_shift_range', 0.0),
        'horizontal_flip': config.get('horizontal_flip', False),
        'vertical_flip': config.get('vertical_flip', False),
        'noise_std': config.get('noise_std', 0.0),
        
        # Analysis parameters
        'run_gradcam': config.get('run_gradcam', False),
        'gradcam_layer': config.get('gradcam_layer', 'auto'),
        
        # Performance metrics - validation
        'mean_val_accuracy': np.mean(val_accuracies),
        'std_val_accuracy': np.std(val_accuracies),
        'best_fold_val_accuracy': np.max(val_accuracies),
        'mean_val_loss': np.mean(val_losses),
        'std_val_loss': np.std(val_losses),
        
        # Performance metrics - training
        'mean_train_accuracy': np.mean(train_accuracies),
        'std_train_accuracy': np.std(train_accuracies),
        
        # Additional metrics
        'mean_precision': np.mean(precisions),
        'std_precision': np.std(precisions),
        'mean_recall': np.mean(recalls),
        'std_recall': np.std(recalls),
        'mean_f1_score': np.mean(f1_scores),
        'std_f1_score': np.std(f1_scores),
        
        # Training statistics
        'mean_epochs_trained': np.mean([r['total_epochs'] for r in fold_results]),
        'mean_best_epoch': np.mean([r['best_epoch'] for r in fold_results]),
        
        # Per-fold results (as strings for CSV storage)
        'fold_val_accuracies': str([round(acc, 4) for acc in val_accuracies]),
        'fold_train_accuracies': str([round(acc, 4) for acc in train_accuracies]),
        'fold_val_losses': str([round(loss, 4) for loss in val_losses]),
        'fold_precisions': str([round(p, 4) for p in precisions]),
        'fold_recalls': str([round(r, 4) for r in recalls]),
        'fold_f1_scores': str([round(f1, 4) for f1 in f1_scores])
    }
    
    # Convert numpy types for JSON serialization
    log_entry = convert_numpy_types(log_entry)
    
    # Write to CSV log
    log_path = get_cwt_experiment_log_path()
    log_df = pd.DataFrame([log_entry])
    
    # Append to existing log or create new one
    if log_path.exists():
        log_df.to_csv(log_path, mode='a', header=False, index=False, encoding='utf-8')
    else:
        log_df.to_csv(log_path, index=False, encoding='utf-8')
    
    print(f"Results logged to: {log_path}")
    
    return log_entry

# -------------------------
# 7. Main Training Function
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
    
    # Generate version for this experiment
    version = format_version(start_time.strftime('%H%M%S'))
    
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
        
        img_size = (config['img_width'], config['img_height'])
        X, y, class_counts, label_encoder = load_cwt_image_data(
            config['cwt_data_dir'], 
            img_size, 
            verbose=args.verbose
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
            fold_result = train_fold(
                fold, train_idx, val_idx, X, y, config, 
                class_weights=class_weights, 
                best_overall_acc=best_overall_acc,
                concise=args.concise
            )
            
            fold_results.append(fold_result)
            
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
        
        end_time = datetime.datetime.now()
        
        # Calculate final metrics
        val_accuracies = [r['val_accuracy'] for r in fold_results]
        mean_val_acc = np.mean(val_accuracies)
        std_val_acc = np.std(val_accuracies)
        
        # Print final results
        if args.concise:
            print(f"\n✅ Training completed | Mean Val Acc: {mean_val_acc:.4f} ± {std_val_acc:.4f} | Best: {best_overall_acc:.4f}")
        else:
            print("\n" + "="*60)
            print("FINAL RESULTS")
            print("="*60)
            print(f"Mean Validation Accuracy: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
            print(f"Best Fold Accuracy: {best_overall_acc:.4f}")
            print(f"Training Duration: {(end_time - start_time).total_seconds()/60:.1f} minutes")
            print("="*60)
        
        # Log results
        log_entry = log_experiment_results(config, fold_results, version, start_time, end_time)
        
        # Run Grad-CAM analysis on the best model
        if config.get('run_gradcam', False):
            best_fold_idx = np.argmax(val_accuracies)
            best_model = fold_results[best_fold_idx]['model']
            
            # Create output directory for this experiment
            output_dir = CWT_MODELS_DIR / f"cwt_experiment_{version}"
            ensure_path_exists(output_dir)
            
            # Save best model
            best_model.save(output_dir / f"best_model_{version}.h5")
            
            # Run Grad-CAM on validation set from best fold
            gradcam_results = run_gradcam_analysis(
                best_model,
                fold_results[best_fold_idx]['X_val'],
                fold_results[best_fold_idx]['y_val'],
                config,
                output_dir
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