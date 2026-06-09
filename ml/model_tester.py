#!/usr/bin/env python3
"""
Model Tester Script

This script evaluates a trained model on a held-out test set to provide
unbiased performance metrics for model presentation and reporting.

Key features:
- Loads a trained model and evaluates it on test data
- Provides detailed metrics including accuracy, precision, recall, F1
- Saves results to the model's output directory
- Supports both individual predictions and aggregated metrics

Author: AI Assistant
"""

# Ensure UTF-8 encoding for all I/O operations
import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import time
from datetime import datetime

# Local imports first — config registers the CUDA DLL directory before TF loads
from config import (convert_numpy_types, CWT_OUTPUTS_DIR, PD_OUTPUTS_DIR, ML_ROOT,
                    format_version, load_dataset_variant_info,
                    get_cwt_experiment_log_path, get_pd_experiment_log_path)
from data_utils import normalize_image, load_cwt_test_images, load_pd_test_images

# CPU-only inference — faster than GPU for small test sets due to transfer overhead
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF info/warning logs

import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)

def _resolve_paths_from_version(version_str, classifier_type='cwt_image'):
    """Resolve model, test_data, and output_dir paths from a version string."""
    base_dir = PD_OUTPUTS_DIR if classifier_type == 'pd_signal' else CWT_OUTPUTS_DIR
    version_dir = base_dir / version_str
    # Search in priority order: best_model* at root, then models/ subdir
    model_candidates = (
        sorted(version_dir.glob('best_model*.h5')) +
        sorted(version_dir.glob('best_model*.keras')) +
        sorted(version_dir.glob('models/final_model*.h5')) +
        sorted(version_dir.glob('models/final_model*.keras'))
    )
    model_path = str(model_candidates[-1]) if model_candidates else str(version_dir / f'best_model_{version_str}.h5')
    return {
        'model':      model_path,
        'test_data':  str(version_dir / 'test_set_data.pkl'),
        'output_dir': str(version_dir / 'test_evaluation'),
    }

class ModelTester:
    """Evaluates trained models on test data."""
    
    def __init__(self, model_path, test_data_path=None, output_dir='.', verbose=False):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path) if test_data_path else None
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _safe_correlation(self, x, y):
        """Safely calculate correlation with error handling."""
        try:
            # Ensure inputs are 1-D arrays
            x = np.asarray(x).flatten()
            y = np.asarray(y).flatten()
            
            # Check if arrays are valid
            if len(x) < 2 or len(y) < 2 or len(x) != len(y):
                return 0.0
            
            # Check for constant arrays
            if np.std(x) == 0 or np.std(y) == 0:
                return 0.0
            
            # Calculate correlation
            corr_matrix = np.corrcoef(x, y)
            correlation = corr_matrix[0, 1]
            
            # Handle NaN results
            if np.isnan(correlation):
                return 0.0
                
            return float(correlation)
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Correlation calculation failed: {e}")
            return 0.0
        
    def load_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if self.verbose:
            print(f"Loading model from: {self.model_path}")
            
        try:
            model = load_model(str(self.model_path))
            if self.verbose:
                print(f"Model loaded successfully")
                model.summary()
            return model
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def load_test_data(self):
        """Load test data from pickle file."""
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"Test data not found: {self.test_data_path}")

        if self.verbose:
            print(f"Loading test data from: {self.test_data_path}")

        try:
            with open(self.test_data_path, 'rb') as f:
                test_data = pickle.load(f)

            X_test = test_data['X_test']
            y_test = test_data['y_test']
            test_files = test_data.get('test_files', None)
            classifier_type = test_data.get('classifier_type', None)

            if self.verbose:
                print(f"Test data loaded: {len(X_test)} samples")
                print(f"Test data shape: {X_test.shape if hasattr(X_test, 'shape') else 'Multiple arrays'}")
                print(f"Class distribution: {np.bincount(y_test)}")

            return X_test, y_test, test_files, classifier_type

        except Exception as e:
            raise Exception(f"Failed to load test data: {e}")
    
    def prepare_test_data(self, X_test, y_test):
        """Prepare test data for model evaluation."""
        # Handle tuple format (pd1_data, pd2_data) for dual-branch model
        if isinstance(X_test, tuple):
            pd1_test, pd2_test = X_test
            
            # Keep as separate arrays for dual-branch model (DO NOT combine)
            pd1_test = pd1_test.astype(np.float32)
            pd2_test = pd2_test.astype(np.float32)
            
            # Check if we need to adapt data shape for legacy models
            # Legacy models expect (N, 2, 1), new models expect (N, 100, 1)
            if len(pd1_test) > 0:
                expected_shape = self._get_expected_input_shape()
                current_shape = pd1_test[0].shape
                
                if expected_shape and expected_shape[1] != current_shape[0]:
                    if self.verbose:
                        print(f"Shape mismatch detected:")
                        print(f"  Model expects: {expected_shape}")
                        print(f"  Data provides: (None, {current_shape[0]}, {current_shape[1]})")
                    
                    # Handle legacy model compatibility
                    if expected_shape[1] == 2 and current_shape[0] == 100:
                        if self.verbose:
                            print("  Adapting data for legacy model (100,1) -> (2,1)")
                        
                        # Take first 2 values from each signal for legacy compatibility
                        pd1_test_adapted = []
                        pd2_test_adapted = []
                        for pd1_signal, pd2_signal in zip(pd1_test, pd2_test):
                            pd1_test_adapted.append(pd1_signal[:2])  # Take first 2 values
                            pd2_test_adapted.append(pd2_signal[:2])  # Take first 2 values
                        
                        pd1_test = np.array(pd1_test_adapted)
                        pd2_test = np.array(pd2_test_adapted)
                        
                        if self.verbose:
                            print(f"  Adapted shapes: PD1 {pd1_test.shape}, PD2 {pd2_test.shape}")
                    
                    elif expected_shape[1] == 100 and current_shape[0] == 2:
                        # Handle opposite case (if needed in future)
                        print("Warning: Model expects (100,1) but data provides (2,1) - this may indicate an error")
            
            # Data should already be normalized from the final model trainer
            # Just ensure it's in the right format
            X_test = [pd1_test, pd2_test]  # List format for dual inputs
            
            if self.verbose:
                print(f"Prepared dual-branch test data:")
                print(f"  PD1 shape: {pd1_test.shape}, range: [{pd1_test.min():.4f}, {pd1_test.max():.4f}]")
                print(f"  PD2 shape: {pd2_test.shape}, range: [{pd2_test.min():.4f}, {pd2_test.max():.4f}]")
        else:
            # Single input format (fallback for other model types)
            X_test = X_test.astype(np.float32)
            # Note: Data should already be normalized by the data pipeline
            
            # Add channel dimension if needed
            if len(X_test.shape) == 3:
                X_test = np.expand_dims(X_test, axis=-1)
            
            if self.verbose:
                print(f"Prepared single-input test data shape: {X_test.shape}")
                print(f"Test data range: [{X_test.min():.4f}, {X_test.max():.4f}]")
        
        return X_test, y_test
    
    def _benchmark_cwt_time(self, n_scales, n_time_samples, n_repeats=5):
        """Time a representative CWT on a synthetic signal. Returns ms per sample."""
        import pywt
        wavelet = 'cmor1.5-1.0'
        sampling_period = 1e-5  # 100 kHz — typical for this dataset
        scales = np.arange(1, n_scales + 1, dtype=float)
        signal = np.random.randn(n_time_samples)
        # Warm-up run to exclude any one-time initialisation overhead
        pywt.cwt(signal, scales, wavelet, sampling_period=sampling_period)
        elapsed = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            pywt.cwt(signal, scales, wavelet, sampling_period=sampling_period)
            elapsed.append(time.perf_counter() - t0)
        return float(np.median(elapsed)) * 1000  # ms per sample

    def _get_expected_input_shape(self):
        """Get the expected input shape from the loaded model."""
        try:
            if hasattr(self, '_model_cache'):
                model = self._model_cache
            else:
                # Try to load model to check input shape
                from tensorflow.keras.models import load_model
                model = load_model(str(self.model_path))
                self._model_cache = model
            
            if hasattr(model, 'inputs') and len(model.inputs) > 0:
                return model.inputs[0].shape
        except Exception as e:
            if self.verbose:
                print(f"Could not determine expected input shape: {e}")
        
        return None
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model on test data and compute comprehensive metrics."""
        if self.verbose:
            print("Evaluating model on test data...")
        
        # Get predictions and time the inference
        n_samples = len(y_test)
        _t0 = time.perf_counter()
        y_pred_proba = model.predict(X_test, verbose=0)
        _inference_seconds = time.perf_counter() - _t0
        predictions_per_second = n_samples / _inference_seconds if _inference_seconds > 0 else float('inf')
        ms_per_sample_classify = (_inference_seconds / n_samples) * 1000 if n_samples > 0 else 0.0

        # CWT timing benchmark (CWT image classifiers only — inferred from 4-D array input)
        if isinstance(X_test, np.ndarray) and X_test.ndim == 4:
            ms_per_sample_cwt = self._benchmark_cwt_time(X_test.shape[1], X_test.shape[2])
        else:
            ms_per_sample_cwt = None

        ms_per_sample_total = (
            ms_per_sample_classify + ms_per_sample_cwt
            if ms_per_sample_cwt is not None
            else ms_per_sample_classify
        )
        
        THRESHOLD = 0.4
        print(f"Using classification threshold: {THRESHOLD} (hardcoded — run final_model_trainer for optimized threshold)")

        # Handle softmax output (n_samples, n_classes) vs sigmoid output (n_samples, 1)
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            # Multi-class softmax output: argmax ignores threshold
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_pred_proba_binary = y_pred_proba[:, 1]
        else:
            # Binary sigmoid output
            y_pred_proba_binary = y_pred_proba.flatten()
            y_pred = (y_pred_proba_binary > THRESHOLD).astype(int)
        
        # Ensure y_test is in the right format
        if len(y_test.shape) > 1:
            y_test = y_test.flatten()
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        # ROC AUC (if binary classification)
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba_binary)
        else:
            roc_auc = None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Compile results
        results = {
            'test_accuracy': float(accuracy),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1_score': float(f1),
            'test_roc_auc': float(roc_auc) if roc_auc is not None else None,
            'test_samples': int(len(y_test)),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': {
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.flatten().tolist() if len(y_pred_proba.shape) > 1 else y_pred_proba.tolist()
            },
            'inference_time_seconds': float(_inference_seconds),
            'ms_per_sample_classify': float(ms_per_sample_classify),
            'predictions_per_second': float(predictions_per_second),
            'ms_per_sample_cwt': float(ms_per_sample_cwt) if ms_per_sample_cwt is not None else None,
            'ms_per_sample_total': float(ms_per_sample_total),
        }
        
        if self.verbose:
            print(f"\nTest Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            if roc_auc is not None:
                print(f"  ROC AUC: {roc_auc:.4f}")
            print(f"  Classification speed: {predictions_per_second:.1f} pred/s ({ms_per_sample_classify:.2f} ms/sample)")
            if ms_per_sample_cwt is not None:
                print(f"  CWT time (benchmark):  {ms_per_sample_cwt:.2f} ms/sample")
                print(f"  End-to-end estimate:   {ms_per_sample_total:.2f} ms/sample")
            print(f"\nConfusion Matrix:")
            print(cm)
        
        return results
    
    def analyze_pd_activations(self, model, X_test, y_test, num_samples_per_class=10):
        """
        Analyze PD classifier activations to understand what the model focuses on.
        
        Args:
            model: Trained dual-branch model
            X_test: Test data [pd1_data, pd2_data]
            y_test: Test labels
            num_samples_per_class: Number of samples to analyze per class
            
        Returns:
            dict: Comprehensive activation analysis results
        """
        if self.verbose:
            print("\nAnalyzing PD classifier activations...")
        
        # Create activation analysis directory
        activation_dir = self.output_dir / 'activation_analysis'
        activation_dir.mkdir(exist_ok=True)
        
        # Check if this is a dual-branch model
        if not isinstance(X_test, list) or len(X_test) != 2:
            if self.verbose:
                print("Activation analysis requires dual-branch PD model with [PD1, PD2] inputs")
            return None
        
        pd1_test, pd2_test = X_test
        
        # Get model predictions and intermediate layer outputs
        intermediate_layer_model = self._create_intermediate_model(model)
        if intermediate_layer_model is None:
            return None
        
        # Analyze activations by class
        unique_classes = np.unique(y_test)
        activation_results = {
            'metadata': {
                'model_architecture': str([layer.name for layer in model.layers]),
                'pd1_shape': pd1_test.shape,
                'pd2_shape': pd2_test.shape,
                'num_classes': len(unique_classes),
                'num_samples_per_class': num_samples_per_class
            },
            'by_class': {},
            'overall_patterns': {}
        }
        
        # Sample data for analysis
        sample_indices = self._get_sample_indices_per_class(y_test, unique_classes, num_samples_per_class)
        
        # Analyze each class
        for class_label in unique_classes:
            class_name = f"class_{int(class_label)}"
            indices = sample_indices[class_label]
            
            if self.verbose:
                print(f"  Analyzing {class_name} ({len(indices)} samples)...")
            
            # Get samples for this class
            pd1_samples = pd1_test[indices]
            pd2_samples = pd2_test[indices]
            y_samples = y_test[indices]
            
            # Get predictions and activations
            predictions = model.predict([pd1_samples, pd2_samples], verbose=0)
            activations = intermediate_layer_model.predict([pd1_samples, pd2_samples], verbose=0)
            
            # Analyze channel-specific patterns
            class_analysis = self._analyze_class_activations(
                pd1_samples, pd2_samples, y_samples, predictions, activations, 
                class_label, activation_dir
            )
            
            activation_results['by_class'][class_name] = class_analysis
        
        # Generate overall pattern analysis
        activation_results['overall_patterns'] = self._analyze_overall_patterns(
            activation_results['by_class'], activation_dir
        )
        
        # Save activation analysis results
        results_file = activation_dir / 'activation_analysis.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(activation_results, f, indent=2, ensure_ascii=False, default=str)
        
        if self.verbose:
            print(f"  Activation analysis saved to: {activation_dir}")
        
        return activation_results
    
    def _create_intermediate_model(self, model):
        """Create model to extract intermediate activations."""
        try:
            # Find key intermediate layers to extract
            layer_names = [layer.name for layer in model.layers]
            
            # Look for common patterns in dual-branch PD models
            target_layers = []
            
            # Find dense layers after branches merge
            for layer in model.layers:
                if 'dense' in layer.name.lower() and 'merge' not in layer.name.lower():
                    target_layers.append(layer.name)
                elif 'concatenate' in layer.name.lower():
                    target_layers.append(layer.name)
                elif 'attention' in layer.name.lower():
                    target_layers.append(layer.name)
            
            if not target_layers:
                # Fallback: use last few layers before output
                target_layers = [layer.name for layer in model.layers[-3:-1]]
            
            if self.verbose:
                print(f"  Extracting activations from layers: {target_layers}")
            
            # Create intermediate model
            outputs = [model.get_layer(name).output for name in target_layers]
            outputs.append(model.output)  # Include final predictions
            
            intermediate_model = tf.keras.Model(inputs=model.input, outputs=outputs)
            return intermediate_model
            
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Could not create intermediate model: {e}")
            return None
    
    def _get_sample_indices_per_class(self, y_test, unique_classes, num_samples_per_class):
        """Get representative sample indices for each class."""
        sample_indices = {}
        
        for class_label in unique_classes:
            class_indices = np.where(y_test == class_label)[0]
            
            # Select samples: some from beginning, middle, and end for variety
            if len(class_indices) >= num_samples_per_class:
                step = len(class_indices) // num_samples_per_class
                selected = class_indices[::step][:num_samples_per_class]
            else:
                selected = class_indices
            
            sample_indices[class_label] = selected
        
        return sample_indices
    
    def _analyze_class_activations(self, pd1_samples, pd2_samples, y_samples, 
                                  predictions, activations, class_label, output_dir):
        """Analyze activations for a specific class."""
        
        class_analysis = {
            'sample_count': len(pd1_samples),
            'prediction_confidence': {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions))
            },
            'pd1_channel_analysis': self._analyze_channel_patterns(pd1_samples, 'PD1'),
            'pd2_channel_analysis': self._analyze_channel_patterns(pd2_samples, 'PD2'),
            'cross_channel_correlation': self._safe_correlation(
                np.mean(pd1_samples, axis=1), 
                np.mean(pd2_samples, axis=1)
            ),
            'activation_maps': []
        }
        
        # Create activation visualization plots
        try:
            self._create_activation_plots(
                pd1_samples, pd2_samples, predictions, class_label, output_dir
            )
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Failed to create activation plots for class_{class_label}: {e}")
            class_analysis['plot_error'] = str(e)
        
        return class_analysis
    
    def _analyze_channel_patterns(self, channel_data, channel_name):
        """Analyze patterns in a specific channel (PD1 or PD2)."""
        
        # Calculate statistics across all samples and time points
        mean_signal = np.mean(channel_data, axis=0)  # Average across samples
        std_signal = np.std(channel_data, axis=0)    # Variability across samples
        
        return {
            'channel_name': channel_name,
            'signal_length': int(channel_data.shape[1]),
            'mean_amplitude': float(np.mean(channel_data)),
            'std_amplitude': float(np.std(channel_data)),
            'mean_signal_profile': mean_signal.tolist(),
            'std_signal_profile': std_signal.tolist(),
            'peak_locations': self._find_signal_peaks(mean_signal),
            'energy_distribution': self._calculate_energy_distribution(channel_data),
            'temporal_patterns': {
                'early_phase_energy': float(np.mean(channel_data[:, :channel_data.shape[1]//3])),
                'middle_phase_energy': float(np.mean(channel_data[:, channel_data.shape[1]//3:2*channel_data.shape[1]//3])),
                'late_phase_energy': float(np.mean(channel_data[:, 2*channel_data.shape[1]//3:]))
            }
        }
    
    def _find_signal_peaks(self, signal):
        """Find significant peaks in the signal."""
        try:
            # Ensure signal is 1-D array
            signal = np.asarray(signal).flatten()
            
            if len(signal) < 3:
                return {
                    'peak_indices': [],
                    'peak_values': [],
                    'num_peaks': 0
                }
            
            # Calculate threshold safely
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            
            if std_val == 0:  # Constant signal
                return {
                    'peak_indices': [],
                    'peak_values': [],
                    'num_peaks': 0
                }
            
            threshold = mean_val + 0.5 * std_val
            
            try:
                from scipy.signal import find_peaks
                peaks, properties = find_peaks(signal, height=threshold)
                return {
                    'peak_indices': peaks.tolist(),
                    'peak_values': signal[peaks].tolist(),
                    'num_peaks': len(peaks)
                }
            except ImportError:
                # Fallback if scipy not available
                peaks = []
                for i in range(1, len(signal)-1):
                    if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                        peaks.append(i)
                
                return {
                    'peak_indices': peaks,
                    'peak_values': signal[peaks].tolist() if peaks else [],
                    'num_peaks': len(peaks)
                }
                
        except Exception as e:
            if self.verbose:
                print(f"Warning: Peak finding failed: {e}")
            return {
                'peak_indices': [],
                'peak_values': [],
                'num_peaks': 0
            }
    
    def _calculate_energy_distribution(self, channel_data):
        """Calculate energy distribution across time."""
        # Calculate RMS energy in sliding windows
        window_size = max(10, channel_data.shape[1] // 10)
        num_windows = (channel_data.shape[1] - window_size) // (window_size // 2) + 1
        
        energy_windows = []
        for i in range(num_windows):
            start = i * (window_size // 2)
            end = start + window_size
            if end > channel_data.shape[1]:
                end = channel_data.shape[1]
            
            window_data = channel_data[:, start:end]
            window_energy = np.sqrt(np.mean(window_data**2))
            energy_windows.append(float(window_energy))
        
        return energy_windows
    
    def _create_activation_plots(self, pd1_samples, pd2_samples, predictions, class_label, output_dir):
        """Create visualization plots for activations."""
        
        # Set up the plot style
        plt.style.use('default')
        fig_size = (15, 10)
        
        # Create comprehensive activation plot
        fig, axes = plt.subplots(2, 3, figsize=fig_size)
        fig.suptitle(f'PD Activation Analysis - Class {int(class_label)}', fontsize=16, fontweight='bold')
        
        # 1. Average signal profiles
        axes[0, 0].plot(np.mean(pd1_samples, axis=0), label='PD1 Average', color='blue', linewidth=2)
        axes[0, 0].fill_between(range(pd1_samples.shape[1]), 
                                np.mean(pd1_samples, axis=0) - np.std(pd1_samples, axis=0),
                                np.mean(pd1_samples, axis=0) + np.std(pd1_samples, axis=0),
                                alpha=0.3, color='blue')
        axes[0, 0].plot(np.mean(pd2_samples, axis=0), label='PD2 Average', color='red', linewidth=2)
        axes[0, 0].fill_between(range(pd2_samples.shape[1]), 
                                np.mean(pd2_samples, axis=0) - np.std(pd2_samples, axis=0),
                                np.mean(pd2_samples, axis=0) + np.std(pd2_samples, axis=0),
                                alpha=0.3, color='red')
        axes[0, 0].set_title('Average Signal Profiles')
        axes[0, 0].set_xlabel('Time Points')
        axes[0, 0].set_ylabel('Signal Amplitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Individual sample overlay
        sample_indices = np.random.choice(len(pd1_samples), min(5, len(pd1_samples)), replace=False)
        for i, idx in enumerate(sample_indices):
            alpha = 0.7 - i*0.1
            axes[0, 1].plot(pd1_samples[idx], alpha=alpha, color='blue', linewidth=1)
            axes[0, 1].plot(pd2_samples[idx], alpha=alpha, color='red', linewidth=1)
        axes[0, 1].set_title(f'Individual Samples (n={len(sample_indices)})')
        axes[0, 1].set_xlabel('Time Points')
        axes[0, 1].set_ylabel('Signal Amplitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Prediction confidence distribution
        pred_flat = predictions.flatten()
        axes[0, 2].hist(pred_flat, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 2].axvline(np.mean(pred_flat), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(pred_flat):.3f}')
        axes[0, 2].set_title('Prediction Confidence Distribution')
        axes[0, 2].set_xlabel('Prediction Probability')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Cross-channel correlation heatmap
        pd1_mean = np.mean(pd1_samples, axis=1)
        pd2_mean = np.mean(pd2_samples, axis=1)
        axes[1, 0].scatter(pd1_mean, pd2_mean, alpha=0.6, s=50)
        correlation = self._safe_correlation(pd1_mean, pd2_mean)
        axes[1, 0].set_title(f'PD1 vs PD2 Correlation (r={correlation:.3f})')
        axes[1, 0].set_xlabel('PD1 Mean Amplitude')
        axes[1, 0].set_ylabel('PD2 Mean Amplitude')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Energy distribution over time
        energy_pd1 = self._calculate_energy_distribution(pd1_samples)
        energy_pd2 = self._calculate_energy_distribution(pd2_samples)
        x_energy = range(len(energy_pd1))
        axes[1, 1].plot(x_energy, energy_pd1, 'o-', label='PD1 Energy', color='blue', linewidth=2, markersize=4)
        axes[1, 1].plot(x_energy, energy_pd2, 's-', label='PD2 Energy', color='red', linewidth=2, markersize=4)
        axes[1, 1].set_title('Energy Distribution Over Time')
        axes[1, 1].set_xlabel('Time Window')
        axes[1, 1].set_ylabel('RMS Energy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Signal difference analysis
        diff_signals = pd1_samples - pd2_samples
        axes[1, 2].plot(np.mean(diff_signals, axis=0), color='purple', linewidth=2, label='Mean Difference')
        axes[1, 2].fill_between(range(diff_signals.shape[1]),
                                np.mean(diff_signals, axis=0) - np.std(diff_signals, axis=0),
                                np.mean(diff_signals, axis=0) + np.std(diff_signals, axis=0),
                                alpha=0.3, color='purple')
        axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 2].set_title('PD1 - PD2 Difference')
        axes[1, 2].set_xlabel('Time Points')
        axes[1, 2].set_ylabel('Amplitude Difference')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / f'activation_analysis_class_{int(class_label)}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"    Activation plot saved: {plot_file}")
    
    def _analyze_overall_patterns(self, class_results, output_dir):
        """Analyze patterns across all classes."""
        
        overall_patterns = {
            'cross_class_comparison': {},
            'channel_importance': {},
            'discriminative_features': {}
        }
        
        # Compare patterns between classes
        if len(class_results) == 2:  # Binary classification
            class_names = list(class_results.keys())
            class0_data = class_results[class_names[0]]
            class1_data = class_results[class_names[1]]
            
            # Compare channel patterns
            pd1_diff = np.array(class1_data['pd1_channel_analysis']['mean_signal_profile']) - \
                       np.array(class0_data['pd1_channel_analysis']['mean_signal_profile'])
            pd2_diff = np.array(class1_data['pd2_channel_analysis']['mean_signal_profile']) - \
                       np.array(class0_data['pd2_channel_analysis']['mean_signal_profile'])
            
            overall_patterns['cross_class_comparison'] = {
                'pd1_difference_profile': pd1_diff.tolist(),
                'pd2_difference_profile': pd2_diff.tolist(),
                'pd1_max_difference': float(np.max(np.abs(pd1_diff))),
                'pd2_max_difference': float(np.max(np.abs(pd2_diff))),
                'most_discriminative_channel': 'PD1' if np.max(np.abs(pd1_diff)) > np.max(np.abs(pd2_diff)) else 'PD2'
            }
            
            # Create comparison plot
            self._create_cross_class_comparison_plot(class0_data, class1_data, output_dir)
        
        return overall_patterns
    
    def _create_cross_class_comparison_plot(self, class0_data, class1_data, output_dir):
        """Create comparison plot between classes."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Cross-Class Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Get signal profiles
        pd1_class0 = np.array(class0_data['pd1_channel_analysis']['mean_signal_profile'])
        pd1_class1 = np.array(class1_data['pd1_channel_analysis']['mean_signal_profile'])
        pd2_class0 = np.array(class0_data['pd2_channel_analysis']['mean_signal_profile'])
        pd2_class1 = np.array(class1_data['pd2_channel_analysis']['mean_signal_profile'])
        
        # PD1 comparison
        axes[0, 0].plot(pd1_class0, label='Class 0', color='blue', linewidth=2)
        axes[0, 0].plot(pd1_class1, label='Class 1', color='orange', linewidth=2)
        axes[0, 0].set_title('PD1 Channel Comparison')
        axes[0, 0].set_xlabel('Time Points')
        axes[0, 0].set_ylabel('Mean Signal Amplitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # PD2 comparison
        axes[0, 1].plot(pd2_class0, label='Class 0', color='blue', linewidth=2)
        axes[0, 1].plot(pd2_class1, label='Class 1', color='orange', linewidth=2)
        axes[0, 1].set_title('PD2 Channel Comparison')
        axes[0, 1].set_xlabel('Time Points')
        axes[0, 1].set_ylabel('Mean Signal Amplitude')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Difference plots
        pd1_diff = pd1_class1 - pd1_class0
        pd2_diff = pd2_class1 - pd2_class0
        
        axes[1, 0].plot(pd1_diff, color='green', linewidth=2, label='Class1 - Class0')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('PD1 Difference (Class 1 - Class 0)')
        axes[1, 0].set_xlabel('Time Points')
        axes[1, 0].set_ylabel('Amplitude Difference')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(pd2_diff, color='red', linewidth=2, label='Class1 - Class0')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('PD2 Difference (Class 1 - Class 0)')
        axes[1, 1].set_xlabel('Time Points')
        axes[1, 1].set_ylabel('Amplitude Difference')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / 'cross_class_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"    Cross-class comparison plot saved: {plot_file}")
    
    def save_results(self, results, model_version):
        """Save test results to output directory."""
        # Add metadata
        results_with_metadata = {
            'metadata': {
                'model_version': model_version,
                'model_path': str(self.model_path),
                'test_data_path': str(self.test_data_path),
                'evaluation_timestamp': datetime.now().isoformat(),
                'output_directory': str(self.output_dir)
            },
            'results': results
        }
        
        # Convert numpy types for JSON serialization
        results_with_metadata = convert_numpy_types(results_with_metadata)
        
        # Save detailed results as JSON
        results_file = self.output_dir / 'test_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)

        # Save summary CSV for easy viewing
        summary_data = {
            'model_version': [model_version],
            'test_accuracy': [results['test_accuracy']],
            'test_precision': [results['test_precision']],
            'test_recall': [results['test_recall']],
            'test_f1_score': [results['test_f1_score']],
            'test_roc_auc': [results['test_roc_auc']],
            'test_samples': [results['test_samples']],
            'ms_per_sample_classify': [results['ms_per_sample_classify']],
            'predictions_per_second': [results['predictions_per_second']],
            'ms_per_sample_cwt': [results['ms_per_sample_cwt']],
            'ms_per_sample_total': [results['ms_per_sample_total']],
            'evaluation_timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / 'test_summary.csv'
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')

        if self.verbose:
            print(f"\nResults saved:")
            print(f"  Detailed: {results_file}")
            print(f"  Summary:  {summary_file}")

        return results_file, summary_file

    def load_test_data_from_variant(self, dataset_variant, model, classifier_type):
        """
        Load test data from a dataset variant CSV when test_set_data.pkl is absent.
        Saves the resulting pkl so subsequent runs can use the fast path.

        Args:
            dataset_variant: Dataset variant name (used by load_dataset_variant_info)
            model: Loaded Keras model (needed for input shape)
            classifier_type: 'cwt_image' or 'pd_signal'

        Returns:
            (X_test, y_test, test_files, classifier_type)
        """
        print(f"Loading test data from dataset variant: {dataset_variant}")
        dataset_info = load_dataset_variant_info(dataset_variant)
        test_csv = dataset_info['dataset_dir'] / 'test.csv'
        if not test_csv.exists():
            raise FileNotFoundError(f"Test CSV not found: {test_csv}")

        df_test = pd.read_csv(test_csv, encoding='utf-8')

        # Get data_dir from dataset_config.json saved alongside the model
        dataset_config_path = self.output_dir.parent / 'dataset_config.json'
        if dataset_config_path.exists():
            with open(dataset_config_path, 'r', encoding='utf-8') as f:
                dataset_config = json.load(f)
            data_dir = dataset_config.get('data_dir')
        else:
            data_dir = dataset_info['config'].get('data_dir')
        if not data_dir:
            raise ValueError("Cannot determine data directory from dataset variant or dataset_config.json")

        is_multi_channel = isinstance(data_dir, dict)
        channel_paths = list(data_dir.values()) if is_multi_channel else [data_dir]

        if is_multi_channel:
            print(f"Loading {len(channel_paths)}-channel test images")
        else:
            print(f"Loading test images from: {data_dir}")

        test_files, test_labels = [], []
        for _, row in df_test.iterrows():
            filename = row['filename']
            label = int(row['has_porosity'])
            check_path = Path(channel_paths[0]) / filename if is_multi_channel else Path(data_dir) / filename
            if check_path.exists():
                test_files.append(filename if is_multi_channel else str(check_path))
                test_labels.append(label)

        if not test_files:
            raise ValueError("No test files found matching CSV entries")
        print(f"Found {len(test_files)} test images")

        test_files_arr = np.array(test_files)
        test_labels_arr = np.array(test_labels)

        if classifier_type == 'cwt_image':
            img_shape = model.input_shape
            img_height, img_width = img_shape[1], img_shape[2]
            img_channels = img_shape[3] if len(img_shape) > 3 else 1
            X_test, y_test_filtered, test_files_filtered = load_cwt_test_images(
                test_files_arr, test_labels_arr, img_width, img_height, img_channels,
                channel_paths=channel_paths if is_multi_channel else None,
                verbose=self.verbose,
            )
        else:
            img_width = model.input_shape[0][1]
            X_test, y_test_filtered, test_files_filtered = load_pd_test_images(
                test_files_arr, test_labels_arr, img_width, verbose=self.verbose)

        test_data = {
            'X_test':           X_test,
            'y_test':           np.array(y_test_filtered),
            'test_files':       test_files_filtered,
            'classifier_type':  classifier_type,
            'dataset_variant':  dataset_variant,
        }
        pkl_path = self.output_dir.parent / 'test_set_data.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(test_data, f)
        print(f"Saved test set definition to: {pkl_path}")

        return X_test, np.array(y_test_filtered), test_files_filtered, classifier_type

    def _load_val_threshold(self, version_str, classifier_type):
        """Return the val-set optimised threshold saved during training, or None if absent."""
        try:
            log_path = (get_cwt_experiment_log_path() if classifier_type == 'cwt_image'
                        else get_pd_experiment_log_path())
            df = pd.read_csv(log_path, encoding='utf-8')
            row = df[df['version'] == version_str]
            if not row.empty and 'best_val_threshold' in row.columns:
                val = row.iloc[-1]['best_val_threshold']
                if pd.notna(val):
                    return float(val)
        except Exception:
            pass
        return None

    def run_evaluation(self, model_version):
        """Run complete model evaluation pipeline."""
        # To add Grad-CAM: from gradcam_utils import generate_comprehensive_gradcam_analysis
        try:
            # Load model and test data
            model = self.load_model()
            X_test, y_test, test_files, classifier_type = self.load_test_data()

            # Prepare data
            X_test, y_test = self.prepare_test_data(X_test, y_test)

            # Evaluate model
            results = self.evaluate_model(model, X_test, y_test)

            y_true_arr = np.array(results['predictions']['y_true'])
            y_pred_arr = np.array(results['predictions']['y_pred'])
            y_proba_arr = np.array(results['predictions']['y_pred_proba'])

            if test_files is not None:
                from visualize_track_predictions import (
                    generate_track_predictions_viz,
                    generate_confusion_matrix,
                )

                # Track prediction figures — same path as final_model_trainer
                generate_track_predictions_viz(
                    test_files=test_files,
                    y_true=y_true_arr,
                    y_pred=y_pred_arr,
                    output_dir=self.output_dir.parent,
                    version=model_version,
                    use_time_labels=True,
                    unlabelled=False,
                    y_proba=y_proba_arr,
                )

                # Confusion matrix figure — same path and filename as final_model_trainer
                class_labels = ['Conduct', 'Keyhole'] if classifier_type == 'pd_signal' else ['No Porosity', 'Porosity']
                generate_confusion_matrix(
                    y_true=y_true_arr,
                    y_pred=y_pred_arr,
                    output_dir=self.output_dir.parent,
                    version=model_version,
                    threshold=0.4,
                    test_files=test_files,
                    class_labels=class_labels,
                    subdir='test_evaluation',
                )

                # Save test_predictions_{version}.pkl so visualize_track_predictions.py
                # --version mode works without re-running final_model_trainer
                y_pred_proba = np.array(results['predictions']['y_pred_proba'])
                predictions_pkl = self.output_dir / f'test_predictions_{model_version}.pkl'
                with open(predictions_pkl, 'wb') as f:
                    pickle.dump({
                        'y_pred':          y_pred_arr,
                        'y_proba':         y_pred_proba,
                        'y_true':          y_true_arr,
                        'best_threshold':  0.4,
                        'test_files':      test_files,
                        'classifier_type': classifier_type,
                    }, f)

            # Run activation analysis for PD models
            activation_results = self.analyze_pd_activations(model, X_test, y_test, num_samples_per_class=10)
            if activation_results is not None:
                results['activation_analysis'] = activation_results

            # Save results
            results_files = self.save_results(results, model_version)

            return results, results_files

        except Exception as e:
            raise Exception(f"Evaluation failed: {e}")

    def run_full_evaluation(self, model_version, model, X_test, y_test, test_files, classifier_type, gradcam=False):
        """
        Complete evaluation: threshold optimization, Grad-CAM, P-V map, classification report,
        and all standard outputs. Produces the same file set as final_model_trainer's
        evaluate_with_threshold_optimization(), so results are directly comparable.

        Args:
            model_version: Version string (e.g. 'v229')
            model: Loaded Keras model
            X_test: Test images / dual-branch list
            y_test: Ground-truth labels (np.ndarray)
            test_files: List of filenames for visualization
            classifier_type: 'cwt_image' or 'pd_signal'
        """
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                     f1_score as sk_f1, roc_auc_score, classification_report as sk_report)
        from visualize_track_predictions import generate_track_predictions_viz, generate_confusion_matrix

        test_eval_dir = self.output_dir  # already points to version/test_evaluation

        # --- Predictions ---
        if classifier_type == 'cwt_image':
            y_proba = model.predict(X_test, verbose=0)
        else:
            pd1, pd2 = X_test if isinstance(X_test, (list, tuple)) else (X_test, X_test)
            y_proba = model.predict([pd1, pd2], verbose=0)
        y_proba_flat = y_proba.flatten()

        # --- Load val-set threshold from training log ---
        best_threshold = self._load_val_threshold(model_version, classifier_type)
        if best_threshold is None:
            print("Warning: val-set threshold not found in log; falling back to 0.4")
            best_threshold = 0.4
        else:
            print(f"Using val-set threshold from training: {best_threshold:.2f}")

        y_pred = (y_proba_flat >= best_threshold).astype(int)

        best_result = {
            'threshold': best_threshold,
            'accuracy':  float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall':    float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score':  float(sk_f1(y_test, y_pred, zero_division=0)),
        }
        print(f"Threshold {best_threshold:.2f}  F1={best_result['f1_score']:.4f}  "
              f"Acc={best_result['accuracy']:.4f}")

        try:
            auc_score = float(roc_auc_score(y_test, y_proba_flat))
        except ValueError:
            auc_score = None

        # --- test_predictions pkl (same format as final_model_trainer) ---
        predictions_pkl = test_eval_dir / f'test_predictions_{model_version}.pkl'
        with open(predictions_pkl, 'wb') as f:
            pickle.dump({
                'y_pred':          y_pred,
                'y_proba':         y_proba_flat,
                'y_true':          y_test,
                'best_threshold':  best_threshold,
                'test_files':      test_files,
                'classifier_type': classifier_type,
            }, f)

        # --- Classification report ---
        report_str = sk_report(y_test, y_pred)
        report_path = test_eval_dir / f'classification_report_{model_version}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Classification Report — Threshold: {best_threshold:.2f}\n{'='*50}\n")
            f.write(report_str)
            f.write(f"\nAUC: {auc_score:.4f}" if auc_score else "\nAUC: N/A")

        # --- Confusion matrix (identical path/filename to trainer) ---
        class_labels = ['Conduct', 'Keyhole'] if classifier_type == 'pd_signal' else ['No Porosity', 'Porosity']
        generate_confusion_matrix(
            y_true=y_test, y_pred=y_pred,
            output_dir=test_eval_dir.parent,
            version=model_version, threshold=best_threshold,
            test_files=test_files, class_labels=class_labels,
            subdir='test_evaluation',
        )

        # --- Track prediction figures (identical path to trainer) ---
        if test_files is not None:
            generate_track_predictions_viz(
                test_files=test_files, y_true=y_test, y_pred=y_pred,
                output_dir=test_eval_dir.parent,
                version=model_version, use_time_labels=True, unlabelled=False,
                y_proba=y_proba_flat,
            )

        # --- Grad-CAM (CWT only, --full mode) ---
        gradcam_results = None
        if gradcam and classifier_type == 'cwt_image':
            try:
                from gradcam_utils import generate_comprehensive_gradcam_analysis
                num_channels = X_test.shape[-1] if hasattr(X_test, 'shape') and len(X_test.shape) == 4 else 1
                channel_labels = [f'Channel_{i+1}' for i in range(num_channels)] if num_channels > 1 else None
                gradcam_results = generate_comprehensive_gradcam_analysis(
                    model, X_test, y_test, y_pred, y_proba_flat,
                    best_threshold, test_eval_dir, model_version, test_files,
                    channel_labels=channel_labels,
                )
            except Exception as e:
                print(f"Warning: Grad-CAM failed: {e}")

        # --- P-V map ---
        pv_map_results = None
        if test_files is not None:
            try:
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).parent.parent))
                from tools import generate_pv_map, get_logbook
                test_trackids = sorted({
                    f"{Path(fp).name.split('_')[0]}_{Path(fp).name.split('_')[1]}"
                    for fp in test_files if len(Path(fp).name.split('_')) >= 2
                })
                logbook = get_logbook()
                mask = (
                    (logbook['Substrate material'] == 'AlSi10Mg') &
                    (logbook['Layer'] == 1) &
                    (logbook['Point jump delay [us]'] == 0) &
                    (logbook['Powder material'] != 'None')
                )
                bg_trackids = logbook[mask]['trackid'].unique().tolist()
                output_path = test_eval_dir / f'pv_map_test_set_{model_version}.png'
                generate_pv_map(
                    trackids=bg_trackids, output_path=output_path,
                    highlight_trackids=test_trackids,
                )
                pv_map_results = {'unique_tracks': len(test_trackids), 'output_file': str(output_path)}
                print(f"P-V map saved: {output_path}")
            except Exception as e:
                print(f"Warning: P-V map generation failed: {e}")

        # --- PD activation analysis ---
        activation_results = None
        if classifier_type == 'pd_signal':
            activation_results = self.analyze_pd_activations(model, X_test, y_test, num_samples_per_class=10)

        # --- Comprehensive JSON ---
        from sklearn.metrics import confusion_matrix as sk_cm
        cm = sk_cm(y_test, y_pred)
        eval_results = {
            'version':               model_version,
            'classifier_type':       classifier_type,
            'test_samples':          int(len(y_test)),
            'best_threshold':        best_threshold,
            'best_metrics':          best_result,
            'auc_score':             auc_score,
            'confusion_matrix':      cm.tolist(),
            'classification_report': sk_report(y_test, y_pred, output_dict=True),
            'gradcam_results':       gradcam_results,
            'activation_results':    activation_results,
            'pv_map':                pv_map_results,
            'output_files': {
                'confusion_matrix':      str(test_eval_dir / f'confusion_matrix_{model_version}.png'),
                'classification_report': str(report_path),
            },
        }
        json_path = test_eval_dir / f'comprehensive_evaluation_{model_version}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(eval_results), f, indent=2, default=str)

        # --- Standard quick-eval outputs too ---
        self.save_results({
            'test_accuracy':          best_result['accuracy'],
            'test_precision':         best_result['precision'],
            'test_recall':            best_result['recall'],
            'test_f1_score':          best_result['f1_score'],
            'test_roc_auc':           auc_score,
            'test_samples':           int(len(y_test)),
            'confusion_matrix':       cm.tolist(),
            'classification_report':  sk_report(y_test, y_pred, output_dict=True),
            'predictions': {
                'y_true':       y_test.tolist(),
                'y_pred':       y_pred.tolist(),
                'y_pred_proba': y_proba_flat.tolist(),
            },
            'inference_time_seconds':   0.0,
            'ms_per_sample_classify':   0.0,
            'predictions_per_second':   0.0,
            'ms_per_sample_cwt':        None,
            'ms_per_sample_total':      0.0,
        }, model_version)

        print(f"Full evaluation complete. Results saved to: {test_eval_dir}")
        return eval_results

    def load_images_from_dir(self, image_dir, model):
        """Load unlabelled PNG images from a flat directory using the training preprocessing pipeline."""
        from PIL import Image as PILImage

        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(
                f"Directory not found: {image_dir}\n"
                f"  Tip: in Git Bash use forward slashes (F:/path/to/dir) or "
                f"double-quote backslash paths (\"F:\\path\\to\\dir\")"
            )
        paths = sorted(image_dir.glob('*.png'))
        if not paths:
            raise FileNotFoundError(f"No PNG files found in {image_dir}")

        # Derive resize target from model input shape (H, W)
        H, W = model.input_shape[1], model.input_shape[2]

        images, filenames = [], []
        for p in paths:
            try:
                img = np.array(PILImage.open(p).convert('L'))
            except Exception as e:
                print(f"  Warning: could not load {p.name}: {e}, skipping")
                continue
            img = PILImage.fromarray(img).resize((W, H), PILImage.LANCZOS)
            img = normalize_image(np.array(img))        # float32 [0,1]
            img = np.expand_dims(img, axis=-1)          # (H, W, 1)
            images.append(img)
            filenames.append(p.name)

        X = np.stack(images, axis=0)                    # (N, H, W, 1)
        print(f"Loaded {len(filenames)} images  shape={X.shape}")
        return X, filenames

    def predict_unlabelled(self, model, X, filenames):
        """Run model on unlabelled images; return DataFrame with per-image predictions."""
        y_pred_proba = model.predict(X, verbose=0)

        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            # Softmax multi-class
            y_pred      = np.argmax(y_pred_proba, axis=1)
            confidence  = np.max(y_pred_proba, axis=1)
        else:
            # Sigmoid binary
            proba_flat  = y_pred_proba.flatten()
            y_pred      = (proba_flat > 0.5).astype(int)
            confidence  = np.where(y_pred == 1, proba_flat, 1.0 - proba_flat)

        return pd.DataFrame({
            'filename':        filenames,
            'predicted_class': y_pred.tolist(),
            'confidence':      confidence.tolist(),
        })

    def generate_track_figures(self, df, output_dir, vote_windows=False):
        """Generate per-track prediction box figures from unlabelled predictions DataFrame."""
        from visualize_track_predictions import generate_track_predictions_viz
        filenames = df['filename'].tolist()
        y_pred    = df['predicted_class'].to_numpy()
        generate_track_predictions_viz(
            test_files=filenames,
            y_true=y_pred,       # unused when unlabelled=True
            y_pred=y_pred,
            output_dir=Path(output_dir),
            version='unlabelled',
            use_time_labels=True,
            unlabelled=True,
            vote_windows=vote_windows,
        )

    def save_predictions(self, df, output_dir, model_info=None):
        """Save unlabelled predictions to CSV and print a brief summary."""
        import json
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / 'predictions.csv'
        df.to_csv(out_path, index=False, encoding='utf-8')

        counts = df['predicted_class'].value_counts().sort_index()
        print(f"\nPredictions saved: {out_path}")
        print(f"  Total images : {len(df)}")
        for cls, n in counts.items():
            mean_conf = df.loc[df['predicted_class'] == cls, 'confidence'].mean()
            print(f"  Class {cls}      : {n} images  (mean confidence {mean_conf:.3f})")

        if model_info is not None:
            info_path = output_dir / 'model_info.json'
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2)
            print(f"  Model info   : {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained model on test data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Usage:\n"
            "  python ml/model_tester.py --version 229\n"
            "  python ml/model_tester.py --version 229 --full\n"
            "  python ml/model_tester.py --version 229 --full --dataset_variant MyVariant\n"
            "  python ml/model_tester.py --version 229 --classifier_type pd_signal\n"
            "  python ml/model_tester.py --version 229 --image_dir /path/to/images\n\n"
            "Paths resolve from the version folder. Use --model/--test_data/--output_dir to override."
        )
    )
    parser.add_argument('--version', type=str, default=None,
                        help='Model version (e.g. 229 or v229).')
    parser.add_argument('--classifier_type', type=str, default='cwt_image',
                        choices=['cwt_image', 'pd_signal'],
                        help='Classifier type — determines which output directory to search (default: cwt_image).')
    parser.add_argument('--model', type=str, default=None,
                        help='Override: explicit path to trained model (.h5 or .keras).')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Override: explicit path to test data pickle.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override: output directory for results.')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output.')
    parser.add_argument('--full', action='store_true', default=False,
                        help='Add Grad-CAM analysis on top of the standard evaluation. '
                             'All other outputs (threshold optimisation, P-V map, classification report) '
                             'are produced by default.')
    parser.add_argument('--dataset_variant', type=str, default=None,
                        help='Dataset variant name. When test_set_data.pkl is absent, loads test '
                             'data from the variant CSV and saves the pkl for future runs.')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory of unlabelled PNG images to predict (skips ground-truth evaluation). '
                             'On Windows in Git Bash, use forward slashes (F:/path/to/dir) or '
                             'double-quote the path to preserve backslashes.')
    parser.add_argument('--vote_windows', action='store_true', default=False,
                        help='Show majority-voted label per offset step in track figures.')

    args = parser.parse_args()

    if args.version is None and args.model is None:
        parser.error('Provide --version (e.g. --version 229) or explicit --model / --test_data paths.')

    version_str = format_version(args.version) if args.version else None

    if version_str:
        resolved = _resolve_paths_from_version(version_str, args.classifier_type)
        model_path     = args.model      or resolved['model']
        test_data_path = args.test_data  or resolved['test_data']
        output_dir     = args.output_dir or resolved['output_dir']
    else:
        missing = [n for n, v in [('--test_data', args.test_data), ('--output_dir', args.output_dir)] if v is None]
        if missing:
            parser.error(f'Without --version you must also supply: {", ".join(missing)}')
        model_path     = args.model
        test_data_path = args.test_data
        output_dir     = args.output_dir
        version_str    = Path(model_path).stem

    try:
        model_path     = Path(model_path)
        test_data_path = Path(test_data_path) if test_data_path else None
        output_dir     = Path(output_dir)
        image_dir      = Path(args.image_dir) if args.image_dir else None

        output_dir.mkdir(parents=True, exist_ok=True)

        tester = ModelTester(
            model_path=model_path,
            test_data_path=test_data_path,
            output_dir=output_dir,
            verbose=args.verbose,
        )

        # --- Unlabelled prediction mode ---
        if image_dir:
            from datetime import datetime
            model = tester.load_model()
            X, filenames = tester.load_images_from_dir(image_dir, model)
            df = tester.predict_unlabelled(model, X, filenames)

            pred_dir = image_dir / 'predictions'
            pred_dir.mkdir(parents=True, exist_ok=True)

            counts = df['predicted_class'].value_counts().sort_index()
            model_info = {
                'model_path':    str(model_path),
                'model_version': version_str,
                'timestamp':     datetime.now().isoformat(timespec='seconds'),
                'image_dir':     str(image_dir),
                'n_images':      len(df),
                'class_counts':  {str(k): int(v) for k, v in counts.items()},
                'mean_confidence_per_class': {
                    str(cls): float(df.loc[df['predicted_class'] == cls, 'confidence'].mean())
                    for cls in counts.index
                },
            }
            tester.save_predictions(df, pred_dir, model_info=model_info)
            tester.generate_track_figures(df, pred_dir, vote_windows=args.vote_windows)
            print(f"\nDone. Results saved to: {pred_dir}")
            return

        # --- Ground-truth evaluation modes ---
        model = tester.load_model()

        # If pkl is absent and --dataset_variant given, build it from CSV
        if test_data_path is not None and not test_data_path.exists():
            if args.dataset_variant:
                X_test, y_test, test_files, classifier_type = tester.load_test_data_from_variant(
                    args.dataset_variant, model, args.classifier_type)
                # pkl now saved; update tester's test_data_path so run_evaluation can find it
                tester.test_data_path = output_dir.parent / 'test_set_data.pkl'
            else:
                parser.error(
                    f"Test data not found: {test_data_path}\n"
                    f"Supply --dataset_variant to load from CSV, or point --test_data at an existing pkl."
                )

        if 'X_test' not in dir():
            X_test, y_test, test_files, classifier_type = tester.load_test_data()
            X_test, y_test = tester.prepare_test_data(X_test, y_test)
        results = tester.run_full_evaluation(
            version_str, model, X_test, y_test, test_files, classifier_type,
            gradcam=args.full,
        )
        print(f"\nEvaluation complete.  Accuracy: {results['best_metrics']['accuracy']:.4f}")
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        print(f"\n❌ Model evaluation failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()