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
from datetime import datetime

# TensorFlow setup
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

# Local imports
from config import convert_numpy_types
from data_utils import normalize_image

class ModelTester:
    """Evaluates trained models on test data."""
    
    def __init__(self, model_path, test_data_path, output_dir, verbose=False):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
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
            
            if self.verbose:
                print(f"Test data loaded: {len(X_test)} samples")
                print(f"Test data shape: {X_test.shape if hasattr(X_test, 'shape') else 'Multiple arrays'}")
                print(f"Class distribution: {np.bincount(y_test)}")
                
            return X_test, y_test
            
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
        
        # Get predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        
        # Handle softmax output (n_samples, n_classes) vs sigmoid output (n_samples, 1)
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            # Multi-class softmax output: take argmax for predictions, class 1 prob for metrics
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_pred_proba_binary = y_pred_proba[:, 1]  # Probability of class 1 for ROC AUC
        else:
            # Binary sigmoid output: use threshold
            y_pred_proba_binary = y_pred_proba.flatten()
            y_pred = (y_pred_proba_binary > 0.5).astype(int)
        
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
            }
        }
        
        if self.verbose:
            print(f"\nTest Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            if roc_auc is not None:
                print(f"  ROC AUC: {roc_auc:.4f}")
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
        
        # Save as JSON
        results_file = self.output_dir / 'test_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)
        
        # Save as pickle for easy loading
        pickle_file = self.output_dir / 'test_results.pkl'
        with open(pickle_file, 'wb') as f:
            pickle.dump(results_with_metadata, f)
        
        # Save summary CSV for easy viewing
        summary_data = {
            'model_version': [model_version],
            'test_accuracy': [results['test_accuracy']],
            'test_precision': [results['test_precision']],
            'test_recall': [results['test_recall']],
            'test_f1_score': [results['test_f1_score']],
            'test_roc_auc': [results['test_roc_auc']],
            'test_samples': [results['test_samples']],
            'evaluation_timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / 'test_summary.csv'
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        
        if self.verbose:
            print(f"\nResults saved:")
            print(f"  Detailed: {results_file}")
            print(f"  Pickle: {pickle_file}")
            print(f"  Summary: {summary_file}")
        
        return results_file, pickle_file, summary_file
    
    def run_evaluation(self, model_version):
        """Run complete model evaluation pipeline."""
        try:
            # Load model and test data
            model = self.load_model()
            X_test, y_test = self.load_test_data()
            
            # Prepare data
            X_test, y_test = self.prepare_test_data(X_test, y_test)
            
            # Evaluate model
            results = self.evaluate_model(model, X_test, y_test)
            
            # Run activation analysis for PD models
            activation_results = self.analyze_pd_activations(model, X_test, y_test, num_samples_per_class=10)
            if activation_results is not None:
                results['activation_analysis'] = activation_results
            
            # Save results
            results_files = self.save_results(results, model_version)
            
            return results, results_files
            
        except Exception as e:
            raise Exception(f"Evaluation failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on test data')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file (.h5 or .keras)')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data pickle file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--model_version', type=str, required=True,
                       help='Model version identifier')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    try:
        # Initialize tester
        tester = ModelTester(
            model_path=args.model,
            test_data_path=args.test_data,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        # Run evaluation
        results, result_files = tester.run_evaluation(args.model_version)
        
        print(f"\n✅ Model evaluation completed successfully!")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Results saved to: {args.output_dir}")
        
        # Print activation analysis summary if available
        if 'activation_analysis' in results:
            activation_data = results['activation_analysis']
            print(f"\n🔍 Activation Analysis Summary:")
            
            if 'overall_patterns' in activation_data and 'cross_class_comparison' in activation_data['overall_patterns']:
                comparison = activation_data['overall_patterns']['cross_class_comparison']
                if 'most_discriminative_channel' in comparison:
                    channel = comparison['most_discriminative_channel']
                    pd1_diff = comparison.get('pd1_max_difference', 0)
                    pd2_diff = comparison.get('pd2_max_difference', 0)
                    print(f"  Most discriminative channel: {channel}")
                    print(f"  PD1 max difference: {pd1_diff:.4f}")
                    print(f"  PD2 max difference: {pd2_diff:.4f}")
            
            if 'by_class' in activation_data:
                for class_name, class_data in activation_data['by_class'].items():
                    pred_conf = class_data['prediction_confidence']
                    corr = class_data['cross_channel_correlation']
                    print(f"  {class_name}: Pred confidence {pred_conf['mean']:.3f}±{pred_conf['std']:.3f}, PD1-PD2 correlation {corr:.3f}")
            
            print(f"  Detailed plots saved to: {args.output_dir}/activation_analysis/")
        
    except Exception as e:
        print(f"\n❌ Model evaluation failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()