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
        
    except Exception as e:
        print(f"\n❌ Model evaluation failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()