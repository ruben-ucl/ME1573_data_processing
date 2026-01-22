#!/usr/bin/env python3
"""Test complexity calculation accuracy vs real Keras model."""

import sys
import os
os.chdir('D:/ME1573_data_processing')
sys.path.insert(0, 'D:/ME1573_data_processing/ml')

from hyperparameter_tuner import HyperparameterTuner
from hyperparameter_registry import get_default_config

# CWT Test
print("="*60)
print("CWT IMAGE CLASSIFIER")
print("="*60)

tuner_cwt = HyperparameterTuner(classifier_type='cwt_image')
config_cwt = tuner_cwt.base_config

print(f"\nConfig:")
print(f"  conv_filters: {config_cwt['conv_filters']}")
print(f"  dense_units: {config_cwt['dense_units']}")
print(f"  img_width x img_height x channels: {config_cwt['img_width']} x {config_cwt['img_height']} x {config_cwt['img_channels']}")
print(f"  pool_layers: {config_cwt.get('pool_layers', 'N/A')}")
print(f"  pool_size: {config_cwt.get('pool_size', 'N/A')}")

calculated = tuner_cwt.calculate_model_complexity(config_cwt)
print(f"\nCalculated complexity: {calculated:,} parameters")
print(f"Real Keras complexity: 9,738,609 parameters (from timing_database.json)")
print(f"Error: {(calculated - 9738609) / 9738609 * 100:+.1f}%")

# Let's build the actual model to see
print("\n" + "="*60)
print("Building actual Keras model...")
print("="*60)

from CWT_image_classifier_v3 import build_model

model = build_model(
    input_shape=(config_cwt['img_height'], config_cwt['img_width'], config_cwt['img_channels']),
    conv_filters=config_cwt['conv_filters'],
    dense_units=config_cwt['dense_units'],
    conv_dropout=config_cwt['conv_dropout'],
    dense_dropout=config_cwt['dense_dropout'],
    l2_reg=config_cwt['l2_regularization'],
    use_batch_norm=config_cwt['use_batch_norm'],
    pool_layers=config_cwt.get('pool_layers', [2, 5]),
    pool_size=config_cwt.get('pool_size', [2, 2]),
    conv_kernel_size=config_cwt.get('conv_kernel_size', [3, 3]),
    num_classes=2,
    verbose=False
)

real_params = model.count_params()
print(f"\nReal Keras model: {real_params:,} parameters")
print(f"Calculated: {calculated:,} parameters")
print(f"Error: {(calculated - real_params) / real_params * 100:+.1f}%")
