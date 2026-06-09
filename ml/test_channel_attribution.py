#!/usr/bin/env python3
"""Quick test script for channel attribution feature"""

import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import sys
sys.path.insert(0, 'D:/ME1573_data_processing/ml')

import numpy as np
import tensorflow as tf
from pathlib import Path
from gradcam_utils import generate_comprehensive_gradcam_analysis

# Load the model from v216
model_path = Path('D:/ME1573_data_processing/ml/outputs/cwt/v216/best_model_v216.h5')
print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# Load a small subset of validation data from fold 2 (which had best results)
# For testing, we'll create dummy multi-channel data
print("\nCreating test data...")
n_samples = 20
H, W, C = 256, 100, 4
X_test = np.random.rand(n_samples, H, W, C).astype(np.float32) * 0.5 + 0.25  # Random images
y_test = np.random.randint(0, 2, n_samples)  # Random labels
y_pred = np.random.randint(0, 2, n_samples)  # Random predictions
y_proba = np.random.rand(n_samples)  # Random probabilities

print(f"Test data shape: {X_test.shape}")
print(f"Number of channels: {C}")

# Define channel labels
channel_labels = ['PD1_cmor1.5-1.0', 'PD1_mexh', 'PD2_cmor1.5-1.0', 'PD2_mexh']
print(f"Channel labels: {channel_labels}")

# Create output directory
output_dir = Path('D:/ME1573_data_processing/test_channel_attribution_output')
output_dir.mkdir(exist_ok=True)

# Run comprehensive GradCAM analysis with channel attribution
print("\n🔥 Running comprehensive Grad-CAM analysis with channel attribution...")
try:
    results = generate_comprehensive_gradcam_analysis(
        model=model,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        threshold=0.5,
        output_dir=output_dir,
        version='test',
        test_files=None,
        channel_labels=channel_labels
    )

    print("\n✅ SUCCESS! GradCAM analysis completed")
    print(f"\nResults summary:")
    print(f"  Saved images: {results.get('saved_images', 0)}")
    print(f"  Target layer: {results.get('target_layer', 'N/A')}")

    if 'channel_attribution' in results:
        print(f"\n📊 Channel attribution results:")
        ch_attr = results['channel_attribution']
        print(f"  Number of samples: {ch_attr['num_samples']}")
        print(f"  Number of channels: {ch_attr['num_channels']}")
        print(f"  Dominant channel: {ch_attr['dominant_channel']} (score: {ch_attr['dominant_channel_score']:.4f})")
        print(f"\n  Channel ranking:")
        for i, (ch, score) in enumerate(zip(ch_attr['channel_ranking'], ch_attr['mean_attributions'])):
            print(f"    {i+1}. {ch}: {score:.4f}")

        print(f"\n  Output files:")
        for key, path in ch_attr['output_files'].items():
            print(f"    {key}: {path}")
    else:
        print("\n⚠️  Warning: No channel attribution in results")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test completed. Check output directory for results:")
print(f"  {output_dir}")
