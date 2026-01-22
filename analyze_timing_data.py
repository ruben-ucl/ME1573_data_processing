#!/usr/bin/env python3
"""Analyze timing database to understand the relationship."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load CWT timing data
with open('D:/ME1573_data_processing/ml/logs/cwt/hyperopt_results/timing_database.json', 'r') as f:
    data = json.load(f)

records = data['records']

# Extract data
complexities = [r['complexity'] for r in records]
times = [r['time'] for r in records]
batch_factors = [r['batch_factor'] for r in records]
fold_factors = [r['fold_factor'] for r in records]

# Normalize times by batch_factor to see raw complexity relationship
normalized_times = [t / (bf * ff) for t, bf, ff in zip(times, batch_factors, fold_factors)]

print("="*60)
print("TIMING DATA ANALYSIS")
print("="*60)

print(f"\nDataset: {len(records)} records")
print(f"Complexity: All {complexities[0]:,} parameters (constant)")
print(f"Time range: {min(times):.1f} - {max(times):.1f} minutes")
print(f"Mean time: {np.mean(times):.1f} minutes")
print(f"Std time: {np.std(times):.1f} minutes")

print(f"\nBatch factors: {sorted(set(batch_factors))}")
print(f"Fold factors: {sorted(set(fold_factors))}")

# Group by batch_factor to see the pattern
from collections import defaultdict
by_batch = defaultdict(list)
for t, bf in zip(times, batch_factors):
    by_batch[bf].append(t)

print("\n" + "="*60)
print("TIME BY BATCH FACTOR")
print("="*60)
for bf in sorted(by_batch.keys()):
    times_list = by_batch[bf]
    print(f"\nBatch factor {bf} (batch_size={16/bf:.0f}):")
    print(f"  Samples: {len(times_list)}")
    print(f"  Mean time: {np.mean(times_list):.1f} ± {np.std(times_list):.1f} min")
    print(f"  Range: {min(times_list):.1f} - {max(times_list):.1f} min")

# Check what SimpleTimingEstimator would predict
print("\n" + "="*60)
print("SIMPLE TIMING ESTIMATOR PREDICTIONS")
print("="*60)

# Test config: batch_size=16, k_folds=5
test_config = {'batch_size': 16, 'k_folds': 5}
complexity_real = 9738609

from scipy.stats import linregress

# Fit power law on normalized times
if len(records) >= 3:
    # All same complexity, so can't fit power law to complexity
    # But we can see the mean normalized time
    print(f"\nAll records have same complexity: {complexity_real:,}")
    print(f"Mean normalized time (time / (batch_factor * fold_factor)): {np.mean(normalized_times):.2f} min")
    print(f"This means SimpleTimingEstimator uses power law fitted to DIFFERENT complexity models")

# What does SimpleTimingEstimator predict?
import sys
sys.path.insert(0, 'D:/ME1573_data_processing/ml')
from simple_timing_estimator import SimpleTimingEstimator

estimator = SimpleTimingEstimator(
    'D:/ME1573_data_processing/ml/logs/cwt/hyperopt_results/timing_database.json',
    'cwt_image'
)

# Predict for batch_size=16
for bs in [16, 32, 64, 128]:
    config = {'batch_size': bs, 'k_folds': 5}
    predicted = estimator.estimate_time(config, real_complexity=complexity_real)
    print(f"\nPredicted for batch_size={bs}: {predicted:.2f} min")

    # Find actual times for this batch size
    bf = 16.0 / bs
    actual_times = by_batch.get(bf, [])
    if actual_times:
        actual_mean = np.mean(actual_times)
        print(f"  Actual mean: {actual_mean:.2f} min")
        print(f"  Error: {(predicted - actual_mean) / actual_mean * 100:+.1f}%")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Time vs record index (time series)
ax = axes[0, 0]
ax.scatter(range(len(times)), times, alpha=0.6, s=50)
ax.set_xlabel('Record Index')
ax.set_ylabel('Training Time (minutes)')
ax.set_title('Training Time Over Time')
ax.grid(True, alpha=0.3)

# Plot 2: Time distribution by batch factor
ax = axes[0, 1]
batch_factor_labels = {0.125: 'BS=128', 0.25: 'BS=64', 0.5: 'BS=32', 1.0: 'BS=16'}
data_by_batch = []
labels = []
for bf in sorted(by_batch.keys()):
    data_by_batch.append(by_batch[bf])
    labels.append(batch_factor_labels.get(bf, f'BF={bf}'))

bp = ax.boxplot(data_by_batch, labels=labels)
ax.set_ylabel('Training Time (minutes)')
ax.set_title('Training Time by Batch Size')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Normalized time (should be more consistent)
ax = axes[1, 0]
ax.scatter(range(len(normalized_times)), normalized_times, alpha=0.6, s=50, c='green')
ax.axhline(y=np.mean(normalized_times), color='r', linestyle='--', label=f'Mean: {np.mean(normalized_times):.1f}')
ax.set_xlabel('Record Index')
ax.set_ylabel('Normalized Time (time / batch_factor)')
ax.set_title('Normalized Training Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Time vs batch_factor scatter
ax = axes[1, 1]
colors = {'0.125': 'red', '0.25': 'orange', '0.5': 'blue', '1.0': 'green'}
for bf in sorted(set(batch_factors)):
    bf_times = [t for t, b in zip(times, batch_factors) if b == bf]
    bf_indices = [i for i, b in enumerate(batch_factors) if b == bf]
    ax.scatter([bf] * len(bf_times), bf_times, alpha=0.6, s=50,
               label=f'{batch_factor_labels.get(bf, f"BF={bf}")}')
ax.set_xlabel('Batch Factor (16 / batch_size)')
ax.set_ylabel('Training Time (minutes)')
ax.set_title('Time vs Batch Factor')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/ME1573_data_processing/timing_analysis.png', dpi=150)
print(f"\n{'='*60}")
print(f"Visualization saved to: timing_analysis.png")
print(f"{'='*60}")

plt.show()
