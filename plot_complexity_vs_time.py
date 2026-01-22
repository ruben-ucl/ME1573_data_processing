#!/usr/bin/env python3
"""Plot model complexity vs training time, grouped by batch size."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Load CWT timing data
timing_db_path = 'D:/ME1573_data_processing/ml/logs/cwt/hyperopt_results/timing_database.json'
with open(timing_db_path, 'r') as f:
    data = json.load(f)

records = data['records']

# Extract data
complexities = [r['complexity'] for r in records]
times = [r['time'] for r in records]
batch_factors = [r['batch_factor'] for r in records]

# Convert batch_factor to batch_size
batch_sizes = [int(16.0 / bf) for bf in batch_factors]

# Group by batch size
by_batch_size = defaultdict(lambda: {'complexities': [], 'times': []})
for c, t, bs in zip(complexities, times, batch_sizes):
    by_batch_size[bs]['complexities'].append(c)
    by_batch_size[bs]['times'].append(t)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Complexity vs Time (all data, colored by batch size)
ax = axes[0]
colors = {32: 'green', 64: 'orange', 128: 'blue'}
markers = {32: 'o', 64: 's', 128: '^'}

for bs in sorted(by_batch_size.keys()):
    c_vals = by_batch_size[bs]['complexities']
    t_vals = by_batch_size[bs]['times']
    ax.scatter(c_vals, t_vals,
               c=colors.get(bs, 'gray'),
               marker=markers.get(bs, 'o'),
               s=80, alpha=0.6,
               label=f'BS={bs} (n={len(t_vals)})')

ax.set_xlabel('Model Complexity (parameters)', fontsize=12)
ax.set_ylabel('Training Time (minutes)', fontsize=12)
ax.set_title('Training Time vs Model Complexity by Batch Size', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Format x-axis to show millions
ax.ticklabel_format(style='plain', axis='x')
xticks = ax.get_xticks()
ax.set_xticklabels([f'{x/1e6:.1f}M' for x in xticks])

# Plot 2: Box plots of time by batch size (to show variance)
ax = axes[1]
data_by_bs = []
labels_by_bs = []
for bs in sorted(by_batch_size.keys()):
    data_by_bs.append(by_batch_size[bs]['times'])
    labels_by_bs.append(f'BS={bs}')

bp = ax.boxplot(data_by_bs, tick_labels=labels_by_bs, patch_artist=True)

# Color boxes to match scatter plot
for patch, bs in zip(bp['boxes'], sorted(by_batch_size.keys())):
    patch.set_facecolor(colors.get(bs, 'gray'))
    patch.set_alpha(0.6)

ax.set_ylabel('Training Time (minutes)', fontsize=12)
ax.set_xlabel('Batch Size', fontsize=12)
ax.set_title('Training Time Distribution by Batch Size', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add statistics text
for i, bs in enumerate(sorted(by_batch_size.keys())):
    times = by_batch_size[bs]['times']
    median = np.median(times)
    mean = np.mean(times)
    std = np.std(times)
    ax.text(i+1, max(times) + 2, f'μ={mean:.1f}\nσ={std:.1f}',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('D:/ME1573_data_processing/complexity_vs_time.png', dpi=150, bbox_inches='tight')

print("="*60)
print("COMPLEXITY vs TIME ANALYSIS")
print("="*60)

print(f"\nTotal records: {len(records)}")
print(f"\nComplexity range: {min(complexities):,} - {max(complexities):,} parameters")
print(f"Time range: {min(times):.1f} - {max(times):.1f} minutes")

print(f"\n{'='*60}")
print("STATISTICS BY BATCH SIZE")
print("="*60)

for bs in sorted(by_batch_size.keys()):
    c_vals = by_batch_size[bs]['complexities']
    t_vals = by_batch_size[bs]['times']

    print(f"\nBatch Size {bs}:")
    print(f"  Samples: {len(t_vals)}")
    print(f"  Complexity: {min(c_vals):,} - {max(c_vals):,} parameters")
    if min(c_vals) == max(c_vals):
        print(f"              (all same: {c_vals[0]:,})")
    print(f"  Time: {min(t_vals):.1f} - {max(t_vals):.1f} min")
    print(f"  Mean: {np.mean(t_vals):.1f} ± {np.std(t_vals):.1f} min")
    print(f"  Median: {np.median(t_vals):.1f} min")
    print(f"  25th percentile: {np.percentile(t_vals, 25):.1f} min")
    print(f"  75th percentile: {np.percentile(t_vals, 75):.1f} min")

# Check if complexity varies
unique_complexities = set(complexities)
print(f"\n{'='*60}")
print("COMPLEXITY VARIATION")
print("="*60)
print(f"Unique complexity values: {len(unique_complexities)}")
if len(unique_complexities) == 1:
    print(f"All records have SAME complexity: {list(unique_complexities)[0]:,} parameters")
    print("\n⚠️  WARNING: Cannot fit complexity-based model with single complexity value!")
    print("   Power law requires varying complexity to establish relationship.")
else:
    print(f"Complexity values: {sorted(unique_complexities)}")

print(f"\n{'='*60}")
print(f"Visualization saved to: complexity_vs_time.png")
print("="*60)

plt.show()
