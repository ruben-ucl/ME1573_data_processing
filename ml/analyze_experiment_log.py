#!/usr/bin/env python3
"""Analyze experiment log to understand timing data for building estimator."""

import sys
sys.path.insert(0, 'ml')

import pandas as pd
import numpy as np

# Load CWT experiment log
df = pd.read_csv('ml/logs/cwt/cwt_experiment_log.csv')

print("="*70)
print("EXPERIMENT LOG ANALYSIS")
print("="*70)

print(f"\nTotal experiments: {len(df)}")

# Check columns
print(f"\nTiming columns:")
print(f"  - duration_minutes: {df['duration_minutes'].notna().sum()} non-null")
print(f"  - total_training_time_minutes: {df['total_training_time_minutes'].notna().sum()} non-null")

# Check if they're identical
both_exist = df['duration_minutes'].notna() & df['total_training_time_minutes'].notna()
if both_exist.any():
    subset = df[both_exist]
    identical = (subset['duration_minutes'] == subset['total_training_time_minutes']).all()
    print(f"\nIdentical where both exist? {identical}")
    if not identical:
        print(f"  duration_minutes: {subset['duration_minutes'].describe()}")
        print(f"  total_training_time_minutes: {subset['total_training_time_minutes'].describe()}")

# Extract timing data with complete info
timing_cols = ['batch_size', 'k_folds', 'model_complexity', 'total_training_time_minutes']
timing_df = df[timing_cols].copy()

# Clean data
timing_df = timing_df.dropna()
print(f"\nRows with complete timing data: {len(timing_df)}")

if len(timing_df) > 0:
    print("\n" + "="*70)
    print("TIMING DATA SUMMARY")
    print("="*70)
    
    print(f"\nBatch sizes: {sorted(timing_df['batch_size'].unique())}")
    print(f"K-folds: {sorted(timing_df['k_folds'].unique())}")
    print(f"Model complexity range: {timing_df['model_complexity'].min():,.0f} - {timing_df['model_complexity'].max():,.0f}")
    print(f"Training time range: {timing_df['total_training_time_minutes'].min():.1f} - {timing_df['total_training_time_minutes'].max():.1f} minutes")
    
    # Group by batch_size
    print("\n" + "="*70)
    print("BY BATCH SIZE")
    print("="*70)
    for bs in sorted(timing_df['batch_size'].unique()):
        subset = timing_df[timing_df['batch_size'] == bs]
        times = subset['total_training_time_minutes']
        complexities = subset['model_complexity']
        
        print(f"\nBatch Size {bs}:")
        print(f"  Samples: {len(subset)}")
        print(f"  Time: {times.min():.1f} - {times.max():.1f} min (mean={times.mean():.1f}, median={times.median():.1f})")
        print(f"  Complexity: {complexities.min():,.0f} - {complexities.max():,.0f} params")
        print(f"  Time per 1M params: {(times / (complexities/1e6)).mean():.2f} min/1M")
    
    # Group by k_folds
    print("\n" + "="*70)
    print("BY K-FOLDS")
    print("="*70)
    for kf in sorted(timing_df['k_folds'].unique()):
        subset = timing_df[timing_df['k_folds'] == kf]
        times = subset['total_training_time_minutes']
        
        print(f"\nK-folds {kf}:")
        print(f"  Samples: {len(subset)}")
        print(f"  Time: {times.min():.1f} - {times.max():.1f} min (mean={times.mean():.1f}, median={times.median():.1f})")
    
    # Check for patterns
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    
    corr_matrix = timing_df.corr()
    print(f"\nCorrelations with training time:")
    print(f"  batch_size: {corr_matrix.loc['total_training_time_minutes', 'batch_size']:.3f}")
    print(f"  k_folds: {corr_matrix.loc['total_training_time_minutes', 'k_folds']:.3f}")
    print(f"  model_complexity: {corr_matrix.loc['total_training_time_minutes', 'model_complexity']:.3f}")
    
    # Simple linear model
    from sklearn.linear_model import LinearRegression
    
    X = timing_df[['batch_size', 'k_folds', 'model_complexity']]
    y = timing_df['total_training_time_minutes']
    
    model = LinearRegression()
    model.fit(X, y)
    
    print(f"\nLinear Regression (time = a*BS + b*kf + c*complexity + d):")
    print(f"  Coefficients:")
    print(f"    batch_size: {model.coef_[0]:.4f}")
    print(f"    k_folds: {model.coef_[1]:.4f}")
    print(f"    model_complexity: {model.coef_[2]:.8f}")
    print(f"    intercept: {model.intercept_:.4f}")
    print(f"  R² score: {model.score(X, y):.3f}")
    
    # Test predictions
    y_pred = model.predict(X)
    errors = np.abs(y - y_pred)
    print(f"  Mean absolute error: {errors.mean():.2f} minutes")
    print(f"  Median absolute error: {errors.median():.2f} minutes")
    print(f"  90th percentile error: {errors.quantile(0.9):.2f} minutes")

else:
    print("\n⚠️  No complete timing data found!")
    print("\nShowing sample of available data:")
    print(df[timing_cols].head(10))
