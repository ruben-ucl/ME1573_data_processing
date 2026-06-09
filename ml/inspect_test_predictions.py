#!/usr/bin/env python3
"""Inspect test predictions PKL file structure"""

import pickle
from pathlib import Path

pkl_file = Path('D:/ME1573_data_processing/ml/outputs/cwt/v206/test_evaluation/test_predictions_v206.pkl')

print(f"Loading: {pkl_file}")
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

print(f"\nType: {type(data)}")

if isinstance(data, dict):
    print(f"\nKeys: {list(data.keys())}")
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {type(value).__name__} with shape {value.shape}")
        elif isinstance(value, (list, tuple)):
            print(f"  {key}: {type(value).__name__} with length {len(value)}")
            if len(value) > 0:
                print(f"    First item type: {type(value[0])}")
                if hasattr(value[0], '__len__') and not isinstance(value[0], str):
                    print(f"    First item: {value[0][:100] if len(value[0]) > 100 else value[0]}")
                else:
                    print(f"    First few items: {value[:3]}")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")
