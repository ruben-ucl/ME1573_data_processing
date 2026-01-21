# Multi-Channel CWT Image Analysis - Usage Guide

## Overview

The CWT image classifier now supports multi-channel analysis, allowing you to combine multiple types of CWT images (e.g., velocity components, pressure fields, temperature distributions) into a single multi-channel model for enhanced classification performance.

## Quick Start

### Single Channel (Backward Compatible)
```json
{
  "cwt_data_dir": "/path/to/single/channel/data",
  "img_channels": 1
}
```

### Multi-Channel Configuration
```json
{
  "cwt_data_channels": {
    "velocity_magnitude": "/path/to/velocity_magnitude",
    "pressure_field": "/path/to/pressure_field", 
    "temperature": "/path/to/temperature"
  },
  "img_channels": 3
}
```

## Configuration Options

### 1. Dictionary-Based Channel Configuration
```json
{
  "cwt_data_channels": {
    "velocity_x": "/data/fluid/velocity_x_cwt",
    "velocity_y": "/data/fluid/velocity_y_cwt",
    "pressure": "/data/fluid/pressure_cwt"
  },
  "img_channels": 3,
  "img_width": 100,
  "img_height": 256
}
```

### 2. Command-Line Multi-Channel Mode

Multi-channel paths are hardcoded in `config.py` (CWT_MULTI_CHANNEL_DIRS). To configure your own paths, edit the CWT_MULTI_CHANNEL_DIRS dictionary:

```python
# In ml/config.py
CWT_MULTI_CHANNEL_DIRS = {
    "velocity_magnitude": r'F:\path\to\velocity_magnitude\cwt\images',
    "pressure_field": r'F:\path\to\pressure_field\cwt\images',
    "temperature": r'F:\path\to\temperature\cwt\images'
}
```

Use the `--multi-channel` flag to enable:

```bash
# Enable multi-channel mode for any optimization
python ml/hyperparameter_tuner.py --classifier cwt_image --multi-channel --mode smart
```

## Data Directory Structure

Each channel must have identical directory structure:

```
channel_directory/
├── 0/              # Class 0 (e.g., "No Pore")
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── 1/              # Class 1 (e.g., "Pore Present")
    ├── image003.png
    ├── image004.png
    └── ...
```

**Important**: All channels must contain the exact same image filenames in corresponding class directories.

## Usage Examples

### 1. Basic Multi-Channel Training
```bash
python ml/CWT_image_classifier_v3.py --config multi_channel_config.json --verbose
```

### 2. Hyperparameter Optimization
```bash
python ml/hyperparameter_tuner.py --classifier cwt_image --mode smart-augmentation --verbose
```

### 3. Channel Ablation Study
```bash
# Channel-ablation mode automatically uses multi-channel configuration
python ml/hyperparameter_tuner.py --classifier cwt_image --mode channel-ablation --verbose

# Or explicitly enable multi-channel for other modes
python ml/hyperparameter_tuner.py --classifier cwt_image --multi-channel --mode smart --verbose
```


### 4. Analyze Channel Contributions
```bash
# After running channel-ablation mode, analyze results
python ml/channel_analysis.py --results ml/logs/cwt/experiment_log.csv --study "ablation_3ch"
```

Or programmatically:
```python
from channel_analysis import analyze_channel_contributions, load_ablation_results_from_csv

# Load results from experiment log
results = load_ablation_results_from_csv("ml/logs/cwt/experiment_log.csv", "ablation_3ch")

# Analyze contributions
analysis = analyze_channel_contributions(results)
print(f"Best individual channel: {analysis['summary']['best_individual_channel']}")
print(f"Multi-channel benefit: +{analysis['summary']['multi_channel_benefit_percent']:.1f}%")
```

## Configuration Examples

### Fluid Dynamics Analysis (3 Channels)
```json
{
  "cwt_data_channels": {
    "velocity_magnitude": "/data/cfd/velocity_magnitude_cwt",
    "pressure_field": "/data/cfd/pressure_field_cwt",
    "vorticity": "/data/cfd/vorticity_cwt"
  },
  "img_channels": 3,
  "img_width": 100,
  "img_height": 256,
  "learning_rate": 0.001,
  "batch_size": 16,
  "conv_filters": [32, 64, 128],
  "epochs": 50
}
```

### Thermal Analysis (2 Channels)
```json
{
  "cwt_data_channels": {
    "temperature": "/data/thermal/temperature_cwt",
    "heat_flux": "/data/thermal/heat_flux_cwt"
  },
  "img_channels": 2,
  "img_width": 100,
  "img_height": 256
}
```

### Material Analysis (3 Channels)
```json
{
  "cwt_data_channels": {
    "stress_xx": "/data/material/stress_xx_cwt",
    "stress_yy": "/data/material/stress_yy_cwt", 
    "strain_energy": "/data/material/strain_energy_cwt"
  },
  "img_channels": 3
}
```

## Channel Ablation Studies

### Automated Ablation Study
The channel-ablation mode automatically uses multi-channel configuration and generates all necessary configurations:
```bash
# Run complete ablation study (uses hardcoded paths from config.py)
python ml/hyperparameter_tuner.py --classifier cwt_image --mode channel-ablation --verbose

# For 3 channels (velocity_magnitude, pressure_field, temperature), automatically generates and runs:
# 1. All 3 channels (baseline)
# 2. velocity_magnitude only
# 3. pressure_field only  
# 4. temperature only
# 5. velocity_magnitude + pressure_field
# 6. velocity_magnitude + temperature
# 7. pressure_field + temperature
```

### Analyzing Results
```bash
# Analyze results using the dedicated script
python ml/channel_analysis.py --results ml/logs/cwt/experiment_log.csv --study "ablation_3ch" --verbose
```

Or programmatically:
```python
from channel_analysis import analyze_channel_contributions, load_ablation_results_from_csv

# Load and analyze ablation results
results = load_ablation_results_from_csv("ml/logs/cwt/experiment_log.csv", "ablation_3ch")
analysis = analyze_channel_contributions(results)

# Individual contributions
for channel, contrib in analysis['individual_contributions'].items():
    if contrib['individual_accuracy'] is not None:
        print(f"{channel}: {contrib['individual_accuracy']:.3f} "
              f"({contrib['relative_contribution']:+.1f}% relative contribution)")

# Interaction effects  
for pair, interaction in analysis['interaction_effects'].items():
    synergy = interaction['synergy_factor']
    print(f"{' + '.join(pair)}: synergy factor = {synergy:.3f}x")
```

## Experiment Logging

The system automatically logs channel information:

- `channel_names`: Comma-separated list of channel labels
- `img_channels`: Number of channels used
- `cwt_data_dir`: Primary data directory (for compatibility)

Example log entry:
```csv
version,channel_names,img_channels,mean_val_accuracy,...
v001,"velocity,pressure,temperature",3,0.924,...
v002,"velocity",1,0.789,...
```

## Best Practices

### 1. Channel Naming
Use descriptive, consistent names:
```python
# Good
"velocity_x", "velocity_y", "pressure"

# Avoid
"ch1", "ch2", "ch3"
```

### 2. Data Consistency
- Ensure all channels have identical image sets
- Use consistent image dimensions across channels
- Maintain same class structure in all directories

### 3. Memory Management
- Multi-channel data uses more memory (2-3x)
- Consider reducing batch size for multi-channel training
- Monitor GPU/CPU memory usage

### 4. Channel Selection
- Start with domain knowledge to select relevant channels
- Use ablation studies to validate channel importance
- Consider correlation between channels to avoid redundancy

## Troubleshooting

### Common Issues

1. **Missing images in channels**
   ```
   Error: Image not found in channel: /path/to/channel2/0/image001.png
   ```
   Solution: Ensure all channels contain the same image files

2. **Channel count mismatch**
   ```
   Error: Number of channels (2) doesn't match img_channels (3)
   ```
   Solution: Update img_channels to match number of channel directories

3. **Memory issues**
   ```
   Error: Out of memory
   ```
   Solution: Reduce batch_size or use fewer channels

### Debugging Tips

1. **Verbose mode**: Use `--verbose` flag to see detailed loading information
2. **Validation**: Use `validate_multichannel_structure()` to check directory consistency
3. **Test loading**: Load a small subset first to verify configuration

## Performance Considerations

### Memory Usage
- Single channel: ~N × H × W × 1 bytes
- Multi-channel: ~N × H × W × C bytes (C = number of channels)
- Recommend 16GB+ RAM for 3-channel datasets

### Training Time
- Multi-channel models train ~20-30% slower than single-channel
- Larger models due to increased input dimensionality
- Consider using mixed precision training for acceleration

### Model Architecture
- Conv2D layers automatically handle multi-channel inputs
- First layer adapts to input channels automatically
- Consider channel-specific processing if needed

## Advanced Features

### Custom Channel Fusion
```python
# In model architecture (future enhancement)
def apply_channel_fusion(x, strategy='early'):
    if strategy == 'early':
        return x  # Standard multi-channel processing
    elif strategy == 'late':
        # Process channels separately then fuse
        pass
    elif strategy == 'attention':
        # Attention-based channel weighting
        pass
```

### Channel-Specific Augmentation
```python
# Future enhancement: different augmentation per channel
"channel_augmentation": {
    "velocity": {"noise_std": 0.01},
    "pressure": {"brightness_range": 0.1},
    "temperature": {"contrast_range": 0.05}
}
```

This multi-channel implementation provides a robust foundation for advanced CWT image analysis while maintaining full backward compatibility with existing single-channel workflows.