# CWT Image Classifier Integration Guide

## Overview

The `CWT_image_classifier_v3.py` script provides a comprehensive CNN-based binary classifier for CWT (Continuous Wavelet Transform) image data, designed to integrate seamlessly with the existing hyperparameter optimization infrastructure while maintaining separation from the photodiode signal classification pipeline.

## Key Features

### 🔗 **Unified Architecture**
- Same command-line interface as `PD_signal_classifier_v3.py`
- Compatible with `hyperparameter_tuner.py` without modifications
- Consistent experiment logging and version management

### 🏗️ **Specialized for CWT Images**
- CNN architecture optimized for CWT scalograms
- Configurable convolutional layers, pooling, and dense networks
- Image-specific data augmentation (rotation, shift, flip)
- Grad-CAM visualization for model interpretability

### 📊 **Separate Experiment Management**
- Independent logging system: `ml/logs/cwt/cwt_experiment_log.csv`
- CWT-specific columns capturing image and CNN parameters
- Dedicated model storage: `ml/models/cwt/`

## Directory Structure

```
ml/
├── logs/
│   ├── experiment_log.csv              # PD signal experiments
│   └── cwt/
│       ├── cwt_experiment_log.csv      # CWT image experiments
│       └── hyperopt_results/
├── models/
│   ├── [PD_signal_models.h5]           # PD signal models
│   └── cwt/
│       └── [CWT_image_models.h5]       # CWT image models
└── CWT_image_classifier_v3.py          # New CWT classifier
```

## Usage Examples

### Basic Usage
```bash
conda activate ml
python ml/CWT_image_classifier_v3.py --epochs 20 --k_folds 5 --verbose
```

### With Configuration File
```bash
python ml/CWT_image_classifier_v3.py --config ml/cwt_example_config.json --concise
```

### Hyperparameter Optimization
The script integrates directly with `hyperparameter_tuner.py`:
```bash
python ml/hyperparameter_tuner.py --script CWT_image_classifier_v3.py --mode smart-architecture
```

## Configuration Parameters

### Data Configuration
```json
{
  "cwt_data_dir": "path/to/cwt/images",
  "img_width": 100,
  "img_height": 256,
  "img_channels": 1
}
```

### CNN Architecture
```json
{
  "conv_filters": [16, 16, 32, 32, 64, 64],
  "conv_kernel_size": [3, 3],
  "pool_size": [2, 2],
  "pool_layers": [2, 5],
  "dense_units": [128],
  "conv_dropout": 0.0,
  "dense_dropout": 0.5,
  "use_batch_norm": false
}
```

### Data Augmentation (CWT-suitable for time series)
```json
{
  "augment_fraction": 0.2,
  "width_shift_range": 0.05,
  "noise_std": 0.01,
  "brightness_range": 0.1,
  "contrast_range": 0.1
}
```

### Grad-CAM Analysis
```json
{
  "run_gradcam": true,
  "gradcam_layer": "auto",
  "save_gradcam_images": false,
  "gradcam_threshold": 0.7
}
```

## CWT Experiment Log Columns

The CWT-specific experiment log includes:

| Column | Description |
|--------|-------------|
| `version` | Experiment version identifier |
| `cwt_data_dir` | Path to CWT image data |
| `img_width/height/channels` | Image dimensions |
| `conv_filters` | CNN filter configuration |
| `conv_kernel_size` | Convolution kernel size |
| `pool_layers` | Pooling layer positions |
| `dense_units` | Dense layer configuration |
| `conv_dropout/dense_dropout` | Dropout rates |
| `augment_fraction` | Data augmentation intensity |
| `width_shift_range/noise_std/brightness_range/contrast_range` | CWT-suitable augmentation parameters |
| `run_gradcam` | Whether Grad-CAM was performed |
| `mean_val_accuracy` | Cross-validation accuracy |
| `mean_precision/recall/f1_score` | Additional metrics |
| `fold_val_accuracies` | Per-fold results |

## Data Directory Structure

Expected CWT image data organization:
```
cwt_data_dir/
├── 0/              # Class 0 (e.g., "No Pore")
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── 1/              # Class 1 (e.g., "Pore")
    ├── image003.png
    ├── image004.png
    └── ...
```

## Integration with Hyperparameter Tuner

### Add CWT Parameter Groups
Update `hyperparameter_tuner.py` to include CWT-specific parameter groups:

```python
# In _get_parameter_groups method
elif mode == 'cwt-architecture':
    return {
        'conv_filters': [[16, 32, 64], [16, 16, 32, 32, 64, 64], [32, 64, 128]],
        'dense_units': [[64], [128], [256], [128, 64]],
        'pool_layers': [[1, 3], [2, 4], [2, 5]]
    }
elif mode == 'cwt-augmentation':
    return {
        'augment_fraction': [0.0, 0.2, 0.5],
        'width_shift_range': [0.0, 0.05, 0.1],
        'noise_std': [0.0, 0.01, 0.02],
        'brightness_range': [0.0, 0.1, 0.2],
        'contrast_range': [0.0, 0.1, 0.2]
    }
```

### Run CWT Hyperparameter Optimization
```bash
python ml/hyperparameter_tuner.py \
    --script CWT_image_classifier_v3.py \
    --mode cwt-architecture \
    --max_time 240 \
    --concise
```

## Benefits

1. **🔄 Clean Separation**: CWT and PD experiments tracked independently
2. **📈 Enhanced Analysis**: Built-in Grad-CAM for CNN interpretability  
3. **🎛️ Full Configurability**: All CNN parameters tunable via config
4. **🔧 Zero Tuner Changes**: Hyperparameter tuner works unchanged
5. **📊 Rich Logging**: CWT-specific metrics and parameters captured
6. **🏗️ Proven Architecture**: Built on battle-tested PD classifier foundation

## Next Steps

1. **Test with Real Data**: Run initial experiments with CWT image dataset
2. **Parameter Exploration**: Use hyperparameter tuner to find optimal configurations  
3. **Architecture Search**: Experiment with different CNN architectures
4. **Grad-CAM Analysis**: Analyze what features the CNN focuses on
5. **Performance Comparison**: Compare results with original CWT_image_binary_classifier.py