# Flat Directory + CSV Labels Migration Guide

**Date**: 2025-10-16
**Status**: Ready for migration

---

## Overview

The CWT image classifier now supports **flat directory structure with CSV-based labels** as the primary workflow. This eliminates the need for class subdirectories (0/, 1/) and enables:
- ✅ **Continuous/regression labels** (not just binary classification)
- ✅ **Categorical labels** with any number of classes
- ✅ **Flexible relabeling** without moving image files
- ✅ **Multiple label sets** for the same images (different CSV files)
- ✅ **Time-based filtering** (skip early windows with `--skip_time_ms`)

---

## Current Implementation Status

### ✅ **COMPLETE: Training Pipeline**

1. **CWT_image_classifier_v3.py**
   - CSV-based labeling is now **primary workflow**
   - Folder-based labeling (0/, 1/) is **fallback only** for backward compatibility
   - Supports: binary, categorical, continuous labels
   - Location: Lines 1485-1548 (data loading), Lines 437-617 (CSV loader function)

2. **hyperparameter_tuner.py**
   - Automatically passes CSV label arguments to training subprocess
   - Location: Lines 1439-1447 (subprocess argument passing)

3. **config.py**
   - Logs label metadata: `label_type`, `label_column`, `label_file`, `skip_time_ms`
   - Tracks regression metrics: MAE, RMSE, R² score
   - Location: Lines 336-541 (logging function)

### ✅ **COMPLETE: Image Generation Scripts**

1. **dataset_labeller.py**
   - **Status**: ✅ Updated (2025-10-16)
   - Saves all images in flat directory structure (no class subdirectories)
   - Generates CSV with required schema: `image_filename`, `trackid`, `window_n`, `window_start_ms`, `window_end_ms`, `label`
   - Location: Lines 759-779 (get_label_folder), Lines 582-614 (CSV logging)

2. **generate_test_holdout.py**
   - **Status**: ✅ Compatible
   - Scans existing directory structures, works with both flat and nested
   - No changes needed

3. **final_model_trainer.py**
   - **Status**: ✅ Compatible
   - Uses CWT_image_classifier_v3.py which supports CSV labels
   - No changes needed

### ✅ **COMPLETE: Default Label Configuration**

**config.py** now includes centralized default label configuration:
- **Constants**: `DEFAULT_CWT_LABEL_FILE`, `DEFAULT_CWT_LABEL_COLUMN`, `DEFAULT_CWT_LABEL_TYPE`
- **Helper function**: `get_default_cwt_labels()` with environment variable override support
- **Template integration**: `get_cwt_config_template()` automatically includes default labels
- **Location**: Lines 31-52 (constants and helper), Lines 167-170 (template defaults)

---

## Recommended Directory Structure

### **NEW: Flat Directory + CSV** (Recommended)

```
F:/AlSi10Mg single layer ffc/
├── CWT_images/
│   └── PD1/
│       └── flat_directory/               # All images together
│           ├── 0105_01_0.2-1.2ms.png
│           ├── 0105_01_0.4-1.4ms.png
│           ├── 0105_01_1.4-2.4ms.png
│           ├── 0563_06_0.8-1.8ms.png
│           └── ...                       # 4,431 images
└── CWT_labelled_windows/
    ├── porosity_binary_labels.csv       # Binary classification
    ├── keyhole_depth_labels.csv         # Continuous regression
    └── defect_type_labels.csv           # Categorical classification
```

### **OLD: Class Subdirectories** (Deprecated but still supported)

```
F:/AlSi10Mg single layer ffc/
└── CWT_images/
    └── PD1/
        ├── 0/                            # Class 0 images
        │   ├── image001.png
        │   └── ...
        └── 1/                            # Class 1 images
            ├── image002.png
            └── ...
```

---

## CSV Label File Schema

### Required Columns

All CSV label files must have these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `image_filename` | string | Image filename (with extension) | `0105_01_0.2-1.2ms.png` |
| `trackid` | string | Track identifier | `0105_01` |
| `window_n` | int | Window number (sequential) | `0`, `1`, `2`, ... |
| `window_start_ms` | float | Window start time (ms) | `0.2`, `0.4`, `0.6`, ... |
| `window_end_ms` | float | Window end time (ms) | `1.2`, `1.4`, `1.6`, ... |
| `<label_column>` | varies | Your label column (any name) | `has_porosity`, `depth_norm`, `defect_type` |

### Example: Binary Classification

```csv
image_filename,trackid,window_n,window_start_ms,window_end_ms,has_porosity
0105_01_0.2-1.2ms.png,0105_01,0,0.2,1.2,0
0105_01_0.4-1.4ms.png,0105_01,1,0.4,1.4,0
0105_01_1.4-2.4ms.png,0105_01,6,1.4,2.4,1
0563_06_0.8-1.8ms.png,0563_06,3,0.8,1.8,0
```

### Example: Continuous Regression

```csv
image_filename,trackid,window_n,window_start_ms,window_end_ms,normalized_depth
0105_01_0.2-1.2ms.png,0105_01,0,0.2,1.2,0.15
0105_01_0.4-1.4ms.png,0105_01,1,0.4,1.4,0.23
0105_01_1.4-2.4ms.png,0105_01,6,1.4,2.4,0.87
0563_06_0.8-1.8ms.png,0563_06,3,0.8,1.8,0.42
```

### Example: Categorical Classification

```csv
image_filename,trackid,window_n,window_start_ms,window_end_ms,defect_type
0105_01_0.2-1.2ms.png,0105_01,0,0.2,1.2,none
0105_01_0.4-1.4ms.png,0105_01,1,0.4,1.4,keyhole
0105_01_1.4-2.4ms.png,0105_01,6,1.4,2.4,porosity
0563_06_0.8-1.8ms.png,0563_06,3,0.8,1.8,spatter
```

---

## Training with CSV Labels

### Binary Classification
```bash
python ml/CWT_image_classifier_v3.py \
    --label_file "F:/AlSi10Mg single layer ffc/CWT_labelled_windows/porosity_binary_labels.csv" \
    --label_column "has_porosity" \
    --epochs 50 \
    --k_folds 5 \
    --concise
```

### Continuous Regression
```bash
python ml/CWT_image_classifier_v3.py \
    --label_file "F:/AlSi10Mg single layer ffc/CWT_labelled_windows/keyhole_depth_labels.csv" \
    --label_column "normalized_depth" \
    --label_type continuous \
    --skip_time_ms 0.5 \
    --epochs 100 \
    --concise
```

### Categorical Classification
```bash
python ml/CWT_image_classifier_v3.py \
    --label_file "F:/AlSi10Mg single layer ffc/CWT_labelled_windows/defect_type_labels.csv" \
    --label_column "defect_type" \
    --label_type categorical \
    --epochs 75 \
    --k_folds 5
```

---

## Hyperparameter Optimization

### With CSV Labels

When using the hyperparameter tuner, add label parameters to your config JSON:

```json
{
    "label_file": "F:/AlSi10Mg single layer ffc/CWT_labelled_windows/porosity_binary_labels.csv",
    "label_column": "has_porosity",
    "label_type": "binary",
    "skip_time_ms": 0.0,
    "learning_rate": 0.001,
    "batch_size": 32,
    ...
}
```

Then run hyperparameter tuner normally:

```bash
python ml/hyperparameter_tuner.py cwt_image --mode quick --config path/to/config.json
```

The tuner will automatically pass `--label_file`, `--label_column`, `--label_type`, and `--skip_time_ms` arguments to each training subprocess.

---

## Migration Steps

### For Existing Binary Classification Projects

If you have existing images in 0/ and 1/ folders:

1. **Create flat directory**:
   ```bash
   mkdir -p "F:/AlSi10Mg single layer ffc/CWT_images/PD1/flat_directory"
   ```

2. **Move all images to flat directory**:
   ```bash
   # Move class 0 images
   cp "F:/AlSi10Mg single layer ffc/CWT_images/PD1/0/"*.png \
      "F:/AlSi10Mg single layer ffc/CWT_images/PD1/flat_directory/"

   # Move class 1 images
   cp "F:/AlSi10Mg single layer ffc/CWT_images/PD1/1/"*.png \
      "F:/AlSi10Mg single layer ffc/CWT_images/PD1/flat_directory/"
   ```

3. **Generate CSV labels** using `verify_and_enhance_labels.py` or manually create CSV with required columns

4. **Update config to use flat directory**:
   ```json
   {
       "cwt_data_dir": "F:/AlSi10Mg single layer ffc/CWT_images/PD1/flat_directory",
       "label_file": "F:/AlSi10Mg single layer ffc/CWT_labelled_windows/labels.csv",
       "label_column": "has_porosity"
   }
   ```

5. **Verify** with a test training run:
   ```bash
   python ml/CWT_image_classifier_v3.py \
       --label_file "path/to/labels.csv" \
       --label_column "has_porosity" \
       --epochs 5 \
       --verbose
   ```

### For New Projects

1. **Generate images directly to flat directory** (requires updating `dataset_labeller.py`)
2. **Labels are automatically saved to CSV** by labeller tool
3. **Train directly** with CSV labels using commands above

---

## Backward Compatibility

### Folder-Based Training Still Works

If you DON'T specify `--label_file` and `--label_column`, the classifier will use the old folder-based approach:

```bash
# This still works (uses 0/ and 1/ folders)
python ml/CWT_image_classifier_v3.py \
    --config config.json \
    --epochs 50
```

Expected directory structure for folder-based:
```
cwt_data_dir/
├── 0/
│   └── *.png
└── 1/
    └── *.png
```

### When to Use Folder-Based

- **Legacy projects** with existing folder structure
- **Quick prototyping** without CSV overhead
- **Binary classification only** (no regression/multiclass)

---

## Advantages of Flat Directory + CSV

| Feature | Folder-Based | CSV-Based |
|---------|--------------|-----------|
| Binary classification | ✅ | ✅ |
| Multiclass classification | ❌ Limited | ✅ |
| Continuous regression | ❌ Not possible | ✅ |
| Multiple label sets | ❌ Requires copying images | ✅ Just create new CSV |
| Relabeling | ❌ Must move files | ✅ Edit CSV only |
| Time filtering | ❌ Not available | ✅ `--skip_time_ms` |
| Experiment tracking | ⚠️ Label not logged | ✅ Full metadata logged |
| Hyperparameter tuning | ⚠️ Manual setup | ✅ Automatic |

---

## Troubleshooting

### Error: "Label CSV missing required columns"

**Cause**: CSV doesn't have all required columns
**Solution**: Ensure CSV has: `image_filename`, `trackid`, `window_n`, `window_start_ms`, `window_end_ms`, and your label column

### Error: "No valid labeled images found"

**Cause**: Images referenced in CSV don't exist in specified directory
**Solution**:
1. Check `image_filename` column matches actual filenames
2. Verify `cwt_data_dir` points to correct directory
3. Check for case sensitivity issues (Windows: insensitive, Linux: sensitive)

### Images load but wrong labels

**Cause**: Wrong `label_column` specified
**Solution**: Use `--verbose` to see detected label type and distribution:
```bash
python ml/CWT_image_classifier_v3.py \
    --label_file "path/to/labels.csv" \
    --label_column "your_column" \
    --verbose
```

### Want to use both binary and regression labels

**Solution**: Keep images in flat directory, create separate CSV files:
```bash
# Binary classification
python ml/CWT_image_classifier_v3.py \
    --label_file "labels_binary.csv" \
    --label_column "has_porosity"

# Regression (same images, different labels)
python ml/CWT_image_classifier_v3.py \
    --label_file "labels_depth.csv" \
    --label_column "normalized_depth" \
    --label_type continuous
```

---

## Migration Status

✅ **MIGRATION COMPLETE** (2025-10-16)

All components have been updated to support flat directory + CSV label workflow:
- ✅ Training pipeline (CWT_image_classifier_v3.py)
- ✅ Hyperparameter tuner (hyperparameter_tuner.py)
- ✅ Image generation (dataset_labeller.py)
- ✅ Experiment logging (config.py)
- ✅ Default configuration (config.py with `get_default_cwt_labels()`)

### Recommended Next Steps

1. **Use flat directory for new projects**: Generate images directly with updated `dataset_labeller.py`
2. **Migrate existing projects**: Follow migration steps above to consolidate class subdirectories
3. **Leverage flexible labeling**: Create multiple CSV files for different label sets without copying images
4. **Environment overrides**: Use `ML_CWT_LABEL_FILE`, `ML_CWT_LABEL_COLUMN`, `ML_CWT_LABEL_TYPE` environment variables to override defaults

---

**Generated**: 2025-10-16
**Updated**: 2025-10-16
**Version**: CWT_image_classifier_v3.py with continuous label support + flat directory defaults
