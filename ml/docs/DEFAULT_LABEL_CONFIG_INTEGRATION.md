# Default Label Configuration Integration

**Date**: 2025-10-16
**Status**: ✅ Complete

---

## Overview

The CWT image classifier now includes centralized default label configuration in `config.py`, making it easy to:
- Use consistent default labels across all training scripts
- Override defaults via environment variables
- Integrate seamlessly with hyperparameter tuning

---

## Implementation

### 1. Constants (config.py:31-33)

```python
DEFAULT_CWT_LABEL_FILE = r'F:\AlSi10Mg single layer ffc\CWT_labelled_windows\1.0ms-window_0.2ms-offset_labels.csv'
DEFAULT_CWT_LABEL_COLUMN = 'has_porosity'
DEFAULT_CWT_LABEL_TYPE = 'binary'
```

### 2. Helper Function (config.py:35-52)

```python
def get_default_cwt_labels():
    """Get default CWT label configuration with environment variable override support."""
    return {
        'label_file': os.environ.get('ML_CWT_LABEL_FILE', DEFAULT_CWT_LABEL_FILE),
        'label_column': os.environ.get('ML_CWT_LABEL_COLUMN', DEFAULT_CWT_LABEL_COLUMN),
        'label_type': os.environ.get('ML_CWT_LABEL_TYPE', DEFAULT_CWT_LABEL_TYPE)
    }
```

### 3. Template Integration (config.py:152-171)

`get_cwt_config_template()` now automatically includes:
```python
default_labels = get_default_cwt_labels()
config.update({
    'label_file': default_labels['label_file'],
    'label_column': default_labels['label_column'],
    'label_type': default_labels['label_type'],
    'skip_time_ms': 0.0,
    ...
})
```

---

## Usage

### Basic Training (Uses Defaults)

```bash
python ml/CWT_image_classifier_v3.py --epochs 50 --k_folds 5
```

The training will automatically use:
- Label file: `F:\AlSi10Mg single layer ffc\CWT_labelled_windows\1.0ms-window_0.2ms-offset_labels.csv`
- Label column: `has_porosity`
- Label type: `binary`

### Override with Command Line

```bash
python ml/CWT_image_classifier_v3.py \
    --label_file "path/to/other_labels.csv" \
    --label_column "depth_normalized" \
    --label_type continuous \
    --epochs 50
```

### Override with Environment Variables

```bash
export ML_CWT_LABEL_FILE="F:/custom_labels.csv"
export ML_CWT_LABEL_COLUMN="defect_type"
export ML_CWT_LABEL_TYPE="categorical"

python ml/CWT_image_classifier_v3.py --epochs 50
```

### Override with Config File

Create `my_config.json`:
```json
{
    "label_file": "F:/custom_labels.csv",
    "label_column": "my_label",
    "label_type": "continuous",
    "epochs": 100,
    "batch_size": 32
}
```

Then run:
```bash
python ml/CWT_image_classifier_v3.py --config my_config.json
```

---

## Hyperparameter Tuning Integration

The hyperparameter tuner automatically uses default labels when generating configs:

```bash
python ml/hyperparameter_tuner.py cwt_image --mode quick
```

To override defaults in hyperparameter tuning, add to your config JSON:
```json
{
    "label_file": "F:/custom_labels.csv",
    "label_column": "my_label",
    "label_type": "continuous",
    ...
}
```

---

## Environment Variable Reference

| Variable | Purpose | Example |
|----------|---------|---------|
| `ML_CWT_LABEL_FILE` | Override default label CSV path | `export ML_CWT_LABEL_FILE="F:/my_labels.csv"` |
| `ML_CWT_LABEL_COLUMN` | Override default label column | `export ML_CWT_LABEL_COLUMN="keyhole_depth"` |
| `ML_CWT_LABEL_TYPE` | Override default label type | `export ML_CWT_LABEL_TYPE="continuous"` |

---

## Advantages

✅ **Consistency**: Same defaults used across all scripts
✅ **Flexibility**: Override at any level (environment, CLI, config file)
✅ **Maintainability**: Single source of truth for default configuration
✅ **Integration**: Seamlessly works with hyperparameter tuner
✅ **Backward Compatibility**: Folder-based loading still works as fallback

---

## Verification

Run the verification script to test the integration:

```bash
python ml/verify_default_labels.py
```

Expected output:
```
============================================================
DEFAULT LABEL CONFIGURATION VERIFICATION
============================================================

1. Testing Constants:
   DEFAULT_CWT_LABEL_FILE: F:\AlSi10Mg single layer ffc\CWT_labelled_windows\1.0ms-window_0.2ms-offset_labels.csv
   DEFAULT_CWT_LABEL_COLUMN: has_porosity
   DEFAULT_CWT_LABEL_TYPE: binary

2. Testing get_default_cwt_labels():
   ✅ All values match constants

3. Testing Environment Variable Override:
   ✅ Environment variable overrides work

============================================================
✅ ALL TESTS PASSED
============================================================
```

---

## Related Documentation

- [FLAT_DIRECTORY_MIGRATION_GUIDE.md](FLAT_DIRECTORY_MIGRATION_GUIDE.md) - Complete migration guide for flat directory + CSV workflow
- [LABEL_VERIFICATION_REPORT.md](LABEL_VERIFICATION_REPORT.md) - Verification of existing 4,431 binary labels
- [config.py](config.py) - Centralized configuration module

---

**Integration completed**: 2025-10-16
**Tested**: ✅ Verified with verify_default_labels.py
