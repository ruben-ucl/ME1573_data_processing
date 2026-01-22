# Continuous Label Support Implementation - Progress Report

**Date**: 2025-10-15
**Status**: Phase 1 Complete, Phase 2 In Progress

---

## Completed Tasks ✅

### 1. Model Architecture Generalization (`CWT_image_classifier_v3.py`)

**Status**: ✅ COMPLETE

**Changes Made**:

#### a) Created Shared Feature Extractor
- `build_cnn_feature_extractor(input_shape, config, verbose=False)`
  - Builds convolutional + dense layers
  - Shared by both classification and regression models
  - Location: Lines 681-768

#### b) Created Classification Model Builder
- `build_classification_model(input_shape, config, num_classes=2, verbose=False)`
  - Supports binary (num_classes=2) and multiclass (num_classes>2)
  - Binary: 1 output unit + sigmoid activation
  - Multiclass: num_classes output units + softmax activation
  - Uses binary_crossentropy for binary, sparse_categorical_crossentropy for multiclass
  - Location: Lines 770-832

#### c) Created Regression Model Builder
- `build_regression_model(input_shape, config, verbose=False)`
  - 1 output unit + linear activation
  - Loss: MSE (mean squared error)
  - Metrics: MAE, RMSE
  - Location: Lines 834-889

#### d) Unified Model Creation Wrapper
- `create_cnn_model(input_shape, config, label_type='binary', num_classes=2, verbose=False)`
  - Routes to appropriate builder based on label_type
  - Supports: 'binary', 'categorical', 'continuous'
  - Location: Lines 891-917

---

## In Progress 🔄

### 2. Training Loop Unification (`train_fold` function)

**Current Location**: Lines 922-1053 (approx)

**Required Changes**:

#### a) Add `label_type` parameter to `train_fold()` signature
```python
def train_fold(fold, train_idx, val_idx, X, y, config,
               label_type='binary', num_classes=2,  # NEW
               class_weights=None, best_overall_acc=0.0,
               concise=False, output_dir=None):
```

#### b) Update model creation call (currently line ~950)
**Current**:
```python
model = create_cnn_model(input_shape, config, verbose=(fold == 1 and not concise))
```

**Needs to be**:
```python
model = create_cnn_model(input_shape, config, label_type=label_type,
                         num_classes=num_classes, verbose=(fold == 1 and not concise))
```

#### c) Update callback monitoring metrics
**Current** (lines ~962-978):
- Early stopping monitors: `'val_accuracy'`
- LR reduction monitors: `'val_accuracy'`
- Model checkpoint monitors: `'val_accuracy'`

**Needs conditional logic**:
```python
# Determine monitor metric based on label_type
if label_type == 'continuous':
    monitor_metric = 'val_loss'  # For regression, minimize loss
    monitor_mode = 'min'
else:
    monitor_metric = 'val_accuracy'  # For classification, maximize accuracy
    monitor_mode = 'max'

# Update callbacks
callbacks.EarlyStopping(monitor=monitor_metric, mode=monitor_mode, ...)
callbacks.ReduceLROnPlateau(monitor=monitor_metric, mode=monitor_mode, ...)
callbacks.ModelCheckpoint(monitor=monitor_metric, mode=monitor_mode, ...)
```

#### d) Update metrics evaluation (lines ~1037-1045)
**Current** (classification-only):
```python
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)

y_pred_proba = model.predict(X_val, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
y_val_flat = y_val.flatten()

precision = metrics.precision_score(y_val_flat, y_pred, average='binary', zero_division=0)
recall = metrics.recall_score(y_val_flat, y_pred, average='binary', zero_division=0)
f1 = metrics.f1_score(y_val_flat, y_pred, average='binary', zero_division=0)

best_epoch = np.argmax(history.history['val_accuracy']) + 1
```

**Needs**:
```python
if label_type == 'continuous':
    # Regression metrics
    eval_results = model.evaluate(X_val, y_val, verbose=0)
    val_loss = eval_results[0]  # MSE
    val_mae = eval_results[1]   # MAE
    val_rmse = eval_results[2]  # RMSE

    train_results = model.evaluate(X_train, y_train, verbose=0)
    train_loss = train_results[0]
    train_mae = train_results[1]
    train_rmse = train_results[2]

    y_pred = model.predict(X_val, verbose=0).flatten()
    y_val_flat = y_val.flatten()

    # R² score for regression
    from sklearn.metrics import r2_score
    r2 = r2_score(y_val_flat, y_pred)

    best_epoch = np.argmin(history.history['val_loss']) + 1

    return {
        'model': model,
        'history': history,
        'fold': fold,
        'label_type': label_type,
        'train_loss': train_loss,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'val_loss': val_loss,
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'r2_score': r2,
        'best_epoch': best_epoch,
        'total_epochs': len(history.history['loss']),
        'model_complexity': getattr(model, '_model_complexity', 0),
        'X_val': X_val,
        'y_val': y_val,
        'y_pred': y_pred,
        'y_val_flat': y_val_flat
    }
else:
    # Classification metrics (existing code)
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    # ... existing classification code ...
```

#### e) Update progress callback for regression
The `ConciseProgressCallback` class (lines 136-189) currently only tracks accuracy. Need to add support for regression metrics.

---

## Pending Tasks ⏳

### 3. Update `config.py` Logging

**Location**: `D:\ME1573_data_processing\ml\config.py`

**Required Changes**:
- Update `log_experiment_results()` to handle regression metrics
- Add label_type, label_column, label_source to log entries
- Support logging MAE, RMSE, R² instead of accuracy/precision/recall for regression

### 4. Add CLI Arguments

**Location**: `CWT_image_classifier_v3.py` main() function

**New Arguments Needed**:
```python
parser.add_argument('--label_file', type=str,
                    help='Path to CSV file with labels (for CSV-based labeling)')
parser.add_argument('--label_column', type=str,
                    help='Column name for labels in CSV file')
parser.add_argument('--label_type', type=str,
                    choices=['binary', 'categorical', 'continuous'],
                    help='Type of labels: binary, categorical, or continuous')
parser.add_argument('--skip_time_ms', type=float,
                    help='Skip images with window_start_ms < this value')
```

**Data Loading Logic Update**:
```python
if args.label_file and args.label_column:
    # CSV-based labeling
    X, y, label_info, label_stats = load_cwt_image_data_from_csv(
        channel_paths, img_size,
        label_file=args.label_file,
        label_column=args.label_column,
        skip_time_ms=args.skip_time_ms,
        verbose=args.verbose,
        exclude_files=exclude_files
    )
    label_type = label_info['label_type']
    # Derive num_classes from label_stats if needed
else:
    # Folder-based labeling (existing code)
    X, y, class_counts, label_encoder = load_cwt_image_data(...)
    label_type = 'binary'  # Assume binary for folder structure
```

### 5. Update Hyperparameter Tuner

**Location**: `D:\ME1573_data_processing\ml\hyperparameter_tuner.py`

**Required Changes**:
- Pass new CLI arguments (`--label_file`, `--label_column`, `--label_type`, `--skip_time_ms`) to training subprocess
- Handle regression configurations (different metrics to optimize)
- Update config space to include label-related parameters

---

## Testing Requirements

### Unit Tests Needed:
1. ✅ Model architecture functions (build_classification_model, build_regression_model)
2. ⏳ Training loop with continuous labels
3. ⏳ CSV-based data loading with various label types
4. ⏳ Logging system with regression metrics

### Integration Tests Needed:
1. ⏳ End-to-end training with continuous labels from HDF5-extracted CSVs
2. ⏳ Hyperparameter tuning with regression tasks
3. ⏳ Model evaluation and prediction with regression outputs

---

## Key Design Decisions

1. **Shared Feature Extractor**: Allows same CNN architecture for both tasks, only output layer differs
2. **Metric Routing**: Use `label_type` parameter to route to appropriate metrics/loss functions
3. **CSV-based Labels**: Flat directory structure, labels in CSV (vs folder-based binary labels)
4. **Backwards Compatibility**: Existing binary classification workflow unchanged when using folder structure

---

## Files Modified

1. ✅ `ml/CWT_image_classifier_v3.py`
   - Lines 678-917: New model architecture functions
   - Lines 920+: train_fold() function (IN PROGRESS)

2. ⏳ `ml/config.py` (PENDING)
   - log_experiment_results() function

3. ⏳ `ml/hyperparameter_tuner.py` (PENDING)
   - CLI argument passing
   - Metrics handling

---

## Next Immediate Steps

1. **Complete train_fold() updates** (Current task)
   - Add label_type and num_classes parameters
   - Add conditional metric evaluation logic
   - Update callbacks to use appropriate monitoring metrics
   - Update return dictionary for regression

2. **Update ConciseProgressCallback**
   - Support both accuracy (classification) and loss (regression) monitoring

3. **Update main() function data loading**
   - Add CLI arguments
   - Implement CSV vs folder-based loading logic
   - Pass label_type to training functions

4. **Test end-to-end with keyhole depth regression**
   - Use `ml/test_labels/keyhole_depth_labels.csv`
   - Train regression model
   - Verify metrics logging

---

## Estimated Completion

- Phase 2 (Training Loop): 50% complete
- Phase 3 (Config Logging): 0% complete
- Phase 4 (CLI Arguments): 0% complete
- Phase 5 (Hyperparameter Tuner): 0% complete

**Total Progress**: ~30% of feature implementation complete
