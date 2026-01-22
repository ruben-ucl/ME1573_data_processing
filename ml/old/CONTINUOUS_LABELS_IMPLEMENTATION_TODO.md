# Continuous Labels Implementation - Remaining Tasks

## Status: Partially Complete

### ✅ Completed Tasks

1. **extract_labels_from_hdf5.py** (ml/extract_labels_from_hdf5.py)
   - Extracts labels from HDF5 timeseries data
   - Parses image filenames for trackid and time windows
   - Supports max/mean aggregation
   - Outputs CSV with required schema

2. **migrate_binary_labels_to_csv.py** (ml/migrate_binary_labels_to_csv.py)
   - Converts folder-based binary labels to unified CSV format
   - Maintains compatibility with new labeling system

3. **CSV-based data loading** (CWT_image_classifier_v3.py:437-617)
   - New function: `load_cwt_image_data_from_csv()`
   - Supports time-based filtering via `skip_time_ms`
   - Auto-detects label type (binary/continuous/categorical)
   - Returns label_info and label_stats for logging

### ⏳ Remaining Tasks

#### 4. Generalize Model Building (CWT_image_classifier_v3.py)

**Current state:** Single `create_cnn_model()` function hardcoded for binary classification

**Required changes:**
- Split into two functions:
  ```python
  def build_classification_model(input_shape, config, num_classes=2, verbose=False):
      # ... existing architecture ...
      # Output layer:
      if num_classes == 2:
          model.add(layers.Dense(1, activation='sigmoid', name='output'))
          loss = 'binary_crossentropy'
      else:
          model.add(layers.Dense(num_classes, activation='softmax', name='output'))
          loss = 'categorical_crossentropy'
      model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
      return model

  def build_regression_model(input_shape, config, verbose=False):
      # ... same architecture, different output ...
      model.add(layers.Dense(1, activation='linear', name='output'))
      model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
      return model
  ```

- Add wrapper function:
  ```python
  def create_model(input_shape, config, label_type, verbose=False):
      """Unified model creation based on label type."""
      if label_type in ['binary', 'categorical']:
          num_classes = config.get('num_classes', 2)
          return build_classification_model(input_shape, config, num_classes, verbose)
      else:  # continuous
          return build_regression_model(input_shape, config, verbose)
  ```

**Files to modify:**
- ml/CWT_image_classifier_v3.py (lines ~449-568)

#### 5. Unify Training Loop (CWT_image_classifier_v3.py)

**Current state:** `train_fold()` hardcoded for classification metrics

**Required changes:**
- Add conditional metrics based on label_type:
  ```python
  def train_fold(fold, train_idx, val_idx, X, y, config, label_type='binary',
                 class_weights=None, best_overall_metric=0.0, concise=False, output_dir=None):
      # ... existing setup ...

      # Create model based on label type
      model = create_model(input_shape, config, label_type, verbose=(fold == 1 and not concise))

      # Setup callbacks with appropriate monitor metric
      if label_type in ['binary', 'categorical']:
          monitor_metric = 'val_accuracy'
      else:  # continuous
          monitor_metric = 'val_mae'  # or val_loss

      # Adjust callbacks...
      callbacks.EarlyStopping(monitor=monitor_metric, ...)
      callbacks.ReduceLROnPlateau(monitor=monitor_metric, ...)

      # Calculate metrics based on label type
      if label_type in ['binary', 'categorical']:
          # ... existing classification metrics ...
          precision = metrics.precision_score(...)
          recall = metrics.recall_score(...)
          f1 = metrics.f1_score(...)
      else:  # continuous
          # Regression metrics
          from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
          mae = mean_absolute_error(y_val, y_pred)
          mse = mean_squared_error(y_val, y_pred)
          r2 = r2_score(y_val, y_pred)
          precision, recall, f1 = None, None, None  # Not applicable

      return {
          # ... existing fields ...
          'mae': mae if label_type == 'continuous' else None,
          'mse': mse if label_type == 'continuous' else None,
          'r2': r2 if label_type == 'continuous' else None,
      }
  ```

**Files to modify:**
- ml/CWT_image_classifier_v3.py (lines ~573-723)

#### 6. Update Main Training Function (CWT_image_classifier_v3.py)

**Current state:** Only uses `load_cwt_image_data()` (folder-based)

**Required changes:**
- Add CLI argument parsing for new parameters
- Add data loading logic that chooses between folder-based and CSV-based:
  ```python
  # In main() around line ~1067
  if args.label_file:
      # CSV-based loading
      X, y, label_info, label_stats = load_cwt_image_data_from_csv(
          channel_paths, img_size,
          label_file=args.label_file,
          label_column=args.label_column,
          skip_time_ms=args.skip_time_ms,
          verbose=args.verbose,
          exclude_files=exclude_files
      )
      label_type = label_info['label_type']
      class_counts = label_stats.get('class_distribution', {})

      # Store label info in config for logging
      config.update(label_info)
      config['label_stats'] = label_stats
  else:
      # Folder-based loading (existing)
      X, y, class_counts, label_encoder = load_cwt_image_data(...)
      label_type = 'binary'  # Assumed
      config['label_type'] = 'binary'
      config['label_source'] = 'folder'

  # Pass label_type to training
  fold_result = train_fold(..., label_type=label_type, ...)
  ```

- Update k-fold strategy:
  ```python
  if label_type in ['binary', 'categorical']:
      skf = StratifiedKFold(...)  # Stratified for classification
  else:
      from sklearn.model_selection import KFold
      skf = KFold(...)  # Regular k-fold for regression
  ```

**Files to modify:**
- ml/CWT_image_classifier_v3.py (lines ~922-1340)

#### 7. Update Experiment Logging (config.py)

**Current state:** Only logs classification metrics

**Required changes:**
- Add columns to experiment log:
  ```python
  'label_type': config.get('label_type', 'binary'),
  'label_column': config.get('label_column', ''),
  'label_source': config.get('label_source', 'folder'),
  'skip_time_ms': config.get('skip_time_ms', 0),

  # Regression metrics
  'mean_mae': np.mean([r['mae'] for r in fold_results if r['mae'] is not None]),
  'mean_mse': np.mean([r['mse'] for r in fold_results if r['mse'] is not None]),
  'mean_r2': np.mean([r['r2'] for r in fold_results if r['r2'] is not None]),
  ```

- Update `log_experiment_results()` function

**Files to modify:**
- ml/config.py (around line 415 in log_experiment_results function)

#### 8. Add CLI Arguments (CWT_image_classifier_v3.py)

**Required additions:**
```python
parser.add_argument('--label_file', type=str, help='Path to CSV file containing labels')
parser.add_argument('--label_column', type=str, help='Name of column in CSV containing labels')
parser.add_argument('--label_type', type=str, choices=['auto', 'binary', 'continuous', 'categorical'],
                   default='auto', help='Label type (auto-detected if not specified)')
parser.add_argument('--skip_time_ms', type=float, help='Skip images with window_start_ms < this value')
```

**Files to modify:**
- ml/CWT_image_classifier_v3.py (lines ~925-942)

#### 9. Update Hyperparameter Tuner (hyperparameter_tuner.py)

**Required changes:**
- Add new parameters to tuner:
  ```python
  def __init__(self, ..., label_file=None, label_column=None, label_type='auto', skip_time_ms=None):
      self.label_file = label_file
      self.label_column = label_column
      self.label_type = label_type
      self.skip_time_ms = skip_time_ms
  ```

- Save to config:
  ```python
  if self.label_file:
      config['label_file'] = self.label_file
      config['label_column'] = self.label_column
      config['label_type'] = self.label_type
  if self.skip_time_ms:
      config['skip_time_ms'] = self.skip_time_ms
  ```

- Pass to subprocess:
  ```python
  if self.label_file:
      cmd.extend(['--label_file', self.label_file])
      cmd.extend(['--label_column', self.label_column])
      cmd.extend(['--label_type', self.label_type])
  if self.skip_time_ms:
      cmd.extend(['--skip_time_ms', str(self.skip_time_ms)])
  ```

- Add CLI arguments:
  ```python
  parser.add_argument('--label_file', type=str, help='Path to CSV label file')
  parser.add_argument('--label_column', type=str, help='Label column name')
  parser.add_argument('--label_type', type=str, default='auto', help='Label type')
  parser.add_argument('--skip_time_ms', type=float, help='Skip early time windows')
  ```

**Files to modify:**
- ml/hyperparameter_tuner.py (multiple locations)

## Testing Checklist

Once implementation is complete:

1. **Test extract_labels_from_hdf5.py:**
   ```bash
   python ml/extract_labels_from_hdf5.py \
       --data_dir "path/to/cwt/images" \
       --hdf5_dir "path/to/hdf5" \
       --label_columns "depth" \
       --aggregation max \
       --output_csv "test_labels.csv"
   ```

2. **Test migrate_binary_labels_to_csv.py:**
   ```bash
   python ml/migrate_binary_labels_to_csv.py \
       --data_dir "path/to/labeled/folders" \
       --label_column_name "has_keyhole" \
       --folder_mapping "keyhole=1,no_keyhole=0" \
       --output_csv "binary_labels.csv"
   ```

3. **Test continuous label training:**
   ```bash
   python ml/CWT_image_classifier_v3.py \
       --label_file "test_labels.csv" \
       --label_column "depth" \
       --label_type auto \
       --epochs 10 \
       --k_folds 3
   ```

4. **Test with time filtering:**
   ```bash
   python ml/CWT_image_classifier_v3.py \
       --label_file "test_labels.csv" \
       --label_column "depth" \
       --skip_time_ms 1.0 \
       --verbose
   ```

5. **Test hyperparameter tuning with continuous labels:**
   ```bash
   python ml/hyperparameter_tuner.py \
       --mode test \
       --label_file "test_labels.csv" \
       --label_column "depth" \
       --label_type continuous
   ```

## Notes

- Backward compatibility is maintained: existing folder-based training still works
- Label type auto-detection means users don't need to specify it explicitly
- Time-based filtering is applied BEFORE any dataset composition calculations
- All new features are optional and don't break existing workflows
