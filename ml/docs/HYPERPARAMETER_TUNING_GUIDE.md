# Hyperparameter Tuning Guide for PD Signal Classifier

This guide explains how to use the hyperparameter optimization system for the PD Signal Classifier.

## 📁 Files Overview

### Main Scripts (in `ml/` directory)
- `PD_signal_classifier_v3.py` - Main training script (supports manual and automated use)
- `hyperparameter_tuner.py` - Automated hyperparameter optimization script  
- `example_config.json` - Example configuration file
- `HYPERPARAMETER_TUNING_GUIDE.md` - This guide

### Generated Folders
- `outputs/` - Individual experiment results (versioned: v001/, v002/, etc.)
- `ml/logs/hyperopt_results/` - Hyperparameter optimization results and progress
- `logs/` - **Unified experiment log** for all runs (manual + automated)

## 🎯 Quick Start

**Important: All commands should be run from the project root with the ML conda environment:**
```bash
conda activate ml
```

### Option 1: Manual Training (Original Approach)
```bash
# Run with default settings
python ml/PD_signal_classifier_v3.py

# Override specific parameters
python ml/PD_signal_classifier_v3.py --learning_rate 0.0005 --batch_size 32 --epochs 30

# Use a custom configuration file
python ml/PD_signal_classifier_v3.py --config my_config.json
```

### Option 2: Automated Hyperparameter Tuning (Recommended)
```bash
# Adaptive Smart tuning (recommended - searches around best previous results)
python ml/hyperparameter_tuner.py --mode smart

# Smart tuning with wider search radius (±2 values per parameter)
python ml/hyperparameter_tuner.py --mode smart --search-radius 2

# Smart tuning with deduplication disabled (allows retesting previous configs)
python ml/hyperparameter_tuner.py --mode smart --skip-deduplication

# Smart tuning with grid search (automatically limits to reasonable size ~27-100 configs)
python ml/hyperparameter_tuner.py --mode smart --grid-search

# Smart tuning with ignored parameters (use best previous values for specified params)
python ml/hyperparameter_tuner.py --mode smart --ignore batch_size conv_dropout

# Medium tuning (~25-30 configs, systematic OFAT approach)
python ml/hyperparameter_tuner.py --mode medium --concise

# Quick tuning (~15-20 configs, basic exploration)
python ml/hyperparameter_tuner.py --mode quick

# Full grid search (tests hundreds of configurations - use with caution!)
python ml/hyperparameter_tuner.py --mode full --max_configs 50
```

## 🔧 Manual Training Usage

### Command Line Arguments
- `--config PATH` - Load configuration from JSON file
- `--learning_rate FLOAT` - Set learning rate (e.g., 0.001)
- `--batch_size INT` - Set batch size (e.g., 16)
- `--epochs INT` - Set number of epochs (e.g., 50)
- `--k_folds INT` - Set number of K-fold splits (e.g., 5)
- `--data_dir PATH` - Set data directory path
- `--output_root PATH` - Set output root directory

### Configuration File Format
Create a JSON file (see `example_config.json`):
```json
{
  "learning_rate": 0.001,
  "batch_size": 16,
  "epochs": 50,
  "conv_filters": [16, 32, 64],
  "dense_units": [128, 64],
  "conv_dropout": 0.2,
  "dense_dropout": [0.3, 0.2],
  "l2_regularization": 0.001
}
```

### Examples
```bash
# Test different learning rates
python ml/PD_signal_classifier_v3.py --learning_rate 0.0005
python ml/PD_signal_classifier_v3.py --learning_rate 0.002

# Test different batch sizes  
python ml/PD_signal_classifier_v3.py --batch_size 8
python ml/PD_signal_classifier_v3.py --batch_size 32

# Use custom config with specific settings
python ml/PD_signal_classifier_v3.py --config my_experiment.json
```

## 🤖 Automated Hyperparameter Tuning

### 🧠 Smart Deduplication (NEW)

By default, the hyperparameter tuner now includes **smart deduplication** to avoid wasting time on configurations that have already been tested:

- **Automatically detects** previously tested configurations from `experiment_log.csv`
- **Skips duplicate experiments** to focus only on new parameter combinations
- **Works across all modes** (test, quick, smart, full)
- **Saves time and compute** by filling in gaps in the hyperparameter space

**How it works:**
1. Before running, the tuner checks `ml/logs/experiment_log.csv` for previous experiments
2. Compares key hyperparameters: learning_rate, batch_size, architecture, dropout, etc.
3. Removes any configurations that match previous experiments
4. Reports how many configurations were skipped

**Example output:**
```
Smart deduplication: Skipped 7/25 configurations (already tested)
Running 18 new configurations...
```

**To disable deduplication:**
```bash
# Allow retesting previous configurations
python ml/hyperparameter_tuner.py --mode smart --skip-deduplication
```

**When to disable:**
- You want to replicate previous experiments for validation
- You've made changes to the training code and want to retest
- You're comparing different random seeds or other non-tracked parameters

### 🎯 Adaptive Smart Mode (NEW)

**Smart mode is now truly smart!** It automatically builds its search strategy around your best previous results:

**How it works:**
1. **Finds your best config**: Identifies the experiment with highest validation accuracy
2. **Extracts optimal values**: Uses those parameter values as the starting point
3. **Searches neighbors**: Tests values immediately above and below each optimal parameter
4. **Focuses effort**: Only explores the most promising parameter region

**Example:** If your best config had `LR=0.001`, smart mode will test `LR=0.0005` and `LR=0.002` (neighboring values in the search space).

**Search Radius Control:**
```bash
# Search ±1 value around best config (default, ~6-12 configs)
python ml/hyperparameter_tuner.py --mode smart --search-radius 1

# Search ±2 values around best config (wider search, ~12-24 configs)  
python ml/hyperparameter_tuner.py --mode smart --search-radius 2
```

**Boundary Handling:**
- If optimal value is at edge of search space (e.g., highest learning rate), smart mode will notify you
- You can extend the search space in `define_search_space()` and re-run if needed

**Grid Search vs OFAT:**
```bash
# OFAT mode (default) - varies one parameter at a time (~6-15 configs)
python ml/hyperparameter_tuner.py --mode smart --search-radius 1

# Grid search mode - automatically limits to reasonable size (~27-100 configs) 
python ml/hyperparameter_tuner.py --mode smart --grid-search
```

**Parameter Ignoring:**
```bash
# Skip optimization of specific parameters, using best previous values
python ml/hyperparameter_tuner.py --mode smart --ignore batch_size conv_dropout

# Useful when you know optimal values for some parameters
python ml/hyperparameter_tuner.py --mode smart --ignore l2_regularization use_class_weights
```

**Fallback Behavior:**
- If no previous experiments exist, smart mode uses default parameter exploration
- Ensures robust behavior whether it's your first run or hundredth

### 🔄 Optimization Modes

**Available Modes:**

| Mode | Strategy | Best For |
|------|----------|----------|
| **test** | Basic validation | Testing setup (~15 minutes) |
| **smart** | Registry-based adaptive search | All optimization scenarios |
| **channel-ablation** | Multi-channel analysis study | CWT channel contribution analysis |

### **Smart Mode with Registry Filtering**

Smart mode now uses the centralized hyperparameter registry with flexible filtering:

```bash
# Basic smart mode - all parameters
python ml/hyperparameter_tuner.py --mode smart

# Focus on specific categories
python ml/hyperparameter_tuner.py --mode smart --category training regularization

# Focus on high-priority parameters only
python ml/hyperparameter_tuner.py --mode smart --priority 1 2

# Combine category and priority filtering
python ml/hyperparameter_tuner.py --mode smart --category architecture --priority 1

# Focus on critical parameters only (fastest optimization)
python ml/hyperparameter_tuner.py --mode smart --priority 1

# Focus on critical + high impact parameters
python ml/hyperparameter_tuner.py --mode smart --priority 1 2

# Full grid search within training category
python ml/hyperparameter_tuner.py --mode smart --category training --grid_search

# Exclude fixed parameters explicitly (default behavior)
python ml/hyperparameter_tuner.py --mode smart --category training regularization architecture

# Include fixed parameters for research (not recommended for optimization)
python ml/hyperparameter_tuner.py --mode smart --category fixed
```

**Available Categories:**
- `training`: Core training parameters (learning_rate, batch_size)
- `regularization`: Overfitting control (dropout, L2 reg, batch norm)
- `architecture`: Model structure (conv filters, dense units)
- `training_control`: Training behavior (early stopping, LR scheduling, class weights)
- `augmentation`: Data augmentation parameters (PD/CWT specific)
- `fixed`: Parameters typically not optimized (epochs, k_folds, optimizer, conv_kernel_size, pool_size)

**Research-Based Priority Tiers:**
- **Tier 1** (Critical): `learning_rate`, `batch_size`, `conv_dropout`, `dense_dropout`
- **Tier 2** (High Impact): `conv_filters`, `dense_units`, `l2_regularization`, `augment_fraction`
- **Tier 3** (Moderate): `early_stopping_patience`, `use_batch_norm`, `use_class_weights`, `time_shift_probability`
- **Tier 4** (Low Impact): `conv_kernel_size`, `pool_size`, `lr_reduction_*`, `noise_probability`
- **Tier 5** (Minimal): Specialist augmentation parameters (`*_range`, `*_scale`, etc.)
- **Fixed**: `epochs`, `k_folds`, `optimizer` (typically not optimized)

**Recommended workflow:**
1. **Start with `test`** - Verify setup works correctly
2. **Use `smart`** - Comprehensive search using all parameters
3. **Focus with filters** - Target specific categories/priorities for refinement
4. **Use `channel-ablation`** - For CWT multi-channel studies when applicable

### Output Modes

The hyperparameter tuner supports three different output modes:

#### 1. Silent Mode (Default)
- No real-time training output shown
- Only shows configuration start/completion messages
- Best for running unattended or in background
- Monitor progress via separate terminal watching CSV files

#### 2. Concise Mode (Recommended for Monitoring)
```bash
python ml/hyperparameter_tuner.py --mode smart --concise
```
- Shows one-line progress updates per epoch: `Fold 2/5 | Epoch 23/50 | Val Acc: 0.8234 | Best This Fold: 0.8456 | Best Overall: 0.8456`
- Shows configuration completion with timing: `Configuration 1/26 - COMPLETED | Val Acc: 0.8456 | Completed: 14:32:15 | Duration: 45.2m`
- Perfect balance of information and readability
- Still captures full output for result parsing

#### 3. Verbose Mode (Full Output)
```bash
python ml/hyperparameter_tuner.py --mode smart --verbose
```
- Shows complete training output in real-time
- All TensorFlow logs, callbacks, and debugging info
- Good for debugging specific configurations
- Note: Result parsing is limited in this mode

### Optimization Modes

#### 1. Smart Mode (Recommended)
- Tests ~35 carefully selected configurations
- Balances thoroughness with computation time
- Based on ML best practices and common patterns

```bash
python ml/hyperparameter_tuner.py --mode smart
```

**What it tests:**
- Learning rates: [0.0001, 0.0005, 0.001, 0.002, 0.005]
- Batch sizes: [8, 16, 32] 
- Dropout combinations: 6 different combinations
- Architecture variations: 3 different model sizes
- L2 regularization: [0.0, 0.0001, 0.001, 0.01]
- Class weighting: With and without

#### 2. Quick Mode
- Tests ~20 configurations
- Good for initial exploration or limited time

```bash
python ml/hyperparameter_tuner.py --mode quick
```

#### 3. Full Mode
- Tests all combinations (can be hundreds!)
- Use `--max_configs` to limit

```bash
python ml/hyperparameter_tuner.py --mode full --max_configs 50
```

### Advanced Options
```bash
# Limit number of configurations
python ml/hyperparameter_tuner.py --mode smart --max_configs 20

# Use custom output directory
python ml/hyperparameter_tuner.py --mode smart --output_dir my_hyperopt_results

# Resume interrupted run
python ml/hyperparameter_tuner.py --mode smart --resume

# Concise mode (recommended for monitoring progress)
python ml/hyperparameter_tuner.py --mode smart --concise

# Verbose mode (full training output in real-time)
python ml/hyperparameter_tuner.py --mode smart --verbose
```

## 📊 Understanding Results

### Experiment Tracking
Every training run creates:
- **Versioned folders** (`outputs/v001/`, `outputs/v002/`, etc.) with all outputs  
- **logs/experiment_log.csv** - **UNIFIED** master tracking file for ALL experiments (manual + automated)
- **Individual model files** - Trained models for each fold

### Hyperparameter Optimization Results
The tuner creates:
- **ml/logs/hyperopt_results/hyperparameter_results.csv** - All optimization results sorted by performance
- **ml/logs/hyperopt_results/top_10_results.csv** - Best 10 configurations
- **ml/logs/hyperopt_results/failed_configs.json** - Any failed configurations
- **ml/logs/hyperopt_results/tuning_progress.json** - Progress tracking for resume

### Key Metrics to Track
- **mean_val_accuracy** - Average validation accuracy across folds
- **std_val_accuracy** - Standard deviation (lower = more stable)
- **best_val_accuracy** - Best single fold performance
- **training_time_minutes** - Time efficiency

## 🎯 Hyperparameter Priorities

Based on ML research, focus on these in order:

### Tier 1: Highest Impact
1. **Learning Rate** - Most critical parameter
   - Start with: [0.0001, 0.001, 0.01]
   - Too high = unstable, too low = slow convergence

2. **Batch Size** - Affects gradient quality
   - Try: [8, 16, 32]
   - Smaller often better for small datasets

3. **Class Weights** - Critical for imbalanced data
   - Always test both True and False

### Tier 2: High Impact  
4. **Dropout Rates** - Controls overfitting
   - Conv dropout: [0.1, 0.2, 0.3]
   - Dense dropout: [[0.2, 0.1], [0.3, 0.2], [0.4, 0.3]]

5. **Model Architecture** - Capacity vs. overfitting
   - Simple: [16, 32] conv, [64] dense
   - Default: [16, 32, 64] conv, [128, 64] dense  
   - Complex: [32, 64, 128] conv, [256, 128] dense

6. **L2 Regularization** - Prevents overfitting
   - Try: [0.0, 0.0001, 0.001, 0.01]

### Tier 3: Moderate Impact
7. **Early Stopping Patience** - [8, 10, 12]
8. **LR Reduction Settings** - Factor and patience

## 💡 Best Practices

### 1. Start Small
```bash
# Begin with quick mode to get baseline
python ml/hyperparameter_tuner.py --mode quick --max_configs 10
```

### 2. Iterative Refinement
1. Run smart mode to find good ranges
2. Manually test variations of best configs
3. Focus on most impactful parameters

### 3. Resource Management
- Each config takes ~30-60 minutes (depends on data size)
- Smart mode ≈ 20-35 hours total
- Use `--max_configs` to limit time
- Use `--resume` to continue interrupted runs

### 4. Interpreting Results
- Look for **stable performance** (low std_val_accuracy)
- Consider **training time** vs. performance trade-offs
- **Overfit warning**: Large gap between train and validation accuracy

## 🔍 Troubleshooting

### Common Issues

#### 1. "Stuck on One Class" Problem
- Usually caused by severe class imbalance
- Check diagnostic output for class distribution
- Ensure `use_class_weights: true`
- Try different learning rates (often too high)

#### 2. Very Low Accuracy
- Check data normalization in diagnostic output
- Verify data ranges are [0, 1] or close
- Try smaller learning rates

#### 3. Training Takes Too Long
- Reduce `epochs` to 30 or add aggressive early stopping
- Use smaller batch sizes (8 or 16)
- Consider simpler model architecture

#### 4. Memory Issues
- Reduce batch size to 8
- Use simpler model architecture
- Check available GPU memory

### Resume Interrupted Runs
```bash
# The tuner automatically saves progress
python ml/hyperparameter_tuner.py --mode smart --resume
```

### Check Progress
```bash
# Monitor hyperparameter optimization results
tail -f ml/logs/hyperopt_results/hyperparameter_results.csv

# Monitor unified experiment log (all experiments)
tail -f logs/experiment_log.csv

# Check optimization progress
cat ml/logs/hyperopt_results/tuning_progress.json
```

## 📈 Example Workflow

### Week 1: Initial Exploration
```bash
# Quick baseline
python ml/hyperparameter_tuner.py --mode quick --max_configs 10

# Check results, identify promising ranges
# Manually test a few variations
python ml/PD_signal_classifier_v3.py --learning_rate 0.0005 --batch_size 8
```

### Week 2: Focused Optimization  
```bash
# Smart mode with findings from Week 1
python ml/hyperparameter_tuner.py --mode smart --max_configs 25

# Test top 3 configurations with longer training
python ml/PD_signal_classifier_v3.py --config best_config.json --epochs 100
```

### Week 3: Final Validation
```bash
# Train best model with full data and cross-validation
python ml/PD_signal_classifier_v3.py --config final_config.json --k_folds 10
```

## 🎯 Success Metrics

**Good Results:**
- Validation accuracy > 80%
- Std deviation < 0.05
- No "stuck on one class" behavior
- Reasonable training time (<2 hours per fold)

**Red Flags:**
- All predictions in one class
- Very high validation loss
- Huge train/validation accuracy gap
- Extremely long training times

---

## 📞 Quick Reference Commands

**First, navigate to the ml directory and activate the environment:**
```bash
cd ml/
conda activate ml
```

**Then run these commands:**
```bash
# Manual single run with defaults
python ml/PD_signal_classifier_v3.py

# Manual run with custom parameters  
python ml/PD_signal_classifier_v3.py --learning_rate 0.0005 --batch_size 32

# Smart hyperparameter optimization (recommended)
python ml/hyperparameter_tuner.py --mode smart

# Smart optimization with concise progress (recommended for monitoring)
python ml/hyperparameter_tuner.py --mode smart --concise

# Quick optimization for testing
python ml/hyperparameter_tuner.py --mode quick --max_configs 15

# Resume interrupted optimization
python ml/hyperparameter_tuner.py --mode smart --resume
```

Happy tuning! 🚀