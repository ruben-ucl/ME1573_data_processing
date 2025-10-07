# ANOVA Improvement Configuration Guide

**⚠️ DEPRECATED:** The `anova` mode has been removed as part of the hyperparameter consolidation effort. This functionality is now integrated into the `smart` mode which uses the centralized hyperparameter registry for more systematic parameter exploration.

For ANOVA analysis improvements, use the focused smart modes:
- `smart-training` for learning rate and batch size optimization
- `smart-regularization` for dropout and L2 regularization analysis

---

## Historical Documentation

This guide previously explained how to use the `anova` mode in the hyperparameter tuner to improve ANOVA analysis quality.

## Problem Statement

Based on comprehensive analysis of 125 experiments, the ANOVA decomposition shows high uncertainty for key parameters:

- **learning_rate**: 33.9% importance but ±0.251 uncertainty (Method agreement: Medium)
- **batch_size**: 20.5% importance but ±0.332 uncertainty (Method agreement: Medium)  
- **augment_fraction**: 8.8% importance but ±0.315 uncertainty (Method agreement: Medium)

## Solution: ANOVA Mode

The `anova` mode generates ~67-80 targeted configurations to:

1. **Reduce parameter uncertainty** through systematic sweeps
2. **Capture interaction effects** between top parameters
3. **Balance architecture sampling** for better statistical power
4. **Focus on high-impact parameters** with poor current coverage

## Usage

### Basic Usage
```bash
conda activate ml
python ml/hyperparameter_tuner.py --mode anova --max-configs 80
```

### With Resume (Recommended)
```bash
python ml/hyperparameter_tuner.py --mode anova --resume --verbose
```

### Test Run (2 configs only)
```bash
python ml/hyperparameter_tuner.py --mode anova --max-configs 2 --verbose
```

## Generated Configuration Structure

### Phase 1: High-Impact Parameter Sweeps (~37 configs)

1. **Learning Rate Sweep** (16 configs)
   - 8 learning rates: [0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005]
   - 2 batch sizes: [16, 32]
   - Systematic coverage to reduce ±0.251 uncertainty

2. **Batch Size Sweep** (6 configs)  
   - 6 batch sizes: [8, 12, 20, 24, 48, 64]
   - Fixed learning_rate: 0.002
   - Fills gaps in current 19-experiment coverage

3. **Augment Fraction Sweep** (9 configs)
   - 9 fractions: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
   - Comprehensive coverage to resolve ±0.315 uncertainty

4. **Dropout/Regularization Sweep** (6 configs)
   - 6 combinations of dropout_rates_mean and l2_reg
   - Better coverage for moderate-importance parameters

### Phase 2: Interaction Effects & Balance (~30 configs)

1. **Learning Rate × Batch Size Factorial** (12 configs)
   - 4 learning rates × 3 batch sizes  
   - Captures interaction effects missing from ANOVA

2. **Learning Rate × Augment Fraction Factorial** (9 configs)
   - 3 learning rates × 3 augment fractions
   - Tests augmentation effectiveness across learning rates

3. **Architecture Balance** (9 configs)
   - Additional experiments for undersampled architectures
   - Balances current imbalance: [32,64,128] has 64 vs others ~10

## Expected Benefits

After running anova mode experiments, expect:

- **Reduced Uncertainty**:
  - learning_rate: ±0.251 → ~±0.15
  - batch_size: ±0.332 → ~±0.20  
  - augment_fraction: ±0.315 → ~±0.18

- **Better Method Agreement**: Medium → High for top parameters
- **Interaction Detection**: Capture learning_rate × batch_size effects
- **Improved Statistical Power**: Balanced architecture sampling

## Validation

After running the experiments, use the comprehensive analyzer to verify improvements:

```bash
python ml/comprehensive_hyperopt_analyzer.py --output-dir ml/outputs/anova_validation
```

Compare the new analysis report with the previous one to confirm:
- Reduced parameter uncertainties
- Improved method agreement scores  
- Better OFAT trend detection
- More balanced architecture analysis

## Integration with Existing Workflow

The anova mode integrates seamlessly with the existing hyperparameter optimization pipeline:

1. **Deduplication**: Automatically skips previously tested configurations
2. **Resume Support**: Can resume interrupted ANOVA improvement runs
3. **Logging**: Uses standard experiment logging format
4. **Analysis**: Compatible with all existing analysis tools

## Computational Cost

- **Estimated Runtime**: ~67 configs × 2-3 hours/config = ~134-200 hours
- **Recommendation**: Run on multiple GPUs or in batches
- **Priority**: Focus on Phase 1 first (highest impact parameters)

## Next Steps

1. Run the anova mode experiments
2. Analyze results with comprehensive_hyperopt_analyzer.py  
3. Compare ANOVA uncertainty reductions
4. If successful, consider additional targeted experiments for remaining high-uncertainty parameters