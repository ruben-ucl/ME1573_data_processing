# Design of Experiments (DoE) Strategy for Neural Network Hyperparameter Optimization

## Executive Summary

Based on parameter sensitivity analysis, this document provides a structured experimental design to improve hyperparameter space coverage and identify optimal configurations with statistical rigor.

## Current Analysis Gaps

### Parameters with Insufficient Sampling (n < 3 levels)
*These require immediate attention for ANOVA validity:*

1. **Network Architecture Parameters**
   - `conv_filters`: Limited architectural diversity
   - `dense_units`: Few hidden layer configurations tested
   - `pool_layers`: Insufficient pooling strategy variation

2. **Regularization Parameters** 
   - `l2_regularization`: Narrow range explored
   - `batch_norm`: Binary parameter needs systematic A/B testing

3. **Training Parameters**
   - `optimizer`: Limited algorithm comparison
   - `lr_reduction_factor`: Few decay strategies tested

## Recommended DoE Strategy: Three-Phase Approach

### Phase 1: Screening Design (Immediate Priority)
**Objective**: Identify active factors and interactions
**Method**: Fractional Factorial Design (2^(k-1))
**Duration**: 32-64 experiments

#### Parameter Selection (Top 6 factors)
Based on current sensitivity analysis prioritization:

1. **learning_rate**: [0.0001, 0.001, 0.01] (3 levels)
2. **batch_size**: [8, 16, 32, 64] (4 levels) 
3. **conv_filters**: [[16,32], [16,32,64], [32,64,128]] (3 architectures)
4. **dense_units**: [[64], [128], [256,128]] (3 configurations)
5. **dropout**: [0.0, 0.2, 0.5] (3 levels)
6. **l2_regularization**: [0.0, 0.001, 0.01] (3 levels)

#### Experimental Design Matrix
```
Fractional Factorial 2^(6-1) = 32 base experiments
+ 8 center point replicates
+ 4 axial points for curvature detection
= 44 total experiments
```

#### Success Metrics
- Identify 2-3 most important parameters
- Detect significant 2-factor interactions  
- Achieve >80% variance explained by model

### Phase 2: Response Surface Methodology (Follow-up)
**Objective**: Optimize around promising regions
**Method**: Central Composite Design (CCD)
**Duration**: 40-80 experiments

#### Focus Areas
Based on Phase 1 results, implement CCD on:
- Top 3 most significant parameters
- Parameters involved in significant interactions
- Narrow ranges around optimal regions

#### Design Specifications
- 5 levels per factor: (-α, -1, 0, +1, +α) where α = √k
- Include center points for pure error estimation
- Rotatable design for uniform prediction variance

### Phase 3: Validation & Fine-tuning
**Objective**: Confirm optimal settings and robustness
**Method**: Optimum seeking experiments
**Duration**: 20-30 experiments

## Specific Architecture Sampling Strategy

### Network Complexity Progression
**Problem**: Current architecture parameters show n=1 sampling

**Solution**: Systematic Complexity Ladder
```
Small:    conv_filters=[16, 32], dense_units=[64]
Medium:   conv_filters=[16, 32, 64], dense_units=[128, 64] 
Large:    conv_filters=[32, 64, 128], dense_units=[256, 128]
XLarge:   conv_filters=[64, 128, 256], dense_units=[512, 256]
```

### Pooling Strategy Matrix
```
Conservative: pool_layers=[2, 4] (every 2nd conv layer)
Balanced:     pool_layers=[1, 3, 5] (every 2nd, starting early)
Aggressive:   pool_layers=[1, 2, 3, 4] (frequent pooling)
```

## Statistical Considerations

### Power Analysis
- **Target Effect Size**: 0.05 accuracy improvement (medium effect)
- **Statistical Power**: 80% (β = 0.20)
- **Significance Level**: α = 0.05
- **Required Sample Size**: 5-8 replicates per factor level combination

### Blocking Strategy
To control confounding variables:
- **Time Block**: Run experiments in random order within time blocks
- **Hardware Block**: Balance experiments across available GPUs
- **Data Block**: Ensure consistent train/val/test splits

### Quality Control
- **Center Point Replication**: Minimum 4 replicates for pure error
- **Duplicate Experiments**: 10% of total experiments
- **Outlier Detection**: Studentized residuals > 3.0
- **Model Validation**: R² > 0.8, adequate precision > 4

## Interaction Effects Investigation

### Priority Interactions to Test
Based on domain knowledge and preliminary analysis:

1. **learning_rate × batch_size**: Training dynamics interaction
2. **dropout × l2_regularization**: Regularization synergy
3. **conv_filters × dense_units**: Architecture balance
4. **learning_rate × optimizer**: Algorithm-specific tuning

### Interaction Detection Protocol
- **Main Effects**: Single parameter sweeps
- **2FI Model**: Include all two-factor interactions
- **Screening**: Use interaction plots and ANOVA
- **Validation**: Confirm significant interactions with additional runs

## Resource Allocation

### Computational Budget
- **Phase 1**: 44 experiments × 2 hours = 88 GPU-hours
- **Phase 2**: 60 experiments × 2 hours = 120 GPU-hours  
- **Phase 3**: 25 experiments × 2 hours = 50 GPU-hours
- **Total**: ~260 GPU-hours over 3 phases

### Timeline
- **Phase 1**: 2-3 weeks (screening)
- **Phase 2**: 3-4 weeks (optimization) 
- **Phase 3**: 1-2 weeks (validation)
- **Analysis**: 1 week between phases
- **Total Duration**: 8-10 weeks

## Success Criteria

### Phase 1 (Screening)
- [ ] Identify ≥3 active factors (p < 0.05)
- [ ] Detect ≥1 significant interaction
- [ ] Achieve model R² > 0.7
- [ ] Reduce parameter space by 50%

### Phase 2 (Optimization) 
- [ ] Find optimal region with 95% confidence
- [ ] Achieve >5% accuracy improvement over baseline
- [ ] Confirm model adequacy (lack of fit p > 0.05)
- [ ] Validate robustness with confirmation runs

### Phase 3 (Validation)
- [ ] Reproduce optimal results within 1% accuracy
- [ ] Demonstrate statistical significance vs. baseline
- [ ] Generate prediction intervals for new configurations
- [ ] Document optimal hyperparameter configuration

## Implementation Guide

### Experiment Generation
```python
# Use enhanced comprehensive analyzer
python ml/comprehensive_hyperopt_analyzer.py --mode cwt --verbose

# Generate DoE experimental design
python ml/generate_doe_experiments.py --phase 1 --design fractional_factorial
```

### Data Collection Protocol
1. **Randomization**: Use Latin Square for run order
2. **Replication**: Minimum 3 repeats for center points
3. **Documentation**: Log all experimental conditions
4. **Quality Checks**: Monitor training convergence

### Analysis Workflow  
1. **Screening Analysis**: Main effects and interaction plots
2. **Model Building**: Stepwise regression, AIC/BIC selection
3. **Diagnostics**: Residual analysis, influence statistics
4. **Optimization**: Ridge analysis, response surface plots
5. **Validation**: Confirmation experiments, prediction intervals

---

*This DoE strategy follows established statistical methodology for industrial experimentation and is specifically adapted for neural network hyperparameter optimization constraints.*