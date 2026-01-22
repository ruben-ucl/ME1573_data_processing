# Auto-Alignment Analysis and Proposed Improvements

## Current Issues

The auto-alignment cross-correlation is finding lag values that don't match the visual best alignment when looking at the plotted signals.

## Potential Root Causes

### 1. **Windowing Problems**
- **Current**: Uses a small window (0.001s = 1ms) from center of signal
- **Issue**: This window might not capture representative features
- **Solution**: Try multiple windows or use full signal with tapering

### 2. **Signal Preprocessing Mismatch**
- **Current**: Applies normalization (zero-mean, unit variance)
- **Issue**: PD and KH signals have very different characteristics:
  - PD: High frequency, sharp transitions, relatively clean
  - KH: Lower frequency, gradual changes, more noise
- **Solution**:
  - Apply bandpass filtering to focus on common frequency range
  - Use envelope detection for KH signals
  - Apply derivative to emphasize edges/transitions

### 3. **Ambiguous Correlation Peaks**
- **Issue**: Multiple similar correlation peaks can lead to wrong lag selection
- **Solution**:
  - Plot correlation function to visualize all peaks
  - Use peak prominence/width criteria
  - Consider physical constraints (max reasonable lag)

### 4. **Anti-Phase Correlation**
- **Issue**: Signals might be negatively correlated at correct lag
- **Current**: Uses absolute value of correlation
- **Solution**: Check sign of correlation peak

### 5. **Time-Varying Lag**
- **Issue**: Optimal lag might vary across the signal duration
- **Solution**:
  - Use sliding window correlation
  - Report lag uncertainty/variance

## Proposed Diagnostic Additions

1. **Save correlation plots** showing:
   - Full cross-correlation function
   - Detected peak
   - Alternative peaks
   - Window used for correlation

2. **Add correlation quality metrics**:
   - Peak height
   - Peak prominence (vs next highest peak)
   - Peak width
   - Signal-to-noise ratio

3. **Multi-scale correlation**:
   - Try correlation at different frequency scales
   - Use wavelet cross-correlation

## Proposed Improvements (Priority Order)

### High Priority

1. **Add diagnostic plotting**
   - Plot cross-correlation function
   - Mark detected peak
   - Save to alignment output directory

2. **Use envelope detection for KH signals**
   - Extract signal envelope using Hilbert transform
   - Correlate PD with KH envelope

3. **Add peak quality checks**
   - Reject alignment if peak prominence < threshold
   - Warn user if multiple similar peaks exist

### Medium Priority

4. **Bandpass filtering before correlation**
   - Find common frequency range
   - Apply matched filtering

5. **Multi-window correlation**
   - Correlate multiple windows
   - Use median/consensus lag

### Low Priority

6. **Adaptive correlation method selection**
   - Auto-detect best method based on signal characteristics
   - Try multiple methods and select best

7. **Machine learning approach**
   - Train alignment model on manually corrected examples
   - Use features beyond simple correlation

## Implementation Plan

1. ✅ **DONE**: Add manual lag correction option
2. **NEXT**: Add diagnostic correlation plotting
3. **THEN**: Implement envelope detection option
4. **FINALLY**: Add peak quality metrics
