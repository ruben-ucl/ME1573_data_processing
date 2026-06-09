# Reduced CWT Scalogram Feature Extraction Recommendation

## Objective

Implement a compact, robust feature extraction pipeline for CWT scalograms that captures:

- overall energy,
- spectral distribution,
- temporal structure,
- dominant oscillatory behaviour,
- band-specific energy structure.

The aim is to maximise information density while minimising feature redundancy and dimensionality.

---

# Recommended Feature Set

## 1. Global Features

Computed over the full scalogram power matrix:

\[
S(f,t) = |W(f,t)|^2
\]

### Features

| Feature | Purpose |
|---|---|
| `energy_total` | Overall signal strength |
| `entropy` | Energy distribution complexity |
| `sparsity` | Degree of energy concentration |
| `kurtosis` | Impulsiveness / burstiness |

---

# 2. Frequency Marginal Features

Compute frequency marginal:

\[
M_f(f)=\sum_t S(f,t)
\]

### Features

| Feature | Purpose |
|---|---|
| `spectral_centroid` | Mean frequency location |
| `spectral_spread` | Effective bandwidth |
| `dominant_frequency` | Frequency with highest energy |

---

# 3. Time Marginal Features

Compute time marginal:

\[
M_t(t)=\sum_f S(f,t)
\]

### Features

| Feature | Purpose |
|---|---|
| `activity_ratio` | Fraction of active/high-energy time |
| `peak_count` | Number of energetic events |
| `temporal_variance` | Temporal energy variability |

---

# 4. Ridge Features

Extract dominant ridge:

\[
f_r(t)=\arg\max_f S(f,t)
\]

### Features

| Feature | Purpose |
|---|---|
| `ridge_mean_freq` | Average dominant frequency |
| `ridge_freq_variance` | Frequency modulation/chirp behaviour |
| `ridge_smoothness` | Ridge continuity/coherence |

---

# 5. User-Defined Frequency Band Features

For each user-defined band:

```python
band_mask = (freqs >= f_low) & (freqs < f_high)
S_band = S[band_mask, :]
```

### Recommended Features Per Band

| Feature | Purpose |
|---|---|
| `band_energy_ratio` | Relative energy contribution |
| `band_entropy` | Complexity within band |

### Example Bands

```python
bands = {
    "low":  (0, 10e3),
    "mid":  (10e3, 30e3),
    "high": (30e3, 50e3),
}
```

---

# Recommended Minimal Practical Set

## Core Features (~15 total)

### Global
- `energy_total`
- `entropy`
- `sparsity`
- `kurtosis`

### Frequency Marginal
- `spectral_centroid`
- `spectral_spread`
- `dominant_frequency`

### Time Marginal
- `activity_ratio`
- `peak_count`
- `temporal_variance`

### Ridge
- `ridge_mean_freq`
- `ridge_freq_variance`
- `ridge_smoothness`

### Bands
- `low_band_energy_ratio`
- `high_band_energy_ratio`

---

# Recommended Preprocessing

## Suggested pipeline

```text
Signal
  ↓
CWT
  ↓
Power scalogram
  ↓
Log compression
  ↓
Optional normalization
  ↓
Feature extraction
```

## Recommended Transform

```python
S = np.log1p(np.abs(coeffs)**2)
```

---

# Features Not Recommended Initially

The following are often redundant, unstable, or low-value relative to dimensionality:

- mean/std/min/max
- large percentile sets
- higher-order moments
- fractal features
- Haralick/GLCM texture features
- LBP features
- edge-density metrics

Add only if baseline performance plateaus.

---

# Recommended Priorities

If further dimensionality reduction is required, prioritise:

1. band energy ratios
2. entropy
3. spectral centroid
4. ridge frequency variance
5. activity ratio

These capture most of the discriminative structure in many CWT-based classification tasks.
