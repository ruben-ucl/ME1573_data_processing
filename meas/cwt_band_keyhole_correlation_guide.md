# CWT Feature Extraction and Keyhole Correlation Analysis
## Course Guide — Signal Processing for Additive Manufacturing

---

## 1. Problem Context

In laser powder bed fusion (LPBF) additive manufacturing, a high-power laser melts successive layers of metal powder to build a three-dimensional part. Under certain process conditions the laser creates a **keyhole**: a deep, narrow vapour cavity that forms when the energy density is high enough to vaporise the molten metal. Keyhole geometry — depth, area, length — is directly linked to print quality. Deep, unstable keyholes collapse and trap gas, creating subsurface porosity that degrades mechanical properties.

We cannot observe the keyhole directly during a live build (X-ray synchrotron imaging exists but is impractical industrially). However, an **in-process photodiode** pointed at the melt pool records a continuous electrical signal. This signal encodes acoustic and optical emissions from the process — fluctuations that carry information about keyhole state.

The central question of this analysis is:

> **Which time-frequency features of the photodiode signal are most predictive of keyhole geometry, and at what frequency scales?**

Answering this question is the first step toward building a real-time keyhole monitor using only the photodiode — a sensor that is cheap, fast, and industrially deployable.

---

## 2. The Raw Signal and Its Time-Frequency Structure

### 2.1 Why standard statistics are insufficient

The photodiode records a 1-D time series sampled at approximately 100–500 kHz. A naive approach would compute the mean or variance of a short signal window and correlate it with keyhole depth. This fails because:

- The signal contains **multiple overlapping physical mechanisms** active at different frequencies (acoustic resonance, vapour plume oscillation, laser-material coupling fluctuations).
- A single window mean conflates all of these — a high mean could come from sustained low-frequency rumble or from intermittent high-frequency bursts.
- **When** and **where in frequency** the energy occurs matters, not just how much there is in total.

A richer representation is required: one that simultaneously resolves both time and frequency.

### 2.2 The Short-Time Fourier Transform — and its limitation

The Short-Time Fourier Transform (STFT) applies a Fourier transform to a sliding window of the signal. For a signal $x(t)$, with window function $g$:

$$\text{STFT}(\tau, \omega) = \int_{-\infty}^{\infty} x(t)\, g(t - \tau)\, e^{-i\omega t}\, dt$$

The result is a 2-D map of frequency content versus time. The problem is the **Heisenberg–Gabor uncertainty principle**: you cannot simultaneously have arbitrarily fine resolution in both time and frequency. A long window gives good frequency resolution but smears transient events in time. A short window captures transients but blurs frequency information.

In practice, different physical phenomena operate at different time scales — a slow thermal drift unfolds over milliseconds while a high-frequency acoustic resonance rings for a few microseconds. No single STFT window suits both.

### 2.3 The Continuous Wavelet Transform (CWT)

The CWT solves the fixed-window problem by using **scale-adaptive windows**: at low frequencies (large scales) the window is wide, capturing slowly-varying structure; at high frequencies (small scales) the window is narrow, capturing transients precisely.

For a signal $x(t)$, the CWT with mother wavelet $\psi$ is:

$$W(a, b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t)\, \psi^*\!\left(\frac{t - b}{a}\right) dt$$

where $a > 0$ is the **scale** (inversely related to frequency) and $b$ is the **translation** (time position). The factor $1/\sqrt{a}$ normalises energy across scales.

The **scalogram** is $|W(a, b)|^2$ — the squared magnitude — which gives the power distribution across scale and time.

#### The complex Morlet wavelet

This analysis uses the **complex Morlet wavelet** (`cmor1.5-1.0` in PyWavelets notation), defined as:

$$\psi(t) = \frac{1}{\sqrt{\pi B}} e^{2\pi i f_c t} e^{-t^2/B}$$

where $f_c$ is the centre frequency and $B$ is the bandwidth. The Morlet wavelet is a complex sinusoid modulated by a Gaussian envelope. Its properties make it well-suited to this application:

- **Good frequency localisation**: the Gaussian envelope limits spectral leakage.
- **Good time localisation at high frequencies**: the window narrows at small scales.
- **Analytic signal**: the imaginary part is the Hilbert transform of the real part, so $|W(a,b)|$ directly measures instantaneous amplitude at each scale — no need to separate carrier from envelope.

#### Scale–frequency relationship

For the Morlet wavelet, scale and centre frequency are related by:

$$f = \frac{f_c}{a \cdot \Delta t}$$

where $\Delta t$ is the sampling period. This allows us to specify the analysis in physically meaningful frequency bands (Hz) rather than abstract scales.

### 2.4 Padding to reduce edge artefacts

The CWT is a convolution. Near the start and end of the signal, part of the wavelet extends beyond the data — the **cone of influence** — producing unreliable estimates. The script mitigates this with **symmetric padding**: the signal is reflected about each end before computing the CWT, then the padded regions are discarded. With `pad_factor = 2`, each end is padded by $2N$ samples (where $N$ is the signal length), ensuring the central analysis region is free of boundary effects.

---

## 3. Windowing Strategy

Rather than computing the CWT over an entire track (which may contain tens of millions of samples), the analysis operates on **short, overlapping windows** of 1 ms duration with a 0.2 ms step. This serves two purposes:

1. **Temporal localisation**: keyhole geometry measurements are associated with specific time stamps. By windowing the signal we create time-localised feature vectors that can be matched against instantaneous keyhole state.

2. **Statistical power**: each window becomes one row in the final dataset. Many windows per track means sufficient samples for robust correlation analysis even with a small number of physical tracks.

The first and last windows of each track are discarded to avoid residual edge effects and signal transients at track start/end.

---

## 4. Log-Compression of Power

Before computing any features, the raw scalogram amplitudes are log-compressed:

$$S = \log(1 + |W|^2)$$

This is not merely a cosmetic transformation. It is physically motivated.

Acoustic and optical emission signals in manufacturing tend to follow **lognormal amplitude distributions** — the signal can fluctuate over several orders of magnitude. Linear power would be dominated by rare, very large spikes, causing means and standard deviations to be unstable estimators.

Log-compression maps multiplicative variation (factors of 10, 100, …) onto additive variation, which is what our statistical features are designed for. It also mirrors human auditory perception (the decibel scale) and is standard in speech processing, sonar, and vibration analysis.

The $+1$ inside the logarithm (`log1p`) ensures the function is defined when $|W|^2 = 0$ and avoids numerical issues near zero without altering large values appreciably.

---

## 5. Frequency Band Decomposition

The CWT scalogram covers 1–50 kHz divided into six octave-like bands:

| Band | Range |
|------|-------|
| 1–2 kHz   | Low-frequency melt pool oscillations |
| 2–4 kHz   | Acoustic resonance, plume dynamics |
| 4–8 kHz   | Mid-band structural resonance |
| 8–16 kHz  | High-frequency acoustic emission |
| 16–32 kHz | Ultrasonic regime |
| 32–50 kHz | Near-ultrasonic, fine-scale emission |

Dividing the scalogram into bands and computing separate statistics for each is motivated by **physics**: different mechanisms couple to the keyhole at different frequency ranges. By keeping bands separate we preserve this physical specificity. Aggregating the whole spectrum into a single number would discard it.

---

## 6. Per-Band Features

For each frequency band and each time window, the following statistics are computed on $S_\text{band}$, the log-compressed power within that band.

### 6.1 Mean

$$\mu = \frac{1}{N} \sum_{k} S_k$$

The arithmetic mean of the log-power. Captures the **typical signal level** in the band. Because we are in log-space, this is the geometric mean of raw power — it is robust to outliers in a way the linear mean is not.

### 6.2 Standard deviation

$$\sigma = \sqrt{\frac{1}{N} \sum_k (S_k - \mu)^2}$$

Measures **variability** within the window. A large $\sigma$ means the energy in that band fluctuates strongly across time and scale — indicative of intermittent, bursty events. A small $\sigma$ means the energy is steady.

### 6.3 Min and max

The minimum and maximum log-power values in the band. Together with mean and std, they characterise the range. Max is particularly sensitive to brief, high-intensity bursts.

### 6.4 Median

The middle value of the sorted distribution. Unlike the mean, the median is unaffected by extreme outliers. Comparing mean and median reveals **skewness**: if mean > median, the distribution has a heavy upper tail (occasional bursts dominate).

### 6.5 Energy

$$E = \sum_k S_k$$

The total log-power summed over all time steps and frequency bins within the band. While proportional to mean for a fixed window size, energy is more natural when comparing windows of varying length or when aggregating across scales.

### 6.6 Energy ratio

$$\rho = \frac{E_\text{band}}{E_\text{total}}$$

The fraction of total scalogram energy residing in this frequency band. This is a **normalised** feature: it is invariant to overall signal amplitude, capturing only the **spectral shape** — how energy is distributed across bands regardless of whether the signal is loud or quiet.

This is particularly valuable for correlation analysis because overall signal level may vary between tracks due to sensor positioning or laser power settings, while the spectral shape may still carry the keyhole signature.

### 6.7 Band entropy

$$H_\text{band} = -\sum_k p_k \log(p_k + \varepsilon), \quad p_k = \frac{S_k}{\sum_j S_j + \varepsilon}$$

Shannon entropy quantifies **spread** of energy within the band. If energy is concentrated at one scale and one time instant, $p_k$ is peaked and $H$ is small. If energy is spread uniformly across all scales and times, $H$ is maximised.

Entropy is thus a measure of signal **complexity** or **unpredictability** within a band. A collapsing keyhole might produce concentrated bursts of energy (low entropy); a stable keyhole might produce diffuse, broadband noise (high entropy).

---

## 7. Global Features

These are computed once per window from the entire scalogram slice (all frequency bands together). They capture properties of the **whole time-frequency distribution** that no single band can represent.

Let $S \in \mathbb{R}^{F \times T}$ be the full log-compressed scalogram for the window, with $F$ frequency bins and $T$ time steps.

Define the two **marginals**:

$$M_f[i] = \sum_j S[i,j] \quad \text{(total power at frequency bin } i\text{)}$$

$$M_t[j] = \sum_i S[i,j] \quad \text{(total power at time step } j\text{)}$$

### 7.1 CWT entropy (`cwt_entropy`)

$$H = -\sum_{i,j} p_{ij} \log(p_{ij} + \varepsilon), \quad p_{ij} = \frac{S[i,j]}{\sum_{i,j} S[i,j]}$$

The Shannon entropy of the full scalogram, treating it as a 2-D probability distribution over time-frequency space. This is the most general measure of **energy spread**: a small value means energy is concentrated at a specific (time, frequency) point; a large value means it is diffuse. This is sensitive to structure that band-level entropy cannot capture — for instance, energy that is simultaneously concentrated in both time and frequency but spread across multiple bands.

### 7.2 Kurtosis (`cwt_kurtosis`)

$$\kappa = \frac{\mathbb{E}[(S - \mu)^4]}{\sigma^4} - 3$$

(Excess kurtosis, where $-3$ makes the Gaussian kurtosis equal zero.)

Kurtosis measures **tail heaviness** of the distribution of log-power values. A high positive kurtosis means the distribution has heavy tails — rare but extreme events dominate. In signal processing terms, high kurtosis indicates **impulsiveness**: a signal that is mostly quiet with occasional sharp bursts, which is a known signature of keyhole instability and vapour plume ejection events.

### 7.3 Spectral centroid (`cwt_spectral_centroid`)

$$\bar{f} = \frac{\sum_i f_i \cdot M_f[i]}{\sum_i M_f[i]}$$

A weighted average of frequency, with weights given by the marginal power at each frequency. The spectral centroid is the **centre of mass** of the spectrum. A high centroid means most energy resides at high frequencies; a low centroid means low-frequency content dominates.

In manufacturing acoustics, the spectral centroid tends to shift with process state — keyhole mode operation produces different acoustic profiles than conduction mode, and the centroid encodes this shift in a single scalar.

### 7.4 Spectral spread (`cwt_spectral_spread`)

$$\sigma_f = \sqrt{\frac{\sum_i M_f[i] \cdot (f_i - \bar{f})^2}{\sum_i M_f[i]}}$$

The power-weighted standard deviation of frequency around the centroid. This measures **bandwidth**: a narrow spread means energy is concentrated in a thin frequency band (a tonal signal); a wide spread means broadband noise.

Spectral centroid and spread together describe the location and width of the spectral distribution — analogous to mean and standard deviation, but in the frequency domain.

### 7.5 Dominant frequency (`cwt_dominant_freq`)

$$f^* = f[\arg\max_i M_f[i]]$$

Simply the frequency bin carrying the most total power across the window. Unlike the centroid, this is the **mode** of the frequency distribution — it is more sensitive to sharp spectral peaks and less influenced by broadband background noise.

### 7.6 Activity ratio (`cwt_activity_ratio`)

$$\alpha = \frac{1}{T} \sum_j \mathbf{1}\left[M_t[j] > \overline{M_t}\right]$$

The fraction of time steps in the window where total power exceeds the window mean. If activity is intermittent — some time steps are very active, others quiet — this ratio will be below 0.5 (because the active steps push the mean up, leaving most steps below it). A value near 0.5 implies roughly symmetric activity.

This feature quantifies the **duty cycle** of acoustic activity: is the signal persistently elevated, or does it burst on and off?

### 7.7 Temporal variance (`cwt_temporal_variance`)

$$\sigma_t = \text{std}(M_t)$$

The standard deviation of the temporal marginal power. High temporal variance means the signal power fluctuates strongly from time step to time step — a hallmark of intermittent, non-stationary processes like keyhole oscillation.

### 7.8 Peak count (`cwt_peak_count`)

The number of local maxima in $M_t$ that exceed the mean:

$$N_\text{peaks} = \left|\left\{j : M_t[j] \text{ is a local maximum and } M_t[j] > \overline{M_t}\right\}\right|$$

Each peak corresponds to a burst of acoustic energy. A high peak count within a 1 ms window suggests rapid, repeated energy releases — consistent with a fluctuating or collapsing keyhole. This feature has no analogue in standard spectral analysis; it is only accessible because the CWT preserves the temporal profile of energy.

### 7.9 Ridge mean frequency (`cwt_ridge_mean_freq`)

At each time step $j$, the **instantaneous dominant frequency** is:

$$f_\text{ridge}[j] = f\left[\arg\max_i S[i, j]\right]$$

The collection of $f_\text{ridge}[j]$ across all time steps traces a path through the scalogram called the **ridge**. The ridge tracks the most energetically dominant frequency as it evolves through time.

The ridge mean frequency is:

$$\bar{f}_\text{ridge} = \frac{1}{T}\sum_j f_\text{ridge}[j]$$

This is subtly different from the spectral centroid: the centroid reflects the time-averaged spectrum, while the ridge mean reflects where power is concentrated at each instant. They diverge when the dominant frequency varies over time.

### 7.10 Ridge frequency standard deviation (`cwt_ridge_freq_std`)

$$\sigma_\text{ridge} = \text{std}(f_\text{ridge})$$

How much the instantaneous dominant frequency wanders over the window. A large $\sigma_\text{ridge}$ means the dominant acoustic mode switches between frequencies — the signal is **non-stationary** in its dominant component. This is a qualitatively different measure from spectral spread: a signal can have a wide spectrum (high spread) but a stable ridge (low $\sigma_\text{ridge}$) if the broadband noise is constant while a single dominant tone stays fixed.

### 7.11 Ridge smoothness (`cwt_ridge_smoothness`)

$$\rho_\text{smooth} = \frac{1}{1 + \text{std}(\Delta f_\text{ridge})}$$

where $\Delta f_\text{ridge}[j] = f_\text{ridge}[j+1] - f_\text{ridge}[j]$ is the step-to-step change in ridge frequency.

This measures how **continuously** the dominant frequency evolves. A value near 1 means the ridge changes only gradually — the dominant acoustic mode drifts smoothly. A value near 0 means the ridge jumps erratically between frequencies from step to step. In keyhole dynamics, smooth ridge evolution might suggest a stable oscillation mode while erratic jumping could indicate chaotic keyhole behaviour.

---

## 8. Keyhole Geometry Statistics

For each time window the script extracts matched keyhole measurements from X-ray imaging data stored alongside the photodiode signal. Five geometric quantities are used:

| Variable | Physical meaning |
|---|---|
| `max_depth` | Maximum keyhole depth in the window — primary indicator of keyhole regime |
| `max_length` | Maximum keyhole length (horizontal extent) |
| `area` | Projected cross-sectional area — integrates depth and length |
| `depth_at_max_length` | Depth at the moment of maximum length — captures asymmetric dynamics |
| `fkw_angle` | Front keyhole wall angle — related to energy absorption geometry |

For each keyhole variable in each time window, the mean (and standard deviation, min, max, median) of all measurements falling within the window are recorded. The **mean** is used as the primary target for correlation analysis because it is the most stable aggregate — individual keyhole measurements can be noisy, and averaging over a window reduces this noise.

---

## 9. Correlation Analysis

### 9.1 Pearson correlation coefficient

For each (CWT feature, KH target) pair, the **Pearson correlation coefficient** is computed:

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2 \cdot \sum_i (y_i - \bar{y})^2}}$$

where $x_i$ are the CWT feature values and $y_i$ are the KH target values across all $n$ windows.

$r$ lies in $[-1, +1]$:
- $r = +1$: perfect positive linear relationship (feature increases as target increases)
- $r = -1$: perfect negative linear relationship (feature increases as target decreases)
- $r = 0$: no linear relationship

The **sign** of $r$ is physically informative. A negative correlation between `cwt_kurtosis` and `max_depth`, for example, would mean deeper keyholes produce less impulsive signals — not obvious a priori, and only visible by preserving the sign.

### 9.2 Coefficient of determination R²

The square of the Pearson coefficient:

$$R^2 = r^2 \in [0, 1]$$

$R^2$ is used for ranking features because it gives an unambiguous measure of **proportion of variance explained**. An $R^2$ of 0.3 means 30% of the variance in the KH target can be linearly predicted from the feature. $R^2$ discards sign, so it is used only for ranking, not for understanding direction of correlation.

### 9.3 Why linear correlation first?

You might wonder whether Pearson correlation, which only measures **linear** association, is the right tool given that the physical relationship between acoustic features and keyhole geometry could be nonlinear.

This is a valid concern. However, computing Pearson $r$ as a first pass is standard practice in feature selection for several reasons:

1. **Speed**: computing $r$ for every (feature, target) pair is $O(n)$ and runs in seconds even on large datasets.
2. **Interpretability**: a high $|r|$ immediately tells you the feature is linearly predictive. A low $|r|$ does not rule out nonlinear relationships, but features with high linear correlations are always worth including in any model.
3. **Baseline**: linear correlation benchmarks against which more complex methods (mutual information, spearman $\rho$, nonlinear regression) can be compared. If your neural network achieves 70% accuracy but a linear model on the top-r² features achieves 65%, you have learned something.

---

## 10. Visualisations

### 10.1 Signed Pearson-r heatmap

This figure displays the full matrix of Pearson $r$ values for all CWT features (rows) against all KH targets (columns), using a diverging colormap (red = positive correlation, blue = negative, white = zero).

Reading the heatmap:

- **Row groups** separate feature types (global, per-band energy ratio, entropy, etc.). Comparing groups reveals which class of feature is most informative overall.
- **Column structure** reveals which KH targets are well-predicted by acoustic signals. A column that is mostly white means that target is difficult to predict from the photodiode alone.
- **Colour asymmetry within a group** reveals which frequency bands carry the correlation signal. If the 4–8 kHz band has strong red cells but the 32–50 kHz band is white, you have a physically specific finding: the relevant acoustic mechanism operates at mid-frequencies.
- **Sign patterns**: a feature that is positively correlated with `max_depth` but negatively with `fkw_angle` suggests these two geometric quantities respond inversely to the same acoustic mechanism.

### 10.2 Band × feature-type matrix

One figure is produced per KH target. Rows are frequency bands (low to high), columns are feature types (energy ratio, entropy, mean, std, energy). Each cell shows the Pearson $r$ for that (band, feature) combination against the chosen KH target.

This figure answers a specific question: *for this keyhole variable, does a simple statistic (mean) in the right frequency band perform as well as a more complex feature (entropy ratio), and which band matters most?*

If the top row (1–2 kHz) is consistently dark red while the bottom rows (32–50 kHz) are white, this suggests the relevant physics is a low-frequency oscillation, and higher-frequency measurements add noise rather than signal. This has direct implications for instrumentation design: if only 1–4 kHz matters, a lower-bandwidth sensor might suffice.

---

## 11. Summary of Feature Design Rationale

| Feature class | What it captures | When it is useful |
|---|---|---|
| Band mean/std | Average level and variability in a frequency range | Baseline; often surprisingly effective |
| Band energy ratio | Spectral shape independent of amplitude | Removes sensor gain variation between tracks |
| Band entropy | Complexity/spread within a band | Distinguishes diffuse noise from tonal emissions |
| Spectral centroid | Centre of mass of the spectrum | Tracks shifts in dominant mechanism frequency |
| Spectral spread | Bandwidth of the emission | Tonal vs broadband |
| Dominant frequency | Mode of the spectrum | Sharp peaks; less influenced by noise floor |
| Kurtosis | Impulsiveness of the full scalogram | Keyhole collapse events |
| Activity ratio | Temporal duty cycle | Intermittent vs sustained activity |
| Temporal variance | How much power fluctuates over time | Oscillatory vs steady emission |
| Peak count | Number of energy bursts in window | Event rate; rapid keyhole dynamics |
| Ridge mean/std | Average and stability of instantaneous dominant frequency | Non-stationarity of the acoustic mode |
| Ridge smoothness | Continuity of frequency evolution | Chaotic vs ordered dynamics |

These features collectively span the major dimensions of time-frequency structure: **level**, **shape**, **spread**, **impulsiveness**, **temporal dynamics**, and **frequency evolution**. A machine learning model trained on these features — rather than raw amplitude statistics — is equipped to discover structure in the acoustic signal that directly reflects keyhole physics.

---

## 12. Further Reading

- Mallat, S. (2009). *A Wavelet Tour of Signal Processing*. Academic Press. — Comprehensive mathematical treatment of wavelets and the CWT.
- Cohen, L. (1995). *Time-Frequency Analysis*. Prentice Hall. — Covers the Heisenberg uncertainty principle and time-frequency representations.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*. — Original paper on information entropy.
- Hyer, H. et al. (2022). Laser powder bed fusion keyhole detection via in-situ photodiode monitoring. *Additive Manufacturing*. — Practical context for the analysis performed here.
- Scikit-learn documentation: Feature selection — https://scikit-learn.org/stable/modules/feature_selection.html — includes the role of correlation-based feature ranking in ML pipelines.
