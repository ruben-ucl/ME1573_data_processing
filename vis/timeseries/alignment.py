"""
Time series alignment operations.

This module contains all alignment-related functionality including
cross-correlation, time series synchronization, and cropping operations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr

# Try to import statsmodels for cross-correlation function
try:
    from statsmodels.tsa.stattools import ccf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from .config import DatasetConfig


class AlignmentMixin:
    """
    Mixin class containing all time series alignment operations.

    This class expects the following attributes to be available from the parent:
    - self.datasets: List[DatasetConfig]
    - self.processed_data: Dict[str, np.ndarray]
    - self.time_vectors: Dict[str, np.ndarray]
    - self.original_time_vectors: Dict[str, np.ndarray]
    - self.alignment_info: Dict[str, Dict]
    - self.sampling_rates: Dict[str, float]
    - self.raw_data: Dict[str, np.ndarray]
    - self.full_processed_data: Dict[str, np.ndarray]
    """

    def _compute_cross_correlation(self, signal1: np.ndarray, signal2: np.ndarray,
                                  max_lag: Optional[int] = None,
                                  method: str = 'statsmodels') -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Compute cross-correlation between two signals with lag detection.

        This is a unified cross-correlation function used for:
        - Signal alignment/synchronization
        - Lag detection and plotting
        - Cross-correlation analysis

        Checks both positive and negative lags to find the optimal alignment.

        Args:
            signal1: Reference signal (y = signal1)
            signal2: Target signal (x = signal2)
            max_lag: Maximum lag to check (in samples). If None, uses full signal length
            method: 'statsmodels' (robust, uses CCF) or 'numpy' (fast, uses correlate)

        Returns:
            Tuple of (lags, correlation_values, peak_lag):
            - lags: Array of lag values (negative = signal2 leads, positive = signal2 lags)
            - correlation_values: Cross-correlation at each lag
            - peak_lag: Lag with maximum correlation (samples to shift signal2)
        """
        n1, n2 = len(signal1), len(signal2)

        # Determine max lag
        if max_lag is None:
            max_lag = min(n1, n2) - 1
        else:
            max_lag = min(max_lag, min(n1, n2) - 1)

        if method == 'statsmodels' and STATSMODELS_AVAILABLE:
            # Use statsmodels CCF (most robust, handles normalization properly)
            ccf_result = ccf(signal2, signal1, nlags=max_lag, adjusted=False)

            # CCF returns [0, 1, 2, ...max_lag] for positive lags
            # We need to compute negative lags by reversing signal roles
            ccf_negative = ccf(signal1, signal2, nlags=max_lag, adjusted=False)

            # Combine: negative lags (reversed, excluding zero) + positive lags (including zero)
            # ccf_negative[max_lag:0:-1] gives us lags from -1 to -max_lag
            # ccf_result gives us lags from 0 to max_lag
            correlation_values = np.concatenate([ccf_negative[max_lag:0:-1], ccf_result])

            # Create symmetric lags array to match correlation_values length
            lags = np.arange(-max_lag, max_lag + 1)

            # Ensure lags and correlation_values have the same length
            if len(lags) != len(correlation_values):
                min_len = min(len(lags), len(correlation_values))
                lags = lags[:min_len]
                correlation_values = correlation_values[:min_len]

        else:
            # Fallback to numpy correlate (faster but less robust)
            s1_zm = signal1 - np.mean(signal1)
            s2_zm = signal2 - np.mean(signal2)

            # Normalize by standard deviations
            std1, std2 = np.std(s1_zm), np.std(s2_zm)
            if std1 > 0:
                s1_zm /= std1
            if std2 > 0:
                s2_zm /= std2

            # Full cross-correlation
            correlation_full = np.correlate(s1_zm, s2_zm, mode='full')

            # Extract relevant portion based on max_lag
            center_idx = len(correlation_full) // 2
            start_idx = max(0, center_idx - max_lag)
            end_idx = min(len(correlation_full), center_idx + max_lag + 1)

            correlation_values = correlation_full[start_idx:end_idx]
            lags = np.arange(-max_lag, max_lag + 1)[:len(correlation_values)]

            # Normalize correlation
            n_overlap = min(n1, n2)
            correlation_values /= n_overlap

        # Find peak correlation (best alignment)
        peak_idx = np.argmax(np.abs(correlation_values))
        peak_lag = lags[peak_idx]

        return lags, correlation_values, peak_lag

    def _detect_signal_features(self, signal: np.ndarray,
                                feature_method: str = 'peak',
                                n_features: int = 5,
                                prominence_threshold: Optional[float] = None,
                                envelope_smoothing: int = 3) -> np.ndarray:
        """
        Detect prominent features in a signal for alignment.

        Args:
            signal: Input signal
            feature_method: 'peak' (maxima), 'edge' (transitions), or 'energy' (energy bursts)
            n_features: Number of features to detect
            prominence_threshold: Minimum prominence for peak detection (auto if None)
            envelope_smoothing: Smoothing window for envelope computation

        Returns:
            Array of feature indices (sample positions)
        """
        from scipy.signal import hilbert, find_peaks
        from scipy.ndimage import median_filter

        # Compute signal envelope using Hilbert transform
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)

        # Apply smoothing to envelope if requested
        if envelope_smoothing > 1:
            envelope = median_filter(envelope, size=envelope_smoothing, mode='nearest')

        if feature_method == 'peak':
            # Detect peaks in envelope
            # Auto-compute prominence threshold using MAD
            if prominence_threshold is None:
                envelope_median = np.median(envelope)
                envelope_mad = np.median(np.abs(envelope - envelope_median))
                prominence_threshold = envelope_median + 2.0 * envelope_mad

            # Find peaks with minimum prominence
            peaks, properties = find_peaks(envelope, prominence=prominence_threshold)

            # Sort by prominence and take top n_features
            if len(peaks) > 0:
                prominences = properties['prominences']
                top_indices = np.argsort(prominences)[-n_features:]
                features = peaks[top_indices]
                features = np.sort(features)  # Return in temporal order
            else:
                # Fallback: use maximum values if no peaks found
                features = np.argsort(envelope)[-n_features:]
                features = np.sort(features)

        elif feature_method == 'edge':
            # Detect edges using gradient
            gradient = np.gradient(envelope)
            abs_gradient = np.abs(gradient)

            # Find top n_features gradient positions
            features = np.argsort(abs_gradient)[-n_features:]
            features = np.sort(features)

        elif feature_method == 'energy':
            # Detect energy bursts using windowed energy
            window_size = max(len(signal) // 100, 10)
            energy = np.convolve(signal**2, np.ones(window_size)/window_size, mode='same')

            # Find top n_features energy positions
            features = np.argsort(energy)[-n_features:]
            features = np.sort(features)

        else:
            raise ValueError(f"Unknown feature_method: {feature_method}")

        return features

    def _align_by_feature_matching(self, signal1: np.ndarray, signal2: np.ndarray,
                                   feature_method: str = 'peak',
                                   n_features: int = 5,
                                   max_lag: Optional[int] = None) -> Tuple[int, float, Dict]:
        """
        Align two signals by matching temporal positions of detected features.

        This method:
        1. Detects prominent features in both signals (peaks, edges, or energy bursts)
        2. Computes average temporal offset between corresponding features
        3. Returns the lag that aligns the features

        Args:
            signal1: Reference signal
            signal2: Target signal to align
            feature_method: 'peak', 'edge', or 'energy'
            n_features: Number of features to detect and match
            max_lag: Maximum allowed lag (samples), None for unlimited

        Returns:
            Tuple of (lag, quality_score, diagnostics):
            - lag: Optimal lag in samples (positive = signal2 lags behind signal1)
            - quality_score: Alignment quality metric (0-1, higher is better)
            - diagnostics: Dict with detected features and matching info
        """
        # Detect features in both signals
        features1 = self._detect_signal_features(signal1, feature_method=feature_method,
                                                 n_features=n_features)
        features2 = self._detect_signal_features(signal2, feature_method=feature_method,
                                                 n_features=n_features)

        # Compute all pairwise temporal offsets
        n_feat1 = len(features1)
        n_feat2 = len(features2)

        if n_feat1 == 0 or n_feat2 == 0:
            # Fallback: no features detected
            return 0, 0.0, {'features1': features1, 'features2': features2,
                           'error': 'No features detected'}

        # Compute pairwise lags (signal2 index - signal1 index)
        # Positive lag means signal2 feature occurs later than signal1 feature
        pairwise_lags = []
        for f2 in features2:
            for f1 in features1:
                lag = f2 - f1
                if max_lag is None or abs(lag) <= max_lag:
                    pairwise_lags.append(lag)

        if len(pairwise_lags) == 0:
            # All lags exceed max_lag
            return 0, 0.0, {'features1': features1, 'features2': features2,
                           'error': 'All lags exceed max_lag'}

        # Use median lag (robust to outliers)
        optimal_lag = int(np.median(pairwise_lags))

        # Compute quality score based on lag consistency
        # Low MAD of lags = high consistency = high quality
        lag_mad = np.median(np.abs(np.array(pairwise_lags) - optimal_lag))
        lag_spread = np.ptp(pairwise_lags) if len(pairwise_lags) > 1 else 0

        # Quality score: inverse of normalized spread (0-1 range)
        # Good alignment: low spread → high quality
        if lag_spread > 0:
            quality_score = 1.0 / (1.0 + lag_spread / (0.1 * len(signal1)))
        else:
            quality_score = 1.0

        # Diagnostic information
        diagnostics = {
            'features1': features1,
            'features2': features2,
            'pairwise_lags': np.array(pairwise_lags),
            'optimal_lag': optimal_lag,
            'lag_mad': lag_mad,
            'lag_spread': lag_spread,
            'n_matches': len(pairwise_lags)
        }

        return optimal_lag, quality_score, diagnostics

    def _compute_mutual_information(self, signal1: np.ndarray, signal2: np.ndarray,
                                   n_bins: int = 50) -> float:
        """
        Compute mutual information between two signals.

        MI measures how much information one signal provides about the other.
        Higher MI indicates stronger relationship/alignment.

        Args:
            signal1: First signal
            signal2: Second signal
            n_bins: Number of bins for histogram (affects MI resolution)

        Returns:
            Mutual information value (higher = more information shared)
        """
        from sklearn.metrics import mutual_info_score

        # Ensure signals have same length
        min_len = min(len(signal1), len(signal2))
        s1 = signal1[:min_len]
        s2 = signal2[:min_len]

        # Discretize signals into bins for MI computation
        s1_binned = np.digitize(s1, bins=np.linspace(s1.min(), s1.max(), n_bins))
        s2_binned = np.digitize(s2, bins=np.linspace(s2.min(), s2.max(), n_bins))

        # Compute mutual information
        mi = mutual_info_score(s1_binned, s2_binned)

        return mi

    def _align_by_mutual_information(self, signal1: np.ndarray, signal2: np.ndarray,
                                    max_lag: Optional[int] = None,
                                    n_bins: int = 50,
                                    lag_step: int = 1) -> Tuple[int, float, Dict]:
        """
        Align two signals by maximizing mutual information across lag range.

        This method:
        1. Shifts signal2 relative to signal1 across lag range
        2. Computes MI at each lag
        3. Returns the lag with maximum MI

        Args:
            signal1: Reference signal
            signal2: Target signal to align
            max_lag: Maximum lag to search (samples), None uses 20% of signal length
            n_bins: Number of bins for MI histogram
            lag_step: Step size for lag search (1=every sample, larger=faster)

        Returns:
            Tuple of (lag, mi_score, diagnostics):
            - lag: Optimal lag in samples (positive = signal2 lags behind signal1)
            - mi_score: Mutual information at optimal lag (higher is better)
            - diagnostics: Dict with MI curve and search info
        """
        n1, n2 = len(signal1), len(signal2)

        # Determine max lag
        if max_lag is None:
            max_lag = min(n1, n2) // 5  # Search ±20% of signal length
        else:
            max_lag = min(max_lag, min(n1, n2) - 1)

        # Generate lag range to search
        lags = np.arange(-max_lag, max_lag + 1, lag_step)
        mi_scores = np.zeros(len(lags))

        # Compute MI at each lag
        for i, lag in enumerate(lags):
            if lag == 0:
                # No shift
                overlap_len = min(n1, n2)
                s1_overlap = signal1[:overlap_len]
                s2_overlap = signal2[:overlap_len]
            elif lag > 0:
                # signal2 lags behind signal1
                overlap_len = min(n1 - lag, n2)
                s1_overlap = signal1[lag:lag + overlap_len]
                s2_overlap = signal2[:overlap_len]
            else:
                # signal2 leads signal1 (negative lag)
                overlap_len = min(n1, n2 + lag)
                s1_overlap = signal1[:overlap_len]
                s2_overlap = signal2[-lag:-lag + overlap_len]

            # Compute MI for this alignment
            if overlap_len > n_bins:  # Need sufficient samples
                mi_scores[i] = self._compute_mutual_information(s1_overlap, s2_overlap, n_bins=n_bins)
            else:
                mi_scores[i] = 0.0

        # Find optimal lag (maximum MI)
        optimal_idx = np.argmax(mi_scores)
        optimal_lag = lags[optimal_idx]
        optimal_mi = mi_scores[optimal_idx]

        # Normalize MI score to 0-1 range for quality metric
        mi_range = mi_scores.max() - mi_scores.min()
        if mi_range > 0:
            quality_score = (optimal_mi - mi_scores.min()) / mi_range
        else:
            quality_score = 1.0

        # Diagnostic information
        diagnostics = {
            'lags': lags,
            'mi_scores': mi_scores,
            'optimal_lag': optimal_lag,
            'optimal_mi': optimal_mi,
            'mi_range': mi_range
        }

        return optimal_lag, quality_score, diagnostics

    def load_data(self) -> None:
        """Load data from HDF5 file with individual time vectors"""
        print(f"Loading data from {self.hdf5_path}")
        
        with h5py.File(self.hdf5_path, 'r') as file:
            for dataset_config in self.datasets:
                try:
                    # Load the main dataset
                    if dataset_config.group:
                        group = file[dataset_config.group]
                        data = np.array(group[dataset_config.name])
                    else:
                        data = np.array(file[dataset_config.name])
                    
                    self.raw_data[dataset_config.label] = data
                    
                    # Load or generate time vector
                    time_vector = None
                    
                    # Try to load explicit time vector
                    if dataset_config.time_name:
                        try:
                            if dataset_config.time_group:
                                time_group = file[dataset_config.time_group]
                                time_vector = np.array(time_group[dataset_config.time_name])
                            else:
                                # Try in same group as data
                                if dataset_config.group:
                                    group = file[dataset_config.group]
                                    time_vector = np.array(group[dataset_config.time_name])
                                else:
                                    time_vector = np.array(file[dataset_config.time_name])
                            
                            # Convert time units to seconds
                            time_vector = self._convert_time_units(time_vector, dataset_config.time_units)
                            
                            # Ensure time vector matches data length
                            if len(time_vector) != len(data):
                                print(f"Warning: Time vector length ({len(time_vector)}) doesn't match data length ({len(data)}) for {dataset_config.label}")
                                # Take minimum length
                                min_len = min(len(time_vector), len(data))
                                time_vector = time_vector[:min_len]
                                data = data[:min_len]
                                self.raw_data[dataset_config.label] = data
                            
                            print(f"✓ Loaded time vector for {dataset_config.label}")
                            
                        except KeyError as e:
                            print(f"Warning: Could not load time vector for {dataset_config.label}: {e}")
                            time_vector = None
                    
                    # Generate time vector if not loaded
                    if time_vector is None:
                        # Use individual sampling rate or default
                        sr = dataset_config.sampling_rate if dataset_config.sampling_rate else self.default_sampling_rate
                        time_vector = np.arange(len(data)) / sr
                        print(f"✓ Generated time vector for {dataset_config.label} at {sr} Hz")
                    
                    # Store original time vector
                    self.original_time_vectors[dataset_config.label] = time_vector.copy()
                    
                    # Apply time shift for phase alignment
                    if dataset_config.time_shift != 0.0:
                        time_vector = time_vector + dataset_config.time_shift
                        print(f"✓ Applied time shift of {dataset_config.time_shift:.3f}s to {dataset_config.label}")
                        self.alignment_info[dataset_config.label] = {
                            'time_shift': dataset_config.time_shift,
                            'shift_type': 'manual'
                        }
                    else:
                        self.alignment_info[dataset_config.label] = {
                            'time_shift': 0.0,
                            'shift_type': 'none'
                        }
                    
                    # Store time vector and calculate actual sampling rate
                    self.time_vectors[dataset_config.label] = time_vector
                    actual_sr = self._calculate_sampling_rate(time_vector)
                    self.sampling_rates[dataset_config.label] = actual_sr
                    
                    print(f"✓ Loaded {dataset_config.label}: {len(data)} samples, SR: {actual_sr:.2f} Hz")
                    
                except KeyError as e:
                    print(f"✗ Failed to load {dataset_config.label}: {e}")
                    continue
    
    def process_data(self, preserve_full_processed: bool = True) -> None:
        """Process all loaded datasets using individual sampling rates"""
        print("\nProcessing signals...")
        
        # Clear any existing processed data backups
        if preserve_full_processed:
            self.full_processed_data = {}  # Store uncropped processed signals
        
        for label, data in self.raw_data.items():
            print(f"Processing {label}...")
            # Use individual sampling rate for processing
            sr = self.sampling_rates[label]
            processed = self.processor.process_signal(data, sr)
            
            # Store the full processed signal
            self.processed_data[label] = processed
            
            # Preserve full processed signal before any cropping
            if preserve_full_processed:
                self.full_processed_data[label] = processed.copy()
                print(f"✓ Preserved full processed signal for {label}: {len(processed)} samples")
            
            # Update time vector if resampling was applied
            if self.processing_config.apply_resampling:
                # Create new time vector for resampled data
                original_duration = self.time_vectors[label][-1] - self.time_vectors[label][0]
                self.time_vectors[label] = np.linspace(
                    self.time_vectors[label][0], 
                    self.time_vectors[label][-1], 
                    len(processed)
                )
                # Update sampling rate
                self.sampling_rates[label] = len(processed) / original_duration
    
    def calculate_statistics(self) -> None:
        """Calculate comprehensive statistics for all datasets"""
        print("\nCalculating statistics...")
        
        for label, data in self.processed_data.items():
            stats = {
                'mean': np.mean(data),
                'median': np.median(data),
                'std': np.std(data),
                'var': np.var(data),
                'min': np.min(data),
                'max': np.max(data),
                'range': np.ptp(data),
                'skewness': self._calculate_skewness(data),
                'kurtosis': self._calculate_kurtosis(data),
                'rms': np.sqrt(np.mean(data**2)),
                'energy': np.sum(data**2),
                'zero_crossings': self._count_zero_crossings(data)
            }
            self.statistics[label] = stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of the data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of the data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _count_zero_crossings(self, data: np.ndarray) -> int:
        """Count zero crossings in the signal"""
        return len(np.where(np.diff(np.signbit(data)))[0])
    
    def _calculate_effective_sample_size(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Calculate effective sample size for autocorrelated time series

        Uses the Bretherton et al. (1999) / Bayley-Hammersley correction:
        n_eff = n * (1 - ρ₁) / (1 + ρ₁)

        where ρ₁ is the lag-1 autocorrelation coefficient.

        Parameters:
        -----------
        data : np.ndarray
            Time series data

        Returns:
        --------
        n_eff : float
            Effective sample size accounting for autocorrelation
        rho_1 : float
            Lag-1 autocorrelation coefficient
        """
        n = len(data)

        # Calculate lag-1 autocorrelation
        data_centered = data - np.mean(data)
        autocorr_full = np.correlate(data_centered, data_centered, mode='full')
        autocorr_full = autocorr_full / autocorr_full[len(autocorr_full)//2]  # Normalize

        # Get lag-1 autocorrelation (index n corresponds to lag 0, so n+1 is lag 1)
        rho_1 = autocorr_full[len(autocorr_full)//2 + 1]

        # Calculate effective sample size
        # Bound rho_1 to avoid numerical issues
        rho_1_bounded = np.clip(rho_1, -0.99, 0.99)
        n_eff = n * (1 - rho_1_bounded) / (1 + rho_1_bounded)

        # Ensure n_eff is at least 2 (minimum for correlation)
        n_eff = max(n_eff, 2.0)

        return n_eff, rho_1

    def _corrected_pearson_pvalue(self, r: float, n_eff: float) -> float:
        """
        Calculate p-value for Pearson correlation with effective sample size

        Uses t-distribution: t = r * sqrt((n_eff - 2) / (1 - r²))

        Parameters:
        -----------
        r : float
            Pearson correlation coefficient
        n_eff : float
            Effective sample size (accounting for autocorrelation)

        Returns:
        --------
        p_value : float
            Two-tailed p-value corrected for autocorrelation
        """
        from scipy.stats import t as t_dist

        # Avoid division by zero for perfect correlations
        if abs(r) >= 0.9999:
            return 0.0 if abs(r) > 0.9999 else 1e-16

        # Calculate t-statistic
        t_stat = r * np.sqrt((n_eff - 2) / (1 - r**2))

        # Two-tailed p-value
        p_value = 2 * t_dist.sf(abs(t_stat), n_eff - 2)

        return p_value

    def calculate_correlations(self) -> Dict[str, float]:
        """
        Calculate correlations between all pairs of datasets with autocorrelation-corrected p-values

        P-values are corrected for time series autocorrelation using effective sample size
        based on the Bretherton et al. (1999) method.
        """
        correlations = {}
        labels = list(self.processed_data.keys())

        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels[i+1:], i+1):
                data1 = self.processed_data[label1]
                data2 = self.processed_data[label2]
                time1 = self.time_vectors[label1]
                time2 = self.time_vectors[label2]

                # Synchronize time series for correlation analysis
                data1_sync, data2_sync = self._synchronize_time_series(
                    data1, time1, data2, time2
                )

                # Calculate standard correlations and p-values (uncorrected)
                pearson_corr, pearson_p_uncorr = pearsonr(data1_sync, data2_sync)
                spearman_corr, spearman_p_uncorr = spearmanr(data1_sync, data2_sync)

                # Calculate effective sample sizes for both series
                n_eff_1, rho1_1 = self._calculate_effective_sample_size(data1_sync)
                n_eff_2, rho1_2 = self._calculate_effective_sample_size(data2_sync)

                # Use the more conservative (smaller) effective sample size
                n_eff = min(n_eff_1, n_eff_2)

                # Calculate corrected p-values using effective sample size
                pearson_p_corr = self._corrected_pearson_pvalue(pearson_corr, n_eff)

                # For Spearman, use the same correction approach
                # (Spearman is just Pearson on ranks, so same correction applies)
                spearman_p_corr = self._corrected_pearson_pvalue(spearman_corr, n_eff)

                pair_key = f"{label1} vs {label2}"
                correlations[pair_key] = {
                    'pearson': pearson_corr,
                    'pearson_p_uncorrected': pearson_p_uncorr,
                    'pearson_p_corrected': pearson_p_corr,
                    'spearman': spearman_corr,
                    'spearman_p_uncorrected': spearman_p_uncorr,
                    'spearman_p_corrected': spearman_p_corr,
                    'n_actual': len(data1_sync),
                    'n_effective': n_eff,
                    'autocorr_lag1_series1': rho1_1,
                    'autocorr_lag1_series2': rho1_2
                }

        return correlations
    
    def calculate_differences(self) -> Dict[str, Dict[str, float]]:
        """Calculate various difference metrics between datasets using synchronized time series"""
        differences = {}
        labels = list(self.processed_data.keys())
        
        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels[i+1:], i+1):
                data1 = self.processed_data[label1]
                data2 = self.processed_data[label2]
                time1 = self.time_vectors[label1]
                time2 = self.time_vectors[label2]
                
                # Synchronize time series for difference analysis
                data1_sync, data2_sync = self._synchronize_time_series(
                    data1, time1, data2, time2
                )
                
                # Calculate various metrics
                mse = mean_squared_error(data1_sync, data2_sync)
                mae = mean_absolute_error(data1_sync, data2_sync)
                rmse = np.sqrt(mse)
                
                # Normalized metrics
                range1 = np.ptp(data1_sync)
                range2 = np.ptp(data2_sync)
                avg_range = (range1 + range2) / 2
                normalized_rmse = rmse / avg_range if avg_range > 0 else 0
                
                pair_key = f"{label1} vs {label2}"
                differences[pair_key] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'normalized_rmse': normalized_rmse
                }
        
        return differences
    
    def _synchronize_time_series(self, data1: np.ndarray, time1: np.ndarray,
                                data2: np.ndarray, time2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synchronize two time series with potentially different sampling rates

        Args:
            data1, time1: First time series and its time vector
            data2, time2: Second time series and its time vector

        Returns:
            Tuple of synchronized data arrays
        """
        # Ensure data and time arrays match in length
        if len(data1) != len(time1):
            min_len1 = min(len(data1), len(time1))
            data1 = data1[:min_len1]
            time1 = time1[:min_len1]

        if len(data2) != len(time2):
            min_len2 = min(len(data2), len(time2))
            data2 = data2[:min_len2]
            time2 = time2[:min_len2]

        # Find common time range
        t_start = max(time1[0], time2[0])
        t_end = min(time1[-1], time2[-1])

        if t_start >= t_end:
            raise ValueError("Time series do not overlap")

        # Create common time grid (use finer resolution of the two)
        dt1 = np.mean(np.diff(time1)) if len(time1) > 1 else 1.0
        dt2 = np.mean(np.diff(time2)) if len(time2) > 1 else 1.0
        dt_common = min(dt1, dt2)

        # Create common time vector
        t_common = np.arange(t_start, t_end, dt_common)

        # Interpolate both series to common time grid
        data1_interp = np.interp(t_common, time1, data1)
        data2_interp = np.interp(t_common, time2, data2)

        return data1_interp, data2_interp
    
    def _calculate_group_correlation(self, ref_group_name: str, ref_group_labels: List[str],
                                   target_group_name: str, target_group_labels: List[str],
                                   data_dict: Dict, time_vectors: Dict,
                                   cross_correlation_window: Optional[int],
                                   max_shift_time: Optional[float],
                                   correlation_method: str, normalize_signals: bool,
                                   alignment_method: str = 'ccf',
                                   feature_method: str = 'peak',
                                   n_features: int = 5,
                                   mi_bins: int = 50) -> float:
        """
        Calculate optimal shift between two groups of signals using various alignment methods

        Args:
            ref_group_name: Name of reference group
            ref_group_labels: List of signal labels in reference group
            target_group_name: Name of target group to align
            target_group_labels: List of signal labels in target group
            data_dict: Dictionary of signal data
            time_vectors: Dictionary of time vectors
            cross_correlation_window: Window size for correlation
            max_shift_time: Maximum shift to search
            correlation_method: Correlation method to use
            normalize_signals: Whether to normalize signals
            alignment_method: 'ccf', 'feature', or 'mutual_info'
            feature_method: For 'feature' method - 'peak', 'edge', or 'energy'
            n_features: For 'feature' method - number of features to match
            mi_bins: For 'mutual_info' method - number of bins for MI histogram

        Returns:
            Optimal time shift for target group
        """
        
        def normalize_for_correlation(signal, method='normalized'):
            """Normalize signal for better correlation"""
            if method == 'zero_mean':
                return signal - np.mean(signal)
            elif method == 'normalized':
                signal_zm = signal - np.mean(signal)
                std = np.std(signal_zm)
                return signal_zm / std if std > 0 else signal_zm
            elif method == 'minmax':
                signal_min, signal_max = np.min(signal), np.max(signal)
                range_val = signal_max - signal_min
                return (signal - signal_min) / range_val if range_val > 0 else signal - signal_min
            else:
                return signal
        # Create composite signals by averaging normalized signals within each group
        print(f"  Correlating group '{target_group_name}' against reference group '{ref_group_name}'")
        
        # Process reference group - use first signal as base and average others onto it
        ref_composite_data = None
        ref_composite_time = None
        
        for i, label in enumerate(ref_group_labels):
            data = data_dict[label]
            time_vec = time_vectors[label]
            
            # Ensure data and time vector have same length
            min_len = min(len(data), len(time_vec))
            data = data[:min_len]
            time_vec = time_vec[:min_len]
            
            # Normalize individual signal before averaging
            if normalize_signals:
                data_norm = normalize_for_correlation(data, correlation_method)
            else:
                data_norm = data
            
            if i == 0:
                # First signal becomes the reference
                ref_composite_time = time_vec.copy()
                ref_composite_data = data_norm.copy()
                print(f"    Reference base: {label} ({len(data)} samples)")
            else:
                # Synchronize subsequent signals to the reference
                try:
                    data_sync, ref_sync = self._synchronize_time_series(data_norm, time_vec, ref_composite_data, ref_composite_time)
                    ref_composite_data = (ref_sync + data_sync) / 2  # Running average
                    print(f"    Added to reference: {label} (sync: {len(data_sync)} samples)")
                except Exception as e:
                    print(f"    Warning: Could not sync {label} to reference group: {e}")
                    # Skip this signal if synchronization fails
                    continue
        
        # Process target group - use first signal as base and average others onto it
        target_composite_data = None
        target_composite_time = None
        
        for i, label in enumerate(target_group_labels):
            data = data_dict[label]
            time_vec = time_vectors[label]
            
            # Ensure data and time vector have same length
            min_len = min(len(data), len(time_vec))
            data = data[:min_len]
            time_vec = time_vec[:min_len]
            
            # Normalize individual signal before averaging
            if normalize_signals:
                data_norm = normalize_for_correlation(data, correlation_method)
            else:
                data_norm = data
            
            if i == 0:
                # First signal becomes the target base
                target_composite_time = time_vec.copy()
                target_composite_data = data_norm.copy()
                print(f"    Target base: {label} ({len(data)} samples)")
            else:
                # Synchronize subsequent signals to the target base
                try:
                    data_sync, target_sync = self._synchronize_time_series(data_norm, time_vec, target_composite_data, target_composite_time)
                    target_composite_data = (target_sync + data_sync) / 2  # Running average
                    print(f"    Added to target: {label} (sync: {len(data_sync)} samples)")
                except Exception as e:
                    print(f"    Warning: Could not sync {label} to target group: {e}")
                    # Skip this signal if synchronization fails
                    continue
        
        # Validate that we have composite signals
        if ref_composite_data is None or target_composite_data is None:
            print(f"    Error: Could not create composite signals for group correlation")
            return 0.0
        
        if len(ref_composite_data) == 0 or len(target_composite_data) == 0:
            print(f"    Error: Empty composite signals")
            return 0.0
        
        print(f"    Composite signals: ref={len(ref_composite_data)}, target={len(target_composite_data)}")
        
        # Synchronize composite signals
        try:
            ref_sync, target_sync = self._synchronize_time_series(ref_composite_data, ref_composite_time, 
                                                                target_composite_data, target_composite_time)
            print(f"    Synchronized composites: ref={len(ref_sync)}, target={len(target_sync)}")
        except Exception as e:
            print(f"    Error synchronizing composite signals: {e}")
            return 0.0
        common_time = np.linspace(max(ref_composite_time[0], target_composite_time[0]), 
                                 min(ref_composite_time[-1], target_composite_time[-1]), 
                                 len(ref_sync))
        
        # Apply windowing
        if cross_correlation_window and cross_correlation_window < len(ref_sync):
            center = len(ref_sync) // 2
            half_window = cross_correlation_window // 2
            start_idx = max(0, center - half_window)
            end_idx = min(len(ref_sync), center + half_window)
            ref_windowed = ref_sync[start_idx:end_idx]
            target_windowed = target_sync[start_idx:end_idx]
            window_time = common_time[start_idx:end_idx]
        else:
            ref_windowed = ref_sync
            target_windowed = target_sync
            window_time = common_time
        
        # Calculate correlation using unified helper
        # Normalize signals if requested before computing cross-correlation
        if normalize_signals:
            ref_for_corr = normalize_for_correlation(ref_windowed, correlation_method)
            target_for_corr = normalize_for_correlation(target_windowed, correlation_method)
        else:
            ref_for_corr = ref_windowed
            target_for_corr = target_windowed

        # Determine max lag for search
        max_lag_samples = None
        if max_shift_time is not None:
            dt = np.mean(np.diff(ref_composite_time))
            max_lag_samples = int(max_shift_time / dt)

        # Select alignment method
        if alignment_method == 'ccf':
            # Use unified cross-correlation helper (checks both positive and negative lags)
            lags_samples, correlation, peak_lag = self._compute_cross_correlation(
                ref_for_corr, target_for_corr,
                max_lag=max_lag_samples,
                method='statsmodels'
            )

            # Convert lags to time
            dt = np.mean(np.diff(ref_composite_time))
            lags_time = lags_samples * dt

            # Extract results
            time_shift = peak_lag * dt
            peak_idx = np.where(lags_samples == peak_lag)[0][0]
            max_corr_value = correlation[peak_idx]
            quality_score = max_corr_value

            # Store cross-correlation diagnostics for later plotting
            if not hasattr(self, 'alignment_diagnostics'):
                self.alignment_diagnostics = {}

            pair_key = f"{ref_group_name} vs {target_group_name}"
            self.alignment_diagnostics[pair_key] = {
                'method': 'ccf',
                'lags_samples': lags_samples.copy(),
                'lags_time': lags_time.copy(),
                'correlation': correlation.copy(),
                'peak_lag_samples': peak_lag,
                'peak_lag_time': time_shift,
                'max_correlation': max_corr_value,
                'label1': ref_group_name,
                'label2': target_group_name,
                'data_length': len(ref_for_corr)
            }

            print(f"    CCF: max_corr={max_corr_value:.4f}, shift={time_shift*1000:.3f}ms")

        elif alignment_method == 'feature':
            # Use feature-based alignment
            peak_lag, quality_score, diagnostics = self._align_by_feature_matching(
                ref_for_corr, target_for_corr,
                feature_method=feature_method,
                n_features=n_features,
                max_lag=max_lag_samples
            )

            # Convert lag to time
            dt = np.mean(np.diff(ref_composite_time))
            time_shift = peak_lag * dt

            # Store diagnostics for later plotting
            if not hasattr(self, 'alignment_diagnostics'):
                self.alignment_diagnostics = {}

            pair_key = f"{ref_group_name} vs {target_group_name}"
            self.alignment_diagnostics[pair_key] = {
                'method': 'feature',
                'peak_lag_samples': peak_lag,
                'peak_lag_time': time_shift,
                'quality_score': quality_score,
                'label1': ref_group_name,
                'label2': target_group_name,
                'data_length': len(ref_for_corr),
                **diagnostics
            }

            print(f"    Feature ({feature_method}): quality={quality_score:.4f}, shift={time_shift*1000:.3f}ms, n_matches={diagnostics.get('n_matches', 0)}")

        elif alignment_method == 'mutual_info':
            # Use mutual information alignment
            peak_lag, quality_score, diagnostics = self._align_by_mutual_information(
                ref_for_corr, target_for_corr,
                max_lag=max_lag_samples,
                n_bins=mi_bins,
                lag_step=1
            )

            # Convert lag to time
            dt = np.mean(np.diff(ref_composite_time))
            time_shift = peak_lag * dt

            # Store diagnostics for later plotting
            if not hasattr(self, 'alignment_diagnostics'):
                self.alignment_diagnostics = {}

            pair_key = f"{ref_group_name} vs {target_group_name}"
            self.alignment_diagnostics[pair_key] = {
                'method': 'mutual_info',
                'peak_lag_samples': peak_lag,
                'peak_lag_time': time_shift,
                'quality_score': quality_score,
                'label1': ref_group_name,
                'label2': target_group_name,
                'data_length': len(ref_for_corr),
                **diagnostics
            }

            # Convert lags to time for compatibility
            lags_samples = diagnostics['lags']
            lags_time = lags_samples * dt
            correlation = diagnostics['mi_scores']  # For compatibility with downstream code

            print(f"    Mutual Info: MI={diagnostics['optimal_mi']:.4f}, shift={time_shift*1000:.3f}ms, quality={quality_score:.4f}")

        else:
            raise ValueError(f"Unknown alignment_method: {alignment_method}. Use 'ccf', 'feature', or 'mutual_info'.")

        return time_shift
    
    def auto_align_time_series(self, reference_label: str = None,
                              reference_group: str = None,
                              cross_correlation_window: Optional[int] = None,
                              correlation_window_time: Optional[float] = None,
                              use_original_positions: bool = False,
                              use_raw_data: bool = True,
                              max_shift_time: float = None,
                              normalize_signals: bool = True,
                              correlation_method: str = 'normalized',
                              sync_within_groups: bool = True,
                              use_precomputed_correlations: bool = False,
                              alignment_method: str = 'ccf',
                              feature_method: str = 'peak',
                              n_features: int = 5,
                              mi_bins: int = 50) -> Dict[str, float]:
        """
        Automatically align time series using various alignment methods with group synchronization

        Args:
            reference_label: Label of the reference time series (overrides reference_group)
            reference_group: Group to use as reference (e.g., 'AMPM', 'KH')
            cross_correlation_window: Window size for cross-correlation in SAMPLES
            correlation_window_time: Window size for cross-correlation in SECONDS
            use_original_positions: Use original or current time vectors
            use_raw_data: Use raw (uncropped) or processed data
            max_shift_time: Maximum shift to search in seconds
            normalize_signals: Remove DC offset and normalize amplitude before correlation
            correlation_method: 'normalized', 'standard', or 'zero_mean'
            sync_within_groups: If True, maintain synchronization within dataset groups
            alignment_method: Alignment method - 'ccf' (cross-correlation), 'feature' (feature matching), 'mutual_info' (MI)
            feature_method: For 'feature' method - 'peak', 'edge', or 'energy'
            n_features: For 'feature' method - number of features to detect and match
            mi_bins: For 'mutual_info' method - number of bins for MI histogram

        Returns:
            Dictionary of calculated time shifts for each dataset/group
        """
        
        def normalize_for_correlation(signal, method='normalized'):
            """Normalize signal for better correlation"""
            if method == 'zero_mean':
                # Remove DC offset only
                return signal - np.mean(signal)
            elif method == 'normalized':
                # Remove DC offset and normalize to unit variance
                signal_zm = signal - np.mean(signal)
                std = np.std(signal_zm)
                return signal_zm / std if std > 0 else signal_zm
            elif method == 'minmax':
                # Scale to [0, 1] range
                signal_min, signal_max = np.min(signal), np.max(signal)
                range_val = signal_max - signal_min
                return (signal - signal_min) / range_val if range_val > 0 else signal - signal_min
            else:
                # No normalization
                return signal
        # Determine reference approach
        if reference_label and reference_label not in self.processed_data:
            raise ValueError(f"Reference dataset '{reference_label}' not found")
        
        if not reference_label and not reference_group:
            raise ValueError("Must specify either reference_label or reference_group")
        
        position_type = "original" if use_original_positions else "current"
        data_type = "raw" if use_raw_data else "processed"
        
        if sync_within_groups:
            if reference_group:
                print(f"\nAuto-aligning time series using '{reference_group}' group as reference...")
            else:
                ref_group = next((d.group for d in self.datasets if d.label == reference_label), None)
                print(f"\nAuto-aligning time series using '{reference_label}' (group: {ref_group}) as reference...")
            print(f"Group sync mode: Signals within same group stay synchronized")
        else:
            print(f"\nAuto-aligning time series using '{reference_label}' as reference...")
            print(f"Individual signal mode: Each signal aligned independently")
        
        print(f"Alignment method: {alignment_method.upper()}")
        if alignment_method == 'ccf':
            print(f"  Correlation: {correlation_method}, normalize: {normalize_signals}")
        elif alignment_method == 'feature':
            print(f"  Feature method: {feature_method}, n_features: {n_features}")
        elif alignment_method == 'mutual_info':
            print(f"  MI bins: {mi_bins}")
        print(f"Using {position_type} positions with {data_type} data")
        
        # Choose time vectors and data [same as before]
        if use_original_positions:
            time_vectors = self.original_time_vectors
        else:
            time_vectors = self.time_vectors
        
        if use_raw_data and hasattr(self, 'full_processed_data') and self.full_processed_data:
            # Use full processed data (after filtering, before cropping)
            data_dict = self.full_processed_data
            print("Using full processed data (filtered but not cropped)")
        elif use_raw_data and hasattr(self, 'original_processed_data') and self.original_processed_data:
            # Use original processed data (before cropping)
            data_dict = self.original_processed_data
            print("Using original processed data (before cropping)")
        elif use_raw_data:
            # Use raw data
            data_dict = self.raw_data
            print("Using raw data for correlation")
        else:
            # Use current processed data
            data_dict = self.processed_data
            print("Using current processed data for correlation")
        
        calculated_shifts = {}
        
        # Organize datasets by group
        groups = {}
        for dataset_config in self.datasets:
            if dataset_config.group not in groups:
                groups[dataset_config.group] = []
            if dataset_config.label in data_dict:
                groups[dataset_config.group].append(dataset_config.label)
        
        print(f"Dataset groups: {dict(groups)}")
        
        if sync_within_groups:
            # Group-based alignment: correlate groups against each other
            if reference_group:
                # Use composite group signals for alignment
                ref_group_name = reference_group
            else:
                # Use single reference signal but maintain group sync
                ref_group_name = next((d.group for d in self.datasets if d.label == reference_label), None)

            if ref_group_name not in groups:
                raise ValueError(f"Reference group '{ref_group_name}' not found")

            # Calculate window size using reference signal
            ref_sample_label = reference_label if reference_label else groups[ref_group_name][0]
            if correlation_window_time is not None:
                ref_dt = np.mean(np.diff(time_vectors[ref_sample_label])) if len(time_vectors[ref_sample_label]) > 1 else 1.0 / self.sampling_rates[ref_sample_label]
                cross_correlation_window = int(correlation_window_time / ref_dt)
                print(f"Using correlation window: {correlation_window_time:.3f}s ({cross_correlation_window} samples)")

            # Reference group has no shift
            for label in groups[ref_group_name]:
                calculated_shifts[label] = 0.0

            # Align other groups to reference
            for group_name, group_labels in groups.items():
                if group_name == ref_group_name:
                    continue

                if reference_label and not reference_group:
                    # Use single reference signal (not composite)
                    # Align first signal in target group to reference, then apply to all
                    first_target = group_labels[0]

                    ref_data = data_dict[reference_label]
                    ref_time = time_vectors[reference_label]
                    target_data = data_dict[first_target]
                    target_time = time_vectors[first_target]

                    # Synchronize signals
                    ref_sync, target_sync = self._synchronize_time_series(ref_data, ref_time, target_data, target_time)

                    # Apply windowing
                    if cross_correlation_window and cross_correlation_window < len(ref_sync):
                        center = len(ref_sync) // 2
                        half_window = cross_correlation_window // 2
                        start_idx = max(0, center - half_window)
                        end_idx = min(len(ref_sync), center + half_window)
                        ref_windowed = ref_sync[start_idx:end_idx]
                        target_windowed = target_sync[start_idx:end_idx]
                    else:
                        ref_windowed = ref_sync
                        target_windowed = target_sync

                    dt = np.mean(np.diff(time_vectors[reference_label]))
                    pair_key = f"{reference_label} vs {first_target}"

                    # Check if we should use precomputed correlation
                    if use_precomputed_correlations and hasattr(self, 'alignment_diagnostics') and pair_key in self.alignment_diagnostics:
                        # Use precomputed results
                        diag = self.alignment_diagnostics[pair_key]
                        peak_lag = diag['peak_lag_samples']
                        group_shift = diag['peak_lag_time']
                        lags_samples = diag['lags_samples']
                        correlation = diag['correlation']
                        print(f"✓ Using precomputed correlation for {pair_key}: {group_shift*1000:.3f}ms shift")
                    else:
                        # Compute cross-correlation
                        # Normalize if requested
                        if normalize_signals:
                            def normalize_for_correlation_local(signal, method='normalized'):
                                if method == 'normalized':
                                    signal_zm = signal - np.mean(signal)
                                    std = np.std(signal_zm)
                                    return signal_zm / std if std > 0 else signal_zm
                                return signal
                            ref_for_corr = normalize_for_correlation_local(ref_windowed, correlation_method)
                            target_for_corr = normalize_for_correlation_local(target_windowed, correlation_method)
                        else:
                            ref_for_corr = ref_windowed
                            target_for_corr = target_windowed

                        # Calculate shift using unified helper
                        max_lag_samples = None
                        if max_shift_time is not None:
                            max_lag_samples = int(max_shift_time / dt)

                        lags_samples, correlation, peak_lag = self._compute_cross_correlation(
                            ref_for_corr, target_for_corr,
                            max_lag=max_lag_samples,
                            method='statsmodels'
                        )

                        group_shift = peak_lag * dt

                        # Store cross-correlation diagnostics for later plotting
                        if not hasattr(self, 'alignment_diagnostics'):
                            self.alignment_diagnostics = {}

                    peak_idx = np.where(lags_samples == peak_lag)[0][0]

                    # Only store diagnostics if not using precomputed (avoid duplicates)
                    if not use_precomputed_correlations or pair_key not in self.alignment_diagnostics:
                        self.alignment_diagnostics[pair_key] = {
                            'lags_samples': lags_samples.copy(),
                            'lags_time': lags_samples * dt,
                            'correlation': correlation.copy(),
                        'peak_lag_samples': peak_lag,
                        'peak_lag_time': group_shift,
                        'max_correlation': correlation[peak_idx],
                        'label1': reference_label,
                        'label2': first_target,
                        'data_length': len(ref_for_corr)
                    }

                    print(f"✓ Aligned {first_target} to {reference_label}: {group_shift*1000:.3f}ms shift")

                else:
                    # Use composite group signals for alignment
                    group_shift = self._calculate_group_correlation(
                        ref_group_name, groups[ref_group_name],
                        group_name, group_labels,
                        data_dict, time_vectors,
                        cross_correlation_window, max_shift_time,
                        correlation_method, normalize_signals,
                        alignment_method, feature_method, n_features, mi_bins
                    )

                # Apply same shift to all signals in the target group
                for label in group_labels:
                    calculated_shifts[label] = group_shift

                print(f"✓ Group '{group_name}': {group_shift*1000:.3f}ms shift (applied to {len(group_labels)} signals)")
        
        else:
            # Individual signal alignment (original behavior)
            if not reference_label:
                raise ValueError("Individual signal mode requires reference_label")
                
            # Calculate window size
            if correlation_window_time is not None:
                ref_dt = np.mean(np.diff(time_vectors[reference_label])) if len(time_vectors[reference_label]) > 1 else 1.0 / self.sampling_rates[reference_label]
                cross_correlation_window = int(correlation_window_time / ref_dt)
                print(f"Using correlation window: {correlation_window_time:.3f}s ({cross_correlation_window} samples)")
            
            ref_data = data_dict[reference_label]
            ref_time = time_vectors[reference_label]
            calculated_shifts[reference_label] = 0.0

            # Individual signal processing loop
            for label, data in data_dict.items():
                if label == reference_label:
                    continue
                
                time_vec = time_vectors[label]
                
                # Synchronize signals
                ref_sync, data_sync = self._synchronize_time_series(ref_data, ref_time, data, time_vec)
                common_time = np.linspace(max(ref_time[0], time_vec[0]), 
                                         min(ref_time[-1], time_vec[-1]), 
                                         len(ref_sync))
                
                # Apply windowing
                if cross_correlation_window and cross_correlation_window < len(ref_sync):
                    center = len(ref_sync) // 2
                    half_window = cross_correlation_window // 2
                    start_idx = max(0, center - half_window)
                    end_idx = min(len(ref_sync), center + half_window)
                    ref_windowed = ref_sync[start_idx:end_idx]
                    data_windowed = data_sync[start_idx:end_idx]
                    window_time = common_time[start_idx:end_idx]
                else:
                    ref_windowed = ref_sync
                    data_windowed = data_sync
                    window_time = common_time
                
                # Calculate correlation using unified helper
                # Normalize signals if requested before computing cross-correlation
                if normalize_signals:
                    ref_for_corr = normalize_for_correlation(ref_windowed, correlation_method)
                    data_for_corr = normalize_for_correlation(data_windowed, correlation_method)
                else:
                    ref_for_corr = ref_windowed
                    data_for_corr = data_windowed

                # Determine max lag for search
                max_lag_samples = None
                if max_shift_time is not None:
                    dt = np.mean(np.diff(time_vectors[reference_label]))
                    max_lag_samples = int(max_shift_time / dt)

                # Use unified cross-correlation helper (checks both positive and negative lags)
                lags_samples, correlation, peak_lag = self._compute_cross_correlation(
                    ref_for_corr, data_for_corr,
                    max_lag=max_lag_samples,
                    method='statsmodels'
                )

                # Convert peak lag to time
                dt = np.mean(np.diff(time_vectors[reference_label]))
                time_shift = peak_lag * dt
                calculated_shifts[label] = time_shift

                # Store cross-correlation diagnostics for later plotting
                if not hasattr(self, 'alignment_diagnostics'):
                    self.alignment_diagnostics = {}

                pair_key = f"{reference_label} vs {label}"
                peak_idx = np.where(lags_samples == peak_lag)[0][0]
                self.alignment_diagnostics[pair_key] = {
                    'lags_samples': lags_samples.copy(),
                    'lags_time': lags_samples * dt,
                    'correlation': correlation.copy(),
                    'peak_lag_samples': peak_lag,
                    'peak_lag_time': time_shift,
                    'max_correlation': correlation[peak_idx],
                    'label1': reference_label,
                    'label2': label,
                    'data_length': len(ref_for_corr)
                }

                # Enhanced diagnostics
                max_corr_value = correlation[peak_idx]

                # For compatibility with downstream code
                lags_time = lags_samples * dt
                correlation_limited = correlation
                lags_time_limited = lags_time
                corr_mean = np.mean(correlation_limited)
                corr_std = np.std(correlation_limited)
                
                # Signal statistics before and after normalization
                ref_stats = f"mean={np.mean(ref_windowed):.3f}, std={np.std(ref_windowed):.3f}"
                target_stats = f"mean={np.mean(data_windowed):.3f}, std={np.std(data_windowed):.3f}"
                
                if normalize_signals:
                    ref_norm = normalize_for_correlation(ref_windowed, correlation_method)
                    target_norm = normalize_for_correlation(data_windowed, correlation_method)
                    ref_norm_stats = f"mean={np.mean(ref_norm):.3f}, std={np.std(ref_norm):.3f}"
                    target_norm_stats = f"mean={np.mean(target_norm):.3f}, std={np.std(target_norm):.3f}"
                
                print(f"✓ {label}: {time_shift*1000:.3f}ms shift, max_corr={max_corr_value:.4f}")
                print(f"  Original - Ref: {ref_stats}, Target: {target_stats}")
                if normalize_signals:
                    print(f"  Normalized - Ref: {ref_norm_stats}, Target: {target_norm_stats}")

            # Visualization removed - alignment details shown in processing_and_alignment_summary.png

        return calculated_shifts
    
    def apply_calculated_shifts(self, calculated_shifts: Dict[str, float],
                               relative_to_original: bool = False) -> None:
        """
        Apply calculated time shifts to align time series
        
        Args:
            calculated_shifts: Dictionary of time shifts for each dataset
            relative_to_original: If True, apply shifts relative to original positions.
                                 If False, apply shifts relative to current positions (additive).
        """
        application_type = "original" if relative_to_original else "current"
        print(f"\nApplying calculated time shifts relative to {application_type} positions...")
        
        for label, shift in calculated_shifts.items():
            if label in self.time_vectors and shift != 0.0:
                if relative_to_original:
                    # Apply shift relative to original time vector
                    self.time_vectors[label] = self.original_time_vectors[label] + shift
                    shift_type = 'auto_from_original'
                    total_shift = shift
                else:
                    # Apply shift relative to current time vector (additive)
                    self.time_vectors[label] = self.time_vectors[label] + shift
                    shift_type = 'auto_additive'
                    # Calculate total shift from original
                    current_manual_shift = self.alignment_info[label].get('time_shift', 0.0)
                    total_shift = current_manual_shift + shift
                
                # Update alignment info
                self.alignment_info[label] = {
                    'time_shift': total_shift,
                    'shift_type': shift_type,
                    'manual_shift': self.alignment_info[label].get('time_shift', 0.0) if not relative_to_original else 0.0,
                    'auto_shift': shift,
                    'group': next((d.group for d in self.datasets if d.label == label), None)
                }
                
                print(f"✓ Applied calculated shift of {shift:.6f}s to {label}")
                print(f"  Total shift from original: {total_shift:.6f}s")

    def compute_all_cross_correlations(self, max_shift_time: float = 0.002,
                                       correlation_method: str = 'normalized',
                                       use_raw_data: bool = True) -> None:
        """
        Compute cross-correlations for ALL signal pairs and store diagnostics.

        This provides comprehensive cross-correlation data for visualization,
        not just the pairs used for alignment calculation.

        Args:
            max_shift_time: Maximum time shift to search (seconds)
            correlation_method: 'normalized' or 'covariance'
            use_raw_data: If True, use raw data for correlation; if False, use processed
        """
        if not hasattr(self, 'alignment_diagnostics'):
            self.alignment_diagnostics = {}

        # Get all labels
        labels = list(self.raw_data.keys()) if use_raw_data else list(self.processed_data.keys())

        # Compute cross-correlation for all pairs
        for i, label1 in enumerate(labels):
            for label2 in labels[i+1:]:
                pair_key = f"{label1} vs {label2}"

                # Skip if already computed during alignment
                if pair_key in self.alignment_diagnostics:
                    continue

                # Get data and time vectors
                if use_raw_data:
                    data1 = self.raw_data[label1]
                    data2 = self.raw_data[label2]
                else:
                    data1 = self.full_processed_data.get(label1, self.processed_data[label1])
                    data2 = self.full_processed_data.get(label2, self.processed_data[label2])

                time1 = self.time_vectors[label1]
                time2 = self.time_vectors[label2]

                # Synchronize time series
                data1_sync, data2_sync = self._synchronize_time_series(data1, time1, data2, time2)

                # Normalize if requested
                if correlation_method == 'normalized':
                    # Inline normalization (z-score)
                    data1_zm = data1_sync - np.mean(data1_sync)
                    std1 = np.std(data1_zm)
                    data1_for_corr = data1_zm / std1 if std1 > 0 else data1_zm

                    data2_zm = data2_sync - np.mean(data2_sync)
                    std2 = np.std(data2_zm)
                    data2_for_corr = data2_zm / std2 if std2 > 0 else data2_zm
                else:
                    data1_for_corr = data1_sync
                    data2_for_corr = data2_sync

                # Determine max lag
                dt = np.mean(np.diff(time1))
                max_lag_samples = int(max_shift_time / dt)

                # Compute cross-correlation
                lags_samples, correlation, peak_lag = self._compute_cross_correlation(
                    data1_for_corr, data2_for_corr,
                    max_lag=max_lag_samples,
                    method='statsmodels'
                )

                # Store diagnostics
                peak_idx = np.where(lags_samples == peak_lag)[0][0]
                self.alignment_diagnostics[pair_key] = {
                    'lags_samples': lags_samples.copy(),
                    'lags_time': lags_samples * dt,
                    'correlation': correlation.copy(),
                    'peak_lag_samples': peak_lag,
                    'peak_lag_time': peak_lag * dt,
                    'max_correlation': correlation[peak_idx],
                    'label1': label1,
                    'label2': label2,
                    'data_length': len(data1_for_corr)
                }

    def get_alignment_summary(self) -> pd.DataFrame:
        """Get summary of all time alignments applied"""
        alignment_data = []

        for label, info in self.alignment_info.items():
            # Format time shift with high precision to show small shifts (e.g., 0.010ms)
            time_shift_ms = info['time_shift'] * 1000  # Convert to milliseconds
            time_shift_str = f"{info['time_shift']:.6f} ({time_shift_ms:+.3f}ms)"

            alignment_data.append({
                'Dataset': label,
                'Time Shift': time_shift_str,
                'Shift Type': info['shift_type'],
                'Original Duration [s]': f"{(self.original_time_vectors[label][-1] - self.original_time_vectors[label][0]):.6f}" if label in self.original_time_vectors else 'N/A',
                'Current Duration [s]': f"{(self.time_vectors[label][-1] - self.time_vectors[label][0]):.6f}" if label in self.time_vectors else 'N/A'
            })

        return pd.DataFrame(alignment_data)
    
    def crop_to_shortest_signal(self, use_processed_data: bool = True, 
                               preserve_original: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Crop all signals to match the duration of the shortest signal
        
        Args:
            use_processed_data: If True, crop processed data. If False, crop raw data.
            preserve_original: If True, preserve original data before cropping.
            
        Returns:
            Dictionary with cropping information for each dataset
        """
        data_dict = self.processed_data if use_processed_data else self.raw_data
        data_type = "processed" if use_processed_data else "raw"
        
        if not data_dict:
            print(f"No {data_type} data available for cropping")
            return {}
        
        print(f"\nCropping {data_type} data to shortest signal duration...")
        
        # Preserve original data if requested
        if preserve_original:
            if use_processed_data:
                for label, data in self.processed_data.items():
                    if label not in self.original_processed_data:
                        self.original_processed_data[label] = data.copy()
            else:
                for label, data in self.raw_data.items():
                    if label not in self.original_raw_data:
                        self.original_raw_data[label] = data.copy()
        
        # Find the time range limits for all signals
        earliest_start = float('-inf')
        latest_end = float('inf')
        duration_info = {}
        
        for label, data in data_dict.items():
            time_vec = self.time_vectors[label]
            start_time = time_vec[0]
            end_time = time_vec[-1]
            duration = end_time - start_time
            
            duration_info[label] = {
                'original_start': start_time,
                'original_end': end_time,
                'original_duration': duration,
                'original_samples': len(data)
            }
            
            # Update global limits
            earliest_start = max(earliest_start, start_time)
            latest_end = min(latest_end, end_time)
            
            print(f"  {label}: {duration:.6f}s ({len(data)} samples) - Start: {start_time:.6f}s, End: {end_time:.6f}s")
        
        if earliest_start >= latest_end:
            print("Error: No overlapping time range found between signals!")
            return duration_info
        
        common_duration = latest_end - earliest_start
        print(f"\nCommon time range: {earliest_start:.6f}s to {latest_end:.6f}s")
        print(f"Common duration: {common_duration:.6f}s")
        
        # Crop each signal to the common time range
        cropping_info = {}

        for label, data in data_dict.items():
            time_vec = self.time_vectors[label]

            # Find indices for cropping
            start_idx = np.argmin(np.abs(time_vec - earliest_start))
            end_idx = np.argmin(np.abs(time_vec - latest_end))

            # Ensure end_idx is after start_idx
            if end_idx <= start_idx:
                end_idx = len(time_vec) - 1

            # Crop data and time vector
            cropped_data = data[start_idx:end_idx+1]
            cropped_time = time_vec[start_idx:end_idx+1]

            # Update the data structures
            if use_processed_data:
                self.processed_data[label] = cropped_data
            else:
                self.raw_data[label] = cropped_data

            # Update time vectors to match cropped data length
            # IMPORTANT: Preserve any time shifts that were applied
            self.time_vectors[label] = cropped_time
            
            # Update sampling rate for consistency
            if len(cropped_time) > 1:
                self.sampling_rates[label] = self._calculate_sampling_rate(cropped_time)
            
            # Store cropping information
            cropping_info[label] = {
                'original_start': duration_info[label]['original_start'],
                'original_end': duration_info[label]['original_end'],
                'original_duration': duration_info[label]['original_duration'],
                'original_samples': duration_info[label]['original_samples'],
                'cropped_start': cropped_time[0],
                'cropped_end': cropped_time[-1],
                'cropped_duration': cropped_time[-1] - cropped_time[0],
                'cropped_samples': len(cropped_data),
                'start_idx': start_idx,
                'end_idx': end_idx,
                'samples_removed_start': start_idx,
                'samples_removed_end': duration_info[label]['original_samples'] - end_idx - 1
            }
            
            samples_removed = duration_info[label]['original_samples'] - len(cropped_data)
            print(f"✓ Cropped {label}: {len(cropped_data)} samples ({cropped_time[-1] - cropped_time[0]:.6f}s) - Removed {samples_removed} samples")
        
        print(f"\n✓ All signals cropped to common duration of {common_duration:.6f}s")
        
        return cropping_info
    
    def restore_original_length(self, use_processed_data: bool = True) -> None:
        """
        Restore signals to their original length before cropping
        
        Args:
            use_processed_data: If True, restore processed data. If False, restore raw data.
        """
        data_type = "processed" if use_processed_data else "raw"
        
        if use_processed_data:
            if not self.original_processed_data:
                print(f"No original {data_type} data available for restoration")
                return
            restore_from = self.original_processed_data
            restore_to = self.processed_data
        else:
            if not self.original_raw_data:
                print(f"No original {data_type} data available for restoration")
                return
            restore_from = self.original_raw_data
            restore_to = self.raw_data
        
        print(f"\nRestoring {data_type} data to original lengths...")
        
        for label, original_data in restore_from.items():
            if label in restore_to:
                restore_to[label] = original_data.copy()
                
                # Restore original time vectors
                if label in self.original_time_vectors:
                    self.time_vectors[label] = self.original_time_vectors[label].copy()
                    # Reapply any time shifts
                    if label in self.alignment_info and self.alignment_info[label]['time_shift'] != 0.0:
                        shift = self.alignment_info[label]['time_shift']
                        self.time_vectors[label] = self.original_time_vectors[label] + shift
                    
                    # Recalculate sampling rate
                    self.sampling_rates[label] = self._calculate_sampling_rate(self.time_vectors[label])
                
                print(f"✓ Restored {label} to {len(original_data)} samples")
        
        print(f"✓ All {data_type} signals restored to original lengths")
    
    def get_cropping_summary(self, cropping_info: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Create a summary DataFrame of cropping information
        
        Args:
            cropping_info: Dictionary returned by crop_to_shortest_signal()
            
        Returns:
            DataFrame with cropping summary
        """
        if not cropping_info:
            return pd.DataFrame()
        
        summary_data = []
        for label, info in cropping_info.items():
            summary_data.append({
                'Dataset': label,
                'Original Duration [s]': f"{info['original_duration']:.6f}",
                'Original Samples': info['original_samples'],
                'Cropped Duration [s]': f"{info['cropped_duration']:.6f}",
                'Cropped Samples': info['cropped_samples'],
                'Samples Removed': info['original_samples'] - info['cropped_samples'],
                'Removal Ratio [%]': f"{((info['original_samples'] - info['cropped_samples']) / info['original_samples'] * 100):.1f}",
                'Start Time [s]': f"{info['cropped_start']:.6f}",
                'End Time [s]': f"{info['cropped_end']:.6f}"
            })
        
        return pd.DataFrame(summary_data)
