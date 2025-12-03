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
                                   visualize: bool = False) -> float:
        """
        Calculate optimal shift between two groups of signals using composite correlation
        
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
            visualize: Whether to show diagnostic plots
            
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

        # For compatibility with downstream code
        correlation_limited = correlation
        lags_time_limited = lags_time
        
        print(f"    Composite correlation: max={max_corr_value:.4f}, shift={time_shift*1000:.3f}ms")
        
        # Visualization for group correlation
        if visualize:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Plot composite signals
            axes[0].plot(window_time, ref_windowed, label=f'{ref_group_name} (composite)', alpha=0.8)
            axes[0].plot(window_time, target_windowed, label=f'{target_group_name} (composite)', alpha=0.8)
            axes[0].set_title('Composite Group Signals')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot correlation
            correlation_norm = correlation_limited / np.max(np.abs(correlation_limited))
            axes[1].plot(lags_time_limited * 1000, correlation_norm, 'b-', linewidth=1)
            axes[1].axvline(time_shift * 1000, color='red', linestyle='--', 
                           label=f'{time_shift*1000:.3f}ms')
            axes[1].set_xlabel('Lag [ms]')
            axes[1].set_ylabel('Normalized Correlation')
            axes[1].set_title(f'Group Cross-Correlation')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Plot aligned result
            target_shifted = np.interp(window_time - time_shift, window_time, target_windowed)
            axes[2].plot(window_time, ref_windowed, label=f'{ref_group_name}', alpha=0.8)
            axes[2].plot(window_time, target_shifted, label=f'{target_group_name} (shifted)', alpha=0.8)
            axes[2].set_title('Aligned Group Signals')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.suptitle(f'Group Alignment: {ref_group_name} ↔ {target_group_name}')
            plt.tight_layout()

            plt.show()
            plt.close()
        
        return time_shift
    
    def auto_align_time_series(self, reference_label: str = None,
                              reference_group: str = None, 
                              cross_correlation_window: Optional[int] = None,
                              correlation_window_time: Optional[float] = None,
                              use_original_positions: bool = False,
                              use_raw_data: bool = True,
                              visualize: bool = False,
                              max_shift_time: float = None,
                              normalize_signals: bool = True,
                              correlation_method: str = 'normalized',
                              sync_within_groups: bool = True) -> Dict[str, float]:
        """
        Automatically align time series using cross-correlation with group synchronization
        
        Args:
            reference_label: Label of the reference time series (overrides reference_group)
            reference_group: Group to use as reference (e.g., 'AMPM', 'KH')
            cross_correlation_window: Window size for cross-correlation in SAMPLES
            correlation_window_time: Window size for cross-correlation in SECONDS
            use_original_positions: Use original or current time vectors
            use_raw_data: Use raw (uncropped) or processed data
            visualize: Show correlation plots for each signal pair
            max_shift_time: Maximum shift to search in seconds
            normalize_signals: Remove DC offset and normalize amplitude before correlation
            correlation_method: 'normalized', 'standard', or 'zero_mean'
            sync_within_groups: If True, maintain synchronization within dataset groups
            
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
        
        print(f"Method: {correlation_method} correlation, normalize_signals: {normalize_signals}")
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
                        dt = np.mean(np.diff(time_vectors[reference_label]))
                        max_lag_samples = int(max_shift_time / dt)

                    lags_samples, correlation, peak_lag = self._compute_cross_correlation(
                        ref_for_corr, target_for_corr,
                        max_lag=max_lag_samples,
                        method='statsmodels'
                    )

                    dt = np.mean(np.diff(time_vectors[reference_label]))
                    group_shift = peak_lag * dt

                    print(f"✓ Aligned {first_target} to {reference_label}: {group_shift*1000:.3f}ms shift")

                    # Visualization for single reference signal alignment
                    if visualize:
                        import matplotlib.pyplot as plt

                        # MinMax normalization for visual comparison
                        def minmax_norm(signal):
                            s_min, s_max = np.min(signal), np.max(signal)
                            if s_max > s_min:
                                return (signal - s_min) / (s_max - s_min)
                            return signal - s_min

                        ref_minmax = minmax_norm(ref_windowed)
                        target_minmax = minmax_norm(target_windowed)

                        # Get common time for plotting
                        common_time = np.linspace(max(ref_time[0], target_time[0]),
                                                 min(ref_time[-1], target_time[-1]),
                                                 len(ref_sync))
                        if cross_correlation_window and cross_correlation_window < len(ref_sync):
                            center = len(ref_sync) // 2
                            half_window = cross_correlation_window // 2
                            start_idx = max(0, center - half_window)
                            end_idx = min(len(ref_sync), center + half_window)
                            window_time = common_time[start_idx:end_idx]
                        else:
                            window_time = common_time

                        # Create figure
                        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                        # Plot 1: Original signals (MinMax normalized)
                        axes[0].plot(window_time, ref_minmax, label=f'{reference_label}', alpha=0.8, linewidth=1.5)
                        axes[0].plot(window_time, target_minmax, label=f'{first_target}', alpha=0.8, linewidth=1.5)
                        axes[0].set_title(f'Signals: {reference_label} vs {first_target} (MinMax norm)')
                        axes[0].set_ylabel('Normalized Amplitude [0-1]')
                        axes[0].legend()
                        axes[0].grid(True, alpha=0.3)

                        # Plot 2: Signals used for correlation
                        axes[1].plot(window_time, ref_for_corr, label=f'{reference_label}', alpha=0.8, linewidth=1.5)
                        axes[1].plot(window_time, target_for_corr, label=f'{first_target}', alpha=0.8, linewidth=1.5)
                        axes[1].set_title(f'For Correlation ({correlation_method} norm)')
                        axes[1].set_ylabel('Amplitude')
                        axes[1].legend()
                        axes[1].grid(True, alpha=0.3)

                        # Plot 3: Cross-correlation
                        lags_time = lags_samples * dt
                        correlation_norm = correlation / np.max(np.abs(correlation))
                        peak_idx = np.where(lags_samples == peak_lag)[0][0]
                        max_corr_value = correlation[peak_idx]

                        axes[2].plot(lags_time * 1000, correlation_norm, 'b-', linewidth=2)
                        axes[2].axvline(group_shift * 1000, color='red', linestyle='--', linewidth=2,
                                       label=f'Peak: {group_shift*1000:.3f}ms')
                        axes[2].axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
                        axes[2].set_xlabel('Lag [ms]')
                        axes[2].set_ylabel('Correlation')
                        axes[2].set_title(f'Cross-Correlation (max={max_corr_value:.3f})')
                        axes[2].legend()
                        axes[2].grid(True, alpha=0.3)

                        # Plot 4: Aligned result (MinMax normalized)
                        target_shifted = np.interp(window_time - group_shift, window_time, target_windowed)
                        target_shifted_minmax = minmax_norm(target_shifted)
                        axes[3].plot(window_time, ref_minmax, label=f'{reference_label}', alpha=0.8, linewidth=1.5)
                        axes[3].plot(window_time, target_shifted_minmax,
                                    label=f'{first_target} (shifted {group_shift*1000:.3f}ms)',
                                    alpha=0.8, linewidth=1.5)
                        axes[3].set_title(f'After Alignment (MinMax norm)\nApplied to all {group_name} signals')
                        axes[3].set_ylabel('Normalized Amplitude [0-1]')
                        axes[3].legend()
                        axes[3].grid(True, alpha=0.3)

                        plt.suptitle(f'Group Alignment: {reference_label} → {group_name} group (via {first_target})', fontsize=14)
                        plt.tight_layout()
                        plt.show()

                else:
                    # Use composite group signals for alignment
                    group_shift = self._calculate_group_correlation(
                        ref_group_name, groups[ref_group_name],
                        group_name, group_labels,
                        data_dict, time_vectors,
                        cross_correlation_window, max_shift_time,
                        correlation_method, normalize_signals, visualize
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
            
            # Visualization setup for individual signal mode
            if visualize:
                import matplotlib.pyplot as plt
                n_targets = len([label for label in data_dict.keys() if label != reference_label])
                if n_targets > 0:
                    fig, axes = plt.subplots(n_targets, 4, figsize=(20, 5*n_targets))
                    if n_targets == 1:
                        axes = axes.reshape(1, -1)
                    plot_idx = 0
            
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

                # Enhanced diagnostics
                peak_idx = np.where(lags_samples == peak_lag)[0][0]
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
                
                # Visualization
                if visualize:
                    # MinMax normalization for visual comparison
                    def minmax_norm(signal):
                        s_min, s_max = np.min(signal), np.max(signal)
                        if s_max > s_min:
                            return (signal - s_min) / (s_max - s_min)
                        return signal - s_min

                    ref_minmax = minmax_norm(ref_windowed)
                    data_minmax = minmax_norm(data_windowed)

                    # Plot 1: Original signals (MinMax normalized for visual comparison)
                    axes[plot_idx, 0].plot(window_time, ref_minmax, label=f'{reference_label}', alpha=0.8, linewidth=1.5)
                    axes[plot_idx, 0].plot(window_time, data_minmax, label=f'{label}', alpha=0.8, linewidth=1.5)
                    axes[plot_idx, 0].set_title(f'Signals: {reference_label} vs {label} (MinMax norm)')
                    axes[plot_idx, 0].set_ylabel('Normalized Amplitude [0-1]')
                    axes[plot_idx, 0].legend()
                    axes[plot_idx, 0].grid(True, alpha=0.3)

                    # Plot 2: Signals used for correlation (after processing)
                    if normalize_signals:
                        ref_for_plot = normalize_for_correlation(ref_windowed, correlation_method)
                        target_for_plot = normalize_for_correlation(data_windowed, correlation_method)
                        axes[plot_idx, 1].plot(window_time, ref_for_plot, label=f'{reference_label}', alpha=0.8, linewidth=1.5)
                        axes[plot_idx, 1].plot(window_time, target_for_plot, label=f'{label}', alpha=0.8, linewidth=1.5)
                        axes[plot_idx, 1].set_title(f'For Correlation ({correlation_method} norm)')
                    else:
                        axes[plot_idx, 1].plot(window_time, ref_minmax, label=f'{reference_label}', alpha=0.8, linewidth=1.5)
                        axes[plot_idx, 1].plot(window_time, data_minmax, label=f'{label}', alpha=0.8, linewidth=1.5)
                        axes[plot_idx, 1].set_title(f'For Correlation (MinMax norm)')
                    axes[plot_idx, 1].set_ylabel('Amplitude')
                    axes[plot_idx, 1].legend()
                    axes[plot_idx, 1].grid(True, alpha=0.3)

                    # Plot 3: Cross-correlation
                    correlation_norm = correlation_limited / np.max(np.abs(correlation_limited))
                    axes[plot_idx, 2].plot(lags_time_limited * 1000, correlation_norm, 'b-', linewidth=2)
                    axes[plot_idx, 2].axvline(time_shift * 1000, color='red', linestyle='--', linewidth=2,
                                            label=f'Peak: {time_shift*1000:.3f}ms')
                    axes[plot_idx, 2].axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
                    axes[plot_idx, 2].set_xlabel('Lag [ms]')
                    axes[plot_idx, 2].set_ylabel('Correlation')
                    axes[plot_idx, 2].set_title(f'Cross-Correlation (max={max_corr_value:.3f})')
                    axes[plot_idx, 2].legend()
                    axes[plot_idx, 2].grid(True, alpha=0.3)

                    # Plot 4: Aligned result (MinMax normalized)
                    target_shifted = np.interp(window_time - time_shift, window_time, data_windowed)
                    target_shifted_minmax = minmax_norm(target_shifted)
                    axes[plot_idx, 3].plot(window_time, ref_minmax, label=f'{reference_label}', alpha=0.8, linewidth=1.5)
                    axes[plot_idx, 3].plot(window_time, target_shifted_minmax, label=f'{label} (shifted {time_shift*1000:.3f}ms)',
                                          alpha=0.8, linewidth=1.5)
                    axes[plot_idx, 3].set_title(f'After Alignment (MinMax norm)')
                    axes[plot_idx, 3].set_ylabel('Normalized Amplitude [0-1]')
                    axes[plot_idx, 3].legend()
                    axes[plot_idx, 3].grid(True, alpha=0.3)
                    
                    plot_idx += 1
            
            if visualize and 'n_targets' in locals() and n_targets > 0:
                plt.tight_layout()
                plt.show()

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

    def get_alignment_summary(self) -> pd.DataFrame:
        """Get summary of all time alignments applied"""
        alignment_data = []
        
        for label, info in self.alignment_info.items():
            alignment_data.append({
                'Dataset': label,
                'Time Shift [s]': f"{info['time_shift']:.4f}",
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
            
            # Always update time vectors to match cropped data length
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
