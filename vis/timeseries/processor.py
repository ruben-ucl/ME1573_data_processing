"""
Signal processing operations for time series data.

This module contains the TimeSeriesProcessor class which handles all
signal processing operations including filtering, smoothing, detrending,
normalization, and outlier removal.
"""

import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .config import ProcessingConfig


class TimeSeriesProcessor:
    """Advanced time series processing and analysis class"""

    def __init__(self, processing_config: ProcessingConfig, verbose: bool = True):
        self.config = processing_config
        self.verbose = verbose
        self.outlier_masks = {}  # Store outlier masks per signal label
        self.gradient_diagnostics = {}  # Store gradient diagnostic data per signal label for plotting

    def _scale_window_to_sampling_rate(self, window_samples: int, sampling_rate: float) -> int:
        """
        Scale window size to account for different sampling rates.

        Ensures that filters have the same temporal characteristics across signals
        with different sampling rates.

        Args:
            window_samples: Window size at reference sampling rate
            sampling_rate: Actual sampling rate of the signal

        Returns:
            Scaled window size (odd number for Savgol compatibility)
        """
        scaling_factor = sampling_rate / self.config.reference_sampling_rate
        scaled_window = int(window_samples * scaling_factor)

        # Ensure odd number for Savgol filter
        if scaled_window % 2 == 0:
            scaled_window += 1

        # Ensure minimum window size
        scaled_window = max(3, scaled_window)

        return scaled_window

    def process_signal(self, data: np.ndarray, sampling_rate: float = 1.0, label: str = None, group: str = None, dataset_name: str = None) -> np.ndarray:
        """
        Apply comprehensive signal processing pipeline

        Args:
            data: Input signal array
            sampling_rate: Sampling rate of the signal
            label: Optional label for the signal (used to store outlier mask)
            group: Optional group name (used to skip outlier removal for AMPM signals)
            dataset_name: Full dataset path (e.g., 'AMPM/Photodiode1Bits') for global normalization

        Returns:
            Processed signal array
        """
        processed_data = data.copy()

        # Remove NaN values
        if self.config.handle_nans:
            processed_data = self._handle_nan_values(processed_data)

        # Remove statistical outliers (measurement errors)
        # Skip outlier removal for AMPM group signals
        if self.config.remove_outliers and group != 'AMPM':
            processed_data, outlier_mask = self._remove_outliers(processed_data, sampling_rate, label=label)
            if label is not None:
                self.outlier_masks[label] = outlier_mask
        elif self.config.remove_outliers and group == 'AMPM':
            if self.verbose:
                print(f"  Outlier removal skipped for AMPM group signal")
            # Create empty outlier mask for consistency
            if label is not None:
                self.outlier_masks[label] = np.zeros(len(data), dtype=bool)

        # Apply detrending
        if self.config.apply_detrend:
            processed_data = self._apply_detrending(processed_data)

        # Apply filtering
        if self.config.apply_savgol:
            processed_data = self._apply_savgol_filter(processed_data, sampling_rate)

        if self.config.apply_lowpass:
            processed_data = self._apply_lowpass_filter(processed_data, sampling_rate)

        if self.config.apply_highpass:
            processed_data = self._apply_highpass_filter(processed_data, sampling_rate)

        if self.config.apply_bandpass:
            processed_data = self._apply_bandpass_filter(processed_data, sampling_rate)

        # Apply smoothing
        if self.config.apply_smoothing:
            processed_data = self._apply_smoothing(processed_data, sampling_rate)

        # Apply resampling
        if self.config.apply_resampling:
            processed_data = self._apply_resampling(processed_data)

        # Apply normalization
        if self.config.apply_normalization:
            processed_data = self._apply_normalization(processed_data, dataset_name)

        return processed_data

    def _handle_nan_values(self, data: np.ndarray) -> np.ndarray:
        """Handle NaN values using interpolation"""
        if np.any(np.isnan(data)):
            mask = ~np.isnan(data)
            if np.sum(mask) > 0:
                # Linear interpolation for NaN values
                indices = np.arange(len(data))
                data = np.interp(indices, indices[mask], data[mask])
        return data

    def _remove_outliers(self, data: np.ndarray, sampling_rate: float, label: str = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove statistical outliers using single or multiple sequential methods.
        Can operate globally or on local windows for time series data.

        Supports sequential application: methods are applied in order, each operating on
        the signal cleaned by the previous method. For example, ['gradient', 'iqr'] will
        first remove gradient-based outliers, then apply IQR to the cleaned signal.

        Args:
            data: Input signal array
            sampling_rate: Sampling rate in Hz
            label: Optional signal label for storing gradient diagnostics

        Returns:
            Tuple of (cleaned signal array, combined outlier mask boolean array)
        """
        # Normalize methods and thresholds to lists for uniform processing
        if isinstance(self.config.outlier_method, str):
            methods = [self.config.outlier_method]
        else:
            methods = self.config.outlier_method

        if isinstance(self.config.outlier_threshold, (int, float)):
            thresholds = [self.config.outlier_threshold] * len(methods)
        else:
            thresholds = self.config.outlier_threshold
            if len(thresholds) != len(methods):
                raise ValueError(f"Number of thresholds ({len(thresholds)}) must match number of methods ({len(methods)})")

        # Initialize with original data
        data_cleaned = data.copy()
        combined_outlier_mask = np.zeros(len(data), dtype=bool)

        # Apply each method sequentially
        for method_idx, (method, threshold) in enumerate(zip(methods, thresholds)):
            print(f"  Applying outlier method {method_idx+1}/{len(methods)}: {method} (threshold={threshold})")

            # Store current method and threshold temporarily
            original_method = self.config.outlier_method
            original_threshold = self.config.outlier_threshold
            self.config.outlier_method = method
            self.config.outlier_threshold = threshold

            # Detect outliers using current method
            method_outlier_mask = np.zeros(len(data_cleaned), dtype=bool)

            # For gradient/second_derivative methods: compute diagnostics on CLEANED signal (after previous passes)
            # Only do this for the first occurrence of these methods and if label is provided
            if method in ['gradient', 'second_derivative'] and label is not None and self.config.outlier_window > 0:
                if method_idx == 0 or method not in methods[:method_idx]:  # First occurrence
                    _ = self._detect_outliers_global(data_cleaned, label=label, method=method, threshold=threshold)

            # Global outlier detection
            if self.config.outlier_window == 0:
                method_outlier_mask = self._detect_outliers_global(data_cleaned, label=(label if method_idx==0 else None),
                                                                   method=method, threshold=threshold)

            # Local (windowed) outlier detection
            else:
                # Scale window based on sampling rate
                window = self._scale_window_to_sampling_rate(self.config.outlier_window, sampling_rate)
                half_window = window // 2

                for i in range(len(data_cleaned)):
                    # Define window boundaries
                    start_idx = max(0, i - half_window)
                    end_idx = min(len(data_cleaned), i + half_window + 1)
                    window_data = data_cleaned[start_idx:end_idx]

                    # Detect outliers within this window
                    local_outliers = self._detect_outliers_global(window_data, method=method, threshold=threshold)

                    # Map back to position in window
                    local_i = i - start_idx
                    if local_i < len(local_outliers):
                        method_outlier_mask[i] = local_outliers[local_i]

            # Count and report outliers for this method
            n_outliers = np.sum(method_outlier_mask)
            outlier_percentage = 100 * n_outliers / len(data_cleaned)

            if n_outliers > 0:
                if self.config.outlier_window == 0:
                    mode_str = "global"
                else:
                    mode_str = f"local (window={window} samples, {window/sampling_rate*1000:.1f}ms)"
                print(f"    {method} ({mode_str}): {n_outliers} outliers detected ({outlier_percentage:.2f}%) - interpolating...")

                # Get outlier statistics
                outlier_values = data_cleaned[method_outlier_mask]
                if len(outlier_values) > 0:
                    print(f"      Outlier range: [{np.min(outlier_values):.4f}, {np.max(outlier_values):.4f}]")
                    print(f"      Signal range: [{np.min(data_cleaned):.4f}, {np.max(data_cleaned):.4f}]")

                # Replace outliers with NaN for interpolation
                data_cleaned[method_outlier_mask] = np.nan

                # Interpolate the outliers
                if np.any(np.isnan(data_cleaned)):
                    mask = ~np.isnan(data_cleaned)
                    if np.sum(mask) > 1:
                        indices = np.arange(len(data_cleaned))
                        data_cleaned = np.interp(indices, indices[mask], data_cleaned[mask])

                # Update combined mask
                combined_outlier_mask = combined_outlier_mask | method_outlier_mask
            else:
                print(f"    {method}: No outliers detected")

            # Restore original config values
            self.config.outlier_method = original_method
            self.config.outlier_threshold = original_threshold

        # Final summary
        total_outliers = np.sum(combined_outlier_mask)
        total_percentage = 100 * total_outliers / len(data)
        print(f"  Total outliers removed across all methods: {total_outliers} ({total_percentage:.2f}%)")

        # Store cleaned data and combined outlier mask in diagnostics if gradient/second_derivative was used
        if label is not None and label in self.gradient_diagnostics:
            self.gradient_diagnostics[label]['data_cleaned'] = data_cleaned.copy()
            # Update outliers to include ALL passes (not just first method)
            self.gradient_diagnostics[label]['outliers'] = combined_outlier_mask.copy()

        return data_cleaned, combined_outlier_mask

    def _detect_outliers_global(self, data: np.ndarray, label: str = None,
                               method: str = None, threshold: float = None) -> np.ndarray:
        """
        Detect outliers using specified statistical method.

        Args:
            data: Input signal array
            label: Optional signal label for storing gradient diagnostics
            method: Optional method override (uses config if None)
            threshold: Optional threshold override (uses config if None)

        Returns:
            Boolean mask where True indicates outlier
        """
        method = method if method is not None else self.config.outlier_method
        threshold = threshold if threshold is not None else self.config.outlier_threshold

        if method == 'iqr':
            # Interquartile Range method
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = (data < lower_bound) | (data > upper_bound)

        elif method == 'zscore':
            # Z-score method
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                z_scores = np.abs((data - mean) / std)
                outliers = z_scores > threshold
            else:
                outliers = np.zeros(len(data), dtype=bool)

        elif method == 'mad':
            # Median Absolute Deviation method
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            if mad > 0:
                # Modified z-score using MAD
                modified_z_scores = 0.6745 * (data - median) / mad
                outliers = np.abs(modified_z_scores) > threshold
            else:
                outliers = np.zeros(len(data), dtype=bool)

        elif method == 'gradient':
            # Gradient-based outlier detection for sudden spikes
            # Compute gradient (rate of change)
            gradient = np.gradient(data)

            # Optional median filtering to smooth gradient and reduce noise sensitivity
            if self.config.outlier_gradient_smoothing > 1:
                from scipy.ndimage import median_filter
                gradient_smoothed = median_filter(gradient, size=self.config.outlier_gradient_smoothing, mode='nearest')
            else:
                gradient_smoothed = gradient

            # Compute MAD of absolute gradient
            abs_gradient = np.abs(gradient_smoothed)

            # Exclude near-zero values from median/MAD calculation (handles flat signal regions)
            # Use small epsilon to avoid excluding legitimate small gradients
            epsilon = 1e-10
            non_zero_mask = abs_gradient > epsilon

            if np.sum(non_zero_mask) > 0:
                # Calculate statistics on non-zero gradient values only
                median_grad = np.median(abs_gradient[non_zero_mask])
                mad_grad = np.median(np.abs(abs_gradient[non_zero_mask] - median_grad))
            else:
                # All gradients are essentially zero (completely flat signal)
                median_grad = 0.0
                mad_grad = 0.0

            if mad_grad > 0:
                # Detect points where gradient exceeds threshold × MAD
                grad_threshold = median_grad + threshold * mad_grad
                outliers = abs_gradient > grad_threshold
            else:
                outliers = np.zeros(len(data), dtype=bool)

            # Store diagnostic data for plotting (keyed by signal label)
            if label is not None:
                self.gradient_diagnostics[label] = {
                    'data': data.copy(),
                    'gradient_raw': gradient.copy(),
                    'gradient_smoothed': gradient_smoothed.copy(),
                    'abs_gradient': abs_gradient.copy(),
                    'median_grad': median_grad,
                    'mad_grad': mad_grad,
                    'threshold_value': median_grad + threshold * mad_grad if mad_grad > 0 else 0,
                    'outliers': outliers.copy(),
                    'method': 'gradient'
                }

        elif method == 'second_derivative':
            # Second derivative-based outlier detection for acceleration/curvature spikes
            # Compute first derivative (velocity)
            first_deriv = np.gradient(data)

            # Compute second derivative (acceleration/curvature)
            second_deriv = np.gradient(first_deriv)

            # Optional median filtering to smooth second derivative
            if self.config.outlier_gradient_smoothing > 1:
                from scipy.ndimage import median_filter
                second_deriv_smoothed = median_filter(second_deriv, size=self.config.outlier_gradient_smoothing, mode='nearest')
            else:
                second_deriv_smoothed = second_deriv

            # Compute MAD of absolute second derivative
            abs_second_deriv = np.abs(second_deriv_smoothed)

            # Exclude near-zero values from median/MAD calculation (handles flat signal regions)
            # Use small epsilon to avoid excluding legitimate small second derivatives
            epsilon = 1e-10
            non_zero_mask = abs_second_deriv > epsilon

            if np.sum(non_zero_mask) > 0:
                # Calculate statistics on non-zero second derivative values only
                median_second = np.median(abs_second_deriv[non_zero_mask])
                mad_second = np.median(np.abs(abs_second_deriv[non_zero_mask] - median_second))
            else:
                # All second derivatives are essentially zero (completely flat signal)
                median_second = 0.0
                mad_second = 0.0

            if mad_second > 0:
                # Detect points where second derivative exceeds threshold × MAD
                second_threshold = median_second + threshold * mad_second
                outliers = abs_second_deriv > second_threshold
            else:
                outliers = np.zeros(len(data), dtype=bool)

            # Store diagnostic data for plotting (keyed by signal label)
            if label is not None:
                self.gradient_diagnostics[label] = {
                    'data': data.copy(),
                    'gradient_raw': second_deriv.copy(),  # Raw second derivative (for plotting)
                    'gradient_smoothed': second_deriv_smoothed.copy(),  # Smoothed second derivative (for plotting)
                    'abs_gradient': abs_second_deriv.copy(),
                    'median_grad': median_second,
                    'mad_grad': mad_second,
                    'threshold_value': median_second + threshold * mad_second if mad_second > 0 else 0,
                    'outliers': outliers.copy(),
                    'method': 'second_derivative',
                    'first_deriv': first_deriv.copy()  # Store first derivative for reference
                }

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        return outliers

    def _apply_detrending(self, data: np.ndarray) -> np.ndarray:
        """Apply detrending to remove linear or constant trends"""
        if self.config.detrend_method == 'linear':
            return signal.detrend(data, type='linear')
        elif self.config.detrend_method == 'constant':
            return signal.detrend(data, type='constant')
        return data

    def _apply_savgol_filter(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """
        Apply Savitzky-Golay filter for smoothing.

        Window size is scaled based on sampling rate to ensure consistent
        temporal characteristics across signals with different sampling rates.
        """
        # Scale window based on sampling rate
        window_length = self._scale_window_to_sampling_rate(self.config.savgol_window, sampling_rate)
        window_length = min(window_length, len(data))

        # Ensure window is valid
        if window_length >= self.config.savgol_polyorder + 1:
            print(f"  Savgol filter: window={window_length} samples ({window_length/sampling_rate*1000:.1f}ms), order={self.config.savgol_polyorder}")
            return signal.savgol_filter(data, window_length, self.config.savgol_polyorder)
        return data

    def _apply_lowpass_filter(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """
        Apply low-pass Butterworth filter using zero-phase filtering (filtfilt).

        Uses filtfilt instead of sosfilt to:
        - Eliminate phase distortion
        - Reduce ringing artifacts (Gibbs phenomenon) at step changes
        - Double the effective filter order (forward + backward pass)

        Note: If you see sharp oscillations/spikes at step changes, try:
        - Lower filter order (e.g., order=2 instead of 4)
        - Higher cutoff frequency (less aggressive filtering)
        - Use Savitzky-Golay filter instead (apply_savgol=True)
        """
        # Calculate actual frequency cutoff
        nyquist = sampling_rate / 2
        cutoff_freq = self.config.lowpass_cutoff * nyquist

        print(f"  Lowpass filter (order={self.config.lowpass_order}): "
              f"cutoff = {cutoff_freq:.2f} Hz "
              f"(normalized: {self.config.lowpass_cutoff}, Nyquist: {nyquist:.2f} Hz)")

        sos = signal.butter(self.config.lowpass_order,
                           self.config.lowpass_cutoff,
                           btype='low', output='sos')

        # Use filtfilt for zero-phase filtering (reduces ringing artifacts)
        return signal.sosfiltfilt(sos, data)

    def _apply_highpass_filter(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Apply high-pass Butterworth filter"""
        # Calculate actual frequency cutoff
        nyquist = sampling_rate / 2
        cutoff_freq = self.config.highpass_cutoff * nyquist

        print(f"  Highpass filter (order={self.config.highpass_order}): "
              f"cutoff = {cutoff_freq:.2f} Hz "
              f"(normalized: {self.config.highpass_cutoff}, Nyquist: {nyquist:.2f} Hz)")

        sos = signal.butter(self.config.highpass_order,
                           self.config.highpass_cutoff,
                           btype='high', output='sos')
        return signal.sosfilt(sos, data)

    def _apply_bandpass_filter(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Apply band-pass Butterworth filter"""
        # Calculate actual frequency cutoffs
        nyquist = sampling_rate / 2
        low_freq = self.config.bandpass_low * nyquist
        high_freq = self.config.bandpass_high * nyquist

        print(f"  Bandpass filter (order={self.config.bandpass_order}): "
              f"band = [{low_freq:.2f}, {high_freq:.2f}] Hz "
              f"(normalized: [{self.config.bandpass_low}, {self.config.bandpass_high}], Nyquist: {nyquist:.2f} Hz)")

        sos = signal.butter(self.config.bandpass_order,
                           [self.config.bandpass_low, self.config.bandpass_high],
                           btype='band', output='sos')
        return signal.sosfilt(sos, data)

    def _apply_smoothing(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """
        Apply smoothing using specified method.

        Window size is scaled based on sampling rate to ensure consistent
        temporal characteristics across signals with different sampling rates.
        """
        # Scale window based on sampling rate
        window = self._scale_window_to_sampling_rate(self.config.smoothing_window, sampling_rate)

        print(f"  Smoothing ({self.config.smoothing_method}): window={window} samples ({window/sampling_rate*1000:.1f}ms)")

        if self.config.smoothing_method == 'gaussian':
            # Gaussian smoothing
            kernel = signal.gaussian(window, std=window/6)
            kernel = kernel / np.sum(kernel)
            return np.convolve(data, kernel, mode='same')
        elif self.config.smoothing_method == 'uniform':
            # Uniform (moving average) smoothing
            kernel = np.ones(window) / window
            return np.convolve(data, kernel, mode='same')
        elif self.config.smoothing_method == 'exponential':
            # Exponential smoothing
            alpha = 2.0 / (window + 1)
            smoothed = np.zeros_like(data)
            smoothed[0] = data[0]
            for i in range(1, len(data)):
                smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
            return smoothed
        return data

    def _apply_resampling(self, data: np.ndarray) -> np.ndarray:
        """Apply resampling to change the number of samples"""
        if len(data) == self.config.target_samples:
            return data
        return signal.resample(data, self.config.target_samples)

    def _apply_normalization(self, data: np.ndarray, dataset_name: str = None) -> np.ndarray:
        """
        Apply normalization using specified method.

        Args:
            data: Input signal array
            dataset_name: Full dataset path (e.g., 'AMPM/Photodiode1Bits') for global normalization

        Returns:
            Normalized signal array
        """
        data_reshaped = data.reshape(-1, 1)

        # Global normalization using pre-computed statistics
        if self.config.use_global_normalization and dataset_name is not None:
            try:
                from tools import get_dataset_normalization_params
                from pathlib import Path

                # Determine stats file path
                hdf5_dir = None
                if self.config.global_stats_file is not None:
                    hdf5_dir = Path(self.config.global_stats_file).parent

                if self.config.normalization_method == 'standard':
                    # Z-score normalization using global mean and std
                    mean_val, std_val = get_dataset_normalization_params(
                        dataset_name, method='zscore', hdf5_dir=hdf5_dir
                    )
                    if std_val > 0:
                        normalized = (data_reshaped - mean_val) / std_val
                    else:
                        normalized = data_reshaped - mean_val
                    print(f"  ✓ Using GLOBAL normalization (zscore): mean={mean_val:.4f}, std={std_val:.4f}")

                elif self.config.normalization_method == 'minmax':
                    # Min-max normalization using global min and max
                    min_val, max_val = get_dataset_normalization_params(
                        dataset_name, method='minmax', hdf5_dir=hdf5_dir
                    )
                    range_val = max_val - min_val
                    if range_val > 0:
                        normalized = (data_reshaped - min_val) / range_val
                    else:
                        normalized = data_reshaped - min_val
                    print(f"  ✓ Using GLOBAL normalization (minmax): min={min_val:.4f}, max={max_val:.4f}, range={range_val:.4f}")

                elif self.config.normalization_method == 'robust':
                    # Robust normalization not yet supported for global stats
                    print("  Warning: Robust normalization with global stats not yet implemented. Using per-track normalization.")
                    normalized = self._apply_local_normalization(data_reshaped)

                else:
                    normalized = data_reshaped

            except Exception as e:
                print(f"  Warning: Global normalization failed ({e}). Falling back to per-track normalization.")
                normalized = self._apply_local_normalization(data_reshaped)

        # Per-track (local) normalization
        else:
            if self.verbose:
                print(f"  Using PER-TRACK normalization ({self.config.normalization_method})")
            normalized = self._apply_local_normalization(data_reshaped)

        return normalized.flatten()

    def _apply_local_normalization(self, data_reshaped: np.ndarray) -> np.ndarray:
        """
        Apply per-track normalization using local statistics.

        Args:
            data_reshaped: Input signal array reshaped to (-1, 1)

        Returns:
            Normalized signal array (still reshaped)
        """
        if self.config.normalization_method == 'standard':
            # Create fresh scaler for each signal
            scaler = StandardScaler()
            normalized = scaler.fit_transform(data_reshaped)
        elif self.config.normalization_method == 'minmax':
            # Create fresh scaler for each signal
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(data_reshaped)
        elif self.config.normalization_method == 'robust':
            # Robust scaling using median and IQR
            data_flat = data_reshaped.flatten()
            median = np.median(data_flat)
            q75, q25 = np.percentile(data_flat, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                normalized = (data_reshaped - median) / iqr
            else:
                normalized = data_reshaped - median
        else:
            normalized = data_reshaped

        return normalized
