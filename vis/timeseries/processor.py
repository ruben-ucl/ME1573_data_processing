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

    def __init__(self, processing_config: ProcessingConfig):
        self.config = processing_config

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

    def process_signal(self, data: np.ndarray, sampling_rate: float = 1.0) -> np.ndarray:
        """
        Apply comprehensive signal processing pipeline

        Args:
            data: Input signal array
            sampling_rate: Sampling rate of the signal

        Returns:
            Processed signal array
        """
        processed_data = data.copy()

        # Remove NaN values
        if self.config.handle_nans:
            processed_data = self._handle_nan_values(processed_data)

        # Remove statistical outliers (measurement errors)
        if self.config.remove_outliers:
            processed_data = self._remove_outliers(processed_data, sampling_rate)

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
            processed_data = self._apply_normalization(processed_data)

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

    def _remove_outliers(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """
        Remove statistical outliers using IQR, Z-score, or MAD methods.
        Can operate globally or on local windows for time series data.

        Window size is scaled based on sampling rate to ensure consistent
        temporal characteristics across signals with different sampling rates.

        Args:
            data: Input signal array
            sampling_rate: Sampling rate in Hz

        Returns:
            Signal array with outliers replaced by interpolated values
        """
        data_cleaned = data.copy()
        outlier_mask = np.zeros(len(data), dtype=bool)

        # Global outlier detection
        if self.config.outlier_window == 0:
            outlier_mask = self._detect_outliers_global(data)

        # Local (windowed) outlier detection
        else:
            # Scale window based on sampling rate
            window = self._scale_window_to_sampling_rate(self.config.outlier_window, sampling_rate)
            half_window = window // 2

            for i in range(len(data)):
                # Define window boundaries
                start_idx = max(0, i - half_window)
                end_idx = min(len(data), i + half_window + 1)
                window_data = data[start_idx:end_idx]

                # Detect outliers within this window
                local_outliers = self._detect_outliers_global(window_data)

                # Map back to position in window
                local_i = i - start_idx
                if local_i < len(local_outliers):
                    outlier_mask[i] = local_outliers[local_i]

        # Count and report outliers
        n_outliers = np.sum(outlier_mask)
        outlier_percentage = 100 * n_outliers / len(data)

        if n_outliers > 0:
            if self.config.outlier_window == 0:
                mode_str = "global"
            else:
                mode_str = f"local (window={window} samples, {window/sampling_rate*1000:.1f}ms)"
            print(f"  Outlier removal ({self.config.outlier_method}, {mode_str}, threshold={self.config.outlier_threshold}): "
                  f"{n_outliers} outliers detected ({outlier_percentage:.2f}%) - interpolating...")

            # Get outlier statistics
            outlier_values = data[outlier_mask]
            if len(outlier_values) > 0:
                print(f"    Outlier range: [{np.min(outlier_values):.4f}, {np.max(outlier_values):.4f}]")
                print(f"    Signal range: [{np.min(data):.4f}, {np.max(data):.4f}]")
        else:
            print(f"  Outlier removal ({self.config.outlier_method}): No outliers detected")

        # Replace outliers with NaN for interpolation
        data_cleaned[outlier_mask] = np.nan

        # Interpolate the outliers
        if np.any(np.isnan(data_cleaned)):
            mask = ~np.isnan(data_cleaned)
            if np.sum(mask) > 1:
                indices = np.arange(len(data_cleaned))
                data_cleaned = np.interp(indices, indices[mask], data_cleaned[mask])

        return data_cleaned

    def _detect_outliers_global(self, data: np.ndarray) -> np.ndarray:
        """
        Detect outliers using specified statistical method.

        Args:
            data: Input signal array

        Returns:
            Boolean mask where True indicates outlier
        """
        method = self.config.outlier_method
        threshold = self.config.outlier_threshold

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

    def _apply_normalization(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization using specified method"""
        data_reshaped = data.reshape(-1, 1)

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
            median = np.median(data)
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                normalized = (data_reshaped - median) / iqr
            else:
                normalized = data_reshaped - median
        else:
            normalized = data_reshaped

        return normalized.flatten()
