#!/usr/bin/env python
"""
Enhanced Time Series Comparison Script

This script provides comprehensive time series analysis and comparison capabilities
for an arbitrary number of curves stored in HDF5 datasets. It includes advanced
signal processing options and statistical analysis.

Author: AI Assistant
Version: 2.1
Based on: timeseries_compare.py
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("magma")

@dataclass
class DatasetConfig:
    """Configuration for dataset groups and names in HDF5 file"""
    group: str
    name: str
    label: str
    color: Optional[str] = None
    linestyle: Optional[str] = None
    # Time vector configuration
    time_group: Optional[str] = None  # Group containing time vector (can be same as data group)
    time_name: Optional[str] = None   # Name of time vector dataset
    sampling_rate: Optional[float] = None  # If no time vector, use this sampling rate
    time_units: str = 's'  # Units for time vector ('s', 'ms', 'us', etc.)
    # Phase alignment
    time_shift: float = 0.0  # Time shift in seconds to align with other series (positive = delay, negative = advance)

@dataclass
class ProcessingConfig:
    """Configuration for signal processing options"""
    # Filtering options
    apply_savgol: bool = False
    savgol_window: int = 51
    savgol_polyorder: int = 3
    
    # Low-pass filtering
    apply_lowpass: bool = False
    lowpass_cutoff: float = 0.1  # Normalized frequency (0-1)
    lowpass_order: int = 4
    
    # High-pass filtering
    apply_highpass: bool = False
    highpass_cutoff: float = 0.01  # Normalized frequency (0-1)
    highpass_order: int = 4
    
    # Bandpass filtering
    apply_bandpass: bool = False
    bandpass_low: float = 0.01
    bandpass_high: float = 0.1
    bandpass_order: int = 4
    
    # Smoothing
    apply_smoothing: bool = False
    smoothing_window: int = 5
    smoothing_method: str = 'uniform'  # 'gaussian', 'uniform', 'exponential'
    
    # Detrending
    apply_detrend: bool = False
    detrend_method: str = 'linear'  # 'linear', 'constant'
    
    # Normalization
    apply_normalization: bool = False
    normalization_method: str = 'standard'  # 'standard', 'minmax', 'robust'
    
    # Resampling
    apply_resampling: bool = False
    target_samples: int = 1000
    resampling_method: str = 'linear'  # 'linear', 'cubic', 'nearest'

class TimeSeriesProcessor:
    """Advanced time series processing and analysis class"""
    
    def __init__(self, processing_config: ProcessingConfig):
        self.config = processing_config
        self.scaler = None
        
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
        processed_data = self._handle_nan_values(processed_data)
        
        # Apply detrending
        if self.config.apply_detrend:
            processed_data = self._apply_detrending(processed_data)
        
        # Apply filtering
        if self.config.apply_savgol:
            processed_data = self._apply_savgol_filter(processed_data)
            
        if self.config.apply_lowpass:
            processed_data = self._apply_lowpass_filter(processed_data, sampling_rate)
            
        if self.config.apply_highpass:
            processed_data = self._apply_highpass_filter(processed_data, sampling_rate)
            
        if self.config.apply_bandpass:
            processed_data = self._apply_bandpass_filter(processed_data, sampling_rate)
        
        # Apply smoothing
        if self.config.apply_smoothing:
            processed_data = self._apply_smoothing(processed_data)
        
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
    
    def _apply_detrending(self, data: np.ndarray) -> np.ndarray:
        """Apply detrending to remove linear or constant trends"""
        if self.config.detrend_method == 'linear':
            return signal.detrend(data, type='linear')
        elif self.config.detrend_method == 'constant':
            return signal.detrend(data, type='constant')
        return data
    
    def _apply_savgol_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay filter for smoothing"""
        window_length = min(self.config.savgol_window, len(data))
        if window_length % 2 == 0:
            window_length -= 1  # Ensure odd window length
        if window_length >= self.config.savgol_polyorder + 1:
            return signal.savgol_filter(data, window_length, self.config.savgol_polyorder)
        return data
    
    def _apply_lowpass_filter(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Apply low-pass Butterworth filter"""
        sos = signal.butter(self.config.lowpass_order, 
                           self.config.lowpass_cutoff, 
                           btype='low', output='sos')
        return signal.sosfilt(sos, data)
    
    def _apply_highpass_filter(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Apply high-pass Butterworth filter"""
        sos = signal.butter(self.config.highpass_order, 
                           self.config.highpass_cutoff, 
                           btype='high', output='sos')
        return signal.sosfilt(sos, data)
    
    def _apply_bandpass_filter(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Apply band-pass Butterworth filter"""
        sos = signal.butter(self.config.bandpass_order, 
                           [self.config.bandpass_low, self.config.bandpass_high], 
                           btype='band', output='sos')
        return signal.sosfilt(sos, data)
    
    def _apply_smoothing(self, data: np.ndarray) -> np.ndarray:
        """Apply smoothing using specified method"""
        if self.config.smoothing_method == 'gaussian':
            # Gaussian smoothing
            kernel = signal.gaussian(self.config.smoothing_window, std=self.config.smoothing_window/6)
            kernel = kernel / np.sum(kernel)
            return np.convolve(data, kernel, mode='same')
        elif self.config.smoothing_method == 'uniform':
            # Uniform (moving average) smoothing
            kernel = np.ones(self.config.smoothing_window) / self.config.smoothing_window
            return np.convolve(data, kernel, mode='same')
        elif self.config.smoothing_method == 'exponential':
            # Exponential smoothing
            alpha = 2.0 / (self.config.smoothing_window + 1)
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
            if self.scaler is None:
                self.scaler = StandardScaler()
                normalized = self.scaler.fit_transform(data_reshaped)
            else:
                normalized = self.scaler.transform(data_reshaped)
        elif self.config.normalization_method == 'minmax':
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                normalized = self.scaler.fit_transform(data_reshaped)
            else:
                normalized = self.scaler.transform(data_reshaped)
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


class TimeSeriesComparator:
    """Main class for comparing multiple time series"""
    
    def __init__(self, hdf5_path: str, datasets: List[DatasetConfig], 
                 processing_config: ProcessingConfig, default_sampling_rate: float = 100.0):
        """
        Initialize the time series comparator
        
        Args:
            hdf5_path: Path to HDF5 file
            datasets: List of dataset configurations
            processing_config: Signal processing configuration
            default_sampling_rate: Default sampling rate in Hz (used as fallback)
        """
        self.hdf5_path = Path(hdf5_path)
        self.datasets = datasets
        self.processing_config = processing_config
        self.default_sampling_rate = default_sampling_rate
        self.processor = TimeSeriesProcessor(processing_config)
        
        self.raw_data = {}
        self.processed_data = {}
        self.time_vectors = {}
        self.original_time_vectors = {}  # Store original time vectors before shifting
        self.sampling_rates = {}  # Store individual sampling rates
        self.statistics = {}
        self.alignment_info = {}  # Store time shift information
        
        # For cropping functionality
        self.original_raw_data = {}
        self.original_processed_data = {}  # This stores cropped→original restoration
        self.full_processed_data = {}      # This stores the full filtered signals
        self.last_cropping_info = {}
        
    def _convert_time_units(self, time_vector: np.ndarray, units: str) -> np.ndarray:
        """Convert time vector to seconds based on specified units"""
        conversion_factors = {
            's': 1.0,
            'ms': 1e-3,
            'us': 1e-6,
            'μs': 1e-6,
            'ns': 1e-9
        }
        
        factor = conversion_factors.get(units.lower(), 1.0)
        return time_vector * factor
    
    def _calculate_sampling_rate(self, time_vector: np.ndarray) -> float:
        """Calculate sampling rate from time vector"""
        if len(time_vector) < 2:
            return self.default_sampling_rate
        
        # Calculate average time step
        dt = np.mean(np.diff(time_vector))
        if dt <= 0:
            return self.default_sampling_rate
        
        return 1.0 / dt
        
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
    
    def calculate_correlations(self) -> Dict[str, float]:
        """Calculate correlations between all pairs of datasets using interpolation for different sampling rates"""
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
                
                # Calculate correlations
                pearson_corr, _ = pearsonr(data1_sync, data2_sync)
                spearman_corr, _ = spearmanr(data1_sync, data2_sync)
                
                pair_key = f"{label1} vs {label2}"
                correlations[pair_key] = {
                    'pearson': pearson_corr,
                    'spearman': spearman_corr
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
    
    def auto_align_time_series(self, reference_label: str, 
                              cross_correlation_window: Optional[int] = None,
                              correlation_window_time: Optional[float] = None,
                              use_original_positions: bool = False,
                              use_raw_data: bool = True,
                              visualize: bool = False,
                              max_shift_time: float = None,
                              normalize_signals: bool = True,
                              correlation_method: str = 'normalized') -> Dict[str, float]:
        """
        Automatically align time series using cross-correlation with normalization options
        
        Args:
            reference_label: Label of the reference time series
            cross_correlation_window: Window size for cross-correlation in SAMPLES
            correlation_window_time: Window size for cross-correlation in SECONDS
            use_original_positions: Use original or current time vectors
            use_raw_data: Use raw (uncropped) or processed data
            visualize: Show correlation plots for each signal pair
            max_shift_time: Maximum shift to search in seconds
            normalize_signals: Remove DC offset and normalize amplitude before correlation
            correlation_method: 'normalized', 'standard', or 'zero_mean'
            
        Returns:
            Dictionary of calculated time shifts for each dataset
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
        
        def calculate_correlation(ref_sig, target_sig, method='normalized'):
            """Calculate cross-correlation with different methods"""
            if method == 'normalized':
                # Normalized cross-correlation (template matching)
                # This is less sensitive to amplitude and offset differences
                ref_norm = normalize_for_correlation(ref_sig, 'normalized')
                target_norm = normalize_for_correlation(target_sig, 'normalized')
                
                # Use scipy's correlation for better numerical stability
                from scipy.signal import correlate
                correlation = correlate(ref_norm, target_norm, mode='full')
                
                # Normalize by signal lengths for proper scaling
                n = min(len(ref_norm), len(target_norm))
                correlation = correlation / n
                
            elif method == 'zero_mean':
                # Zero-mean correlation (removes DC offset only)
                ref_zm = normalize_for_correlation(ref_sig, 'zero_mean')
                target_zm = normalize_for_correlation(target_sig, 'zero_mean')
                correlation = np.correlate(ref_zm, target_zm, mode='full')
                
            else:
                # Standard correlation (most sensitive to offset/amplitude)
                correlation = np.correlate(ref_sig, target_sig, mode='full')
            
            return correlation
        
        # [Previous setup code remains the same...]
        if reference_label not in self.processed_data:
            raise ValueError(f"Reference dataset '{reference_label}' not found")
        
        position_type = "original" if use_original_positions else "current"
        data_type = "raw" if use_raw_data else "processed"
        print(f"\nAuto-aligning time series using '{reference_label}' as reference...")
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
        
        ref_data = data_dict[reference_label]
        ref_time = time_vectors[reference_label]
        calculated_shifts = {}
        
        # [Window calculation code remains the same...]
        if correlation_window_time is not None:
            ref_dt = np.mean(np.diff(ref_time)) if len(ref_time) > 1 else 1.0 / self.sampling_rates[reference_label]
            cross_correlation_window = int(correlation_window_time / ref_dt)
            print(f"Using correlation window: {correlation_window_time:.3f}s ({cross_correlation_window} samples)")
        
        # Visualization setup
        if visualize:
            import matplotlib.pyplot as plt
            n_targets = len([label for label in data_dict.keys() if label != reference_label])
            if n_targets > 0:
                fig, axes = plt.subplots(n_targets, 4, figsize=(20, 5*n_targets))  # Added 4th column
                if n_targets == 1:
                    axes = axes.reshape(1, -1)
                plot_idx = 0
        
        for label, data in data_dict.items():
            if label == reference_label:
                calculated_shifts[label] = 0.0
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
            
            # Calculate correlation with chosen method
            correlation = calculate_correlation(ref_windowed, data_windowed, correlation_method)
            
            # Create lag vectors
            lags_samples = np.arange(-len(data_windowed) + 1, len(ref_windowed))
            dt = np.mean(np.diff(time_vectors[reference_label]))
            lags_time = lags_samples * dt
            
            # Limit search range
            if max_shift_time is not None:
                mask = np.abs(lags_time) <= max_shift_time
                correlation_limited = correlation[mask]
                lags_time_limited = lags_time[mask]
            else:
                correlation_limited = correlation
                lags_time_limited = lags_time
            
            # Find optimal shift
            max_corr_idx = np.argmax(correlation_limited)
            time_shift = lags_time_limited[max_corr_idx]
            calculated_shifts[label] = time_shift
            
            # Enhanced diagnostics
            max_corr_value = correlation_limited[max_corr_idx]
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
                # Plot 1: Original signals
                axes[plot_idx, 0].plot(window_time, ref_windowed, label=f'{reference_label}', alpha=0.8)
                axes[plot_idx, 0].plot(window_time, data_windowed, label=f'{label}', alpha=0.8)
                axes[plot_idx, 0].set_title(f'Original Signals')
                axes[plot_idx, 0].legend()
                axes[plot_idx, 0].grid(True, alpha=0.3)
                
                # Plot 2: Normalized signals (if applicable)
                if normalize_signals:
                    ref_norm = normalize_for_correlation(ref_windowed, correlation_method)
                    target_norm = normalize_for_correlation(data_windowed, correlation_method)
                    axes[plot_idx, 1].plot(window_time, ref_norm, label=f'{reference_label} (norm)', alpha=0.8)
                    axes[plot_idx, 1].plot(window_time, target_norm, label=f'{label} (norm)', alpha=0.8)
                    axes[plot_idx, 1].set_title(f'Normalized Signals ({correlation_method})')
                else:
                    axes[plot_idx, 1].plot(window_time, ref_windowed, label=f'{reference_label}', alpha=0.8)
                    axes[plot_idx, 1].plot(window_time, data_windowed, label=f'{label}', alpha=0.8)
                    axes[plot_idx, 1].set_title(f'Signals (no normalization)')
                axes[plot_idx, 1].legend()
                axes[plot_idx, 1].grid(True, alpha=0.3)
                
                # Plot 3: Cross-correlation
                correlation_norm = correlation_limited / np.max(np.abs(correlation_limited))
                axes[plot_idx, 2].plot(lags_time_limited * 1000, correlation_norm, 'b-', linewidth=1)
                axes[plot_idx, 2].axvline(time_shift * 1000, color='red', linestyle='--', 
                                        label=f'{time_shift*1000:.3f}ms')
                axes[plot_idx, 2].set_xlabel('Lag [ms]')
                axes[plot_idx, 2].set_ylabel('Normalized Correlation')
                axes[plot_idx, 2].set_title(f'Cross-Correlation ({correlation_method})')
                axes[plot_idx, 2].legend()
                axes[plot_idx, 2].grid(True, alpha=0.3)
                
                # Plot 4: Aligned result
                target_shifted = np.interp(window_time - time_shift, window_time, data_windowed)
                axes[plot_idx, 3].plot(window_time, ref_windowed, label=f'{reference_label}', alpha=0.8)
                axes[plot_idx, 3].plot(window_time, target_shifted, label=f'{label} (shifted)', alpha=0.8)
                axes[plot_idx, 3].set_title(f'Aligned Result')
                axes[plot_idx, 3].legend()
                axes[plot_idx, 3].grid(True, alpha=0.3)
                
                plot_idx += 1
        
        if visualize and n_targets > 0:
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
                    'auto_shift': shift
                }
                
                print(f"✓ Applied calculated shift of {shift:.6f}s to {label}")
                print(f"  Total shift from original: {total_shift:.6f}s")
    
    def reset_time_alignment(self, labels: Optional[List[str]] = None) -> None:
        """
        Reset time vectors to original (unshifted) state
        
        Args:
            labels: List of dataset labels to reset (None for all)
        """
        if labels is None:
            labels = list(self.time_vectors.keys())
        
        print(f"\nResetting time alignment for: {', '.join(labels)}")
        
        for label in labels:
            if label in self.original_time_vectors:
                self.time_vectors[label] = self.original_time_vectors[label].copy()
                self.alignment_info[label] = {
                    'time_shift': 0.0,
                    'shift_type': 'reset'
                }
                print(f"✓ Reset {label} to original time vector")
        
        print(f"✓ All specified signals reset to original time alignment")
    
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
    
    def plot_time_series(self, save_path: Optional[str] = None, 
                        show_raw: bool = True, show_processed: bool = True) -> None:
        """Plot time series comparison"""
        n_plots = int(show_raw) + int(show_processed)
        if n_plots == 0:
            return
            
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 6*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot raw data
        if show_raw and self.raw_data:
            ax = axes[plot_idx]
            for i, (label, data) in enumerate(self.raw_data.items()):
                dataset_config = next((d for d in self.datasets if d.label == label), None)
                if dataset_config:
                    color = dataset_config.color if dataset_config.color else None
                    linestyle = dataset_config.linestyle if dataset_config.linestyle else '-'
                else:
                    color = None
                    linestyle = '-'
                
                # Use original time vectors for raw data if available
                time_vec = self.original_time_vectors.get(label, self.time_vectors[label])
                # Ensure time vector matches data length
                if len(time_vec) != len(data):
                    time_vec = self.time_vectors[label]
                
                ax.plot(time_vec[:len(data)], data, 
                       label=f"{label} (raw)", 
                       color=color, linestyle=linestyle, alpha=0.8)
            
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Amplitude')
            ax.set_title('Raw Time Series Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot processed data
        if show_processed and self.processed_data:
            ax = axes[plot_idx]
            for i, (label, data) in enumerate(self.processed_data.items()):
                dataset_config = next((d for d in self.datasets if d.label == label), None)
                if dataset_config:
                    color = dataset_config.color if dataset_config.color else None
                    linestyle = dataset_config.linestyle if dataset_config.linestyle else '-'
                else:
                    color = None
                    linestyle = '-'
                
                # Use current time vectors for processed data
                time_vec = self.time_vectors[label]
                # Ensure time vector matches data length
                time_vec = time_vec[:len(data)]
                
                ax.plot(time_vec, data, 
                       label=f"{label} (processed)", 
                       color=color, linestyle=linestyle, alpha=0.8)
            
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Amplitude')
            ax.set_title('Processed Time Series Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Time series plot saved to {save_path}")
        
        plt.show()
    
    def plot_statistics_summary(self, save_path: Optional[str] = None) -> None:
        """Plot comprehensive statistics summary"""
        if not self.statistics:
            print("No statistics calculated. Run calculate_statistics() first.")
            return
        
        # Create statistics DataFrame
        stats_df = pd.DataFrame(self.statistics).T
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Basic statistics bar plot
        basic_stats = ['mean', 'median', 'std', 'range']
        stats_df[basic_stats].plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Basic Statistics')
        axes[0,0].set_ylabel('Value')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Distribution characteristics
        dist_stats = ['skewness', 'kurtosis']
        stats_df[dist_stats].plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Distribution Characteristics')
        axes[0,1].set_ylabel('Value')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Signal energy and RMS
        energy_stats = ['rms', 'energy']
        stats_df[energy_stats].plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Signal Energy Characteristics')
        axes[1,0].set_ylabel('Value')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Zero crossings
        stats_df[['zero_crossings']].plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Zero Crossings')
        axes[1,1].set_ylabel('Count')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Statistics plot saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, save_path: Optional[str] = None) -> None:
        """Plot correlation matrix heatmap"""
        correlations = self.calculate_correlations()
        
        if not correlations:
            print("No correlations to plot (need at least 2 datasets)")
            return
        
        # Create correlation matrices
        labels = list(self.processed_data.keys())
        n_labels = len(labels)
        
        pearson_matrix = np.eye(n_labels)
        spearman_matrix = np.eye(n_labels)
        
        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels):
                if i != j:
                    pair_key = f"{label1} vs {label2}" if i < j else f"{label2} vs {label1}"
                    if pair_key in correlations:
                        pearson_matrix[i,j] = correlations[pair_key]['pearson']
                        spearman_matrix[i,j] = correlations[pair_key]['spearman']
        
        # Plot correlation matrices
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Pearson correlation
        sns.heatmap(pearson_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_title('Pearson Correlation Matrix')
        
        # Spearman correlation
        sns.heatmap(spearman_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=labels, yticklabels=labels, ax=axes[1])
        axes[1].set_title('Spearman Correlation Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved to {save_path}")
        
        plt.show()
    
    def plot_alignment_comparison(self, save_path: Optional[str] = None, 
                                 use_full_data: bool = True) -> None:
        """Plot comparison of original vs aligned time series"""
        if not self.original_time_vectors:
            print("No original time vectors available for comparison")
            return
        
        # Count datasets with non-zero shifts
        shifted_datasets = [label for label, info in self.alignment_info.items() 
                          if info['time_shift'] != 0.0]
        
        if not shifted_datasets:
            print("No time shifts applied - nothing to compare")
            return
        
        # Choose data source based on use_full_data
        if use_full_data and hasattr(self, 'full_processed_data') and self.full_processed_data:
            plot_data = self.full_processed_data
            data_type = "full processed"
            print("Plotting full processed data (before cropping)")
        else:
            plot_data = self.processed_data
            data_type = "current processed"
            print("Plotting current processed data (potentially cropped)")
        
        n_datasets = len(plot_data)
        fig, axes = plt.subplots(n_datasets, 1, figsize=(15, 4*n_datasets), sharex=True)
        if n_datasets == 1:
            axes = [axes]
        
        for i, (label, data) in enumerate(plot_data.items()):
            ax = axes[i]
            
            # Get shift information
            shift = self.alignment_info[label]['time_shift']
            shift_type = self.alignment_info[label]['shift_type']
            
            # Plot original (unshifted) time series
            if label in self.original_time_vectors:
                orig_time = self.original_time_vectors[label]
                
                if use_full_data:
                    # Use full original time vector with full data
                    ax.plot(orig_time, data, 
                           label=f"{label} (original)", 
                           alpha=0.7, linestyle='--')
                else:
                    # Ensure time vector matches cropped data length
                    orig_time_cropped = orig_time[:len(data)]
                    ax.plot(orig_time_cropped, data, 
                           label=f"{label} (original)", 
                           alpha=0.7, linestyle='--')
            
            # Plot shifted time series
            if shift != 0.0:
                label_text = f"{label} (shifted {shift:+.6f}s, {shift_type})"
                
                if use_full_data:
                    # Create shifted time vector for full data
                    shifted_time = self.original_time_vectors[label] + shift
                    ax.plot(shifted_time, data, 
                           label=label_text, 
                           alpha=0.9, linewidth=1.5, color='red')
                else:
                    # Use current (cropped and shifted) time vectors
                    curr_time = self.time_vectors[label][:len(data)]
                    ax.plot(curr_time, data, 
                           label=label_text, 
                           alpha=0.9, linewidth=1.5, color='red')
            else:
                label_text = f"{label} (no shift)"
                if use_full_data:
                    ax.plot(self.original_time_vectors[label], data, 
                           label=label_text, 
                           alpha=0.9, linewidth=1.5)
                else:
                    curr_time = self.time_vectors[label][:len(data)]
                    ax.plot(curr_time, data, 
                           label=label_text, 
                           alpha=0.9, linewidth=1.5)
            
            # Add shift annotation if significant
            if abs(shift) > 1e-6:
                ax.text(0.02, 0.98, f'Time shift: {shift*1000:+.3f}ms', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Time Alignment Comparison - {label} ({data_type})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time [s]')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Alignment comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, output_dir: str = 'analysis_output') -> None:
        """Generate comprehensive analysis report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\nGenerating comprehensive analysis report in {output_path}")
        
        # Calculate all metrics
        correlations = self.calculate_correlations()
        differences = self.calculate_differences()
        
        # Generate plots
        self.plot_time_series(save_path=output_path / 'time_series_comparison.png')
        self.plot_statistics_summary(save_path=output_path / 'statistics_summary.png')
        self.plot_correlation_matrix(save_path=output_path / 'correlation_matrix.png')
        self.plot_alignment_comparison(save_path=output_path / 'alignment_comparison.png')
        
        # Save alignment summary
        alignment_df = self.get_alignment_summary()
        alignment_df.to_csv(output_path / 'alignment_summary.csv', index=False)
        
        # Save cropping summary if available
        if self.last_cropping_info:
            cropping_df = self.get_cropping_summary(self.last_cropping_info)
            cropping_df.to_csv(output_path / 'cropping_summary.csv', index=False)
        
        # Create text report
        report_path = output_path / 'analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write("TIME SERIES ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset information
            f.write("DATASET INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"HDF5 File: {self.hdf5_path}\n")
            f.write(f"Default Sampling Rate: {self.default_sampling_rate} Hz\n")
            f.write(f"Number of datasets: {len(self.datasets)}\n\n")
            
            for dataset_config in self.datasets:
                f.write(f"Dataset: {dataset_config.label}\n")
                f.write(f"  Group: {dataset_config.group}\n")
                f.write(f"  Name: {dataset_config.name}\n")
                if dataset_config.time_name:
                    f.write(f"  Time vector: {dataset_config.time_group}/{dataset_config.time_name}\n")
                    f.write(f"  Time units: {dataset_config.time_units}\n")
                elif dataset_config.sampling_rate:
                    f.write(f"  Individual sampling rate: {dataset_config.sampling_rate} Hz\n")
                else:
                    f.write(f"  Using default sampling rate: {self.default_sampling_rate} Hz\n")
                
                if dataset_config.time_shift != 0.0:
                    f.write(f"  Manual time shift: {dataset_config.time_shift:.6f} s\n")
                
                if dataset_config.label in self.processed_data:
                    f.write(f"  Samples: {len(self.processed_data[dataset_config.label])}\n")
                    f.write(f"  Actual sampling rate: {self.sampling_rates[dataset_config.label]:.2f} Hz\n")
                    duration = self.time_vectors[dataset_config.label][-1] - self.time_vectors[dataset_config.label][0]
                    f.write(f"  Duration: {duration:.6f} s\n")
                f.write("\n")
            
            # Time alignment summary
            f.write("TIME ALIGNMENT SUMMARY\n")
            f.write("-" * 22 + "\n")
            for label, info in self.alignment_info.items():
                f.write(f"{label}:\n")
                f.write(f"  Time shift: {info['time_shift']:.6f} s\n")
                f.write(f"  Shift type: {info['shift_type']}\n")
                f.write("\n")
            
            # Processing configuration
            f.write("PROCESSING CONFIGURATION\n")
            f.write("-" * 25 + "\n")
            config_dict = self.processing_config.__dict__
            for key, value in config_dict.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Statistics
            f.write("STATISTICS SUMMARY\n")
            f.write("-" * 18 + "\n")
            for label, stats in self.statistics.items():
                f.write(f"{label}:\n")
                for stat_name, stat_value in stats.items():
                    f.write(f"  {stat_name}: {stat_value:.6f}\n")
                f.write("\n")
            
            # Correlations
            f.write("CORRELATION ANALYSIS\n")
            f.write("-" * 20 + "\n")
            for pair, corr_data in correlations.items():
                f.write(f"{pair}:\n")
                f.write(f"  Pearson: {corr_data['pearson']:.6f}\n")
                f.write(f"  Spearman: {corr_data['spearman']:.6f}\n")
                f.write("\n")
            
            # Differences
            f.write("DIFFERENCE ANALYSIS\n")
            f.write("-" * 19 + "\n")
            for pair, diff_data in differences.items():
                f.write(f"{pair}:\n")
                for metric, value in diff_data.items():
                    f.write(f"  {metric}: {value:.6f}\n")
                f.write("\n")
            
            # Cropping information if available
            if self.last_cropping_info:
                f.write("CROPPING INFORMATION\n")
                f.write("-" * 20 + "\n")
                for label, crop_info in self.last_cropping_info.items():
                    f.write(f"{label}:\n")
                    f.write(f"  Original: {crop_info['original_samples']} samples ({crop_info['original_duration']:.6f}s)\n")
                    f.write(f"  Cropped: {crop_info['cropped_samples']} samples ({crop_info['cropped_duration']:.6f}s)\n")
                    f.write(f"  Removed: {crop_info['original_samples'] - crop_info['cropped_samples']} samples\n")
                    f.write(f"  Time range: {crop_info['cropped_start']:.6f}s to {crop_info['cropped_end']:.6f}s\n")
                    f.write("\n")
        
        print(f"✓ Analysis report saved to {report_path}")
        print(f"✓ Alignment summary saved to {output_path / 'alignment_summary.csv'}")
        if self.last_cropping_info:
            print(f"✓ Cropping summary saved to {output_path / 'cropping_summary.csv'}")

    def get_data_summary(self) -> None:
        """Print summary of all stored data"""
        print("\n=== Data Storage Summary ===")
        
        for attr_name in ['raw_data', 'processed_data', 'full_processed_data', 
                          'original_processed_data', 'original_raw_data']:
            if hasattr(self, attr_name):
                data_dict = getattr(self, attr_name)
                if data_dict:
                    print(f"\n{attr_name}:")
                    for label, data in data_dict.items():
                        print(f"  {label}: {len(data)} samples")
                else:
                    print(f"\n{attr_name}: Empty")
            else:
                print(f"\n{attr_name}: Not initialized")

def main():
    """Main function demonstrating usage"""
    
    # Hardcoded dataset configurations (modify as needed)
    datasets = [
        DatasetConfig(
            group='AMPM',
            name='Photodiode1Bits',
            label='PD1',
            # color='#f98e09',
            linestyle='-',
            time_group='AMPM',  # Time vector in same group
            time_name='Time',   # Explicit time vector
            time_units='s',     # Time in seconds
            time_shift=0.0      # No phase shift
        ),
        DatasetConfig(
            group='AMPM',
            name='Photodiode2Bits', 
            label='PD2',
            # color='#57106e',
            linestyle='-',
            time_group='AMPM',  # Time vector in same group
            time_name='Time',   # Shared time vector
            time_units='s',     # Time in seconds
            time_shift=0.0      # No phase shift (was 0.005 in example)
        ),
        DatasetConfig(
            group='KH',
            name='max_depth', 
            label='KH depth',
            # color='#57106e',
            linestyle='-',
            time_group='KH',  # Time vector in same group
            time_name='time',   # Shared time vector
            time_units='s',     # Time in seconds
            time_shift=0.00165      # No phase shift (was 0.005 in example)
        ),
        # DatasetConfig(
            # group='KH',
            # name='area', 
            # label='KH area',
            # color='#57106e',
            # linestyle='-',
            # time_group='KH',  # Time vector in same group
            # time_name='time',   # Shared time vector
            # time_units='s',     # Time in seconds
            # time_shift=0.00165      # No phase shift (was 0.005 in example)
        # ),
        DatasetConfig(
            group='KH',
            name='fkw_angle', 
            label='FKW angle',
            # color='#57106e',
            linestyle='-',
            time_group='KH',  # Time vector in same group
            time_name='time',   # Shared time vector
            time_units='s',     # Time in seconds
            time_shift=0.00165      # No phase shift (was 0.005 in example)
        ),
        # Example with individual sampling rate and time shift
        # DatasetConfig(
        #     group='ProcessData',
        #     name='LaserPower',
        #     label='Laser Power',
        #     color='red',
        #     linestyle=':',
        #     sampling_rate=50.0,  # Individual sampling rate
        #     time_units='s',
        #     time_shift=-0.002    # 2ms advance
        # ),
        # Example with time vector in different units and phase shift
        # DatasetConfig(
        #     group='HighSpeed',
        #     name='Pyrometer',
        #     label='Temperature',
        #     color='orange',
        #     linestyle='-.',
        #     time_group='HighSpeed',
        #     time_name='TimeMs',  # Time vector in milliseconds
        #     time_units='ms',     # Will be converted to seconds
        #     time_shift=0.010     # 10ms delay
        # )
    ]
    
    # Processing configuration
    processing_config = ProcessingConfig(
        # Enable Savitzky-Golay filtering
        apply_savgol=False,
        savgol_window=5,
        savgol_polyorder=3,
        
        # Enable low-pass filtering
        apply_lowpass=False,
        lowpass_cutoff=0.3,
        lowpass_order=4,
        
        # Enable detrending
        apply_detrend=False,
        detrend_method='linear',
        
        # Enable normalization
        apply_normalization=True,
        normalization_method='standard',
        
        # Disable other options for this example
        apply_highpass=False,
        apply_bandpass=False,
        apply_smoothing=True,
        apply_resampling=False
    )
    
    # Example usage
    # hdf5_file = "E:/ESRF ME1573 LTP 6 Al data HDF5/ffc/1112_06.hdf5"  # Update this path
    folder = get_paths()['hdf5']  # Update this path
    trackid = '1112_01'
    hdf5_file = Path(folder, trackid+'.hdf5')
    
    default_sampling_rate = 100.0  # kHz - used as fallback
    
    # Initialize comparator
    comparator = TimeSeriesComparator(
        hdf5_path=hdf5_file,
        datasets=datasets,
        processing_config=processing_config,
        default_sampling_rate=default_sampling_rate
    )
    
    try:
        # Load and process data
        comparator.load_data()
        comparator.process_data()
        
        # Example of automatic alignment (optional)
        # Option 1: Auto-align using sample-based window
        # calculated_shifts = comparator.auto_align_time_series('PD1', cross_correlation_window=20)
        
        # Option 2: Auto-align using time-based window (recommended)
        # With visualization to debug correlation
        # Normalized correlation (recommended)
        calculated_shifts = comparator.auto_align_time_series('PD1', 
                                                              correlation_window_time=0.005,
                                                              use_raw_data=True,
                                                              correlation_method='normalized',
                                                              visualize=True,
                                                              max_shift_time=0.0005)
        
        # Option 3: Auto-align from original positions (ignoring manual shifts)
        # calculated_shifts = comparator.auto_align_time_series('PD1', 
        #                                                      correlation_window_time=1.0,
        #                                                      use_original_positions=True)
        
        # Apply calculated shifts
        # comparator.apply_calculated_shifts(calculated_shifts)
        
        # Debug time shifts
        print("\n=== Time Vector Debug ===")
        for label in ['PD1', 'PD2', 'KH depth']:
            if label in comparator.time_vectors:
                current_shift = comparator.alignment_info[label]['time_shift']
                time_start = comparator.time_vectors[label][0]
                original_start = comparator.original_time_vectors[label][0]
                print(f"{label}:")
                print(f"  Total shift: {current_shift:.6f}s")
                print(f"  Original start: {original_start:.6f}s")
                print(f"  Current start: {time_start:.6f}s")
                print(f"  Expected start: {original_start + current_shift:.6f}s")
                print(f"  Match: {abs(time_start - (original_start + current_shift)) < 1e-6}")
        
        # Or apply relative to original positions
        # comparator.apply_calculated_shifts(calculated_shifts, relative_to_original=True)
        
        # Example of cropping to shortest signal (optional)
        # Option 1: Crop processed data
        cropping_info = comparator.crop_to_shortest_signal(use_processed_data=True)
        comparator.last_cropping_info = cropping_info  # Store for reporting
        
        # Option 2: Crop raw data  
        # cropping_info = comparator.crop_to_shortest_signal(use_processed_data=False)
        
        # View cropping summary
        print("\nCropping Summary:")
        print(comparator.get_cropping_summary(cropping_info).to_string(index=False))
        
        # Restore original lengths if needed
        # comparator.restore_original_length(use_processed_data=True)
        
        # Print alignment summary
        print("\nTime Alignment Summary:")
        print(comparator.get_alignment_summary().to_string(index=False))
        
        # Calculate statistics after cropping
        comparator.calculate_statistics()
        
        # Show all stored datasets
        comparator.get_data_summary()
        
        # Generate comprehensive report
        output_parent = Path(hdf5_file).parent
        output_path = Path(output_parent, 'timeseries_compare_analysis_results')
        comparator.generate_report(output_path)
        
        print("\n✓ Analysis complete! Check the 'timeseries_compare_analysis_results' folder for outputs.")
        print("✓ Alignment comparison plot shows original vs shifted time series.")
        print("✓ Cropping summary shows data reduction details.")
        
    except FileNotFoundError:
        print(f"Error: Could not find HDF5 file at {hdf5_file}")
        print("Please update the hdf5_file variable with the correct path.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()