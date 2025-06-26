#!/usr/bin/env python
"""
Enhanced Time Series Comparison Script

This script provides comprehensive time series analysis and comparison capabilities
for an arbitrary number of curves stored in HDF5 datasets. It includes advanced
signal processing options and statistical analysis.

Author: AI Assistant
Version: 2.0
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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
    smoothing_window: int = 10
    smoothing_method: str = 'gaussian'  # 'gaussian', 'uniform', 'exponential'
    
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
    
    def process_data(self) -> None:
        """Process all loaded datasets using individual sampling rates"""
        print("\nProcessing signals...")
        
        for label, data in self.raw_data.items():
            print(f"Processing {label}...")
            # Use individual sampling rate for processing
            sr = self.sampling_rates[label]
            processed = self.processor.process_signal(data, sr)
            self.processed_data[label] = processed
            
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
                              use_original_positions: bool = False) -> Dict[str, float]:
        """
        Automatically align time series using cross-correlation
        
        Args:
            reference_label: Label of the reference time series
            cross_correlation_window: Window size for cross-correlation in SAMPLES (None for full signal)
            correlation_window_time: Window size for cross-correlation in SECONDS (overrides cross_correlation_window if provided)
            use_original_positions: If True, calculate shifts relative to original (unshifted) positions.
                                   If False, calculate shifts relative to current (manually shifted) positions.
            
        Returns:
            Dictionary of calculated time shifts for each dataset
        """
        if reference_label not in self.processed_data:
            raise ValueError(f"Reference dataset '{reference_label}' not found")
        
        position_type = "original" if use_original_positions else "current"
        print(f"\nAuto-aligning time series using '{reference_label}' as reference...")
        print(f"Calculating shifts relative to {position_type} positions")
        
        # Choose time vectors based on use_original_positions flag
        if use_original_positions:
            time_vectors = self.original_time_vectors
            print("Using original (unshifted) time vectors for alignment calculation")
        else:
            time_vectors = self.time_vectors
            print("Using current (potentially shifted) time vectors for alignment calculation")
        
        ref_data = self.processed_data[reference_label]
        ref_time = time_vectors[reference_label]
        calculated_shifts = {}
        
        # Convert time-based window to samples if provided
        if correlation_window_time is not None:
            ref_dt = np.mean(np.diff(ref_time)) if len(ref_time) > 1 else 1.0 / self.sampling_rates[reference_label]
            cross_correlation_window = int(correlation_window_time / ref_dt)
            print(f"Using correlation window: {correlation_window_time:.3f}s ({cross_correlation_window} samples)")
        elif cross_correlation_window is not None:
            ref_dt = np.mean(np.diff(ref_time)) if len(ref_time) > 1 else 1.0 / self.sampling_rates[reference_label]
            window_time = cross_correlation_window * ref_dt
            print(f"Using correlation window: {cross_correlation_window} samples ({window_time:.3f}s)")
        else:
            print("Using full signal length for correlation")
        
        for label, data in self.processed_data.items():
            if label == reference_label:
                calculated_shifts[label] = 0.0
                continue
            
            time_vec = time_vectors[label]
            
            # Synchronize for cross-correlation analysis
            ref_sync, data_sync = self._synchronize_time_series(ref_data, ref_time, data, time_vec)
            
            # Apply windowing if specified
            if cross_correlation_window and cross_correlation_window < len(ref_sync):
                center = len(ref_sync) // 2
                half_window = cross_correlation_window // 2
                start_idx = max(0, center - half_window)
                end_idx = min(len(ref_sync), center + half_window)
                ref_windowed = ref_sync[start_idx:end_idx]
                data_windowed = data_sync[start_idx:end_idx]
            else:
                ref_windowed = ref_sync
                data_windowed = data_sync
            
            # Calculate cross-correlation
            correlation = np.correlate(ref_windowed, data_windowed, mode='full')
            
            # Find the lag with maximum correlation
            max_corr_idx = np.argmax(correlation)
            lag_samples = max_corr_idx - (len(data_windowed) - 1)
            
            # Convert lag to time shift
            dt = np.mean(np.diff(time_vectors[reference_label]))
            time_shift = lag_samples * dt
            
            calculated_shifts[label] = time_shift
            
            print(f"✓ Calculated time shift for {label}: {time_shift:.6f}s")
        
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
    
    def get_alignment_summary(self) -> pd.DataFrame:
        """Get summary of all time alignments applied"""
        alignment_data = []
        
        for label, info in self.alignment_info.items():
            alignment_data.append({
                'Dataset': label,
                'Time Shift [s]': info['time_shift'],
                'Shift Type': info['shift_type'],
                'Original Duration [s]': (self.original_time_vectors[label][-1] - 
                                        self.original_time_vectors[label][0]) if label in self.original_time_vectors else 'N/A',
                'Current Duration [s]': (self.time_vectors[label][-1] - 
                                       self.time_vectors[label][0]) if label in self.time_vectors else 'N/A'
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
                if not hasattr(self, 'original_processed_data'):
                    self.original_processed_data = {}
                for label, data in self.processed_data.items():
                    if label not in self.original_processed_data:
                        self.original_processed_data[label] = data.copy()
            else:
                if not hasattr(self, 'original_raw_data'):
                    self.original_raw_data = {}
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
            
            print(f"  {label}: {duration:.3f}s ({len(data)} samples) - Start: {start_time:.3f}s, End: {end_time:.3f}s")
        
        if earliest_start >= latest_end:
            print("Error: No overlapping time range found between signals!")
            return duration_info
        
        common_duration = latest_end - earliest_start
        print(f"\nCommon time range: {earliest_start:.3f}s to {latest_end:.3f}s")
        print(f"Common duration: {common_duration:.3f}s")
        
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
            
            self.time_vectors[label] = cropped_time
            
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
            print(f"✓ Cropped {label}: {len(cropped_data)} samples ({cropped_time[-1] - cropped_time[0]:.3f}s) - Removed {samples_removed} samples")
        
        print(f"\n✓ All signals cropped to common duration of {common_duration:.3f}s")
        
        return cropping_info
    
    def restore_original_length(self, use_processed_data: bool = True) -> None:
        """
        Restore signals to their original length before cropping
        
        Args:
            use_processed_data: If True, restore processed data. If False, restore raw data.
        """
        data_type = "processed" if use_processed_data else "raw"
        
        if use_processed_data:
            if not hasattr(self, 'original_processed_data') or not self.original_processed_data:
                print(f"No original {data_type} data available for restoration")
                return
            restore_from = self.original_processed_data
            restore_to = self.processed_data
        else:
            if not hasattr(self, 'original_raw_data') or not self.original_raw_data:
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
                'Original Duration [s]': f"{info['original_duration']:.3f}",
                'Original Samples': info['original_samples'],
                'Cropped Duration [s]': f"{info['cropped_duration']:.3f}",
                'Cropped Samples': info['cropped_samples'],
                'Samples Removed': info['original_samples'] - info['cropped_samples'],
                'Removal Ratio [%]': f"{((info['original_samples'] - info['cropped_samples']) / info['original_samples'] * 100):.1f}",
                'Start Time [s]': f"{info['cropped_start']:.3f}",
                'End Time [s]': f"{info['cropped_end']:.3f}"
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
                dataset_config = next(d for d in self.datasets if d.label == label)
                color = dataset_config.color if dataset_config.color else None
                linestyle = dataset_config.linestyle if dataset_config.linestyle else '-'
                
                ax.plot(self.time_vectors[label], data, 
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
                dataset_config = next(d for d in self.datasets if d.label == label)
                color = dataset_config.color if dataset_config.color else None
                linestyle = dataset_config.linestyle if dataset_config.linestyle else '-'
                
                ax.plot(self.time_vectors[label], data, 
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
    
    def plot_alignment_comparison(self, save_path: Optional[str] = None) -> None:
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
        
        n_datasets = len(self.processed_data)
        fig, axes = plt.subplots(n_datasets, 1, figsize=(15, 4*n_datasets))
        if n_datasets == 1:
            axes = [axes]
        
        for i, (label, data) in enumerate(self.processed_data.items()):
            ax = axes[i]
            
            # Plot original (unshifted) time series
            if label in self.original_time_vectors:
                ax.plot(self.original_time_vectors[label], data, 
                       label=f"{label} (original)", 
                       alpha=0.7, linestyle='--')
            
            # Plot current (potentially shifted) time series
            shift = self.alignment_info[label]['time_shift']
            shift_type = self.alignment_info[label]['shift_type']
            
            if shift != 0.0:
                label_text = f"{label} (shifted {shift:+.3f}s, {shift_type})"
            else:
                label_text = f"{label} (no shift)"
            
            ax.plot(self.time_vectors[label], data, 
                   label=label_text, 
                   alpha=0.9, linewidth=1.5)
            
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Time Alignment Comparison - {label}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time [s]')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Alignment comparison plot saved to {save_path}")
        
        plt.show()
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
                
                if dataset_config.label in self.processed_data:
                    f.write(f"  Samples: {len(self.processed_data[dataset_config.label])}\n")
                    f.write(f"  Actual sampling rate: {self.sampling_rates[dataset_config.label]:.2f} Hz\n")
                    duration = self.time_vectors[dataset_config.label][-1] - self.time_vectors[dataset_config.label][0]
                    f.write(f"  Duration: {duration:.3f} s\n")
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
        if hasattr(self, 'last_cropping_info') and self.last_cropping_info:
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
                    f.write(f"  Duration: {duration:.3f} s\n")
                f.write("\n")
            
            # Time alignment summary
            f.write("TIME ALIGNMENT SUMMARY\n")
            f.write("-" * 22 + "\n")
            for label, info in self.alignment_info.items():
                f.write(f"{label}:\n")
                f.write(f"  Time shift: {info['time_shift']:.6f} s\n")
                f.write(f"  Shift type: {info['shift_type']}\n")
                f.write("\n")
                
            # Cropping information if available
            if hasattr(self, 'last_cropping_info') and self.last_cropping_info:
                f.write("CROPPING INFORMATION\n")
                f.write("-" * 20 + "\n")
                for label, crop_info in self.last_cropping_info.items():
                    f.write(f"{label}:\n")
                    f.write(f"  Original: {crop_info['original_samples']} samples ({crop_info['original_duration']:.3f}s)\n")
                    f.write(f"  Cropped: {crop_info['cropped_samples']} samples ({crop_info['cropped_duration']:.3f}s)\n")
                    f.write(f"  Removed: {crop_info['original_samples'] - crop_info['cropped_samples']} samples\n")
                    f.write(f"  Time range: {crop_info['cropped_start']:.3f}s to {crop_info['cropped_end']:.3f}s\n")
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
        
        print(f"✓ Analysis report saved to {report_path}")
        print(f"✓ Alignment summary saved to {output_path / 'alignment_summary.csv'}")
        if hasattr(self, 'last_cropping_info') and self.last_cropping_info:
            print(f"✓ Cropping summary saved to {output_path / 'cropping_summary.csv'}")


def main():
    """Main function demonstrating usage"""
    
    # Hardcoded dataset configurations (modify as needed)
    datasets = [
        DatasetConfig(
            group='AMPM',
            name='Photodiode1Bits',
            label='PD1',
            color='#f98e09',
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
            color='#bc3754',
            linestyle='-',
            time_group='AMPM',  # Time vector in same group
            time_name='Time',   # Shared time vector
            time_units='s',     # Time in seconds
            time_shift=0.0    
        ),
        DatasetConfig(
            group='KH',
            name='max_depth', 
            label='KH depth',
            color='#57106e',
            linestyle='-',
            time_group='KH',  # Time vector in same group
            time_name='time',   # Shared time vector
            time_units='s',     # Time in seconds
            time_shift=0.0017    # 1.7 ms delay to align with PD1
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
        apply_savgol=True,
        savgol_window=11,
        savgol_polyorder=3,
        
        # Enable low-pass filtering
        apply_lowpass=False,
        lowpass_cutoff=0.1,
        lowpass_order=4,
        
        # Enable detrending
        apply_detrend=False,
        detrend_method='linear',
        
        # Enable normalization
        apply_normalization=False,
        normalization_method='standard',
        
        # Disable other options for this example
        apply_highpass=False,
        apply_bandpass=False,
        apply_smoothing=False,
        apply_resampling=False
    )
    
    # Example usage
    hdf5_file = "E:/ESRF ME1573 LTP 6 Al data HDF5/ffc/1112_01.hdf5"  # Update this path
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
        comparator.calculate_statistics()
        
        # Example of automatic alignment (optional)
        # Option 1: Auto-align using sample-based window
        # calculated_shifts = comparator.auto_align_time_series('PD1', cross_correlation_window=1000)
        
        # Option 2: Auto-align using time-based window (recommended)
        calculated_shifts = comparator.auto_align_time_series('PD1', correlation_window_time=0.0002)  # 0.5 second window
        
        # Option 3: Auto-align from original positions (ignoring manual shifts)
        # calculated_shifts = comparator.auto_align_time_series('PD1', 
        #                                                      correlation_window_time=1.0,
        #                                                      use_original_positions=True)
        
        # Apply calculated shifts
        comparator.apply_calculated_shifts(calculated_shifts)
        
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
        
        # Generate comprehensive report
        comparator.generate_report('analysis_results')
        
        print("\n✓ Analysis complete! Check the 'analysis_results' folder for outputs.")
        print("✓ Alignment comparison plot shows original vs shifted time series.")
        
    except FileNotFoundError:
        print(f"Error: Could not find HDF5 file at {hdf5_file}")
        print("Please update the hdf5_file variable with the correct path.")
    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == '__main__':
    main()


