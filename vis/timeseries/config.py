"""
Configuration dataclasses for time series comparison.

This module contains the configuration classes used throughout the
timeseries comparison package.
"""

from dataclasses import dataclass
from typing import Optional


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
    # Reference sampling rate for time-based window calculations
    reference_sampling_rate: float = 100000.0  # Hz (100 kHz - typical AMPM rate)

    # Filtering options
    apply_savgol: bool = False
    savgol_window: int = 51  # Will be scaled based on sampling rate
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
    smoothing_window: int = 3  # Will be scaled based on sampling rate
    smoothing_method: str = 'uniform'  # 'gaussian', 'uniform', 'exponential'

    # Detrending
    apply_detrend: bool = False
    detrend_method: str = 'linear'  # 'linear', 'constant'

    # Normalization
    apply_normalization: bool = False
    normalization_method: str = 'standard'  # 'standard', 'minmax', 'robust'
    use_global_normalization: bool = False  # Use global statistics across all tracks instead of per-track
    global_stats_file: Optional[str] = None  # Path to hdf5_dataset_statistics.csv (auto-detected if None)

    # Resampling
    apply_resampling: bool = False
    target_samples: int = 1000
    resampling_method: str = 'linear'  # 'linear', 'cubic', 'nearest'

    # Outlier removal
    remove_outliers: bool = False
    outlier_method: str | list[str] = 'iqr'  # Single method or list of methods: 'iqr', 'zscore', 'mad', 'gradient', 'second_derivative'
    outlier_threshold: float | list[float] = 3.0  # Single threshold or list of thresholds (one per method)
    outlier_window: int = 50  # Will be scaled based on sampling rate (0=global)
    outlier_gradient_smoothing: int = 1  # Median filter window for gradient smoothing (gradient/second_derivative methods)
    handle_nans: bool = True  # Handle NaN values by default

    # Auto alignment
    apply_auto_alignment: bool = True  # Enable/disable automatic time series alignment
    apply_manual_lag_corrections: bool = False  # Enable/disable manual per-track lag corrections (overrides auto-alignment)

    # Alignment reference configuration
    alignment_reference_label: str = 'PD1'  # Reference signal for alignment
    alignment_reference_group: Optional[str] = None  # Reference group (overrides reference_label if set)

    # Alignment method selection
    alignment_method: str = 'mutual_info'  # 'ccf' (cross-correlation), 'feature' (feature matching), 'mutual_info' (mutual information)

    # General alignment parameters
    correlation_window_time: float = 0.001  # Window size for cross-correlation in seconds
    max_shift_time: float = 0.0005  # Maximum shift to search in seconds
    correlation_method: str = 'normalized'  # 'normalized', 'standard', or 'zero_mean'
    normalize_signals: bool = True  # Remove DC offset and normalize amplitude before correlation
    sync_within_groups: bool = True  # Maintain synchronization within dataset groups
    use_raw_data: bool = True  # Use raw (uncropped) or processed data for alignment

    # Feature-based alignment parameters (used when alignment_method='feature')
    feature_method: str = 'peak'  # 'peak' (envelope peaks), 'edge' (transitions), 'energy' (energy bursts)
    n_features: int = 5  # Number of features to detect and match

    # Mutual information alignment parameters (used when alignment_method='mutual_info')
    mi_bins: int = 50  # Number of bins for MI histogram
