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

    # Resampling
    apply_resampling: bool = False
    target_samples: int = 1000
    resampling_method: str = 'linear'  # 'linear', 'cubic', 'nearest'

    # Outlier removal
    remove_outliers: bool = False
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'mad' (median absolute deviation)
    outlier_threshold: float = 3.0  # IQR multiplier (1.5=mild, 3.0=extreme) or z-score threshold
    outlier_window: int = 50  # Will be scaled based on sampling rate (0=global)
    handle_nans: bool = True  # Handle NaN values by default

    # Auto alignment
    apply_auto_alignment: bool = True  # Enable/disable automatic time series alignment
