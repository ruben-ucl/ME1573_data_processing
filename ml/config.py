"""
Centralized configuration for ML scripts.

This module provides centralized path management and configuration constants
to make it easy to update paths when directory structure changes.
"""

import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent  # Go up to ME1573_data_processing
ML_ROOT = Path(__file__).parent  # Current ml/ directory

# Data directories - Update these paths as needed
DEFAULT_DATA_DIR = r"E:\AlSi10Mg single layer ffc\Photodiode_1ms_window_plots\window_plots_16bit"
CWT_DATA_DIR = r'E:\AlSi10Mg single layer ffc\CWT_labelled_windows\cmor1_5-1_0\1.0_ms\781-50000_Hz_256_steps\grey'

# Output directories (relative to ML_ROOT)
OUTPUTS_DIR = ML_ROOT / "outputs"
LOGS_DIR = ML_ROOT / "logs"
HYPEROPT_RESULTS_DIR = LOGS_DIR / "hyperopt_results"
MODELS_DIR = ML_ROOT / "models"

# CWT-specific directories
CWT_LOGS_DIR = ML_ROOT / "logs" / "cwt"
CWT_HYPEROPT_RESULTS_DIR = CWT_LOGS_DIR / "hyperopt_results"
CWT_MODELS_DIR = ML_ROOT / "models" / "cwt"

# File patterns
TIFF_PATTERN = "*.tiff"
CONFIG_PATTERN = "config_*.json"

# Ensure essential directories exist
def ensure_directories():
    """Create essential directories if they don't exist."""
    for directory in [OUTPUTS_DIR, LOGS_DIR, HYPEROPT_RESULTS_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def ensure_cwt_directories():
    """Create CWT-specific directories if they don't exist."""
    for directory in [CWT_LOGS_DIR, CWT_HYPEROPT_RESULTS_DIR, CWT_MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def get_data_dir():
    """
    Get the data directory path. 
    Checks for environment variable first, then falls back to default.
    """
    return os.environ.get('ML_DATA_DIR', DEFAULT_DATA_DIR)

def get_cwt_data_dir():
    """
    Get the CWT data directory path.
    Checks for environment variable first, then falls back to default.
    """
    return os.environ.get('ML_CWT_DATA_DIR', CWT_DATA_DIR)

def get_experiment_log_path():
    """Get the path to the unified experiment log."""
    return LOGS_DIR / "experiment_log.csv"

def get_cwt_experiment_log_path():
    """Get the path to the CWT-specific experiment log."""
    return CWT_LOGS_DIR / "cwt_experiment_log.csv"

def get_timing_database_path():
    """Get the path to the timing database for hyperparameter tuning."""
    return HYPEROPT_RESULTS_DIR / "timing_database.json"

def get_config_template():
    """Get the default configuration template."""
    return {
        'data_dir': get_data_dir(),
        'img_width': 100,
        'output_root': str(OUTPUTS_DIR),
        'k_folds': 5,
        'epochs': 50,
        'learning_rate': 0.001,
        'batch_size': 16,
        'conv_filters': [16, 32, 64],
        'dense_units': [128, 64],
        'conv_dropout': 0.2,
        'dense_dropout': [0.3, 0.2],
        'l2_regularization': 0.001,
        'use_batch_norm': True,
        'early_stopping_patience': 10,
        'lr_reduction_patience': 5,
        'lr_reduction_factor': 0.5,
        'use_class_weights': True,
        
        # Data augmentation parameters
        'time_shift_range': 5,
        'stretch_probability': 0.3,
        'stretch_scale': 0.1,
        'noise_probability': 0.5,
        'noise_std': 0.02,
        'amplitude_scale_probability': 0.5,
        'amplitude_scale': 0.1,
        'augment_fraction': 0.5,
    }

def get_cwt_config_template():
    """Get the default CWT configuration template."""
    return {
        'cwt_data_dir': get_cwt_data_dir(),
        'img_width': 100,
        'img_height': 256,
        'img_channels': 1,
        'output_root': str(OUTPUTS_DIR),
        'k_folds': 5,
        'epochs': 50,
        'learning_rate': 0.001,
        'batch_size': 16,
        'optimizer': 'adam',
        
        # CNN architecture parameters
        'conv_filters': [16, 16, 32, 32, 64, 64],
        'conv_kernel_size': [3, 3],
        'pool_size': [2, 2],
        'pool_layers': [2, 5],  # Which conv layers to add pooling after
        'dense_units': [128],
        'conv_dropout': 0.0,
        'dense_dropout': 0.5,
        'l2_regularization': 0.001,
        'use_batch_norm': False,
        
        # Training parameters
        'early_stopping_patience': 10,
        'lr_reduction_patience': 5,
        'lr_reduction_factor': 0.5,
        'use_class_weights': True,
        
        # Data augmentation parameters for CWT images
        'rotation_range': 0.0,
        'width_shift_range': 0.0,
        'height_shift_range': 0.0,
        'horizontal_flip': False,
        'vertical_flip': False,
        'noise_std': 0.0,
        'brightness_range': 0.0,
        'contrast_range': 0.0,
        'augment_fraction': 0.0,
        
        # Analysis parameters
        'run_gradcam': True,
        'gradcam_layer': 'auto',  # Auto-detect last conv layer
        'save_gradcam_images': False,
        'gradcam_threshold': 0.7,
    }

def load_config(config_path=None, **overrides):
    """
    Load configuration from file with command line overrides.
    
    Args:
        config_path: Path to JSON config file (optional)
        **overrides: Command line arguments to override config values
    
    Returns:
        dict: Merged configuration
    """
    import json
    
    # Start with default configuration
    config = get_config_template()
    
    # Load from file if provided
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
        except IOError as e:
            raise IOError(f"Could not read config file {config_path}: {e}")
    elif config_path:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Apply command line overrides (filter out None values)
    overrides = {k: v for k, v in overrides.items() if v is not None}
    config.update(overrides)
    
    return config

# Path utility functions
def ensure_path_exists(path):
    """Ensure a directory path exists, creating it if necessary."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def normalize_path(path):
    """Normalize a path to use forward slashes and resolve it."""
    return str(Path(path).resolve()).replace('\\', '/')

# Centralized version management functions
def format_version(version_num):
    """Format version number consistently as v001, v002, etc."""
    if isinstance(version_num, str):
        # Handle string inputs like 'v001', '001', '1', 'v1'
        if version_num.startswith('v'):
            try:
                num = int(version_num[1:])
                return f'v{num:03d}'
            except (ValueError, IndexError):
                return 'v001'  # Fallback
        else:
            try:
                num = int(version_num)
                return f'v{num:03d}'
            except ValueError:
                return 'v001'  # Fallback
    elif isinstance(version_num, int):
        return f'v{version_num:03d}'
    else:
        return 'v001'  # Fallback

def parse_version(version_str):
    """Parse version string to get integer number. Returns None if invalid."""
    if not version_str:
        return None
    
    try:
        version_str = str(version_str).strip()
        if version_str.startswith('v'):
            return int(version_str[1:])
        else:
            return int(version_str)
    except (ValueError, AttributeError, IndexError):
        return None

def get_next_version_from_log(log_path=None):
    """Get next version number based on experiment log entries."""
    import pandas as pd
    
    if log_path is None:
        log_path = get_experiment_log_path()
    
    if not Path(log_path).exists():
        return 1
    
    try:
        df = pd.read_csv(log_path, encoding='utf-8')
        if 'version' not in df.columns or len(df) == 0:
            return 1
        
        existing_versions = []
        for version_str in df['version'].dropna():
            version_num = parse_version(version_str)
            if version_num is not None:
                existing_versions.append(version_num)
        
        return max(existing_versions) + 1 if existing_versions else 1
        
    except (pd.errors.EmptyDataError, FileNotFoundError, Exception):
        return 1

def get_next_version_from_directory(directory, pattern='v*'):
    """Get next version number based on versioned directories."""
    from glob import glob
    
    if not Path(directory).exists():
        return 1
    
    try:
        version_dirs = glob(str(Path(directory) / pattern))
        existing_versions = []
        
        for dir_path in version_dirs:
            dir_name = Path(dir_path).name
            version_num = parse_version(dir_name)
            if version_num is not None:
                existing_versions.append(version_num)
        
        return max(existing_versions) + 1 if existing_versions else 1
        
    except Exception:
        return 1

def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Initialize directories when module is imported
ensure_directories()