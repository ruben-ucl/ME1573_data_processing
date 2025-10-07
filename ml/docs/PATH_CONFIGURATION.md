# Path Configuration System

This document explains the centralized path configuration system implemented for the ML scripts.

## Overview

To make the codebase more maintainable and easier to update when directory structures change, all file paths are now managed through a centralized configuration system in `config.py`.

## Configuration Module (`config.py`)

### Key Features

1. **Centralized Path Management**: All file paths are defined in one place
2. **Environment Variable Support**: Data directories can be overridden using environment variables
3. **Cross-Platform Compatibility**: Uses `pathlib.Path` for proper path handling
4. **Automatic Directory Creation**: Essential directories are created automatically

### Main Components

#### Directory Constants
- `PROJECT_ROOT`: Base project directory (ME1573_data_processing)
- `ML_ROOT`: ML scripts directory (ml/)
- `PD_OUTPUTS_DIR`: Training outputs (ml/outputs/)
- `PD_LOGS_DIR`: Log files (ml/logs/)
- `PD_HYPEROPT_RESULTS_DIR`: Hyperparameter optimization results (ml/logs/hyperopt_results/)

#### Data Directory Functions
- `get_data_dir()`: Returns photodiode data directory (supports `ML_DATA_DIR` env var)
- `get_cwt_data_dir()`: Returns CWT data directory (supports `ML_CWT_DATA_DIR` env var)

#### Path Utility Functions
- `get_pd_experiment_log_path()`: PD signal experiment log file path
- `get_pd_timing_database_path()`: PD signal hyperparameter timing database path
- `get_pd_config_template()`: PD signal default configuration template
- `get_cwt_experiment_log_path()`: CWT image experiment log file path  
- `get_cwt_timing_database_path()`: CWT image hyperparameter timing database path
- `get_cwt_config_template()`: CWT image default configuration template
- `normalize_path()`: Cross-platform path normalization
- `ensure_path_exists()`: Directory creation helper

## Environment Variables

You can override default data directories using environment variables:

```bash
# Set custom data directory
export ML_DATA_DIR="/path/to/your/photodiode/data"

# Set custom CWT data directory  
export ML_CWT_DATA_DIR="/path/to/your/cwt/data"
```

### Windows
```cmd
set ML_DATA_DIR=C:\path\to\your\photodiode\data
set ML_CWT_DATA_DIR=C:\path\to\your\cwt\data
```

### PowerShell
```powershell
$env:ML_DATA_DIR="C:\path\to\your\photodiode\data"
$env:ML_CWT_DATA_DIR="C:\path\to\your\cwt\data"
```

## Migration from Hardcoded Paths

### Before (Hardcoded)
```python
# BAD: Hardcoded absolute path
data_dir = r"E:/AlSi10Mg single layer ffc/Photodiode_1ms_window_plots/window_plots_16bit"
log_file = os.path.join('logs', 'experiment_log.csv')
```

### After (Centralized)
```python
# GOOD: Centralized configuration
from config import get_data_dir, get_pd_experiment_log_path  # or get_cwt_experiment_log_path

data_dir = get_data_dir()
log_file = str(get_pd_experiment_log_path())  # or get_cwt_experiment_log_path() for CWT
```

## Updated Files

The following files have been updated to use the centralized configuration:

1. **`hyperparameter_tuner.py`**: Uses strategy pattern with classifier-specific functions
2. **`PD_signal_classifier_v3.py`**: Uses `get_pd_config_template()`, `get_pd_experiment_log_path()`
3. **`CWT_image_binary_classifier.py`**: Uses `get_cwt_data_dir()`
4. **`example_config.json`**: Updated documentation to mention environment variables

## Benefits

1. **Easy Updates**: Change paths in one place (`config.py`) instead of searching through multiple files
2. **Environment Flexibility**: Different users can set their own data paths via environment variables
3. **No Hardcoded Paths**: Eliminates brittle absolute path dependencies
4. **Cross-Platform**: Proper path handling works on Windows, macOS, and Linux
5. **Automatic Setup**: Essential directories are created automatically
6. **Clear Documentation**: All path-related logic is documented in one place

## Best Practices for Future Development

1. **Always use `config.py`**: Import path functions instead of hardcoding paths
2. **Use `pathlib.Path`**: For new path manipulations, prefer `pathlib` over `os.path`
3. **Support Environment Variables**: Allow users to override paths when reasonable
4. **Document Path Changes**: Update this file when adding new path configurations
5. **Test Cross-Platform**: Ensure paths work on different operating systems

## Example Usage

```python
from config import (
    get_data_dir, get_experiment_log_path, PD_OUTPUTS_DIR,
    ensure_path_exists, normalize_path
)

# Get data directory (respects environment variables)
data_path = get_data_dir()

# Get experiment log path
log_path = get_pd_experiment_log_path()  # or get_cwt_experiment_log_path() for CWT

# Create output directory for experiment
experiment_dir = PD_OUTPUTS_DIR / "v001"
ensure_path_exists(experiment_dir)

# Normalize path for cross-platform compatibility
normalized = normalize_path(some_path)
```

## Troubleshooting

### "Module not found" errors
Make sure you're running scripts from the project root with the ml conda environment:
```bash
conda activate ml
python ml/script_name.py
```

### Path not found errors
1. Check if the environment variable is set correctly
2. Verify the default path in `config.py` exists
3. Ensure you have proper file permissions

### Directory creation errors
The `ensure_directories()` function runs automatically when importing `config.py`. If you get permission errors, check write permissions in the ml/ directory.