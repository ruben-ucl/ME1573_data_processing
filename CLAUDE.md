# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains scripts for analyzing greyscale radiography data from laser powder bed fusion (LPBF) additive manufacturing experiments. The data is primarily stored in HDF5 format and processed for keyhole measurements, melt pool analysis, and machine learning classification.

## Architecture & Key Components

### Core Library
- `tools.py`: Central library containing shared functions across the entire project
  - Path management via `get_paths()` - reads directory paths from `dirs/*.txt` files
  - Logbook data handling with `get_logbook()` and `get_logbook_data()`
  - Column label definitions for consistent plotting with `define_collumn_labels()`
  - Image processing utilities (median filtering, histogram equalization)
  - Timeseries data quality validation and interpolation functions
  - CWT scales optimization for different wavelets

### Module Structure
- `file/`: File management and format conversion
  - HDF5 operations (`csv_to_hdf5.py`, `tiff_to_hdf5.py`, `inspect_hdf5.py`)
  - AMPM data processing (`read_AMPM.py`, `trim_AMPM.py`)
  - Keyhole measurements conversion (`keyhole_measurements_csv_to_hdf5.py`)

- `prep/`: Image preprocessing and preparation
  - Background subtraction, denoising, filtering
  - Flat field correction, histogram equalization
  - Coordinate transformation (`to_lagrangian.py`)

- `segm/`: Binary segmentation and analysis
  - Thresholding and connected component analysis
  - Keyhole refinement and centroid filtering
  - WEKA-based keyhole measurements

- `meas/`: Feature measurement and analysis
  - Keyhole depth, porosity, layer thickness measurements
  - Signal correlation and time series summarization
  - FKW angle measurements

- `vis/`: Data visualization and plotting
  - 3D plotting with multi-contour support
  - Statistical boxplots and heatmaps (`boxplotter.py`)
  - Continuous wavelet transform visualization (`cwt.py`)
  - FFT analysis and video generation

- `ml/`: Machine learning models
  - CWT image binary classification using TensorFlow/Keras
  - Photodiode signal classification
  - Pre-trained models stored in `ml/models/`

- `sim/`: Simulation scripts
  - Ansys Additive integration for LPBF simulation

### Data Organization
- `dirs/`: Text files containing file paths referenced by scripts
  - `AMPM.txt`, `FKW_meas.txt`, `KH_meas.txt`, `hdf5.txt`, etc.
  - Always use `get_paths()` from tools.py to read these paths

## Development Practices

### Core Principles
- **Minimize Complexity**: Always prefer deletion over addition when solving problems
  - If columns/data already exist, use them instead of recreating
  - If a bug can be fixed by removing unnecessary code, do that instead of adding workarounds
  - Question whether new code is truly necessary before adding it
  - Simpler solutions are more maintainable and less error-prone

### Environment Setup
- **Conda Environment**: Always use the `ml` conda environment
- **Script Execution**: Run from repository root directory
  ```bash
  conda activate ml
  python ml/script_name.py
  ```
- **Working Directory**: Scripts expect the working directory to be the project root
- **Module Imports**: `from tools import get_paths, get_logbook`

### Encoding & Text Handling  
- **UTF-8 by Default**: All ML scripts automatically set `PYTHONIOENCODING='utf-8'`
- **File Operations**: Text files use UTF-8 encoding automatically
- **CSV Operations**: pandas operations explicitly specify `encoding='utf-8'` for reliability
- **No Encoding Errors**: Scripts handle special characters and Unicode properly in Git Bash/Windows

### Path Management
- **No Hardcoded Paths**: Never hardcode file paths in scripts
- **Centralized Configuration**: 
  - General scripts: Use `get_paths()` function to read paths from `dirs/*.txt` files
  - ML scripts: Use `config.py` for centralized path management
- **Environment Variable Support**: Override default paths using:
  - `ML_DATA_DIR`: Custom photodiode data directory
  - `ML_CWT_DATA_DIR`: Custom CWT data directory  
- **Cross-Platform Paths**: Use `pathlib.Path` for new code, automatic normalization

### Version Management
- **Consistent Formatting**: All version numbers use format `v001, v002, v003...`
- **Centralized Functions**: Use `format_version()` and `get_next_version_from_log()` from `config.py`
- **Automatic Incrementing**: Version numbers auto-increment based on experiment log

### Data Handling
- **Primary Format**: HDF5 for time series and image data
- **Logbook Data**: Excel format with specific column naming conventions
- **Column Labels**: Use `define_collumn_labels()` from tools.py for consistency
- **Data Quality**: Validate timeseries using `interpolate_low_quality_data()` and `validate_timeseries_quality()`

### Machine Learning Conventions
- **Framework**: TensorFlow/Keras for classification tasks
- **Feature Extraction**: CWT-based features for time series classification
- **Model Storage**: Artifacts and logs stored in `ml/models/`
- **Experiment Tracking**: Unified CSV log in `ml/logs/experiment_log.csv`
- **Hyperparameter Optimization**:
  - Use modes: `test` → `quick` → `medium` → `smart` → `full`
  - Smart deduplication prevents duplicate experiments
  - Resume functionality for interrupted runs
  - Progress tracking with accurate config counting

### Code Quality
- **Error Handling**: Proper exception handling with descriptive messages
- **Logging**: Structured experiment logging with JSON + human-readable formats
- **Progress Display**: Clear progress indicators showing `X/Y` configs completed
- **Verbosity Levels**: Support `--verbose`, `--concise` modes appropriately

## Key Dependencies
- Scientific: numpy, pandas, scipy, scikit-learn, scikit-image
- Visualization: matplotlib, seaborn
- ML: tensorflow, keras
- Data: h5py, PIL, opencv-python
- Signal processing: pywt (PyWavelets)
- Statistics: scikit_posthocs

## Troubleshooting

### Common Issues & Solutions

#### Encoding Errors (CRITICAL - Updated 2025-08-21)
- **Primary Error**: `'charmap' codec can't decode byte 0x8f in position X: character maps to <undefined>`
- **Root Cause**: Git Bash on Windows defaults to CP1252 (charmap) encoding, but Python scripts output UTF-8 characters
- **When It Occurs**: 
  - During subprocess execution in hyperparameter tuner
  - When training scripts output progress bars, special characters, or Unicode symbols
  - Intermittent - depends on specific output content generated during training

**Progressive Solution History:**
1. **Initial Fix**: Added `encoding='utf-8'` to file operations ❌ (Insufficient)
2. **Script-Level Fix**: Added `os.environ.setdefault('PYTHONIOENCODING', 'utf-8')` at script startup ❌ (Still occurring)  
3. **Subprocess Environment Fix**: Explicit environment variable passing to child processes ❌ (Still occurring)
4. **Direct Subprocess Encoding**: Add explicit encoding to subprocess calls ✅ **WORKING SOLUTION**

**FINAL WORKING SOLUTION (Confirmed 2025-08-21):**
```python
# Direct subprocess encoding specification:
result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace',  # Replace problematic chars instead of crashing
    env=env,  # Still include environment variables
    cwd=str(Path(__file__).parent)
)

# And for Popen:
process = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, encoding='utf-8', errors='replace',
    env=env, cwd=str(Path(__file__).parent)
)
```

**Key Insights from Resolution:**
- Environment variables (`PYTHONIOENCODING`) are insufficient for subprocess output handling
- Direct encoding specification in subprocess calls is the reliable approach  
- The `errors='replace'` parameter prevents crashes and allows graceful handling of any remaining encoding issues
- This solution works consistently across different Git Bash/Windows configurations

**Status**: ✅ **RESOLVED** - No more charmap codec errors occurring

**Historical Alternative Solutions (no longer needed):**
1. **Shell Environment**: Set before running scripts:
   ```bash
   export PYTHONIOENCODING=utf-8
   export PYTHONLEGACYWINDOWSFSENCODING=0
   conda activate ml
   python ml/hyperparameter_tuner.py
   ```

2. **Python UTF-8 Mode**: Use Python's UTF-8 mode flag:
   ```bash
   python -X utf8 ml/hyperparameter_tuner.py
   ```

3. **Console Code Page**: Change Windows console encoding:
   ```cmd
   chcp 65001  # UTF-8
   ```

**Why This Is Tricky:**
- Environment variables set in Python script don't affect subprocess encoding by default
- Git Bash inherits Windows console encoding settings  
- TensorFlow/Keras may output Unicode characters in progress bars
- Issue is intermittent - depends on specific model output during training

**Monitoring Notes:**
- Track if error occurs in specific training phases (progress bars, validation output)
- Note if error correlates with specific hyperparameter configurations
- Monitor if additional subprocess environment variables are needed

#### Module Import Errors  
- **Error**: `ModuleNotFoundError: No module named '...'`
- **Solution**: Always run from project root with ml environment:
  ```bash
  conda activate ml
  python ml/script_name.py  # Not cd ml && python script_name.py
  ```

#### Path Configuration Issues
- **ML Scripts**: Use environment variables to override default paths:
  ```bash
  export ML_DATA_DIR="/custom/path/to/data"
  export ML_CWT_DATA_DIR="/custom/path/to/cwt/data"  
  ```
- **General Scripts**: Update paths in `dirs/*.txt` files

#### Hyperparameter Tuning Issues
- **Duplicate Experiments**: Smart deduplication automatically skips previously tested configurations
- **Resume Not Working**: Ensure you're in the same directory as the previous run
- **Progress Display**: Shows actual running configs (e.g., "Config 3/6") not total generated configs

#### Version Number Issues  
- **Inconsistent Formats**: All scripts now use centralized `format_version()` function for consistent `v001` format
- **Version Conflicts**: Versions auto-increment based on experiment log, no manual management needed

## Important Notes
- The `print` function is overridden in tools.py to fix console output buffering
- Image data is typically uint8 grayscale format  
- CWT analysis uses optimized scales defined in `get_cwt_scales()`
- All file operations use UTF-8 encoding by default
- Progress tracking shows meaningful counts after deduplication and resume filtering
- the above list
- when encountering a similar bug/issue on more than one occassion, promt the user to add the solution to memory to prevent further instances
- keep concise temp logs of troubleshooting, and save final working solutions to claude.md
- no hardcoded paths in scripts, always refer to config.py and dirs/ text files.