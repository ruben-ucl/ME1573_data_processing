# How to Run the Time Series Comparison Tool

## Quick Start

The tool works **exactly the same as before** - the refactoring is completely transparent to users!

### Method 1: Run the built-in example (Easiest)

```bash
conda activate ml
python vis/timeseries_compare.py
```

This runs the example script with hardcoded dataset configuration.

### Method 2: Import in your own script

```python
from vis.timeseries import TimeSeriesComparator, DatasetConfig, ProcessingConfig
from pathlib import Path

# Configure your datasets
datasets = [
    DatasetConfig(
        group='AMPM',
        name='Photodiode1Bits',
        label='PD1',
        time_group='AMPM',
        time_name='Time',
        time_units='s'
    ),
    DatasetConfig(
        group='KH',
        name='max_depth',
        label='KH depth',
        time_group='KH',
        time_name='time',
        time_units='s',
        time_shift=0.00165  # Optional manual shift
    ),
]

# Configure processing
processing_config = ProcessingConfig(
    apply_lowpass=True,
    lowpass_cutoff=0.5,
    apply_normalization=True,
    normalization_method='minmax'
)

# Create comparator
comparator = TimeSeriesComparator(
    hdf5_path='path/to/your/file.hdf5',
    datasets=datasets,
    processing_config=processing_config
)

# Run analysis
comparator.load_data()
comparator.process_data()

# Auto-align using PD1 as reference
shifts = comparator.auto_align_time_series(
    reference_label='PD1',
    correlation_window_time=0.001,
    sync_within_groups=True,
    visualize=True
)
comparator.apply_calculated_shifts(shifts)

# Crop and analyze
comparator.crop_to_shortest_signal()
comparator.calculate_statistics()

# Generate report
comparator.generate_report('output_directory')
```

## What's Different?

**Nothing from a user perspective!**

The only visible change is:
- Old import: `from vis.timeseries_compare import ...` (shows deprecation warning)
- New import: `from vis.timeseries import ...` (no warning)

## Configuration

Edit the dataset configuration in `vis/timeseries_compare.py` around line 3095 or create your own script.

Key parameters:
- **HDF5 path**: Path to your data file
- **Datasets**: List of signals to analyze
- **Processing config**: Filtering, normalization options
- **Alignment**: Reference signal and parameters

## Output

The tool generates:
- Processing and alignment summary plots
- Statistics summaries
- Correlation matrices
- Autocorrelation plots
- Cross-correlation plots
- Scatterplot matrices

All saved to the specified output directory.

## Troubleshooting

**Import error?**
- Make sure you're in the project root directory
- Activate the `ml` conda environment

**Can't find HDF5 file?**
- Update the path in the main() function
- Or set it via `get_paths()['hdf5']` from tools.py

**Module not found?**
- Run from project root: `python vis/timeseries_compare.py`
- Not from vis/ directory

## Examples

### Example 1: Basic analysis
```bash
conda activate ml
cd D:/ME1573_data_processing
python vis/timeseries_compare.py
```

### Example 2: Custom script
```python
# my_analysis.py
from vis.timeseries import TimeSeriesComparator, DatasetConfig, ProcessingConfig

# Your custom analysis here
```

Run: `python my_analysis.py`

## Advanced Usage

See `vis/timeseries/README.md` for module structure and details about the refactored package.
