#!/usr/bin/env python
"""
Backward compatibility wrapper for timeseries_compare.py

This file maintains backward compatibility with existing code while
delegating all functionality to the new modular timeseries package.

DEPRECATED: This file is maintained for backward compatibility only.
New code should use: from vis.timeseries import TimeSeriesComparator

Author: AI Assistant
Version: 2.1 (Refactored)
"""

import warnings
import sys
from pathlib import Path

# Add parent directory to path to allow imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import everything from the new modular package
from vis.timeseries import (
    DatasetConfig,
    ProcessingConfig,
    TimeSeriesComparator,
)

# Issue deprecation warning
warnings.warn(
    "Importing from vis.timeseries_compare is deprecated. "
    "Please use 'from vis.timeseries import TimeSeriesComparator' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'DatasetConfig',
    'ProcessingConfig',
    'TimeSeriesComparator',
]

# Import additional dependencies for main() function
from tools import get_paths

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
            time_shift=0.0      # Phase shift
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
            time_shift=0.0      # Phase shift
        ),
        DatasetConfig(
            group='AMPM',
            name='BeamDumpDiodeBits', 
            label='BD',
            # color='#57106e',
            linestyle='-',
            time_group='AMPM',  # Time vector in same group
            time_name='Time',   # Shared time vector
            time_units='s',     # Time in seconds
            time_shift=0.0      # Phase shift
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
            time_shift=0.00165      # Phase shift (~0.00165 in 504kfps)
        ),
        DatasetConfig(
            group='KH',
            name='area', 
            label='KH area',
            color='#57106e',
            linestyle='-',
            time_group='KH',  # Time vector in same group
            time_name='time',   # Shared time vector
            time_units='s',     # Time in seconds
            time_shift=0.00165      # Phase shift
        ),
        DatasetConfig(
            group='KH',
            name='max_length', 
            label='KH length',
            color='#57106e',
            linestyle='-',
            time_group='KH',  # Time vector in same group
            time_name='time',   # Shared time vector
            time_units='s',     # Time in seconds
            time_shift=0.00165      # Phase shift
        ),
        DatasetConfig(
            group='KH',
            name='fkw_angle', 
            label='FKW angle',
            # color='#57106e',
            linestyle='-',
            time_group='KH',  # Time vector in same group
            time_name='time',   # Shared time vector
            time_units='s',     # Time in seconds
            time_shift=0.00165      # Phase shift
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
        savgol_window=21,
        savgol_polyorder=2,

        # Enable low-pass filtering
        apply_lowpass=True,
        lowpass_cutoff=0.5,
        lowpass_order=4,

        # Enable detrending
        apply_detrend=False,
        detrend_method='linear',

        # Enable normalization
        apply_normalization=True,
        normalization_method='minmax',

        # Remove statistical outliers (measurement errors in KH data)
        remove_outliers=True,
        outlier_method='iqr',  # 'iqr', 'zscore', 'mad'
        outlier_threshold=3.0,  # IQR multiplier or z-score threshold
        outlier_window=50,  # Local window size (0=global)

        # Disable other options for this example
        apply_highpass=False,
        apply_bandpass=False,
        apply_smoothing=False,
        apply_resampling=False
    )

    # Example usage
    # hdf5_file = "E:/ESRF ME1573 LTP 6 Al data HDF5/ffc/1112_06.hdf5"  # Update this path
    folder = get_paths()['hdf5']  # Update this path
    trackid = '1112_01'
    hdf5_file = Path(folder, trackid + '.hdf5')
    
    default_sampling_rate = 100.0  # kHz - used as fallback
    
    # Initialize comparator
    comparator = TimeSeriesComparator(
        hdf5_path=hdf5_file,
        datasets=datasets,
        processing_config=processing_config
    )
    
    try:
        # Load and process data
        comparator.load_data()
        comparator.process_data()

        # Automatic alignment (optional - controlled by processing_config)
        if processing_config.apply_auto_alignment:
            # Option 1: Auto-align using PD1 as reference (single signal, not composite)
            # Maintains group synchronization: PD2 stays with PD1, KH signals shift together
            calculated_shifts = comparator.auto_align_time_series(reference_label='PD1',
                                                                  correlation_window_time=0.001,
                                                                  use_raw_data=True,
                                                                  correlation_method='normalized',
                                                                  visualize=True,
                                                                  max_shift_time=0.002,  # 2ms search window
                                                                  sync_within_groups=True)

            # Option 2: Group-based alignment (use composite AMPM group signal)
            # calculated_shifts = comparator.auto_align_time_series(reference_group='AMPM',
            #                                                       correlation_window_time=0.001,
            #                                                       use_raw_data=True,
            #                                                       correlation_method='normalized',
            #                                                       visualize=True,
            #                                                       max_shift_time=0.0005,
            #                                                       sync_within_groups=True)

            # Option 3: Auto-align from original positions (ignoring manual shifts)
            # calculated_shifts = comparator.auto_align_time_series('PD1',
            #                                                      correlation_window_time=1.0,
            #                                                      use_original_positions=True)

            # Apply calculated shifts
            comparator.apply_calculated_shifts(calculated_shifts)
        
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