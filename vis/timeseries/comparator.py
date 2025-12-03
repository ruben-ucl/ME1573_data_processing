"""
Main time series comparator class.

This module contains the TimeSeriesComparator class which orchestrates
all time series analysis operations by combining functionality from
the specialized mixin classes.
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for tools import
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from tools import get_paths

from .config import DatasetConfig, ProcessingConfig
from .processor import TimeSeriesProcessor
from .alignment import AlignmentMixin
from .statistics import StatisticsMixin
from .plotting import PlottingMixin


class TimeSeriesComparator(AlignmentMixin, StatisticsMixin, PlottingMixin):
    """
    Comprehensive time series analysis and comparison tool.

    This class combines functionality from multiple specialized mixins to provide
    a complete suite of time series analysis operations including:
    - Data loading from HDF5 files
    - Signal processing and filtering
    - Time series alignment
    - Statistical analysis
    - Visualization and reporting
    """

    def __init__(self, hdf5_path: str, datasets: List[DatasetConfig],
                 processing_config: Optional[ProcessingConfig] = None):
        """
        Initialize the TimeSeriesComparator.

        Args:
            hdf5_path: Path to HDF5 file containing time series data
            datasets: List of dataset configurations
            processing_config: Optional processing configuration (uses defaults if None)
        """
        self.hdf5_path = hdf5_path
        self.datasets = datasets
        self.processing_config = processing_config or ProcessingConfig()

        # Initialize data storage
        self.raw_data: Dict[str, np.ndarray] = {}
        self.processed_data: Dict[str, np.ndarray] = {}
        self.full_processed_data: Dict[str, np.ndarray] = {}
        self.original_processed_data: Dict[str, np.ndarray] = {}  # For cropping/restoration
        self.time_vectors: Dict[str, np.ndarray] = {}
        self.original_time_vectors: Dict[str, np.ndarray] = {}
        self.sampling_rates: Dict[str, float] = {}

        # Initialize analysis storage
        self.statistics: Dict = {}
        self.correlations: Dict = {}
        self.alignment_info: Dict[str, Dict] = {}

        # Initialize processor
        self.processor = TimeSeriesProcessor(self.processing_config)

        print(f"Initialized TimeSeriesComparator with {len(datasets)} datasets")
        print(f"HDF5 file: {hdf5_path}")

    def _convert_time_units(self, time_vector: np.ndarray, units: str) -> np.ndarray:
        """Convert time vector to seconds based on specified units"""
        conversion_factors = {
            's': 1.0,
            'ms': 1e-3,
            'us': 1e-6,
            'ns': 1e-9
        }
        factor = conversion_factors.get(units.lower(), 1.0)
        return time_vector * factor

    def _calculate_sampling_rate(self, time_vector: np.ndarray) -> float:
        """Calculate sampling rate from time vector"""
        if len(time_vector) < 2:
            return 1.0
        dt = np.mean(np.diff(time_vector))
        return 1.0 / dt if dt > 0 else 1.0

    def load_data(self) -> None:
        """Load data from HDF5 file"""
        print("\n" + "=" * 80)
        print("LOADING DATA FROM HDF5")
        print("=" * 80)

        with h5py.File(self.hdf5_path, 'r') as f:
            for dataset_config in self.datasets:
                label = dataset_config.label
                print(f"\nLoading {label}...")

                # Load data
                try:
                    data_path = f"{dataset_config.group}/{dataset_config.name}"
                    data = np.array(f[data_path])
                    self.raw_data[label] = data
                    print(f"  Data loaded: {len(data)} samples")

                    # Load or create time vector
                    if dataset_config.time_group and dataset_config.time_name:
                        time_path = f"{dataset_config.time_group}/{dataset_config.time_name}"
                        time_vector = np.array(f[time_path])
                        time_vector = self._convert_time_units(time_vector, dataset_config.time_units)
                        print(f"  Time vector loaded: {len(time_vector)} samples")
                    elif dataset_config.sampling_rate:
                        time_vector = np.arange(len(data)) / dataset_config.sampling_rate
                        print(f"  Time vector generated from sampling rate: {dataset_config.sampling_rate} Hz")
                    else:
                        time_vector = np.arange(len(data))
                        print(f"  Time vector generated (indices)")

                    # Apply manual time shift
                    if dataset_config.time_shift != 0.0:
                        time_vector = time_vector + dataset_config.time_shift
                        print(f"  Applied manual time shift: {dataset_config.time_shift*1000:.3f}ms")

                    self.time_vectors[label] = time_vector
                    self.original_time_vectors[label] = time_vector.copy()

                    # Calculate sampling rate
                    sampling_rate = self._calculate_sampling_rate(time_vector)
                    self.sampling_rates[label] = sampling_rate
                    print(f"  Calculated sampling rate: {sampling_rate:.2f} Hz")

                    # Initialize alignment info
                    self.alignment_info[label] = {
                        'time_shift': dataset_config.time_shift,
                        'shift_type': 'manual' if dataset_config.time_shift != 0.0 else 'none',
                        'group': dataset_config.group
                    }

                except Exception as e:
                    print(f"  Error loading {label}: {e}")
                    continue

        print(f"\n✓ Loaded {len(self.raw_data)} datasets successfully")

    def process_data(self, preserve_full_processed: bool = True) -> None:
        """Apply signal processing to all datasets"""
        print("\n" + "=" * 80)
        print("PROCESSING SIGNALS")
        print("=" * 80)

        for label, data in self.raw_data.items():
            print(f"\nProcessing {label}...")
            sampling_rate = self.sampling_rates.get(label, 1.0)
            processed = self.processor.process_signal(data, sampling_rate)
            self.processed_data[label] = processed

            if preserve_full_processed:
                self.full_processed_data[label] = processed.copy()

            print(f"  ✓ Processing complete ({len(processed)} samples)")

        print(f"\n✓ Processed {len(self.processed_data)} datasets")

    def generate_report(self, output_dir: str = 'analysis_output') -> None:
        """Generate comprehensive analysis report"""
        print("\n" + "=" * 80)
        print("GENERATING ANALYSIS REPORT")
        print("=" * 80)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path.absolute()}")

        # Generate plots
        print("\nGenerating plots...")
        self.plot_processing_and_alignment_summary(save_path=output_path / 'processing_and_alignment_summary.png')
        self.plot_statistics_summary(save_path=output_path / 'statistics_summary.png')
        self.plot_correlation_matrix(save_path=output_path / 'correlation_matrix.png')
        self.plot_autocorrelation(save_path=output_path / 'autocorrelation.png')

        if len(self.processed_data) >= 2:
            self.plot_cross_correlation(save_path=output_path / 'cross_correlation.png')
            self.plot_scatterplot_matrix(save_path=output_path / 'scatterplot_matrix.png')
            self.plot_scatterplot_matrix_compact(save_path=output_path / 'scatterplot_matrix_compact.png')

        print("\n✓ Report generation complete")

    def get_data_summary(self) -> None:
        """Print summary of loaded data"""
        print("\n" + "=" * 80)
        print("DATA SUMMARY")
        print("=" * 80)

        for label in self.raw_data.keys():
            print(f"\n{label}:")
            print(f"  Samples: {len(self.raw_data[label])}")
            print(f"  Duration: {self.time_vectors[label][-1] - self.time_vectors[label][0]:.6f}s")
            print(f"  Sampling rate: {self.sampling_rates[label]:.2f} Hz")
            if label in self.statistics:
                stats = self.statistics[label]
                print(f"  Mean: {stats['mean']:.4f}")
                print(f"  Std: {stats['std']:.4f}")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
