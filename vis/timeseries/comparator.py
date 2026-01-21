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
from .logging import ProcessingLog


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
                 processing_config: Optional[ProcessingConfig] = None,
                 verbose: bool = True):
        """
        Initialize the TimeSeriesComparator.

        Args:
            hdf5_path: Path to HDF5 file containing time series data
            datasets: List of dataset configurations
            processing_config: Optional processing configuration (uses defaults if None)
            verbose: If True, print detailed progress messages; if False, only print critical errors
        """
        self.hdf5_path = hdf5_path
        self.datasets = datasets
        self.processing_config = processing_config or ProcessingConfig()
        self.verbose = verbose

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

        # Initialize processor with verbose flag
        self.processor = TimeSeriesProcessor(self.processing_config, verbose=verbose)

        # Initialize processing log
        self.processing_log = ProcessingLog(hdf5_path)

        if self.verbose:
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

                    # DEBUG: Check if data is constant
                    data_min, data_max = np.min(data), np.max(data)
                    data_unique = len(np.unique(data))
                    print(f"  DEBUG: min={data_min:.6f}, max={data_max:.6f}, unique_values={data_unique}")
                    if data_min == data_max:
                        print(f"  WARNING: Data is constant! All values = {data_min}")

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
                        'group': dataset_config.group,
                        'dataset_name': data_path  # Full HDF5 path for global normalization
                    }

                except Exception as e:
                    print(f"  Error loading {label}: {e}")
                    continue

        print(f"\n✓ Loaded {len(self.raw_data)} datasets successfully")

        # Log loading information
        raw_data_shapes = {label: (len(data),) for label, data in self.raw_data.items()}
        self.processing_log.add_loading_info(self.datasets, self.sampling_rates, raw_data_shapes)

    def process_data(self, preserve_full_processed: bool = True) -> None:
        """Apply signal processing to all datasets"""
        print("\n" + "=" * 80)
        print("PROCESSING SIGNALS")
        print("=" * 80)

        for label, data in self.raw_data.items():
            print(f"\nProcessing {label}...")
            sampling_rate = self.sampling_rates.get(label, 1.0)
            group = self.alignment_info.get(label, {}).get('group', None)
            dataset_name = self.alignment_info.get(label, {}).get('dataset_name', None)
            processed = self.processor.process_signal(data, sampling_rate, label=label, group=group, dataset_name=dataset_name)
            self.processed_data[label] = processed

            if preserve_full_processed:
                self.full_processed_data[label] = processed.copy()

            # DEBUG: Check if processed data is constant
            proc_min, proc_max = np.min(processed), np.max(processed)
            proc_unique = len(np.unique(processed))
            print(f"  DEBUG PROCESSED: min={proc_min:.6f}, max={proc_max:.6f}, unique_values={proc_unique}")
            if proc_min == proc_max:
                print(f"  WARNING: Processed data is constant! All values = {proc_min}")

            print(f"  ✓ Processing complete ({len(processed)} samples)")

        print(f"\n✓ Processed {len(self.processed_data)} datasets")

        # Log processing information
        self.processing_log.add_processing_info(self.processing_config, self.processor.outlier_masks)

    def generate_report(self, output_dir: str = 'analysis_output') -> None:
        """Generate comprehensive analysis report"""
        print("\n" + "=" * 80)
        print("GENERATING ANALYSIS REPORT")
        print("=" * 80)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path.absolute()}")

        # Log statistics information
        if self.statistics:
            self.processing_log.add_statistics_info(
                self.statistics,
                self.correlations if hasattr(self, 'correlations') else None,
                self.silhouette_scores if hasattr(self, 'silhouette_scores') else None
            )

        # Save processing log
        print("\nSaving processing log...")
        self.processing_log.save(output_path)

        # Generate plots
        print("\nGenerating plots...")
        self.plot_processing_and_alignment_summary(save_path=output_path / 'processing_and_alignment_summary.png')
        self.plot_outlier_removal(save_path=output_path / 'outlier_removal.png')

        # Generate gradient/second derivative diagnostic plot if either method is being used
        methods = self.processor.config.outlier_method if isinstance(self.processor.config.outlier_method, list) else [self.processor.config.outlier_method]
        if any(m in ['gradient', 'second_derivative'] for m in methods) and len(self.processor.gradient_diagnostics) > 0:
            self.plot_gradient_diagnostics(save_path=output_path / 'gradient_diagnostics.png')

        self.plot_statistics_summary(save_path=output_path / 'statistics_summary.png')
        self.plot_correlation_matrix(save_path=output_path / 'correlation_matrix.png')
        self.plot_autocorrelation(save_path=output_path / 'autocorrelation.png')

        if len(self.processed_data) >= 2:
            self.plot_cross_correlation(save_path=output_path / 'cross_correlation.png')

            # Generate alignment diagnostic plot for feature-based and mutual info methods
            if hasattr(self, 'alignment_diagnostics') and self.alignment_diagnostics:
                # Check if any non-CCF methods were used
                has_advanced_methods = any(
                    diag.get('method') in ['feature', 'mutual_info']
                    for diag in self.alignment_diagnostics.values()
                )
                if has_advanced_methods:
                    self.plot_alignment_diagnostics(save_path=output_path / 'alignment_diagnostics.png')

            self.plot_scatterplot_matrix(save_path=output_path / 'scatterplot_matrix.png')
            self.plot_scatterplot_matrix_compact(save_path=output_path / 'scatterplot_matrix_compact.png')

        print("\n✓ Report generation complete")

    def get_batch_summary_row(self, trackid: str) -> 'pd.Series':
        """
        Generate a summary row for batch processing with all numerical metrics.

        Creates a pandas Series containing:
        - Trackid identifier
        - Basic statistics for each signal
        - Correlation coefficients for each signal pair
        - Silhouette scores (k=2, k=3) and optimal k for each pair

        Parameters:
        -----------
        trackid : str
            Identifier for this data file

        Returns:
        --------
        pd.Series
            Row containing all metrics with descriptive column names
        """
        import pandas as pd

        if not self.statistics:
            raise ValueError("Statistics not calculated. Call calculate_statistics() first.")

        row_data = {'trackid': trackid}

        # Add statistics for each signal
        for label, stats in self.statistics.items():
            for stat_name, stat_value in stats.items():
                col_name = f"{label}_{stat_name}"
                row_data[col_name] = stat_value

        # Add correlation coefficients if available
        if hasattr(self, 'correlations') and self.correlations:
            for pair_key, corr_data in self.correlations.items():
                # Clean up pair key for column name (replace ' vs ' with '_vs_')
                pair_clean = pair_key.replace(' vs ', '_vs_').replace(' ', '_')
                row_data[f"{pair_clean}_pearson"] = corr_data['pearson']
                row_data[f"{pair_clean}_spearman"] = corr_data['spearman']
                row_data[f"{pair_clean}_pearson_p"] = corr_data['pearson_p_corrected']
                row_data[f"{pair_clean}_spearman_p"] = corr_data['spearman_p_corrected']

        # Add silhouette scores if available
        if hasattr(self, 'silhouette_scores') and self.silhouette_scores:
            for pair_key, silhouette_data in self.silhouette_scores.items():
                # Clean up pair key for column name
                pair_clean = pair_key.replace(' vs ', '_vs_').replace(' ', '_')
                row_data[f"{pair_clean}_silhouette_k2"] = silhouette_data['silhouette_k2']
                row_data[f"{pair_clean}_silhouette_k3"] = silhouette_data['silhouette_k3']
                row_data[f"{pair_clean}_optimal_k"] = silhouette_data['optimal_k']

        return pd.Series(row_data)

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
