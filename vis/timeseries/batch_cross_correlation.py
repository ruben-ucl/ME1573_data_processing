"""
Multi-file cross-correlation analysis (statistics-only mode).

This module provides batch statistical analysis across multiple HDF5 files
by concatenating data from multiple tracks and performing cross-correlation
analysis. Unlike single-track time series analysis, this skips time-series
specific operations (alignment, autocorrelation) since the data is not
continuous across files.

Analyses performed:
- Basic statistics (mean, std, skewness, kurtosis, etc.)
- Cross-correlation matrix (Pearson, Spearman)
- Scatter plot matrix
- Silhouette clustering analysis
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
from .statistics import StatisticsMixin
from .plotting import PlottingMixin


class BatchCrossCorrelator(StatisticsMixin, PlottingMixin):
    """
    Batch cross-correlation analysis across multiple HDF5 files.

    This class loads data from multiple HDF5 files, concatenates the signals,
    and performs statistical cross-correlation analysis. It reuses the statistical
    analysis and plotting functionality from the TimeSeriesComparator mixins but
    skips time-series specific operations since the concatenated data is not
    continuous across file boundaries.

    Attributes:
    -----------
    hdf5_files : List[Path]
        List of HDF5 file paths to process
    datasets : List[DatasetConfig]
        Dataset configurations specifying which signals to load
    processing_config : ProcessingConfig
        Processing configuration (normalization, outlier removal, etc.)
    processed_data : Dict[str, np.ndarray]
        Concatenated and processed data for each signal label
    file_info : List[Dict]
        Metadata for each file (trackid, path, sample counts)
    file_boundaries : Dict[str, List[int]]
        Track file boundaries within concatenated arrays for diagnostics
    statistics : Dict
        Statistical measures for each signal
    correlations : Dict
        Cross-correlation coefficients between signal pairs
    silhouette_scores : Dict
        Clustering quality metrics for signal pairs
    """

    def __init__(self, hdf5_files: List[Path], datasets: List[DatasetConfig],
                 processing_config: Optional[ProcessingConfig] = None,
                 verbose: bool = True):
        """
        Initialize the BatchCrossCorrelator.

        Args:
            hdf5_files: List of HDF5 file paths to analyze
            datasets: List of dataset configurations (signal names, groups, etc.)
            processing_config: Optional processing configuration (uses defaults if None)
            verbose: If True, print detailed progress messages
        """
        self.hdf5_files = hdf5_files
        self.datasets = datasets
        self.processing_config = processing_config or ProcessingConfig()
        self.verbose = verbose

        # Initialize data storage (required by mixins)
        self.processed_data: Dict[str, np.ndarray] = {}
        self.statistics: Dict = {}
        self.correlations: Dict = {}
        self.silhouette_scores: Dict = {}

        # Dummy time_vectors (not used in batch mode, but required by StatisticsMixin)
        self.time_vectors: Dict[str, np.ndarray] = {}

        # File tracking
        self.file_info: List[Dict] = []
        self.file_boundaries: Dict[str, List[int]] = {}

        # Initialize processor
        self.processor = TimeSeriesProcessor(self.processing_config, verbose=verbose)

        # Color palette for plotting (from PlottingMixin)
        self.COLOR_PRIMARY = '#0173B2'
        self.COLOR_SECONDARY = '#DE8F05'
        self.COLOR_TERTIARY = '#029E73'
        self.COLOR_SIGNIFICANCE = '#CC78BC'

        if self.verbose:
            print(f"Initialized BatchCrossCorrelator with {len(hdf5_files)} files")
            print(f"Datasets to analyze: {[d.label for d in datasets]}")

    def load_and_concatenate_data(self) -> None:
        """
        Load data from all HDF5 files, synchronize per-file, then concatenate.

        For each file, per signal:
        1. Load raw data + time vectors from HDF5
        2. Synchronize signals to common time grid (interpolate to consistent sampling)
        3. Crop to overlapping time window
        4. Store synchronized arrays for concatenation

        Then concatenate synchronized arrays across all files and apply global processing.

        This ensures:
        - Points correspond correctly across signals (critical for correlation)
        - Sampling rate differences handled via interpolation
        - Time lags corrected before concatenation
        """
        print("\n" + "=" * 80)
        print("LOADING AND SYNCHRONIZING DATA FROM MULTIPLE FILES")
        print("=" * 80)
        print(f"\nProcessing {len(self.hdf5_files)} files...")

        # Initialize storage for synchronized data from each file
        synchronized_signal_arrays = {dataset.label: [] for dataset in self.datasets}
        sampling_rates = {}

        # Process each file: load, synchronize, crop
        for file_idx, hdf5_path in enumerate(self.hdf5_files):
            trackid = hdf5_path.stem  # Extract trackid from filename

            if self.verbose:
                print(f"\n[{file_idx + 1}/{len(self.hdf5_files)}] Processing {trackid}...")

            file_samples = {}

            # Storage for this file's data and time vectors
            file_raw_data = {}
            file_time_vectors = {}

            # Try to open file - skip if it doesn't exist or can't be opened
            try:
                f = h5py.File(hdf5_path, 'r')
            except (FileNotFoundError, OSError) as e:
                print(f"  ERROR: Cannot open file ({e}) - skipping")
                continue

            try:
                # Use context manager for the successfully opened file
                with f:
                    for dataset_config in self.datasets:
                        label = dataset_config.label

                        try:
                            # Load raw data
                            data_path = f"{dataset_config.group}/{dataset_config.name}"
                            data = np.array(f[data_path])

                            # Load or generate time vector
                            time_vector = None
                            if dataset_config.time_group and dataset_config.time_name:
                                try:
                                    time_path = f"{dataset_config.time_group}/{dataset_config.time_name}"
                                    time_vector = np.array(f[time_path])
                                    # Convert time units
                                    conversion_factors = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9}
                                    factor = conversion_factors.get(dataset_config.time_units.lower(), 1.0)
                                    time_vector = time_vector * factor
                                except KeyError:
                                    time_vector = None

                            # Generate time vector if not loaded
                            if time_vector is None:
                                if dataset_config.sampling_rate:
                                    sampling_rate = dataset_config.sampling_rate
                                else:
                                    sampling_rate = 1.0
                                time_vector = np.arange(len(data)) / sampling_rate

                            # Ensure data and time vector match length
                            if len(time_vector) != len(data):
                                min_len = min(len(time_vector), len(data))
                                time_vector = time_vector[:min_len]
                                data = data[:min_len]

                            # Crop PD signals (before synchronization)
                            if dataset_config.group == 'AMPM':
                                data = data[510:-510]
                                time_vector = time_vector[510:-510]

                            # Store for synchronization
                            file_raw_data[label] = data
                            file_time_vectors[label] = time_vector

                            # Calculate sampling rate for global tracking
                            dt = np.mean(np.diff(time_vector)) if len(time_vector) > 1 else 1.0
                            sampling_rate = 1.0 / dt if dt > 0 else 1.0
                            sampling_rates[label] = sampling_rate

                            if self.verbose:
                                print(f"  {label}: {len(data)} samples, SR: {sampling_rate:.2f} Hz")

                        except Exception as e:
                            print(f"  ERROR loading {label}: {e}")
                            continue

                # Synchronize signals within this file
                if len(file_raw_data) > 0:
                    synced_data = self._synchronize_file_signals(
                        file_raw_data, file_time_vectors, trackid
                    )

                    # Store synchronized data for concatenation
                    for label, data in synced_data.items():
                        synchronized_signal_arrays[label].append(data)
                        file_samples[label] = len(data)

                    # Record file metadata
                    self.file_info.append({
                        'trackid': trackid,
                        'path': hdf5_path,
                        'samples': file_samples
                    })
                else:
                    print(f"  WARNING: No data loaded from {trackid}")

            except Exception as e:
                print(f"  ERROR processing file: {e}")
                continue

        # Concatenate synchronized data and apply global processing
        print(f"\n{'=' * 80}")
        print("CONCATENATING SYNCHRONIZED DATA AND APPLYING GLOBAL PROCESSING")
        print("=" * 80)

        for dataset_config in self.datasets:
            label = dataset_config.label

            if len(synchronized_signal_arrays[label]) == 0:
                print(f"WARNING: No synchronized data for {label}")
                continue

            # Concatenate synchronized arrays
            concatenated_data = np.concatenate(synchronized_signal_arrays[label])

            # Calculate file boundaries (before processing in case length changes)
            boundaries = [0]
            cumsum = 0
            for arr in synchronized_signal_arrays[label]:
                cumsum += len(arr)
                boundaries.append(cumsum)
            self.file_boundaries[label] = boundaries

            print(f"\n{label}:")
            print(f"  Synchronized samples concatenated: {len(concatenated_data):,}")
            print(f"  Files contributing: {len(synchronized_signal_arrays[label])}")

            # Now apply processing to concatenated data (with global normalization)
            sampling_rate = sampling_rates.get(label, 1.0)

            # Temporarily disable global normalization in processor
            # (we'll do it manually after to use true global stats)
            original_use_global = self.processing_config.use_global_normalization
            self.processing_config.use_global_normalization = False

            processed = self.processor.process_signal(
                concatenated_data,
                sampling_rate,
                label=label,
                group=dataset_config.group,
                dataset_name=None  # No dataset_name to avoid global norm lookup
            )

            # Restore original setting
            self.processing_config.use_global_normalization = original_use_global

            self.processed_data[label] = processed

            # Create dummy time vector (indices) for StatisticsMixin compatibility
            self.time_vectors[label] = np.arange(len(processed))

            print(f"  Processed samples: {len(processed):,}")
            print(f"  Samples per file: min={min(len(a) for a in synchronized_signal_arrays[label]):,}, "
                  f"max={max(len(a) for a in synchronized_signal_arrays[label]):,}, "
                  f"mean={np.mean([len(a) for a in synchronized_signal_arrays[label]]):.0f}")

        print(f"\n✓ Synchronization, concatenation and processing complete: {len(self.processed_data)} signals ready for analysis")

    def _synchronize_file_signals(self, raw_data_dict: Dict[str, np.ndarray],
                                   time_vectors_dict: Dict[str, np.ndarray],
                                   trackid: str) -> Dict[str, np.ndarray]:
        """
        Synchronize signals within a single file by interpolating to common time grid
        and cropping to overlapping region.

        This implements the same logic as alignment.py's _synchronize_time_series()
        and crop_to_shortest_signal() but operates on a single file's data.

        Args:
            raw_data_dict: Dictionary of signal label -> raw data array
            time_vectors_dict: Dictionary of signal label -> time vector
            trackid: Track ID for verbose output

        Returns:
            Dictionary of signal label -> synchronized data array
        """
        if len(raw_data_dict) == 0:
            return {}

        if self.verbose:
            print(f"  Synchronizing {len(raw_data_dict)} signals for {trackid}...")

        # Step 1: Find overlapping time range across all signals
        earliest_start = float('-inf')
        latest_end = float('inf')

        for label in raw_data_dict.keys():
            time_vec = time_vectors_dict[label]
            earliest_start = max(earliest_start, time_vec[0])
            latest_end = min(latest_end, time_vec[-1])

        if earliest_start >= latest_end:
            if self.verbose:
                print(f"  WARNING: No overlapping time range in {trackid}, using minimum length truncation")
            # Fallback: truncate to minimum length
            min_len = min(len(data) for data in raw_data_dict.values())
            return {label: data[:min_len] for label, data in raw_data_dict.items()}

        # Step 2: Determine common time grid (use finest sampling rate)
        dt_values = []
        for label in raw_data_dict.keys():
            time_vec = time_vectors_dict[label]
            if len(time_vec) > 1:
                dt = np.mean(np.diff(time_vec))
                dt_values.append(dt)

        if len(dt_values) > 0:
            dt_common = min(dt_values)  # Use finest resolution
        else:
            dt_common = 1.0

        # Create common time grid
        t_common = np.arange(earliest_start, latest_end, dt_common)

        if self.verbose:
            print(f"    Time range: [{earliest_start:.6f}s, {latest_end:.6f}s], dt={dt_common:.6f}s, {len(t_common)} samples")

        # Step 3: Interpolate all signals to common time grid
        synchronized_data = {}
        for label, data in raw_data_dict.items():
            time_vec = time_vectors_dict[label]

            # Interpolate to common time grid
            data_interp = np.interp(t_common, time_vec, data)
            synchronized_data[label] = data_interp

            if self.verbose:
                print(f"    {label}: {len(data)} → {len(data_interp)} samples")

        return synchronized_data

    def _synchronize_time_series(self, data1=None, time1=None, data2=None, time2=None, labels=None):
        """
        Compatibility method for both StatisticsMixin and PlottingMixin.

        In batch mode, data is already concatenated (not time-synchronized).
        This method handles two calling conventions:
        1. From StatisticsMixin: _synchronize_time_series(data1, time1, data2, time2)
        2. From PlottingMixin: _synchronize_time_series(labels=[...])

        Args:
            data1: First data array (for StatisticsMixin)
            time1: First time vector (for StatisticsMixin)
            data2: Second data array (for StatisticsMixin)
            time2: Second time vector (for StatisticsMixin)
            labels: List of signal labels (for PlottingMixin)

        Returns:
            - If called from StatisticsMixin: Tuple of (data1, data2)
            - If called from PlottingMixin: Tuple of (synced_data_dict, synced_time_dict)
        """
        # Called from StatisticsMixin (correlation calculation)
        if data1 is not None and data2 is not None:
            # In batch mode, data is already concatenated and aligned
            # Just return the minimum overlapping length
            min_len = min(len(data1), len(data2))
            return data1[:min_len], data2[:min_len]

        # Called from PlottingMixin (plotting operations)
        if labels is not None:
            synced_data = {label: self.processed_data[label] for label in labels if label in self.processed_data}
            synced_time = {label: self.time_vectors[label] for label in labels if label in self.time_vectors}
            return synced_data, synced_time

        # Fallback - should not reach here
        raise ValueError("Invalid arguments to _synchronize_time_series")

    def generate_report(self, output_dir: str = 'batch_analysis_output') -> None:
        """
        Generate comprehensive batch analysis report.

        Creates:
        - Statistical plots (summary, correlation matrix, scatter plots)
        - Text report with file list and metrics
        - JSON log with machine-readable data

        Args:
            output_dir: Directory to save output files
        """
        print("\n" + "=" * 80)
        print("GENERATING BATCH ANALYSIS REPORT")
        print("=" * 80)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path.absolute()}")

        # Generate plots
        print("\nGenerating plots...")
        self.plot_statistics_summary(save_path=output_path / 'statistics_summary.png')
        self.plot_correlation_matrix(save_path=output_path / 'correlation_matrix.png')

        if len(self.processed_data) >= 2:
            self.plot_scatterplot_matrix(save_path=output_path / 'scatterplot_matrix.png')
            self.plot_scatterplot_matrix_compact(save_path=output_path / 'scatterplot_matrix_compact.png')

        # Generate text report
        print("\nGenerating text report...")
        self._write_text_report(output_path / 'batch_analysis_report.txt')

        # Generate JSON log
        print("\nGenerating JSON log...")
        self._write_json_log(output_path / 'batch_analysis_log.json')

        print("\n✓ Report generation complete")

    def _write_text_report(self, filepath: Path) -> None:
        """Write human-readable text report."""
        import json

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BATCH CROSS-CORRELATION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # File list
            f.write(f"FILES ANALYZED: {len(self.file_info)}\n")
            f.write("-" * 80 + "\n")
            for i, file_info in enumerate(self.file_info, 1):
                f.write(f"{i}. {file_info['trackid']}\n")
                for label, count in file_info['samples'].items():
                    f.write(f"   {label}: {count:,} samples\n")
            f.write("\n")

            # Total samples
            f.write("CONCATENATED DATA SUMMARY\n")
            f.write("-" * 80 + "\n")
            for label, data in self.processed_data.items():
                f.write(f"{label}: {len(data):,} total samples "
                       f"({len(self.file_boundaries[label])-1} files)\n")
            f.write("\n")

            # Statistics
            f.write("SIGNAL STATISTICS\n")
            f.write("-" * 80 + "\n")
            for label, stats in self.statistics.items():
                f.write(f"\n{label}:\n")
                for stat_name, value in stats.items():
                    f.write(f"  {stat_name}: {value:.6f}\n")
            f.write("\n")

            # Correlations
            if self.correlations:
                f.write("CROSS-CORRELATIONS\n")
                f.write("-" * 80 + "\n")
                for pair_key, corr_data in self.correlations.items():
                    f.write(f"\n{pair_key}:\n")
                    f.write(f"  Pearson r:  {corr_data['pearson']:.6f} "
                           f"(p={corr_data['pearson_p_corrected']:.6e})\n")
                    f.write(f"  Spearman ρ: {corr_data['spearman']:.6f} "
                           f"(p={corr_data['spearman_p_corrected']:.6e})\n")
                f.write("\n")

            # Silhouette scores
            if hasattr(self, 'silhouette_scores') and self.silhouette_scores:
                f.write("CLUSTERING QUALITY (Silhouette Scores)\n")
                f.write("-" * 80 + "\n")
                for pair_key, sil_data in self.silhouette_scores.items():
                    f.write(f"\n{pair_key}:\n")
                    f.write(f"  k=2: {sil_data['silhouette_k2']:.6f}\n")
                    f.write(f"  k=3: {sil_data['silhouette_k3']:.6f}\n")
                    f.write(f"  Optimal k: {sil_data['optimal_k']}\n")

        print(f"  Saved: {filepath}")

    def _write_json_log(self, filepath: Path) -> None:
        """Write machine-readable JSON log."""
        import json

        log_data = {
            'files': [
                {
                    'trackid': info['trackid'],
                    'path': str(info['path']),
                    'samples': info['samples']
                }
                for info in self.file_info
            ],
            'concatenated_data': {
                label: {
                    'total_samples': len(data),
                    'num_files': len(self.file_boundaries[label]) - 1,
                    'boundaries': self.file_boundaries[label]
                }
                for label, data in self.processed_data.items()
            },
            'statistics': self.statistics,
            'correlations': self.correlations,
            'silhouette_scores': self.silhouette_scores if hasattr(self, 'silhouette_scores') else {},
            'processing_config': {
                'apply_normalization': self.processing_config.apply_normalization,
                'normalization_method': self.processing_config.normalization_method,
                'remove_outliers': self.processing_config.remove_outliers,
                'outlier_method': self.processing_config.outlier_method,
                'outlier_threshold': self.processing_config.outlier_threshold
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2)

        print(f"  Saved: {filepath}")

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Create a summary DataFrame with all metrics.

        Returns a single-row DataFrame containing all statistics, correlations,
        and silhouette scores for integration with batch analysis pipelines.

        Returns:
            pd.DataFrame: Single row with all metrics
        """
        row_data = {}

        # Add statistics for each signal
        for label, stats in self.statistics.items():
            for stat_name, stat_value in stats.items():
                col_name = f"{label}_{stat_name}"
                row_data[col_name] = stat_value

        # Add correlations
        if self.correlations:
            for pair_key, corr_data in self.correlations.items():
                pair_clean = pair_key.replace(' vs ', '_vs_').replace(' ', '_')
                row_data[f"{pair_clean}_pearson"] = corr_data['pearson']
                row_data[f"{pair_clean}_spearman"] = corr_data['spearman']
                row_data[f"{pair_clean}_pearson_p"] = corr_data['pearson_p_corrected']
                row_data[f"{pair_clean}_spearman_p"] = corr_data['spearman_p_corrected']

        # Add silhouette scores
        if hasattr(self, 'silhouette_scores') and self.silhouette_scores:
            for pair_key, silhouette_data in self.silhouette_scores.items():
                pair_clean = pair_key.replace(' vs ', '_vs_').replace(' ', '_')
                row_data[f"{pair_clean}_silhouette_k2"] = silhouette_data['silhouette_k2']
                row_data[f"{pair_clean}_silhouette_k3"] = silhouette_data['silhouette_k3']
                row_data[f"{pair_clean}_optimal_k"] = silhouette_data['optimal_k']

        return pd.DataFrame([row_data])
