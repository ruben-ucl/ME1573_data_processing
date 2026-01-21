"""
Processing log for time series analysis.

This module provides comprehensive logging of all data processing operations
including data loading, signal processing, alignment, cropping, and statistics.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np


class ProcessingLog:
    """
    Comprehensive logging of time series processing pipeline.

    Captures all parameters and operations that affect the final output:
    - Data loading configurations
    - Signal processing parameters
    - Alignment operations and shifts
    - Cropping operations
    - Statistical analysis settings
    """

    def __init__(self, hdf5_path: str):
        """
        Initialize processing log.

        Args:
            hdf5_path: Path to HDF5 file being processed
        """
        self.hdf5_path = str(hdf5_path)
        self.timestamp = datetime.now().isoformat()

        # Processing sections
        self.loading_info: Dict[str, Any] = {}
        self.processing_info: Dict[str, Any] = {}
        self.alignment_info: Dict[str, Any] = {}
        self.cropping_info: Dict[str, Any] = {}
        self.statistics_info: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def add_loading_info(self, datasets: List[Any],
                        sampling_rates: Dict[str, float],
                        raw_data_shapes: Dict[str, tuple]) -> None:
        """
        Record data loading information.

        Args:
            datasets: List of DatasetConfig objects
            sampling_rates: Dictionary of calculated sampling rates
            raw_data_shapes: Dictionary of raw data array shapes
        """
        self.loading_info = {
            'hdf5_file': self.hdf5_path,
            'datasets': [],
            'sampling_rates': {},
            'data_shapes': {}
        }

        for ds in datasets:
            self.loading_info['datasets'].append({
                'label': ds.label,
                'group': ds.group,
                'name': ds.name,
                'time_group': ds.time_group,
                'time_name': ds.time_name,
                'time_units': ds.time_units,
                'manual_time_shift': ds.time_shift,
                'linestyle': ds.linestyle,
                'color': ds.color if hasattr(ds, 'color') else None
            })

        for label, rate in sampling_rates.items():
            self.loading_info['sampling_rates'][label] = float(rate)

        for label, shape in raw_data_shapes.items():
            self.loading_info['data_shapes'][label] = shape

    def add_processing_info(self, config: Any,
                           outlier_masks: Optional[Dict[str, np.ndarray]] = None) -> None:
        """
        Record signal processing parameters.

        Args:
            config: ProcessingConfig object
            outlier_masks: Optional dictionary of outlier masks per signal
        """
        self.processing_info = {
            'reference_sampling_rate': config.reference_sampling_rate,

            # NaN handling
            'handle_nans': config.handle_nans,

            # Outlier removal
            'remove_outliers': config.remove_outliers,
            'outlier_method': config.outlier_method if config.remove_outliers else None,
            'outlier_threshold': config.outlier_threshold if config.remove_outliers else None,
            'outlier_window': config.outlier_window if config.remove_outliers else None,

            # Detrending
            'apply_detrend': config.apply_detrend,
            'detrend_method': config.detrend_method if config.apply_detrend else None,

            # Savitzky-Golay filter
            'apply_savgol': config.apply_savgol,
            'savgol_window': config.savgol_window if config.apply_savgol else None,
            'savgol_polyorder': config.savgol_polyorder if config.apply_savgol else None,

            # Lowpass filter
            'apply_lowpass': config.apply_lowpass,
            'lowpass_cutoff': config.lowpass_cutoff if config.apply_lowpass else None,
            'lowpass_order': config.lowpass_order if config.apply_lowpass else None,

            # Highpass filter
            'apply_highpass': config.apply_highpass,
            'highpass_cutoff': config.highpass_cutoff if config.apply_highpass else None,
            'highpass_order': config.highpass_order if config.apply_highpass else None,

            # Bandpass filter
            'apply_bandpass': config.apply_bandpass,
            'bandpass_low': config.bandpass_low if config.apply_bandpass else None,
            'bandpass_high': config.bandpass_high if config.apply_bandpass else None,
            'bandpass_order': config.bandpass_order if config.apply_bandpass else None,

            # Smoothing
            'apply_smoothing': config.apply_smoothing,
            'smoothing_method': config.smoothing_method if config.apply_smoothing else None,
            'smoothing_window': config.smoothing_window if config.apply_smoothing else None,

            # Resampling
            'apply_resampling': config.apply_resampling,
            'target_samples': config.target_samples if config.apply_resampling else None,

            # Normalization
            'apply_normalization': config.apply_normalization,
            'normalization_method': config.normalization_method if config.apply_normalization else None,
        }

        # Add outlier statistics if available
        if outlier_masks:
            self.processing_info['outlier_statistics'] = {}
            for label, mask in outlier_masks.items():
                n_outliers = int(np.sum(mask))
                total_samples = len(mask)
                self.processing_info['outlier_statistics'][label] = {
                    'n_outliers': n_outliers,
                    'total_samples': total_samples,
                    'outlier_percentage': float(100 * n_outliers / total_samples) if total_samples > 0 else 0.0
                }

    def add_alignment_info(self, alignment_info: Dict[str, Dict],
                          apply_auto_alignment: bool,
                          auto_alignment_params: Optional[Dict[str, Any]] = None,
                          calculated_shifts: Optional[Dict[str, float]] = None,
                          alignment_diagnostics: Optional[Dict[str, Dict]] = None) -> None:
        """
        Record time series alignment information.

        Args:
            alignment_info: Dictionary of alignment info per signal
            apply_auto_alignment: Whether auto-alignment was applied
            auto_alignment_params: Parameters used for auto-alignment
            calculated_shifts: Dictionary of calculated shifts
            alignment_diagnostics: Cross-correlation diagnostics from alignment
        """
        self.alignment_info = {
            'auto_alignment_enabled': apply_auto_alignment,
            'shifts_per_signal': {}
        }

        # Record auto-alignment parameters
        if apply_auto_alignment and auto_alignment_params:
            self.alignment_info['auto_alignment_params'] = {
                'reference_label': auto_alignment_params.get('reference_label'),
                'reference_group': auto_alignment_params.get('reference_group'),
                'correlation_window_time': auto_alignment_params.get('correlation_window_time'),
                'max_shift_time': auto_alignment_params.get('max_shift_time'),
                'correlation_method': auto_alignment_params.get('correlation_method'),
                'normalize_signals': auto_alignment_params.get('normalize_signals'),
                'sync_within_groups': auto_alignment_params.get('sync_within_groups'),
                'use_raw_data': auto_alignment_params.get('use_raw_data')
            }

        # Record calculated shifts
        if calculated_shifts:
            self.alignment_info['calculated_shifts'] = {
                label: float(shift) for label, shift in calculated_shifts.items()
            }

        # Record cross-correlation diagnostics
        if alignment_diagnostics:
            self.alignment_info['cross_correlation_diagnostics'] = {}
            for pair_key, diag in alignment_diagnostics.items():
                self.alignment_info['cross_correlation_diagnostics'][pair_key] = {
                    'label1': diag['label1'],
                    'label2': diag['label2'],
                    'peak_lag_time': float(diag['peak_lag_time']),
                    'peak_lag_samples': int(diag['peak_lag_samples']),
                    'max_correlation': float(diag['max_correlation']),
                    'data_length': int(diag['data_length'])
                    # Note: lags_samples, lags_time, and correlation arrays not stored to keep log size reasonable
                }

        # Record final alignment state per signal
        for label, info in alignment_info.items():
            self.alignment_info['shifts_per_signal'][label] = {
                'total_time_shift': float(info['time_shift']),
                'shift_type': info['shift_type'],
                'manual_shift': float(info.get('manual_shift', 0.0)),
                'auto_shift': float(info.get('auto_shift', 0.0)),
                'group': info.get('group')
            }

    def add_cropping_info(self, cropping_info: Dict[str, Dict]) -> None:
        """
        Record signal cropping information.

        Args:
            cropping_info: Dictionary returned by crop_to_shortest_signal()
        """
        if not cropping_info:
            self.cropping_info = {'cropping_applied': False}
            return

        self.cropping_info = {
            'cropping_applied': True,
            'per_signal': {}
        }

        # Find common range
        if cropping_info:
            first_signal = next(iter(cropping_info.values()))
            self.cropping_info['common_time_range'] = {
                'start': float(first_signal['cropped_start']),
                'end': float(first_signal['cropped_end']),
                'duration': float(first_signal['cropped_duration'])
            }

        # Record per-signal cropping details
        for label, info in cropping_info.items():
            self.cropping_info['per_signal'][label] = {
                'original_samples': int(info['original_samples']),
                'cropped_samples': int(info['cropped_samples']),
                'samples_removed': int(info['original_samples'] - info['cropped_samples']),
                'removal_percentage': float(100 * (info['original_samples'] - info['cropped_samples']) / info['original_samples']),
                'original_duration': float(info['original_duration']),
                'cropped_duration': float(info['cropped_duration']),
                'samples_removed_start': int(info['samples_removed_start']),
                'samples_removed_end': int(info['samples_removed_end'])
            }

    def add_statistics_info(self, statistics: Dict[str, Dict],
                           correlations: Optional[Dict[str, Dict]] = None,
                           silhouette_scores: Optional[Dict[str, Dict]] = None) -> None:
        """
        Record statistical analysis results.

        Args:
            statistics: Dictionary of statistics per signal
            correlations: Optional dictionary of correlation results
            silhouette_scores: Optional dictionary of silhouette scores
        """
        self.statistics_info = {
            'per_signal_statistics': {},
            'correlations_calculated': correlations is not None,
            'silhouette_scores_calculated': silhouette_scores is not None
        }

        # Record basic statistics per signal
        for label, stats in statistics.items():
            self.statistics_info['per_signal_statistics'][label] = {
                key: float(value) for key, value in stats.items()
            }

        # Record correlation information (summary only to avoid bloat)
        if correlations:
            self.statistics_info['correlations'] = {}
            for pair_key, corr_data in correlations.items():
                self.statistics_info['correlations'][pair_key] = {
                    'pearson': float(corr_data['pearson']),
                    'pearson_p_corrected': float(corr_data['pearson_p_corrected']),
                    'spearman': float(corr_data['spearman']),
                    'spearman_p_corrected': float(corr_data['spearman_p_corrected']),
                    'n_effective': float(corr_data['n_effective'])
                }

        # Record silhouette scores
        if silhouette_scores:
            self.statistics_info['silhouette_scores'] = {}
            for pair_key, scores in silhouette_scores.items():
                self.statistics_info['silhouette_scores'][pair_key] = {
                    'silhouette_k2': float(scores['silhouette_k2']),
                    'silhouette_k3': float(scores['silhouette_k3']),
                    'optimal_k': int(scores['optimal_k'])
                }

    def add_metadata(self, **kwargs) -> None:
        """
        Add custom metadata fields.

        Args:
            **kwargs: Arbitrary key-value pairs to add to metadata
        """
        self.metadata.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert log to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the log
        """
        return {
            'timestamp': self.timestamp,
            'hdf5_file': self.hdf5_path,
            'loading': self.loading_info,
            'processing': self.processing_info,
            'alignment': self.alignment_info,
            'cropping': self.cropping_info,
            'statistics': self.statistics_info,
            'metadata': self.metadata
        }

    def to_text(self) -> str:
        """
        Generate human-readable text summary.

        Returns:
            Formatted text summary
        """
        lines = []
        lines.append("=" * 80)
        lines.append("TIME SERIES PROCESSING LOG")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {self.timestamp}")
        lines.append(f"HDF5 File: {self.hdf5_path}")
        lines.append("")

        # Data Loading
        lines.append("-" * 80)
        lines.append("DATA LOADING")
        lines.append("-" * 80)
        if self.loading_info:
            for ds in self.loading_info.get('datasets', []):
                lines.append(f"\n{ds['label']}:")
                lines.append(f"  Group: {ds['group']}")
                lines.append(f"  Dataset: {ds['name']}")
                lines.append(f"  Time source: {ds['time_group']}/{ds['time_name']} ({ds['time_units']})")
                lines.append(f"  Manual time shift: {ds['manual_time_shift']}s")
                if ds['label'] in self.loading_info.get('sampling_rates', {}):
                    lines.append(f"  Sampling rate: {self.loading_info['sampling_rates'][ds['label']]:.2f} Hz")
                if ds['label'] in self.loading_info.get('data_shapes', {}):
                    lines.append(f"  Data shape: {self.loading_info['data_shapes'][ds['label']]}")
        lines.append("")

        # Signal Processing
        lines.append("-" * 80)
        lines.append("SIGNAL PROCESSING")
        lines.append("-" * 80)
        if self.processing_info:
            proc = self.processing_info

            lines.append(f"Reference sampling rate: {proc.get('reference_sampling_rate', 'N/A')} Hz")
            lines.append(f"Handle NaNs: {proc.get('handle_nans', False)}")
            lines.append("")

            if proc.get('remove_outliers'):
                lines.append("Outlier Removal:")
                lines.append(f"  Method: {proc['outlier_method']}")
                lines.append(f"  Threshold: {proc['outlier_threshold']}")
                lines.append(f"  Window: {proc['outlier_window']} samples")
                if 'outlier_statistics' in proc:
                    for label, stats in proc['outlier_statistics'].items():
                        lines.append(f"  {label}: {stats['n_outliers']} outliers ({stats['outlier_percentage']:.2f}%)")
                lines.append("")

            if proc.get('apply_detrend'):
                lines.append(f"Detrending: {proc['detrend_method']}")
                lines.append("")

            if proc.get('apply_savgol'):
                lines.append("Savitzky-Golay Filter:")
                lines.append(f"  Window: {proc['savgol_window']} samples")
                lines.append(f"  Polynomial order: {proc['savgol_polyorder']}")
                lines.append("")

            if proc.get('apply_lowpass'):
                lines.append("Lowpass Filter:")
                lines.append(f"  Cutoff: {proc['lowpass_cutoff']} (normalized)")
                lines.append(f"  Order: {proc['lowpass_order']}")
                lines.append("")

            if proc.get('apply_highpass'):
                lines.append("Highpass Filter:")
                lines.append(f"  Cutoff: {proc['highpass_cutoff']} (normalized)")
                lines.append(f"  Order: {proc['highpass_order']}")
                lines.append("")

            if proc.get('apply_bandpass'):
                lines.append("Bandpass Filter:")
                lines.append(f"  Low cutoff: {proc['bandpass_low']} (normalized)")
                lines.append(f"  High cutoff: {proc['bandpass_high']} (normalized)")
                lines.append(f"  Order: {proc['bandpass_order']}")
                lines.append("")

            if proc.get('apply_smoothing'):
                lines.append("Smoothing:")
                lines.append(f"  Method: {proc['smoothing_method']}")
                lines.append(f"  Window: {proc['smoothing_window']} samples")
                lines.append("")

            if proc.get('apply_resampling'):
                lines.append(f"Resampling: {proc['target_samples']} samples")
                lines.append("")

            if proc.get('apply_normalization'):
                lines.append(f"Normalization: {proc['normalization_method']}")
                lines.append("")

        # Alignment
        lines.append("-" * 80)
        lines.append("TIME SERIES ALIGNMENT")
        lines.append("-" * 80)
        if self.alignment_info:
            align = self.alignment_info
            lines.append(f"Auto-alignment enabled: {align.get('auto_alignment_enabled', False)}")

            if align.get('auto_alignment_enabled') and 'auto_alignment_params' in align:
                params = align['auto_alignment_params']
                lines.append("\nAuto-alignment parameters:")
                lines.append(f"  Reference: {params.get('reference_label') or params.get('reference_group')}")
                lines.append(f"  Correlation window: {params.get('correlation_window_time')}s")
                lines.append(f"  Max shift: {params.get('max_shift_time')}s")
                lines.append(f"  Correlation method: {params.get('correlation_method')}")
                lines.append(f"  Normalize signals: {params.get('normalize_signals')}")
                lines.append(f"  Sync within groups: {params.get('sync_within_groups')}")

            if 'shifts_per_signal' in align:
                lines.append("\nApplied shifts:")
                for label, shift_info in align['shifts_per_signal'].items():
                    total_shift_ms = shift_info['total_time_shift'] * 1000
                    manual_ms = shift_info['manual_shift'] * 1000
                    auto_ms = shift_info['auto_shift'] * 1000

                    if abs(total_shift_ms) < 0.001:
                        lines.append(f"  {label}: Reference (0.0ms)")
                    else:
                        components = []
                        if abs(manual_ms) > 0.001:
                            components.append(f"{manual_ms:+.3f}ms manual")
                        if abs(auto_ms) > 0.001:
                            components.append(f"{auto_ms:+.3f}ms auto")
                        comp_str = " + ".join(components) if components else ""
                        lines.append(f"  {label}: {total_shift_ms:+.3f}ms total ({comp_str})")
        lines.append("")

        # Cropping
        lines.append("-" * 80)
        lines.append("SIGNAL CROPPING")
        lines.append("-" * 80)
        if self.cropping_info:
            crop = self.cropping_info
            if crop.get('cropping_applied'):
                if 'common_time_range' in crop:
                    time_range = crop['common_time_range']
                    lines.append(f"Common time range: {time_range['start']:.6f}s to {time_range['end']:.6f}s")
                    lines.append(f"Common duration: {time_range['duration']:.6f}s")

                lines.append("\nPer-signal cropping:")
                for label, info in crop.get('per_signal', {}).items():
                    lines.append(f"  {label}:")
                    lines.append(f"    Original: {info['original_samples']} samples ({info['original_duration']:.6f}s)")
                    lines.append(f"    Cropped: {info['cropped_samples']} samples ({info['cropped_duration']:.6f}s)")
                    lines.append(f"    Removed: {info['samples_removed']} samples ({info['removal_percentage']:.1f}%)")
            else:
                lines.append("No cropping applied")
        lines.append("")

        # Statistics Summary
        lines.append("-" * 80)
        lines.append("STATISTICAL ANALYSIS")
        lines.append("-" * 80)
        if self.statistics_info:
            stats = self.statistics_info
            lines.append(f"Correlations calculated: {stats.get('correlations_calculated', False)}")
            lines.append(f"Silhouette scores calculated: {stats.get('silhouette_scores_calculated', False)}")

            if 'per_signal_statistics' in stats:
                lines.append("\nPer-signal statistics:")
                for label, signal_stats in stats['per_signal_statistics'].items():
                    lines.append(f"  {label}:")
                    lines.append(f"    Mean: {signal_stats['mean']:.4f}")
                    lines.append(f"    Std: {signal_stats['std']:.4f}")
                    lines.append(f"    Range: [{signal_stats['min']:.4f}, {signal_stats['max']:.4f}]")
        lines.append("")

        # Metadata
        if self.metadata:
            lines.append("-" * 80)
            lines.append("METADATA")
            lines.append("-" * 80)
            for key, value in self.metadata.items():
                lines.append(f"{key}: {value}")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def save(self, output_dir: Path) -> None:
        """
        Save log to both JSON and text files.

        Args:
            output_dir: Directory to save log files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_dir / 'processing_log.json'
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save text summary
        txt_path = output_dir / 'processing_summary.txt'
        with open(txt_path, 'w') as f:
            f.write(self.to_text())

        print(f"Processing log saved:")
        print(f"  JSON: {json_path}")
        print(f"  Text: {txt_path}")
