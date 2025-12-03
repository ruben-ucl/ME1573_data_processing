"""
Visualization and plotting operations for time series.

This module contains all plotting functionality including statistics summaries,
correlation matrices, scatterplot matrices, and time series visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr

# Try to import statsmodels for cross-correlation function
try:
    from statsmodels.tsa.stattools import ccf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class PlottingMixin:
    """
    Mixin class containing all visualization operations.

    This class expects the following attributes to be available from the parent:
    - self.datasets: List[DatasetConfig]
    - self.processed_data: Dict[str, np.ndarray]
    - self.raw_data: Dict[str, np.ndarray]
    - self.time_vectors: Dict[str, np.ndarray]
    - self.original_time_vectors: Dict[str, np.ndarray]
    - self.statistics: Dict
    - self.correlations: Dict
    - self.alignment_info: Dict
    - self.full_processed_data: Dict[str, np.ndarray]
    """

    def plot_statistics_summary(self, save_path: Optional[str] = None) -> None:
        """Plot comprehensive statistics summary"""
        if not self.statistics:
            print("No statistics calculated. Run calculate_statistics() first.")
            return
        
        # Create statistics DataFrame
        stats_df = pd.DataFrame(self.statistics).T
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Basic statistics bar plot
        basic_stats = ['mean', 'median', 'std', 'range']
        stats_df[basic_stats].plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Basic Statistics')
        axes[0,0].set_ylabel('Value')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Distribution characteristics
        dist_stats = ['skewness', 'kurtosis']
        stats_df[dist_stats].plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Distribution Characteristics')
        axes[0,1].set_ylabel('Value')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Signal energy and RMS
        energy_stats = ['rms', 'energy']
        stats_df[energy_stats].plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Signal Energy Characteristics')
        axes[1,0].set_ylabel('Value')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Zero crossings
        stats_df[['zero_crossings']].plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Zero Crossings')
        axes[1,1].set_ylabel('Count')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Statistics plot saved to {save_path}")

        plt.show()
        plt.close()
    
    def plot_correlation_matrix(self, save_path: Optional[str] = None) -> None:
        """Plot correlation matrix heatmap"""
        correlations = self.calculate_correlations()

        if not correlations:
            print("No correlations to plot (need at least 2 datasets)")
            return

        # Create correlation matrices
        labels = list(self.processed_data.keys())
        n_labels = len(labels)

        pearson_matrix = np.eye(n_labels)
        spearman_matrix = np.eye(n_labels)

        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels):
                if i != j:
                    pair_key = f"{label1} vs {label2}" if i < j else f"{label2} vs {label1}"
                    if pair_key in correlations:
                        pearson_matrix[i,j] = correlations[pair_key]['pearson']
                        spearman_matrix[i,j] = correlations[pair_key]['spearman']

        # Plot correlation matrices
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Pearson correlation
        sns.heatmap(pearson_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_title('Pearson Correlation Matrix')

        # Spearman correlation
        sns.heatmap(spearman_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=labels, yticklabels=labels, ax=axes[1])
        axes[1].set_title('Spearman Correlation Matrix')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved to {save_path}")

        plt.show()
        plt.close()

    def plot_scatterplot_matrix(self, save_path: Optional[str] = None,
                                max_points: int = 5000,
                                show_correlation: bool = True,
                                show_regression: bool = True) -> None:
        """
        Plot scatterplot matrix (pair plot) showing all pairwise relationships

        Also known as SPLOM (Scatterplot Matrix). Shows scatter plots for each
        pair of variables with optional correlation coefficients and regression lines.

        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        max_points : int, default=5000
            Downsample to this many points if signals are longer (for performance)
        show_correlation : bool, default=True
            Show Pearson correlation coefficient on each subplot
        show_regression : bool, default=True
            Show linear regression line on scatter plots
        """
        if not self.processed_data:
            print("No processed data available. Load and process data first.")
            return

        if len(self.processed_data) < 2:
            print("Need at least 2 datasets for scatterplot matrix")
            return

        # Get synchronized data for all pairs
        labels = list(self.processed_data.keys())
        n_labels = len(labels)

        # Prepare synchronized data dictionary
        sync_data = {}
        for label in labels:
            sync_data[label] = []

        # Synchronize all pairs to get data on common time base
        # We'll use the first signal as reference and sync all others to it
        ref_label = labels[0]
        ref_data = self.processed_data[ref_label]
        ref_time = self.time_vectors[ref_label]

        for label in labels:
            if label == ref_label:
                sync_data[label] = ref_data
            else:
                data = self.processed_data[label]
                time = self.time_vectors[label]
                _, data_sync = self._synchronize_time_series(
                    ref_data, ref_time, data, time
                )
                sync_data[label] = data_sync

        # Ensure all have same length (use minimum)
        min_len = min(len(sync_data[label]) for label in labels)
        for label in labels:
            sync_data[label] = sync_data[label][:min_len]

        # Downsample if needed for performance
        if min_len > max_points:
            downsample_factor = min_len // max_points
            for label in labels:
                sync_data[label] = sync_data[label][::downsample_factor]
            print(f"Downsampled from {min_len} to {len(sync_data[labels[0]])} points for visualization")

        # Create figure with subplots
        fig, axes = plt.subplots(n_labels, n_labels, figsize=(4*n_labels, 4*n_labels))

        # Handle single subplot case
        if n_labels == 2:
            axes = np.array([[None, None], [None, None]])
            axes_flat = axes.flatten()
            fig, axes_flat_new = plt.subplots(1, 4, figsize=(16, 4))
            # We'll only use the off-diagonal
            axes = axes_flat_new.reshape(2, 2)

        # Iterate through all subplot positions
        for i, label_y in enumerate(labels):
            for j, label_x in enumerate(labels):
                if n_labels == 2:
                    if i == j:
                        continue
                    ax = axes[i, j]
                else:
                    ax = axes[i, j]

                # Diagonal: show histogram
                if i == j:
                    data = sync_data[label_y]
                    ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                    ax.set_ylabel('Frequency', fontsize=9)
                    ax.set_xlabel(label_y if i == n_labels-1 else '', fontsize=9)
                    ax.set_title(f'{label_y}\n(histogram)', fontsize=10, fontweight='bold')
                    ax.grid(True, alpha=0.3)

                # Off-diagonal: show scatter plot
                else:
                    data_x = sync_data[label_x]
                    data_y = sync_data[label_y]

                    # Scatter plot
                    ax.scatter(data_x, data_y, alpha=0.3, s=1, color='steelblue', rasterized=True)

                    # Add regression line if requested
                    if show_regression:
                        # Calculate linear fit
                        z = np.polyfit(data_x, data_y, 1)
                        p = np.poly1d(z)
                        x_line = np.array([np.min(data_x), np.max(data_x)])
                        y_line = p(x_line)
                        ax.plot(x_line, y_line, 'r-', linewidth=2, alpha=0.8, label='Linear fit')

                    # Add correlation coefficient if requested
                    if show_correlation:
                        corr, _ = pearsonr(data_x, data_y)
                        # Position text in corner
                        ax.text(0.05, 0.95, f'r = {corr:.3f}',
                               transform=ax.transAxes,
                               fontsize=9, fontweight='bold',
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                    # Labels
                    if j == 0:
                        ax.set_ylabel(label_y, fontsize=9)
                    if i == n_labels - 1:
                        ax.set_xlabel(label_x, fontsize=9)

                    ax.grid(True, alpha=0.3)

                # Tick label size
                ax.tick_params(labelsize=8)

        # Overall title
        fig.suptitle('Scatterplot Matrix (Pairwise Correlations)',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scatterplot matrix saved to {save_path}")

        plt.show()
        plt.close()

    def plot_scatterplot_matrix_compact(self, save_path: Optional[str] = None,
                                        max_points: int = 5000,
                                        point_size: float = 2.0,
                                        point_alpha: float = 0.5) -> None:
        """
        Plot compact scatterplot matrix for small-size display

        Minimal design with no axis ticks, numbers, regression lines, or correlation values.
        Shows only scatter points and variable labels on edges.

        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        max_points : int, default=5000
            Downsample to this many points if signals are longer
        point_size : float, default=2.0
            Size of scatter points (increase for better visibility)
        point_alpha : float, default=0.5
            Transparency of scatter points (0=transparent, 1=opaque)
        """
        if not self.processed_data:
            print("No processed data available. Load and process data first.")
            return

        if len(self.processed_data) < 2:
            print("Need at least 2 datasets for scatterplot matrix")
            return

        # Get synchronized data for all pairs
        labels = list(self.processed_data.keys())
        n_labels = len(labels)

        # Prepare synchronized data dictionary
        sync_data = {}
        for label in labels:
            sync_data[label] = []

        # Synchronize all pairs to get data on common time base
        ref_label = labels[0]
        ref_data = self.processed_data[ref_label]
        ref_time = self.time_vectors[ref_label]

        for label in labels:
            if label == ref_label:
                sync_data[label] = ref_data
            else:
                data = self.processed_data[label]
                time = self.time_vectors[label]
                _, data_sync = self._synchronize_time_series(
                    ref_data, ref_time, data, time
                )
                sync_data[label] = data_sync

        # Ensure all have same length (use minimum)
        min_len = min(len(sync_data[label]) for label in labels)
        for label in labels:
            sync_data[label] = sync_data[label][:min_len]

        # Downsample if needed for performance
        if min_len > max_points:
            downsample_factor = min_len // max_points
            for label in labels:
                sync_data[label] = sync_data[label][::downsample_factor]
            print(f"Downsampled from {min_len} to {len(sync_data[labels[0]])} points for visualization")

        # Create figure with smaller size for compact display
        fig, axes = plt.subplots(n_labels, n_labels,
                                figsize=(2*n_labels, 2*n_labels))

        # Handle single pair case
        if n_labels == 2:
            axes = np.array([[axes[0], axes[1]], [axes[2], axes[3]]])

        # Iterate through all subplot positions
        for i, label_y in enumerate(labels):
            for j, label_x in enumerate(labels):
                ax = axes[i, j] if n_labels > 2 else axes.flatten()[i*n_labels + j]

                # Off-diagonal: show scatter plot only
                if i != j:
                    data_x = sync_data[label_x]
                    data_y = sync_data[label_y]

                    # Minimal scatter plot
                    ax.scatter(data_x, data_y, alpha=point_alpha, s=point_size,
                              color='steelblue', rasterized=True)

                # Diagonal: show histogram
                else:
                    data = sync_data[label_y]
                    ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='none')

                # Remove all ticks and tick labels
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(left=False, bottom=False)

                # Add labels only on edges
                if i == n_labels - 1:  # Bottom row
                    ax.set_xlabel(label_x, fontsize=10, fontweight='bold')
                if j == 0:  # Left column
                    ax.set_ylabel(label_y, fontsize=10, fontweight='bold')

                # Minimal styling
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)
                    spine.set_color('gray')

        # Tight layout with minimal spacing
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Compact scatterplot matrix saved to {save_path}")

        plt.show()
        plt.close()

    def plot_processing_and_alignment_summary(self, save_path: Optional[str] = None, 
                                             use_full_data: bool = True) -> None:
        """Consolidated plot showing complete data processing pipeline: Raw → Processed → Aligned"""
        if not self.original_time_vectors:
            print("No original time vectors available for alignment comparison")
            return
        
        # Choose data source
        if use_full_data and hasattr(self, 'full_processed_data') and self.full_processed_data:
            plot_data = self.full_processed_data
            data_type = "full processed"
            print("Plotting full processed data (before cropping)")
        else:
            plot_data = self.processed_data
            data_type = "current processed"
            print("Plotting current processed data (potentially cropped)")

        # Create label-to-group mapping from dataset configs
        label_to_group = {ds.label: ds.group for ds in self.datasets}

        # Get unique groups and determine row assignment
        unique_groups = sorted(set(label_to_group.values()))
        if len(unique_groups) != 2:
            print(f"Warning: Expected 2 groups, found {len(unique_groups)}: {unique_groups}")

        # Create 2x3 grid: Row assignment based on group order
        # Columns: Raw → Processed → Final Aligned
        fig, axes = plt.subplots(2, 3, figsize=(20, 10), facecolor='white')
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, wspace=0.25, hspace=0.30)

        # Unpack axes: row 0 = first group, row 1 = second group
        ax_row0 = {col: axes[0, col] for col in range(3)}  # raw, proc, final
        ax_row1 = {col: axes[1, col] for col in range(3)}

        # Map groups to rows
        group_to_row = {unique_groups[0]: 0, unique_groups[1]: 1}
        row_to_axes = {0: ax_row0, 1: ax_row1}
        row_to_group = {0: unique_groups[0], 1: unique_groups[1]}

        # Define colorblind-friendly colors (using Okabe-Ito palette)
        colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#FBAFE4', '#949494', '#ECE133']
        linewidth_thin = 1.0
        linewidth_medium = 1.5

        # Store alignment info for final summary
        alignment_summary = []

        # Set titles for all subplots
        for row in [0, 1]:
            group_name = row_to_group[row]
            row_to_axes[row][0].set_title(f'Raw Signals ({group_name} group)', fontsize=12, fontweight='bold', pad=10)
            row_to_axes[row][1].set_title(f'Processed Signals ({group_name} group)', fontsize=12, fontweight='bold', pad=10)
            row_to_axes[row][2].set_title(f'Final Aligned ({group_name} group)', fontsize=12, fontweight='bold', pad=10)

        # Panel 1: Raw signals
        for i, (label, processed_data) in enumerate(plot_data.items()):
            color = colors[i % len(colors)]

            if label in self.original_time_vectors and label in self.raw_data:
                orig_time = self.original_time_vectors[label]
                raw_data = self.raw_data[label]

                if use_full_data:
                    time_vec = orig_time
                    # Use raw data at full length to match
                    raw_plot_data = raw_data[:len(time_vec)] if len(raw_data) >= len(time_vec) else raw_data
                    time_vec = time_vec[:len(raw_plot_data)]
                else:
                    time_vec = orig_time[:len(processed_data)]
                    raw_plot_data = raw_data[:len(processed_data)]

                # Get group and row for this signal
                signal_group = label_to_group.get(label)
                if signal_group in group_to_row:
                    row = group_to_row[signal_group]
                    ax = row_to_axes[row][0]  # Column 0 = raw
                    ax.plot(time_vec, raw_plot_data,
                           label=f"{label}",
                           color=color, linewidth=linewidth_thin, alpha=0.8)
        
        # Panel 2: Processed signals
        for i, (label, data) in enumerate(plot_data.items()):
            color = colors[i % len(colors)]

            if label in self.original_time_vectors:
                orig_time = self.original_time_vectors[label]

                if use_full_data:
                    time_vec = orig_time
                else:
                    time_vec = orig_time[:len(data)]

                # Get group and row for this signal
                signal_group = label_to_group.get(label)
                if signal_group in group_to_row:
                    row = group_to_row[signal_group]
                    ax = row_to_axes[row][1]  # Column 1 = processed
                    ax.plot(time_vec, data,
                           label=f"{label}",
                           color=color, linewidth=linewidth_thin, alpha=0.9)
        
        # Panel 3: Final aligned signals
        # Use current processed data (which includes cropping) for the final panel
        final_data = self.processed_data if not use_full_data or not hasattr(self, 'full_processed_data') else self.processed_data

        for i, (label, data) in enumerate(final_data.items()):
            color = colors[i % len(colors)]

            # Get shift information
            shift_info = self.alignment_info[label]
            total_shift = shift_info['time_shift']
            manual_shift = shift_info.get('manual_shift', 0.0)
            auto_shift = shift_info.get('auto_shift', 0.0)

            # Always apply normalization for final panel display (regardless of processing config)
            # This ensures the final panel shows normalized data for comparison
            data_min, data_max = np.min(data), np.max(data)
            if data_max > data_min:  # Avoid division by zero
                data_norm = (data - data_min) / (data_max - data_min)
            else:
                data_norm = np.zeros_like(data)

            # Use current time vectors (which include alignment and cropping)
            time_vec = self.time_vectors[label][:len(data)]

            # Create concise labels
            if abs(total_shift) > 1e-6:
                label_text = f"{label} ({total_shift*1000:+.1f}ms)"
            else:
                label_text = f"{label} (ref)"

            # Get group and row for this signal
            signal_group = label_to_group.get(label)
            if signal_group in group_to_row:
                row = group_to_row[signal_group]
                ax = row_to_axes[row][2]  # Column 2 = final
                ax.plot(time_vec, data_norm,
                       label=label_text,
                       color=color, linewidth=linewidth_medium, alpha=0.95)

            # Build alignment summary for text box
            if total_shift != 0.0:
                if manual_shift != 0.0 and auto_shift != 0.0:
                    alignment_summary.append(f"{label}: {manual_shift*1000:+.1f}ms (manual) + {auto_shift*1000:+.1f}ms (auto) = {total_shift*1000:+.1f}ms")
                elif manual_shift != 0.0:
                    alignment_summary.append(f"{label}: {total_shift*1000:+.1f}ms (manual only)")
                elif auto_shift != 0.0:
                    alignment_summary.append(f"{label}: {total_shift*1000:+.1f}ms (auto only)")
                else:
                    alignment_summary.append(f"{label}: {total_shift*1000:+.1f}ms")
            else:
                alignment_summary.append(f"{label}: Reference (0.0ms)")
        
        # Configure axes labels and styling for all 6 subplots
        for row in [0, 1]:
            for col in [0, 1, 2]:
                ax = row_to_axes[row][col]
                group_name = row_to_group[row]

                ax.set_xlabel('Time [s]', fontsize=10)

                # Set y-labels based on group and column
                if col == 2:  # Final (normalized)
                    ax.set_ylabel(f'{group_name} Signals\n(normalized)', fontsize=10)
                else:
                    ax.set_ylabel(f'{group_name} Signals', fontsize=10)

                # Configure tick marks
                ax.tick_params(axis='both', labelsize=9, direction='out',
                              colors='black', width=1, length=4)

                # Set plot background and borders
                ax.set_facecolor('white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1)
                    spine.set_visible(True)

                # Add legend to each subplot
                lines, labels_list = ax.get_legend_handles_labels()
                if lines:
                    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        
        # Add vertical guides to final aligned plots for phase comparison
        if len(final_data) > 1:
            all_final_times = []
            for label, data in final_data.items():
                time_vec = self.time_vectors[label][:len(data)]
                all_final_times.extend([time_vec[0], time_vec[-1]])

            if all_final_times:
                t_min, t_max = min(all_final_times), max(all_final_times)
                t_range = t_max - t_min
                n_guides = 5  # Fewer guides to reduce clutter
                guide_times = [t_min + i * t_range / (n_guides - 1) for i in range(n_guides)]

                # Add subtle guides to both final aligned plots
                for guide_time in guide_times:
                    for row in [0, 1]:
                        ax_final = row_to_axes[row][2]  # Column 2 = final
                        ax_final.axvline(x=guide_time, color='lightgray', linestyle=':',
                                  alpha=0.6, linewidth=0.5, zorder=0)
        
        # Add compact processing information as text box
        processing_info = []
        if self.processing_config.apply_normalization:
            processing_info.append(f"Norm: {self.processing_config.normalization_method}")
        if self.processing_config.apply_savgol:
            processing_info.append(f"Savgol({self.processing_config.savgol_window})")
        if self.processing_config.apply_lowpass:
            processing_info.append(f"Lowpass({self.processing_config.lowpass_cutoff})")
        if self.processing_config.apply_smoothing:
            processing_info.append(f"Smooth: {self.processing_config.smoothing_method}")
        
        processing_text = "Processing: " + ", ".join(processing_info) if processing_info else "Processing: None"
        
        # Compact alignment summary
        alignment_compact = []
        for label, info in self.alignment_info.items():
            shift = info['time_shift']
            if abs(shift) > 1e-6:
                alignment_compact.append(f"{label}: {shift*1000:+.1f}ms")
            else:
                alignment_compact.append(f"{label}: ref")
        
        alignment_text = "Alignment: " + ", ".join(alignment_compact) if alignment_compact else "Alignment: None"
        
        # Cleaner overall title centered
        fig.suptitle('Data Processing Pipeline Summary', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Place compact summary in bottom center
        summary_text = processing_text + " | " + alignment_text
        fig.text(0.5, 0.02, summary_text, ha='center', va='bottom', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=0.3),
                fontsize=9, transform=fig.transFigure)
        
        # Don't use tight_layout since we have custom spacing
        # plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Processing and alignment summary saved to {save_path}")
        
        plt.show()
        plt.close()

    def plot_autocorrelation(self, max_lag: Optional[int] = None,
                            save_path: Optional[str] = None,
                            normalize: bool = True) -> None:
        """
        Plot autocorrelation function (ACF) for each time series

        Parameters:
        -----------
        max_lag : int, optional
            Maximum lag to compute (default: min(len(data)//2, 1000))
        save_path : str, optional
            Path to save the figure
        normalize : bool, default=True
            If True, normalize autocorrelation to [-1, 1] range
        """
        if not self.processed_data:
            print("No processed data available. Load data first.")
            return

        n_series = len(self.processed_data)

        # Determine subplot layout
        if n_series == 1:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            axes = [axes]
        elif n_series == 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        elif n_series <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
        elif n_series <= 6:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            axes = axes.flatten()
        else:
            n_cols = 3
            n_rows = int(np.ceil(n_series / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
            axes = axes.flatten()

        # Compute and plot autocorrelation for each series
        for idx, (label, data) in enumerate(self.processed_data.items()):
            ax = axes[idx]

            # Determine max lag if not specified
            if max_lag is None:
                lag_limit = min(len(data) // 2, 1000)
            else:
                lag_limit = min(max_lag, len(data) - 1)

            # Compute autocorrelation using numpy correlate
            # Remove mean for proper autocorrelation
            data_centered = data - np.mean(data)

            # Full autocorrelation
            autocorr = np.correlate(data_centered, data_centered, mode='full')

            # Take only positive lags
            autocorr = autocorr[len(autocorr)//2:]

            # Normalize if requested
            if normalize:
                autocorr = autocorr / autocorr[0]  # Normalize by zero-lag value

            # Limit to max_lag
            autocorr = autocorr[:lag_limit+1]
            lags = np.arange(len(autocorr))

            # Convert lag indices to time if time vector exists
            if label in self.time_vectors and len(self.time_vectors[label]) > 0:
                time_vec = self.time_vectors[label]
                # Calculate average sampling interval
                dt = np.mean(np.diff(time_vec))
                time_lags = lags * dt
                x_label = f'Lag (seconds)'
                x_values = time_lags
            else:
                x_label = 'Lag (samples)'
                x_values = lags

            # Plot autocorrelation
            ax.plot(x_values, autocorr, linewidth=2, color='#0173B2', alpha=0.8)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

            # Add confidence interval (approximate 95% CI for white noise)
            conf_interval = 1.96 / np.sqrt(len(data))
            if normalize:
                ax.axhline(y=conf_interval, color='red', linestyle='--',
                          linewidth=1, alpha=0.5, label=f'95% CI (±{conf_interval:.3f})')
                ax.axhline(y=-conf_interval, color='red', linestyle='--',
                          linewidth=1, alpha=0.5)

            # Find first zero crossing (decorrelation time)
            zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
            if len(zero_crossings) > 0:
                first_zero = zero_crossings[0]
                ax.axvline(x=x_values[first_zero], color='green', linestyle=':',
                          linewidth=1.5, alpha=0.7,
                          label=f'First zero: {x_values[first_zero]:.4f}')

            # Find where autocorrelation drops below 1/e (decay time)
            if normalize:
                decay_threshold = 1.0 / np.e
                below_threshold = np.where(autocorr < decay_threshold)[0]
                if len(below_threshold) > 0:
                    decay_idx = below_threshold[0]
                    ax.axvline(x=x_values[decay_idx], color='orange', linestyle=':',
                              linewidth=1.5, alpha=0.7,
                              label=f'1/e decay: {x_values[decay_idx]:.4f}')

            # Labels and title
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel('Autocorrelation' if normalize else 'Autocovariance', fontsize=11)
            ax.set_title(f'{label}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)

            # Set y-axis limits for normalized case
            if normalize:
                ax.set_ylim([-0.5, 1.1])

        # Hide unused subplots
        for idx in range(n_series, len(axes)):
            axes[idx].axis('off')

        fig.suptitle('Autocorrelation Functions', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Autocorrelation plot saved to {save_path}")

        plt.show()
        plt.close()

    def calculate_cross_correlation_lags(self, max_lag: Optional[int] = None,
                                        use_statsmodels: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Calculate cross-correlation between all pairs of time series to detect lags

        Parameters:
        -----------
        max_lag : int, optional
            Maximum lag to consider (default: min(len(data)//4, 500))
        use_statsmodels : bool, default=True
            If True and available, use statsmodels CCF; otherwise use numpy

        Returns:
        --------
        lag_info : dict
            Dictionary with lag information for each pair:
            - 'optimal_lag_samples': Lag at maximum correlation (in samples)
            - 'optimal_lag_time': Lag in time units (seconds)
            - 'max_correlation': Maximum correlation value
            - 'correlation_at_zero': Correlation at zero lag
        """
        if not self.processed_data:
            print("No processed data available. Load data first.")
            return {}

        lag_info = {}
        labels = list(self.processed_data.keys())

        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels[i+1:], i+1):
                data1 = self.processed_data[label1]
                data2 = self.processed_data[label2]
                time1 = self.time_vectors[label1]
                time2 = self.time_vectors[label2]

                # Synchronize time series
                data1_sync, data2_sync = self._synchronize_time_series(
                    data1, time1, data2, time2
                )

                # Determine max lag
                if max_lag is None:
                    lag_limit = min(len(data1_sync) // 4, 500)
                else:
                    lag_limit = min(max_lag, len(data1_sync) - 1)

                # Use unified cross-correlation helper (checks both positive and negative lags)
                method_to_use = 'statsmodels' if (use_statsmodels and STATSMODELS_AVAILABLE) else 'numpy'
                lags, cross_corr, optimal_lag = self._compute_cross_correlation(
                    data1_sync, data2_sync,
                    max_lag=lag_limit,
                    method=method_to_use
                )

                # Find correlation values
                optimal_idx = np.where(lags == optimal_lag)[0][0]
                max_corr = cross_corr[optimal_idx]
                zero_idx = np.where(lags == 0)[0]
                corr_at_zero = cross_corr[zero_idx[0]] if len(zero_idx) > 0 else max_corr

                # Convert lag to time units
                if label1 in self.time_vectors and len(self.time_vectors[label1]) > 1:
                    dt = np.mean(np.diff(self.time_vectors[label1]))
                    optimal_lag_time = optimal_lag * dt
                else:
                    optimal_lag_time = float('nan')

                pair_key = f"{label1} vs {label2}"
                lag_info[pair_key] = {
                    'optimal_lag_samples': int(optimal_lag),
                    'optimal_lag_time': optimal_lag_time,
                    'max_correlation': float(max_corr),
                    'correlation_at_zero': float(corr_at_zero),
                    'lag_interpretation': self._interpret_lag(optimal_lag, label1, label2)
                }

        return lag_info

    def _interpret_lag(self, lag: int, label1: str, label2: str) -> str:
        """
        Interpret the meaning of a lag value

        Parameters:
        -----------
        lag : int
            Lag in samples (positive means label2 lags behind label1)
        label1, label2 : str
            Labels of the two series

        Returns:
        --------
        interpretation : str
            Human-readable interpretation of the lag
        """
        if lag == 0:
            return "No lag detected (synchronized)"
        elif lag > 0:
            return f"{label2} lags behind {label1} by {abs(lag)} samples"
        else:
            return f"{label1} lags behind {label2} by {abs(lag)} samples"

    def plot_cross_correlation(self, max_lag: Optional[int] = None,
                              save_path: Optional[str] = None,
                              use_statsmodels: bool = True) -> None:
        """
        Plot cross-correlation functions for all pairs of time series

        Parameters:
        -----------
        max_lag : int, optional
            Maximum lag to compute (default: min(len(data)//4, 500))
        save_path : str, optional
            Path to save the figure
        use_statsmodels : bool, default=True
            If True and available, use statsmodels CCF
        """
        if not self.processed_data:
            print("No processed data available. Load data first.")
            return

        labels = list(self.processed_data.keys())
        n_pairs = len(labels) * (len(labels) - 1) // 2

        if n_pairs == 0:
            print("Need at least 2 datasets for cross-correlation analysis")
            return

        # Calculate lag information
        lag_info = self.calculate_cross_correlation_lags(max_lag=max_lag,
                                                         use_statsmodels=use_statsmodels)

        # Determine subplot layout
        if n_pairs == 1:
            fig, axes = plt.subplots(1, 1, figsize=(12, 6))
            axes = [axes]
        elif n_pairs == 2:
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        elif n_pairs <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            axes = axes.flatten()
        elif n_pairs <= 6:
            fig, axes = plt.subplots(2, 3, figsize=(24, 12))
            axes = axes.flatten()
        else:
            n_cols = 3
            n_rows = int(np.ceil(n_pairs / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows))
            axes = axes.flatten()

        # Plot cross-correlation for each pair
        pair_idx = 0
        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels[i+1:], i+1):
                ax = axes[pair_idx]

                data1 = self.processed_data[label1]
                data2 = self.processed_data[label2]
                time1 = self.time_vectors[label1]
                time2 = self.time_vectors[label2]

                # Synchronize time series
                data1_sync, data2_sync = self._synchronize_time_series(
                    data1, time1, data2, time2
                )

                # Determine max lag
                if max_lag is None:
                    lag_limit = min(len(data1_sync) // 4, 500)
                else:
                    lag_limit = min(max_lag, len(data1_sync) - 1)

                # Use unified cross-correlation helper (checks both positive and negative lags)
                method_to_use = 'statsmodels' if (use_statsmodels and STATSMODELS_AVAILABLE) else 'numpy'
                lags, cross_corr, peak_lag = self._compute_cross_correlation(
                    data1_sync, data2_sync,
                    max_lag=lag_limit,
                    method=method_to_use
                )

                # Convert lags to time if possible
                if label1 in self.time_vectors and len(self.time_vectors[label1]) > 1:
                    dt = np.mean(np.diff(self.time_vectors[label1]))
                    time_lags = lags * dt
                    x_label = 'Lag (seconds)'
                    x_values = time_lags
                else:
                    x_label = 'Lag (samples)'
                    x_values = lags

                # Plot cross-correlation
                ax.plot(x_values, cross_corr, linewidth=2, color='#0173B2', alpha=0.8)
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

                # Mark optimal lag
                pair_key = f"{label1} vs {label2}"
                if pair_key in lag_info:
                    opt_lag_samples = lag_info[pair_key]['optimal_lag_samples']
                    if label1 in self.time_vectors and len(self.time_vectors[label1]) > 1:
                        opt_lag_display = lag_info[pair_key]['optimal_lag_time']
                    else:
                        opt_lag_display = opt_lag_samples

                    # Find the correlation value at optimal lag
                    if use_statsmodels and STATSMODELS_AVAILABLE:
                        opt_lag_idx = abs(opt_lag_samples)
                        if opt_lag_idx < len(cross_corr):
                            opt_corr = cross_corr[opt_lag_idx]
                        else:
                            opt_corr = lag_info[pair_key]['max_correlation']
                    else:
                        opt_lag_idx = np.argmin(np.abs(lags - opt_lag_samples))
                        opt_corr = cross_corr[opt_lag_idx]

                    ax.axvline(x=x_values[opt_lag_idx] if opt_lag_idx < len(x_values) else opt_lag_display,
                              color='red', linestyle=':', linewidth=2, alpha=0.7,
                              label=f'Max corr at lag: {opt_lag_display:.4f}')

                    # Add text annotation
                    ax.text(0.02, 0.98,
                           f"Max correlation: {lag_info[pair_key]['max_correlation']:.3f}\n"
                           f"At zero lag: {lag_info[pair_key]['correlation_at_zero']:.3f}",
                           transform=ax.transAxes, fontsize=9,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                # Labels and title
                ax.set_xlabel(x_label, fontsize=11)
                ax.set_ylabel('Cross-correlation', fontsize=11)
                ax.set_title(f'{label1} vs {label2}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=9)
                ax.set_ylim([-1.1, 1.1])

                pair_idx += 1

        # Hide unused subplots
        for idx in range(pair_idx, len(axes)):
            axes[idx].axis('off')

        method_str = "statsmodels CCF" if (use_statsmodels and STATSMODELS_AVAILABLE) else "numpy correlate"
        fig.suptitle(f'Cross-Correlation Functions ({method_str})',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cross-correlation plot saved to {save_path}")

        plt.show()
        plt.close()
