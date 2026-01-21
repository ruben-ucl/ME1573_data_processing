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

    # Colorblind-friendly color palette (Okabe-Ito)
    COLORS_OKABE_ITO = [
        '#0173B2',  # Blue
        '#DE8F05',  # Orange
        '#029E73',  # Green
        '#CC78BC',  # Purple
        '#CA9161',  # Brown
        '#FBAFE4',  # Pink
        '#949494',  # Gray
        '#ECE133'   # Yellow
    ]

    # Diverging colormap for heatmaps
    CMAP_DIVERGING = 'coolwarm' 

    # Accent colors
    COLOR_ACCENT_BG = '#E8F4F8'      # Light blue-gray background for text boxes
    COLOR_ACCENT_GUIDE = '#CCCCCC'    # Medium gray for guide lines
    COLOR_SIGNIFICANCE = '#D55E00'    # Orange for significance markers

    def _get_signal_color(self, index: int) -> str:
        """Get color for signal by index from Okabe-Ito palette"""
        return self.COLORS_OKABE_ITO[index % len(self.COLORS_OKABE_ITO)]

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

        # Don't show plots - only save them
        # plt.show()
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
        sns.heatmap(pearson_matrix, annot=True, cmap=self.CMAP_DIVERGING, center=0,
                   xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_title('Pearson Correlation Matrix')

        # Spearman correlation
        sns.heatmap(spearman_matrix, annot=True, cmap=self.CMAP_DIVERGING, center=0,
                   xticklabels=labels, yticklabels=labels, ax=axes[1])
        axes[1].set_title('Spearman Correlation Matrix')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved to {save_path}")

        # Don't show plots - only save them
        # plt.show()
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
                # Interpolate to reference time grid (ensures all signals have same length)
                data_sync = np.interp(ref_time, time, data)
                sync_data[label] = data_sync

        # All signals now have same length as reference
        min_len = len(ref_data)

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
                    ax.hist(data, bins=30, alpha=0.7, color=self._get_signal_color(i), edgecolor='black')
                    ax.set_ylabel('Frequency', fontsize=9)
                    ax.set_xlabel(label_y if i == n_labels-1 else '', fontsize=9)
                    ax.set_title(f'{label_y}\n(histogram)', fontsize=10, fontweight='bold')
                    ax.grid(True, alpha=0.3)

                # Off-diagonal: show scatter plot
                else:
                    data_x = sync_data[label_x]
                    data_y = sync_data[label_y]

                    # Scatter plot - use color based on first signal (x-axis)
                    ax.scatter(data_x, data_y, alpha=0.3, s=1, color=self._get_signal_color(j), rasterized=True)

                    # Add regression line if requested
                    if show_regression:
                        try:
                            # Calculate linear fit
                            z = np.polyfit(data_x, data_y, 1)
                            p = np.poly1d(z)
                            x_line = np.array([np.min(data_x), np.max(data_x)])
                            y_line = p(x_line)
                            ax.plot(x_line, y_line, 'r-', linewidth=2, alpha=0.8, label='Linear fit')
                        except (np.linalg.LinAlgError, ValueError) as e:
                            # Skip regression line if fit fails (e.g., constant data, NaN values)
                            pass

                    # Add correlation coefficient if requested
                    if show_correlation:
                        corr, _ = pearsonr(data_x, data_y)
                        # Position text in corner
                        ax.text(0.05, 0.95, f'r = {corr:.3f}',
                               transform=ax.transAxes,
                               fontsize=9, fontweight='bold',
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor=self.COLOR_ACCENT_BG, alpha=0.8))

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

        # Don't show plots - only save them
        # plt.show()
        plt.close()

    def plot_scatterplot_matrix_compact(self, save_path: Optional[str] = None,
                                        max_points: int = 5000,
                                        point_size: float = 2.0,
                                        point_alpha: float = 0.3) -> None:
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
                # Interpolate to reference time grid (ensures all signals have same length)
                data_sync = np.interp(ref_time, time, data)
                sync_data[label] = data_sync

        # All signals now have same length as reference
        min_len = len(ref_data)

        # Downsample if needed for performance
        if min_len > max_points:
            downsample_factor = min_len // max_points
            for label in labels:
                sync_data[label] = sync_data[label][::downsample_factor]
            print(f"Downsampled from {min_len} to {len(sync_data[labels[0]])} points for visualization")

        # Create figure: 1/2 A4 width (105mm = 4.13 inches)
        # Scale total size by number of subplots while keeping overall width ~4.13"
        fig_width = 4.13  # 1/2 A4 page width in inches
        fig_height = 4.13  # Square aspect ratio
        fig, axes = plt.subplots(n_labels, n_labels,
                                figsize=(fig_width, fig_height))

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

                    # Minimal scatter plot - use color based on x-axis signal
                    # No rasterization for sharper points at high DPI
                    ax.scatter(data_x, data_y, alpha=point_alpha, s=point_size,
                              color=self._get_signal_color(j), edgecolors='none')

                # Diagonal: show histogram
                else:
                    data = sync_data[label_y]
                    ax.hist(data, bins=20, alpha=0.7, color=self._get_signal_color(i),
                           edgecolor='black', linewidth=0.5)

                # Remove all ticks and tick labels
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(left=False, bottom=False)

                # Add labels only on edges (9pt font)
                if i == n_labels - 1:  # Bottom row
                    ax.set_xlabel(label_x, fontsize=9, rotation=45, ha='right')
                if j == 0:  # Left column
                    ax.set_ylabel(label_y, fontsize=9, rotation=0, ha='right')

                # Minimal styling with appropriate line weights for 600 DPI
                for spine in ax.spines.values():
                    spine.set_linewidth(0.75)  # Thicker lines for better visibility at high DPI
                    spine.set_color('gray')

        # Tight layout with minimal spacing
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        if save_path:
            # Save with high quality settings for sharp rendering
            plt.savefig(save_path, dpi=600, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Compact scatterplot matrix saved to {save_path}")

        # Don't show plots - only save them
        # plt.show()
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
        fig.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.95, wspace=0.25, hspace=0.30)

        # Unpack axes: row 0 = first group, row 1 = second group
        ax_row0 = {col: axes[0, col] for col in range(3)}  # raw, proc, final
        ax_row1 = {col: axes[1, col] for col in range(3)}

        # Map groups to rows
        group_to_row = {unique_groups[0]: 0, unique_groups[1]: 1}
        row_to_axes = {0: ax_row0, 1: ax_row1}
        row_to_group = {0: unique_groups[0], 1: unique_groups[1]}

        # Use centralized color palette
        colors = self.COLORS_OKABE_ITO
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
        # Track secondary axes for KH area (which has much larger magnitude)
        secondary_axes = {}  # Store secondary axes by row

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

                    # Use secondary axis for KH area due to large magnitude
                    if label == 'KH area':
                        # Create secondary axis if not already created for this row
                        if row not in secondary_axes:
                            secondary_axes[row] = ax.twinx()
                        ax_plot = secondary_axes[row]
                        ax_plot.plot(time_vec, raw_plot_data,
                                   label=f"{label} (right axis)",
                                   color=color, linewidth=linewidth_thin, alpha=0.8, linestyle='--')
                        ax_plot.set_ylabel('KH area', fontsize=10, color=color)
                        ax_plot.tick_params(axis='y', labelcolor=color)
                    else:
                        ax_plot = ax
                        ax_plot.plot(time_vec, raw_plot_data,
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
                label_text = f"{label} ({total_shift*1000:+.3f}ms)"
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

                # Add legend to each subplot (combine primary and secondary axes)
                lines, labels_list = ax.get_legend_handles_labels()

                # Only add secondary axis legend entries to column 0 (Raw signals)
                # where the secondary axis is actually used
                if col == 0 and row in secondary_axes:
                    lines2, labels2 = secondary_axes[row].get_legend_handles_labels()
                    lines.extend(lines2)
                    labels_list.extend(labels2)

                if lines:
                    ax.legend(lines, labels_list, loc='upper right', fontsize=8, framealpha=0.9)
        
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
                        ax_final.axvline(x=guide_time, color=self.COLOR_ACCENT_GUIDE, linestyle=':',
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
                    fontsize=16, fontweight='bold', y=0.96)
        
        # Place compact summary in bottom center
        summary_text = processing_text + " | " + alignment_text
        fig.text(0.5, 0.02, summary_text, ha='center', va='bottom',
                bbox=dict(boxstyle='round', facecolor=self.COLOR_ACCENT_BG, alpha=0.7, pad=0.3),
                fontsize=9, transform=fig.transFigure)
        
        # Don't use tight_layout since we have custom spacing
        # plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Processing and alignment summary saved to {save_path}")
        
        # Don't show plots - only save them
        # plt.show()
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
                ax.axhline(y=conf_interval, color=self.COLOR_SIGNIFICANCE, linestyle='--',
                          linewidth=1, alpha=0.5, label=f'95% CI (±{conf_interval:.3f})')
                ax.axhline(y=-conf_interval, color=self.COLOR_SIGNIFICANCE, linestyle='--',
                          linewidth=1, alpha=0.5)

            # Find first zero crossing (decorrelation time)
            zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
            if len(zero_crossings) > 0:
                first_zero = zero_crossings[0]
                ax.axvline(x=x_values[first_zero], color=self.COLOR_SIGNIFICANCE, linestyle=':',
                          linewidth=1.5, alpha=0.7,
                          label=f'First zero: {x_values[first_zero]:.4f}')

            # Find where autocorrelation drops below 1/e (decay time)
            if normalize:
                decay_threshold = 1.0 / np.e
                below_threshold = np.where(autocorr < decay_threshold)[0]
                if len(below_threshold) > 0:
                    decay_idx = below_threshold[0]
                    ax.axvline(x=x_values[decay_idx], color=self.COLOR_SIGNIFICANCE, linestyle=':',
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

        # Don't show plots - only save them
        # plt.show()
        plt.close()

    def plot_cross_correlation(self, max_lag: Optional[int] = None,
                              max_shift_time: Optional[float] = None,
                              save_path: Optional[str] = None,
                              use_statsmodels: bool = True) -> None:
        """
        Plot cross-correlation functions using diagnostics stored during alignment.

        This method visualizes the ACTUAL cross-correlation calculations performed
        during auto_align_time_series(), ensuring consistency between computed shifts
        and displayed plots. No cross-correlation recomputation occurs here.

        Parameters:
        -----------
        max_lag : int, optional
            DEPRECATED - Included only for backward compatibility, not used
        max_shift_time : float, optional
            DEPRECATED - Included only for backward compatibility, not used
        save_path : str, optional
            Path to save the figure
        use_statsmodels : bool, default=True
            DEPRECATED - Included only for backward compatibility, not used
        """
        # Check if alignment diagnostics are available
        if not hasattr(self, 'alignment_diagnostics') or not self.alignment_diagnostics:
            print("No alignment diagnostics available. Run auto_align_time_series() first.")
            return

        n_pairs = len(self.alignment_diagnostics)

        if n_pairs == 0:
            print("No cross-correlation data available for plotting.")
            return

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

        # Plot cross-correlation for each pair using stored diagnostics
        pair_idx = 0
        for pair_key, diag_data in self.alignment_diagnostics.items():
            ax = axes[pair_idx]

            # Extract stored diagnostic data
            lags_time = diag_data['lags_time']
            correlation = diag_data['correlation']
            peak_lag_time = diag_data['peak_lag_time']
            max_correlation = diag_data['max_correlation']
            label1 = diag_data['label1']
            label2 = diag_data['label2']

            # Calculate correlation at zero lag
            zero_idx = np.argmin(np.abs(lags_time))
            corr_at_zero = correlation[zero_idx]

            # Plot cross-correlation
            ax.plot(lags_time, correlation, linewidth=2, color='#0173B2', alpha=0.8)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

            # Mark optimal lag with vertical line
            ax.axvline(x=peak_lag_time,
                      color=self.COLOR_SIGNIFICANCE, linestyle=':', linewidth=2, alpha=0.7,
                      label=f'Max corr at lag: {peak_lag_time:.6f}s')

            # Add text annotation
            ax.text(0.02, 0.98,
                   f"Max correlation: {max_correlation:.3f}\n"
                   f"At zero lag: {corr_at_zero:.3f}\n"
                   f"Applied shift: {peak_lag_time*1000:.3f}ms",
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=self.COLOR_ACCENT_BG, alpha=0.5))

            # Labels and title
            ax.set_xlabel('Lag (seconds)', fontsize=11)
            ax.set_ylabel('Cross-correlation', fontsize=11)
            ax.set_title(f'{label1} vs {label2}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_ylim([-1.1, 1.1])

            pair_idx += 1

        # Hide unused subplots
        for idx in range(pair_idx, len(axes)):
            axes[idx].axis('off')

        fig.suptitle('Cross-Correlation Functions (from alignment diagnostics)',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cross-correlation plot saved to {save_path}")

        # Don't show plots - only save them
        # plt.show()
        plt.close()

    def plot_outlier_removal(self, save_path: Optional[Path] = None) -> None:
        """
        Plot outlier removal visualization for each signal.

        Shows raw signal with outliers marked in orange 'x' markers,
        with each signal in its own subplot.

        Parameters:
        -----------
        save_path : Path, optional
            Path to save the figure
        """
        if not hasattr(self, 'processor') or not self.processor.outlier_masks:
            print("No outlier masks available. Skipping outlier removal plot.")
            return

        # Get signals that have outlier masks
        signals_with_outliers = [label for label in self.raw_data.keys()
                                 if label in self.processor.outlier_masks]

        if not signals_with_outliers:
            print("No signals with outlier masks. Skipping outlier removal plot.")
            return

        # Create figure with one subplot per signal
        n_signals = len(signals_with_outliers)
        fig, axes = plt.subplots(n_signals, 1, figsize=(12, 3 * n_signals))

        # Handle single subplot case
        if n_signals == 1:
            axes = [axes]

        colors = self.COLORS_OKABE_ITO

        for idx, label in enumerate(signals_with_outliers):
            ax = axes[idx]
            color = colors[idx % len(colors)]

            # Get data
            raw_data = self.raw_data[label]
            time_vec = self.original_time_vectors.get(label, np.arange(len(raw_data)))
            outlier_mask = self.processor.outlier_masks[label]

            # Plot raw signal
            ax.plot(time_vec, raw_data, label=f'{label} (raw)',
                   color=color, linewidth=1.5, alpha=0.8)

            # Mark outliers with orange 'x' markers
            if np.any(outlier_mask):
                outlier_indices = np.where(outlier_mask)[0]
                outlier_times = time_vec[outlier_indices]
                outlier_values = raw_data[outlier_indices]

                ax.scatter(outlier_times, outlier_values, marker='x',
                          color=self.COLOR_SIGNIFICANCE, s=50, linewidths=2,
                          label='Outliers removed', zorder=10)

                # Add text with outlier count
                n_outliers = len(outlier_indices)
                outlier_pct = 100 * n_outliers / len(raw_data)
                ax.text(0.02, 0.98, f'{n_outliers} outliers ({outlier_pct:.2f}%)',
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=self.COLOR_ACCENT_BG, alpha=0.8))
            else:
                # No outliers detected
                ax.text(0.02, 0.98, 'No outliers detected',
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=self.COLOR_ACCENT_BG, alpha=0.8))

            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3)

        fig.suptitle('Outlier Removal Summary', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Outlier removal plot saved to {save_path}")

        # Don't show plots - only save them
        # plt.show()
        plt.close()

    def plot_gradient_diagnostics(self, save_path: Optional[Path] = None) -> None:
        """
        Plot gradient/second derivative outlier detection diagnostics for all signals.

        Shows all signals side-by-side in columns with:
        - Row 1: Original signal with detected outliers
        - Row 2: Raw vs smoothed gradient/second derivative
        - Row 3: Absolute gradient/second derivative with MAD threshold

        Automatically detects whether gradient or second_derivative method was used
        and updates labels accordingly.

        Parameters:
        -----------
        save_path : Path, optional
            Path to save the figure
        """
        if not hasattr(self, 'processor') or len(self.processor.gradient_diagnostics) == 0:
            print("No gradient diagnostics available. Skipping gradient diagnostic plot.")
            return

        # Get all signal labels and their diagnostics
        signal_labels = list(self.processor.gradient_diagnostics.keys())
        n_signals = len(signal_labels)

        # Detect which method was used (check first signal)
        first_diag = self.processor.gradient_diagnostics[signal_labels[0]]
        method = first_diag.get('method', 'gradient')  # Default to 'gradient' for backward compatibility

        # Create figure with 3 rows x n_signals columns with shared x-axis
        fig, axes = plt.subplots(3, n_signals, figsize=(5 * n_signals, 12),
                                sharex='col', gridspec_kw={'hspace': 0.05, 'wspace': 0.3})

        # Handle single signal case
        if n_signals == 1:
            axes = axes.reshape(3, 1)

        # Process each signal
        for col_idx, label in enumerate(signal_labels):
            diag = self.processor.gradient_diagnostics[label]

            # Generate time vector for x-axis (use full duration)
            time_vec = np.arange(len(diag['data']))

            # Row 1: Original signal with cleaned overlay and outliers marked
            ax = axes[0, col_idx]
            # Plot original signal
            ax.plot(time_vec, diag['data'], label='Original', color='#0173B2',
                   linewidth=0.8, alpha=0.4, linestyle='--')

            # Overlay cleaned signal if available
            if 'data_cleaned' in diag:
                ax.plot(time_vec, diag['data_cleaned'], label='Cleaned',
                       color='#029E73', linewidth=1.2, alpha=0.9)

            # Mark outliers
            if np.any(diag['outliers']):
                outlier_indices = np.where(diag['outliers'])[0]
                outlier_values = diag['data'][outlier_indices]
                ax.scatter(outlier_indices, outlier_values, marker='x',
                          color=self.COLOR_SIGNIFICANCE, s=30, linewidths=1.5,
                          label='Removed', zorder=10, alpha=0.7)

                # Add outlier count
                n_outliers = len(outlier_indices)
                outlier_pct = 100 * n_outliers / len(diag['data'])
                ax.text(0.02, 0.98, f'{n_outliers} ({outlier_pct:.1f}%)',
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=self.COLOR_ACCENT_BG, alpha=0.8))

            ax.set_ylabel('Signal Value', fontsize=10, fontweight='bold')
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelbottom=False)  # Hide x-tick labels (shared x-axis)
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

            # Row 2: Raw vs smoothed derivative
            ax = axes[1, col_idx]
            derivative_label = 'Gradient' if method == 'gradient' else 'Second Derivative'
            ax.plot(time_vec, diag['gradient_raw'], label='Raw',
                   color='#DE8F05', linewidth=0.8, alpha=0.5)
            ax.plot(time_vec, diag['gradient_smoothed'], label='Smoothed',
                   color='#029E73', linewidth=1.0, alpha=0.8)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

            ax.set_ylabel(derivative_label, fontsize=10, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelbottom=False)  # Hide x-tick labels (shared x-axis)

            # Row 3: Absolute derivative with MAD threshold
            ax = axes[2, col_idx]
            abs_label = '|Gradient|' if method == 'gradient' else '|Second Derivative|'
            ax.plot(time_vec, diag['abs_gradient'], label=abs_label,
                   color='#CC78BC', linewidth=1.0, alpha=0.8)

            # Plot median and threshold
            ax.axhline(y=diag['median_grad'], color='#0173B2', linestyle='--',
                      linewidth=1.2, label=f'Median={diag["median_grad"]:.4f}')
            ax.axhline(y=diag['threshold_value'], color=self.COLOR_SIGNIFICANCE,
                      linestyle='-', linewidth=1.5,
                      label=f'Threshold={diag["threshold_value"]:.4f}')

            # Add MAD info and windowed detection note
            info_text = f'MAD={diag["mad_grad"]:.4f}\nT=med+{self.processor.config.outlier_threshold}×MAD'
            if self.processor.config.outlier_window > 0:
                info_text += '\n(Global threshold shown;\nLocal windows used)'
            ax.text(0.02, 0.98, info_text,
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=self.COLOR_ACCENT_BG, alpha=0.8))

            ax.set_xlabel('Sample Index', fontsize=10, fontweight='bold')
            ax.set_ylabel(abs_label, fontsize=10, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3)

        # Set title based on method used
        title = 'Gradient-Based Outlier Detection Diagnostics' if method == 'gradient' else 'Second Derivative Outlier Detection Diagnostics'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gradient diagnostic plot saved to {save_path}")

        # Don't show plots - only save them
        # plt.show()
        plt.close()

    def plot_alignment_diagnostics(self, save_path: Optional[Path] = None) -> None:
        """
        Plot alignment diagnostics for feature-based and mutual information methods.

        Creates comprehensive diagnostic visualizations showing:
        - Feature-based: Detected features on signal envelopes with match indicators
        - Mutual Info: MI scores across lag range with optimal lag marked

        Note: CCF diagnostics are plotted separately via plot_cross_correlation()
        """
        if not hasattr(self, 'alignment_diagnostics') or not self.alignment_diagnostics:
            print("No alignment diagnostics available. Run auto_align_time_series() first.")
            return

        # Filter for feature and mutual_info methods only
        feature_pairs = {k: v for k, v in self.alignment_diagnostics.items() if v.get('method') == 'feature'}
        mi_pairs = {k: v for k, v in self.alignment_diagnostics.items() if v.get('method') == 'mutual_info'}

        if not feature_pairs and not mi_pairs:
            print("No feature-based or mutual information diagnostics available.")
            return

        n_feature = len(feature_pairs)
        n_mi = len(mi_pairs)
        total_plots = n_feature + n_mi

        if total_plots == 0:
            return

        # Determine layout
        if total_plots == 1:
            fig, axes = plt.subplots(1, 1, figsize=(14, 6))
            axes = [axes]
        elif total_plots == 2:
            fig, axes = plt.subplots(1, 2, figsize=(24, 6))
        elif total_plots <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(24, 12))
            axes = axes.flatten()
        else:
            n_cols = 2
            n_rows = int(np.ceil(total_plots / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows))
            axes = axes.flatten()

        plot_idx = 0

        # Plot feature-based diagnostics
        for pair_key, diag in feature_pairs.items():
            ax = axes[plot_idx]

            features1 = diag.get('features1', np.array([]))
            features2 = diag.get('features2', np.array([]))
            optimal_lag = diag.get('optimal_lag', 0)
            quality_score = diag.get('quality_score', 0)
            n_matches = diag.get('n_matches', 0)
            lag_mad = diag.get('lag_mad', 0)
            label1 = diag.get('label1', '')
            label2 = diag.get('label2', '')

            # Get the actual signals from processed_data (for visualization)
            if label1 in self.processed_data and label2 in self.processed_data:
                # Create dummy time vectors for plotting (just indices)
                len1 = len(self.processed_data[label1])
                len2 = len(self.processed_data[label2])
                t1 = np.arange(len1)
                t2 = np.arange(len2)

                # Plot feature positions as vertical spans
                for feat_idx in features1:
                    if feat_idx < len1:
                        ax.axvline(x=feat_idx, color='#0173B2', alpha=0.3, linewidth=1, linestyle='--')

                # Plot shifted features2 positions
                for feat_idx in features2:
                    shifted_idx = feat_idx - optimal_lag
                    if 0 <= shifted_idx < len1:
                        ax.axvline(x=shifted_idx, color='#DE8F05', alpha=0.3, linewidth=1, linestyle='--')

                # Add scatter points for features
                if len(features1) > 0:
                    ax.scatter(features1, np.ones(len(features1)) * 0.1, marker='o', s=80,
                             color='#0173B2', alpha=0.7, label=f'{label1} features', zorder=10)
                if len(features2) > 0:
                    features2_shifted = features2 - optimal_lag
                    valid_mask = (features2_shifted >= 0) & (features2_shifted < len1)
                    if np.any(valid_mask):
                        ax.scatter(features2_shifted[valid_mask], np.ones(np.sum(valid_mask)) * 0.9,
                                 marker='s', s=80, color='#DE8F05', alpha=0.7,
                                 label=f'{label2} features (shifted)', zorder=10)

            # Add info text box
            info_text = (f"Optimal lag: {optimal_lag} samples\n"
                        f"Quality score: {quality_score:.3f}\n"
                        f"Matches: {n_matches}\n"
                        f"Lag MAD: {lag_mad:.1f}")

            ax.text(0.02, 0.98, info_text,
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=self.COLOR_ACCENT_BG, alpha=0.7))

            ax.set_ylim([0, 1])
            ax.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
            ax.set_ylabel('Feature Position', fontsize=11, fontweight='bold')
            ax.set_title(f'Feature-Based Alignment: {pair_key}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_yticks([])

            plot_idx += 1

        # Plot mutual information diagnostics
        for pair_key, diag in mi_pairs.items():
            ax = axes[plot_idx]

            lags = diag.get('lags', np.array([]))
            mi_scores = diag.get('mi_scores', np.array([]))
            optimal_lag = diag.get('optimal_lag', 0)
            optimal_mi = diag.get('optimal_mi', 0)
            quality_score = diag.get('quality_score', 0)
            label1 = diag.get('label1', '')
            label2 = diag.get('label2', '')

            if len(lags) > 0 and len(mi_scores) > 0:
                # Plot MI curve
                ax.plot(lags, mi_scores, linewidth=2, color='#029E73', alpha=0.8, label='MI Score')

                # Mark zero lag
                ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Zero lag')

                # Mark optimal lag
                ax.axvline(x=optimal_lag, color=self.COLOR_SIGNIFICANCE, linestyle=':',
                          linewidth=2.5, alpha=0.8, label=f'Optimal lag: {optimal_lag}')

                # Highlight optimal point
                ax.scatter([optimal_lag], [optimal_mi], marker='*', s=300,
                          color=self.COLOR_SIGNIFICANCE, edgecolors='black', linewidths=1,
                          zorder=10, label=f'Max MI: {optimal_mi:.3f}')

                # Add info text box
                mi_at_zero = mi_scores[np.argmin(np.abs(lags))] if len(lags) > 0 else 0
                info_text = (f"Optimal lag: {optimal_lag} samples\n"
                            f"Max MI: {optimal_mi:.3f}\n"
                            f"MI at zero lag: {mi_at_zero:.3f}\n"
                            f"Quality: {quality_score:.3f}")

                ax.text(0.02, 0.98, info_text,
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=self.COLOR_ACCENT_BG, alpha=0.7))

                ax.set_xlabel('Lag (samples)', fontsize=11, fontweight='bold')
                ax.set_ylabel('Mutual Information', fontsize=11, fontweight='bold')
                ax.set_title(f'Mutual Information Alignment: {pair_key}', fontsize=12, fontweight='bold')
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3)

            plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')

        fig.suptitle('Alignment Diagnostics (Feature-Based & Mutual Information)',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Alignment diagnostic plot saved to {save_path}")

        plt.close()
