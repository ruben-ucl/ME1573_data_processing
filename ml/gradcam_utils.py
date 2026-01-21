#!/usr/bin/env python3
"""
Grad-CAM Utilities Module

This module provides Grad-CAM (Gradient-weighted Class Activation Mapping) analysis
functions for CNN model interpretation and visualization.

Functions:
    - generate_gradcam_heatmap: Generate Grad-CAM heatmap for a single image
    - save_gradcam_image: Save Grad-CAM visualization with 3-panel layout
    - generate_comprehensive_gradcam_analysis: Generate comprehensive analysis with class averages

Author: AI Assistant
"""

# Ensure UTF-8 encoding for all I/O operations
import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model


def generate_gradcam_heatmap(gradcam_model, img_array, target_layer_idx=-2):
    """Generate Grad-CAM heatmap for a single image.

    Args:
        gradcam_model: Keras model with outputs [conv_layer_output, model_output]
        img_array: Input image array (batch of 1)
        target_layer_idx: Index for target layer output (default: -2)

    Returns:
        numpy.ndarray: Grad-CAM heatmap (2D array, normalized to 0-1)
        None: If error occurs during generation
    """
    try:
        with tf.GradientTape() as tape:
            conv_outputs, predictions = gradcam_model(img_array)
            loss = predictions[:, 0]  # Binary classification

        # Calculate gradients
        grads = tape.gradient(loss, conv_outputs)

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the feature maps by the gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()

    except Exception as e:
        print(f"Warning: Could not generate Grad-CAM for image: {e}")
        return None


def save_gradcam_image(original_img, heatmap, filepath, title=None, enhance_contrast=False):
    """Save Grad-CAM visualization combining original image and heatmap with frequency axis.

    Args:
        original_img: Original CWT image
        heatmap: Grad-CAM heatmap
        filepath: Output file path
        title: Optional title for the figure
        enhance_contrast: If True, enhance contrast by stretching values to full range
    """
    try:
        # Prepare original image for display
        if len(original_img.shape) == 3 and original_img.shape[-1] == 1:
            display_img = original_img.squeeze()
        else:
            display_img = original_img
        
        # Flip data for plotting (low frequencies at the bottom)
        display_img = np.flipud(display_img)
        heatmap_processed = np.flipud(heatmap.copy())

        # Enhance contrast if requested (for class averages)
        if enhance_contrast and heatmap_processed.max() > 0:
            # Stretch values to full 0-1 range
            heatmap_processed = (heatmap_processed - heatmap_processed.min()) / (heatmap_processed.max() - heatmap_processed.min())

        # Resize heatmap to match image dimensions
        heatmap_resized = cv2.resize(heatmap_processed, (display_img.shape[1], display_img.shape[0]))

        # Get image dimensions
        img_height = display_img.shape[0]
        img_width = display_img.shape[1]

        # CWT frequency range: 1 kHz to 50 kHz (logarithmic)
        # Create logarithmic frequency labels
        freq_min_khz = 1    # 1 kHz
        freq_max_khz = 50   # 50 kHz

        # Logarithmic tick positions (powers of 2: 1, 2, 4, 8, 16, 32, 50)
        freq_ticks_khz = [1, 2, 4, 8, 16, 32, 50]

        # Convert to pixel positions (inverted - high freq at top)
        # Frequency increases from bottom to top, so we need to invert
        freq_ticks_pos = []
        for f in freq_ticks_khz:
            # Logarithmic mapping: log scale from 1 to 50
            log_pos = (np.log(f) - np.log(freq_min_khz)) / (np.log(freq_max_khz) - np.log(freq_min_khz))
            # imshow uses origin at top-left, so invert the position
            # pixel_pos = img_height * (1 - log_pos)
            pixel_pos = img_height * log_pos
            freq_ticks_pos.append(pixel_pos)

        # Time axis: 0 to 1 ms
        # Create time tick positions and labels
        time_ticks_ms = [0, 0.25, 0.5, 0.75, 1.0]
        time_ticks_pos = [t * img_width for t in time_ticks_ms]

        # Calculate figure size to maintain aspect ratio
        # Image is img_width x img_height (100 x 256)
        # Each subplot should maintain this aspect ratio
        aspect_ratio = img_height / img_width  # 256/100 = 2.56
        subplot_width = 3.5  # Width per subplot in inches
        subplot_height = subplot_width * aspect_ratio
        total_width = subplot_width * 3 + 2  # 3 subplots plus spacing

        # Create visualization with narrower figure
        fig, axes = plt.subplots(1, 3, figsize=(total_width, subplot_height + 1))

        # Create coordinate arrays for pcolormesh
        t_ax = np.linspace(0, 1, img_width + 1)  # Time edges in ms
        f_ax = np.geomspace(freq_min_khz, freq_max_khz, img_height + 1)  # Frequency edges in kHz

        # Original image with frequency and time axes
        pcm0 = axes[0].pcolormesh(t_ax, f_ax, display_img, cmap='gray', shading='flat')
        axes[0].set_title('Original CWT Image')
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('Frequency (kHz)')
        axes[0].set_yscale('log', base=2)
        axes[0].set_ylim(freq_min_khz, freq_max_khz)
        axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        axes[0].set_yticks(freq_ticks_khz)
        axes[0].set_xticks(time_ticks_ms)
        axes[0].grid(True, alpha=0.3, linestyle='--')

        # Heatmap with frequency and time axes
        pcm1 = axes[1].pcolormesh(t_ax, f_ax, heatmap_resized, cmap='viridis', shading='flat')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Frequency (kHz)')
        axes[1].set_yscale('log', base=2)
        axes[1].set_ylim(freq_min_khz, freq_max_khz)
        axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        axes[1].set_yticks(freq_ticks_khz)
        axes[1].set_xticks(time_ticks_ms)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(pcm1, ax=axes[1], label='Activation')

        # Overlay with frequency and time axes
        # For overlay, use imshow with origin='lower' and then overlay the heatmap
        axes[2].pcolormesh(t_ax, f_ax, display_img, cmap='gray', alpha=0.7, shading='flat')
        axes[2].pcolormesh(t_ax, f_ax, heatmap_resized, cmap='viridis', alpha=0.3, shading='flat')
        axes[2].set_title('Overlay')
        axes[2].set_xlabel('Time (ms)')
        axes[2].set_ylabel('Frequency (kHz)')
        axes[2].set_yscale('log', base=2)
        axes[2].set_ylim(freq_min_khz, freq_max_khz)
        axes[2].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        axes[2].set_yticks(freq_ticks_khz)
        axes[2].set_xticks(time_ticks_ms)
        axes[2].grid(True, alpha=0.3, linestyle='--')

        if title:
            fig.suptitle(title, fontsize=12, y=1.02)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Warning: Could not save Grad-CAM image to {filepath}: {e}")


def _create_frequency_curve_visualization(avg_heatmap_0_enhanced, avg_heatmap_1_enhanced,
                                         diff_heatmap, freq_ticks_pos, freq_ticks_khz,
                                         summary_dir, img_height, img_width,
                                         title_fontsize=10, label_fontsize=9, tick_fontsize=8):
    """Create frequency curve visualization with Class 0 | Frequency Curve | Class 1 layout.

    Args:
        avg_heatmap_0_enhanced: Enhanced average heatmap for class 0
        avg_heatmap_1_enhanced: Enhanced average heatmap for class 1
        diff_heatmap: Difference heatmap (class 1 - class 0)
        freq_ticks_pos: Frequency tick positions in pixels
        freq_ticks_khz: Frequency tick labels in kHz
        summary_dir: Directory to save the visualization
        img_height: Image height in pixels
        img_width: Image width in pixels
        title_fontsize: Font size for titles
        label_fontsize: Font size for axis labels
        tick_fontsize: Font size for tick labels
    """
    # Compute frequency-averaged difference curve
    # Average along rows (axis=1) to get average difference per frequency
    freq_diff = np.mean(diff_heatmap, axis=1)  # Shape: (img_height,)

    # CWT frequency range
    freq_min_khz = 1    # 1 kHz
    freq_max_khz = 50   # 50 kHz

    # Time axis: 0 to 1 ms
    time_ticks_heatmap = [0, 0.5, 1.0]

    # Saturation threshold for heatmaps
    saturation_threshold = 0.8

    # Calculate figure size - all panels same width for clean layout
    # Aspect ratio of CWT images: 256/100 = 2.56
    aspect_ratio = img_height / img_width
    panel_width = 3.5  # Width per panel in inches (all equal)
    subplot_height = panel_width * aspect_ratio  # ~8.96 inches
    total_width = panel_width * 3 + 1  # 3 panels plus minimal spacing

    # Font sizes scaled for half A4 page width display (8pt target)
    # Half A4 width ≈ 4.13 inches, our figure ≈ 11.5 inches, scale factor ≈ 2.8
    title_fontsize_scaled = 22
    label_fontsize_scaled = 20
    tick_fontsize_scaled = 18

    # Create figure with GridSpec for equal widths, minimal spacing
    fig = plt.figure(figsize=(total_width, subplot_height + 1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.12)

    # Create axes with shared y-axis
    ax0 = fig.add_subplot(gs[0])
    ax_curve = fig.add_subplot(gs[1], sharey=ax0)
    ax1 = fig.add_subplot(gs[2], sharey=ax0)

    # Create coordinate arrays for pcolormesh
    t_ax = np.linspace(0, 1, img_width + 1)  # Time edges in ms
    f_ax = np.geomspace(freq_min_khz, freq_max_khz, img_height + 1)  # Frequency edges in kHz

    # Flip data for plotting (low frequencies at the bottom)
    avg_heatmap_0_flipped = np.flipud(avg_heatmap_0_enhanced)
    avg_heatmap_1_flipped = np.flipud(avg_heatmap_1_enhanced)
    diff_heatmap_flipped = np.flipud(diff_heatmap)

    # Left panel: Class 0 average
    pcm0 = ax0.pcolormesh(t_ax, f_ax, avg_heatmap_0_flipped, cmap='viridis',
                          vmin=0, vmax=saturation_threshold, shading='flat')
    ax0.set_title('No pore', fontsize=title_fontsize_scaled, pad=10)
    ax0.set_xlabel('Time (ms)', fontsize=label_fontsize_scaled)
    ax0.set_ylabel('Frequency (kHz)', fontsize=label_fontsize_scaled)
    ax0.set_yscale('log', base=2)
    ax0.set_ylim(freq_min_khz, freq_max_khz)
    ax0.set_yticks(freq_ticks_khz)
    ax0.set_xticks(time_ticks_heatmap)
    ax0.tick_params(axis='x', labelsize=tick_fontsize_scaled)
    ax0.tick_params(axis='y', labelsize=tick_fontsize_scaled)
    ax0.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)

    # Middle panel: Frequency difference curve
    # Create frequency centers for plotting
    freq_centers = np.sqrt(f_ax[:-1] * f_ax[1:])  # Geometric mean of edges
    ax_curve.plot(freq_diff, freq_centers[::-1], 'k-', linewidth=2.0)

    # Add zero line (black dashed, centered)
    ax_curve.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

    # Center x-axis around zero with 10% margin
    max_abs_diff = np.max(np.abs(freq_diff))
    if max_abs_diff > 0:
        margin = 0.15 * max_abs_diff
        ax_curve.set_xlim(-max_abs_diff - margin, max_abs_diff + margin)

    # Set axis properties
    ax_curve.set_xlabel('Difference', fontsize=label_fontsize_scaled)
    ax_curve.set_yscale('log', base=2)
    ax_curve.set_ylim(freq_min_khz, freq_max_khz)
    ax_curve.tick_params(axis='x', labelsize=tick_fontsize_scaled)
    ax_curve.tick_params(axis='y', labelleft=False)  # Hide y-labels on middle panel
    ax_curve.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y', zorder=0)
    ax_curve.set_title('Difference', fontsize=title_fontsize_scaled, pad=10)

    # Right panel: Class 1 average
    pcm1 = ax1.pcolormesh(t_ax, f_ax, avg_heatmap_1_flipped, cmap='viridis',
                          vmin=0, vmax=saturation_threshold, shading='flat')
    ax1.set_title('Pore', fontsize=title_fontsize_scaled, pad=10)
    ax1.set_xlabel('Time (ms)', fontsize=label_fontsize_scaled)
    ax1.set_ylabel('')  # No y-label on right panel
    ax1.set_yscale('log', base=2)
    ax1.set_ylim(freq_min_khz, freq_max_khz)
    ax1.set_xticks(time_ticks_heatmap)
    ax1.tick_params(axis='x', labelsize=tick_fontsize_scaled)
    ax1.tick_params(axis='y', labelleft=False)  # Hide y-tick labels on right panel
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)

    # Fix y-axis ticks for shared log axes (must be done AFTER all set_yscale calls)
    # Log formatters only label decade ticks by default, so provide explicit labels
    # This is the matplotlib-recommended approach for non-decade ticks on log scales
    ax0.set_yticks(freq_ticks_khz, labels=['1', '2', '4', '8', '16', '32', '50'])

    # Add shared colorbar for heatmaps (thicker with more spacing)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="8%", pad=0.08)
    cbar = plt.colorbar(pcm1, cax=cax, extend='max')
    cbar.set_label('Activation', fontsize=label_fontsize_scaled)
    cbar.ax.tick_params(labelsize=tick_fontsize_scaled)

    plt.savefig(summary_dir / 'class_difference_analysis_with_frequency_curve.png',
                dpi=600, bbox_inches='tight')
    plt.close()


def generate_comprehensive_gradcam_analysis(model, X_test, y_test, y_pred, y_proba, threshold,
                                           output_dir, version, test_files=None):
    """Generate comprehensive Grad-CAM analysis with class-specific folders and averages.

    Args:
        model: Trained CNN model
        X_test: Test images
        y_test: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        threshold: Confidence threshold
        output_dir: Output directory path
        version: Version string for organizing outputs
        test_files: Optional list of test file paths for naming

    Returns:
        dict: Analysis results including saved images count and class statistics
    """
    import re

    print(f"🔥 Generating comprehensive Grad-CAM analysis...")

    gradcam_dir = Path(output_dir) / f'gradcam_analysis_{version}'

    # Create class-specific directories
    class_dirs = {}
    for class_label in [0, 1]:
        class_dirs[class_label] = {
            'correct': gradcam_dir / f'class_{class_label}' / 'correct_predictions',
            'incorrect': gradcam_dir / f'class_{class_label}' / 'incorrect_predictions',
            'all': gradcam_dir / f'class_{class_label}' / 'all_samples'
        }
        for dir_path in class_dirs[class_label].values():
            dir_path.mkdir(parents=True, exist_ok=True)

    # Create summary directory
    summary_dir = gradcam_dir / 'class_averages'
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Find the best layer for Grad-CAM (last convolutional layer)
    conv_layers = [layer for layer in model.layers if 'conv' in layer.name.lower()]
    if not conv_layers:
        print("Warning: No convolutional layers found for Grad-CAM")
        return None

    target_layer = conv_layers[-1]
    print(f"Using layer '{target_layer.name}' for Grad-CAM analysis")

    # Create Grad-CAM model
    gradcam_model = Model(inputs=model.input, outputs=[target_layer.output, model.output])

    # Store heatmaps for averaging
    class_heatmaps = {0: [], 1: []}
    gradcam_results = {
        'target_layer': target_layer.name,
        'threshold': threshold,
        'total_images': len(X_test),
        'class_analysis': {0: {'correct': 0, 'incorrect': 0}, 1: {'correct': 0, 'incorrect': 0}},
        'saved_images': 0
    }

    for i in range(len(X_test)):
        true_label = int(y_test[i])
        pred_label = int(y_pred[i])
        confidence = float(y_proba[i])
        is_correct = (true_label == pred_label)

        # Generate Grad-CAM heatmap
        heatmap = generate_gradcam_heatmap(
            gradcam_model, X_test[i:i+1], target_layer_idx=-2
        )

        if heatmap is not None:
            # Store for class averaging
            class_heatmaps[true_label].append(heatmap)

            # Create filename with track ID and window if available
            prediction_type = 'correct' if is_correct else 'incorrect'

            if test_files is not None and i < len(test_files):
                # Extract track ID and window from filename
                # Example: 0105_01_0.2-1.2ms.png -> track_id = "0105_01", window = "0.2-1.2ms"
                original_filename = Path(test_files[i]).name
                parts = original_filename.split('_')

                if len(parts) >= 2:
                    track_id = f"{parts[0]}_{parts[1]}"  # e.g., "0105_01"
                    # Extract window info (everything after second underscore, before extension)
                    window_info = '_'.join(parts[2:]).replace('.png', '')  # e.g., "0.2-1.2ms"
                    filename = f'{track_id}_{window_info}_true_{true_label}_pred_{pred_label}_conf_{confidence:.3f}.png'
                else:
                    # Fallback if filename doesn't match expected pattern
                    filename = f'{original_filename.replace(".png", "")}_true_{true_label}_pred_{pred_label}_conf_{confidence:.3f}.png'
            else:
                # Fallback if no test_files provided
                filename = f'sample_{i:04d}_true_{true_label}_pred_{pred_label}_conf_{confidence:.3f}.png'

            # Save to class-specific directories
            save_gradcam_image(
                X_test[i], heatmap,
                class_dirs[true_label][prediction_type] / filename,
                title=f'True: {true_label}, Pred: {pred_label}, Conf: {confidence:.3f}'
            )

            save_gradcam_image(
                X_test[i], heatmap,
                class_dirs[true_label]['all'] / filename,
                title=f'True: {true_label}, Pred: {pred_label}, Conf: {confidence:.3f}'
            )

            gradcam_results['saved_images'] += 1
            gradcam_results['class_analysis'][true_label][prediction_type] += 1

    # Generate class average heatmaps
    for class_label in [0, 1]:
        if class_heatmaps[class_label]:
            avg_heatmap = np.mean(class_heatmaps[class_label], axis=0)

            # Create a representative image (mean of class images)
            class_indices = np.where(y_test == class_label)[0]
            representative_img = np.mean(X_test[class_indices], axis=0)

            save_gradcam_image(
                representative_img, avg_heatmap,
                summary_dir / f'class_{class_label}_average_gradcam.png',
                title=f'Class {class_label} Average Grad-CAM (n={len(class_heatmaps[class_label])})',
                enhance_contrast=True  # Enhance contrast for class averages
            )

    # Generate difference heatmaps (Class 1 - Class 0)
    if class_heatmaps[0] and class_heatmaps[1]:
        avg_heatmap_0 = np.mean(class_heatmaps[0], axis=0)
        avg_heatmap_1 = np.mean(class_heatmaps[1], axis=0)

        # Enhance contrast by stretching to full range (for class averages)
        avg_heatmap_0_enhanced = avg_heatmap_0.copy()
        avg_heatmap_1_enhanced = avg_heatmap_1.copy()

        if avg_heatmap_0_enhanced.max() > 0:
            avg_heatmap_0_enhanced = (avg_heatmap_0_enhanced - avg_heatmap_0_enhanced.min()) / (avg_heatmap_0_enhanced.max() - avg_heatmap_0_enhanced.min())
        if avg_heatmap_1_enhanced.max() > 0:
            avg_heatmap_1_enhanced = (avg_heatmap_1_enhanced - avg_heatmap_1_enhanced.min()) / (avg_heatmap_1_enhanced.max() - avg_heatmap_1_enhanced.min())

        diff_heatmap = avg_heatmap_1_enhanced - avg_heatmap_0_enhanced

        # Get image dimensions
        img_height = avg_heatmap_0.shape[0]
        img_width = avg_heatmap_0.shape[1]

        # CWT frequency range: 1 kHz to 50 kHz (logarithmic)
        freq_min_khz = 1
        freq_max_khz = 50
        freq_ticks_khz = [1, 2, 4, 8, 16, 32, 50]

        freq_ticks_pos = []
        for f in freq_ticks_khz:
            log_pos = (np.log(f) - np.log(freq_min_khz)) / (np.log(freq_max_khz) - np.log(freq_min_khz))
            pixel_pos = img_height * log_pos
            freq_ticks_pos.append(pixel_pos)

        # Time axis: 0 to 1 ms
        time_ticks_ms = [0, 0.25, 0.5, 0.75, 1.0]

        # Calculate figure size to match individual class average figures
        # Aspect ratio of CWT images: 256/100 = 2.56
        aspect_ratio = img_height / img_width  # 256/100 = 2.56
        subplot_width = 3.5  # Width per subplot in inches (same as individual figures)
        subplot_height = subplot_width * aspect_ratio  # ~8.96 inches
        total_width = subplot_width * 3 + 2  # 3 subplots plus spacing

        # Font sizes for publication-quality display
        title_fontsize = 10
        label_fontsize = 9
        tick_fontsize = 8

        # ============================================================
        # ORIGINAL VISUALIZATION: Class 0 | Class 1 | Difference Map
        # ============================================================

        # Create difference visualization with frequency and time axes
        fig, axes = plt.subplots(1, 3, figsize=(total_width, subplot_height + 1))

        # Saturate values above 0.8 for better contrast
        saturation_threshold = 0.8

        # Create coordinate arrays for pcolormesh
        t_ax = np.linspace(0, 1, img_width + 1)  # Time edges in ms
        f_ax = np.geomspace(freq_min_khz, freq_max_khz, img_height + 1)  # Frequency edges in kHz

        # Flip data for plotting (low frequencies at the bottom)
        avg_heatmap_0_flipped = np.flipud(avg_heatmap_0_enhanced)
        avg_heatmap_1_flipped = np.flipud(avg_heatmap_1_enhanced)
        diff_heatmap_flipped = np.flipud(diff_heatmap)

        # Class 0 average with saturation
        pcm0 = axes[0].pcolormesh(t_ax, f_ax, avg_heatmap_0_flipped, cmap='viridis',
                                  vmin=0, vmax=saturation_threshold, shading='flat')
        axes[0].set_title('Class 0 (No Porosity)', fontsize=title_fontsize, fontweight='bold', pad=3)
        axes[0].set_xlabel('Time (ms)', fontsize=label_fontsize)
        axes[0].set_ylabel('Frequency (kHz)', fontsize=label_fontsize)
        axes[0].set_yscale('log', base=2)
        axes[0].set_ylim(freq_min_khz, freq_max_khz)
        axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        axes[0].set_yticks(freq_ticks_khz)
        axes[0].set_xticks(time_ticks_ms)
        axes[0].tick_params(axis='x', labelsize=tick_fontsize)
        axes[0].tick_params(axis='y', labelsize=tick_fontsize)
        axes[0].grid(True, alpha=0.2, linestyle='--', linewidth=0.3)

        # Class 1 average with saturation
        pcm1 = axes[1].pcolormesh(t_ax, f_ax, avg_heatmap_1_flipped, cmap='viridis',
                                  vmin=0, vmax=saturation_threshold, shading='flat')
        axes[1].set_title('Class 1 (Porosity)', fontsize=title_fontsize, fontweight='bold', pad=3)
        axes[1].set_xlabel('Time (ms)', fontsize=label_fontsize)
        axes[1].set_ylabel('Frequency (kHz)', fontsize=label_fontsize)
        axes[1].set_yscale('log', base=2)
        axes[1].set_ylim(freq_min_khz, freq_max_khz)
        axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        axes[1].set_yticks(freq_ticks_khz)
        axes[1].set_xticks(time_ticks_ms)
        axes[1].tick_params(axis='x', labelsize=tick_fontsize)
        axes[1].tick_params(axis='y', labelsize=tick_fontsize)
        axes[1].grid(True, alpha=0.2, linestyle='--', linewidth=0.3)

        # Shared colorbar for class 0 and 1 with extend arrow
        # Position: right of axes[1]
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.03)
        cbar_shared = plt.colorbar(pcm1, cax=cax, extend='max')
        cbar_shared.set_label('Activation', fontsize=label_fontsize)
        cbar_shared.ax.tick_params(labelsize=tick_fontsize)

        # Difference with separate colorbar
        max_abs_diff = np.max(np.abs(diff_heatmap))
        pcm_diff = axes[2].pcolormesh(t_ax, f_ax, diff_heatmap_flipped, cmap='RdBu_r',
                                      vmin=-max_abs_diff, vmax=max_abs_diff, shading='flat')
        axes[2].set_title('Difference (1 - 0)', fontsize=title_fontsize, fontweight='bold', pad=3)
        axes[2].set_xlabel('Time (ms)', fontsize=label_fontsize)
        axes[2].set_ylabel('Frequency (kHz)', fontsize=label_fontsize)
        axes[2].set_yscale('log', base=2)
        axes[2].set_ylim(freq_min_khz, freq_max_khz)
        axes[2].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        axes[2].set_yticks(freq_ticks_khz)
        axes[2].set_xticks(time_ticks_ms)
        axes[2].tick_params(axis='x', labelsize=tick_fontsize)
        axes[2].tick_params(axis='y', labelsize=tick_fontsize)
        axes[2].grid(True, alpha=0.2, linestyle='--', linewidth=0.3)

        # Colorbar for difference
        divider_diff = make_axes_locatable(axes[2])
        cax_diff = divider_diff.append_axes("right", size="5%", pad=0.03)
        cbar_diff = plt.colorbar(pcm_diff, cax=cax_diff)
        cbar_diff.set_label('Difference', fontsize=label_fontsize)
        cbar_diff.ax.tick_params(labelsize=tick_fontsize)

        plt.tight_layout()
        plt.savefig(summary_dir / 'class_difference_analysis.png', dpi=600, bbox_inches='tight')
        plt.close()

        # ====================================================================
        # NEW VISUALIZATION: Class 0 | Frequency Curve | Class 1
        # ====================================================================

        _create_frequency_curve_visualization(
            avg_heatmap_0_enhanced, avg_heatmap_1_enhanced, diff_heatmap,
            freq_ticks_pos, freq_ticks_khz, summary_dir, img_height, img_width,
            title_fontsize, label_fontsize, tick_fontsize
        )

    print(f"   Grad-CAM images saved to: {gradcam_dir}")
    print(f"   Generated {gradcam_results['saved_images']} individual heatmaps")
    print(f"   Class 0: {gradcam_results['class_analysis'][0]} samples")
    print(f"   Class 1: {gradcam_results['class_analysis'][1]} samples")

    return gradcam_results
