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


def compute_channel_attribution(model, img_array, method='gradient_magnitude'):
    """
    Compute channel-wise attribution scores showing which input channels
    contribute most to the model's prediction.

    Uses gradient magnitude method: computes gradients w.r.t. input image
    and sums absolute gradients per channel across spatial dimensions.

    Args:
        model: Trained CNN model
        img_array: Input image array (shape: 1, H, W, C)
        method: Attribution method ('gradient_magnitude' is currently supported)

    Returns:
        dict: {
            'channel_scores': numpy array of shape (C,) with normalized scores (sum=1.0),
            'channel_ranks': numpy array of shape (C,) with rank indices (0=most important),
            'raw_scores': numpy array of shape (C,) with unnormalized gradient magnitudes
        }
    """
    num_channels = img_array.shape[-1]

    if method == 'gradient_magnitude':
        # Convert to tensor if needed
        img_tensor = tf.constant(img_array, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = model(img_tensor)
            loss = predictions[:, 0]  # Binary classification

        # Compute gradients w.r.t. input
        input_grads = tape.gradient(loss, img_tensor)  # Shape: (1, H, W, C)

        if input_grads is None:
            # If gradients couldn't be computed, return uniform attribution
            print("Warning: Could not compute gradients for channel attribution")
            uniform_scores = np.ones(num_channels) / num_channels
            return {
                'channel_scores': uniform_scores,
                'channel_ranks': np.arange(num_channels),
                'raw_scores': uniform_scores
            }

        # Per-channel attribution: sum of absolute gradients across spatial dimensions
        channel_scores = tf.reduce_sum(tf.abs(input_grads), axis=(0, 1, 2)).numpy()  # Shape: (C,)

        # Normalize to sum to 1.0
        if channel_scores.sum() > 0:
            channel_scores_normalized = channel_scores / channel_scores.sum()
        else:
            # If all scores are 0, use uniform distribution
            channel_scores_normalized = np.ones(num_channels) / num_channels

        # Compute ranks (0 = most important)
        channel_ranks = np.argsort(-channel_scores_normalized)

        return {
            'channel_scores': channel_scores_normalized,
            'channel_ranks': channel_ranks,
            'raw_scores': channel_scores
        }

    else:
        raise ValueError(f"Unknown attribution method: {method}")


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
        if len(original_img.shape) == 3:
            if original_img.shape[-1] == 1:
                # Single channel with explicit dimension
                display_img = original_img.squeeze()
            else:
                # Multi-channel: average across channels for visualization
                display_img = np.mean(original_img, axis=-1)
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
                                           output_dir, version, test_files=None, channel_labels=None):
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
        channel_labels: Optional list of channel names for multi-channel attribution analysis

    Returns:
        dict: Analysis results including saved images count and class statistics
    """
    import re

    print(f"🔥 Generating comprehensive Grad-CAM analysis...")

    # Determine if multi-channel
    num_channels = X_test.shape[-1] if len(X_test.shape) == 4 else 1
    is_multi_channel = num_channels > 1

    if is_multi_channel and channel_labels is None:
        # Auto-generate labels if not provided
        channel_labels = [f'Channel_{i+1}' for i in range(num_channels)]

    # Initialize channel attribution tracking
    channel_attribution_stats = None
    if is_multi_channel:
        channel_attribution_stats = {
            'per_sample': [],  # List of attribution dicts per sample
            'by_class': {0: [], 1: []},  # Split by true class
            'by_correctness': {'correct': [], 'incorrect': []},  # Split by prediction correctness
        }
        print(f"   Multi-channel analysis enabled for {num_channels} channels: {channel_labels}")

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

            # Compute channel attribution if multi-channel
            if is_multi_channel:
                attribution = compute_channel_attribution(model, X_test[i:i+1])

                # Store attribution with metadata
                attribution_record = {
                    'sample_idx': i,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'is_correct': is_correct,
                    'channel_scores': attribution['channel_scores'],
                    'dominant_channel': channel_labels[np.argmax(attribution['channel_scores'])],
                    'dominant_score': np.max(attribution['channel_scores'])
                }

                channel_attribution_stats['per_sample'].append(attribution_record)
                channel_attribution_stats['by_class'][true_label].append(attribution['channel_scores'])

                correctness_key = 'correct' if is_correct else 'incorrect'
                channel_attribution_stats['by_correctness'][correctness_key].append(attribution['channel_scores'])

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

    # Generate channel attribution summary if multi-channel
    if is_multi_channel and channel_attribution_stats['per_sample']:
        print(f"\n📊 Generating channel attribution analysis...")

        attribution_summary = generate_channel_attribution_summary(
            channel_attribution_stats,
            channel_labels,
            gradcam_dir,
            version
        )

        gradcam_results['channel_attribution'] = attribution_summary

    return gradcam_results


def generate_channel_attribution_summary(attribution_stats, channel_labels, output_dir, version):
    """
    Generate summary statistics and visualizations for channel attribution analysis.

    Args:
        attribution_stats: Dict with per_sample, by_class, by_correctness channel scores
        channel_labels: List of channel names
        output_dir: Output directory for saving results
        version: Version string for file naming

    Returns:
        dict: Summary statistics
    """
    import pandas as pd

    output_dir = Path(output_dir)
    num_channels = len(channel_labels)

    # Compute aggregate statistics
    all_scores = np.array([record['channel_scores'] for record in attribution_stats['per_sample']])  # Shape: (N, C)

    # Overall statistics
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)
    median_scores = np.median(all_scores, axis=0)

    # Statistics by class
    class_0_scores = np.array(attribution_stats['by_class'][0])  # Shape: (N0, C)
    class_1_scores = np.array(attribution_stats['by_class'][1])  # Shape: (N1, C)
    mean_scores_class_0 = np.mean(class_0_scores, axis=0) if len(class_0_scores) > 0 else np.zeros(num_channels)
    mean_scores_class_1 = np.mean(class_1_scores, axis=0) if len(class_1_scores) > 0 else np.zeros(num_channels)

    # Statistics by correctness
    correct_scores = np.array(attribution_stats['by_correctness']['correct'])
    incorrect_scores = np.array(attribution_stats['by_correctness']['incorrect'])
    mean_scores_correct = np.mean(correct_scores, axis=0) if len(correct_scores) > 0 else np.zeros(num_channels)
    mean_scores_incorrect = np.mean(incorrect_scores, axis=0) if len(incorrect_scores) > 0 else np.zeros(num_channels)

    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'channel': channel_labels,
        'mean_attribution': mean_scores,
        'std_attribution': std_scores,
        'median_attribution': median_scores,
        'class_0_mean': mean_scores_class_0,
        'class_1_mean': mean_scores_class_1,
        'correct_pred_mean': mean_scores_correct,
        'incorrect_pred_mean': mean_scores_incorrect,
        'rank': np.argsort(-mean_scores) + 1  # 1 = most important
    })

    # Sort by mean attribution (descending)
    summary_df = summary_df.sort_values('mean_attribution', ascending=False).reset_index(drop=True)

    # Save to CSV
    csv_path = output_dir / f'channel_attribution_summary_{version}.csv'
    summary_df.to_csv(csv_path, index=False, float_format='%.4f', encoding='utf-8')
    print(f"   Channel attribution summary saved to: {csv_path}")

    # Create bar chart visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Channel Attribution Analysis - {version}', fontsize=16, fontweight='bold')

    # Plot 1: Overall mean attribution
    ax1 = axes[0, 0]
    x_pos = np.arange(num_channels)
    bars = ax1.bar(x_pos, mean_scores, yerr=std_scores, capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Channel', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Attribution Score', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Channel Importance', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(channel_labels, rotation=45, ha='right')
    ax1.axhline(1.0 / num_channels, color='red', linestyle='--', linewidth=1, label='Uniform baseline')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mean_scores)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Attribution by class
    ax2 = axes[0, 1]
    x_pos = np.arange(num_channels)
    width = 0.35
    bars1 = ax2.bar(x_pos - width/2, mean_scores_class_0, width, label='Class 0 (No Porosity)', alpha=0.7, color='green')
    bars2 = ax2.bar(x_pos + width/2, mean_scores_class_1, width, label='Class 1 (Porosity)', alpha=0.7, color='red')
    ax2.set_xlabel('Channel', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Attribution Score', fontsize=12, fontweight='bold')
    ax2.set_title('Channel Importance by True Class', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(channel_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Attribution by prediction correctness
    ax3 = axes[1, 0]
    bars1 = ax3.bar(x_pos - width/2, mean_scores_correct, width, label='Correct Predictions', alpha=0.7, color='blue')
    bars2 = ax3.bar(x_pos + width/2, mean_scores_incorrect, width, label='Incorrect Predictions', alpha=0.7, color='orange')
    ax3.set_xlabel('Channel', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean Attribution Score', fontsize=12, fontweight='bold')
    ax3.set_title('Channel Importance by Prediction Correctness', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(channel_labels, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Distribution of dominant channels
    ax4 = axes[1, 1]
    dominant_channels = [record['dominant_channel'] for record in attribution_stats['per_sample']]
    channel_counts = pd.Series(dominant_channels).value_counts()

    # Reindex to ensure all channels appear
    channel_counts = channel_counts.reindex(channel_labels, fill_value=0)

    bars = ax4.bar(range(num_channels), channel_counts.values, alpha=0.7, color='purple')
    ax4.set_xlabel('Channel', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Count (# samples where dominant)', fontsize=12, fontweight='bold')
    ax4.set_title('Distribution of Dominant Channels', fontsize=13, fontweight='bold')
    ax4.set_xticks(range(num_channels))
    ax4.set_xticklabels(channel_labels, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for bar, count in zip(bars, channel_counts.values):
        if count > 0:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / f'channel_attribution_analysis_{version}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Channel attribution visualization saved to: {fig_path}")

    # Create per-sample attribution CSV
    per_sample_df = pd.DataFrame([
        {
            'sample_idx': rec['sample_idx'],
            'true_label': rec['true_label'],
            'pred_label': rec['pred_label'],
            'is_correct': rec['is_correct'],
            'dominant_channel': rec['dominant_channel'],
            'dominant_score': rec['dominant_score'],
            **{f'{ch}_attribution': score for ch, score in zip(channel_labels, rec['channel_scores'])}
        }
        for rec in attribution_stats['per_sample']
    ])

    per_sample_csv_path = output_dir / f'channel_attribution_per_sample_{version}.csv'
    per_sample_df.to_csv(per_sample_csv_path, index=False, float_format='%.4f', encoding='utf-8')
    print(f"   Per-sample attribution data saved to: {per_sample_csv_path}")

    # Return summary dict
    return {
        'num_samples': len(attribution_stats['per_sample']),
        'num_channels': num_channels,
        'channel_labels': channel_labels,
        'mean_attributions': mean_scores.tolist(),
        'channel_ranking': summary_df['channel'].tolist(),
        'dominant_channel': summary_df.iloc[0]['channel'],
        'dominant_channel_score': float(summary_df.iloc[0]['mean_attribution']),
        'output_files': {
            'summary_csv': str(csv_path),
            'visualization': str(fig_path),
            'per_sample_csv': str(per_sample_csv_path)
        }
    }
