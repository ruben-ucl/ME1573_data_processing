"""
Measure keyhole geometry from binary TIFF stack.

Reads a stack of binary TIFF images where white (255) represents the keyhole,
measures geometric properties frame by frame, and outputs measurements to CSV.

Author: Rubén Lambert-Garcia
Version: v1.0
"""

import os
import sys
import glob
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
from mpl_toolkits.axes_grid1 import make_axes_locatable

print = functools.partial(print, flush=True)

# =============================================================================
# CONFIGURATION - Edit these parameters
# =============================================================================

# Input/output paths
INPUT_TIFF_DIR = r"F:\sim_segmented_300W_800mm_s\SLM_Al10SiMg_1st_layer_4mm_300W_800mms_with_phases\animations\centre_slice_alpha_metalVapour\cropped_binary"
OUTPUT_DIR = r"F:\sim_segmented_300W_800mm_s\SLM_Al10SiMg_1st_layer_4mm_300W_800mms_with_phases\extracted_data"

# Pixel resolution (micrometers per pixel)
UM_PER_PIXEL = 1.25

# Substrate surface location in MICROMETERS (measured upward from bottom of image)
# This acts as datum for depth measurements (keyhole depth measured downward from here)
# Any keyhole region above this surface is cropped/ignored in measurements
SUBSTRATE_SURFACE_FROM_BOTTOM_UM = 315  # 237 px * 1.25 um/px

# Frame rate for timestamp calculation (fps) - set to None if timestamps not needed
CAPTURE_FRAMERATE = 100000

# Visualization options
PLOT_KEYHOLES = False  # Set True to visualize individual keyhole measurements
SAVE_MODE = 'preview'  # 'save' or 'preview'

# =============================================================================


def main():
    # Get list of TIFF files
    tiff_pattern = str(Path(INPUT_TIFF_DIR, '*.tif*'))
    tiff_files = sorted(glob.glob(tiff_pattern))

    if not tiff_files:
        print(f"ERROR: No TIFF files found in {INPUT_TIFF_DIR}")
        print(f"Pattern used: {tiff_pattern}")
        sys.exit(1)

    print(f"Found {len(tiff_files)} TIFF files in {INPUT_TIFF_DIR}")

    # Get image dimensions from first file
    first_im = np.array(Image.open(tiff_files[0]))
    im_height, im_width = first_im.shape[:2]
    print(f"Image dimensions: {im_width} x {im_height} pixels")

    # Convert substrate surface from um to pixels, then to row index (top-down)
    substrate_surface_from_bottom_px = int(round(SUBSTRATE_SURFACE_FROM_BOTTOM_UM / UM_PER_PIXEL))
    substrate_surface_row = im_height - substrate_surface_from_bottom_px
    print(f"Pixel resolution: {UM_PER_PIXEL} um/px")
    print(f"Substrate surface: {SUBSTRATE_SURFACE_FROM_BOTTOM_UM} um from bottom = row {substrate_surface_row}")

    # Preview mode: show z-projection with substrate line to verify settings
    if SAVE_MODE == 'preview':
        print("\nGenerating z-projection preview...")
        preview_z_projection(tiff_files, substrate_surface_row, im_height, im_width)

        print("\nGenerating measurement preview on example frame...")
        preview_measurement_steps(tiff_files, substrate_surface_row, im_height, im_width)

    # Initialize measurement storage
    keyhole_data = {
        'frame': [],
        'time_ms': [],
        'area_um2': [],
        'max_depth_um': [],
        'max_width_um': [],
        'depth_at_max_width_um': [],
        'aperture_width_um': [],
        'centroid_x_um': [],
        'centroid_y_um': [],
    }

    # Process each frame
    n_frames = len(tiff_files)
    for i, tiff_path in enumerate(tiff_files):
        frame_num = i + 1

        # Load and binarize image
        im = np.array(Image.open(tiff_path))
        if im.ndim == 3:
            im = im[:, :, 0]  # Take first channel if RGB

        # Ensure binary (threshold at 127)
        im_binary = (im > 127).astype(np.uint8)

        # Calculate timestamp if framerate specified
        if CAPTURE_FRAMERATE:
            time_ms = i * 1000 / CAPTURE_FRAMERATE
        else:
            time_ms = None

        keyhole_data['frame'].append(frame_num)
        keyhole_data['time_ms'].append(time_ms)

        # Label connected regions (note: measure.label only labels non-zero pixels,
        # so background is NOT included as a region)
        labeled = measure.label(im_binary)
        props = measure.regionprops(labeled)

        if len(props) == 0:
            # No regions found (empty frame)
            keyhole_data['area_um2'].append(0)
            keyhole_data['max_depth_um'].append(0)
            keyhole_data['max_width_um'].append(0)
            keyhole_data['depth_at_max_width_um'].append(0)
            keyhole_data['aperture_width_um'].append(0)
            keyhole_data['centroid_x_um'].append(None)
            keyhole_data['centroid_y_um'].append(None)
            print_progress(i + 1, n_frames, 'Measuring keyholes', 'Complete')
            continue

        # Sort regions by area (descending) - largest region is keyhole,
        # smaller regions are noise
        props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
        keyhole_props = props_sorted[0]  # Largest region is keyhole

        # Get keyhole mask
        keyhole_mask = (labeled == keyhole_props.label).astype(np.uint8) * 255

        # Crop keyhole above substrate surface (set to 0)
        # Only measure the portion below the substrate datum
        keyhole_mask[:substrate_surface_row, :] = 0

        # Recalculate area after cropping (only count pixels below substrate)
        area_px2 = np.sum(keyhole_mask == 255)
        area_um2 = area_px2 * (UM_PER_PIXEL ** 2)

        # Check if any keyhole remains after cropping
        if area_px2 == 0:
            # Keyhole was entirely above substrate
            keyhole_data['area_um2'].append(0)
            keyhole_data['max_depth_um'].append(0)
            keyhole_data['max_width_um'].append(0)
            keyhole_data['depth_at_max_width_um'].append(0)
            keyhole_data['aperture_width_um'].append(0)
            keyhole_data['centroid_x_um'].append(None)
            keyhole_data['centroid_y_um'].append(None)
            print_progress(i + 1, n_frames, 'Measuring keyholes', 'Complete')
            continue

        # Centroid of cropped keyhole (relative to substrate datum)
        cropped_props = measure.regionprops(keyhole_mask)
        if cropped_props:
            centroid_y, centroid_x = cropped_props[0].centroid
            centroid_x_um = centroid_x * UM_PER_PIXEL
            centroid_y_um = (centroid_y - substrate_surface_row) * UM_PER_PIXEL  # Depth below datum
        else:
            centroid_x_um = None
            centroid_y_um = None

        # Get detailed measurements (keyhole mask already cropped above substrate)
        measurements = get_keyhole_measurements(
            keyhole_mask,
            substrate_surface_row,
            frame_num if PLOT_KEYHOLES else None
        )

        keyhole_data['area_um2'].append(area_um2)
        keyhole_data['max_depth_um'].append(measurements['max_depth_um'])
        keyhole_data['max_width_um'].append(measurements['max_width_um'])
        keyhole_data['depth_at_max_width_um'].append(measurements['depth_at_max_width_um'])
        keyhole_data['aperture_width_um'].append(measurements['aperture_width_um'])
        keyhole_data['centroid_x_um'].append(centroid_x_um)
        keyhole_data['centroid_y_um'].append(centroid_y_um)

        if not PLOT_KEYHOLES:
            print_progress(i + 1, n_frames, 'Measuring keyholes', 'Complete')

    # Convert to DataFrame
    df = pd.DataFrame(keyhole_data)

    # Remove time column if not used
    if CAPTURE_FRAMERATE is None:
        df = df.drop(columns=['time_ms'])

    # Generate summary statistics
    summary = generate_summary_stats(df)

    # Save or preview
    if SAVE_MODE == 'save':
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full measurements
        measurements_path = output_dir / 'keyhole_measurements.csv'
        df.to_csv(measurements_path, index=False)
        print(f"\nSaved measurements to: {measurements_path}")

        # Save summary
        summary_path = output_dir / 'keyhole_measurements_summary.csv'
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary to: {summary_path}")
    else:
        print("\n--- Measurements (first 10 rows) ---")
        print(df.head(10))
        print("\n--- Summary Statistics ---")
        print(summary)


def preview_z_projection(tiff_files, substrate_row, im_height, im_width):
    """
    Generate and display a z-projection preview with substrate line.

    Args:
        tiff_files: List of TIFF file paths
        substrate_row: Row index of substrate surface
        im_height: Image height in pixels
        im_width: Image width in pixels
    """
    # Calculate mean projection across all frames
    print(f"  Loading {len(tiff_files)} frames...")
    stack_sum = np.zeros((im_height, im_width), dtype=np.float64)

    for i, tiff_path in enumerate(tiff_files):
        im = np.array(Image.open(tiff_path))
        if im.ndim == 3:
            im = im[:, :, 0]
        stack_sum += im.astype(np.float64)

        if (i + 1) % 50 == 0 or i == len(tiff_files) - 1:
            print(f"  Loaded {i + 1}/{len(tiff_files)} frames", end='\r')

    print()

    # Normalize mean projection to 0-1
    mean_projection = stack_sum / len(tiff_files)
    mean_normalized = (mean_projection - mean_projection.min()) / (mean_projection.max() - mean_projection.min())

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(mean_normalized, cmap='gray', aspect='equal')

    # Draw substrate surface line
    ax.axhline(y=substrate_row, color='red', linestyle='--', linewidth=1.5,
               label=f'Substrate surface (row {substrate_row})')

    # Add labels
    ax.set_title(f'Z-Projection (Normalized Mean) - {len(tiff_files)} frames\n'
                 f'Substrate: {SUBSTRATE_SURFACE_FROM_BOTTOM_UM} um from bottom')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.legend(loc='upper right')

    # Add text annotation
    ax.text(10, substrate_row + 10, f'← Substrate ({SUBSTRATE_SURFACE_FROM_BOTTOM_UM} um)',
            color='red', fontsize=9, verticalalignment='top')

    plt.tight_layout()
    plt.show()


def preview_measurement_steps(tiff_files, substrate_row, im_height, im_width):
    """
    Show example frame with original -> cropped -> measurements visualization.

    Args:
        tiff_files: List of TIFF file paths
        substrate_row: Row index of substrate surface
        im_height: Image height in pixels
        im_width: Image width in pixels
    """
    # Find a frame with a valid keyhole (try middle frame first, then search)
    test_indices = [len(tiff_files) // 2]  # Start with middle frame
    test_indices.extend(range(0, len(tiff_files), max(1, len(tiff_files) // 20)))  # Sample every 5%

    example_frame = None
    example_idx = None

    for idx in test_indices:
        im = np.array(Image.open(tiff_files[idx]))
        if im.ndim == 3:
            im = im[:, :, 0]
        im_binary = (im > 127).astype(np.uint8)

        labeled = measure.label(im_binary)
        props = measure.regionprops(labeled)

        if len(props) > 0:
            # Check if keyhole extends below substrate
            largest = sorted(props, key=lambda x: x.area, reverse=True)[0]
            if largest.bbox[2] > substrate_row:  # max_row > substrate
                example_frame = im_binary
                example_idx = idx
                break

    if example_frame is None:
        print("  WARNING: No valid keyhole frame found for preview")
        return

    print(f"  Using frame {example_idx + 1} for measurement preview")

    # Process the example frame
    im_binary = example_frame * 255  # Convert to 0/255

    # Get original keyhole properties
    labeled = measure.label(im_binary // 255)
    props = measure.regionprops(labeled)
    props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
    keyhole_props = props_sorted[0]

    # Create keyhole mask
    keyhole_mask_original = (labeled == keyhole_props.label).astype(np.uint8) * 255

    # Create cropped mask (above substrate removed)
    keyhole_mask_cropped = keyhole_mask_original.copy()
    keyhole_mask_cropped[:substrate_row, :] = 0

    # Get measurements from cropped keyhole
    cropped_props = measure.regionprops(keyhole_mask_cropped)
    if not cropped_props:
        print("  WARNING: No keyhole below substrate in example frame")
        return

    min_row, min_col, max_row, max_col = cropped_props[0].bbox

    # Calculate measurements
    max_depth_px = 0
    max_width_px = 0
    depth_at_max_width_px = 0
    max_width_row = substrate_row
    max_depth_col = min_col

    # Find max width and its depth
    for r in range(substrate_row, max_row):
        row = keyhole_mask_cropped[r]
        white_pixels = np.sum(row == 255)
        if white_pixels > max_width_px:
            max_width_px = white_pixels
            max_width_row = r
            depth_at_max_width_px = r - substrate_row

    # Find max depth and aperture
    aperture_cols = []
    for c in range(min_col, max_col):
        col = keyhole_mask_cropped[:, c]
        white_indices = np.where(col == 255)[0]
        white_below = white_indices[white_indices >= substrate_row]
        if len(white_below) > 0:
            depth_px = white_below[-1] - substrate_row
            if depth_px > max_depth_px:
                max_depth_px = depth_px
                max_depth_col = c
            # Check for aperture (at substrate row)
            if substrate_row in white_indices:
                aperture_cols.append(c)

    # Create figure with 3 panels
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))

    # Panel 1: Original
    ax1 = axes[0]
    ax1.imshow(keyhole_mask_original, cmap='gray', aspect='equal')
    ax1.axhline(y=substrate_row, color='red', linestyle='--', linewidth=1.5)
    ax1.set_title(f'Original (Frame {example_idx + 1})')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')

    # Panel 2: Cropped (substrate mask applied)
    ax2 = axes[1]
    ax2.imshow(keyhole_mask_cropped, cmap='gray', aspect='equal')
    ax2.axhline(y=substrate_row, color='red', linestyle='--', linewidth=1.5)
    ax2.set_title('Cropped (above substrate removed)')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')

    # Panel 3: Measurements overlay
    ax3 = axes[2]
    # Show cropped keyhole with color overlay
    rgb_image = np.stack([keyhole_mask_cropped] * 3, axis=-1).astype(np.float32) / 255

    ax3.imshow(rgb_image, aspect='equal')
    ax3.axhline(y=substrate_row, color='red', linestyle='--', linewidth=1.5,
                label=f'Substrate datum')

    # Draw max depth line (vertical, cyan)
    ax3.plot([max_depth_col, max_depth_col],
             [substrate_row, substrate_row + max_depth_px],
             'c-', linewidth=2, label=f'Max depth: {max_depth_px * UM_PER_PIXEL:.1f} um')

    # Draw max width line (horizontal, yellow)
    # Find the actual left and right extent at max_width_row
    row_at_max_width = keyhole_mask_cropped[max_width_row]
    white_cols = np.where(row_at_max_width == 255)[0]
    if len(white_cols) > 0:
        left_col = white_cols[0]
        right_col = white_cols[-1]
        ax3.plot([left_col, right_col], [max_width_row, max_width_row],
                 'y-', linewidth=2, label=f'Max width: {max_width_px * UM_PER_PIXEL:.1f} um')

    # Draw aperture line (horizontal at substrate, magenta)
    if len(aperture_cols) > 0:
        aperture_width_px = len(aperture_cols)
        ax3.plot([min(aperture_cols), max(aperture_cols)], [substrate_row, substrate_row],
                 'm-', linewidth=2, label=f'Aperture: {aperture_width_px * UM_PER_PIXEL:.1f} um')

    # Draw depth at max width (vertical dashed, green)
    if len(white_cols) > 0:
        mid_col = (left_col + right_col) // 2
        ax3.plot([mid_col, mid_col], [substrate_row, max_width_row],
                 'g--', linewidth=1.5, label=f'Depth@max width: {depth_at_max_width_px * UM_PER_PIXEL:.1f} um')

    ax3.set_title('Measurements')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    ax3.legend(loc='lower right', fontsize=8)

    plt.suptitle(f'Measurement Preview - Frame {example_idx + 1}\n'
                 f'Pixel resolution: {UM_PER_PIXEL} um/px | '
                 f'Substrate: {SUBSTRATE_SURFACE_FROM_BOTTOM_UM} um from bottom',
                 fontsize=11)
    plt.tight_layout()
    plt.show()


def get_keyhole_measurements(im, substrate_row, frame_num=None):
    """
    Measure keyhole geometry from binary image.

    Args:
        im: Binary image (255 = keyhole, 0 = background)
        substrate_row: Row index of substrate surface (datum for depth)
        frame_num: Frame number for plotting (None to skip plotting)

    Returns:
        dict: Measurements in micrometers
    """
    im_height, im_width = im.shape

    # Get bounding box
    props = measure.regionprops(im)
    if not props:
        return {
            'max_depth_um': 0,
            'max_width_um': 0,
            'depth_at_max_width_um': 0,
            'aperture_width_um': 0,
        }

    min_row, min_col, max_row, max_col = props[0].bbox

    # Initialize measurements
    max_width_px = 0
    max_depth_px = 0
    aperture_width_px = 0
    depth_at_max_width_px = 0
    widths_px = []
    depths_px = []

    # Start measuring from substrate surface (or keyhole top if below surface)
    measure_start_row = max(substrate_row, min_row)

    # Iterate through rows to find max_width, depth_at_max_width, and aperture_width
    for r in range(measure_start_row, max_row):
        row = im[r]
        white_pixels = np.sum(row == 255)

        if PLOT_KEYHOLES:
            widths_px.append(white_pixels)

        # Aperture is width at substrate surface
        if r == substrate_row:
            aperture_width_px = white_pixels

        if white_pixels >= max_width_px:
            max_width_px = white_pixels
            depth_at_max_width_px = r - substrate_row

    # Iterate through columns to find max_depth
    for c in range(min_col, max_col):
        col = im[:, c]
        white_indices = np.where(col == 255)[0]

        if len(white_indices) > 0:
            # Only consider pixels at or below substrate surface
            white_below_substrate = white_indices[white_indices >= substrate_row]

            if len(white_below_substrate) > 0:
                # Depth is from substrate to lowest white pixel below substrate
                lowest_white = white_below_substrate[-1]
                depth_px = lowest_white - substrate_row

                if PLOT_KEYHOLES:
                    depths_px.append(depth_px)

                if depth_px > max_depth_px:
                    max_depth_px = depth_px
            else:
                if PLOT_KEYHOLES:
                    depths_px.append(0)
        else:
            if PLOT_KEYHOLES:
                depths_px.append(0)

    # Convert to micrometers
    measurements = {
        'max_depth_um': max_depth_px * UM_PER_PIXEL,
        'max_width_um': max_width_px * UM_PER_PIXEL,
        'depth_at_max_width_um': depth_at_max_width_px * UM_PER_PIXEL,
        'aperture_width_um': aperture_width_px * UM_PER_PIXEL,
    }

    if PLOT_KEYHOLES and frame_num is not None:
        print(f'Frame {frame_num}:')
        print(f'   max_depth = {measurements["max_depth_um"]:.1f} um')
        print(f'   max_width = {measurements["max_width_um"]:.1f} um')
        print(f'   depth_at_max_width = {measurements["depth_at_max_width_um"]:.1f} um')
        print(f'   aperture_width = {measurements["aperture_width_um"]:.1f} um')

        plot_keyhole_dimensions(
            im, frame_num, widths_px, depths_px,
            min_col, max_col, substrate_row, max_row
        )

    return measurements


def generate_summary_stats(df):
    """Generate summary statistics for keyhole measurements."""

    summary_data = {
        'metric': [],
        'n_frames': [],
        'n_valid': [],
        'mean': [],
        'std': [],
        'min': [],
        'max': [],
        'median': [],
    }

    measurement_cols = ['area_um2', 'max_depth_um', 'max_width_um',
                        'depth_at_max_width_um', 'aperture_width_um']

    for col in measurement_cols:
        if col not in df.columns:
            continue

        data = df[col].values
        data_nonzero = data[data > 0]

        summary_data['metric'].append(col)
        summary_data['n_frames'].append(len(data))
        summary_data['n_valid'].append(len(data_nonzero))

        if len(data_nonzero) > 0:
            summary_data['mean'].append(np.mean(data_nonzero))
            summary_data['std'].append(np.std(data_nonzero))
            summary_data['min'].append(np.min(data_nonzero))
            summary_data['max'].append(np.max(data_nonzero))
            summary_data['median'].append(np.median(data_nonzero))
        else:
            summary_data['mean'].append(0)
            summary_data['std'].append(0)
            summary_data['min'].append(0)
            summary_data['max'].append(0)
            summary_data['median'].append(0)

    return pd.DataFrame(summary_data)


def plot_keyhole_dimensions(im, frame_num, widths, depths, min_col, max_col, min_row, max_row):
    """Plot keyhole with dimension profiles."""

    # Crop to region of interest
    keyhole_cropped = im[min_row:max_row, min_col:max_col]
    y_lim, x_lim = keyhole_cropped.shape

    if x_lim == 0 or y_lim == 0:
        return

    aspect = y_lim / x_lim if x_lim > 0 else 1

    fig, ax = plt.subplots()
    fig.suptitle(f'Frame {frame_num}\nKeyhole dimensions (um)')

    # Show keyhole image
    ax.imshow(keyhole_cropped, cmap='gray')

    # Create axes for profiles
    divider = make_axes_locatable(ax)
    ax_depth = divider.append_axes("bottom", size='100%', pad='3%', sharex=ax)
    ax_width = divider.append_axes("right", size='100%', pad='3%', sharey=ax)

    # Turn off ticks on image
    ax.yaxis.set_tick_params(labelleft=False, size=0)
    ax.xaxis.set_tick_params(labelbottom=False, size=0)

    # Plot depth profile (convert to um)
    if depths:
        depths_um = [d * UM_PER_PIXEL for d in depths]
        ax_depth.plot(range(len(depths_um)), depths_um)
        ax_depth.set_box_aspect(aspect)
        ax_depth.set_ylim(max(depths_um) * 1.1 if depths_um else 1, 0)
        ax_depth.set_ylabel('Depth (um)')
        ax_depth.set_xlabel('X position (px)')

    # Plot width profile (convert to um)
    if widths:
        widths_um = [w * UM_PER_PIXEL for w in widths[:y_lim]]
        ax_width.plot(widths_um, range(len(widths_um)))
        ax_width.set_box_aspect(aspect)
        ax_width.yaxis.set_label_position("right")
        ax_width.yaxis.tick_right()
        ax_width.set_xlabel('Width (um)')
        ax_width.set_ylabel('Y position (px)')

    plt.show()


def print_progress(iteration, total, prefix='', suffix='', length=50):
    """Print progress bar to console."""
    percent = f'{100 * iteration / total:.1f}'
    filled = int(length * iteration // total)
    bar = '█' * filled + '-' * (length - filled)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()


if __name__ == "__main__":
    main()
