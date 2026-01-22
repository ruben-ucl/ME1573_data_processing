#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Saturation Upper Bound Analysis

This script analyzes saturated images to recommend optimal upper bounds for
reprocessing. Uses multiple methods focused on the high-intensity signal region.

Author: Rubén Lambert-Garcia
Version: v1.0
"""

import os
import sys
import glob
import functools
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

sys.path.insert(1, os.path.join(sys.path[0], '..'))

__author__ = 'Rubén Lambert-Garcia'
__version__ = 'v1.0'

# Re-implement print to fix console output buffering
print = functools.partial(print, flush=True)


def load_images_from_directory(directory_path, pattern='*.png'):
    """
    Load all PNG images from a directory into a list of numpy arrays.

    Parameters:
        directory_path (str or Path): Path to directory containing images
        pattern (str): Glob pattern for image files

    Returns:
        list: List of numpy arrays, one per image
    """
    directory_path = Path(directory_path)
    image_files = sorted(glob.glob(str(directory_path / pattern)))

    if not image_files:
        raise ValueError(f"No PNG files found in {directory_path}")

    print(f"Found {len(image_files)} images in {directory_path}")

    images = []
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            if img.mode != 'L':
                img = img.convert('L')
            images.append(np.array(img))
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            continue

    print(f"Successfully loaded {len(images)} images")

    if images:
        print(f"Image shape: {images[0].shape}")
        print(f"Image dtype: {images[0].dtype}")

    return images


def compute_histogram(images, bins=256):
    """
    Compute histogram from all images combined.

    Parameters:
        images (list): List of numpy arrays containing image data
        bins (int): Number of histogram bins

    Returns:
        hist (ndarray): Histogram counts
        bin_centers (ndarray): Bin center values
        all_pixels (ndarray): Flattened pixel array
    """
    print("\nComputing histogram from image stack...")
    all_pixels = np.concatenate([img.flatten() for img in images])

    print(f"Total pixels: {len(all_pixels):,}")
    print(f"Min value: {np.min(all_pixels)}")
    print(f"Max value: {np.max(all_pixels)}")
    print(f"Mean value: {np.mean(all_pixels):.2f}")
    print(f"Std dev: {np.std(all_pixels):.2f}")

    hist, bin_edges = np.histogram(all_pixels, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return hist, bin_centers, all_pixels


def method1_percentile_based(all_pixels, current_max=255):
    """
    Method 1: Percentile-based analysis of high-intensity pixels.

    Parameters:
        all_pixels (ndarray): All pixel values
        current_max (int): Current maximum value (e.g., 255 for 8-bit)

    Returns:
        dict: Analysis results
    """
    print("\n" + "="*60)
    print("Method 1: Percentile-Based Analysis")
    print("="*60)

    # Define signal threshold (exclude background)
    median_val = np.median(all_pixels)
    signal_threshold = max(median_val, 100)  # At least 100 to avoid background

    # Extract high-intensity (signal) pixels
    signal_pixels = all_pixels[all_pixels >= signal_threshold]

    print(f"Signal threshold: {signal_threshold:.1f}")
    print(f"Signal pixels: {len(signal_pixels):,} ({len(signal_pixels)/len(all_pixels)*100:.1f}%)")

    # Calculate percentiles
    percentiles = [90, 95, 99, 99.5, 99.9, 99.99]
    percentile_values = np.percentile(signal_pixels, percentiles)

    print("\nSignal Pixel Percentiles:")
    for p, val in zip(percentiles, percentile_values):
        print(f"  {p}th: {val:.2f}")

    # Check saturation
    saturated_count = np.sum(signal_pixels >= current_max)
    saturation_fraction = saturated_count / len(signal_pixels)

    print(f"\nSaturation Analysis:")
    print(f"  Pixels at max ({current_max}): {saturated_count:,}")
    print(f"  Saturation fraction: {saturation_fraction*100:.2f}%")

    # Calculate recommended upper bound
    # Based on how compressed the upper percentiles are
    p99_to_p95 = percentile_values[3] - percentile_values[1]  # 99.5th - 95th
    natural_spread = p99_to_p95 / 0.045  # Expect 4.5% spread naturally

    # If saturation is significant, add headroom
    if saturation_fraction > 0.01:  # >1% saturated
        headroom_needed = natural_spread * saturation_fraction * 10
        recommended_max = current_max + headroom_needed
    else:
        recommended_max = current_max

    print(f"\nRecommended upper bound: {recommended_max:.1f}")

    return {
        'name': 'Percentile-Based',
        'recommended_max': recommended_max,
        'saturation_fraction': saturation_fraction,
        'signal_threshold': signal_threshold,
        'percentile_values': dict(zip(percentiles, percentile_values))
    }


def method2_tail_fitting(hist, bin_centers, current_max=255):
    """
    Method 2: Exponential tail fitting for high-intensity region.

    Parameters:
        hist (ndarray): Histogram counts
        bin_centers (ndarray): Bin center values
        current_max (int): Current maximum value

    Returns:
        dict: Analysis results
    """
    print("\n" + "="*60)
    print("Method 2: Tail Fitting Analysis")
    print("="*60)

    # Focus on upper tail (e.g., 150-255)
    tail_start = max(150, current_max - 100)
    tail_end = current_max - 5  # Exclude last few bins (saturated)

    mask = (bin_centers >= tail_start) & (bin_centers <= tail_end)
    x_tail = bin_centers[mask]
    y_tail = hist[mask]

    print(f"Analyzing tail region: {tail_start:.0f} to {tail_end:.0f}")

    # Fit exponential decay: y = A * exp(-k * x) + B
    def exponential_decay(x, A, k, B):
        return A * np.exp(-k * (x - tail_start)) + B

    try:
        # Initial guess
        p0 = [np.max(y_tail), 0.05, np.min(y_tail)]
        popt, _ = curve_fit(exponential_decay, x_tail, y_tail, p0=p0, maxfev=10000)

        A, k, B = popt
        print(f"Fitted parameters: A={A:.1f}, k={k:.4f}, B={B:.1f}")

        # Extrapolate to find where curve reaches near-zero (1% of peak)
        threshold = 0.01 * A
        if k > 0:
            # Solve: threshold = A * exp(-k * (x - tail_start)) + B
            x_end = tail_start - np.log((threshold - B) / A) / k
            recommended_max = min(x_end, current_max * 2)  # Cap at 2x current
        else:
            recommended_max = current_max * 1.2

        print(f"Extrapolated natural endpoint: {x_end:.1f}")
        print(f"Recommended upper bound: {recommended_max:.1f}")

        # Generate fitted curve for plotting
        x_fit = np.linspace(tail_start, current_max + 50, 200)
        y_fit = exponential_decay(x_fit, *popt)

        success = True

    except Exception as e:
        print(f"Tail fitting failed: {e}")
        recommended_max = current_max * 1.1
        x_fit = None
        y_fit = None
        success = False

    return {
        'name': 'Tail Fitting',
        'recommended_max': recommended_max,
        'tail_start': tail_start,
        'tail_end': tail_end,
        'x_fit': x_fit,
        'y_fit': y_fit,
        'success': success
    }


def method3_saturation_fraction(hist, bin_centers, current_max=255):
    """
    Method 3: Direct saturation fraction method.

    Parameters:
        hist (ndarray): Histogram counts
        bin_centers (ndarray): Bin center values
        current_max (int): Current maximum value

    Returns:
        dict: Analysis results
    """
    print("\n" + "="*60)
    print("Method 3: Saturation Fraction Analysis")
    print("="*60)

    # Count pixels at maximum value
    saturated_pixels = hist[-1]

    # Count high-intensity pixels (e.g., above 150)
    high_threshold = max(150, current_max - 100)
    high_intensity_mask = bin_centers >= high_threshold
    total_high_pixels = np.sum(hist[high_intensity_mask])

    saturation_fraction = saturated_pixels / total_high_pixels

    print(f"Saturated pixels (at {current_max}): {saturated_pixels:,}")
    print(f"High-intensity pixels (>{high_threshold}): {total_high_pixels:,}")
    print(f"Saturation fraction: {saturation_fraction*100:.2f}%")

    # Recommendation based on saturation severity
    if saturation_fraction > 0.10:  # >10% saturated
        safety_factor = 2.0
        severity = "SEVERE"
    elif saturation_fraction > 0.05:  # >5% saturated
        safety_factor = 1.5
        severity = "HIGH"
    elif saturation_fraction > 0.01:  # >1% saturated
        safety_factor = 1.2
        severity = "MODERATE"
    else:
        safety_factor = 1.0
        severity = "LOW"

    recommended_max = current_max * (1 + saturation_fraction * safety_factor)

    print(f"Saturation severity: {severity}")
    print(f"Safety factor: {safety_factor:.1f}")
    print(f"Recommended upper bound: {recommended_max:.1f}")

    return {
        'name': 'Saturation Fraction',
        'recommended_max': recommended_max,
        'saturation_fraction': saturation_fraction,
        'severity': severity,
        'safety_factor': safety_factor
    }


def method4_derivative_detection(hist, bin_centers, current_max=255):
    """
    Method 4: Derivative-based saturation cliff detection.

    Parameters:
        hist (ndarray): Histogram counts
        bin_centers (ndarray): Bin center values
        current_max (int): Current maximum value

    Returns:
        dict: Analysis results
    """
    print("\n" + "="*60)
    print("Method 4: Derivative-Based Saturation Detection")
    print("="*60)

    # Focus on high-intensity region
    region_start = max(100, current_max - 150)
    mask = bin_centers >= region_start
    x_region = bin_centers[mask]
    y_region = hist[mask]

    # Smooth the histogram to reduce noise
    if len(y_region) > 5:
        y_smooth = savgol_filter(y_region, window_length=min(11, len(y_region)//2*2+1), polyorder=2)
    else:
        y_smooth = y_region

    # Calculate derivative
    dy_dx = np.gradient(y_smooth, x_region)

    # Find where derivative becomes very large (saturation cliff)
    derivative_threshold = np.std(dy_dx) * 3  # 3 standard deviations
    cliff_indices = np.where(dy_dx > derivative_threshold)[0]

    if len(cliff_indices) > 0:
        cliff_location = x_region[cliff_indices[0]]
        print(f"Saturation cliff detected at intensity: {cliff_location:.1f}")

        # Find slope before the cliff
        pre_cliff_indices = x_region < cliff_location - 10
        if np.any(pre_cliff_indices):
            pre_cliff_slope = np.mean(dy_dx[pre_cliff_indices])

            # Extrapolate: where would natural decline reach zero?
            if pre_cliff_slope < 0:
                distance_to_zero = -y_smooth[pre_cliff_indices][-1] / pre_cliff_slope
                recommended_max = cliff_location + distance_to_zero
            else:
                recommended_max = current_max * 1.2
        else:
            recommended_max = current_max * 1.2
    else:
        print("No clear saturation cliff detected")
        cliff_location = None
        recommended_max = current_max

    print(f"Recommended upper bound: {recommended_max:.1f}")

    return {
        'name': 'Derivative Detection',
        'recommended_max': recommended_max,
        'cliff_location': cliff_location,
        'x_region': x_region,
        'dy_dx': dy_dx
    }


def method5_iqr_based(all_pixels, current_max=255):
    """
    Method 5: Inter-Quartile Range (IQR) based outlier detection.

    Parameters:
        all_pixels (ndarray): All pixel values
        current_max (int): Current maximum value

    Returns:
        dict: Analysis results
    """
    print("\n" + "="*60)
    print("Method 5: IQR-Based Analysis")
    print("="*60)

    # Focus on signal pixels (exclude background)
    signal_threshold = max(np.median(all_pixels), 100)
    signal_pixels = all_pixels[all_pixels >= signal_threshold]

    # Calculate quartiles and IQR
    Q1 = np.percentile(signal_pixels, 25)
    Q3 = np.percentile(signal_pixels, 75)
    IQR = Q3 - Q1

    # Tukey's fences for outlier detection
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    # Extended fence for far outliers
    upper_far_fence = Q3 + 3.0 * IQR

    print(f"Signal threshold: {signal_threshold:.1f}")
    print(f"Q1: {Q1:.1f}")
    print(f"Q3: {Q3:.1f}")
    print(f"IQR: {IQR:.1f}")
    print(f"Upper fence (Q3 + 1.5*IQR): {upper_fence:.1f}")
    print(f"Upper far fence (Q3 + 3.0*IQR): {upper_far_fence:.1f}")

    # Check if current max is below upper fence (no saturation)
    if current_max < upper_fence:
        print("No saturation detected by IQR method")
        recommended_max = current_max
    else:
        # Saturation detected - recommend far fence + safety margin
        recommended_max = upper_far_fence * 1.1
        print(f"Saturation detected - current max exceeds upper fence")

    print(f"Recommended upper bound: {recommended_max:.1f}")

    return {
        'name': 'IQR-Based',
        'recommended_max': recommended_max,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'upper_fence': upper_fence,
        'upper_far_fence': upper_far_fence
    }


def calculate_consensus_recommendation(results):
    """
    Calculate consensus recommendation from all methods.

    Parameters:
        results (list): List of result dictionaries from each method

    Returns:
        float: Consensus recommended upper bound
    """
    print("\n" + "="*60)
    print("Consensus Recommendation")
    print("="*60)

    # Collect all recommendations
    recommendations = [r['recommended_max'] for r in results]

    print("\nAll method recommendations:")
    for r in results:
        print(f"  {r['name']}: {r['recommended_max']:.1f}")

    # Calculate statistics
    mean_rec = np.mean(recommendations)
    median_rec = np.median(recommendations)
    std_rec = np.std(recommendations)

    print(f"\nStatistics:")
    print(f"  Mean: {mean_rec:.1f}")
    print(f"  Median: {median_rec:.1f}")
    print(f"  Std Dev: {std_rec:.1f}")

    # Use median as consensus (more robust to outliers)
    consensus = median_rec

    # Round to reasonable value
    if consensus < 300:
        consensus = np.round(consensus / 10) * 10  # Round to nearest 10
    else:
        consensus = np.round(consensus / 50) * 50  # Round to nearest 50

    print(f"\nCONSENSUS RECOMMENDATION: {consensus:.0f}")

    return consensus


def plot_analysis_results(hist, bin_centers, all_pixels, results, consensus, current_max, output_path):
    """
    Create comprehensive visualization of all analysis methods.

    Parameters:
        hist (ndarray): Histogram counts
        bin_centers (ndarray): Bin center values
        all_pixels (ndarray): All pixel values
        results (list): List of result dictionaries
        consensus (float): Consensus recommendation
        current_max (int): Current maximum value
        output_path (Path): Path to save figure
    """
    fig = plt.figure(figsize=(16, 10), dpi=150)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main histogram plot
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0],
                alpha=0.6, color='steelblue', label='Histogram')

    # Mark current max and consensus recommendation
    ax_main.axvline(current_max, color='red', linestyle='--', linewidth=2,
                    label=f'Current Max: {current_max}')
    ax_main.axvline(consensus, color='green', linestyle='-', linewidth=2,
                    label=f'Consensus Recommendation: {consensus:.0f}')

    # Mark each method's recommendation
    colors = ['orange', 'purple', 'brown', 'pink', 'cyan']
    for result, color in zip(results, colors):
        ax_main.axvline(result['recommended_max'], color=color, linestyle=':', linewidth=1.5,
                        alpha=0.7, label=f'{result["name"]}: {result["recommended_max"]:.0f}')

    ax_main.set_xlabel('Pixel Intensity')
    ax_main.set_ylabel('Count')
    ax_main.set_title('Saturation Analysis - All Methods Comparison')
    ax_main.legend(loc='upper left', fontsize=8)
    ax_main.grid(True, alpha=0.3)

    # Zoom into high-intensity region
    ax_zoom = fig.add_subplot(gs[1, 0])
    zoom_mask = bin_centers >= 150
    ax_zoom.bar(bin_centers[zoom_mask], hist[zoom_mask],
                width=bin_centers[1]-bin_centers[0], alpha=0.6, color='steelblue')
    ax_zoom.axvline(current_max, color='red', linestyle='--', linewidth=2)
    ax_zoom.axvline(consensus, color='green', linestyle='-', linewidth=2)
    ax_zoom.set_xlabel('Pixel Intensity')
    ax_zoom.set_ylabel('Count')
    ax_zoom.set_title('High-Intensity Region (>150)')
    ax_zoom.grid(True, alpha=0.3)

    # Percentile plot
    ax_percentile = fig.add_subplot(gs[1, 1])
    percentile_result = results[0]  # Method 1
    if 'percentile_values' in percentile_result:
        percentiles = list(percentile_result['percentile_values'].keys())
        values = list(percentile_result['percentile_values'].values())
        ax_percentile.plot(percentiles, values, 'o-', linewidth=2, markersize=6)
        ax_percentile.axhline(current_max, color='red', linestyle='--', label=f'Current Max: {current_max}')
        ax_percentile.set_xlabel('Percentile')
        ax_percentile.set_ylabel('Pixel Intensity')
        ax_percentile.set_title('Signal Pixel Percentiles')
        ax_percentile.legend()
        ax_percentile.grid(True, alpha=0.3)

    # Tail fitting plot
    ax_tail = fig.add_subplot(gs[1, 2])
    tail_result = results[1]  # Method 2
    if tail_result['success'] and tail_result['x_fit'] is not None:
        tail_mask = bin_centers >= tail_result['tail_start']
        ax_tail.bar(bin_centers[tail_mask], hist[tail_mask],
                    width=bin_centers[1]-bin_centers[0], alpha=0.4, color='steelblue', label='Data')
        ax_tail.plot(tail_result['x_fit'], tail_result['y_fit'], 'r-',
                     linewidth=2, label='Exponential Fit')
        ax_tail.axvline(current_max, color='orange', linestyle='--', label=f'Current Max')
        ax_tail.set_xlabel('Pixel Intensity')
        ax_tail.set_ylabel('Count')
        ax_tail.set_title('Tail Fitting (Exponential Decay)')
        ax_tail.legend()
        ax_tail.grid(True, alpha=0.3)

    # Method comparison table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')

    table_data = []
    table_data.append(['Method', 'Recommended Max', 'Key Metric', 'Confidence'])

    for result in results:
        name = result['name']
        rec_max = f"{result['recommended_max']:.1f}"

        if name == 'Percentile-Based':
            key_metric = f"{result['saturation_fraction']*100:.2f}% saturated"
            confidence = "HIGH" if result['saturation_fraction'] > 0.01 else "MEDIUM"
        elif name == 'Tail Fitting':
            key_metric = "Exponential fit" if result['success'] else "Failed"
            confidence = "HIGH" if result['success'] else "LOW"
        elif name == 'Saturation Fraction':
            key_metric = f"{result['severity']} severity"
            confidence = "HIGH"
        elif name == 'Derivative Detection':
            key_metric = f"Cliff at {result['cliff_location']:.1f}" if result['cliff_location'] else "No cliff"
            confidence = "MEDIUM" if result['cliff_location'] else "LOW"
        elif name == 'IQR-Based':
            key_metric = f"IQR = {result['IQR']:.1f}"
            confidence = "MEDIUM"
        else:
            key_metric = "-"
            confidence = "-"

        table_data.append([name, rec_max, key_metric, confidence])

    table_data.append(['', '', '', ''])
    table_data.append(['CONSENSUS', f"{consensus:.0f}", f"Median of all methods", "HIGH"])

    table = ax_table.table(cellText=table_data, cellLoc='left',
                           colWidths=[0.25, 0.20, 0.35, 0.20],
                           loc='center', bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style the header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style the consensus row
    for i in range(4):
        table[(len(table_data)-1, i)].set_facecolor('#FFC107')
        table[(len(table_data)-1, i)].set_text_props(weight='bold')

    plt.suptitle(f'Saturation Upper Bound Analysis - Recommended: {consensus:.0f}',
                 fontsize=14, fontweight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")
    plt.show()


def main():
    """Main execution function."""

    # ========== USER INPUTS ==========
    image_directory = r'F:\AlSi10Mg single layer ffc\CWT_labelled_windows\PD1\cmor1_5-1_0\1.0_ms\781-50000_Hz_256_steps\grey\1'
    current_max = 255  # Current upper bound (8-bit = 255, 16-bit = 65535)
    num_bins = 256
    output_file = Path(image_directory) / 'saturation_analysis_result.png'
    # =================================

    print("="*60)
    print("SATURATION UPPER BOUND ANALYSIS")
    print("="*60)
    print(f"Image directory: {image_directory}")
    print(f"Current max value: {current_max}")
    print()

    # Load images
    try:
        images = load_images_from_directory(image_directory)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Compute histogram
    hist, bin_centers, all_pixels = compute_histogram(images, bins=num_bins)

    # Run all analysis methods
    results = []

    results.append(method1_percentile_based(all_pixels, current_max))
    results.append(method2_tail_fitting(hist, bin_centers, current_max))
    results.append(method3_saturation_fraction(hist, bin_centers, current_max))
    results.append(method4_derivative_detection(hist, bin_centers, current_max))
    results.append(method5_iqr_based(all_pixels, current_max))

    # Calculate consensus
    consensus = calculate_consensus_recommendation(results)

    # Visualize results
    plot_analysis_results(hist, bin_centers, all_pixels, results, consensus,
                          current_max, output_file)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\n🎯 RECOMMENDED UPPER BOUND FOR REPROCESSING: {consensus:.0f}")
    print(f"   (Current: {current_max}, Increase: {consensus - current_max:.0f}, +{(consensus/current_max-1)*100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
