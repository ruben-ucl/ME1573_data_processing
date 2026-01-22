#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Histogram Analysis with Skewed Gaussian Fitting

This script reads all PNG images from a directory, computes a histogram of the
combined pixel intensity values, and fits a skewed Gaussian distribution to
predict the true intensity range for saturated images.

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
        pattern (str): Glob pattern for image files (default: '*.png')

    Returns:
        list: List of numpy arrays, one per image
        list: List of image filenames
    """
    directory_path = Path(directory_path)
    image_files = sorted(glob.glob(str(directory_path / pattern)))

    if not image_files:
        raise ValueError(f"No PNG files found in {directory_path}")

    print(f"Found {len(image_files)} images in {directory_path}")

    images = []
    filenames = []

    for img_path in image_files:
        try:
            img = Image.open(img_path)
            # Convert to grayscale if image has multiple channels
            if img.mode != 'L':
                img = img.convert('L')
            img_array = np.array(img)
            images.append(img_array)
            filenames.append(Path(img_path).name)
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            continue

    print(f"Successfully loaded {len(images)} images")

    # Report image properties
    if images:
        first_img = images[0]
        print(f"Image shape: {first_img.shape}")
        print(f"Image dtype: {first_img.dtype}")

        # Determine bit depth
        if first_img.dtype == np.uint8:
            print("Detected 8-bit images")
        elif first_img.dtype == np.uint16:
            print("Detected 16-bit images")
        else:
            print(f"Detected images with dtype: {first_img.dtype}")

    return images, filenames


def compute_histogram(images, bins=256):
    """
    Compute histogram from all images combined.

    Parameters:
        images (list): List of numpy arrays containing image data
        bins (int): Number of histogram bins (default: 256 for 8-bit, use 65536 for 16-bit)

    Returns:
        hist (ndarray): Histogram counts
        bin_edges (ndarray): Bin edge values
        bin_centers (ndarray): Bin center values
    """
    print("Computing histogram from image stack...")

    # Flatten all images into a single 1D array
    all_pixels = np.concatenate([img.flatten() for img in images])

    print(f"Total pixels: {len(all_pixels):,}")
    print(f"Min value: {np.min(all_pixels)}")
    print(f"Max value: {np.max(all_pixels)}")
    print(f"Mean value: {np.mean(all_pixels):.2f}")
    print(f"Std dev: {np.std(all_pixels):.2f}")

    # Compute histogram
    hist, bin_edges = np.histogram(all_pixels, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return hist, bin_edges, bin_centers


def detect_saturation(hist, bin_centers, saturation_threshold=0.01):
    """
    Detect saturation in histogram by checking for large peaks at extremes.

    Parameters:
        hist (ndarray): Histogram counts
        bin_centers (ndarray): Bin center values
        saturation_threshold (float): Fraction of total pixels to consider saturated

    Returns:
        bool: True if min saturation detected
        bool: True if max saturation detected
        float: Min value to exclude from fitting
        float: Max value to exclude from fitting
    """
    total_pixels = np.sum(hist)
    threshold_count = total_pixels * saturation_threshold

    # Check first and last bins for saturation
    min_saturated = hist[0] > threshold_count
    max_saturated = hist[-1] > threshold_count

    # Find where to exclude data for fitting
    min_exclude = bin_centers[0]
    max_exclude = bin_centers[-1]

    if min_saturated:
        # Find first bin with count below threshold
        for i, count in enumerate(hist):
            if count < threshold_count:
                min_exclude = bin_centers[i]
                print(f"Min saturation detected: excluding values below {min_exclude:.1f}")
                break

    if max_saturated:
        # Find last bin with count below threshold
        for i in range(len(hist)-1, -1, -1):
            if hist[i] < threshold_count:
                max_exclude = bin_centers[i]
                print(f"Max saturation detected: excluding values above {max_exclude:.1f}")
                break

    return min_saturated, max_saturated, min_exclude, max_exclude


def skewed_gaussian(x, amplitude, center, width, skewness):
    """
    Skewed Gaussian distribution (using scipy's skewnorm).

    Parameters:
        x (ndarray): Input values
        amplitude (float): Peak amplitude
        center (float): Distribution center (location parameter)
        width (float): Distribution width (scale parameter)
        skewness (float): Skewness parameter (alpha)

    Returns:
        ndarray: Skewed Gaussian values
    """
    # Normalize and shift the skewnorm distribution
    return amplitude * stats.skewnorm.pdf(x, skewness, loc=center, scale=width)


def gamma_distribution(x, amplitude, shape, scale, offset):
    """
    Gamma distribution.

    Parameters:
        x (ndarray): Input values
        amplitude (float): Peak amplitude scaling
        shape (float): Shape parameter (k)
        scale (float): Scale parameter (theta)
        offset (float): Horizontal offset

    Returns:
        ndarray: Gamma distribution values
    """
    x_shifted = x - offset
    # Set negative values to zero
    result = np.zeros_like(x)
    mask = x_shifted > 0
    result[mask] = amplitude * stats.gamma.pdf(x_shifted[mask], shape, scale=scale)
    return result


def lognormal_distribution(x, amplitude, mu, sigma, offset):
    """
    Log-normal distribution.

    Parameters:
        x (ndarray): Input values
        amplitude (float): Peak amplitude scaling
        mu (float): Mean of underlying normal distribution
        sigma (float): Standard deviation of underlying normal distribution
        offset (float): Horizontal offset

    Returns:
        ndarray: Log-normal distribution values
    """
    x_shifted = x - offset
    result = np.zeros_like(x)
    mask = x_shifted > 0
    result[mask] = amplitude * stats.lognorm.pdf(x_shifted[mask], sigma, scale=np.exp(mu))
    return result


def weibull_distribution(x, amplitude, shape, scale, offset):
    """
    Weibull distribution.

    Parameters:
        x (ndarray): Input values
        amplitude (float): Peak amplitude scaling
        shape (float): Shape parameter (k)
        scale (float): Scale parameter (lambda)
        offset (float): Horizontal offset

    Returns:
        ndarray: Weibull distribution values
    """
    x_shifted = x - offset
    result = np.zeros_like(x)
    mask = x_shifted > 0
    result[mask] = amplitude * stats.weibull_min.pdf(x_shifted[mask], shape, scale=scale)
    return result


def bimodal_gaussian(x, amp1, center1, width1, amp2, center2, width2):
    """
    Bimodal Gaussian (sum of two Gaussians).

    Parameters:
        x (ndarray): Input values
        amp1 (float): Amplitude of first Gaussian
        center1 (float): Center of first Gaussian
        width1 (float): Width of first Gaussian
        amp2 (float): Amplitude of second Gaussian
        center2 (float): Center of second Gaussian
        width2 (float): Width of second Gaussian

    Returns:
        ndarray: Bimodal Gaussian values
    """
    gaussian1 = amp1 * np.exp(-((x - center1) ** 2) / (2 * width1 ** 2))
    gaussian2 = amp2 * np.exp(-((x - center2) ** 2) / (2 * width2 ** 2))
    return gaussian1 + gaussian2


def fit_multiple_distributions(hist, bin_centers, min_exclude, max_exclude):
    """
    Fit multiple distribution types to the histogram and select the best one.

    Parameters:
        hist (ndarray): Histogram counts
        bin_centers (ndarray): Bin center values
        min_exclude (float): Exclude bins below this value
        max_exclude (float): Exclude bins above this value

    Returns:
        dict: Best fit information with keys: 'name', 'popt', 'pcov', 'x_fit', 'y_fit', 'r2', 'rmse'
        list: All fit results for comparison
    """
    print("Fitting multiple distribution types to histogram...")

    # Create mask to exclude saturated regions
    mask = (bin_centers >= min_exclude) & (bin_centers <= max_exclude)

    x_data = bin_centers[mask]
    y_data = hist[mask]

    print(f"Fitting to {np.sum(mask)} bins (excluding saturated regions)")

    # Common initial estimates
    amplitude_guess = np.max(y_data)
    center_guess = x_data[np.argmax(y_data)]
    width_guess = np.std(x_data)

    print(f"Common initial guesses: amplitude={amplitude_guess:.1f}, center={center_guess:.1f}, width={width_guess:.1f}")
    print("")

    results = []

    # 1. Skewed Gaussian
    try:
        print("Fitting Skewed Gaussian...")
        p0 = [amplitude_guess, center_guess, width_guess, 0.0]
        bounds = ([0, 0, width_guess/10, -20],
                  [amplitude_guess * 100, center_guess + 2*width_guess, width_guess*5, 20])

        popt, pcov = curve_fit(skewed_gaussian, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)
        y_fit = skewed_gaussian(x_data, *popt)
        r2 = 1 - np.sum((y_data - y_fit)**2) / np.sum((y_data - np.mean(y_data))**2)
        rmse = np.sqrt(np.mean((y_data - y_fit)**2))

        print(f"  R² = {r2:.4f}, RMSE = {rmse:.1f}")
        print(f"  Parameters: amplitude={popt[0]:.1f}, center={popt[1]:.1f}, width={popt[2]:.1f}, skewness={popt[3]:.3f}")

        results.append({
            'name': 'Skewed Gaussian',
            'function': skewed_gaussian,
            'popt': popt,
            'pcov': pcov,
            'r2': r2,
            'rmse': rmse
        })
    except Exception as e:
        print(f"  Failed: {e}")

    # 2. Gamma Distribution
    try:
        print("Fitting Gamma Distribution...")
        p0 = [amplitude_guess, 2.0, width_guess/2, 0]
        bounds = ([0, 0.1, 0.1, -50],
                  [amplitude_guess * 100, 50, width_guess*3, center_guess])

        popt, pcov = curve_fit(gamma_distribution, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)
        y_fit = gamma_distribution(x_data, *popt)
        r2 = 1 - np.sum((y_data - y_fit)**2) / np.sum((y_data - np.mean(y_data))**2)
        rmse = np.sqrt(np.mean((y_data - y_fit)**2))

        print(f"  R² = {r2:.4f}, RMSE = {rmse:.1f}")
        print(f"  Parameters: amplitude={popt[0]:.1f}, shape={popt[1]:.2f}, scale={popt[2]:.1f}, offset={popt[3]:.1f}")

        results.append({
            'name': 'Gamma',
            'function': gamma_distribution,
            'popt': popt,
            'pcov': pcov,
            'r2': r2,
            'rmse': rmse
        })
    except Exception as e:
        print(f"  Failed: {e}")

    # 3. Log-normal Distribution
    try:
        print("Fitting Log-normal Distribution...")
        p0 = [amplitude_guess, np.log(center_guess + 1), 1.0, 0]
        bounds = ([0, -5, 0.01, -50],
                  [amplitude_guess * 100, 10, 5, center_guess])

        popt, pcov = curve_fit(lognormal_distribution, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)
        y_fit = lognormal_distribution(x_data, *popt)
        r2 = 1 - np.sum((y_data - y_fit)**2) / np.sum((y_data - np.mean(y_data))**2)
        rmse = np.sqrt(np.mean((y_data - y_fit)**2))

        print(f"  R² = {r2:.4f}, RMSE = {rmse:.1f}")
        print(f"  Parameters: amplitude={popt[0]:.1f}, mu={popt[1]:.2f}, sigma={popt[2]:.2f}, offset={popt[3]:.1f}")

        results.append({
            'name': 'Log-normal',
            'function': lognormal_distribution,
            'popt': popt,
            'pcov': pcov,
            'r2': r2,
            'rmse': rmse
        })
    except Exception as e:
        print(f"  Failed: {e}")

    # 4. Weibull Distribution
    try:
        print("Fitting Weibull Distribution...")
        p0 = [amplitude_guess, 2.0, width_guess, 0]
        bounds = ([0, 0.1, 0.1, -50],
                  [amplitude_guess * 100, 10, width_guess*3, center_guess])

        popt, pcov = curve_fit(weibull_distribution, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)
        y_fit = weibull_distribution(x_data, *popt)
        r2 = 1 - np.sum((y_data - y_fit)**2) / np.sum((y_data - np.mean(y_data))**2)
        rmse = np.sqrt(np.mean((y_data - y_fit)**2))

        print(f"  R² = {r2:.4f}, RMSE = {rmse:.1f}")
        print(f"  Parameters: amplitude={popt[0]:.1f}, shape={popt[1]:.2f}, scale={popt[2]:.1f}, offset={popt[3]:.1f}")

        results.append({
            'name': 'Weibull',
            'function': weibull_distribution,
            'popt': popt,
            'pcov': pcov,
            'r2': r2,
            'rmse': rmse
        })
    except Exception as e:
        print(f"  Failed: {e}")

    # 5. Bimodal Gaussian
    try:
        print("Fitting Bimodal Gaussian...")
        # Find two peaks for initial guess
        peak1_idx = np.argmax(y_data)
        peak1_val = y_data[peak1_idx]
        peak1_loc = x_data[peak1_idx]

        # Mask out region around first peak to find second peak
        mask_peak = np.abs(x_data - peak1_loc) > width_guess/2
        if np.any(mask_peak):
            peak2_idx = np.argmax(y_data[mask_peak])
            peak2_val = y_data[mask_peak][peak2_idx]
            peak2_loc = x_data[mask_peak][peak2_idx]
        else:
            # Default to assuming second peak at higher intensity
            peak2_loc = min(center_guess * 2, np.max(x_data))
            peak2_val = amplitude_guess / 2

        p0 = [peak1_val, peak1_loc, width_guess/2, peak2_val, peak2_loc, width_guess/2]
        bounds = ([0, 0, 1, 0, 0, 1],
                  [amplitude_guess * 10, center_guess * 2, width_guess*2, amplitude_guess * 10, np.max(x_data), width_guess*2])

        popt, pcov = curve_fit(bimodal_gaussian, x_data, y_data, p0=p0, bounds=bounds, maxfev=20000)
        y_fit = bimodal_gaussian(x_data, *popt)
        r2 = 1 - np.sum((y_data - y_fit)**2) / np.sum((y_data - np.mean(y_data))**2)
        rmse = np.sqrt(np.mean((y_data - y_fit)**2))

        print(f"  R² = {r2:.4f}, RMSE = {rmse:.1f}")
        print(f"  Parameters: amp1={popt[0]:.1f}, center1={popt[1]:.1f}, width1={popt[2]:.1f}, amp2={popt[3]:.1f}, center2={popt[4]:.1f}, width2={popt[5]:.1f}")

        results.append({
            'name': 'Bimodal Gaussian',
            'function': bimodal_gaussian,
            'popt': popt,
            'pcov': pcov,
            'r2': r2,
            'rmse': rmse
        })
    except Exception as e:
        print(f"  Failed: {e}")

    print("")

    if not results:
        print("ERROR: All fits failed!")
        return None, []

    # Select best fit based on R²
    best_fit = max(results, key=lambda x: x['r2'])
    print(f"Best fit: {best_fit['name']} (R² = {best_fit['r2']:.4f}, RMSE = {best_fit['rmse']:.1f})")
    print("")

    # Generate fitted curve over full range for best fit
    x_fit = np.linspace(bin_centers[0], bin_centers[-1], 1000)
    y_fit = best_fit['function'](x_fit, *best_fit['popt'])

    best_fit['x_fit'] = x_fit
    best_fit['y_fit'] = y_fit

    return best_fit, results


def predict_true_range(best_fit, bin_centers, threshold_fraction=0.001):
    """
    Predict the true min/max intensity range from the fitted distribution.

    Parameters:
        best_fit (dict): Best fit information with 'function', 'popt', 'y_fit' keys
        bin_centers (ndarray): Original bin centers
        threshold_fraction (float): Fraction of peak to consider as min/max bounds

    Returns:
        float: Predicted minimum intensity
        float: Predicted maximum intensity
    """
    if best_fit is None or best_fit['popt'] is None:
        return None, None

    # Get peak amplitude from fitted curve
    peak_amplitude = np.max(best_fit['y_fit'])

    # Define threshold as fraction of peak amplitude
    threshold = peak_amplitude * threshold_fraction

    # Generate extended range for prediction
    x_min = bin_centers[0] - 100
    x_max = bin_centers[-1] + 100
    x_extended = np.linspace(x_min, x_max, 10000)
    y_extended = best_fit['function'](x_extended, *best_fit['popt'])

    # Find where distribution crosses threshold
    above_threshold = y_extended > threshold

    if np.any(above_threshold):
        indices = np.where(above_threshold)[0]
        predicted_min = x_extended[indices[0]]
        predicted_max = x_extended[indices[-1]]

        print(f"\nPredicted true intensity range:")
        print(f"  Min: {predicted_min:.2f}")
        print(f"  Max: {predicted_max:.2f}")
        print(f"  Range: {predicted_max - predicted_min:.2f}")

        return predicted_min, predicted_max
    else:
        print("Warning: Could not determine intensity range from fitted curve")
        return None, None


def plot_histogram_with_fit(hist, bin_centers, best_fit, all_fits,
                            min_exclude, max_exclude,
                            predicted_min, predicted_max,
                            output_path=None):
    """
    Create a comprehensive plot of the histogram with fitted curve.

    Parameters:
        hist (ndarray): Histogram counts
        bin_centers (ndarray): Bin center values
        best_fit (dict): Best fit information
        all_fits (list): All fit results for comparison
        min_exclude (float): Min excluded value
        max_exclude (float): Max excluded value
        predicted_min (float): Predicted minimum intensity
        predicted_max (float): Predicted maximum intensity
        output_path (str or Path): Path to save figure (optional)
    """
    plt.rcParams.update({'font.size': 10})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150)

    # Plot 1: Full histogram with fit
    ax1.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0],
            alpha=0.6, color='steelblue', label='Histogram')

    if best_fit is not None and best_fit['x_fit'] is not None:
        ax1.plot(best_fit['x_fit'], best_fit['y_fit'], 'r-', linewidth=2,
                label=f'Best Fit: {best_fit["name"]} (R²={best_fit["r2"]:.3f})')

    # Mark excluded regions
    ax1.axvline(min_exclude, color='orange', linestyle='--', linewidth=1.5,
                label=f'Excluded < {min_exclude:.1f}')
    ax1.axvline(max_exclude, color='orange', linestyle='--', linewidth=1.5,
                label=f'Excluded > {max_exclude:.1f}')

    # Mark predicted range
    if predicted_min is not None and predicted_max is not None:
        ax1.axvline(predicted_min, color='green', linestyle=':', linewidth=2,
                    label=f'Predicted min: {predicted_min:.1f}')
        ax1.axvline(predicted_max, color='green', linestyle=':', linewidth=2,
                    label=f'Predicted max: {predicted_max:.1f}')

    ax1.set_xlabel('Pixel Intensity')
    ax1.set_ylabel('Count')
    ax1.set_title('Image Stack Histogram with Skewed Gaussian Fit')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Set y-axis limit to better fit the data
    # Consider both histogram peak and fitted curve maximum
    y_max = np.max(hist[1:-1]) if len(hist) > 2 else np.max(hist)
    if best_fit is not None and best_fit['y_fit'] is not None:
        y_max = max(y_max, np.max(best_fit['y_fit']))
    ax1.set_ylim(0, y_max * 1.15)  # Add 15% margin above the peak

    # Plot 2: Log scale to see distribution tails
    ax2.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0],
            alpha=0.6, color='steelblue', label='Histogram')

    if best_fit is not None and best_fit['x_fit'] is not None:
        ax2.plot(best_fit['x_fit'], best_fit['y_fit'], 'r-', linewidth=2,
                label=f'Best Fit: {best_fit["name"]}')

    ax2.axvline(min_exclude, color='orange', linestyle='--', linewidth=1.5)
    ax2.axvline(max_exclude, color='orange', linestyle='--', linewidth=1.5)

    if predicted_min is not None and predicted_max is not None:
        ax2.axvline(predicted_min, color='green', linestyle=':', linewidth=2)
        ax2.axvline(predicted_max, color='green', linestyle=':', linewidth=2)

    ax2.set_xlabel('Pixel Intensity')
    ax2.set_ylabel('Count (log scale)')
    ax2.set_title('Histogram (Log Scale) - Better View of Distribution Tails')
    ax2.set_yscale('log')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Add text box with fit comparison
    if best_fit is not None and all_fits:
        textstr = f'Best Fit: {best_fit["name"]}\n'
        textstr += f'R² = {best_fit["r2"]:.4f}\n'
        textstr += f'RMSE = {best_fit["rmse"]:.1f}\n\n'
        textstr += 'All Fits (R²):\n'
        for fit in sorted(all_fits, key=lambda x: x['r2'], reverse=True):
            textstr += f'{fit["name"]}: {fit["r2"]:.4f}\n'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")

    plt.show()


def main():
    """Main execution function."""

    # ========== USER INPUTS ==========
    # Set your image directory path here
    image_directory = r'F:\AlSi10Mg single layer ffc\CWT_labelled_windows\PD1\mexh\1.0_ms\1000-50000_Hz_256_steps\grey\full_signal'

    # Number of histogram bins (256 for 8-bit, 65536 for 16-bit images)
    num_bins = 256

    # Saturation detection threshold (fraction of total pixels)
    saturation_threshold = 0.01  # 1% of pixels in a single bin indicates saturation

    # Threshold for predicting true range (fraction of peak amplitude)
    range_threshold = 0.001  # 0.1% of peak

    # Output file path (set to None to skip saving)
    output_file = Path(image_directory) / 'histogram_analysis_result.png'
    # =================================

    print("="*60)
    print("Image Histogram Analysis with Skewed Gaussian Fitting")
    print("="*60)

    # Load images
    try:
        images, filenames = load_images_from_directory(image_directory)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Adjust bins for 16-bit images
    if images[0].dtype == np.uint16 and num_bins == 256:
        print("Warning: Using 256 bins for 16-bit images. Consider using num_bins=65536 for better resolution.")

    print("")

    # Compute histogram
    hist, bin_edges, bin_centers = compute_histogram(images, bins=num_bins)

    print("")

    # Detect saturation
    min_sat, max_sat, min_exclude, max_exclude = detect_saturation(
        hist, bin_centers, saturation_threshold
    )

    if not min_sat and not max_sat:
        print("No significant saturation detected in images.")
        print("Fitting will use full histogram range.")

    print("")

    # Fit multiple distributions and select best
    best_fit, all_fits = fit_multiple_distributions(
        hist, bin_centers, min_exclude, max_exclude
    )

    print("")

    # Predict true range
    predicted_min, predicted_max = predict_true_range(
        best_fit, bin_centers, range_threshold
    )

    # Calculate dynamic range
    if predicted_min is not None and predicted_max is not None:
        measured_range = bin_centers[-1] - bin_centers[0]
        predicted_range = predicted_max - predicted_min
        range_extension = predicted_range - measured_range

        print(f"\nDynamic Range Analysis:")
        print(f"  Measured range: {measured_range:.2f}")
        print(f"  Predicted range: {predicted_range:.2f}")
        print(f"  Extension needed: {range_extension:.2f} ({range_extension/measured_range*100:.1f}%)")

    print("")
    print("="*60)

    # Plot results
    plot_histogram_with_fit(
        hist, bin_centers, best_fit, all_fits,
        min_exclude, max_exclude,
        predicted_min, predicted_max,
        output_path=output_file
    )

    print("Analysis complete!")


if __name__ == "__main__":
    main()
