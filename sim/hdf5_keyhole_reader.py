"""
HDF5 Keyhole Perimeter Extraction Module

This module provides functions to extract keyhole perimeter coordinates from
binary HDF5 datasets for use with RayTracer analysis.

Key features:
- Extract contours from binary keyhole images
- Close the keyhole opening at the surface (120 μm width, centered)
- Smooth corners with spline interpolation
- Order points from top-right to top-left
- Convert to physical coordinates (microns)

Author: Claude
Date: 2025-11-20
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import pandas as pd
from scipy import interpolate
from skimage import measure


def read_keyhole_images(hdf5_path: str, dataset_name: str = 'keyhole_bin') -> Tuple[np.ndarray, Dict]:
    """
    Read binary keyhole images from HDF5 file.

    Parameters:
        hdf5_path: Path to HDF5 file
        dataset_name: Name of binary keyhole dataset (default: 'keyhole_bin')

    Returns:
        images: 3D numpy array (n_frames, height, width) with binary keyhole masks
        metadata: Dictionary with dataset information
    """
    hdf5_path = Path(hdf5_path)

    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        if dataset_name not in f:
            available = list(f.keys())
            raise KeyError(f"Dataset '{dataset_name}' not found. Available: {available}")

        images = f[dataset_name][:]

        metadata = {
            'filename': hdf5_path.name,
            'dataset': dataset_name,
            'shape': images.shape,
            'dtype': images.dtype,
            'n_frames': images.shape[0] if len(images.shape) == 3 else 1,
            'height': images.shape[-2],
            'width': images.shape[-1]
        }

    return images, metadata


def find_keyhole_center(contour: np.ndarray) -> float:
    """
    Calculate the keyhole center X-coordinate.

    Parameters:
        contour: Array of (row, col) or (y, x) coordinates from find_contours

    Returns:
        center_x: Mean X-coordinate (col coordinate)
    """
    # Contour format from find_contours is (row, col) = (y, x)
    x_coords = contour[:, 1]
    center_x = np.mean(x_coords)
    return center_x


def identify_opening_endpoints(contour: np.ndarray, top_percentile: float = 5.0) -> Tuple[int, int]:
    """
    Find the top-left and top-right endpoints where the keyhole opens.

    Parameters:
        contour: Array of (row, col) coordinates
        top_percentile: Percentile threshold for "top" points (default: 5%)

    Returns:
        left_idx: Index of top-left point
        right_idx: Index of top-right point
    """
    # Contour format: (row, col) = (y, x)
    y_coords = contour[:, 0]
    x_coords = contour[:, 1]

    # Find top surface points (minimum y values, since y=0 is at top)
    top_threshold = np.percentile(y_coords, top_percentile)
    top_mask = y_coords <= top_threshold

    # Among top points, find leftmost and rightmost
    top_indices = np.where(top_mask)[0]

    if len(top_indices) == 0:
        # Fallback: use absolute minimum y points
        min_y = np.min(y_coords)
        top_indices = np.where(y_coords == min_y)[0]

    top_x = x_coords[top_indices]

    # Left endpoint: minimum X among top points
    left_relative_idx = np.argmin(top_x)
    left_idx = top_indices[left_relative_idx]

    # Right endpoint: maximum X among top points
    right_relative_idx = np.argmax(top_x)
    right_idx = top_indices[right_relative_idx]

    return left_idx, right_idx


def close_keyhole_opening(contour: np.ndarray,
                          keyhole_center_x: float,
                          opening_width_um: float = 120.0,
                          resolution_um_per_pixel: float = 4.3) -> np.ndarray:
    """
    Close the keyhole opening at the surface by adding vertical and horizontal segments.

    The opening is created with:
    - Total width: opening_width_um (120 μm default)
    - Centered on: keyhole_center_x
    - Vertical extensions from endpoints to y=0 (top edge)
    - Horizontal segment along top edge

    Parameters:
        contour: Array of (row, col) coordinates
        keyhole_center_x: Center X-coordinate of keyhole (pixels)
        opening_width_um: Total opening width in microns (default: 120)
        resolution_um_per_pixel: Spatial resolution (default: 4.3)

    Returns:
        closed_contour: Array with opening closure points inserted
    """
    # Find opening endpoints
    left_idx, right_idx = identify_opening_endpoints(contour)

    # Get endpoint coordinates
    left_point = contour[left_idx]  # (y, x)
    right_point = contour[right_idx]  # (y, x)

    # Calculate opening edge positions in pixels
    half_opening_pixels = (opening_width_um / 2.0) / resolution_um_per_pixel
    left_opening_x = keyhole_center_x - half_opening_pixels
    right_opening_x = keyhole_center_x + half_opening_pixels

    # Create new points for opening closure
    # All at y=0 (top edge)
    top_y = 0.0

    # Vertical extensions
    left_vertical_top = np.array([top_y, left_point[1]])  # From left endpoint up to y=0
    right_vertical_top = np.array([top_y, right_point[1]])  # From right endpoint up to y=0

    # Horizontal segment along top edge
    left_horizontal = np.array([top_y, left_opening_x])
    right_horizontal = np.array([top_y, right_opening_x])

    # Determine contour ordering (which way around it goes)
    # Assume contour is ordered, find which endpoint comes first
    if left_idx < right_idx:
        # Left endpoint comes first in contour
        # Insert points: contour[0:left_idx] + left_point + left_vertical_top +
        #                left_horizontal + right_horizontal + right_vertical_top +
        #                right_point + contour[right_idx+1:]

        # Actually, we want to insert between the endpoints
        # Split contour at endpoints and insert new geometry
        part1 = contour[:left_idx+1]  # Up to and including left endpoint
        part2 = contour[right_idx:]    # From right endpoint onwards

        # New points sequence
        new_points = np.array([
            left_vertical_top,
            left_horizontal,
            right_horizontal,
            right_vertical_top
        ])

        closed_contour = np.vstack([part1, new_points, part2])
    else:
        # Right endpoint comes first
        part1 = contour[:right_idx+1]
        part2 = contour[left_idx:]

        new_points = np.array([
            right_vertical_top,
            right_horizontal,
            left_horizontal,
            left_vertical_top
        ])

        closed_contour = np.vstack([part1, new_points, part2])

    return closed_contour


def smooth_contour_spline(contour: np.ndarray, smoothing_factor: float = 0.5) -> np.ndarray:
    """
    Apply cubic spline smoothing to round corners.

    Parameters:
        contour: Array of (row, col) coordinates
        smoothing_factor: Smoothing parameter (0=no smoothing, higher=more smoothing)

    Returns:
        smoothed_contour: Array with same shape, smoothed
    """
    if len(contour) < 4:
        return contour  # Need at least 4 points for cubic spline

    # Create parameter t for spline (cumulative distance along contour)
    distances = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
    t = np.concatenate([[0], np.cumsum(distances)])

    # Normalize t to [0, 1]
    t = t / t[-1] if t[-1] > 0 else t

    try:
        # Fit splines for y and x separately
        # Use s parameter to control smoothing (s=0 means interpolation)
        s_param = smoothing_factor * len(contour)

        spline_y = interpolate.UnivariateSpline(t, contour[:, 0], s=s_param, k=3)
        spline_x = interpolate.UnivariateSpline(t, contour[:, 1], s=s_param, k=3)

        # Evaluate splines at original parameter values
        smoothed_y = spline_y(t)
        smoothed_x = spline_x(t)

        smoothed_contour = np.column_stack([smoothed_y, smoothed_x])

    except Exception as e:
        print(f"Warning: Spline smoothing failed ({e}). Returning original contour.")
        smoothed_contour = contour

    return smoothed_contour


def order_contour_top_right_to_left(contour: np.ndarray) -> np.ndarray:
    """
    Reorder contour points to go from top-right to top-left.

    Parameters:
        contour: Array of (row, col) coordinates

    Returns:
        ordered_contour: Reordered array starting at top-right, ending at top-left
    """
    # Find top-right point (minimum y, maximum x among top points)
    y_coords = contour[:, 0]
    x_coords = contour[:, 1]

    # Top points (minimum 10%)
    top_threshold = np.percentile(y_coords, 10)
    top_mask = y_coords <= top_threshold
    top_indices = np.where(top_mask)[0]

    if len(top_indices) == 0:
        # Fallback to absolute minimum
        top_indices = np.where(y_coords == np.min(y_coords))[0]

    # Top-right: maximum X among top indices
    top_right_relative = np.argmax(x_coords[top_indices])
    top_right_idx = top_indices[top_right_relative]

    # Reorder contour to start at top_right_idx
    ordered_contour = np.vstack([contour[top_right_idx:], contour[:top_right_idx]])

    return ordered_contour


def extract_keyhole_perimeter_with_opening(binary_image: np.ndarray,
                                           opening_width_um: float = 120.0,
                                           smoothing_factor: float = 0.5,
                                           resolution_um_per_pixel: float = 4.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete perimeter extraction pipeline.

    Steps:
    1. Find contours in binary image
    2. Select largest contour (main keyhole)
    3. Calculate keyhole center
    4. Close opening at top surface
    5. Smooth corners with spline
    6. Order points from top-right to top-left
    7. Convert to microns

    Parameters:
        binary_image: 2D binary array (keyhole mask)
        opening_width_um: Opening width in microns (default: 120)
        smoothing_factor: Spline smoothing factor (default: 0.5)
        resolution_um_per_pixel: Spatial resolution (default: 4.3)

    Returns:
        x_coords: X coordinates in microns
        y_coords: Y coordinates in microns
    """
    # Find contours (returns list of contours)
    contours = measure.find_contours(binary_image, level=0.5)

    if len(contours) == 0:
        raise ValueError("No contours found in binary image")

    # Select largest contour (main keyhole)
    largest_contour = max(contours, key=len)

    # Find keyhole center
    keyhole_center_x = find_keyhole_center(largest_contour)

    # Close opening
    closed_contour = close_keyhole_opening(largest_contour, keyhole_center_x,
                                           opening_width_um, resolution_um_per_pixel)

    # Smooth corners
    smoothed_contour = smooth_contour_spline(closed_contour, smoothing_factor)

    # Order from top-right to top-left
    ordered_contour = order_contour_top_right_to_left(smoothed_contour)

    # Convert to microns
    # Contour format: (row, col) = (y, x)
    y_coords_pixels = ordered_contour[:, 0]
    x_coords_pixels = ordered_contour[:, 1]

    x_coords_um = x_coords_pixels * resolution_um_per_pixel
    y_coords_um = y_coords_pixels * resolution_um_per_pixel

    return x_coords_um, y_coords_um


def extract_all_perimeters(hdf5_path: str,
                           dataset_name: str = 'keyhole_bin',
                           opening_width_um: float = 120.0,
                           smoothing_factor: float = 0.5,
                           resolution_um_per_pixel: float = 4.3,
                           frame_range: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
    """
    Extract perimeters from all frames in HDF5 file.

    Parameters:
        hdf5_path: Path to HDF5 file
        dataset_name: Name of binary keyhole dataset
        opening_width_um: Opening width in microns
        smoothing_factor: Spline smoothing factor
        resolution_um_per_pixel: Spatial resolution
        frame_range: Optional (start, end) frame indices to process

    Returns:
        DataFrame with columns: X, Y, Slice
    """
    # Load images
    images, metadata = read_keyhole_images(hdf5_path, dataset_name)
    n_frames = metadata['n_frames']

    # Determine frame range
    if frame_range is None:
        start_frame, end_frame = 0, n_frames
    else:
        start_frame, end_frame = frame_range
        start_frame = max(0, start_frame)
        end_frame = min(n_frames, end_frame)

    # Extract perimeters for all frames
    all_x = []
    all_y = []
    all_slices = []

    for frame_idx in range(start_frame, end_frame):
        try:
            binary_image = images[frame_idx]

            x_coords, y_coords = extract_keyhole_perimeter_with_opening(
                binary_image, opening_width_um, smoothing_factor, resolution_um_per_pixel
            )

            # Add to lists
            all_x.extend(x_coords)
            all_y.extend(y_coords)
            all_slices.extend([frame_idx + 1] * len(x_coords))  # Slice numbering starts at 1

        except Exception as e:
            print(f"Warning: Failed to extract perimeter for frame {frame_idx}: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame({
        'X': all_x,
        'Y': all_y,
        'Slice': all_slices
    })

    return df


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hdf5_keyhole_reader.py <hdf5_file>")
        sys.exit(1)

    hdf5_path = sys.argv[1]

    print(f"Extracting perimeters from: {hdf5_path}")
    df = extract_all_perimeters(hdf5_path)

    print(f"\nExtracted {len(df)} points from {df['Slice'].nunique()} frames")
    print(f"X range: {df['X'].min():.1f} to {df['X'].max():.1f} μm")
    print(f"Y range: {df['Y'].min():.1f} to {df['Y'].max():.1f} μm")
