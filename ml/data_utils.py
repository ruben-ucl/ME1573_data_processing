#!/usr/bin/env python3
"""
Data Processing Utilities for ML Pipeline

Shared utilities for image processing, normalization, and data handling
to ensure consistency between training and testing pipelines.

Author: AI Assistant  
"""

import numpy as np
from pathlib import Path


def normalize_image(img):
    """
    Centralized image normalization logic matching training pipeline.
    
    Args:
        img: Input image array
        
    Returns:
        np.array: Normalized image in float32 format [0,1]
    """
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    else:
        # Unknown format - normalize to [0,1] based on actual range
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            return (img.astype(np.float32) - img_min) / (img_max - img_min)
        else:
            return img.astype(np.float32)


def split_dual_branch_image(img, img_width):
    """
    Split resized image into dual photodiode signals for dual-branch model.
    Handles both (img_width, 2) and (2, img_width) orientations.
    
    Args:
        img: Resized image of shape (img_width, 2) or (2, img_width)
        img_width: Image width for validation
        
    Returns:
        tuple: (pd1_signal, pd2_signal) each of shape (img_width, 1)
    """
    # Handle different image orientations
    if img.shape == (img_width, 2):
        # Standard orientation: (width, 2)
        pd1_signal = img[:, 0].reshape(-1, 1)  # First column: (width, 1)
        pd2_signal = img[:, 1].reshape(-1, 1)  # Second column: (width, 1)
    elif img.shape == (2, img_width):
        # Transposed orientation: (2, width) - transpose to (width, 2)
        img = img.T  # Transpose to (width, 2)
        pd1_signal = img[:, 0].reshape(-1, 1)  # First column: (width, 1)
        pd2_signal = img[:, 1].reshape(-1, 1)  # Second column: (width, 1)
    else:
        raise ValueError(f"Expected image shape ({img_width}, 2) or (2, {img_width}), got {img.shape}")
    
    return pd1_signal, pd2_signal


def estimate_memory_usage_gb(num_files, img_width, signals_per_image=2):
    """
    Estimate memory usage in GB for loading images into memory.

    Args:
        num_files: Number of image files
        img_width: Width of each image after resizing
        signals_per_image: Number of signals per image (default 2 for dual-branch)

    Returns:
        float: Estimated memory usage in GB
    """
    # Each signal: (img_width, 1) float32 values = img_width * 4 bytes
    bytes_per_image = img_width * 1 * 4 * signals_per_image
    total_bytes = num_files * bytes_per_image
    return total_bytes / (1024**3)  # Convert to GB


def load_cwt_test_images(test_files, test_labels, img_width, img_height, img_channels,
                         channel_paths=None, verbose=False):
    """
    Load CWT test images from file paths.

    Args:
        test_files: Array of file paths (single-channel) or filenames (multi-channel)
        test_labels: Array of integer labels
        img_width: Target image width
        img_height: Target image height
        img_channels: Number of channels expected
        channel_paths: List of channel directories for multi-channel, or None
        verbose: Print warnings for skipped files

    Returns:
        tuple: (X_test np.ndarray, y_test_filtered list, files_filtered list)
    """
    import cv2
    X_test, files_filtered, y_test_filtered = [], [], []
    is_multi_channel = channel_paths is not None and len(channel_paths) > 1

    for class_label in [0, 1]:
        class_files = test_files[test_labels == class_label]
        for file_path_or_name in class_files:
            try:
                if is_multi_channel:
                    filename = (Path(file_path_or_name).name
                                if ('/' in str(file_path_or_name) or '\\' in str(file_path_or_name))
                                else file_path_or_name)
                    channels = []
                    for channel_path in channel_paths:
                        img_path = Path(channel_path) / filename
                        if not img_path.exists():
                            raise FileNotFoundError(f"Image not found: {img_path}")
                        channel_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                        if channel_img is None:
                            raise ValueError(f"Failed to read image: {img_path}")
                        channel_img = cv2.resize(channel_img, (img_width, img_height))
                        channel_img = channel_img.astype(np.float32) / 255.0
                        channels.append(channel_img)
                    img = np.stack(channels, axis=-1)
                    files_filtered.append(filename)
                else:
                    file_path = file_path_or_name
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        raise ValueError(f"Failed to read image: {file_path}")
                    img = cv2.resize(img, (img_width, img_height))
                    if img_channels == 1:
                        img = np.expand_dims(img, axis=-1)
                    img = img.astype(np.float32) / 255.0
                    files_filtered.append(file_path)
                X_test.append(img)
                y_test_filtered.append(class_label)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load CWT test image {file_path_or_name}: {e}")

    return np.array(X_test), y_test_filtered, files_filtered


def load_pd_test_images(test_files, test_labels, img_width, verbose=False):
    """
    Load PD test images from file paths into dual-branch format.

    Args:
        test_files: Array of file paths
        test_labels: Array of integer labels
        img_width: Expected signal length
        verbose: Print warnings for skipped files

    Returns:
        tuple: ((pd1_array, pd2_array), y_test_filtered list, files_filtered list)
    """
    import cv2
    pd1_test, pd2_test, y_test_filtered, files_filtered = [], [], [], []

    for class_label in [0, 1]:
        class_files = test_files[test_labels == class_label]
        for file_path in class_files:
            try:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
                if img is None:
                    raise ValueError(f"Failed to read image: {file_path}")
                if img.shape == (2, img_width):
                    img = img.T
                elif img.shape == (img_width, 2):
                    pass
                else:
                    raise ValueError(f"Unexpected shape {img.shape}; expected (2,{img_width}) or ({img_width},2)")
                assert img.shape == (img_width, 2)
                img = normalize_image(img)
                pd1_signal, pd2_signal = split_dual_branch_image(img, img_width)
                pd1_test.append(pd1_signal)
                pd2_test.append(pd2_signal)
                y_test_filtered.append(class_label)
                files_filtered.append(file_path)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load PD test image {file_path}: {e}")

    return (np.array(pd1_test), np.array(pd2_test)), y_test_filtered, files_filtered


def extract_trackid_from_filename(filename):
    """
    Extract trackid from filename using simple split logic.

    Expected format: XXXX_YY_...rest.ext → XXXX_YY
    Example: "0105_01_0.2-1.2ms.png" → "0105_01"

    Args:
        filename: Filename string or Path object

    Returns:
        str: Trackid (e.g., "0105_01") or None if format doesn't match
    """
    from pathlib import Path

    if isinstance(filename, Path):
        filename = filename.name

    stem = Path(filename).stem
    parts = stem.split('_')

    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"

    return None

