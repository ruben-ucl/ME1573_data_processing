#!/usr/bin/env python3
"""
Data Processing Utilities for ML Pipeline

Shared utilities for image processing, normalization, and data handling
to ensure consistency between training and testing pipelines.

Author: AI Assistant  
"""

import numpy as np


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

