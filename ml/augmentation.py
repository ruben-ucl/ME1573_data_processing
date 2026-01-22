#!/usr/bin/env python3
"""
Unified Data Augmentation Module

Simplified augmentation system with 3 intuitive parameters:
- augment_probability: Probability that augmentation is applied
- augment_strength: Intensity preset ('low', 'medium', 'high')
- augment_methods: List of augmentation techniques to choose from

Each augmented sample receives exactly ONE randomly selected augmentation
from the methods list, applied at the specified strength level.

Author: Claude Code
Date: 2025-01-22
"""

import numpy as np
import random
import cv2

# ============================================================================
# AUGMENTATION STRENGTH PRESETS
# ============================================================================

AUGMENTATION_PRESETS = {
    'low': {
        'time_shift_range': 2,
        'stretch_scale': 0.05,
        'noise_std': 0.01,
        'amplitude_scale': 0.05,
        'brightness_range': 0.05,
        'contrast_range': 0.05
    },
    'medium': {
        'time_shift_range': 5,
        'stretch_scale': 0.1,
        'noise_std': 0.02,
        'amplitude_scale': 0.1,
        'brightness_range': 0.1,
        'contrast_range': 0.1
    },
    'high': {
        'time_shift_range': 10,
        'stretch_scale': 0.2,
        'noise_std': 0.05,
        'amplitude_scale': 0.2,
        'brightness_range': 0.2,
        'contrast_range': 0.2
    }
}

# Valid augmentation methods per classifier type
VALID_METHODS = {
    'pd_signal': ['time_shift', 'stretch', 'noise', 'amplitude'],
    'cwt_image': ['time_shift', 'noise', 'brightness', 'contrast']
}

# ============================================================================
# CONFIGURATION PARSING
# ============================================================================

def parse_augmentation_config(config, classifier_type='pd_signal'):
    """
    Convert new 3-parameter augmentation format to expanded parameters.

    Args:
        config (dict): Configuration with new augmentation parameters
        classifier_type (str): 'pd_signal' or 'cwt_image'

    Returns:
        dict: Expanded configuration with all parameter values resolved
    """
    # Extract new parameters
    probability = config.get('augment_probability', 0.5)
    strength = config.get('augment_strength', 'medium')
    methods = config.get('augment_methods', [])

    # Validate strength
    if strength not in AUGMENTATION_PRESETS:
        raise ValueError(f"Invalid augment_strength '{strength}'. Must be 'low', 'medium', or 'high'")

    # Validate methods
    valid_for_classifier = VALID_METHODS.get(classifier_type, [])
    for method in methods:
        if method not in valid_for_classifier:
            raise ValueError(f"Invalid method '{method}' for {classifier_type}. Valid: {valid_for_classifier}")

    # Get strength values from preset
    strength_params = AUGMENTATION_PRESETS[strength]

    # Build expanded config
    expanded_config = {
        'augment_probability': probability,
        'augment_strength': strength,
        'augment_methods': methods,
        # Add strength values for all methods (for backward compatibility logging)
        **strength_params
    }

    return expanded_config


def get_legacy_log_values(config, classifier_type='pd_signal'):
    """
    Extract values in legacy format for experiment logging.

    Only logs augmentation parameters when augmentation is actually active.
    Sets all values to 0 when augmentation is disabled.

    Args:
        config (dict): Augmentation config
        classifier_type (str): 'pd_signal' or 'cwt_image'

    Returns:
        dict: Legacy parameter names and values for logging
    """
    augment_probability = config.get('augment_probability', 0.0)
    strength = config.get('augment_strength', 'medium')
    methods = config.get('augment_methods', [])

    # If augmentation is disabled, return all zeros
    if augment_probability == 0.0 or len(methods) == 0:
        if classifier_type == 'pd_signal':
            return {
                'augment_fraction': 0.0,
                'time_shift_probability': 0.0,
                'time_shift_range': 0.0,
                'stretch_probability': 0.0,
                'stretch_scale': 0.0,
                'noise_probability': 0.0,
                'noise_std': 0.0,
                'amplitude_scale_probability': 0.0,
                'amplitude_scale': 0.0,
            }
        else:  # cwt_image
            return {
                'augment_fraction': 0.0,
                'time_shift_probability': 0.0,
                'time_shift_range': 0.0,
                'brightness_probability': 0.0,
                'brightness_range': 0.0,
                'contrast_probability': 0.0,
                'contrast_range': 0.0,
                'noise_probability': 0.0,
                'noise_std': 0.0,
            }

    # Augmentation is active - get preset values
    presets = AUGMENTATION_PRESETS.get(strength, AUGMENTATION_PRESETS['medium'])

    # Base values
    legacy_values = {
        'augment_fraction': augment_probability,
    }

    # Only include parameter values for methods that are actually used
    if classifier_type == 'pd_signal':
        # Time shift
        if 'time_shift' in methods:
            legacy_values['time_shift_probability'] = 1.0
            legacy_values['time_shift_range'] = presets['time_shift_range']
        else:
            legacy_values['time_shift_probability'] = 0.0
            legacy_values['time_shift_range'] = 0.0

        # Stretch
        if 'stretch' in methods:
            legacy_values['stretch_probability'] = 1.0
            legacy_values['stretch_scale'] = presets['stretch_scale']
        else:
            legacy_values['stretch_probability'] = 0.0
            legacy_values['stretch_scale'] = 0.0

        # Noise
        if 'noise' in methods:
            legacy_values['noise_probability'] = 1.0
            legacy_values['noise_std'] = presets['noise_std']
        else:
            legacy_values['noise_probability'] = 0.0
            legacy_values['noise_std'] = 0.0

        # Amplitude
        if 'amplitude' in methods:
            legacy_values['amplitude_scale_probability'] = 1.0
            legacy_values['amplitude_scale'] = presets['amplitude_scale']
        else:
            legacy_values['amplitude_scale_probability'] = 0.0
            legacy_values['amplitude_scale'] = 0.0

    elif classifier_type == 'cwt_image':
        # Time shift
        if 'time_shift' in methods:
            legacy_values['time_shift_probability'] = 1.0
            legacy_values['time_shift_range'] = presets['time_shift_range']
        else:
            legacy_values['time_shift_probability'] = 0.0
            legacy_values['time_shift_range'] = 0.0

        # Noise
        if 'noise' in methods:
            legacy_values['noise_probability'] = 1.0
            legacy_values['noise_std'] = presets['noise_std']
        else:
            legacy_values['noise_probability'] = 0.0
            legacy_values['noise_std'] = 0.0

        # Brightness
        if 'brightness' in methods:
            legacy_values['brightness_probability'] = 1.0
            legacy_values['brightness_range'] = presets['brightness_range']
        else:
            legacy_values['brightness_probability'] = 0.0
            legacy_values['brightness_range'] = 0.0

        # Contrast
        if 'contrast' in methods:
            legacy_values['contrast_probability'] = 1.0
            legacy_values['contrast_range'] = presets['contrast_range']
        else:
            legacy_values['contrast_probability'] = 0.0
            legacy_values['contrast_range'] = 0.0

    return legacy_values

# ============================================================================
# INDIVIDUAL AUGMENTATION METHODS - PD SIGNALS
# ============================================================================

def augment_time_shift(pd1_signal, pd2_signal, time_shift_range):
    """Apply time shift augmentation to dual PD signals."""
    aug_pd1 = pd1_signal.copy()
    aug_pd2 = pd2_signal.copy()

    max_shift = int(time_shift_range)
    shift = np.random.randint(-max_shift, max_shift + 1)
    aug_pd1 = np.roll(aug_pd1, shift, axis=0)
    aug_pd2 = np.roll(aug_pd2, shift, axis=0)

    return aug_pd1, aug_pd2


def augment_stretch(pd1_signal, pd2_signal, stretch_scale):
    """Apply stretch/compression augmentation to dual PD signals."""
    aug_pd1 = pd1_signal.copy()
    aug_pd2 = pd2_signal.copy()

    # Convert scale parameter to min/max range
    scale_min = 1.0 - stretch_scale
    scale_max = 1.0 + stretch_scale
    scale_factor = random.uniform(scale_min, scale_max)
    new_width = int(aug_pd1.shape[0] * scale_factor)

    # Resize both signals
    aug_pd1_2d = cv2.resize(aug_pd1.squeeze(), (1, new_width), interpolation=cv2.INTER_LINEAR).reshape(-1, 1)
    aug_pd2_2d = cv2.resize(aug_pd2.squeeze(), (1, new_width), interpolation=cv2.INTER_LINEAR).reshape(-1, 1)

    # Pad/crop back to original width
    original_width = pd1_signal.shape[0]
    if new_width < original_width:
        pad_width = original_width - new_width
        aug_pd1 = np.pad(aug_pd1_2d, ((0, pad_width), (0, 0)), mode='constant')[:original_width, :]
        aug_pd2 = np.pad(aug_pd2_2d, ((0, pad_width), (0, 0)), mode='constant')[:original_width, :]
    elif new_width > original_width:
        aug_pd1 = aug_pd1_2d[:original_width, :]
        aug_pd2 = aug_pd2_2d[:original_width, :]
    else:
        aug_pd1 = aug_pd1_2d
        aug_pd2 = aug_pd2_2d

    return aug_pd1, aug_pd2


def augment_noise_pd(pd1_signal, pd2_signal, noise_std):
    """Apply Gaussian noise to dual PD signals."""
    aug_pd1 = pd1_signal.copy()
    aug_pd2 = pd2_signal.copy()

    noise_pd1 = np.random.normal(0, noise_std, aug_pd1.shape)
    noise_pd2 = np.random.normal(0, noise_std, aug_pd2.shape)
    aug_pd1 = np.clip(aug_pd1 + noise_pd1, 0.0, 1.0)
    aug_pd2 = np.clip(aug_pd2 + noise_pd2, 0.0, 1.0)

    return aug_pd1, aug_pd2


def augment_amplitude(pd1_signal, pd2_signal, amplitude_scale):
    """Apply amplitude scaling to dual PD signals."""
    aug_pd1 = pd1_signal.copy()
    aug_pd2 = pd2_signal.copy()

    # Convert scale parameter to min/max range
    scale_min = 1.0 - amplitude_scale
    scale_max = 1.0 + amplitude_scale

    # Scale each signal independently
    scale_pd1 = random.uniform(scale_min, scale_max)
    aug_pd1 = np.clip(aug_pd1 * scale_pd1, 0.0, 1.0)

    scale_pd2 = random.uniform(scale_min, scale_max)
    aug_pd2 = np.clip(aug_pd2 * scale_pd2, 0.0, 1.0)

    return aug_pd1, aug_pd2

# ============================================================================
# INDIVIDUAL AUGMENTATION METHODS - CWT IMAGES
# ============================================================================

def augment_time_shift_image(image, time_shift_range):
    """Apply time shift (horizontal shift) to CWT image."""
    aug_image = image.copy()

    max_shift = int(time_shift_range)
    shift = np.random.randint(-max_shift, max_shift + 1)

    if shift != 0:
        aug_image = np.roll(aug_image, shift, axis=1)  # Shift along width (time)

    return aug_image


def augment_noise_image(image, noise_std):
    """Apply Gaussian noise to CWT image."""
    aug_image = image.copy()

    noise = np.random.normal(0, noise_std, aug_image.shape)
    aug_image = aug_image + noise
    aug_image = np.clip(aug_image, 0, 1)

    return aug_image


def augment_brightness(image, brightness_range):
    """Apply brightness variation to CWT image."""
    aug_image = image.copy()

    brightness_factor = 1.0 + np.random.uniform(-brightness_range, brightness_range)
    aug_image = aug_image * brightness_factor
    aug_image = np.clip(aug_image, 0, 1)

    return aug_image


def augment_contrast(image, contrast_range):
    """Apply contrast variation to CWT image."""
    aug_image = image.copy()

    contrast_factor = 1.0 + np.random.uniform(-contrast_range, contrast_range)
    mean_val = np.mean(aug_image)
    aug_image = mean_val + (aug_image - mean_val) * contrast_factor
    aug_image = np.clip(aug_image, 0, 1)

    return aug_image

# ============================================================================
# MAIN AUGMENTATION FUNCTIONS
# ============================================================================

def apply_single_augmentation_pd(pd1_signal, pd2_signal, method, strength_params):
    """
    Apply a single augmentation method to dual PD signals.

    Args:
        pd1_signal: PD1 signal array
        pd2_signal: PD2 signal array
        method (str): Augmentation method name
        strength_params (dict): Strength parameters for all methods

    Returns:
        tuple: (augmented_pd1, augmented_pd2)
    """
    if method == 'time_shift':
        return augment_time_shift(pd1_signal, pd2_signal, strength_params['time_shift_range'])
    elif method == 'stretch':
        return augment_stretch(pd1_signal, pd2_signal, strength_params['stretch_scale'])
    elif method == 'noise':
        return augment_noise_pd(pd1_signal, pd2_signal, strength_params['noise_std'])
    elif method == 'amplitude':
        return augment_amplitude(pd1_signal, pd2_signal, strength_params['amplitude_scale'])
    else:
        raise ValueError(f"Unknown PD augmentation method: {method}")


def apply_single_augmentation_image(image, method, strength_params):
    """
    Apply a single augmentation method to a CWT image.

    Args:
        image: Image array
        method (str): Augmentation method name
        strength_params (dict): Strength parameters for all methods

    Returns:
        np.ndarray: Augmented image
    """
    if method == 'time_shift':
        return augment_time_shift_image(image, strength_params['time_shift_range'])
    elif method == 'noise':
        return augment_noise_image(image, strength_params['noise_std'])
    elif method == 'brightness':
        return augment_brightness(image, strength_params['brightness_range'])
    elif method == 'contrast':
        return augment_contrast(image, strength_params['contrast_range'])
    else:
        raise ValueError(f"Unknown CWT image augmentation method: {method}")


def augment_pd_sample(pd1_signal, pd2_signal, config):
    """
    Augment a single dual-signal PD sample using new simplified system.

    Applies ONE randomly selected augmentation from augment_methods list.

    Args:
        pd1_signal: PD1 signal array
        pd2_signal: PD2 signal array
        config (dict): Configuration with augmentation parameters

    Returns:
        tuple: (augmented_pd1, augmented_pd2)
    """
    # Parse config to get expanded parameters
    expanded_config = parse_augmentation_config(config, classifier_type='pd_signal')

    methods = expanded_config['augment_methods']
    if not methods:
        # No augmentation methods specified
        return pd1_signal.copy(), pd2_signal.copy()

    # Randomly select ONE method
    method = random.choice(methods)

    # Get strength parameters
    strength = expanded_config['augment_strength']
    strength_params = AUGMENTATION_PRESETS[strength]

    # Apply the selected method
    return apply_single_augmentation_pd(pd1_signal, pd2_signal, method, strength_params)


def augment_cwt_image(image, config):
    """
    Augment a single CWT image using new simplified system.

    Applies ONE randomly selected augmentation from augment_methods list.

    Args:
        image: CWT image array
        config (dict): Configuration with augmentation parameters

    Returns:
        np.ndarray: Augmented image
    """
    # Parse config to get expanded parameters
    expanded_config = parse_augmentation_config(config, classifier_type='cwt_image')

    methods = expanded_config['augment_methods']
    if not methods:
        # No augmentation methods specified
        return image.copy()

    # Randomly select ONE method
    method = random.choice(methods)

    # Get strength parameters
    strength = expanded_config['augment_strength']
    strength_params = AUGMENTATION_PRESETS[strength]

    # Apply the selected method
    return apply_single_augmentation_image(image, method, strength_params)
