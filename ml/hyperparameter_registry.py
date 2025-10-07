#!/usr/bin/env python3
"""
Centralized hyperparameter registry for ML components.
Single source of truth for parameter names, types, ranges, and defaults.
"""

import os
from itertools import product

# Set UTF-8 encoding for Windows compatibility
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

HYPERPARAMETER_REGISTRY = {
    # =====================================================================
    # TRAINING CATEGORY
    # =====================================================================
    
    # Tier 1 (Critical)
    'learning_rate': {
        'type': 'continuous',
        'category': 'training',
        'tier': 1,
        'default': 0.001,
        'search_space': [0.0001, 0.0005, 0.001, 0.002, 0.005],
        'doe_range': (0.0001, 0.01),
        'log_scale': True,
        'description': 'Learning rate for optimizer - critical for convergence'
    },
    'batch_size': {
        'type': 'discrete',
        'category': 'training',
        'tier': 1,
        'default': 16,
        'search_space': {
            'pd_signal': [8, 16, 32],
            'cwt_image': [8, 16, 32, 64, 128]
        },
        'doe_range': {
            'pd_signal': (8, 32),
            'cwt_image': (8, 128)
        },
        'description': 'Batch size for training - affects gradient quality and memory'
    },

    # =====================================================================
    # REGULARIZATION CATEGORY
    # =====================================================================
    
    # Tier 1 (Critical)
    'conv_dropout': {
        'type': 'continuous',
        'category': 'regularization',
        'tier': 1,
        'default': {
            'pd_signal': 0.2,
            'cwt_image': 0.0
        },
        'search_space': {
            'pd_signal': [0.1, 0.2, 0.3],
            'cwt_image': [0.0, 0.1, 0.2, 0.3]
        },
        'doe_range': (0.0, 0.5),
        'description': 'Dropout rate for convolutional layers - critical for overfitting control'
    },
    'dense_dropout': {
        'type': 'categorical',
        'category': 'regularization',
        'tier': 1,
        'default': {
            'pd_signal': [0.3, 0.2],
            'cwt_image': [0.5]
        },
        'search_space': {
            'pd_signal': [[0.2, 0.1], [0.2, 0.2], [0.3, 0.2], [0.3, 0.3], [0.4, 0.3]],
            'cwt_image': [[0.1], [0.2], [0.3], [0.5], [0.3, 0.2], [0.5, 0.3]]
        },
        'description': 'Dropout rates for dense layers - critical for overfitting control'
    },
    
    # Tier 2 (High Impact)
    'l2_regularization': {
        'type': 'continuous',
        'category': 'regularization',
        'tier': 2,
        'default': 0.001,
        'search_space': [0.0, 0.0001, 0.001, 0.01],
        'doe_range': (0.0, 0.1),
        'log_scale': True,
        'description': 'L2 regularization strength - high impact on overfitting'
    },
    
    # Tier 3 (Moderate Impact)
    'use_batch_norm': {
        'type': 'categorical',
        'category': 'regularization',
        'tier': 3,
        'default': {
            'pd_signal': True,
            'cwt_image': False
        },
        'search_space': [True, False],
        'description': 'Whether to use batch normalization - moderate impact on stability'
    },

    # =====================================================================
    # ARCHITECTURE CATEGORY
    # =====================================================================
    
    # Tier 2 (High Impact)
    'conv_filters': {
        'type': 'categorical',
        'category': 'architecture',
        'tier': 2,
        'default': {
            'pd_signal': [16, 32, 64],
            'cwt_image': [16, 16, 32, 32, 64, 64]
        },
        'search_space': {
            'pd_signal': [
                [16, 32],
                [32, 64],
                [16, 32, 64],
                [32, 64, 128]
            ],
            'cwt_image': [
                [16, 32, 64],
                [16, 32, 32, 64],
                [32, 32, 64, 64],
                [16, 32, 64, 64, 128],
                [16, 16, 32, 32, 64, 64],
                [32, 32, 64, 64, 128, 128]
            ]
        },
        'description': 'Number of filters in each convolutional layer - high impact on model capacity'
    },
    'dense_units': {
        'type': 'categorical',
        'category': 'architecture',
        'tier': 2,
        'default': {
            'pd_signal': [128, 64],
            'cwt_image': [128]
        },
        'search_space': {
            'pd_signal': [
                [64],
                [128],
                [128, 64],
                [256, 128]
            ],
            'cwt_image': [
                [64],
                [128],
                [128, 64],
                [256, 128],
                [512, 256]
            ]
        },
        'description': 'Number of units in each dense layer - high impact on model capacity'
    },

    # =====================================================================
    # TRAINING_CONTROL CATEGORY
    # =====================================================================
    
    # Tier 3 (Moderate Impact)
    'early_stopping_patience': {
        'type': 'discrete',
        'category': 'training_control',
        'tier': 3,
        'default': 10,
        'search_space': {
            'pd_signal': [8, 12],
            'cwt_image': [5, 10, 15]
        },
        'doe_range': (5, 20),
        'description': 'Epochs to wait before early stopping - moderate impact on training efficiency'
    },
    'use_class_weights': {
        'type': 'categorical',
        'category': 'training_control',
        'tier': 3,
        'default': True,
        'search_space': [True, False],
        'description': 'Whether to use class weights for imbalanced data - moderate impact on class balance'
    },
    
    # Tier 4 (Low Impact)
    'lr_reduction_patience': {
        'type': 'discrete',
        'category': 'training_control',
        'tier': 4,
        'default': 5,
        'search_space': [3, 5, 7],
        'doe_range': (3, 10),
        'description': 'Epochs to wait before reducing learning rate - low impact on final performance'
    },
    'lr_reduction_factor': {
        'type': 'continuous',
        'category': 'training_control',
        'tier': 4,
        'default': 0.5,
        'search_space': [0.3, 0.5, 0.7],
        'doe_range': (0.1, 0.9),
        'description': 'Factor to reduce learning rate by - low impact on final performance'
    },

    # =====================================================================
    # AUGMENTATION CATEGORY
    # =====================================================================
    
    # Tier 2 (High Impact)
    'augment_fraction': {
        'type': 'continuous',
        'category': 'augmentation',
        'tier': 2,
        'default': {
            'pd_signal': 0.5,
            'cwt_image': 0.0
        },
        'search_space': [0.0, 0.25, 0.5, 0.75, 1.0],
        'doe_range': (0.0, 1.0),
        'description': 'Fraction of data to augment - high impact in data-limited scenarios'
    },
    
    # Tier 3 (Moderate Impact)
    'time_shift_probability': {
        'type': 'continuous',
        'category': 'augmentation',
        'tier': 3,
        'default': {
            'pd_signal': 0.7,
            'cwt_image': 0.0
        },
        'search_space': [0.0, 0.25, 0.5, 0.75, 1.0],
        'doe_range': (0.0, 1.0),
        'description': 'Probability of applying time shift augmentation - moderate impact for PD signals'
    },
    
    # Tier 4 (Low Impact)
    'noise_probability': {
        'type': 'continuous',
        'category': 'augmentation',
        'tier': 4,
        'default': {
            'pd_signal': 0.5,
            'cwt_image': 0.0
        },
        'search_space': [0.0, 0.25, 0.5, 0.75, 1.0],
        'doe_range': (0.0, 1.0),
        'description': 'Probability of applying noise augmentation - low impact on performance'
    },
    
    # Tier 5 (Minimal Impact)
    'time_shift_range': {
        'type': 'discrete',
        'category': 'augmentation',
        'tier': 5,
        'default': {
            'pd_signal': 5,
            'cwt_image': 0
        },
        'search_space': {
            'pd_signal': [2, 5, 10],
            'cwt_image': [0, 10]
        },
        'doe_range': (0, 20),
        'description': 'Range for time shift augmentation - minimal impact on performance'
    },
    'noise_std': {
        'type': 'continuous',
        'category': 'augmentation',
        'tier': 5,
        'default': {
            'pd_signal': 0.02,
            'cwt_image': 0.0
        },
        'search_space': {
            'pd_signal': [0.01, 0.02, 0.03, 0.05],
            'cwt_image': [0.0, 0.02]
        },
        'doe_range': (0.0, 0.1),
        'description': 'Standard deviation for noise augmentation - minimal impact on performance'
    },
    
    # PD-specific augmentation (Tier 5)
    'stretch_probability': {
        'type': 'continuous',
        'category': 'augmentation',
        'tier': 5,
        'default': 0.3,
        'search_space': [0.0, 0.25, 0.5, 0.75, 1.0],
        'doe_range': (0.0, 1.0),
        'description': 'Probability of applying stretch augmentation (PD only)',
        'classifiers': ['pd_signal']
    },
    'stretch_scale': {
        'type': 'continuous',
        'category': 'augmentation',
        'tier': 5,
        'default': 0.1,
        'search_space': [0.005, 0.1, 0.15, 0.2],
        'doe_range': (0.0, 0.3),
        'description': 'Scale for stretch augmentation (PD only)',
        'classifiers': ['pd_signal']
    },
    'amplitude_scale_probability': {
        'type': 'continuous',
        'category': 'augmentation',
        'tier': 5,
        'default': 0.5,
        'search_space': [0.0, 0.25, 0.5, 0.75, 1.0],
        'doe_range': (0.0, 1.0),
        'description': 'Probability of applying amplitude scaling (PD only)',
        'classifiers': ['pd_signal']
    },
    'amplitude_scale': {
        'type': 'continuous',
        'category': 'augmentation',
        'tier': 5,
        'default': 0.1,
        'search_space': [0.005, 0.1, 0.15, 0.2],
        'doe_range': (0.0, 0.3),
        'description': 'Scale for amplitude augmentation (PD only)',
        'classifiers': ['pd_signal']
    },
    
    # CWT-specific augmentation (Tier 5)
    'brightness_probability': {
        'type': 'continuous',
        'category': 'augmentation',
        'tier': 5,
        'default': 0.0,
        'search_space': [0.0, 0.5, 1.0],
        'doe_range': (0.0, 1.0),
        'description': 'Probability of applying brightness augmentation (CWT only)',
        'classifiers': ['cwt_image']
    },
    'brightness_range': {
        'type': 'continuous',
        'category': 'augmentation',
        'tier': 5,
        'default': 0.0,
        'search_space': [0.0, 0.1],
        'doe_range': (0.0, 0.3),
        'description': 'Range for brightness augmentation (CWT only)',
        'classifiers': ['cwt_image']
    },
    'contrast_probability': {
        'type': 'continuous',
        'category': 'augmentation',
        'tier': 5,
        'default': 0.0,
        'search_space': [0.0, 0.5, 1.0],
        'doe_range': (0.0, 1.0),
        'description': 'Probability of applying contrast augmentation (CWT only)',
        'classifiers': ['cwt_image']
    },
    'contrast_range': {
        'type': 'continuous',
        'category': 'augmentation',
        'tier': 5,
        'default': 0.0,
        'search_space': [0.0, 0.1],
        'doe_range': (0.0, 0.3),
        'description': 'Range for contrast augmentation (CWT only)',
        'classifiers': ['cwt_image']
    },

    # =====================================================================
    # FIXED CATEGORY (Not typically optimized in search)
    # =====================================================================
    
    'epochs': {
        'type': 'discrete',
        'category': 'fixed',
        'tier': 'fixed',
        'default': 50,
        'search_space': [30, 50, 75, 100],
        'doe_range': (20, 100),
        'description': 'Number of training epochs - typically fixed with early stopping'
    },
    'k_folds': {
        'type': 'discrete',
        'category': 'fixed',
        'tier': 'fixed',
        'default': 5,
        'search_space': [3, 5, 7],
        'doe_range': (3, 10),
        'description': 'Number of cross-validation folds - typically fixed for consistency'
    },
    'optimizer': {
        'type': 'categorical',
        'category': 'fixed',
        'tier': 'fixed',
        'default': 'adam',
        'search_space': ['adam', 'rmsprop', 'sgd', 'adamw'],
        'description': 'Optimizer algorithm - typically fixed as Adam for most cases'
    },
    'conv_kernel_size': {
        'type': 'categorical',
        'category': 'fixed',
        'tier': 'fixed',
        'default': [3, 3],
        'search_space': {
            'cwt_image': [[3, 3], [3, 5], [5, 5]]
        },
        'description': 'Kernel size for convolutional layers - typically fixed'
    },
    'pool_size': {
        'type': 'categorical',
        'category': 'fixed',
        'tier': 'fixed',
        'default': [2, 2],
        'search_space': {
            'cwt_image': [[2, 2], [3, 3]]
        },
        'description': 'Pooling size for max pooling layers - typically fixed'
    },
    'pool_layers': {
        'type': 'categorical',
        'category': 'fixed',
        'tier': 'fixed',
        'default': [2, 5],
        'search_space': {
            'cwt_image': [
                [1, 2],
                [1, 3],
                [2, 4],
                [2, 5],
                [0, 2],
                [1, 4]
            ]
        },
        'description': 'Which conv layers to add pooling after - typically fixed'
    }
}


def get_categories():
    """Generate category list from registry."""
    categories = set()
    for param_info in HYPERPARAMETER_REGISTRY.values():
        categories.add(param_info['category'])
    return sorted(list(categories))


def get_tiers():
    """Generate tier list from registry."""
    tiers = set()
    for param_info in HYPERPARAMETER_REGISTRY.values():
        tier = param_info['tier']
        if tier != 'fixed':
            tiers.add(tier)
    return sorted(list(tiers))


def get_parameters_by_category(category):
    """Get all parameters in a specific category."""
    params = []
    for param_name, param_info in HYPERPARAMETER_REGISTRY.items():
        if param_info['category'] == category:
            params.append(param_name)
    return params


def get_parameters_by_tier(tier):
    """Get all parameters in a specific tier."""
    params = []
    for param_name, param_info in HYPERPARAMETER_REGISTRY.items():
        if param_info['tier'] == tier:
            params.append(param_name)
    return params


def get_parameter_info(param_name, classifier_type=None):
    """Get parameter information from registry.
    
    Args:
        param_name (str): Parameter name
        classifier_type (str, optional): 'pd_signal' or 'cwt_image'
        
    Returns:
        dict: Parameter information with classifier-specific values resolved
    """
    if param_name not in HYPERPARAMETER_REGISTRY:
        raise ValueError(f"Unknown parameter: {param_name}")
    
    info = HYPERPARAMETER_REGISTRY[param_name].copy()
    
    # Check if parameter is applicable to this classifier
    if 'classifiers' in info and classifier_type:
        if classifier_type not in info['classifiers']:
            return None
    
    # Resolve classifier-specific values
    if classifier_type:
        for key in ['default', 'search_space', 'doe_range']:
            if key in info and isinstance(info[key], dict):
                if classifier_type in info[key]:
                    info[key] = info[key][classifier_type]
                else:
                    # If classifier-specific value not found, remove the key
                    del info[key]
    
    return info


def get_search_space(classifier_type, categories=None, tiers=None, parameters=None):
    """Generate search space with flexible filtering.
    
    Args:
        classifier_type (str): 'pd_signal' or 'cwt_image'
        categories (list, optional): Filter by parameter categories
        tiers (list, optional): Filter by priority tiers
        parameters (list, optional): Specific parameters to include
        
    Returns:
        dict: Filtered parameter search space
    """
    space = {}
    
    # Start with all parameters or specific ones
    params_to_check = parameters if parameters else HYPERPARAMETER_REGISTRY.keys()
    
    for param_name in params_to_check:
        param_info = get_parameter_info(param_name, classifier_type)
        
        # Skip if parameter not applicable to this classifier
        if param_info is None:
            continue
            
        # Apply category filter
        if categories:
            param_category = param_info.get('category')
            if param_category not in categories:
                continue
        
        # Apply tier filter
        if tiers:
            param_tier = param_info.get('tier')
            if param_tier not in tiers:
                continue
        
        # Add to search space
        if 'search_space' in param_info:
            space[param_name] = param_info['search_space']
    
    return space


def get_default_config(classifier_type):
    """Generate default configuration from registry.
    
    Args:
        classifier_type (str): 'pd_signal' or 'cwt_image'
        
    Returns:
        dict: Default configuration
    """
    config = {}
    
    for param_name in HYPERPARAMETER_REGISTRY:
        param_info = get_parameter_info(param_name, classifier_type)
        
        # Skip if parameter not applicable to this classifier
        if param_info is None:
            continue
            
        if 'default' in param_info:
            config[param_name] = param_info['default']
    
    return config