"""
Model Complexity Calculator using Keras Built-in Methods

This module provides accurate model complexity calculation by actually building
the Keras models and using built-in parameter counting methods instead of 
manual estimation.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


def create_pd_model_from_config(config):
    """
    Create PD signal classifier model from config to get exact parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tf.keras.Model: Compiled model for parameter counting
    """
    model = models.Sequential()
    
    # Extract configuration
    conv_filters = config['conv_filters']
    dense_units = config['dense_units']
    l2_reg = config.get('l2_regularization', 0.001)
    use_batch_norm = config.get('use_batch_norm', False)
    
    # Input shape for PD signals: (width, height, channels)
    img_width = config.get('img_width', 100)
    img_height = 2  # PD signals are always 2 pixels high
    input_shape = (img_height, img_width, 1)  # (2, 100, 1) - note TF expects (H, W, C)
    
    # Add convolutional layers (use appropriate kernel size for PD signals)
    # PD signals are only 2 pixels high, so use (1, 3) kernels
    kernel_size = (1, 3)  # Height=1, Width=3 for PD temporal signals
    
    for i, filters in enumerate(conv_filters):
        if i == 0:
            model.add(layers.Conv2D(
                filters, kernel_size, activation='relu',
                input_shape=input_shape,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
                name=f'conv2d_{i}'
            ))
        else:
            model.add(layers.Conv2D(
                filters, kernel_size, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
                name=f'conv2d_{i}'
            ))
        
        if use_batch_norm:
            model.add(layers.BatchNormalization(name=f'batch_norm_{i}'))
        
        # Add pooling (PD typically uses 2x1 pooling to preserve height)
        model.add(layers.MaxPooling2D((2, 1), name=f'max_pool_{i}'))
    
    # Flatten
    model.add(layers.Flatten(name='flatten'))
    
    # Dense layers
    for i, units in enumerate(dense_units):
        model.add(layers.Dense(
            units, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
            name=f'dense_{i}'
        ))
        
        if use_batch_norm:
            model.add(layers.BatchNormalization(name=f'dense_batch_norm_{i}'))
    
    # Output layer (binary classification)
    model.add(layers.Dense(1, activation='sigmoid', name='output'))
    
    return model


def create_cwt_model_from_config(config):
    """
    Create CWT image classifier model from config to get exact parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tf.keras.Model: Compiled model for parameter counting
    """
    model = models.Sequential()
    
    # Extract configuration
    conv_filters = config['conv_filters']
    kernel_size = tuple(config.get('conv_kernel_size', [3, 3]))
    pool_size = tuple(config.get('pool_size', [2, 2]))
    pool_layers = config.get('pool_layers', [])
    dense_units = config['dense_units']
    l2_reg = config.get('l2_regularization', 0.001)
    use_batch_norm = config.get('use_batch_norm', False)
    
    # Input shape for CWT images
    img_height = config.get('img_height', 256)
    img_width = config.get('img_width', 100)
    img_channels = config.get('img_channels', 1)
    input_shape = (img_height, img_width, img_channels)
    
    # Add convolutional layers
    for i, filters in enumerate(conv_filters):
        if i == 0:
            model.add(layers.Conv2D(
                filters, kernel_size, activation='relu',
                input_shape=input_shape,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
                name=f'conv2d_{i}'
            ))
        else:
            model.add(layers.Conv2D(
                filters, kernel_size, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
                name=f'conv2d_{i}'
            ))
        
        if use_batch_norm:
            model.add(layers.BatchNormalization(name=f'batch_norm_{i}'))
        
        # Add pooling after specified layers
        if i in pool_layers:
            model.add(layers.MaxPooling2D(pool_size, name=f'max_pool_{i}'))
    
    # Flatten
    model.add(layers.Flatten(name='flatten'))
    
    # Dense layers
    for i, units in enumerate(dense_units):
        model.add(layers.Dense(
            units, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
            name=f'dense_{i}'
        ))
        
        if use_batch_norm:
            model.add(layers.BatchNormalization(name=f'dense_batch_norm_{i}'))
    
    # Output layer (binary classification)
    model.add(layers.Dense(1, activation='sigmoid', name='output'))
    
    return model


def calculate_model_complexity(config, classifier_type='pd_signal', include_flops=False):
    """
    Calculate exact model complexity using Keras built-in methods.
    
    Args:
        config: Configuration dictionary
        classifier_type: 'pd_signal' or 'cwt_image'
        include_flops: Whether to estimate FLOPs (more computationally expensive)
        
    Returns:
        dict: Complexity metrics including parameter count, layer breakdown, etc.
    """
    try:
        # Build the actual model based on type
        if classifier_type == 'cwt_image':
            model = create_cwt_model_from_config(config)
        else:
            model = create_pd_model_from_config(config)
        
        # Get exact parameter count
        total_params = model.count_params()
        
        # Get layer-wise breakdown
        layer_params = []
        for layer in model.layers:
            if hasattr(layer, 'count_params'):
                layer_info = {
                    'name': layer.name,
                    'type': type(layer).__name__,
                    'params': layer.count_params(),
                    'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'Unknown'
                }
                layer_params.append(layer_info)
        
        # Basic FLOPs estimation (optional)
        estimated_flops = None
        if include_flops:
            estimated_flops = estimate_model_flops(model, config)
        
        complexity_metrics = {
            'total_parameters': total_params,
            'layer_breakdown': layer_params,
            'estimated_flops': estimated_flops,
            'model_summary': get_model_summary_string(model),
            'classifier_type': classifier_type,
            'config_signature': str(sorted(config.items()))
        }
        
        # Clean up model to free memory
        del model
        tf.keras.backend.clear_session()
        
        return complexity_metrics
        
    except Exception as e:
        print(f"Error calculating model complexity: {e}")
        return {
            'total_parameters': 0,
            'error': str(e),
            'classifier_type': classifier_type
        }


def estimate_model_flops(model, config):
    """
    Estimate FLOPs for the model (simplified calculation).
    
    Args:
        model: Keras model
        config: Configuration dictionary
        
    Returns:
        int: Estimated FLOPs
    """
    # This is a simplified FLOP estimation
    # For more accurate FLOPs, you'd need TensorFlow Profiler
    
    total_flops = 0
    
    # Estimate based on layer types and operations
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            # Conv2D FLOPs = output_elements * (kernel_size^2 * input_channels + 1)
            try:
                output_shape = layer.output_shape
                kernel_size = layer.kernel_size
                input_channels = layer.input_shape[-1] if hasattr(layer, 'input_shape') else 1
                
                if output_shape and len(output_shape) >= 3:
                    output_elements = output_shape[1] * output_shape[2] * output_shape[3]
                    kernel_ops = kernel_size[0] * kernel_size[1] * input_channels + 1
                    layer_flops = output_elements * kernel_ops
                    total_flops += layer_flops
            except:
                pass
        
        elif isinstance(layer, tf.keras.layers.Dense):
            # Dense FLOPs = input_size * output_size + output_size (for bias)
            try:
                if hasattr(layer, 'units'):
                    output_size = layer.units
                    input_size = layer.input_shape[-1] if hasattr(layer, 'input_shape') else 1
                    layer_flops = input_size * output_size + output_size
                    total_flops += layer_flops
            except:
                pass
    
    return total_flops


def get_model_summary_string(model):
    """
    Get model summary as a string for logging.
    
    Args:
        model: Keras model
        
    Returns:
        str: Model summary
    """
    import io
    from contextlib import redirect_stdout
    
    summary_buffer = io.StringIO()
    try:
        with redirect_stdout(summary_buffer):
            model.summary()
        return summary_buffer.getvalue()
    except:
        return "Summary not available"


def test_complexity_calculator():
    """Test the complexity calculator with sample configurations."""
    
    # Test PD configuration
    pd_config = {
        'conv_filters': [16, 32, 64],
        'dense_units': [128, 64],
        'img_width': 100,
        'l2_regularization': 0.001,
        'use_batch_norm': False
    }
    
    # Test CWT configuration  
    cwt_config = {
        'conv_filters': [16, 16, 32, 32, 64, 64],
        'conv_kernel_size': [3, 3],
        'pool_size': [2, 2],
        'pool_layers': [2, 5],
        'dense_units': [128],
        'img_width': 100,
        'img_height': 256,
        'img_channels': 1,
        'l2_regularization': 0.001,
        'use_batch_norm': False
    }
    
    print("Testing Model Complexity Calculator")
    print("=" * 50)
    
    # Test PD model
    print("\\nPD Signal Classifier:")
    pd_complexity = calculate_model_complexity(pd_config, 'pd_signal')
    print(f"Total Parameters: {pd_complexity['total_parameters']:,}")
    
    # Test CWT model
    print("\\nCWT Image Classifier:")
    cwt_complexity = calculate_model_complexity(cwt_config, 'cwt_image')
    print(f"Total Parameters: {cwt_complexity['total_parameters']:,}")
    
    return pd_complexity, cwt_complexity


if __name__ == "__main__":
    test_complexity_calculator()