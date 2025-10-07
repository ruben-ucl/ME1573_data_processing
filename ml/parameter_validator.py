#!/usr/bin/env python3
"""
Centralized parameter validation for ML configurations.
Validates configurations against the hyperparameter registry.
"""

import os
from typing import List, Dict, Any, Tuple
from hyperparameter_registry import HYPERPARAMETER_REGISTRY, get_parameter_info

# Set UTF-8 encoding for Windows compatibility
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')


class ParameterValidator:
    """Centralized parameter validation using the hyperparameter registry."""
    
    @staticmethod
    def validate_config(config: Dict[str, Any], classifier_type: str) -> List[str]:
        """Validate complete configuration against registry.
        
        Args:
            config (dict): Configuration to validate
            classifier_type (str): 'pd_signal' or 'cwt_image'
            
        Returns:
            list: List of validation error messages (empty if valid)
        """
        errors = []
        
        for param_name, value in config.items():
            if param_name in HYPERPARAMETER_REGISTRY:
                param_errors = ParameterValidator.validate_parameter(
                    param_name, value, classifier_type
                )
                errors.extend(param_errors)
        
        return errors
    
    @staticmethod
    def validate_parameter(param_name: str, value: Any, classifier_type: str) -> List[str]:
        """Validate a single parameter value.
        
        Args:
            param_name (str): Parameter name
            value: Parameter value to validate
            classifier_type (str): 'pd_signal' or 'cwt_image'
            
        Returns:
            list: List of validation error messages
        """
        errors = []
        
        try:
            param_info = get_parameter_info(param_name, classifier_type)
            if param_info is None:
                errors.append(f"Parameter '{param_name}' not applicable to classifier '{classifier_type}'")
                return errors
        except ValueError as e:
            errors.append(str(e))
            return errors
        
        param_type = param_info.get('type')
        search_space = param_info.get('search_space', [])
        
        # Type-specific validation
        if param_type == 'continuous':
            errors.extend(ParameterValidator._validate_continuous(param_name, value, param_info))
        elif param_type == 'discrete':
            errors.extend(ParameterValidator._validate_discrete(param_name, value, param_info))
        elif param_type == 'categorical':
            errors.extend(ParameterValidator._validate_categorical(param_name, value, search_space))
        
        return errors
    
    @staticmethod
    def _validate_continuous(param_name: str, value: Any, param_info: Dict) -> List[str]:
        """Validate continuous parameter."""
        errors = []
        
        try:
            float_value = float(value)
        except (ValueError, TypeError):
            errors.append(f"Parameter '{param_name}' must be numeric, got {type(value).__name__}")
            return errors
        
        # Check if value is in search space
        search_space = param_info.get('search_space', [])
        if search_space and float_value not in search_space:
            errors.append(f"Parameter '{param_name}' value {float_value} not in search space {search_space}")
        
        # Check DoE range if available
        doe_range = param_info.get('doe_range')
        if doe_range:
            min_val, max_val = doe_range
            if not (min_val <= float_value <= max_val):
                errors.append(f"Parameter '{param_name}' value {float_value} outside valid range [{min_val}, {max_val}]")
        
        return errors
    
    @staticmethod
    def _validate_discrete(param_name: str, value: Any, param_info: Dict) -> List[str]:
        """Validate discrete parameter."""
        errors = []
        
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            errors.append(f"Parameter '{param_name}' must be integer, got {type(value).__name__}")
            return errors
        
        # Check if value is in search space
        search_space = param_info.get('search_space', [])
        if search_space and int_value not in search_space:
            errors.append(f"Parameter '{param_name}' value {int_value} not in search space {search_space}")
        
        # Check DoE range if available
        doe_range = param_info.get('doe_range')
        if doe_range:
            min_val, max_val = doe_range
            if not (min_val <= int_value <= max_val):
                errors.append(f"Parameter '{param_name}' value {int_value} outside valid range [{min_val}, {max_val}]")
        
        return errors
    
    @staticmethod
    def _validate_categorical(param_name: str, value: Any, search_space: List) -> List[str]:
        """Validate categorical parameter."""
        errors = []
        
        if not search_space:
            return errors
        
        # Handle list parameters (like conv_filters, dense_units)
        if isinstance(value, list):
            if value not in search_space:
                errors.append(f"Parameter '{param_name}' list {value} not in search space {search_space}")
        else:
            if value not in search_space:
                errors.append(f"Parameter '{param_name}' value '{value}' not in search space {search_space}")
        
        return errors
    
    @staticmethod
    def validate_and_fix_config(config: Dict[str, Any], classifier_type: str) -> Tuple[Dict[str, Any], List[str]]:
        """Validate config and attempt to fix common issues.
        
        Args:
            config (dict): Configuration to validate and fix
            classifier_type (str): 'pd_signal' or 'cwt_image'
            
        Returns:
            tuple: (fixed_config, warnings) where warnings are non-critical issues that were fixed
        """
        fixed_config = config.copy()
        warnings = []
        
        for param_name, value in config.items():
            if param_name not in HYPERPARAMETER_REGISTRY:
                continue
                
            param_info = get_parameter_info(param_name, classifier_type)
            if param_info is None:
                continue
            
            param_type = param_info.get('type')
            search_space = param_info.get('search_space', [])
            
            # Fix common type issues
            if param_type == 'discrete' and isinstance(value, float):
                fixed_config[param_name] = int(value)
                warnings.append(f"Converted {param_name} from float to int: {value} -> {int(value)}")
            
            elif param_type == 'continuous' and isinstance(value, int):
                fixed_config[param_name] = float(value)
                warnings.append(f"Converted {param_name} from int to float: {value} -> {float(value)}")
            
            # Fix out-of-range values by clamping to search space
            elif param_type in ['continuous', 'discrete'] and search_space:
                if isinstance(search_space, list) and value not in search_space:
                    # Find closest value in search space
                    if param_type == 'continuous':
                        closest = min(search_space, key=lambda x: abs(x - float(value)))
                    else:
                        closest = min(search_space, key=lambda x: abs(x - int(value)))
                    
                    fixed_config[param_name] = closest
                    warnings.append(f"Adjusted {param_name} to closest valid value: {value} -> {closest}")
        
        return fixed_config, warnings
    
    @staticmethod
    def get_parameter_suggestions(classifier_type: str, category: str = None) -> Dict[str, Any]:
        """Get suggested parameter values for a classifier.
        
        Args:
            classifier_type (str): 'pd_signal' or 'cwt_image'
            category (str, optional): Parameter category to filter by
            
        Returns:
            dict: Suggested parameter values
        """
        suggestions = {}
        
        for param_name, param_info in HYPERPARAMETER_REGISTRY.items():
            # Filter by category if specified
            if category and param_info.get('category') != category:
                continue
            
            resolved_info = get_parameter_info(param_name, classifier_type)
            if resolved_info is None:
                continue
            
            # Get middle value from search space as suggestion
            search_space = resolved_info.get('search_space', [])
            if search_space:
                if isinstance(search_space[0], list):
                    # For list parameters, suggest the default or middle option
                    suggestions[param_name] = resolved_info.get('default', search_space[len(search_space)//2])
                else:
                    # For scalar parameters, suggest middle value
                    suggestions[param_name] = search_space[len(search_space)//2]
            else:
                # Fall back to default
                suggestions[param_name] = resolved_info.get('default')
        
        return suggestions


def validate_config_file(config_path: str, classifier_type: str) -> Tuple[bool, List[str]]:
    """Validate a configuration file.
    
    Args:
        config_path (str): Path to JSON configuration file
        classifier_type (str): 'pd_signal' or 'cwt_image'
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    import json
    from pathlib import Path
    
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            return False, [f"Configuration file not found: {config_path}"]
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        errors = ParameterValidator.validate_config(config, classifier_type)
        return len(errors) == 0, errors
        
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON in config file: {e}"]
    except Exception as e:
        return False, [f"Error reading config file: {e}"]