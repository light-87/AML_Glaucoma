"""
Configuration Validator

Validates and documents pipeline configuration.
"""

import os
from pathlib import Path
import yaml
from omegaconf import OmegaConf, DictConfig
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
import re

from glaucoma_detection.logger import get_logger, ERROR_CODES, log_exception_handler

logger = get_logger(__name__)

class ConfigurationError(Exception):
    """Exception for configuration validation errors."""
    pass

# Schema for configuration validation
CONFIG_SCHEMA = {
    "paths": {
        "base_dir": {"type": "str", "required": True, "description": "Base directory for the project"}
    },
    "data": {
        "zip_file": {"type": "str", "required": False, "description": "Path to ZIP file if extraction is needed"},
        "random_state": {"type": "int", "required": False, "default": 42, "description": "Random seed for reproducibility"},
        "split_config": {
            "train_ratio": {"type": "float", "required": False, "default": 0.7, "description": "Ratio of training data"},
            "val_ratio": {"type": "float", "required": False, "default": 0.15, "description": "Ratio of validation data"},
            "test_ratio": {"type": "float", "required": False, "default": 0.15, "description": "Ratio of test data"}
        }
    },
    "model": {
        "architecture": {"type": "str", "required": True, "options": ["unet", "unet++", "deeplabv3", "fpn"], 
                        "description": "Model architecture to use"},
        "encoder": {"type": "str", "required": True, "description": "Backbone encoder for the model"},
        "pretrained": {"type": "bool", "required": False, "default": True, "description": "Whether to use pretrained weights"},
        "in_channels": {"type": "int", "required": False, "default": 3, "description": "Number of input channels"},
        "num_classes": {"type": "int", "required": False, "default": 1, "description": "Number of output classes"}
    },
    "preprocessing": {
        "image_size": {"type": "list", "required": True, "description": "Target image size (width, height)"},
        "image_channels": {"type": "int", "required": False, "default": 3, "description": "Number of image channels"},
        "normalization": {"type": "str", "required": False, "default": "imagenet", 
                         "options": ["imagenet", "instance", "pixel", "none"], 
                         "description": "Normalization method"},
        "mode": {"type": "str", "required": False, "default": "segmentation", 
                "options": ["segmentation", "classification"], 
                "description": "Mode of operation"},
        "augmentation": {
            "enabled": {"type": "bool", "required": False, "default": True, "description": "Whether to use data augmentation"},
            "rotation_range": {"type": "float", "required": False, "default": 15, "description": "Rotation range for augmentation"},
            "width_shift_range": {"type": "float", "required": False, "default": 0.1, "description": "Width shift range for augmentation"},
            "height_shift_range": {"type": "float", "required": False, "default": 0.1, "description": "Height shift range for augmentation"},
            "shear_range": {"type": "float", "required": False, "default": 0.1, "description": "Shear range for augmentation"},
            "zoom_range": {"type": "float", "required": False, "default": 0.1, "description": "Zoom range for augmentation"},
            "horizontal_flip": {"type": "bool", "required": False, "default": True, "description": "Whether to use horizontal flip"},
            "vertical_flip": {"type": "bool", "required": False, "default": False, "description": "Whether to use vertical flip"}
        }
    },
    "training": {
        "epochs": {"type": "int", "required": True, "description": "Number of training epochs"},
        "batch_size": {"type": "int", "required": True, "description": "Batch size for training"},
        "num_workers": {"type": "int", "required": False, "default": 4, "description": "Number of workers for data loading"},
        "learning_rate": {"type": "float", "required": False, "default": 0.001, "description": "Learning rate"},
        "optimizer": {"type": "str", "required": False, "default": "adam", 
                     "options": ["adam", "sgd", "adamw"], 
                     "description": "Optimizer to use"},
        "loss_function": {"type": "str", "required": False, "default": "combined", 
                         "options": ["combined", "dice", "bce", "focal", "jaccard"], 
                         "description": "Loss function to use"},
        "precision": {"type": "str", "required": False, "default": "32-true", 
                     "options": ["16-mixed", "32-true"], 
                     "description": "Precision for training"},
        "use_gpu": {"type": "bool", "required": False, "default": True, "description": "Whether to use GPU for training"},
        "gpu_ids": {"type": "list", "required": False, "default": [0], "description": "List of GPU IDs to use"},
        "gradient_clip_val": {"type": "float", "required": False, "default": 0.0, "description": "Gradient clipping value"},
        "accumulate_grad_batches": {"type": "int", "required": False, "default": 1, "description": "Number of batches to accumulate gradients"},
        "lr_scheduler": {
            "enabled": {"type": "bool", "required": False, "default": True, "description": "Whether to use learning rate scheduler"},
            "factor": {"type": "float", "required": False, "default": 0.1, "description": "Factor by which to reduce learning rate"},
            "patience": {"type": "int", "required": False, "default": 5, "description": "Patience for learning rate scheduler"},
            "min_lr": {"type": "float", "required": False, "default": 0.000001, "description": "Minimum learning rate"},
            "monitor": {"type": "str", "required": False, "default": "val_loss", "description": "Metric to monitor for scheduler"}
        },
        "early_stopping": {
            "enabled": {"type": "bool", "required": False, "default": True, "description": "Whether to use early stopping"},
            "patience": {"type": "int", "required": False, "default": 10, "description": "Patience for early stopping"},
            "monitor": {"type": "str", "required": False, "default": "val_loss", "description": "Metric to monitor for early stopping"},
            "min_delta": {"type": "float", "required": False, "default": 0.001, "description": "Minimum change to qualify as improvement"},
            "mode": {"type": "str", "required": False, "default": "min", 
                    "options": ["min", "max"], 
                    "description": "Mode for early stopping"}
        },
        "checkpointing": {
            "enabled": {"type": "bool", "required": False, "default": True, "description": "Whether to use checkpointing"},
            "save_top_k": {"type": "int", "required": False, "default": 3, "description": "Number of best models to save"},
            "monitor": {"type": "str", "required": False, "default": "val_loss", "description": "Metric to monitor for checkpointing"},
            "mode": {"type": "str", "required": False, "default": "min", 
                    "options": ["min", "max"], 
                    "description": "Mode for checkpointing"}
        },
        "use_class_weights": {"type": "bool", "required": False, "default": True, "description": "Whether to use class weights for imbalanced data"}
    },
    "evaluation": {
        "metrics": {"type": "list", "required": False, 
                   "default": ["dice", "iou", "accuracy", "precision", "recall", "f1"],
                   "description": "Metrics to calculate during evaluation"},
        "threshold": {"type": "float", "required": False, "default": 0.5, "description": "Threshold for binary segmentation"},
        "num_samples": {"type": "int", "required": False, "default": 5, "description": "Number of sample visualizations"},
        "visualization": {
            "enabled": {"type": "bool", "required": False, "default": True, "description": "Whether to generate visualizations"},
            "plot_wrong_predictions": {"type": "bool", "required": False, "default": True, "description": "Whether to plot wrong predictions"},
            "plot_gradcam": {"type": "bool", "required": False, "default": True, "description": "Whether to plot GradCAM visualizations"}
        }
    },
    "logging": {
        "use_wandb": {"type": "bool", "required": False, "default": False, "description": "Whether to use Weights & Biases for logging"},
        "wandb_project": {"type": "str", "required": False, "default": "glaucoma-detection", "description": "Weights & Biases project name"},
        "log_every_n_steps": {"type": "int", "required": False, "default": 10, "description": "How often to log metrics"}
    },
    "pipeline": {
        "steps": {"type": "list", "required": False, 
                 "default": ["extract", "load", "clean", "preprocess", "train", "evaluate"],
                 "description": "Pipeline steps to execute"},
        "force": {"type": "bool", "required": False, "default": False, "description": "Whether to force rerun of steps"},
        "description": {"type": "str", "required": False, "default": "Default pipeline run", "description": "Description of the pipeline run"}
    }
}

class ConfigValidator:
    """Validates and documents configuration files."""
    
    def __init__(self, schema: Dict[str, Any] = CONFIG_SCHEMA):
        """Initialize with schema.
        
        Args:
            schema: Schema for validation
        """
        self.schema = schema
    
    def validate_config(self, config: Union[Dict[str, Any], DictConfig]) -> Tuple[bool, List[str]]:
        """Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        # Convert to dict if DictConfig
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        
        is_valid = True
        error_messages = []
        
        # Validate against schema
        for section_name, section_schema in self.schema.items():
            # Check if section exists
            if section_name not in config:
                if any(param.get('required', False) for param_name, param in section_schema.items()):
                    is_valid = False
                    error_messages.append(f"Required section '{section_name}' is missing")
                continue
            
            # Get section from config
            section = config[section_name]
            
            # Validate each parameter in the section
            for param_name, param_schema in section_schema.items():
                # Skip nested parameters (handled recursively)
                if isinstance(param_schema, dict) and 'type' not in param_schema:
                    # Recursively validate nested section
                    if param_name in section:
                        nested_is_valid, nested_errors = self._validate_nested_section(
                            section[param_name], param_schema, f"{section_name}.{param_name}"
                        )
                        is_valid = is_valid and nested_is_valid
                        error_messages.extend(nested_errors)
                    elif any(sub_param.get('required', False) for sub_param_name, sub_param in param_schema.items()):
                        is_valid = False
                        error_messages.append(f"Required subsection '{section_name}.{param_name}' is missing")
                    continue
                
                # Check if required parameter exists
                if param_schema.get('required', False) and (param_name not in section or section[param_name] is None):
                    is_valid = False
                    error_messages.append(f"Required parameter '{section_name}.{param_name}' is missing")
                    continue
                
                # Skip if parameter doesn't exist and isn't required
                if param_name not in section:
                    continue
                
                # Get parameter value
                value = section[param_name]
                
                # Skip None values for optional parameters
                if value is None and not param_schema.get('required', False):
                    continue
                
                # Validate parameter type
                type_name = param_schema.get('type')
                if type_name == 'str' and not isinstance(value, str):
                    is_valid = False
                    error_messages.append(
                        f"Parameter '{section_name}.{param_name}' should be a string, got {type(value).__name__}"
                    )
                elif type_name == 'int' and not isinstance(value, int):
                    is_valid = False
                    error_messages.append(
                        f"Parameter '{section_name}.{param_name}' should be an integer, got {type(value).__name__}"
                    )
                elif type_name == 'float' and not isinstance(value, (int, float)):
                    is_valid = False
                    error_messages.append(
                        f"Parameter '{section_name}.{param_name}' should be a float, got {type(value).__name__}"
                    )
                elif type_name == 'bool' and not isinstance(value, bool):
                    is_valid = False
                    error_messages.append(
                        f"Parameter '{section_name}.{param_name}' should be a boolean, got {type(value).__name__}"
                    )
                elif type_name == 'list' and not isinstance(value, list):
                    is_valid = False
                    error_messages.append(
                        f"Parameter '{section_name}.{param_name}' should be a list, got {type(value).__name__}"
                    )
                
                # Validate parameter options
                if 'options' in param_schema and value not in param_schema['options']:
                    is_valid = False
                    error_messages.append(
                        f"Parameter '{section_name}.{param_name}' should be one of {param_schema['options']}, got {value}"
                    )
        
        return is_valid, error_messages
    
    def _validate_nested_section(self, section: Dict[str, Any], schema: Dict[str, Any], path: str) -> Tuple[bool, List[str]]:
        """Validate a nested configuration section.
        
        Args:
            section: Section to validate
            schema: Schema for the section
            path: Path to the section
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        is_valid = True
        error_messages = []
        
        # Validate each parameter in the section
        for param_name, param_schema in schema.items():
            # Skip nested parameters (handled recursively)
            if isinstance(param_schema, dict) and 'type' not in param_schema:
                # Recursively validate nested section
                if param_name in section:
                    nested_is_valid, nested_errors = self._validate_nested_section(
                        section[param_name], param_schema, f"{path}.{param_name}"
                    )
                    is_valid = is_valid and nested_is_valid
                    error_messages.extend(nested_errors)
                elif any(sub_param.get('required', False) for sub_param_name, sub_param in param_schema.items()):
                    is_valid = False
                    error_messages.append(f"Required subsection '{path}.{param_name}' is missing")
                continue
            
            # Check if required parameter exists
            if param_schema.get('required', False) and (param_name not in section or section[param_name] is None):
                is_valid = False
                error_messages.append(f"Required parameter '{path}.{param_name}' is missing")
                continue
            
            # Skip if parameter doesn't exist and isn't required
            if param_name not in section:
                continue
            
            # Get parameter value
            value = section[param_name]
            
            # Skip None values for optional parameters
            if value is None and not param_schema.get('required', False):
                continue
            
            # Validate parameter type
            type_name = param_schema.get('type')
            if type_name == 'str' and not isinstance(value, str):
                is_valid = False
                error_messages.append(
                    f"Parameter '{path}.{param_name}' should be a string, got {type(value).__name__}"
                )
            elif type_name == 'int' and not isinstance(value, int):
                is_valid = False
                error_messages.append(
                    f"Parameter '{path}.{param_name}' should be an integer, got {type(value).__name__}"
                )
            elif type_name == 'float' and not isinstance(value, (int, float)):
                is_valid = False
                error_messages.append(
                    f"Parameter '{path}.{param_name}' should be a float, got {type(value).__name__}"
                )
            elif type_name == 'bool' and not isinstance(value, bool):
                is_valid = False
                error_messages.append(
                    f"Parameter '{path}.{param_name}' should be a boolean, got {type(value).__name__}"
                )
            elif type_name == 'list' and not isinstance(value, list):
                is_valid = False
                error_messages.append(
                    f"Parameter '{path}.{param_name}' should be a list, got {type(value).__name__}"
                )
            
            # Validate parameter options
            if 'options' in param_schema and value not in param_schema['options']:
                is_valid = False
                error_messages.append(
                    f"Parameter '{path}.{param_name}' should be one of {param_schema['options']}, got {value}"
                )
        
        return is_valid, error_messages
    
    def fill_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing parameters with default values.
        
        Args:
            config: Configuration to fill
            
        Returns:
            Configuration with defaults filled
        """
        result = {}
        
        # Fill defaults for each section
        for section_name, section_schema in self.schema.items():
            # Create section if it doesn't exist
            if section_name not in config:
                result[section_name] = {}
            else:
                result[section_name] = config[section_name].copy() if isinstance(config[section_name], dict) else config[section_name]
            
            # Skip if not a dict
            if not isinstance(result[section_name], dict):
                continue
            
            # Fill defaults for each parameter
            for param_name, param_schema in section_schema.items():
                # Handle nested sections
                if isinstance(param_schema, dict) and 'type' not in param_schema:
                    # Create nested section if it doesn't exist
                    if param_name not in result[section_name]:
                        result[section_name][param_name] = {}
                    
                    # Recursively fill defaults for nested section
                    nested_config = result[section_name][param_name]
                    if isinstance(nested_config, dict):
                        result[section_name][param_name] = self._fill_nested_defaults(nested_config, param_schema)
                    continue
                
                # Fill default if parameter doesn't exist and has a default
                if param_name not in result[section_name] and 'default' in param_schema:
                    result[section_name][param_name] = param_schema['default']
        
        return result
    
    def _fill_nested_defaults(self, config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Fill defaults for nested section.
        
        Args:
            config: Configuration to fill
            schema: Schema for the section
            
        Returns:
            Configuration with defaults filled
        """
        result = config.copy()
        
        # Fill defaults for each parameter
        for param_name, param_schema in schema.items():
            # Handle nested sections
            if isinstance(param_schema, dict) and 'type' not in param_schema:
                # Create nested section if it doesn't exist
                if param_name not in result:
                    result[param_name] = {}
                
                # Recursively fill defaults for nested section
                nested_config = result[param_name]
                if isinstance(nested_config, dict):
                    result[param_name] = self._fill_nested_defaults(nested_config, param_schema)
                continue
            
            # Fill default if parameter doesn't exist and has a default
            if param_name not in result and 'default' in param_schema:
                result[param_name] = param_schema['default']
        
        return result
    
    def generate_schema_documentation(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """Generate Markdown documentation from schema.
        
        Args:
            output_path: Path to save the documentation
            
        Returns:
            Documentation as Markdown string
        """
        docs = "# Configuration Schema Documentation\n\n"
        
        # Generate documentation for each section
        for section_name, section_schema in self.schema.items():
            docs += f"## {section_name}\n\n"
            
            # Generate table for parameters
            docs += "| Parameter | Type | Required | Default | Description |\n"
            docs += "|-----------|------|----------|---------|-------------|\n"
            
            # Add parameters to table
            for param_name, param_schema in section_schema.items():
                # Handle nested sections
                if isinstance(param_schema, dict) and 'type' not in param_schema:
                    docs += f"| **{param_name}** | object | - | - | Nested configuration section |\n"
                    
                    # Add nested section with indentation
                    docs += "\n### " + param_name + "\n\n"
                    docs += "| Parameter | Type | Required | Default | Description |\n"
                    docs += "|-----------|------|----------|---------|-------------|\n"
                    
                    # Add nested parameters to table
                    # Add nested parameters to table
                    for sub_param_name, sub_param_schema in param_schema.items():
                        docs += self._add_param_to_table(sub_param_name, sub_param_schema)
                    
                    docs += "\n"
                    continue
                
                # Add parameter to table
                # Add parameter to table
                docs += self._add_param_to_table(param_name, param_schema)
            
            docs += "\n"
        
        # Save documentation if output path is provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            with open(output_path, 'w') as f:
                f.write(docs)
            
            logger.info(f"Schema documentation saved to {output_path}")
        
        return docs
    
    def _add_param_to_table(self, param_name: str, param_schema: Dict[str, Any], docs: str = "") -> str:
        """Add parameter to documentation table.
        
        Args:
            param_name: Parameter name
            param_schema: Parameter schema
            docs: Existing documentation string (optional)
            
        Returns:
            Table row as string
        """
        type_name = param_schema.get('type', '')
        required = 'Yes' if param_schema.get('required', False) else 'No'
        default = str(param_schema.get('default', '-'))
        description = param_schema.get('description', '')
        
        # Add options to description if available
        if 'options' in param_schema:
            description += f" (Options: {param_schema['options']})"
        
        row = f"| {param_name} | {type_name} | {required} | {default} | {description} |\n"
        
        # If docs is provided, append the row and return the updated docs
        if docs:
            return docs + row
        
        # Otherwise just return the row
        return row
    
    def export_schema_as_json(self, output_path: Union[str, Path]) -> None:
        """Export schema as JSON file.
        
        Args:
            output_path: Path to save the schema
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.schema, f, indent=2)
        
        logger.info(f"Schema exported to {output_path}")

@log_exception_handler
def validate_config(config: Union[Dict[str, Any], DictConfig]) -> Dict[str, Any]:
    """Validate configuration and fill defaults.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Validated configuration with defaults filled
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    validator = ConfigValidator()
    
    # Convert to dict if DictConfig
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config
    
    # Validate configuration
    is_valid, error_messages = validator.validate_config(config_dict)
    
    if not is_valid:
        error_msg = "Configuration validation failed:\n" + "\n".join(error_messages)
        logger.error(error_msg)
        raise ConfigurationError(error_msg)
    
    # Fill defaults
    config_with_defaults = validator.fill_defaults(config_dict)
    
    logger.info("Configuration validation successful")
    return config_with_defaults