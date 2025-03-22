"""
Configuration Module

This module contains all configurable parameters for the glaucoma detection pipeline.
Centralizing configuration makes it easier to experiment with different settings.

Usage:
    from config import DATA_CONFIG, PREPROCESSING_CONFIG, MODEL_CONFIG, TRAINING_CONFIG

Configuration categories:
- DATA_CONFIG: Data loading, paths, and dataset-specific settings
- PREPROCESSING_CONFIG: Data preprocessing parameters
- MODEL_CONFIG: Model architecture parameters
- TRAINING_CONFIG: Training hyperparameters
- EVALUATION_CONFIG: Evaluation metrics and settings
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(os.getenv('BASE_DIR', os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.getenv('DATA_DIR', os.path.join(BASE_DIR, 'data'))
OUTPUT_DIR = os.getenv('OUTPUT_DIR', os.path.join(BASE_DIR, 'output'))
MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(BASE_DIR, 'models'))
LOG_DIR = os.getenv('LOG_DIR', os.path.join(BASE_DIR, 'logs'))

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data configuration
DATA_CONFIG = {
    # General data settings
    'base_path': DATA_DIR,
    'output_dir': OUTPUT_DIR,
    'consolidated_csv': os.path.join(OUTPUT_DIR, 'consolidated_glaucoma_dataset.csv'),
    'cleaned_csv': os.path.join(OUTPUT_DIR, 'cleaned_glaucoma_dataset.csv'),
    'preprocessed_csv': os.path.join(OUTPUT_DIR, 'preprocessed_glaucoma_dataset.csv'),
    
    # Dataset specific settings
    'datasets': {
        'ORIGA': {
            'path': os.path.join(DATA_DIR, 'ORIGA'),
            'metadata_file': 'OrigaList.csv',
            'info_file': 'origa_info.csv',
            'image_dir': 'Images',
            'mask_dir': 'Masks',
        },
        'REFUGE': {
            'path': os.path.join(DATA_DIR, 'REFUGE'),
            'splits': ['train', 'val', 'test'],
            'metadata_file': 'index.json',
            'image_dir': 'Images',
            'mask_dir': 'Masks',
        },
        'G1020': {
            'path': os.path.join(DATA_DIR, 'G1020'),
            'metadata_file': 'G1020.csv',
            'image_dir': 'Images',
            'mask_dir': 'Masks',
            'nerve_removed_dir': 'NerveRemoved_Images',
        }
    },
    
    # Split configuration
    'split_config': {
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'random_state': 42
    }
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    # Image preprocessing
    'image_size': (224, 224),  # Width, Height
    'image_channels': 3,  # RGB
    'normalization': 'imagenet',  # 'imagenet', 'instance', 'pixel', 'none'
    'use_square_images': True,  # Use square cropped images if available
    'use_cropped_images': True,  # Use cropped optic disc images if available
    
    # Data augmentation
    'augmentation': {
        'enabled': True,
        'rotation_range': 15,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'shear_range': 0.1,
        'zoom_range': 0.1,
        'horizontal_flip': True,
        'vertical_flip': False,
        'fill_mode': 'nearest'
    },
    
    # Batch processing
    'batch_size': 32,
    'shuffle': True,
    'num_workers': 4,
    
    # Class balancing
    'class_balancing': {
        'enabled': True,
        'method': 'oversampling',  # 'oversampling', 'undersampling', 'weighted'
        'target_ratio': 1.0  # Ratio of minority to majority class
    }
}

# Model configuration
MODEL_CONFIG = {
    # Model architecture
    'architecture': 'resnet50',  # 'resnet50', 'vgg16', 'efficientnet', 'custom'
    'pretrained': True,  # Use pretrained weights
    'freeze_backbone': False,  # Freeze backbone layers
    'input_shape': (224, 224, 3),  # Height, Width, Channels
    
    # Custom model settings
    'custom_model': {
        'conv_layers': [64, 128, 256, 512],  # Convolutional layer filters
        'dense_layers': [1024, 512, 256],  # Dense layer units
        'dropout_rate': 0.5,  # Dropout rate for dense layers
        'activation': 'relu',  # Activation function
        'final_activation': 'sigmoid'  # Final activation function
    },
    
    # Model outputs
    'num_classes': 1,  # Binary classification
    'output_activation': 'sigmoid',  # 'sigmoid' for binary, 'softmax' for multi-class
    
    # Model checkpoint
    'checkpoint_dir': MODEL_DIR,
    'model_name': 'glaucoma_detection_model'
}

# Training configuration
TRAINING_CONFIG = {
    # Basic training parameters
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'adam',  # 'adam', 'sgd', 'rmsprop'
    'loss_function': 'binary_crossentropy',  # 'binary_crossentropy', 'categorical_crossentropy'
    
    # Learning rate scheduling
    'lr_scheduler': {
        'enabled': True,
        'factor': 0.1,
        'patience': 5,
        'min_lr': 1e-6,
        'monitor': 'val_loss'
    },
    
    # Early stopping
    'early_stopping': {
        'enabled': True,
        'patience': 10,
        'monitor': 'val_loss',
        'min_delta': 0.001
    },
    
    # Checkpointing
    'checkpointing': {
        'enabled': True,
        'save_best_only': True,
        'monitor': 'val_loss',
        'mode': 'min'
    },
    
    # Class weights for imbalanced data
    'use_class_weights': True,
    
    # Metrics to monitor
    'metrics': ['accuracy', 'precision', 'recall', 'auc'],
    
    # GPU configuration
    'use_gpu': True,
    'multi_gpu': False,
    'gpu_ids': [0]
}

# Evaluation configuration
EVALUATION_CONFIG = {
    # Evaluation metrics
    'metrics': [
        'accuracy', 'precision', 'recall', 'f1', 
        'specificity', 'sensitivity', 'auc'
    ],
    
    # ROC curve
    'plot_roc': True,
    
    # Confusion matrix
    'plot_confusion_matrix': True,
    
    # Threshold optimization
    'optimize_threshold': True,
    'threshold_metric': 'f1',  # Metric to optimize when finding best threshold
    
    # Prediction output
    'prediction_output': os.path.join(OUTPUT_DIR, 'predictions.csv'),
    
    # Visualizations
    'visualization': {
        'enabled': True,
        'output_dir': os.path.join(OUTPUT_DIR, 'visualizations'),
        'plot_wrong_predictions': True,
        'max_samples': 50,  # Maximum number of samples to visualize
        'plot_gradcam': True  # Plot Grad-CAM heatmaps
    }
}

# Combine all configurations
CONFIG = {
    'DATA': DATA_CONFIG,
    'PREPROCESSING': PREPROCESSING_CONFIG,
    'MODEL': MODEL_CONFIG,
    'TRAINING': TRAINING_CONFIG,
    'EVALUATION': EVALUATION_CONFIG
}

def get_config():
    """
    Get the complete configuration dictionary.
    
    Returns:
    --------
    dict
        Complete configuration dictionary
    """
    return CONFIG

def update_config(config_updates):
    """
    Update the configuration with new values.
    
    Parameters:
    -----------
    config_updates : dict
        Dictionary with configuration updates
        
    Returns:
    --------
    dict
        Updated configuration
    """
    # Deep update of nested dictionaries
    def deep_update(original, updates):
        for key, value in updates.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                deep_update(original[key], value)
            else:
                original[key] = value
    
    # Update each configuration category
    for category, category_updates in config_updates.items():
        if category in CONFIG and isinstance(category_updates, dict):
            deep_update(CONFIG[category], category_updates)
    
    return CONFIG

def save_config(output_path=None):
    """
    Save the current configuration to a JSON file.
    
    Parameters:
    -----------
    output_path : str, optional
        Path to save the configuration, by default None
        
    Returns:
    --------
    str
        Path to the saved configuration file
    """
    import json
    
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'config.json')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save configuration
    with open(output_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    return output_path

def load_config(config_path):
    """
    Load configuration from a JSON file.
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration file
        
    Returns:
    --------
    dict
        Loaded configuration
    """
    import json
    
    # Load configuration
    with open(config_path, 'r') as f:
        config_updates = json.load(f)
    
    # Update configuration
    return update_config(config_updates)

if __name__ == "__main__":
    # Example usage
    print("Current configuration:")
    print(f"Data directory: {DATA_CONFIG['base_path']}")
    print(f"Output directory: {DATA_CONFIG['output_dir']}")
    print(f"Model architecture: {MODEL_CONFIG['architecture']}")
    print(f"Training epochs: {TRAINING_CONFIG['epochs']}")
    
    # Save configuration
    config_path = save_config()
    print(f"Configuration saved to {config_path}")