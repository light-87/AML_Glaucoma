"""
Model Architecture Module

Using Segmentation Models PyTorch (SMP) for model architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Dict, Any, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

def create_model(model_config: Dict[str, Any]) -> nn.Module:
    """Create a segmentation model using SMP library."""
    
    architecture = model_config.get('architecture', 'unet')
    encoder_name = model_config.get('encoder', 'resnet34')
    encoder_weights = 'imagenet' if model_config.get('pretrained', True) else None
    in_channels = model_config.get('in_channels', 3)
    classes = model_config.get('num_classes', 1)
    
    logger.info(f"Creating {architecture} model with {encoder_name} encoder")
    
    # Create model based on architecture
    if architecture.lower() == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation='sigmoid' if classes == 1 else 'softmax'
        )
    elif architecture.lower() == 'unet++':
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation='sigmoid' if classes == 1 else 'softmax'
        )
    elif architecture.lower() == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation='sigmoid' if classes == 1 else 'softmax'
        )
    elif architecture.lower() == 'fpn':
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation='sigmoid' if classes == 1 else 'softmax'
        )
    else:
        logger.warning(f"Unknown architecture: {architecture}. Defaulting to UNet.")
        model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=classes,
            activation='sigmoid' if classes == 1 else 'softmax'
        )
    
    return model

def get_loss_function(loss_type: str = 'combined'):
    """Get the appropriate loss function."""
    import segmentation_models_pytorch.losses as smp_losses
    
    if loss_type.lower() == 'dice':
        return smp_losses.DiceLoss(mode='binary')
    elif loss_type.lower() == 'jaccard':
        return smp_losses.JaccardLoss(mode='binary')
    elif loss_type.lower() == 'focal':
        return smp_losses.FocalLoss(mode='binary')
    elif loss_type.lower() == 'lovasz':
        return smp_losses.LovaszLoss(mode='binary')
    elif loss_type.lower() == 'tversky':
        return smp_losses.TverskyLoss(mode='binary')
    elif loss_type.lower() == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type.lower() == 'combined':
        return smp_losses.DiceLoss(mode='binary') + nn.BCEWithLogitsLoss()
    else:
        logger.warning(f"Unknown loss function: {loss_type}. Using combined loss.")
        return smp_losses.DiceLoss(mode='binary') + nn.BCEWithLogitsLoss()

def save_model(model: nn.Module, filepath: str, **kwargs) -> Optional[str]:
    """Save model checkpoint."""
    try:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            **kwargs
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return None

def load_model(
    filepath: str, 
    model_config: Optional[Dict[str, Any]] = None, 
    device: str = 'cuda'
) -> Tuple[Optional[nn.Module], Optional[Dict[str, Any]]]:
    """Load model from checkpoint."""
    try:
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model if config is provided
        if model_config is not None:
            model = create_model(model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            return model, checkpoint
        else:
            logger.warning("No model_config provided. Returning checkpoint only.")
            return None, checkpoint
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None