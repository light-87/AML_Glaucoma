"""
Model Architecture Module

This module defines the neural network architectures for the glaucoma detection pipeline.
It implements different model architectures including U-Net for segmentation using PyTorch.

Classes:
- UNet: U-Net model architecture
- DoubleConv: Double convolution block
- Up: Upsampling block
- Down: Downsampling block
- OutConv: Output convolution

Functions:
- dice_coefficient: Dice coefficient metric
- dice_loss: Dice loss function
- combined_loss: Combined BCE and Dice loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import logging

# Import config
from config import MODEL_CONFIG
from utils import setup_logger

# Set up logging
logger = setup_logger('model')

class DoubleConv(nn.Module):
    """
    Double convolution block: (Conv2D -> BatchNorm -> ReLU) × 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
            
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downscaling block with maxpool and double convolution
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upscaling block with upsampling, concatenation, and double convolution
    """
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Output convolution block
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    U-Net model architecture for image segmentation
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, base_filters=64, dropout_rate=0.1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        
        # Encoder (Contracting path)
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor, dropout_rate=dropout_rate)
        
        # Decoder (Expansive path)
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)
        
        # Output layer
        self.outc = OutConv(base_filters, n_classes)
        
        # Activation for final layer
        self.activation = nn.Sigmoid() if n_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        
        # Apply activation
        output = self.activation(logits)
        
        return output

def dice_coefficient(y_pred, y_true, smooth=1.0):
    """
    Calculate Dice coefficient.
    
    Parameters:
    -----------
    y_pred : torch.Tensor
        Predicted masks
    y_true : torch.Tensor
        Ground truth masks
    alpha : float, optional
        Weight for BCE loss, by default 0.5
    smooth : float, optional
        Smoothing factor for Dice loss, by default 1.0
        
    Returns:
    --------
    torch.Tensor
        Combined loss
    """
    bce = F.binary_cross_entropy(y_pred, y_true)
    dice = dice_loss(y_pred, y_true, smooth)
    
    return alpha * bce + (1 - alpha) * dice

def create_model(model_config=None):
    """
    Create a model based on the provided configuration.
    
    Parameters:
    -----------
    model_config : dict, optional
        Model configuration, by default None
        
    Returns:
    --------
    nn.Module
        PyTorch model
    """
    if model_config is None:
        model_config = MODEL_CONFIG
    
    # Get model parameters
    input_shape = model_config.get('input_shape', (3, 224, 224))
    n_channels = input_shape[0]
    n_classes = model_config.get('num_classes', 1)
    base_filters = model_config.get('custom_model', {}).get('conv_layers', [64])[0]
    dropout_rate = model_config.get('custom_model', {}).get('dropout_rate', 0.1)
    bilinear = model_config.get('custom_model', {}).get('bilinear', True)
    
    # Create model
    logger.info(f"Creating U-Net model with {n_channels} input channels, {n_classes} output classes")
    model = UNet(
        n_channels=n_channels,
        n_classes=n_classes,
        bilinear=bilinear,
        base_filters=base_filters,
        dropout_rate=dropout_rate
    )
    
    return model

def get_optimizer(model, lr=0.001, optimizer_type='adam'):
    """
    Get optimizer for the model.
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model
    lr : float, optional
        Learning rate, by default 0.001
    optimizer_type : str, optional
        Optimizer type ('adam', 'sgd', 'rmsprop'), by default 'adam'
        
    Returns:
    --------
    torch.optim.Optimizer
        PyTorch optimizer
    """
    if optimizer_type.lower() == 'adam':
        return Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type.lower() == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        logger.warning(f"Unknown optimizer: {optimizer_type}. Using Adam.")
        return Adam(model.parameters(), lr=lr)

def get_loss_function(loss_type='combined'):
    """
    Get loss function.
    
    Parameters:
    -----------
    loss_type : str, optional
        Loss function type ('combined', 'dice', 'bce'), by default 'combined'
        
    Returns:
    --------
    function
        Loss function
    """
    if loss_type.lower() == 'combined':
        return combined_loss
    elif loss_type.lower() == 'dice':
        return dice_loss
    elif loss_type.lower() == 'bce':
        return F.binary_cross_entropy
    else:
        logger.warning(f"Unknown loss function: {loss_type}. Using combined loss.")
        return combined_loss

def evaluate_metrics(y_pred, y_true):
    """
    Calculate evaluation metrics for segmentation.
    
    Parameters:
    -----------
    y_pred : torch.Tensor
        Predicted masks
    y_true : torch.Tensor
        Ground truth masks
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    dice = dice_coefficient(y_pred, y_true).item()
    
    # Convert to binary masks for IoU calculation
    pred_binary = (y_pred > 0.5).float()
    true_binary = (y_true > 0.5).float()
    
    # Calculate IoU (Jaccard index)
    intersection = (pred_binary * true_binary).sum().item()
    union = (pred_binary + true_binary).sum().item() - intersection
    iou = intersection / (union + 1e-7)
    
    # Calculate pixel accuracy
    correct = (pred_binary == true_binary).sum().item()
    total = pred_binary.numel()
    accuracy = correct / total
    
    # Calculate precision and recall
    true_positives = (pred_binary * true_binary).sum().item()
    total_predicted_positives = pred_binary.sum().item()
    total_actual_positives = true_binary.sum().item()
    
    precision = true_positives / (total_predicted_positives + 1e-7)
    recall = true_positives / (total_actual_positives + 1e-7)
    
    # F1 score
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    
    metrics = {
        'dice': dice,
        'iou': iou,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

def save_model(model, filepath, optimizer=None, epoch=None, loss=None, metrics=None):
    """
    Save model checkpoint.
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model
    filepath : str
        Path to save the model
    optimizer : torch.optim.Optimizer, optional
        Optimizer, by default None
    epoch : int, optional
        Current epoch, by default None
    loss : float, optional
        Current loss, by default None
    metrics : dict, optional
        Current metrics, by default None
        
    Returns:
    --------
    str
        Path to the saved model
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if loss is not None:
        checkpoint['loss'] = loss
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    try:
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return None

def load_model(filepath, model=None, optimizer=None, device='cuda'):
    """
    Load model checkpoint.
    
    Parameters:
    -----------
    filepath : str
        Path to the model checkpoint
    model : nn.Module, optional
        PyTorch model, by default None
    optimizer : torch.optim.Optimizer, optional
        Optimizer, by default None
    device : str, optional
        Device to load the model to, by default 'cuda'
        
    Returns:
    --------
    tuple
        (model, optimizer, checkpoint)
    """
    try:
        checkpoint = torch.load(filepath, map_location=device)
        
        if model is None:
            model = create_model()
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
        return model, optimizer, checkpoint
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None, None

if __name__ == "__main__":
    # Example usage
    # Create model
    model = create_model()
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Model Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create sample input
    sample_input = torch.randn(1, 3, 224, 224)
    
    # Forward pass
    output = model(sample_input)
    
    print(f"Model output shape: {output.shape}")

def dice_coefficient(y_pred, y_true, smooth=1.0):
    """
    Calculate Dice coefficient.
    
    Parameters:
    -----------
    y_pred : torch.Tensor
        Predicted masks
    y_true : torch.Tensor
        Ground truth masks
    smooth : float, optional
        Smoothing factor to avoid division by zero, by default 1.0
        
    Returns:
    --------
    torch.Tensor
        Dice coefficient
    """
    # Flatten
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    intersection = (y_pred * y_true).sum()
    dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    
    return dice

def dice_loss(y_pred, y_true, smooth=1.0):
    """
    Calculate Dice loss.
    
    Parameters:
    -----------
    y_pred : torch.Tensor
        Predicted masks
    y_true : torch.Tensor
        Ground truth masks
    smooth : float, optional
        Smoothing factor to avoid division by zero, by default 1.0
        
    Returns:
    --------
    torch.Tensor
        Dice loss
    """
    return 1 - dice_coefficient(y_pred, y_true, smooth)

def combined_loss(y_pred, y_true, alpha=0.5, smooth=1.0):
    """
    Calculate combined BCE and Dice loss.
    
    Parameters:
    -----------
    y_pred : torch.Tensor
        Predicted masks
    y_true : torch.Tensor
        Ground truth masks
    alpha : float, optional
        Weight for BCE loss, by default 0.5
    smooth : float, optional
        Smoothing factor for Dice loss, by default 1.0
        
    Returns:
    --------
    torch.Tensor
        Combined loss
    """
    bce = F.binary_cross_entropy(y_pred, y_true)
    dice = dice_loss(y_pred, y_true, smooth)
    
    return alpha * bce + (1 - alpha) * dice