"""
Custom Loss Functions

Combined loss implementations for the glaucoma detection pipeline.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class CombinedLoss(nn.Module):
    """Combined loss function that adds Dice and BCE losses."""
    
    def __init__(self, mode='binary'):
        """Initialize with the specified mode."""
        super().__init__()
        self.dice_loss = smp.losses.DiceLoss(mode=mode)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, y_pred, y_true):
        """Compute the combined loss."""
        dice = self.dice_loss(y_pred, y_true)
        bce = self.bce_loss(y_pred, y_true)
        return dice + bce