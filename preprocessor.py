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

# Import confi