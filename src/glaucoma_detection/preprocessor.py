"""
Preprocessing Module

Enhanced preprocessing with PyTorch Lightning and Albumentations.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Tuple, Dict, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

class GlaucomaDataset(Dataset):
    """Dataset for loading glaucoma images and masks."""
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 target_size: Tuple[int, int] = (224, 224), 
                 augment: bool = False, 
                 mode: str = 'segmentation', 
                 normalization: str = 'imagenet'):
        """Initialize the dataset."""
        self.data = data
        self.target_size = target_size
        self.augment = augment
        self.mode = mode
        
        # Define transforms based on mode and augmentation
        self.transforms = self._get_transforms(normalization, augment)
    
    def _get_transforms(self, normalization: str, augment: bool) -> A.Compose:
        """Get the appropriate transforms."""
        # Set normalization parameters
        if normalization == 'imagenet':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean, std = None, None
        
        if augment:
            return A.Compose([
                A.RandomRotate90(),
                A.Flip(),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.Normalize(mean=mean, std=std) if mean else A.Normalize(),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Normalize(mean=mean, std=std) if mean else A.Normalize(),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.data)
    
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        # Get image and mask paths
        row = self.data.iloc[idx]
        image_path = row['image_path']
        mask_path = row['mask_path'] if 'mask_path' in row else None
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask if available
        if mask_path and os.path.exists(mask_path) and self.mode == 'segmentation':
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Normalize mask to 0-1 range
            mask = mask / 255.0 if mask.max() > 1 else mask
            # Resize mask to target size
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=0)  # Add channel dimension for masks
        elif self.mode == 'classification' and 'label' in row:
            # For classification, use label as target
            label = int(row['label'])
            mask = np.array([label], dtype=np.float32)
        else:
            # Empty mask if not available
            mask = np.zeros((1, *self.target_size), dtype=np.float32)
        
        # Resize image to target size
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Apply transforms
        if self.transforms:
            if self.mode == 'segmentation':
                transformed = self.transforms(image=image, mask=mask[0])
                image = transformed["image"]
                mask = transformed["mask"].unsqueeze(0)  # Add channel dimension back
            else:
                transformed = self.transforms(image=image)
                image = transformed["image"]
                mask = torch.tensor(mask, dtype=torch.float32)
        
        return image, mask

class GlaucomaDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Glaucoma datasets."""
    
    def __init__(self, 
                 data_df: pd.DataFrame,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 target_size: Tuple[int, int] = (224, 224),
                 augment_train: bool = True,
                 mode: str = 'segmentation',
                 val_split: float = 0.15,
                 test_split: float = 0.15,
                 random_state: int = 42):
        """Initialize the data module."""
        super().__init__()
        self.data_df = data_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.augment_train = augment_train
        self.mode = mode
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        
        self.train_df = None
        self.val_df = None
        self.test_df = None
    
    def setup(self, stage: Optional[str] = None):
        """Set up the data splits."""
        if 'split' in self.data_df.columns:
            # Use existing splits
            self.train_df = self.data_df[self.data_df['split'] == 'train']
            self.val_df = self.data_df[self.data_df['split'] == 'val']
            self.test_df = self.data_df[self.data_df['split'] == 'test']
        else:
            # Create splits
            from sklearn.model_selection import train_test_split
            
            # Split off test set
            train_val_df, self.test_df = train_test_split(
                self.data_df, 
                test_size=self.test_split,
                random_state=self.random_state,
                stratify=self.data_df['label'] if 'label' in self.data_df.columns else None
            )
            
            # Split remaining data into train and val
            adjusted_val_size = self.val_split / (1 - self.test_split)
            self.train_df, self.val_df = train_test_split(
                train_val_df,
                test_size=adjusted_val_size,
                random_state=self.random_state,
                stratify=train_val_df['label'] if 'label' in train_val_df.columns else None
            )
    
    def train_dataloader(self) -> DataLoader:
        """Get the training data loader."""
        dataset = GlaucomaDataset(
            data=self.train_df,
            target_size=self.target_size,
            augment=self.augment_train,
            mode=self.mode
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get the validation data loader."""
        dataset = GlaucomaDataset(
            data=self.val_df,
            target_size=self.target_size,
            augment=False,
            mode=self.mode
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get the test data loader."""
        dataset = GlaucomaDataset(
            data=self.test_df,
            target_size=self.target_size,
            augment=False,
            mode=self.mode
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )