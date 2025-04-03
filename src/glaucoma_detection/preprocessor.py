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
                # Use correct Albumentations transforms
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),  # Replace Flip() with HorizontalFlip()
                A.VerticalFlip(p=0.5),    # Add VerticalFlip if needed
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(0.0, 0.05), p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
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
    
    def _process_mask(self, mask: np.ndarray) -> np.ndarray:
        """Ensure mask is binary (0 or 1)."""
        # Normalize to 0-1 range
        if mask.max() > 1:
            mask = mask / 255.0
        
        # Threshold to create binary mask
        mask = (mask > 0.5).astype(np.float32)
        
        return mask
    
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
            
            # Resize mask to target size
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            
            # Process mask to ensure binary values
            mask = self._process_mask(mask)
            
            # Add channel dimension for masks
            mask = np.expand_dims(mask, axis=0)
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
                # For segmentation, we need to ensure mask is 2D for albumentations
                mask_2d = mask[0]  # Get the first channel (remove channel dimension)
                transformed = self.transforms(image=image, mask=mask_2d)
                image = transformed["image"]
                mask = transformed["mask"].unsqueeze(0)  # Add channel dimension back
                
                # Ensure binary mask after transforms
                mask = (mask > 0.5).float()
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
             augment_train: bool = False, 
             mode: str = 'segmentation',
             val_split: float = 0.15,
             test_split: float = 0.15,
             random_state: int = 42,
             use_memory_efficient: bool = False,
             cache_dir: Optional[str] = None,
             use_existing_splits: bool = False,
             train_df: Optional[pd.DataFrame] = None,
             val_df: Optional[pd.DataFrame] = None,
             test_df: Optional[pd.DataFrame] = None):
        
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
        # Add these lines after the other initializations
        self.use_memory_efficient = use_memory_efficient
        self.cache_dir = cache_dir
        self.use_existing_splits = use_existing_splits

        # Set dataframes if using existing splits
        if use_existing_splits:
            self.train_df = train_df
            self.val_df = val_df
            self.test_df = test_df
    
    def setup(self, stage: Optional[str] = None):
        """Set up the data splits with improved stratification."""
        # If using existing splits, just use them directly
        if self.use_existing_splits and self.train_df is not None and self.val_df is not None and self.test_df is not None:
            logger.info("Using existing data splits")
            return
            
        # Check if 'split' column already exists in the dataframe
        if 'split' in self.data_df.columns:
            # Use existing splits from the dataframe
            self.train_df = self.data_df[self.data_df['split'] == 'train']
            self.val_df = self.data_df[self.data_df['split'] == 'val']
            self.test_df = self.data_df[self.data_df['split'] == 'test']
            
            logger.info(f"Using predefined splits: Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")
            return

        # If no predefined splits, create stratified splits
        from sklearn.model_selection import train_test_split
        
        # Ensure we have a label column for stratification
        if 'label' not in self.data_df.columns:
            # If no label, create a placeholder
            self.data_df['label'] = 0
        
        # Stratify by dataset and label to ensure balanced representation
        def stratify_key(row):
            return f"{row['dataset']}_{row['label']}"
        
        self.data_df['stratify_key'] = self.data_df.apply(stratify_key, axis=1)
        
        # First, split off test set
        train_val_df, self.test_df = train_test_split(
            self.data_df, 
            test_size=self.test_split,
            random_state=self.random_state,
            stratify=self.data_df['stratify_key']
        )
        
        # Then split train and validation
        adjusted_val_size = self.val_split / (1 - self.test_split)
        self.train_df, self.val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_size,
            random_state=self.random_state,
            stratify=train_val_df['stratify_key']
        )
        
        # Remove temporary stratify column
        self.data_df.drop(columns=['stratify_key'], inplace=True)
        
        # Logging detailed split information
        logger.info("Dataset Split Details:")
        logger.info(f"Total Samples: {len(self.data_df)}")
        logger.info(f"Train Samples: {len(self.train_df)} ({len(self.train_df)/len(self.data_df)*100:.2f}%)")
        logger.info(f"Validation Samples: {len(self.val_df)} ({len(self.val_df)/len(self.data_df)*100:.2f}%)")
        logger.info(f"Test Samples: {len(self.test_df)} ({len(self.test_df)/len(self.data_df)*100:.2f}%)")
        
        # Dataset distribution in each split
        logger.info("\nDataset Distribution in Splits:")
        for split_name, split_df in [('Train', self.train_df), ('Validation', self.val_df), ('Test', self.test_df)]:
            logger.info(f"\n{split_name} Split:")
            logger.info(split_df['dataset'].value_counts())
            logger.info("Label Distribution:")
            logger.info(split_df['label'].value_counts(normalize=True))
    def train_dataloader(self) -> DataLoader:
        """Get the training data loader."""
        if self.use_memory_efficient:
            from glaucoma_detection.memory_efficient_loader import create_memory_efficient_data_loader
            return create_memory_efficient_data_loader(
                data=self.train_df,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                target_size=self.target_size,
                augment=self.augment_train,
                mode=self.mode,
                shuffle=True,
                cache_dir=self.cache_dir
            )
        else:
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
        if self.use_memory_efficient:
            from glaucoma_detection.memory_efficient_loader import create_memory_efficient_data_loader
            return create_memory_efficient_data_loader(
                data=self.val_df,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                target_size=self.target_size,
                augment=False,  # No augmentation for validation
                mode=self.mode,
                shuffle=False,
                cache_dir=self.cache_dir
            )
        else:
            dataset = GlaucomaDataset(
                data=self.val_df,
                target_size=self.target_size,
                augment=False,  # No augmentation for validation
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
        if self.use_memory_efficient:
            from glaucoma_detection.memory_efficient_loader import create_memory_efficient_data_loader
            return create_memory_efficient_data_loader(
                data=self.test_df,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                target_size=self.target_size,
                augment=False,  # No augmentation for testing
                mode=self.mode,
                shuffle=False,
                cache_dir=self.cache_dir
            )
        else:
            dataset = GlaucomaDataset(
                data=self.test_df,
                target_size=self.target_size,
                augment=False,  # No augmentation for testing
                mode=self.mode
            )
            return DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )