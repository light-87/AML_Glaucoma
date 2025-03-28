import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

from utils import setup_logger

# Set up logging
logger = setup_logger('preprocessor')

class GlaucomaDataset(Dataset):
    """
    Dataset for loading glaucoma images and masks.
    
    Attributes:
    -----------
    data : pandas.DataFrame
        DataFrame containing image paths and labels
    target_size : tuple
        Target size for images (width, height)
    augment : bool
        Whether to apply data augmentation
    mode : str
        Mode of operation ('segmentation' or 'classification')
    transform : albumentations.Compose
        Data augmentation pipeline
    normalize : albumentations.Normalize
        Normalization transform
    """
    
    def __init__(self, data, target_size=(224, 224), augment=False, 
                 mode='segmentation', normalization='imagenet'):
        """
        Initialize the dataset.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing image paths and labels
        target_size : tuple, optional
            Target size for images (width, height), by default (224, 224)
        augment : bool, optional
            Whether to apply data augmentation, by default False
        mode : str, optional
            Mode of operation ('segmentation' or 'classification'), by default 'segmentation'
        normalization : str, optional
            Normalization type ('imagenet', 'instance', 'pixel', 'none'), by default 'imagenet'
        """
        self.data = data
        self.target_size = target_size
        self.augment = augment
        self.mode = mode
        
        # Set up normalization
        if normalization == 'imagenet':
            self.normalize = A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        elif normalization == 'instance':
            self.normalize = A.Normalize()
        elif normalization == 'pixel':
            self.normalize = A.Normalize(mean=0, std=1)
        else:
            self.normalize = None
        
        # Set up augmentation pipeline
        if self.augment:
            self.transform = A.Compose([
                A.RandomRotate90(),
                A.Flip(),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.GaussianBlur(blur_limit=(3, 5)),
                self.normalize if self.normalize else A.NoOp(),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                self.normalize if self.normalize else A.NoOp(),
                ToTensorV2()
            ])
    
    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
        --------
        int
            Number of samples in the dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Parameters:
        -----------
        idx : int
            Index of the sample
            
        Returns:
        --------
        tuple
            (image, mask) for segmentation or (image, label) for classification
        """
        # Get sample data
        sample = self.data.iloc[idx]
        
        # Determine which image path to use
        if self.mode == 'segmentation':
            # Prefer square images, then cropped, then original
            if sample.get('image_square_path') and os.path.exists(sample['image_square_path']):
                img_path = sample['image_square_path']
            elif sample.get('image_cropped_path') and os.path.exists(sample['image_cropped_path']):
                img_path = sample['image_cropped_path']
            elif sample.get('image_path') and os.path.exists(sample['image_path']):
                img_path = sample['image_path']
            else:
                logger.warning(f"No valid image path found for sample {idx}")
                # Return a black image as fallback
                img = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
                mask = np.zeros((self.target_size[1], self.target_size[0], 1), dtype=np.uint8)
                
                # Apply transforms
                transformed = self.transform(image=img, mask=mask)
                return transformed['image'], transformed['mask']
            
            # Determine which mask path to use
            if sample.get('mask_square_path') and os.path.exists(sample['mask_square_path']):
                mask_path = sample['mask_square_path']
            elif sample.get('mask_cropped_path') and os.path.exists(sample['mask_cropped_path']):
                mask_path = sample['mask_cropped_path']
            elif sample.get('mask_path') and os.path.exists(sample['mask_path']):
                mask_path = sample['mask_path']
            else:
                logger.warning(f"No valid mask path found for sample {idx}")
                mask_path = None
        else:  # classification mode
            # Prefer square images, then cropped, then original
            if sample.get('image_square_path') and os.path.exists(sample['image_square_path']):
                img_path = sample['image_square_path']
            elif sample.get('image_cropped_path') and os.path.exists(sample['image_cropped_path']):
                img_path = sample['image_cropped_path']
            elif sample.get('image_path') and os.path.exists(sample['image_path']):
                img_path = sample['image_path']
            else:
                logger.warning(f"No valid image path found for sample {idx}")
                # Return a black image as fallback
                img = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
                label = torch.tensor(0, dtype=torch.float32)
                
                # Apply transforms
                transformed = self.transform(image=img)
                return transformed['image'], label
            
        # Load image
        try:
            # Use OpenCV for faster loading
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            img = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        
        # Resize image to target size
        img = cv2.resize(img, self.target_size)
        
        if self.mode == 'segmentation':
            # Load mask for segmentation
            try:
                if mask_path:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, self.target_size)
                    # Normalize mask to 0-1
                    mask = mask / 255.0 if mask.max() > 1 else mask
                else:
                    # If no valid mask path, create an empty mask
                    mask = np.zeros((self.target_size[1], self.target_size[0]), dtype=np.float32)
            except Exception as e:
                logger.error(f"Error loading mask {mask_path}: {e}")
                mask = np.zeros((self.target_size[1], self.target_size[0]), dtype=np.float32)
            
            # Add channel dimension to mask
            mask = np.expand_dims(mask, axis=2)
            
            # Apply transforms
            transformed = self.transform(image=img, mask=mask)
            return transformed['image'], transformed['mask']
        else:
            # Get label for classification
            label = sample['label']
            if pd.isna(label):
                label = 0
            
            # Apply transforms
            transformed = self.transform(image=img)
            return transformed['image'], torch.tensor(label, dtype=torch.float32)

def preprocess_dataset(df, output_dir=None, save_images=False):
    """
    Preprocess dataset for model training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing dataset information
    output_dir : str, optional
        Directory to save preprocessed data, by default None
    save_images : bool, optional
        Whether to save preprocessed images, by default False
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed DataFrame
    """
    logger.info("Preprocessing dataset...")
    
    # Make a copy to avoid modifying the original
    preprocessed_df = df.copy()
    
    # Convert label to numeric if not already
    if 'label' in preprocessed_df.columns:
        preprocessed_df['label'] = pd.to_numeric(preprocessed_df['label'], errors='coerce')
        # Replace NaN values with -1 (unknown)
        preprocessed_df['label'] = preprocessed_df['label'].fillna(-1)
    
    # Log class distribution
    if 'label' in preprocessed_df.columns:
        label_counts = preprocessed_df['label'].value_counts()
        logger.info(f"Class distribution:\n{label_counts}")
    
    # Save preprocessed dataset
    if output_dir:
        output_file = os.path.join(output_dir, 'preprocessed_glaucoma_dataset.csv')
        preprocessed_df.to_csv(output_file, index=False)
        logger.info(f"Saved preprocessed dataset to {output_file}")
    
    return preprocessed_df

def create_dataset_splits(df, val_size=0.15, test_size=0.15, random_state=42, stratify=True):
    """
    Create train, validation, and test splits from the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing dataset information
    val_size : float, optional
        Proportion of data for validation, by default 0.15
    test_size : float, optional
        Proportion of data for testing, by default 0.15
    random_state : int, optional
        Random seed for reproducibility, by default 42
    stratify : bool, optional
        Whether to stratify splits based on label, by default True
        
    Returns:
    --------
    tuple
        (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    logger.info("Creating dataset splits...")
    
    # Check if the data already has a 'split' column
    if 'split' in df.columns:
        logger.info("Using existing split information")
        train_df = df[df['split'] == 'train'].copy()
        val_df = df[df['split'] == 'val'].copy()
        test_df = df[df['split'] == 'test'].copy()
        
        logger.info(f"Split distribution: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")
        return train_df, val_df, test_df
    
    # Stratify by label if requested and available
    stratify_col = df['label'] if stratify and 'label' in df.columns else None
    
    # First split off the test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    # Update stratify column for the next split
    if stratify_col is not None:
        stratify_col = train_val_df['label']
    
    # Split the remaining data into train and validation sets
    # Adjust validation size to account for the test split
    adjusted_val_size = val_size / (1 - test_size)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    # Add split information to the dataframes
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    logger.info(f"Split distribution: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")
    
    return train_df, val_df, test_df

def create_dataloaders(train_df, val_df, test_df=None, batch_size=32, num_workers=4, 
                      augment_train=True, target_size=(224, 224), mode='segmentation'):
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training data
    val_df : pandas.DataFrame
        Validation data
    test_df : pandas.DataFrame, optional
        Test data, by default None
    batch_size : int, optional
        Batch size, by default 32
    num_workers : int, optional
        Number of worker threads for data loading, by default 4
    augment_train : bool, optional
        Whether to apply data augmentation to training data, by default True
    target_size : tuple, optional
        Target image size (width, height), by default (224, 224)
    mode : str, optional
        Mode of operation ('segmentation' or 'classification'), by default 'segmentation'
        
    Returns:
    --------
    tuple
        (train_loader, val_loader, test_loader)
    """
    logger.info("Creating DataLoaders...")
    
    # Create datasets
    train_dataset = GlaucomaDataset(
        train_df,
        target_size=target_size,
        augment=augment_train,
        mode=mode
    )
    
    val_dataset = GlaucomaDataset(
        val_df,
        target_size=target_size,
        augment=False,  # No augmentation for validation
        mode=mode
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create test loader if test data is provided
    test_loader = None
    if test_df is not None:
        test_dataset = GlaucomaDataset(
            test_df,
            target_size=target_size,
            augment=False,  # No augmentation for test
            mode=mode
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    logger.info(f"Created DataLoaders with batch size {batch_size}")
    logger.info(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    if test_loader:
        logger.info(f"Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader