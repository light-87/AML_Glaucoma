"""
Data Cleaning Module

Simplified data cleaning leveraging pandas features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict, List, Optional

logger = logging.getLogger(__name__)

def standardize_dataset_splits(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Standardize dataset splits across different source datasets."""
    # Make a copy to avoid modifying the input DataFrame
    result_df = df.copy()
    
    # Check if 'split' column already exists
    if 'split' not in result_df.columns:
        # Create split column based on stratified sampling
        from sklearn.model_selection import train_test_split
        
        # Check if we have a label column for stratification
        if 'label' in result_df.columns:
            # Get indices for train, val, test splits
            train_idx, temp_idx = train_test_split(
                range(len(result_df)), 
                test_size=0.3, 
                random_state=random_state, 
                stratify=result_df['label']
            )
            
            val_idx, test_idx = train_test_split(
                temp_idx, 
                test_size=0.5, 
                random_state=random_state, 
                stratify=result_df.iloc[temp_idx]['label']
            )
        else:
            # Get indices for train, val, test splits without stratification
            train_idx, temp_idx = train_test_split(
                range(len(result_df)), 
                test_size=0.3, 
                random_state=random_state
            )
            
            val_idx, test_idx = train_test_split(
                temp_idx, 
                test_size=0.5, 
                random_state=random_state
            )
        
        # Assign splits based on indices
        result_df['split'] = 'train'  # Default value
        result_df.loc[val_idx, 'split'] = 'val'
        result_df.loc[test_idx, 'split'] = 'test'
    
    # Ensure dataset-wise consistency (all samples from same patient should be in same split)
    if 'patient_id' in result_df.columns:
        # Get all unique patient IDs
        patient_ids = result_df['patient_id'].unique()
        
        # Shuffle patient IDs
        np.random.seed(random_state)
        np.random.shuffle(patient_ids)
        
        # Split patient IDs into train, val, test
        n_patients = len(patient_ids)
        n_train = int(0.7 * n_patients)
        n_val = int(0.15 * n_patients)
        
        train_patients = patient_ids[:n_train]
        val_patients = patient_ids[n_train:n_train+n_val]
        test_patients = patient_ids[n_train+n_val:]
        
        # Assign splits based on patient IDs
        result_df['split'] = 'train'  # Default value
        result_df.loc[result_df['patient_id'].isin(val_patients), 'split'] = 'val'
        result_df.loc[result_df['patient_id'].isin(test_patients), 'split'] = 'test'
    
    return result_df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    # Make a copy to avoid modifying the input DataFrame
    result_df = df.copy()
    
    # Handle missing image paths
    if 'image_path' in result_df.columns:
        # Drop rows with missing image paths
        missing_images = result_df['image_path'].isna() | (result_df['image_path'] == '')
        if missing_images.any():
            logger.warning(f"Dropping {missing_images.sum()} rows with missing image paths")
            result_df = result_df[~missing_images].reset_index(drop=True)
    
    # Handle missing mask paths for segmentation tasks
    if 'mask_path' in result_df.columns and 'mode' in result_df.columns:
        # Only check for missing masks in segmentation mode
        segmentation_rows = result_df['mode'] == 'segmentation'
        missing_masks = (result_df['mask_path'].isna() | (result_df['mask_path'] == '')) & segmentation_rows
        
        if missing_masks.any():
            logger.warning(f"Dropping {missing_masks.sum()} segmentation rows with missing mask paths")
            result_df = result_df[~missing_masks].reset_index(drop=True)
    
    # Handle missing labels for classification tasks
    if 'label' in result_df.columns and 'mode' in result_df.columns:
        # Only check for missing labels in classification mode
        classification_rows = result_df['mode'] == 'classification'
        missing_labels = result_df['label'].isna() & classification_rows
        
        if missing_labels.any():
            logger.warning(f"Dropping {missing_labels.sum()} classification rows with missing labels")
            result_df = result_df[~missing_labels].reset_index(drop=True)
    
    # Fill missing values in other columns
    for col in result_df.columns:
        if col not in ['image_path', 'mask_path', 'label']:
            if result_df[col].dtype == 'object':
                # Fill missing strings with 'unknown'
                result_df[col] = result_df[col].fillna('unknown')
            elif pd.api.types.is_numeric_dtype(result_df[col]):
                # Fill missing numbers with median
                result_df[col] = result_df[col].fillna(result_df[col].median())
    
    return result_df

def validate_dataset(df: pd.DataFrame) -> bool:
    """Validate the dataset for basic quality checks."""
    valid = True
    
    # Check for minimum required columns
    required_columns = ['image_path']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Required column '{col}' not found in dataset")
            valid = False
    
    # Check for empty dataset
    if len(df) == 0:
        logger.error("Dataset is empty")
        valid = False
    
    # Check for duplicate image paths
    if 'image_path' in df.columns:
        duplicates = df['image_path'].duplicated()
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate image paths")
            # Not failing validation for duplicates, just warning
    
    # Verify file paths exist if image_path column exists
    if 'image_path' in df.columns:
        sample_size = min(100, len(df))  # Check a sample of paths to save time
        sample_paths = df['image_path'].sample(sample_size).tolist()
        missing_files = sum(1 for path in sample_paths if not os.path.exists(path))
        
        if missing_files > 0:
            missing_ratio = missing_files / sample_size
            logger.warning(f"{missing_files}/{sample_size} sampled image paths do not exist ({missing_ratio:.1%})")
            if missing_ratio > 0.5:  # If more than 50% are missing, consider it invalid
                logger.error("Too many image paths are invalid")
                valid = False
    
    return valid

def clean_dataset(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Clean the dataset by standardizing splits, handling missing values, etc."""
    logger.info("Starting dataset cleaning process...")
    
    # Make a copy to avoid modifying the input DataFrame
    cleaned_df = df.copy()
    
    # Add source column if not present based on dataset column
    if 'source' not in cleaned_df.columns and 'dataset' in cleaned_df.columns:
        cleaned_df['source'] = cleaned_df['dataset'].str.lower()
    
    # Standardize splits
    cleaned_df = standardize_dataset_splits(cleaned_df, random_state)
    
    # Handle missing values
    cleaned_df = handle_missing_values(cleaned_df)
    
    # Convert label to numeric and replace NaN with -1
    if 'label' in cleaned_df.columns:
        cleaned_df['label'] = pd.to_numeric(cleaned_df['label'], errors='coerce').fillna(-1).astype(int)
    
    # Remove redundant columns
    if 'has_glaucoma' in cleaned_df.columns and 'label' in cleaned_df.columns:
        if (cleaned_df['has_glaucoma'].fillna(-1) == cleaned_df['label']).all():
            cleaned_df = cleaned_df.drop(columns=['has_glaucoma'])
            logger.info("Removed redundant 'has_glaucoma' column")
    
    # Validate dataset
    validation_result = validate_dataset(cleaned_df)
    if not validation_result:
        logger.warning("Dataset validation failed. The dataset may have issues.")
    
    logger.info(f"Cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

# Simplified versions of helper functions...