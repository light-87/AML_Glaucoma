"""
Data Cleaning Module

This module handles cleaning and standardization of the consolidated glaucoma dataset:
- Standardize dataset splits across different sources
- Handle missing values and outliers
- Create consistent dataset structure

Functions:
- standardize_dataset_splits(dataframe): Convert diverse split formats to standard train/val/test
- infer_data_sources(dataframe): Add source information if not present
- clean_dataset(dataframe): Main function to perform all cleaning operations
- validate_dataset(dataframe): Validate dataset structure and report issues

Usage:
    from data_cleaner import clean_dataset
    
    # Clean the consolidated dataset
    cleaned_df = clean_dataset(consolidated_df)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def infer_data_sources(dataframe):
    """
    Infer data sources based on split information if source column doesn't exist.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        The DataFrame containing the dataset
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added or updated 'source' column
    """
    df = dataframe.copy()
    
    # Check if source column exists
    if 'source' not in df.columns:
        logger.info("Creating 'source' column based on split information")
        df['source'] = 'unknown'
        
        # Based on the split patterns:
        # - 'unspecified' entries are from g1020 dataset
        # - 'train', 'val', 'test' entries are from refuge
        # - 'A' and 'B' entries are from origa
        df.loc[df['split'] == 'unspecified', 'source'] = 'g1020'
        df.loc[df['split'].isin(['train', 'val', 'test']), 'source'] = 'refuge'
        df.loc[df['split'].isin(['A', 'B']), 'source'] = 'origa'
        
        # Check if any entries still have unknown source
        unknown_count = (df['source'] == 'unknown').sum()
        if unknown_count > 0:
            logger.warning(f"Could not determine source for {unknown_count} entries")
    
    # Verify that the source matches the dataset column if both exist
    if 'dataset' in df.columns:
        # Create a mapping to standardize the case
        source_map = {
            'g1020': 'G1020',
            'refuge': 'REFUGE',
            'origa': 'ORIGA'
        }
        
        # Check for mismatches
        for source, dataset in source_map.items():
            mismatch = ((df['source'] == source) & (df['dataset'] != dataset)).sum()
            if mismatch > 0:
                logger.warning(f"Found {mismatch} entries where source={source} but dataset!={dataset}")
    
    return df

def standardize_dataset_splits(dataframe, random_state=42):
    """
    Standardize the 'split' column in the dataset to use consistent train/val/test values.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        The DataFrame containing the dataset with a 'split' column
    random_state : int, optional
        Random seed for reproducibility of the splits, by default 42
        
    Returns:
    --------
    pandas.DataFrame
        A copy of the DataFrame with standardized split values
    """
    df = dataframe.copy()
    
    # Ensure source column exists
    df = infer_data_sources(df)
    
    logger.info("Standardizing dataset splits...")
    
    # Log original split distribution
    logger.info(f"Original split distribution:\n{df['split'].value_counts()}")
    logger.info(f"Original split distribution by source:\n{df[['split', 'source']].groupby(['source', 'split']).size()}")
    
    # Handle origa dataset (A/B split)
    # Convert 'A' to 'train' and 'B' to 'test'
    origa_train_count = ((df['source'] == 'origa') & (df['split'] == 'A')).sum()
    origa_test_count = ((df['source'] == 'origa') & (df['split'] == 'B')).sum()
    
    if origa_train_count > 0 or origa_test_count > 0:
        logger.info(f"Converting ORIGA splits: {origa_train_count} 'A' to 'train', {origa_test_count} 'B' to 'test'")
        
        df.loc[(df['source'] == 'origa') & (df['split'] == 'A'), 'split'] = 'train'
        df.loc[(df['source'] == 'origa') & (df['split'] == 'B'), 'split'] = 'test'
        
    # Handle g1020 dataset (unspecified split)
    # Split into train (70%), val (15%), test (15%)
    g1020_indices = df[df['source'] == 'g1020'].index
    g1020_count = len(g1020_indices)
    
    if g1020_count > 0:
        logger.info(f"Splitting G1020 dataset ({g1020_count} entries) into train/val/test")
        
        # Get indices for the splits
        g1020_train_idx, g1020_temp_idx = train_test_split(
            g1020_indices, train_size=0.7, random_state=random_state
        )
        g1020_val_idx, g1020_test_idx = train_test_split(
            g1020_temp_idx, train_size=0.5, random_state=random_state
        )
        
        # Assign the splits
        df.loc[g1020_train_idx, 'split'] = 'train'
        df.loc[g1020_val_idx, 'split'] = 'val'
        df.loc[g1020_test_idx, 'split'] = 'test'
        
        logger.info(f"G1020 split: {len(g1020_train_idx)} train, {len(g1020_val_idx)} val, {len(g1020_test_idx)} test")
    
    # Log final split distribution
    logger.info(f"Final split distribution:\n{df['split'].value_counts()}")
    logger.info(f"Final split distribution by source:\n{df[['split', 'source']].groupby(['source', 'split']).size()}")
    
    return df

def handle_missing_values(dataframe):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        The DataFrame containing the dataset
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with handled missing values
    """
    df = dataframe.copy()
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    
    if not missing_cols.empty:
        logger.info(f"Handling missing values in columns:\n{missing_cols}")
        
        # For path columns, replace None with empty strings for easier handling
        path_columns = [col for col in df.columns if 'path' in col]
        for col in path_columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                logger.info(f"Replacing {null_count} missing values in '{col}' with empty strings")
                df[col] = df[col].fillna('')
        
        # For numeric columns, we can't easily impute without domain knowledge
        # Just log the missing values for now
        numeric_cols = ['cdr', 'ecc_cup', 'ecc_disc', 'fovea_x', 'fovea_y', 'image_width', 'image_height']
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                logger.info(f"Column '{col}' has {df[col].isnull().sum()} missing values. Not imputing.")
    
    return df

def remove_redundant_columns(dataframe):
    """
    Remove redundant columns from the dataset.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        The DataFrame containing the dataset
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with redundant columns removed
    """
    df = dataframe.copy()
    
    # Check for redundant columns
    redundant_cols = []
    
    # has_glaucoma is redundant if same as label
    if 'has_glaucoma' in df.columns and 'label' in df.columns:
        if (df['has_glaucoma'] == df['label']).all():
            redundant_cols.append('has_glaucoma')
            logger.info("'has_glaucoma' column is redundant (same as 'label')")
    
    # Remove redundant columns
    if redundant_cols:
        logger.info(f"Removing redundant columns: {redundant_cols}")
        df = df.drop(columns=redundant_cols)
    
    return df

def validate_dataset(dataframe):
    """
    Validate the dataset and report any issues.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        The DataFrame to validate
        
    Returns:
    --------
    bool
        True if validation passes, False otherwise
    """
    # Check if required columns exist
    required_columns = ['file_id', 'dataset', 'split', 'image_path', 'label']
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    
    if missing_columns:
        logger.error(f"Dataset is missing required columns: {missing_columns}")
        return False
    
    # Check if all entries have a valid split
    valid_splits = ['train', 'val', 'test']
    invalid_splits = dataframe[~dataframe['split'].isin(valid_splits)]
    
    if not invalid_splits.empty:
        logger.error(f"Dataset contains {len(invalid_splits)} entries with invalid splits: {invalid_splits['split'].unique()}")
        return False
    
    # Check if all entries have a valid label
    if 'label' in dataframe.columns:
        invalid_labels = dataframe[~dataframe['label'].isin([0, 1, -1])]
        
        if not invalid_labels.empty:
            logger.error(f"Dataset contains {len(invalid_labels)} entries with invalid labels: {invalid_labels['label'].unique()}")
            return False
    
    # All validations passed
    logger.info("Dataset validation passed")
    return True

def clean_dataset(dataframe, random_state=42):
    """
    Clean the dataset by standardizing splits, handling missing values,
    removing redundant columns, and validating the result.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        The DataFrame to clean
    random_state : int, optional
        Random seed for reproducibility, by default 42
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned DataFrame
    """
    logger.info("Starting dataset cleaning process...")
    
    # Make a copy to avoid modifying the input DataFrame
    df = dataframe.copy()
    
    # Step 1: Standardize dataset splits
    df = standardize_dataset_splits(df, random_state=random_state)
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    # Step 3: Remove redundant columns
    df = remove_redundant_columns(df)
    
    # Step 4: Validate the dataset
    validation_result = validate_dataset(df)
    if not validation_result:
        logger.warning("Dataset validation failed. The dataset may have issues.")
    
    logger.info("Dataset cleaning complete")
    logger.info(f"Cleaned dataset shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    # Example usage
    try:
        import os
        
        # Load the consolidated dataset if it exists
        csv_path = 'consolidated_glaucoma_dataset.csv'
        if os.path.exists(csv_path):
            logger.info(f"Loading dataset from {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Clean the dataset
            cleaned_df = clean_dataset(df)
            
            # Save the cleaned dataset
            cleaned_csv_path = 'cleaned_glaucoma_dataset.csv'
            cleaned_df.to_csv(cleaned_csv_path, index=False)
            logger.info(f"Cleaned dataset saved to {cleaned_csv_path}")
        else:
            logger.error(f"Dataset file not found: {csv_path}")
    
    except Exception as e:
        logger.error(f"Error during dataset cleaning: {e}")