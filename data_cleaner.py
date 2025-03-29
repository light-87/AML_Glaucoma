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