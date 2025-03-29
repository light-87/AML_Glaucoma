"""
Utility Functions

Simplified utility functions leveraging modern Python features.
"""

import os
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import hashlib
from PIL import Image
import json
from typing import Optional, Dict, Any, Union, List, Tuple

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with both console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler if not already added
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def compute_file_hash(file_path: str) -> Optional[str]:
    """Compute the MD5 hash of a file."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return None
    
    hash_md5 = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logging.error(f"Error computing hash for {path}: {e}")
        return None

def load_image(image_path: str) -> Optional[Image.Image]:
    """Load an image from disk."""
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return None
    
    try:
        return Image.open(path)
    except Exception as e:
        logging.error(f"Error loading image {path}: {e}")
        return None

def save_image(image: Image.Image, output_path: str) -> bool:
    """Save an image to disk."""
    path = Path(output_path)
    
    try:
        # Ensure output directory exists
        path.parent.mkdir(exist_ok=True, parents=True)
        image.save(path)
        return True
    except Exception as e:
        logging.error(f"Error saving image to {path}: {e}")
        return False

def load_json(json_path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON file."""
    path = Path(json_path)
    if not path.exists() or not path.is_file():
        return None
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON file {path}: {e}")
        return None

def save_json(data: Dict[str, Any], output_path: str) -> bool:
    """Save data as a JSON file."""
    path = Path(output_path)
    
    try:
        # Ensure output directory exists
        path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Error saving JSON to {path}: {e}")
        return False

def load_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """Load a CSV file."""
    path = Path(csv_path)
    if not path.exists() or not path.is_file():
        return None
    
    try:
        return pd.read_csv(path)
    except Exception as e:
        logging.error(f"Error loading CSV file {path}: {e}")
        return None

def save_csv(df: pd.DataFrame, output_path: str) -> bool:
    """Save a DataFrame as a CSV file."""
    path = Path(output_path)
    
    try:
        # Ensure output directory exists
        path.parent.mkdir(exist_ok=True, parents=True)
        
        df.to_csv(path, index=False)
        return True
    except Exception as e:
        logging.error(f"Error saving CSV to {path}: {e}")
        return False

def get_dataset_statistics(dataframe: pd.DataFrame) -> Dict[str, Any]:
    """Get statistics of a dataset."""
    stats = {}
    
    # Basic statistics
    stats['num_samples'] = len(dataframe)
    
    # Class distribution if 'label' column exists
    if 'label' in dataframe.columns:
        label_counts = dataframe['label'].value_counts().to_dict()
        stats['class_distribution'] = {
            'glaucoma': label_counts.get(1, 0),
            'normal': label_counts.get(0, 0),
            'unknown': label_counts.get(-1, 0) if -1 in label_counts else 0
        }
    
    # Dataset and split distribution
    for col in ['dataset', 'split', 'source']:
        if col in dataframe.columns:
            stats[f'{col}_distribution'] = dataframe[col].value_counts().to_dict()
    
    # Check for missing values
    stats['missing_values'] = dataframe.isnull().sum().to_dict()
    
    return stats

def print_dataset_statistics(dataframe: pd.DataFrame) -> None:
    """Print statistics of a dataset."""
    stats = get_dataset_statistics(dataframe)
    
    print("Dataset Statistics:")
    print(f"Total samples: {stats['num_samples']}")
    
    for key, distribution in stats.items():
        if key.endswith('_distribution'):
            print(f"\n{key.replace('_', ' ').title()}:")
            for category, count in distribution.items():
                print(f"  - {category}: {count} ({count/stats['num_samples']*100:.2f}%)")
    
    print("\nMissing Values:")
    missing_values = {k: v for k, v in stats['missing_values'].items() if v > 0}
    if missing_values:
        for col, count in missing_values.items():
            print(f"  - {col}: {count} ({count/stats['num_samples']*100:.2f}%)")
    else:
        print("  No missing values")

def verify_file_paths(dataframe: pd.DataFrame, path_columns: List[str]) -> Dict[str, Any]:
    """Verify that file paths in a DataFrame exist."""
    results = {
        'total_files': 0,
        'existing_files': 0,
        'missing_files': 0,
        'missing_files_by_column': {},
        'missing_file_examples': {}
    }
    
    for col in path_columns:
        if col not in dataframe.columns:
            continue
            
        # Skip empty or missing values
        valid_paths = dataframe[dataframe[col].notna() & (dataframe[col] != '')][col]
        
        total = len(valid_paths)
        results['total_files'] += total
        
        # Check if files exist
        missing_files = [path for path in valid_paths if not Path(path).exists()]
        missing_count = len(missing_files)
        
        results['missing_files'] += missing_count
        results['existing_files'] += (total - missing_count)
        results['missing_files_by_column'][col] = missing_count
        
        # Store examples of missing files
        if missing_files:
            results['missing_file_examples'][col] = missing_files[:5]  # Store up to 5 examples
    
    return results