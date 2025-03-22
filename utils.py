"""
Utility Functions

This module contains utility functions used across the pipeline components:
- File handling
- Path creation and validation
- Common image processing operations
- Logging utilities

Usage:
    from utils import create_directory, get_file_extension, setup_logger
"""

import os
import logging
import sys
import json
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
import pandas as pd

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger with both console and file handlers.
    
    Parameters:
    -----------
    name : str
        Logger name
    log_file : str, optional
        Path to log file, by default None
    level : int, optional
        Logging level, by default logging.INFO
        
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def create_directory(directory_path):
    """
    Create a directory if it doesn't exist.
    
    Parameters:
    -----------
    directory_path : str
        Path to the directory to create
        
    Returns:
    --------
    str
        Path to the created directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    return directory_path

def get_file_extension(file_path):
    """
    Get the extension of a file.
    
    Parameters:
    -----------
    file_path : str
        Path to the file
        
    Returns:
    --------
    str
        File extension without the dot
    """
    return Path(file_path).suffix.lstrip('.')

def file_exists(file_path):
    """
    Check if a file exists and is accessible.
    
    Parameters:
    -----------
    file_path : str
        Path to the file
        
    Returns:
    --------
    bool
        True if the file exists and is accessible, False otherwise
    """
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)

def compute_file_hash(file_path):
    """
    Compute the MD5 hash of a file.
    
    Parameters:
    -----------
    file_path : str
        Path to the file
        
    Returns:
    --------
    str
        MD5 hash of the file
    """
    hash_md5 = hashlib.md5()
    
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logging.error(f"Error computing hash for {file_path}: {e}")
        return None

def load_image(image_path):
    """
    Load an image from disk.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
        
    Returns:
    --------
    PIL.Image or None
        Loaded image or None if loading fails
    """
    try:
        return Image.open(image_path)
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None

def save_image(image, output_path):
    """
    Save an image to disk.
    
    Parameters:
    -----------
    image : PIL.Image
        Image to save
    output_path : str
        Path to save the image
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        image.save(output_path)
        return True
    except Exception as e:
        logging.error(f"Error saving image to {output_path}: {e}")
        return False

def load_json(json_path):
    """
    Load a JSON file.
    
    Parameters:
    -----------
    json_path : str
        Path to the JSON file
        
    Returns:
    --------
    dict or None
        Loaded JSON data or None if loading fails
    """
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON file {json_path}: {e}")
        return None

def save_json(data, output_path):
    """
    Save data as a JSON file.
    
    Parameters:
    -----------
    data : dict
        Data to save
    output_path : str
        Path to save the JSON file
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Error saving JSON to {output_path}: {e}")
        return False

def load_csv(csv_path):
    """
    Load a CSV file.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame or None
        Loaded DataFrame or None if loading fails
    """
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Error loading CSV file {csv_path}: {e}")
        return None

def save_csv(df, output_path):
    """
    Save a DataFrame as a CSV file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to save
    output_path : str
        Path to save the CSV file
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        logging.error(f"Error saving CSV to {output_path}: {e}")
        return False

def get_dataset_statistics(dataframe):
    """
    Get statistics of a dataset.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        DataFrame containing the dataset
        
    Returns:
    --------
    dict
        Dictionary with dataset statistics
    """
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
    if 'dataset' in dataframe.columns:
        stats['dataset_distribution'] = dataframe['dataset'].value_counts().to_dict()
    
    if 'split' in dataframe.columns:
        stats['split_distribution'] = dataframe['split'].value_counts().to_dict()
    
    # Check for missing values
    stats['missing_values'] = dataframe.isnull().sum().to_dict()
    
    return stats

def print_dataset_statistics(dataframe):
    """
    Print statistics of a dataset.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        DataFrame containing the dataset
    """
    stats = get_dataset_statistics(dataframe)
    
    print("Dataset Statistics:")
    print(f"Total samples: {stats['num_samples']}")
    
    if 'class_distribution' in stats:
        print("\nClass Distribution:")
        for cls, count in stats['class_distribution'].items():
            print(f"  - {cls}: {count} ({count/stats['num_samples']*100:.2f}%)")
    
    if 'dataset_distribution' in stats:
        print("\nDataset Distribution:")
        for dataset, count in stats['dataset_distribution'].items():
            print(f"  - {dataset}: {count} ({count/stats['num_samples']*100:.2f}%)")
    
    if 'split_distribution' in stats:
        print("\nSplit Distribution:")
        for split, count in stats['split_distribution'].items():
            print(f"  - {split}: {count} ({count/stats['num_samples']*100:.2f}%)")
    
    print("\nMissing Values:")
    missing_values = {k: v for k, v in stats['missing_values'].items() if v > 0}
    if missing_values:
        for col, count in missing_values.items():
            print(f"  - {col}: {count} ({count/stats['num_samples']*100:.2f}%)")
    else:
        print("  No missing values")

def verify_file_paths(dataframe, path_columns):
    """
    Verify that file paths in a DataFrame exist.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        DataFrame containing file paths
    path_columns : list
        List of column names containing file paths
        
    Returns:
    --------
    dict
        Dictionary with verification results
    """
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
        missing_files = [path for path in valid_paths if not file_exists(path)]
        missing_count = len(missing_files)
        
        results['missing_files'] += missing_count
        results['existing_files'] += (total - missing_count)
        results['missing_files_by_column'][col] = missing_count
        
        # Store examples of missing files
        if missing_files:
            results['missing_file_examples'][col] = missing_files[:5]  # Store up to 5 examples
    
    return results

def print_file_verification_results(results):
    """
    Print file verification results.
    
    Parameters:
    -----------
    results : dict
        Dictionary with verification results from verify_file_paths()
    """
    print("File Verification Results:")
    print(f"Total files: {results['total_files']}")
    print(f"Existing files: {results['existing_files']} ({results['existing_files']/results['total_files']*100:.2f}% if results['total_files'] > 0 else 0}}%)")
    print(f"Missing files: {results['missing_files']} ({results['missing_files']/results['total_files']*100:.2f}% if results['total_files'] > 0 else 0}}%)")
    
    if results['missing_files'] > 0:
        print("\nMissing Files by Column:")
        for col, count in results['missing_files_by_column'].items():
            if count > 0:
                print(f"  - {col}: {count}")
                
                # Print examples
                if col in results['missing_file_examples']:
                    print("    Examples:")
                    for example in results['missing_file_examples'][col]:
                        print(f"{example}")