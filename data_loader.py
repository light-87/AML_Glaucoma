"""
Data Loading & Extraction Module

This module handles data extraction, loading, and consolidation for glaucoma datasets:
- ORIGA
- REFUGE
- G1020

Functions:
- extract_zip(zip_file, output_dir): Extract ZIP files
- load_origa(dataset_path): Load ORIGA dataset
- load_refuge(dataset_path): Load REFUGE dataset
- load_g1020(dataset_path): Load G1020 dataset
- consolidate_datasets(base_path): Combine all datasets into one DataFrame
- save_consolidated_dataset(df, output_file): Save consolidated DataFrame to CSV

Usage:
    from data_loader import consolidate_datasets, save_consolidated_dataset
    
    # Consolidate datasets from base path
    df = consolidate_datasets('/path/to/datasets')
    
    # Save to CSV
    save_consolidated_dataset(df, 'consolidated_glaucoma_dataset.csv')
"""

import os
import json
import pandas as pd
import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
import zipfile
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_zip(zip_file, output_dir=None):
    """
    Extract a ZIP file to the specified output directory.
    
    Parameters:
    -----------
    zip_file : str
        Path to the ZIP file to extract
    output_dir : str, optional
        Directory to extract the ZIP file to. If None, extracts to the same directory.
        
    Returns:
    --------
    str
        Path to the extracted directory
    """
    if output_dir is None:
        output_dir = os.path.dirname(zip_file)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            logger.info(f"Extracting {zip_file} to {output_dir}")
            zip_ref.extractall(output_dir)
            
            # Get the name of the top-level folder in the ZIP file, if any
            top_dirs = {item.split('/')[0] for item in zip_ref.namelist() if '/' in item}
            if len(top_dirs) == 1:
                extracted_dir = os.path.join(output_dir, list(top_dirs)[0])
            else:
                extracted_dir = output_dir
                
        logger.info(f"Extraction complete: {extracted_dir}")
        return extracted_dir
    
    except Exception as e:
        logger.error(f"Error extracting {zip_file}: {e}")
        raise

def load_origa(dataset_path):
    """
    Load and process ORIGA dataset.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the ORIGA dataset directory
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing ORIGA dataset information
    """
    logger.info("Processing ORIGA dataset...")
    
    # Initialize column names
    columns = [
        'file_id', 'dataset', 'split', 'image_path', 'image_cropped_path', 'image_square_path',
        'mask_path', 'mask_cropped_path', 'mask_square_path', 'nerve_removed_path',
        'label', 'has_glaucoma', 'eye_laterality', 'cdr', 'ecc_cup', 'ecc_disc',
        'fovea_x', 'fovea_y', 'image_width', 'image_height'
    ]
    
    # Read metadata files
    origa_list_path = os.path.join(dataset_path, 'OrigaList.csv')
    origa_info_path = os.path.join(dataset_path, 'origa_info.csv')
    
    if not os.path.exists(origa_list_path):
        logger.error(f"ORIGA metadata file not found: {origa_list_path}")
        return pd.DataFrame(columns=columns)
        
    if not os.path.exists(origa_info_path):
        logger.warning(f"ORIGA info file not found: {origa_info_path}")
        origa_info = pd.DataFrame()
    else:
        origa_info = pd.read_csv(origa_info_path)
    
    origa_list = pd.read_csv(origa_list_path)
    
    # Initialize DataFrame
    origa_data = pd.DataFrame(columns=columns)
    
    # Identify all image files
    image_dir = os.path.join(dataset_path, 'Images')
    if not os.path.exists(image_dir):
        logger.error(f"ORIGA images directory not found: {image_dir}")
        return pd.DataFrame(columns=columns)
        
    image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing ORIGA images"):
        img_filename = os.path.basename(img_path)
        img_id = os.path.splitext(img_filename)[0]
        
        # Find metadata for this image
        img_meta = origa_list[origa_list['Filename'] == img_filename]
        
        if len(img_meta) == 0:
            logger.warning(f"No metadata found for {img_filename}, skipping")
            continue
            
        # Find additional info
        img_info = pd.DataFrame()
        if not origa_info.empty:
            img_info = origa_info[origa_info['Image'].str.contains(f"/{img_id}.jpg", na=False)]
        
        # Get image dimensions
        width, height = None, None
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            logger.warning(f"Error reading image dimensions for {img_path}: {e}")
        
        # Construct paths for all versions
        image_path = img_path
        image_cropped_path = os.path.join(dataset_path, 'Images_Cropped', img_filename)
        image_square_path = os.path.join(dataset_path, 'Images_Square', img_filename)
        
        mask_path = os.path.join(dataset_path, 'Masks', f"{img_id}.png")
        mask_cropped_path = os.path.join(dataset_path, 'Masks_Cropped', f"{img_id}.png")
        mask_square_path = os.path.join(dataset_path, 'Masks_Square', f"{img_id}.png")
        
        # Get label info
        has_glaucoma = int(img_meta['Glaucoma'].iloc[0])
        eye_laterality = img_meta['Eye'].iloc[0]  # OD (right) or OS (left)
        
        # Get CDR and other metrics if available
        cdr = float(img_info['CDR'].iloc[0]) if len(img_info) > 0 and 'CDR' in img_info.columns and not img_info['CDR'].iloc[0] in [None, np.nan, ''] else None
        ecc_cup = float(img_info['Ecc-Cup'].iloc[0]) if len(img_info) > 0 and 'Ecc-Cup' in img_info.columns and not img_info['Ecc-Cup'].iloc[0] in [None, np.nan, ''] else None
        ecc_disc = float(img_info['Ecc-Disc'].iloc[0]) if len(img_info) > 0 and 'Ecc-Disc' in img_info.columns and not img_info['Ecc-Disc'].iloc[0] in [None, np.nan, ''] else None
        
        # Add to dataset
        new_row = {
            'file_id': img_id,
            'dataset': 'ORIGA',
            'split': img_meta['Set'].iloc[0],  # A or Z sets
            'image_path': image_path,
            'image_cropped_path': image_cropped_path if os.path.exists(image_cropped_path) else None,
            'image_square_path': image_square_path if os.path.exists(image_square_path) else None,
            'mask_path': mask_path if os.path.exists(mask_path) else None,
            'mask_cropped_path': mask_cropped_path if os.path.exists(mask_cropped_path) else None,
            'mask_square_path': mask_square_path if os.path.exists(mask_square_path) else None,
            'nerve_removed_path': None,  # ORIGA doesn't have nerve removed images
            'label': has_glaucoma,
            'has_glaucoma': has_glaucoma,
            'eye_laterality': eye_laterality,
            'cdr': cdr,
            'ecc_cup': ecc_cup,
            'ecc_disc': ecc_disc,
            'fovea_x': None,  # ORIGA doesn't have fovea coordinates
            'fovea_y': None,
            'image_width': width,
            'image_height': height
        }
        
        origa_data = pd.concat([origa_data, pd.DataFrame([new_row])], ignore_index=True)
    
    logger.info(f"Processed {len(origa_data)} ORIGA images")
    return origa_data

def load_refuge(dataset_path):
    """
    Load and process REFUGE dataset.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the REFUGE dataset directory
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing REFUGE dataset information
    """
    logger.info("Processing REFUGE dataset...")
    
    # Initialize column names
    columns = [
        'file_id', 'dataset', 'split', 'image_path', 'image_cropped_path', 'image_square_path',
        'mask_path', 'mask_cropped_path', 'mask_square_path', 'nerve_removed_path',
        'label', 'has_glaucoma', 'eye_laterality', 'cdr', 'ecc_cup', 'ecc_disc',
        'fovea_x', 'fovea_y', 'image_width', 'image_height'
    ]
    
    # Initialize DataFrame
    refuge_data = pd.DataFrame(columns=columns)
    
    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, split)
        
        if not os.path.exists(split_path):
            logger.warning(f"REFUGE {split} directory not found: {split_path}")
            continue
            
        # Load index.json which contains metadata
        index_path = os.path.join(split_path, 'index.json')
        if not os.path.exists(index_path):
            logger.warning(f"index.json not found for {split} split")
            continue
            
        try:
            with open(index_path, 'r') as f:
                index_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing index.json for {split} split: {e}")
            continue
            
        # Process each image in the split
        for idx, img_info in tqdm(index_data.items(), desc=f"Processing REFUGE {split}"):
            img_filename = img_info['ImgName']
            img_id = os.path.splitext(img_filename)[0]
            
            # Construct paths for all versions
            image_path = os.path.join(split_path, 'Images', img_filename)
            image_cropped_path = os.path.join(split_path, 'Images_Cropped', img_filename)
            
            # Square images are stored at the root level for REFUGE
            image_square_path = os.path.join(dataset_path, 'Images_Square', img_filename)
            
            # Mask paths
            mask_filename = f"{img_id}.png"
            mask_path = os.path.join(split_path, 'Masks', mask_filename)
            mask_cropped_path = os.path.join(split_path, 'Masks_Cropped', mask_filename)
            mask_square_path = os.path.join(dataset_path, 'Masks_Square', mask_filename)
            
            # Get image dimensions from the metadata
            width = img_info.get('Size_X', None)
            height = img_info.get('Size_Y', None)
            
            # Get label info
            has_glaucoma = img_info.get('Label', None)
            if has_glaucoma is None and split == 'test':
                # Test set might not have labels, use placeholder
                has_glaucoma = -1
                
            # Get fovea coordinates (unique to REFUGE)
            fovea_x = img_info.get('Fovea_X', None)
            fovea_y = img_info.get('Fovea_Y', None)
            
            # Add to dataset
            new_row = {
                'file_id': img_id,
                'dataset': 'REFUGE',
                'split': split,
                'image_path': image_path if os.path.exists(image_path) else None,
                'image_cropped_path': image_cropped_path if os.path.exists(image_cropped_path) else None,
                'image_square_path': image_square_path if os.path.exists(image_square_path) else None,
                'mask_path': mask_path if os.path.exists(mask_path) else None,
                'mask_cropped_path': mask_cropped_path if os.path.exists(mask_cropped_path) else None,
                'mask_square_path': mask_square_path if os.path.exists(mask_square_path) else None,
                'nerve_removed_path': None,  # REFUGE doesn't have nerve removed images
                'label': has_glaucoma,
                'has_glaucoma': 1 if has_glaucoma == 1 else 0 if has_glaucoma == 0 else -1,
                'eye_laterality': None,  # REFUGE doesn't specify eye laterality
                'cdr': None,  # REFUGE doesn't provide CDR directly
                'ecc_cup': None,
                'ecc_disc': None,
                'fovea_x': fovea_x,
                'fovea_y': fovea_y,
                'image_width': width,
                'image_height': height
            }
            
            refuge_data = pd.concat([refuge_data, pd.DataFrame([new_row])], ignore_index=True)
    
    logger.info(f"Processed {len(refuge_data)} REFUGE images")
    return refuge_data

def load_g1020(dataset_path):
    """
    Load and process G1020 dataset.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the G1020 dataset directory
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing G1020 dataset information
    """
    logger.info("Processing G1020 dataset...")
    
    # Initialize column names
    columns = [
        'file_id', 'dataset', 'split', 'image_path', 'image_cropped_path', 'image_square_path',
        'mask_path', 'mask_cropped_path', 'mask_square_path', 'nerve_removed_path',
        'label', 'has_glaucoma', 'eye_laterality', 'cdr', 'ecc_cup', 'ecc_disc',
        'fovea_x', 'fovea_y', 'image_width', 'image_height'
    ]
    
    # Read metadata file
    g1020_csv_path = os.path.join(dataset_path, 'G1020.csv')
    
    if not os.path.exists(g1020_csv_path):
        logger.error(f"G1020 metadata file not found: {g1020_csv_path}")
        return pd.DataFrame(columns=columns)
        
    g1020_meta = pd.read_csv(g1020_csv_path)
    
    # Initialize DataFrame
    g1020_data = pd.DataFrame(columns=columns)
    
    # Process each image in the dataset
    for _, row in tqdm(g1020_meta.iterrows(), desc="Processing G1020 images", total=len(g1020_meta)):
        img_filename = row['imageID']
        img_id = os.path.splitext(img_filename)[0]
        
        # Construct paths for all versions
        image_path = os.path.join(dataset_path, 'Images', img_filename)
        
        # Check for different possible locations of cropped images
        image_cropped_path = os.path.join(dataset_path, 'Images_Cropped', 'img', img_filename)
        if not os.path.exists(image_cropped_path):
            image_cropped_path = os.path.join(dataset_path, 'Images_Cropped', img_filename)
            if not os.path.exists(image_cropped_path):
                image_cropped_path = None
                
        image_square_path = os.path.join(dataset_path, 'Images_Square', img_filename)
        
        # Unique to G1020: nerve removed images
        nerve_removed_path = os.path.join(dataset_path, 'NerveRemoved_Images', img_filename)
        
        # Mask paths
        mask_filename = f"{img_id}.png"
        mask_path = os.path.join(dataset_path, 'Masks', mask_filename)
        
        # Check for different possible locations of cropped masks
        mask_cropped_path = os.path.join(dataset_path, 'Masks_Cropped', 'img', mask_filename)
        if not os.path.exists(mask_cropped_path):
            mask_cropped_path = os.path.join(dataset_path, 'Masks_Cropped', mask_filename)
            if not os.path.exists(mask_cropped_path):
                mask_cropped_path = None
                
        mask_square_path = os.path.join(dataset_path, 'Masks_Square', mask_filename)
        
        # Get image dimensions
        width, height = None, None
        try:
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    width, height = img.size
        except Exception as e:
            logger.warning(f"Error reading image dimensions for {image_path}: {e}")
            
        # Get label info
        has_glaucoma = int(row['binaryLabels'])
        
        # Add to dataset
        new_row = {
            'file_id': img_id,
            'dataset': 'G1020',
            'split': 'unspecified',  # G1020 doesn't specify train/test split
            'image_path': image_path if os.path.exists(image_path) else None,
            'image_cropped_path': image_cropped_path,
            'image_square_path': image_square_path if os.path.exists(image_square_path) else None,
            'mask_path': mask_path if os.path.exists(mask_path) else None,
            'mask_cropped_path': mask_cropped_path,
            'mask_square_path': mask_square_path if os.path.exists(mask_square_path) else None,
            'nerve_removed_path': nerve_removed_path if os.path.exists(nerve_removed_path) else None,
            'label': has_glaucoma,
            'has_glaucoma': has_glaucoma,
            'eye_laterality': None,  # G1020 doesn't specify eye laterality
            'cdr': None,  # G1020 doesn't provide CDR directly
            'ecc_cup': None,
            'ecc_disc': None,
            'fovea_x': None,  # G1020 doesn't have fovea coordinates
            'fovea_y': None,
            'image_width': width,
            'image_height': height
        }
        
        g1020_data = pd.concat([g1020_data, pd.DataFrame([new_row])], ignore_index=True)
    
    logger.info(f"Processed {len(g1020_data)} G1020 images")
    return g1020_data

def consolidate_datasets(base_path):
    """
    Consolidate all datasets from the given base path.
    
    Parameters:
    -----------
    base_path : str
        Base directory containing all datasets
        
    Returns:
    --------
    pandas.DataFrame
        Consolidated DataFrame containing all datasets
    """
    logger.info("Starting dataset consolidation...")
    
    # Initialize DataFrame
    columns = [
        'file_id', 'dataset', 'split', 'image_path', 'image_cropped_path', 'image_square_path',
        'mask_path', 'mask_cropped_path', 'mask_square_path', 'nerve_removed_path',
        'label', 'has_glaucoma', 'eye_laterality', 'cdr', 'ecc_cup', 'ecc_disc',
        'fovea_x', 'fovea_y', 'image_width', 'image_height'
    ]
    
    all_data = pd.DataFrame(columns=columns)
    
    # Process ORIGA dataset
    origa_path = os.path.join(base_path, 'ORIGA')
    if os.path.exists(origa_path):
        origa_data = load_origa(origa_path)
        all_data = pd.concat([all_data, origa_data], ignore_index=True)
        logger.info(f"Added {len(origa_data)} ORIGA images")
    else:
        logger.warning(f"ORIGA dataset not found at {origa_path}")
    
    # Process REFUGE dataset
    refuge_path = os.path.join(base_path, 'REFUGE')
    if os.path.exists(refuge_path):
        refuge_data = load_refuge(refuge_path)
        all_data = pd.concat([all_data, refuge_data], ignore_index=True)
        logger.info(f"Added {len(refuge_data)} REFUGE images")
    else:
        logger.warning(f"REFUGE dataset not found at {refuge_path}")
    
    # Process G1020 dataset
    g1020_path = os.path.join(base_path, 'G1020')
    if os.path.exists(g1020_path):
        g1020_data = load_g1020(g1020_path)
        all_data = pd.concat([all_data, g1020_data], ignore_index=True)
        logger.info(f"Added {len(g1020_data)} G1020 images")
    else:
        logger.warning(f"G1020 dataset not found at {g1020_path}")
    
    logger.info(f"Consolidation complete. Total images: {len(all_data)}")
    return all_data

def save_consolidated_dataset(df, output_file):
    """
    Save consolidated dataset to CSV.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Consolidated dataset
    output_file : str
        Path to save the CSV file
        
    Returns:
    --------
    str
        Path to the saved CSV file
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Consolidated dataset saved to {output_file}")
    
    # Log dataset statistics
    logger.info(f"Dataset Statistics:")
    logger.info(f"Total images: {len(df)}")
    logger.info(f"Total Glaucoma cases: {df['has_glaucoma'].sum()}")
    logger.info(f"Total Normal cases: {len(df) - df['has_glaucoma'].sum()}")
    logger.info(f"Distribution by dataset:\n{df['dataset'].value_counts()}")
    logger.info(f"Distribution by split:\n{df['split'].value_counts()}")
    
    return output_file

def main(base_path=None, output_file=None):
    """
    Main function to run the data loading and consolidation process.
    
    Parameters:
    -----------
    base_path : str, optional
        Base directory containing all datasets. If None, uses '/content'.
    output_file : str, optional
        Path to save the consolidated CSV file. If None, uses 'consolidated_glaucoma_dataset.csv'
        in the base_path.
        
    Returns:
    --------
    pandas.DataFrame
        Consolidated DataFrame containing all datasets
    """
    # Set default paths if not provided
    if base_path is None:
        base_path = '/content'
    
    if output_file is None:
        output_file = os.path.join(base_path, 'consolidated_glaucoma_dataset.csv')
    
    # Consolidate datasets
    all_data = consolidate_datasets(base_path)
    
    # Save consolidated dataset
    save_consolidated_dataset(all_data, output_file)
    
    return all_data

if __name__ == "__main__":
    main()