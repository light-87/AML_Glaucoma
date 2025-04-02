"""
Simplified Data Loading Module

Loads datasets for the glaucoma detection pipeline.
"""

import os
import pandas as pd
from pathlib import Path
import zipfile
import logging

logger = logging.getLogger(__name__)

def extract_zip(zip_file, output_dir=None):
    """Extract a ZIP file of datasets."""
    zip_path = Path(zip_file)
    if output_dir is None:
        output_dir = zip_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Extracting {zip_path} to {output_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    return str(output_dir)

def load_origa(dataset_path):
    """Load the ORIGA dataset."""
    dataset_path = Path(dataset_path)
    metadata_file = dataset_path / "OrigaList.csv"
    images_dir = dataset_path / "Images"
    masks_dir = dataset_path / "Masks"
    
    if not metadata_file.exists():
        logger.error(f"ORIGA metadata file not found: {metadata_file}")
        return pd.DataFrame()
    
    # Load metadata
    df = pd.read_csv(metadata_file)
    logger.info(f"Loaded ORIGA metadata with {len(df)} entries")
    
    # Add dataset source
    df['dataset'] = 'ORIGA'
    
    # Get filename column and standardize it
    if 'Filename' in df.columns:
        filename_col = 'Filename'
    elif 'filename' in df.columns:
        filename_col = 'filename'
    else:
        logger.error("No filename column found in ORIGA metadata")
        return pd.DataFrame()
    
    # Extract filename without extension
    df['filename'] = df[filename_col].apply(
        lambda x: str(x).split('.')[0] if '.' in str(x) else str(x)
    )
    
    # Add paths to images and masks
    df['image_path'] = df['filename'].apply(lambda x: str(images_dir / f"{x}.jpg"))
    df['mask_path'] = df['filename'].apply(lambda x: str(masks_dir / f"{x}.png"))
    
    # Map glaucoma labels if available
    if 'Glaucoma' in df.columns:
        df['label'] = df['Glaucoma'].astype(int)
    
    # Filter to include only files that exist
    valid_df = df[df['image_path'].apply(os.path.exists) & 
                  df['mask_path'].apply(os.path.exists)].copy()
    
    logger.info(f"ORIGA dataset: {len(valid_df)} valid samples out of {len(df)}")
    return valid_df

def load_refuge(dataset_path):
    """Load the REFUGE dataset with more flexible directory structure handling."""
    dataset_path = Path(dataset_path)
    splits = ["train", "val", "test"]
    
    all_data = []
    
    # Function to find image files recursively
    def find_images(directory):
        if not directory.exists():
            return []
        
        # First check directly in the directory
        image_files = list(directory.glob("*.jpg")) + list(directory.glob("*.png"))
        
        # If no images found directly, check subdirectories recursively (limit to depth 2)
        if not image_files:
            for subdir in directory.iterdir():
                if subdir.is_dir():
                    image_files.extend(list(subdir.glob("*.jpg")) + list(subdir.glob("*.png")))
                    
                    # Check one level deeper if still nothing
                    if not image_files:
                        for subsubdir in subdir.iterdir():
                            if subsubdir.is_dir():
                                image_files.extend(list(subsubdir.glob("*.jpg")) + list(subsubdir.glob("*.png")))
        
        return image_files
    
    # Process each split
    for split in splits:
        split_dir = dataset_path / split
        if not split_dir.exists():
            logger.warning(f"REFUGE {split} directory not found: {split_dir}")
            continue
        
        logger.info(f"Processing REFUGE {split} split")
        
        # Look for images in multiple possible locations
        image_locations = [
            split_dir,                 # Directly in split dir
            split_dir / "Images",      # In Images subdirectory
            split_dir / "images",      # Case variation
            split_dir / "Glaucoma",    # Specialized directory
            split_dir / "Non-Glaucoma" # Specialized directory
        ]
        
        # Find images in all possible locations
        image_files = []
        for location in image_locations:
            image_files.extend(find_images(location))
        
        if not image_files:
            logger.warning(f"No images found in REFUGE {split} split")
            continue
        
        logger.info(f"Found {len(image_files)} images in REFUGE {split} split")
        
        # Look for masks in multiple possible locations
        mask_locations = [
            split_dir / "Masks", 
            split_dir / "masks",
            split_dir / "GT", 
            split_dir / "gt",
            split_dir / "Groundtruth",
            split_dir / "groundtruth",
            split_dir / "Segmentations",
            split_dir / "segmentations"
        ]
        
        # Process each image
        for img_path in image_files:
            img_name = img_path.stem
            
            # Determine if it's glaucoma based on directory or filename
            is_glaucoma = False
            if "glaucoma" in str(img_path).lower():
                if "non" not in str(img_path).lower() and "non-" not in str(img_path).lower():
                    is_glaucoma = True
            elif "_g_" in img_name.lower() or "glaucoma" in img_name.lower():
                is_glaucoma = True
            
            # Look for corresponding mask in all possible locations
            mask_path = None
            mask_variations = [
                f"{img_name}.png", 
                f"{img_name}_mask.png", 
                f"{img_name}_gt.png",
                f"{img_name}_segmentation.png",
                f"{img_name.lower()}.png",
                f"{img_name.upper()}.png"
            ]
            
            for mask_location in mask_locations:
                if mask_location.exists():
                    for variation in mask_variations:
                        candidate = mask_location / variation
                        if candidate.exists():
                            mask_path = str(candidate)
                            break
                    
                    if mask_path:
                        break
            
            # Add to data list, even without mask (for classification)
            all_data.append({
                'filename': img_name,
                'image_path': str(img_path),
                'mask_path': mask_path,
                'label': 1 if is_glaucoma else 0,
                'dataset': 'REFUGE',
                'split': split
            })
    
    # Create dataframe
    df = pd.DataFrame(all_data)
    logger.info(f"REFUGE dataset: {len(df)} samples")
    
    # Log counts with and without masks
    with_masks = df['mask_path'].notna().sum()
    logger.info(f"REFUGE dataset: {with_masks} samples with masks, {len(df) - with_masks} without masks")
    
    return df

def load_g1020(dataset_path):
    """Load the G1020 dataset."""
    dataset_path = Path(dataset_path)
    metadata_file = dataset_path / "G1020.csv"
    images_dir = dataset_path / "Images"
    masks_dir = dataset_path / "Masks"
    
    if not metadata_file.exists():
        logger.error(f"G1020 metadata file not found: {metadata_file}")
        return pd.DataFrame()
    
    # Load metadata
    df = pd.read_csv(metadata_file)
    logger.info(f"Loaded G1020 metadata with {len(df)} entries")
    
    # Add dataset source
    df['dataset'] = 'G1020'
    
    # Process image IDs
    if 'imageID' in df.columns:
        df['filename'] = df['imageID'].apply(
            lambda x: str(x).split('.')[0] if '.' in str(x) else str(x)
        )
    else:
        logger.error("No imageID column found in G1020 metadata")
        return pd.DataFrame()
    
    # Add paths to images and masks
    df['image_path'] = df['filename'].apply(lambda x: str(images_dir / f"{x}.jpg"))
    df['mask_path'] = df['filename'].apply(lambda x: str(masks_dir / f"{x}.png"))
    
    # Map binary labels if available
    if 'binaryLabels' in df.columns:
        df['label'] = df['binaryLabels'].astype(int)
    
    # Filter to include only files that exist
    valid_df = df[df['image_path'].apply(os.path.exists) & 
                  df['mask_path'].apply(os.path.exists)].copy()
    
    logger.info(f"G1020 dataset: {len(valid_df)} valid samples out of {len(df)}")
    return valid_df

def consolidate_datasets(data_dir):
    """Consolidate datasets from the data directory."""
    data_dir = Path(data_dir)
    datasets = ["ORIGA", "REFUGE", "G1020"]
    all_data = pd.DataFrame()
    
    for dataset in datasets:
        dataset_path = data_dir / dataset
        if dataset_path.exists():
            # Load appropriate dataset
            if dataset == "ORIGA":
                df = load_origa(dataset_path)
            elif dataset == "REFUGE":
                df = load_refuge(dataset_path)
            elif dataset == "G1020":
                df = load_g1020(dataset_path)
                
            # Add to consolidated dataframe
            if not df.empty:
                all_data = pd.concat([all_data, df], ignore_index=True)
                logger.info(f"Added {len(df)} samples from {dataset}")
    
    # Log the result
    if all_data.empty:
        logger.warning("No data loaded from any dataset")
    else:
        logger.info(f"Consolidated {len(all_data)} total samples")
        
        # Log dataset distribution
        if 'dataset' in all_data.columns:
            dataset_counts = all_data['dataset'].value_counts().to_dict()
            logger.info(f"Dataset distribution: {dataset_counts}")
    
    return all_data

def save_consolidated_dataset(df, output_path):
    """Save the consolidated dataset to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} samples to {output_path}")
    return True