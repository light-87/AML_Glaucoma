"""
Fixed Data Loading Module specifically for your dataset structures.
"""

import os
import pandas as pd
from pathlib import Path
import zipfile
import json
from typing import Union, Dict, List, Optional
import logging
from tqdm import tqdm
from glaucoma_detection.path_utils import PathManager

logger = logging.getLogger(__name__)

def load_origa(dataset_path: str) -> pd.DataFrame:
    """Load the ORIGA dataset with specific handling for your file structure."""
    dataset_path = Path(dataset_path)
    metadata_file = dataset_path / "OrigaList.csv"
    images_dir = dataset_path / "Images"
    masks_dir = dataset_path / "Masks"
    
    # Check if necessary files exist
    if not metadata_file.exists():
        logger.error(f"ORIGA metadata file not found at {metadata_file}")
        return pd.DataFrame()
    
    if not images_dir.exists():
        logger.error(f"ORIGA images directory not found at {images_dir}")
        return pd.DataFrame()
    
    # Load metadata
    try:
        df = pd.read_csv(metadata_file)
        logger.info(f"Loaded ORIGA metadata with {len(df)} entries")
        
        # Add dataset source
        df['dataset'] = 'ORIGA'
        
        # Use the 'Filename' column (note: capital F)
        if 'Filename' in df.columns:
            logger.info("Using 'Filename' column for image paths")
            
            # Extract the filename without extension if it has one
            df['filename'] = df['Filename'].apply(
                lambda x: x.split('.')[0] if '.' in x else x
            )
            
            # Add paths to images and masks
            # Note: From the test output, images end with .jpg and masks end with .png
            df['image_path'] = df['filename'].apply(lambda x: str(images_dir / f"{x}.jpg"))
            df['mask_path'] = df['filename'].apply(lambda x: str(masks_dir / f"{x}.png"))
        else:
            logger.error("'Filename' column not found in ORIGA metadata")
            return pd.DataFrame()
        
        # Map glaucoma labels
        if 'Glaucoma' in df.columns:
            df['label'] = df['Glaucoma'].astype(int)
        
        # Validate that files exist
        df['image_exists'] = df['image_path'].apply(os.path.exists)
        df['mask_exists'] = df['mask_path'].apply(os.path.exists)
        
        missing_images = (~df['image_exists']).sum()
        missing_masks = (~df['mask_exists']).sum()
        
        if missing_images > 0:
            logger.warning(f"Found {missing_images} missing images in ORIGA dataset")
        
        if missing_masks > 0:
            logger.warning(f"Found {missing_masks} missing masks in ORIGA dataset")
        
        # Keep only rows where both image and mask exist
        valid_df = df[df['image_exists'] & df['mask_exists']].copy()
        
        # Drop temporary columns
        valid_df = valid_df.drop(columns=['image_exists', 'mask_exists'])
        
        logger.info(f"ORIGA dataset loaded with {len(valid_df)} valid samples")
        return valid_df
    except Exception as e:
        logger.error(f"Error loading ORIGA metadata: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def load_g1020(dataset_path: str) -> pd.DataFrame:
    """Load the G1020 dataset specifically for your file structure."""
    dataset_path = Path(dataset_path)
    metadata_file = dataset_path / "G1020.csv"
    images_dir = dataset_path / "Images"
    masks_dir = dataset_path / "Masks"
    
    # Check if necessary files exist
    if not metadata_file.exists():
        logger.error(f"G1020 metadata file not found at {metadata_file}")
        return pd.DataFrame()
    
    if not images_dir.exists():
        logger.error(f"G1020 images directory not found at {images_dir}")
        return pd.DataFrame()
    
    # Load metadata
    try:
        df = pd.read_csv(metadata_file)
        logger.info(f"Loaded G1020 metadata with {len(df)} entries")
        
        # Add dataset source
        df['dataset'] = 'G1020'
        
        # Use the 'imageID' column 
        if 'imageID' in df.columns:
            logger.info("Using 'imageID' column for image paths")
            
            # Extract just the base filename without extension
            df['filename'] = df['imageID'].apply(
                lambda x: x.split('.')[0] if '.' in x else x
            )
            
            # Add paths to images and masks - from test, we can see the masks are .png
            df['image_path'] = df['filename'].apply(lambda x: str(images_dir / f"{x}.jpg"))
            df['mask_path'] = df['filename'].apply(lambda x: str(masks_dir / f"{x}.png"))
        else:
            logger.error("'imageID' column not found in G1020 metadata")
            return pd.DataFrame()
        
        # Map glaucoma labels
        if 'binaryLabels' in df.columns:
            df['label'] = df['binaryLabels'].astype(int)
        
        # Validate that files exist
        df['image_exists'] = df['image_path'].apply(os.path.exists)
        df['mask_exists'] = df['mask_path'].apply(os.path.exists)
        
        missing_images = (~df['image_exists']).sum()
        missing_masks = (~df['mask_exists']).sum()
        
        if missing_images > 0:
            logger.warning(f"Found {missing_images} missing images in G1020 dataset")
        
        if missing_masks > 0:
            logger.warning(f"Found {missing_masks} missing masks in G1020 dataset")
        
        # Keep only rows where both image and mask exist
        valid_df = df[df['image_exists'] & df['mask_exists']].copy()
        
        # Drop temporary columns
        valid_df = valid_df.drop(columns=['image_exists', 'mask_exists'])
        
        logger.info(f"G1020 dataset loaded with {len(valid_df)} valid samples")
        return valid_df
    except Exception as e:
        logger.error(f"Error loading G1020 metadata: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

# Keep the REFUGE loader from the previous version since it's working
def load_refuge(dataset_path: str) -> pd.DataFrame:
    """Load the REFUGE dataset."""
    dataset_path = Path(dataset_path)
    splits = ["train", "val", "test"]
    
    all_data = []
    
    for split in splits:
        split_dir = dataset_path / split
        
        if not split_dir.exists():
            logger.warning(f"REFUGE {split} directory not found at {split_dir}")
            continue
        
        images_dir = split_dir / "Images"
        if not images_dir.exists():
            # Try looking for images directly in the split directory
            images_dir = split_dir
            logger.warning(f"No Images subdirectory found. Using {images_dir} instead.")
        
        masks_dir = split_dir / "Masks"
        if not masks_dir.exists():
            logger.warning(f"REFUGE {split} masks directory not found. Masks will be marked as missing.")
        
        metadata_file = split_dir / "index.json"
        
        # Load metadata if available
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded REFUGE {split} metadata with {len(metadata)} entries")
            except Exception as e:
                logger.warning(f"Error loading REFUGE {split} metadata: {e}")
        
        # Get all image files
        image_extensions = ['.jpg', '.png', '.jpeg', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
        
        if not image_files:
            logger.warning(f"No image files found in {images_dir}")
            continue
        
        logger.info(f"Found {len(image_files)} images in REFUGE {split} split")
        
        # Create dataframe for this split
        split_data = []
        
        for img_path in image_files:
            img_name = img_path.stem
            
            # Look for mask with various possible naming patterns
            mask_patterns = [
                masks_dir / f"{img_name}_mask.png",
                masks_dir / f"{img_name}.png",
                masks_dir / f"{img_name}_segmentation.png",
                masks_dir / f"{img_name}_gt.png"
            ]
            
            mask_path = None
            for pattern in mask_patterns:
                if pattern.exists():
                    mask_path = str(pattern)
                    break
            
            # Get label from metadata if available
            label = -1  # Default unknown
            if img_name in metadata and 'glaucoma' in metadata[img_name]:
                label = int(metadata[img_name]['glaucoma'])
            
            split_data.append({
                'filename': img_name,
                'image_path': str(img_path),
                'mask_path': mask_path,
                'label': label,
                'dataset': 'REFUGE',
                'split': split
            })
        
        all_data.extend(split_data)
        logger.info(f"Added {len(split_data)} samples from REFUGE {split} split")
    
    # Create dataframe
    df = pd.DataFrame(all_data)
    
    # Validate that files exist
    valid_mask_count = df['mask_path'].apply(lambda x: x is not None and os.path.exists(x)).sum()
    
    logger.info(f"REFUGE dataset loaded with {len(df)} samples, {valid_mask_count} with valid masks")
    return df

def save_consolidated_dataset(df: pd.DataFrame, output_path: str) -> bool:
    """Save the consolidated dataset to a CSV file."""
    try:
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved consolidated dataset with {len(df)} samples to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving consolidated dataset: {e}")
        return False

def extract_zip(zip_file: str, output_dir: Optional[str] = None) -> str:
    """Extract a ZIP file to the specified output directory."""
    zip_path = Path(zip_file)
    if output_dir is None:
        output_dir = zip_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        logger.info(f"Extracting {zip_path} to {output_dir}")
        zip_ref.extractall(output_dir)
    
    # Check for top-level directory
    extracted_items = [output_dir / item.split('/')[0] for item in zip_ref.namelist() 
                      if '/' in item]
    unique_dirs = set(item for item in extracted_items if item.is_dir())
    
    if len(unique_dirs) == 1:
        return str(unique_dirs.pop())
    return str(output_dir)

def load_dataset(dataset_type: str, dataset_path: str) -> pd.DataFrame:
    """Load a specific dataset based on type."""
    if dataset_type.upper() == "ORIGA":
        return load_origa(dataset_path)
    elif dataset_type.upper() == "REFUGE":
        return load_refuge(dataset_path)
    elif dataset_type.upper() == "G1020":
        return load_g1020(dataset_path)
    else:
        logger.error(f"Unknown dataset type: {dataset_type}")
        return pd.DataFrame()
    
def consolidate_datasets(base_path: str) -> pd.DataFrame:
    """Consolidate all datasets from the given base path or package path."""
    datasets = ["ORIGA", "REFUGE", "G1020"]
    all_data = pd.DataFrame()
    
    # First, try to get paths from PathManager to handle both locations
    path_manager = PathManager(os.path.dirname(base_path))
    
    for dataset in datasets:
        try:
            # Get dataset path from PathManager (handles both locations)
            dataset_path = path_manager.get_dataset_path(dataset)
            if os.path.exists(dataset_path):
                df = load_dataset(dataset, str(dataset_path))
                if not df.empty:
                    all_data = pd.concat([all_data, df], ignore_index=True)
                    logger.info(f"Added {len(df)} {dataset} images")
            else:
                logger.warning(f"{dataset} dataset not found at {dataset_path}")
        except Exception as e:
            logger.warning(f"Error loading {dataset} dataset: {str(e)}")
    
    # Check if we found any data
    if all_data.empty:
        logger.error("No data loaded from any dataset")
    else:
        logger.info(f"Consolidation complete. Total images: {len(all_data)}")
        
    return all_data