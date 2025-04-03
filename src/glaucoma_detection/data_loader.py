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
    """Process REFUGE dataset and return a DataFrame"""
    import os
    import json
    import pandas as pd
    from pathlib import Path
    
    # Initialize DataFrame
    columns = [
        'filename', 'image_path', 'mask_path', 'label', 
        'dataset', 'split', 'image_width', 'image_height'
    ]
    refuge_data = []

    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        split_path = Path(dataset_path) / split
        
        # Look for index.json in the split directory
        index_path = split_path / 'index.json'
        
        if not index_path.exists():
            logger.warning(f"No index.json found for {split} split")
            continue
        
        # Read index file
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        # Process each image in the split
        for img_id, img_info in index_data.items():
            img_filename = img_info.get('ImgName')
            
            if not img_filename:
                logger.warning(f"No image filename found for {img_id}")
                continue
            
            # Construct image path
            image_path = split_path / 'Images' / img_filename
            
            # Try to find mask
            mask_candidates = [
                split_path / 'Masks' / f"{os.path.splitext(img_filename)[0]}.png",
                split_path / 'Masks' / img_filename,
                split_path / 'Masks_Square' / f"{os.path.splitext(img_filename)[0]}.png"
            ]
            
            mask_path = None
            for candidate in mask_candidates:
                if candidate.exists():
                    mask_path = str(candidate)
                    break
            
            # Determine label
            has_glaucoma = img_info.get('Label', -1)
            
            # Try to get image dimensions
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                logger.warning(f"Could not read image dimensions for {image_path}: {e}")
                width, height = None, None
            
            # Add to dataset
            refuge_data.append({
                'filename': img_filename,
                'image_path': str(image_path),
                'mask_path': mask_path,
                'label': 1 if has_glaucoma == 1 else 0 if has_glaucoma == 0 else -1,
                'dataset': 'REFUGE',
                'split': split,
                'image_width': width,
                'image_height': height
            })
    
    # Create DataFrame
    df = pd.DataFrame(refuge_data)
    
    # Log dataset information
    logger.info(f"REFUGE dataset: Total samples: {len(df)}")
    logger.info("Split distribution:")
    logger.info(df['split'].value_counts())
    logger.info("Label distribution:")
    logger.info(df['label'].value_counts())
    
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