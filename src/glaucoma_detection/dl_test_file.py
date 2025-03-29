"""
Focused data loading test script to diagnose and fix data loading issues.
"""

import os
import sys
from pathlib import Path
import logging
import pandas as pd

# Configure detailed logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the src directory to Python path if needed
current_file = Path(__file__).resolve()
project_root = current_file.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

def inspect_directory(directory_path):
    """Inspect a directory and log its contents."""
    path = Path(directory_path)
    if not path.exists():
        logger.error(f"Directory does not exist: {path}")
        return
    
    logger.info(f"Inspecting directory: {path}")
    
    # List subdirectories
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    logger.info(f"Subdirectories: {[d.name for d in subdirs]}")
    
    # List files directly in this directory
    files = [f for f in path.iterdir() if f.is_file()]
    logger.info(f"Files: {[f.name for f in files]}")
    
    # Check for CSV files
    csv_files = [f for f in files if f.suffix.lower() == '.csv']
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"CSV file: {csv_file.name}, Columns: {list(df.columns)}, Rows: {len(df)}")
            # Print first few rows to see structure
            logger.info(f"First row: {df.iloc[0].to_dict() if not df.empty else 'Empty DataFrame'}")
        except Exception as e:
            logger.error(f"Error reading CSV file {csv_file.name}: {e}")
    
    # Recursively check important subdirectories
    for dataset_dir in ['ORIGA', 'REFUGE', 'G1020']:
        if (path / dataset_dir).exists():
            inspect_dataset_directory(path / dataset_dir, dataset_dir)

def inspect_dataset_directory(dataset_path, dataset_name):
    """Inspect a specific dataset directory."""
    logger.info(f"Inspecting {dataset_name} dataset at {dataset_path}")
    
    # Check for expected subdirectories
    for subdir in ['Images', 'Masks']:
        subdir_path = dataset_path / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob('*'))
            logger.info(f"  {subdir} directory exists with {len(files)} files")
            # Log a few example files
            if files:
                logger.info(f"  Example files: {[f.name for f in files[:5]]}")
        else:
            logger.error(f"  {subdir} directory not found at {subdir_path}")
    
    # Check for dataset-specific metadata files
    if dataset_name == 'ORIGA':
        metadata_file = dataset_path / 'OrigaList.csv'
    elif dataset_name == 'G1020':
        metadata_file = dataset_path / 'G1020.csv'
    elif dataset_name == 'REFUGE':
        subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        for split_dir in subdirs:
            metadata_file = split_dir / 'index.json'
            if metadata_file.exists():
                logger.info(f"  Found metadata file for split {split_dir.name}: {metadata_file}")
            else:
                logger.warning(f"  Metadata file not found for split {split_dir.name}: {metadata_file}")
        return
    
    # Check the main metadata file
    if metadata_file.exists():
        logger.info(f"  Metadata file exists: {metadata_file}")
        if metadata_file.suffix.lower() == '.csv':
            try:
                df = pd.read_csv(metadata_file)
                logger.info(f"  CSV columns: {list(df.columns)}")
                logger.info(f"  CSV rows: {len(df)}")
                # Print first few rows to see structure
                if not df.empty:
                    logger.info(f"  First row: {df.iloc[0].to_dict()}")
            except Exception as e:
                logger.error(f"  Error reading CSV file {metadata_file}: {e}")
    else:
        logger.error(f"  Metadata file not found: {metadata_file}")

def test_data_loading():
    """Test the data loading functionality."""
    # Data path
    data_dir = project_root / "data"
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Data directory path: {data_dir}")
    
    # Inspect data directory
    inspect_directory(data_dir)
    
    # Try fixing the data loading issue
    try:
        # Import the data loader
        from glaucoma_detection.data_loader import load_origa, load_refuge, load_g1020
        
        # Create a patched version of load_origa that's more flexible
        def patched_load_origa(dataset_path):
            """Patched version of load_origa that's more flexible with column names."""
            dataset_path = Path(dataset_path)
            metadata_file = dataset_path / "OrigaList.csv"
            images_dir = dataset_path / "Images"
            masks_dir = dataset_path / "Masks"
            
            if not metadata_file.exists():
                logger.error(f"ORIGA metadata file not found at {metadata_file}")
                return pd.DataFrame()
            
            try:
                df = pd.read_csv(metadata_file)
                logger.info(f"Loaded ORIGA metadata with {len(df)} entries")
                
                # Add the 'dataset' column
                df['dataset'] = 'ORIGA'
                
                # Check for filename column or alternative
                if 'filename' not in df.columns:
                    # Look for alternatives
                    filename_alternatives = ['ID', 'id', 'Name', 'name', 'image_id', 'image']
                    found = False
                    for alt in filename_alternatives:
                        if alt in df.columns:
                            logger.info(f"Using '{alt}' column instead of 'filename'")
                            df['filename'] = df[alt]
                            found = True
                            break
                    
                    if not found:
                        # If we have an "OrigaList.csv" without a filename column, we have to create it
                        logger.warning("No suitable filename column found. Creating sequential filenames.")
                        df['filename'] = [f"image_{i:04d}" for i in range(len(df))]
                
                # Check if image files exist based on the directory structure
                sample_extensions = ['.jpg', '.png', '.tif', '.jpeg']
                
                # Try to figure out the correct file extension by checking a sample
                if len(df) > 0:
                    first_filename = df['filename'].iloc[0]
                    for ext in sample_extensions:
                        test_path = images_dir / f"{first_filename}{ext}"
                        if test_path.exists():
                            logger.info(f"Found image file with extension {ext}")
                            # Add image paths with the correct extension
                            df['image_path'] = df['filename'].apply(
                                lambda x: str(images_dir / f"{x}{ext}"))
                            df['mask_path'] = df['filename'].apply(
                                lambda x: str(masks_dir / f"{x}_mask.png"))
                            break
                    else:
                        logger.warning(f"Could not determine file extension for images")
                        return pd.DataFrame()
                
                # Check which paths actually exist
                df['image_exists'] = df['image_path'].apply(os.path.exists)
                df['mask_exists'] = df['mask_path'].apply(os.path.exists)
                
                # Log missing files
                missing_images = (~df['image_exists']).sum()
                missing_masks = (~df['mask_exists']).sum()
                
                if missing_images > 0:
                    logger.warning(f"Found {missing_images} missing images in ORIGA dataset")
                
                if missing_masks > 0:
                    logger.warning(f"Found {missing_masks} missing masks in ORIGA dataset")
                
                # Keep only rows where both image and mask exist
                valid_df = df[df['image_exists'] & df['mask_exists']].copy()
                
                # Add glaucoma label if available
                if 'glaucoma' in df.columns:
                    valid_df['label'] = valid_df['glaucoma'].astype(int)
                elif 'Glaucoma' in df.columns:
                    valid_df['label'] = valid_df['Glaucoma'].astype(int)
                
                # Drop temporary columns
                valid_df = valid_df.drop(columns=['image_exists', 'mask_exists'])
                
                logger.info(f"ORIGA dataset loaded with {len(valid_df)} valid samples")
                return valid_df
                
            except Exception as e:
                logger.error(f"Error loading ORIGA metadata: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return pd.DataFrame()
        
        # Test the patched loader on the ORIGA dataset
        origa_path = data_dir / "ORIGA"
        if origa_path.exists():
            logger.info("Testing patched ORIGA data loader...")
            origa_df = patched_load_origa(origa_path)
            logger.info(f"Loaded {len(origa_df)} valid ORIGA samples")
            
            if not origa_df.empty:
                sample_row = origa_df.iloc[0]
                logger.info(f"Sample row: {sample_row.to_dict()}")
        
        # Try the G1020 dataset
        g1020_path = data_dir / "G1020"
        if g1020_path.exists():
            logger.info("Testing G1020 data loader...")
            g1020_df = load_g1020(str(g1020_path))
            logger.info(f"Loaded {len(g1020_df)} G1020 samples")
        
        # Try the REFUGE dataset
        refuge_path = data_dir / "REFUGE"
        if refuge_path.exists():
            logger.info("Testing REFUGE data loader...")
            refuge_df = load_refuge(str(refuge_path))
            logger.info(f"Loaded {len(refuge_df)} REFUGE samples")
    
    except Exception as e:
        logger.error(f"Error in data loading testing: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("Data loading test completed!")

if __name__ == "__main__":
    test_data_loading()