"""
Data Loading & Extraction Module

Simplified module leveraging pandas and pathlib.
"""

import os
import pandas as pd
from pathlib import Path
import zipfile
from typing import Union, Dict, List, Optional
import logging
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)

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

# Simplified versions of load_origa, load_refuge, load_g1020 functions...

def consolidate_datasets(base_path: str) -> pd.DataFrame:
    """Consolidate all datasets from the given base path."""
    datasets = ["ORIGA", "REFUGE", "G1020"]
    all_data = pd.DataFrame()
    
    for dataset in datasets:
        dataset_path = os.path.join(base_path, dataset)
        if os.path.exists(dataset_path):
            df = load_dataset(dataset, dataset_path)
            if not df.empty:
                all_data = pd.concat([all_data, df], ignore_index=True)
                logger.info(f"Added {len(df)} {dataset} images")
        else:
            logger.warning(f"{dataset} dataset not found at {dataset_path}")
    
    logger.info(f"Consolidation complete. Total images: {len(all_data)}")
    return all_data