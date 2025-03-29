"""
Path Utilities Module

Centralized path management for consistent file handling.
"""

import os
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import logging
import shutil

from glaucoma_detection.logger import get_logger, ERROR_CODES, log_exception_handler

logger = get_logger(__name__)

class PathManager:
    """Manages paths throughout the pipeline for consistency."""
    
    def __init__(self, base_dir: Union[str, Path]):
        """Initialize with base directory.
        
        Args:
            base_dir: Base directory for the project
        """
        self.base_dir = Path(base_dir).resolve()
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "output"
        self.model_dir = self.output_dir / "models"
        self.log_dir = self.output_dir / "logs"
        
        # Update dataset paths to look in the package directory
        package_dir = self.base_dir / "src" / "glaucoma_detection" / "data"
        
        # Check if datasets exist in package directory
        if (package_dir / "ORIGA").exists() or (package_dir / "REFUGE").exists() or (package_dir / "G1020").exists():
            self.dataset_paths = {
                "ORIGA": package_dir / "ORIGA",
                "REFUGE": package_dir / "REFUGE",
                "G1020": package_dir / "G1020"
            }
        else:
            # Default to data directory
            self.dataset_paths = {
                "ORIGA": self.data_dir / "ORIGA",
                "REFUGE": self.data_dir / "REFUGE",
                "G1020": self.data_dir / "G1020"
            }
        
        # Ensure critical directories exist
        self._ensure_dirs()
    
    def _ensure_dirs(self) -> None:
        """Ensure that critical directories exist."""
        for dir_path in [self.data_dir, self.output_dir, self.model_dir, self.log_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
    
    def resolve(self, path: Union[str, Path]) -> Path:
        """Resolve a path relative to the base directory.
        
        Args:
            path: Path to resolve
        
        Returns:
            Resolved absolute path
        """
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return (self.base_dir / path_obj).resolve()
    
    def get_dataset_path(self, dataset_name: str) -> Path:
        """Get path for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset (ORIGA, REFUGE, G1020)
        
        Returns:
            Path to the dataset directory
        
        Raises:
            ValueError: If dataset name is not recognized
        """
        dataset_name = dataset_name.upper()
        if dataset_name not in self.dataset_paths:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return self.dataset_paths[dataset_name]
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path for a model file.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Path to the model file
        """
        return self.model_dir / f"{model_name}.pt"
    
    def get_log_path(self, log_name: str) -> Path:
        """Get path for a log file.
        
        Args:
            log_name: Name of the log
        
        Returns:
            Path to the log file
        """
        return self.log_dir / f"{log_name}.log"
    
    def ensure_file_exists(self, file_path: Union[str, Path]) -> Path:
        """Ensure that a file exists.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Absolute path to the file
        
        Raises:
            FileNotFoundError: If file does not exist
        """
        path = self.resolve(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return path
    
    def ensure_dir_exists(self, dir_path: Union[str, Path]) -> Path:
        """Ensure that a directory exists, creating it if necessary.
        
        Args:
            dir_path: Path to the directory
        
        Returns:
            Absolute path to the directory
        """
        path = self.resolve(dir_path)
        path.mkdir(exist_ok=True, parents=True)
        return path
    
    def find_files(self, directory: Union[str, Path], pattern: str = "*") -> List[Path]:
        """Find files matching a pattern in a directory.
        
        Args:
            directory: Directory to search in
            pattern: Glob pattern to match
        
        Returns:
            List of matching file paths
        """
        dir_path = self.resolve(directory)
        return list(dir_path.glob(pattern))
    
    def get_dataset_info(self) -> Dict[str, Dict[str, Union[bool, int]]]:
        """Get information about available datasets.
        
        Returns:
            Dictionary with dataset information
        """
        info = {}
        for name, path in self.dataset_paths.items():
            dataset_exists = path.exists()
            num_files = 0
            if dataset_exists:
                num_files = len(list(path.glob("**/*")))
            
            info[name] = {
                "exists": dataset_exists,
                "path": str(path),
                "num_files": num_files
            }
        return info

@log_exception_handler
def verify_image_mask_pairs(image_paths: List[str], mask_paths: List[str]) -> Tuple[List[str], List[str]]:
    """Verify that image and mask paths are valid pairs.
    
    Args:
        image_paths: List of image file paths
        mask_paths: List of mask file paths
    
    Returns:
        Tuple of (valid_image_paths, valid_mask_paths)
    """
    valid_images = []
    valid_masks = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        img_path_obj = Path(img_path)
        mask_path_obj = Path(mask_path)
        
        if not img_path_obj.exists():
            logger.warning(f"Image file does not exist: {img_path}")
            continue
            
        if not mask_path_obj.exists():
            logger.warning(f"Mask file does not exist: {mask_path}")
            continue
        
        valid_images.append(str(img_path_obj))
        valid_masks.append(str(mask_path_obj))
    
    logger.info(f"Found {len(valid_images)} valid image-mask pairs out of {len(image_paths)}")
    return valid_images, valid_masks

@log_exception_handler
def create_backup(file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Path:
    """Create a backup of a file.
    
    Args:
        file_path: Path to the file to backup
        backup_dir: Directory to store backup (defaults to same directory)
    
    Returns:
        Path to the backup file
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if backup_dir:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(exist_ok=True, parents=True)
    else:
        backup_dir = file_path.parent
    
    # Create backup filename with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
    
    # Create the backup
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup: {backup_path}")
    
    return backup_path

def get_path_manager(base_dir: Optional[Union[str, Path]] = None) -> PathManager:
    """Get a PathManager instance.
    
    Args:
        base_dir: Base directory (defaults to current working directory)
    
    Returns:
        PathManager instance
    """
    if base_dir is None:
        base_dir = Path.cwd()
    return PathManager(base_dir)