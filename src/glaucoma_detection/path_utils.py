"""
Simplified Path Management Module

Handles paths for the glaucoma detection pipeline.
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PathManager:
    """Simplified path manager that only looks in the package data directory."""
    
    def __init__(self, base_dir=None):
        """Initialize with optional base directory."""
        # If base_dir is not provided, use the current directory
        if base_dir is None:
            self.base_dir = Path.cwd()
        else:
            self.base_dir = Path(base_dir)
            
        # Use the package data directory
        self.data_dir = self.base_dir / "src" / "glaucoma_detection" / "data"
        self.output_dir = self.base_dir / "output"
        self.model_dir = self.output_dir / "models"
        self.log_dir = self.output_dir / "logs"
        
        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Log the data directory location
        if self.data_dir.exists():
            logger.info(f"Using data directory: {self.data_dir}")
        else:
            logger.warning(f"Data directory not found: {self.data_dir}")
    
    def get_dataset_path(self, dataset_name):
        """Get path for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset (ORIGA, REFUGE, G1020)
        
        Returns:
            Path to the dataset directory
        """
        dataset_name = dataset_name.upper()
        dataset_path = self.data_dir / dataset_name
        
        if not dataset_path.exists():
            logger.warning(f"Dataset directory not found: {dataset_path}")
        
        return dataset_path

def get_path_manager(base_dir=None):
    """Get a path manager instance."""
    return PathManager(base_dir)