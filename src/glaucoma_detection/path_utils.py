from pathlib import Path
from typing import Union
from glaucoma_detection.logger import get_logger

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
        
        # Create these directories if they don't exist
        for dir_path in [self.data_dir, self.output_dir, self.model_dir, self.log_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # First, check the package directory for datasets
        package_dir = self.base_dir / "src" / "glaucoma_detection" / "data"
        
        # If package directory exists and contains at least one dataset, use it
        if package_dir.exists() and any((package_dir / dataset).exists() 
                                       for dataset in ["ORIGA", "REFUGE", "G1020"]):
            logger.info(f"Using datasets from package directory: {package_dir}")
            self.dataset_paths = {
                "ORIGA": package_dir / "ORIGA",
                "REFUGE": package_dir / "REFUGE",
                "G1020": package_dir / "G1020"
            }
        else:
            # Default to data directory with warning if datasets are expected
            logger.info(f"Using datasets from main data directory: {self.data_dir}")
            self.dataset_paths = {
                "ORIGA": self.data_dir / "ORIGA",
                "REFUGE": self.data_dir / "REFUGE",
                "G1020": self.data_dir / "G1020"
            }
            
            # Check if any datasets exist and log warning if not
            if not any((self.data_dir / dataset).exists() for dataset in ["ORIGA", "REFUGE", "G1020"]):
                logger.warning(f"No datasets found in {self.data_dir} or {package_dir}")
    
    def get_dataset_path(self, dataset_name: str) -> Path:
        """Get path for a specific dataset with validation.
        
        Args:
            dataset_name: Name of the dataset (ORIGA, REFUGE, G1020)
        
        Returns:
            Path to the dataset directory
        
        Raises:
            ValueError: If dataset name is not recognized or dataset doesn't exist
        """
        dataset_name = dataset_name.upper()
        if dataset_name not in self.dataset_paths:
            raise ValueError(f"Unknown dataset: {dataset_name}. Expected one of: ORIGA, REFUGE, G1020")
        
        dataset_path = self.dataset_paths[dataset_name]
        
        # Verify that dataset exists
        if not dataset_path.exists():
            error_msg = f"Dataset not found: {dataset_name} at {dataset_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        return dataset_path