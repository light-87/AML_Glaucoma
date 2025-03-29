"""
Testing Utilities

Comprehensive testing framework for the glaucoma detection pipeline.
"""

import os
import sys
import tempfile
import unittest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import shutil
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import json
import pytest

from glaucoma_detection.logger import get_logger, ERROR_CODES
from glaucoma_detection.path_utils import PathManager, get_path_manager
from glaucoma_detection.data_validator import DataValidator

logger = get_logger(__name__)

class TestPipelineBase(unittest.TestCase):
    """Base class for pipeline tests."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.temp_data_dir = os.path.join(self.temp_dir, "data")
        self.temp_output_dir = os.path.join(self.temp_dir, "output")
        
        # Create directory structure
        os.makedirs(self.temp_data_dir, exist_ok=True)
        os.makedirs(self.temp_output_dir, exist_ok=True)
        
        # Set up path manager
        self.path_manager = PathManager(self.temp_dir)
        
        # Log test setup
        logger.info(f"Test environment set up in {self.temp_dir}")
    
    def tearDown(self):
        """Clean up after test."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        logger.info(f"Test environment cleaned up: {self.temp_dir}")
    
    def create_test_image(self, path: Union[str, Path], width: int = 224, height: int = 224) -> Path:
        """Create a test image.
        
        Args:
            path: Path to save the image
            width: Image width
            height: Image height
            
        Returns:
            Path to the created image
        """
        import cv2
        
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        
        # Create a random image
        img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        cv2.imwrite(str(path), img)
        
        return path
    
    def create_test_mask(self, path: Union[str, Path], width: int = 224, height: int = 224) -> Path:
        """Create a test mask.
        
        Args:
            path: Path to save the mask
            width: Mask width
            height: Mask height
            
        Returns:
            Path to the created mask
        """
        import cv2
        
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        
        # Create a random binary mask
        mask = np.zeros((height, width), dtype=np.uint8)
        # Add a circle in the center
        cv2.circle(mask, (width // 2, height // 2), min(width, height) // 4, 255, -1)
        cv2.imwrite(str(path), mask)
        
        return path
    
    def create_test_dataset(self, num_samples: int = 10) -> pd.DataFrame:
        """Create a test dataset.
        
        Args:
            num_samples: Number of samples to create
            
        Returns:
            DataFrame with test dataset
        """
        # Create directories for test dataset
        os.makedirs(os.path.join(self.temp_data_dir, "ORIGA", "Images"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_data_dir, "ORIGA", "Masks"), exist_ok=True)
        
        data = []
        for i in range(num_samples):
            # Define paths
            image_path = os.path.join(self.temp_data_dir, "ORIGA", "Images", f"img_{i}.jpg")
            mask_path = os.path.join(self.temp_data_dir, "ORIGA", "Masks", f"img_{i}.png")
            
            # Create image and mask
            self.create_test_image(image_path)
            self.create_test_mask(mask_path)
            
            # Add to dataset
            data.append({
                "filename": f"img_{i}",
                "image_path": image_path,
                "mask_path": mask_path,
                "label": i % 2,  # Alternate 0 and 1 for balanced dataset
                "dataset": "ORIGA",
                "split": "train" if i < 7 else "val" if i < 9 else "test"
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        metadata_path = os.path.join(self.temp_data_dir, "ORIGA", "OrigaList.csv")
        df.to_csv(metadata_path, index=False)
        
        logger.info(f"Created test dataset with {num_samples} samples")
        
        return df
    
    def create_test_model_config(self) -> Dict[str, Any]:
        """Create a test model configuration.
        
        Returns:
            Model configuration dictionary
        """
        return {
            "architecture": "unet",
            "encoder": "resnet18",  # Smaller model for testing
            "pretrained": False,    # No need for pretrained weights in tests
            "in_channels": 3,
            "num_classes": 1
        }

def create_test_environment(base_dir: Optional[Union[str, Path]] = None) -> Tuple[Path, PathManager]:
    """Create a test environment for the pipeline.
    
    Args:
        base_dir: Base directory for the test environment
        
    Returns:
        Tuple of (temp_dir, path_manager)
    """
    if base_dir is None:
        temp_dir = Path(tempfile.mkdtemp())
    else:
        temp_dir = Path(base_dir)
        temp_dir.mkdir(exist_ok=True, parents=True)
    
    # Create directory structure
    data_dir = temp_dir / "data"
    output_dir = temp_dir / "output"
    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    
    for d in [data_dir, output_dir, models_dir, logs_dir]:
        d.mkdir(exist_ok=True, parents=True)
    
    # Create path manager
    path_manager = PathManager(temp_dir)
    
    logger.info(f"Created test environment in {temp_dir}")
    
    return temp_dir, path_manager

def generate_test_dataset(path_manager: PathManager, 
                        dataset_name: str = "ORIGA", 
                        num_samples: int = 10,
                        image_size: Tuple[int, int] = (224, 224)) -> pd.DataFrame:
    """Generate a test dataset.
    
    Args:
        path_manager: Path manager
        dataset_name: Name of the dataset
        num_samples: Number of samples
        image_size: Image size
        
    Returns:
        DataFrame with the test dataset
    """
    import cv2
    
    # Determine paths
    dataset_path = path_manager.get_dataset_path(dataset_name)
    images_dir = dataset_path / "Images"
    masks_dir = dataset_path / "Masks"
    
    # Create directories
    images_dir.mkdir(exist_ok=True, parents=True)
    masks_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate data
    data = []
    for i in range(num_samples):
        # Define paths
        image_path = images_dir / f"img_{i}.jpg"
        mask_path = masks_dir / f"img_{i}.png"
        
        # Create dummy image (random colored image)
        img = np.random.randint(0, 256, (image_size[1], image_size[0], 3), dtype=np.uint8)
        cv2.imwrite(str(image_path), img)
        
        # Create dummy mask (circle in the center)
        mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        center = (image_size[0] // 2, image_size[1] // 2)
        radius = min(image_size) // 4
        cv2.circle(mask, center, radius, 255, -1)
        cv2.imwrite(str(mask_path), mask)
        
        # Add to dataset
        data.append({
            "filename": f"img_{i}",
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "label": i % 2,  # Alternate between 0 and 1
            "dataset": dataset_name,
            "split": "train" if i < 7 else "val" if i < 9 else "test"
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    metadata_file = "OrigaList.csv" if dataset_name == "ORIGA" else f"{dataset_name}.csv"
    metadata_path = dataset_path / metadata_file
    df.to_csv(metadata_path, index=False)
    
    logger.info(f"Generated test dataset '{dataset_name}' with {num_samples} samples")
    
    return df

def create_test_config(path_manager: PathManager) -> Dict[str, Any]:
    """Create a test configuration.
    
    Args:
        path_manager: Path manager
        
    Returns:
        Test configuration dictionary
    """
    return {
        "paths": {
            "base_dir": str(path_manager.base_dir),
            "data_dir": str(path_manager.data_dir),
            "output_dir": str(path_manager.output_dir),
            "model_dir": str(path_manager.model_dir),
            "log_dir": str(path_manager.log_dir)
        },
        "data": {
            "random_state": 42,
            "split_config": {
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15
            }
        },
        "model": {
            "architecture": "unet",
            "encoder": "resnet18",  # Smaller model for testing
            "pretrained": False,
            "in_channels": 3,
            "num_classes": 1
        },
        "preprocessing": {
            "image_size": [224, 224],
            "image_channels": 3,
            "mode": "segmentation",
            "augmentation": {
                "enabled": True
            }
        },
        "training": {
            "epochs": 1,  # Just 1 epoch for testing
            "batch_size": 2,
            "num_workers": 1,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss_function": "combined",
            "use_gpu": torch.cuda.is_available(),
            "early_stopping": {
                "enabled": False  # Disable for testing
            },
            "checkpointing": {
                "enabled": True,
                "save_top_k": 1
            }
        },
        "evaluation": {
            "metrics": ["dice", "iou", "accuracy"],
            "threshold": 0.5,
            "visualization": {
                "enabled": True
            }
        },
        "logging": {
            "use_wandb": False
        },
        "pipeline": {
            "steps": ["extract", "load", "clean", "preprocess", "train", "evaluate"],
            "force": True,
            "description": "Test pipeline run"
        }
    }

def cleanup_test_environment(temp_dir: Union[str, Path]):
    """Clean up the test environment.
    
    Args:
        temp_dir: Directory to clean up
    """
    shutil.rmtree(str(temp_dir))
    logger.info(f"Cleaned up test environment: {temp_dir}")