"""
Data Validation Module

Comprehensive validation for dataset integrity and format.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import cv2
from PIL import Image
import logging

from glaucoma_detection.logger import get_logger, ERROR_CODES, log_exception_handler
from glaucoma_detection.path_utils import PathManager, get_path_manager

logger = get_logger(__name__)

class DataValidator:
    """Validates data structure and content for the glaucoma pipeline."""
    
    def __init__(self, path_manager: Optional[PathManager] = None):
        """Initialize with path manager.
        
        Args:
            path_manager: Path manager instance
        """
        self.path_manager = path_manager or get_path_manager()
    
    def validate_dataset_structure(self, dataset_name: str) -> Dict[str, Any]:
        """Validate the structure of a specific dataset.
        
        Args:
            dataset_name: Name of the dataset to validate
        
        Returns:
            Validation results
        """
        dataset_path = self.path_manager.get_dataset_path(dataset_name)
        results = {
            "dataset": dataset_name,
            "path": str(dataset_path),
            "exists": dataset_path.exists(),
            "valid": False,
            "issues": []
        }
        
        if not dataset_path.exists():
            results["issues"].append(f"Dataset directory does not exist: {dataset_path}")
            return results
        
        # Check for required subdirectories
        required_dirs = ["Images", "Masks"]
        for subdir in required_dirs:
            subdir_path = dataset_path / subdir
            if not subdir_path.exists():
                results["issues"].append(f"Required subdirectory not found: {subdir}")
            elif not subdir_path.is_dir():
                results["issues"].append(f"Required path is not a directory: {subdir}")
        
        # Check for dataset-specific metadata files
        if dataset_name.upper() == "ORIGA":
            metadata_file = dataset_path / "OrigaList.csv"
            if not metadata_file.exists():
                results["issues"].append(f"ORIGA metadata file not found: OrigaList.csv")
        elif dataset_name.upper() == "G1020":
            metadata_file = dataset_path / "G1020.csv"
            if not metadata_file.exists():
                results["issues"].append(f"G1020 metadata file not found: G1020.csv")
        elif dataset_name.upper() == "REFUGE":
            # Check for split directories
            split_dirs = ["train", "val", "test"]
            found_splits = []
            for split in split_dirs:
                split_path = dataset_path / split
                if split_path.exists() and split_path.is_dir():
                    found_splits.append(split)
            
            if not found_splits:
                results["issues"].append(f"No split directories found for REFUGE")
        
        # Set valid flag if no issues were found
        results["valid"] = len(results["issues"]) == 0
        
        return results
    
    def validate_dataframe(self, df: pd.DataFrame, mode: str = "segmentation") -> Dict[str, Any]:
        """Validate a dataframe for required columns and data integrity.
        
        Args:
            df: DataFrame to validate
            mode: Mode of operation ('segmentation' or 'classification')
        
        Returns:
            Validation results
        """
        results = {
            "valid": False,
            "issues": [],
            "warnings": [],
            "stats": {}
        }
        
        # Check if DataFrame is empty
        if df.empty:
            results["issues"].append("DataFrame is empty")
            return results
        
        # Check for required columns
        required_cols = ["image_path"]
        if mode == "segmentation":
            required_cols.append("mask_path")
        if mode == "classification":
            required_cols.append("label")
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results["issues"].append(f"Missing required columns: {missing_cols}")
        
        # Check for file existence
        if "image_path" in df.columns:
            # Sample a subset of paths to check for large datasets
            sample_size = min(100, len(df))
            sample_df = df.sample(sample_size, random_state=42)
            
            missing_images = 0
            for idx, row in sample_df.iterrows():
                if not Path(row['image_path']).exists():
                    missing_images += 1
            
            missing_pct = (missing_images / sample_size) * 100
            if missing_pct > 0:
                results["warnings"].append(
                    f"Approximately {missing_pct:.1f}% of image files are missing"
                )
                if missing_pct > 50:
                    results["issues"].append(
                        f"More than 50% of image files are missing"
                    )
        
        # Check for mask existence in segmentation mode
        if mode == "segmentation" and "mask_path" in df.columns:
            missing_masks = 0
            for idx, row in sample_df.iterrows():
                if 'mask_path' in row and not Path(row['mask_path']).exists():
                    missing_masks += 1
            
            missing_mask_pct = (missing_masks / sample_size) * 100
            if missing_mask_pct > 0:
                results["warnings"].append(
                    f"Approximately {missing_mask_pct:.1f}% of mask files are missing"
                )
                if missing_mask_pct > 50:
                    results["issues"].append(
                        f"More than 50% of mask files are missing"
                    )
        
        # Check for valid labels in classification mode
        if mode == "classification" and "label" in df.columns:
            invalid_labels = df['label'].isna().sum()
            if invalid_labels > 0:
                results["warnings"].append(
                    f"{invalid_labels} rows have missing labels"
                )
        
        # Generate statistics
        results["stats"] = {
            "num_samples": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns)
        }
        
        # Add class distribution if label column exists
        if "label" in df.columns:
            label_counts = df['label'].value_counts().to_dict()
            results["stats"]["class_distribution"] = label_counts
        
        # Add dataset distribution if dataset column exists
        if "dataset" in df.columns:
            dataset_counts = df['dataset'].value_counts().to_dict()
            results["stats"]["dataset_distribution"] = dataset_counts
        
        # Set valid flag if no critical issues were found
        results["valid"] = len(results["issues"]) == 0
        
        return results
    
    @log_exception_handler
    def validate_image_file(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate a single image file.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Validation results
        """
        img_path = Path(image_path)
        results = {
            "path": str(img_path),
            "valid": False,
            "exists": img_path.exists(),
            "is_file": img_path.is_file() if img_path.exists() else False,
            "file_size": img_path.stat().st_size if img_path.exists() and img_path.is_file() else 0,
            "issues": []
        }
        
        if not results["exists"]:
            results["issues"].append("File does not exist")
            return results
        
        if not results["is_file"]:
            results["issues"].append("Path is not a file")
            return results
        
        # Try to open and read the image
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                results["issues"].append("Could not read image with OpenCV")
                return results
            
            # Get image dimensions and channels
            h, w = img.shape[:2]
            c = img.shape[2] if len(img.shape) > 2 else 1
            
            results.update({
                "width": w,
                "height": h,
                "channels": c,
                "min_value": float(img.min()),
                "max_value": float(img.max()),
                "mean_value": float(img.mean())
            })
            
            # Check for reasonable dimensions
            if w < 10 or h < 10:
                results["issues"].append(f"Image dimensions too small: {w}x{h}")
            
            # Check if image is empty or has strange values
            if img.min() == img.max():
                results["issues"].append("Image has uniform values (possibly empty)")
            
            # Set valid flag if no issues were found
            results["valid"] = len(results["issues"]) == 0
            
        except Exception as e:
            results["issues"].append(f"Error reading image: {str(e)}")
        
        return results
    
    @log_exception_handler
    def validate_mask_file(self, mask_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate a single mask file.
        
        Args:
            mask_path: Path to the mask file
        
        Returns:
            Validation results
        """
        mask_path = Path(mask_path)
        results = {
            "path": str(mask_path),
            "valid": False,
            "exists": mask_path.exists(),
            "is_file": mask_path.is_file() if mask_path.exists() else False,
            "file_size": mask_path.stat().st_size if mask_path.exists() and mask_path.is_file() else 0,
            "issues": []
        }
        
        if not results["exists"]:
            results["issues"].append("File does not exist")
            return results
        
        if not results["is_file"]:
            results["issues"].append("Path is not a file")
            return results
        
        # Try to open and read the mask
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                results["issues"].append("Could not read mask with OpenCV")
                return results
            
            # Get mask dimensions
            h, w = mask.shape[:2]
            
            # Check unique values (binary masks should have at most 2 unique values)
            unique_values = np.unique(mask)
            
            results.update({
                "width": w,
                "height": h,
                "min_value": float(mask.min()),
                "max_value": float(mask.max()),
                "mean_value": float(mask.mean()),
                "unique_values": [int(v) for v in unique_values]
            })
            
            # Check if mask is binary
            if len(unique_values) > 2:
                results["issues"].append(
                    f"Mask has {len(unique_values)} unique values, expected 2 for binary mask"
                )
            
            # Check if mask dimensions make sense
            if w < 10 or h < 10:
                results["issues"].append(f"Mask dimensions too small: {w}x{h}")
            
            # Set valid flag if no issues were found
            results["valid"] = len(results["issues"]) == 0
            
        except Exception as e:
            results["issues"].append(f"Error reading mask: {str(e)}")
        
        return results
    
    def validate_image_mask_pair(self, 
                               image_path: Union[str, Path], 
                               mask_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate an image-mask pair for compatibility.
        
        Args:
            image_path: Path to the image file
            mask_path: Path to the mask file
        
        Returns:
            Validation results
        """
        img_result = self.validate_image_file(image_path)
        mask_result = self.validate_mask_file(mask_path)
        
        results = {
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "valid": False,
            "issues": []
        }
        
        # Check individual validity
        if not img_result["valid"]:
            results["issues"].append(f"Invalid image: {img_result['issues']}")
        
        if not mask_result["valid"]:
            results["issues"].append(f"Invalid mask: {mask_result['issues']}")
        
        # If both files are valid, check compatibility
        if img_result["valid"] and mask_result["valid"]:
            # Check dimensions match
            if img_result["width"] != mask_result["width"] or img_result["height"] != mask_result["height"]:
                results["issues"].append(
                    f"Image and mask dimensions don't match: "
                    f"Image {img_result['width']}x{img_result['height']} vs "
                    f"Mask {mask_result['width']}x{mask_result['height']}"
                )
        
        # Set valid flag if no issues were found
        results["valid"] = len(results["issues"]) == 0
        
        return results
    
    def validate_dataset_sample(self, df: pd.DataFrame, sample_size: int = 10) -> Dict[str, Any]:
        """Validate a random sample of dataset entries.
        
        Args:
            df: DataFrame with dataset entries
            sample_size: Number of samples to validate
        
        Returns:
            Validation results
        """
        # Sample entries
        sample_df = df.sample(min(sample_size, len(df)), random_state=42)
        
        results = {
            "total_samples": len(df),
            "validated_samples": len(sample_df),
            "valid_samples": 0,
            "invalid_samples": 0,
            "sample_results": []
        }
        
        for idx, row in sample_df.iterrows():
            sample_result = {
                "row_idx": idx,
                "valid": True,
                "issues": []
            }
            
            # Check image path
            if 'image_path' in row:
                image_valid = Path(row['image_path']).exists()
                sample_result["image_exists"] = image_valid
                if not image_valid:
                    sample_result["valid"] = False
                    sample_result["issues"].append("Image file doesn't exist")
            else:
                sample_result["valid"] = False
                sample_result["issues"].append("No image_path in row")
            
            # Check mask path for segmentation
            if 'mask_path' in row:
                mask_valid = Path(row['mask_path']).exists()
                sample_result["mask_exists"] = mask_valid
                if not mask_valid:
                    sample_result["valid"] = False
                    sample_result["issues"].append("Mask file doesn't exist")
            
            # If both image and mask exist, validate the pair
            if sample_result.get("image_exists", False) and sample_result.get("mask_exists", False):
                pair_result = self.validate_image_mask_pair(row['image_path'], row['mask_path'])
                if not pair_result["valid"]:
                    sample_result["valid"] = False
                    sample_result["issues"].extend(pair_result["issues"])
            
            # Add to results
            results["sample_results"].append(sample_result)
            if sample_result["valid"]:
                results["valid_samples"] += 1
            else:
                results["invalid_samples"] += 1
        
        # Calculate validity percentage
        if results["validated_samples"] > 0:
            results["validity_percentage"] = (results["valid_samples"] / results["validated_samples"]) * 100
        else:
            results["validity_percentage"] = 0
        
        return results
    
    def generate_validation_report(self, df: pd.DataFrame, output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Generate a comprehensive validation report for a dataset.
        
        Args:
            df: DataFrame with dataset entries
            output_path: Path to save the report
        
        Returns:
            Validation report
        """
        logger.info(f"Generating validation report for dataset with {len(df)} entries")
        
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "dataset_size": len(df),
            "dataframe_validation": self.validate_dataframe(df),
            "dataset_sample_validation": self.validate_dataset_sample(df, sample_size=min(20, len(df))),
            "dataset_sources": {}
        }
        
        # Validate by dataset source if available
        if 'dataset' in df.columns:
            for dataset_name in df['dataset'].unique():
                dataset_df = df[df['dataset'] == dataset_name]
                report["dataset_sources"][dataset_name] = {
                    "size": len(dataset_df),
                    "validation": self.validate_dataframe(dataset_df),
                    "structure_validation": self.validate_dataset_structure(dataset_name) 
                    if dataset_name in ["ORIGA", "REFUGE", "G1020"] else None
                }
        
        # Overall validity
        report["overall_valid"] = report["dataframe_validation"]["valid"] and \
                                 report["dataset_sample_validation"]["valid_samples"] > 0
        
        # Save report if output path is provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Validation report saved to {output_path}")
        
        return report