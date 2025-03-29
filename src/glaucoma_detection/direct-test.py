"""
Simple test script that tests basic functionality without Hydra.
This bypasses configuration issues by hardcoding test values.
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the src directory to Python path if needed
current_file = Path(__file__).resolve()
project_root = current_file.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

def run_simple_test():
    """Run a simple test of the glaucoma detection pipeline components."""
    
    # Basic configurations
    data_dir = project_root / "data"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Try to import and test model creation
    try:
        from glaucoma_detection.model import create_model
        
        # Create a simple model configuration
        model_config = {
            'architecture': 'unet',
            'encoder': 'resnet34',
            'pretrained': True,
            'in_channels': 3,
            'num_classes': 1
        }
        
        # Create the model
        logger.info("Creating model...")
        model = create_model(model_config)
        logger.info(f"Successfully created model: {type(model).__name__}")
        
        # Test with a sample input
        import torch
        sample_input = torch.randn(1, 3, 224, 224)
        sample_output = model(sample_input)
        logger.info(f"Model output shape: {sample_output.shape}")
    except Exception as e:
        logger.error(f"Error in model testing: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Try loading dataset
    try:
        from glaucoma_detection.data_loader import consolidate_datasets, save_consolidated_dataset
        
        if data_dir.exists():
            logger.info("Testing data loading functionality...")
            df = consolidate_datasets(str(data_dir))
            logger.info(f"Loaded dataset with {len(df)} samples")
            
            # Save consolidated dataset
            consolidated_csv = output_dir / "test_consolidated_dataset.csv"
            save_consolidated_dataset(df, str(consolidated_csv))
            logger.info(f"Saved test consolidated dataset to {consolidated_csv}")
        else:
            logger.warning(f"Data directory {data_dir} does not exist. Skipping data loading test.")
    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("Simple test completed!")

if __name__ == "__main__":
    run_simple_test()