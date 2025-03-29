import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the project root directory
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # Adjust based on where you put the script

# Define the test run function
@hydra.main(config_path=str(project_root / "conf"), config_name="config", version_base=None)
def test_run(cfg: DictConfig) -> None:
    """Inspect and run a quick test of the pipeline."""
    
    # First, inspect the configuration structure
    logger.info("Configuration structure:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Check what fields exist in the config
    logger.info(f"Top-level keys: {list(cfg.keys())}")
    
    # Safely modify configuration based on what's available
    # Only modify if the appropriate sections exist
    if hasattr(cfg, 'training'):
        logger.info(f"Training section keys: {list(cfg.training.keys())}")
        if hasattr(cfg.training, 'epochs'):
            cfg.training.epochs = 2
            logger.info("Set training epochs to 2")
        else:
            logger.warning("No 'epochs' field found in training section")
    else:
        logger.warning("No 'training' section found in config")
        
    if hasattr(cfg, 'pipeline') and hasattr(cfg.pipeline, 'steps'):
        cfg.pipeline.steps = ["extract", "load", "clean", "preprocess", "train", "evaluate"]
        cfg.pipeline.description = "Quick test run with debugging"
    
    # Try to run the pipeline if possible
    try:
        logger.info("Attempting to run the pipeline...")
        # Import run_pipeline after Hydra has initialized
        from glaucoma_detection.run_pipeline import run_with_hydra
        
        # Execute the pipeline
        run_with_hydra(cfg)
        
        logger.info("Test run completed!")
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Show full error traces
    test_run()