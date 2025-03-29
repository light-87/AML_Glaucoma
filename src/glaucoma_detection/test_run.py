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
# Adjust this logic based on your project structure
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # Navigate from test_run.py to project root

# Print paths to help debug
logger.info(f"Current file: {current_file}")
logger.info(f"Project root: {project_root}")
logger.info(f"Config path: {project_root / 'conf'}")

# Define the test run function
@hydra.main(config_path=str(project_root / "conf"), config_name="config", version_base=None)
def test_run(cfg: DictConfig) -> None:
    """Run a quick test of the glaucoma detection pipeline with 2 epochs."""
    
    # Override configuration for quick testing
    cfg.training.epochs = 2
    cfg.training.batch_size = 8
    cfg.logging.use_wandb = False
    
    # Specify which steps to run
    cfg.pipeline.steps = ["extract", "load", "clean", "preprocess", "train", "evaluate"]
    cfg.pipeline.description = "Quick test run with 2 epochs"
    
    # Print the configuration
    logger.info("Test run configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Import run_pipeline after Hydra has initialized
    from glaucoma_detection.run_pipeline import run_with_hydra
    
    # Execute the pipeline
    run_with_hydra(cfg)
    
    logger.info("Test run completed!")

if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Show full error traces
    test_run()