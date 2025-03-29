"""
Final test script for the glaucoma pipeline.
Tests the full pipeline with a 2-epoch run.
"""

import os
import sys
from pathlib import Path
import logging
import subprocess

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

def run_test():
    """Run a 2-epoch test of the glaucoma detection pipeline."""
    logger.info("Starting 2-epoch test run of the glaucoma detection pipeline...")
    
    # Choose which steps to run
    steps = ["extract", "load", "clean", "preprocess", "train", "evaluate"]
    
    # Construct the command with appropriate arguments
    cmd = [
        sys.executable,  # Use the current Python interpreter
        "-m", "glaucoma_detection.run_pipeline",
        "--steps", ",".join(steps),
        "--description", "Test run with 2 epochs",
        "--force",  # Force rerun of steps
        "training.epochs=2",  # Set to 2 epochs for quick test
        "training.batch_size=8",  # Smaller batch size for testing
        "logging.use_wandb=false"  # Disable wandb logging
    ]
    
    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Set PYTHONPATH to include src directory
        env = os.environ.copy()
        env["PYTHONPATH"] = str(src_dir)
        
        # Run the process with live output
        process = subprocess.Popen(
            cmd,
            cwd=project_root,  # Run from project root
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream the output in real time
        for line in process.stdout:
            logger.info(line.strip())
        
        # Wait for process to complete and get return code
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("Test completed successfully!")
        else:
            logger.error(f"Test failed with return code: {return_code}")
    
    except Exception as e:
        logger.error(f"Error running test: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_test()