"""
Run Pipeline With SSL Fix

This script applies the SSL fix and then runs the pipeline with appropriate settings.
"""

import os
import sys
import subprocess
from glaucoma_detection.ssl_fix import fix_ssl_for_torch_downloads

def run_pipeline_with_fix():
    """Apply SSL fix and run the pipeline."""
    print("Step 1: Applying SSL certificate fix...")
    fix_ssl_for_torch_downloads()
    
    print("\nStep 2: Running pipeline with appropriate settings...")
    
    # Define the command with settings for your GPU
    cmd = [
        sys.executable, 
        "-m", 
        "glaucoma_detection.run_pipeline",
        "pipeline.steps=[load,clean,preprocess,train]",  # Skip evaluation for now
        "training.batch_size=16",  # Based on your GPU
        "training.epochs=1",       # Just 1 epoch for testing
        "model.encoder=resnet18",  # Lighter encoder
        "preprocessing.image_size=[224,224]"  # Full resolution is fine for your GPU
    ]
    
    # Run the command
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                             universal_newlines=True, bufsize=1)
    
    # Print output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code == 0:
        print("\nPipeline completed successfully!")
        print("You can now run the full pipeline with your desired settings.")
    else:
        print(f"\nPipeline failed with return code {return_code}")
        print("Check the error messages above for troubleshooting.")

if __name__ == "__main__":
    run_pipeline_with_fix()