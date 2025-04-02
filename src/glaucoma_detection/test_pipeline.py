"""
Lightweight Pipeline Test Script

Run this to test the pipeline with lighter settings that should work on most GPUs.
"""

import os
import sys
import subprocess

def run_memory_check():
    """Run the GPU memory check first."""
    print("Running GPU memory check...")
    result = subprocess.run([sys.executable, "-m", "glaucoma_detection.memory_check"], 
                          capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    
    return "No GPU detected" not in result.stdout

def run_pipeline_with_lighter_settings():
    """Run the pipeline with settings optimized for lower memory usage."""
    print("\nRunning pipeline with lightweight settings...")
    
    # Define the command with overrides for lighter processing
    cmd = [
        sys.executable, 
        "-m", 
        "glaucoma_detection.run_pipeline",
        "pipeline.steps=[load,clean,preprocess,train]",  # Skip evaluation for now
        "training.batch_size=8",  # Smaller batch size
        "training.epochs=1",  # Just 1 epoch for testing
        "model.encoder=resnet18",  # Lighter encoder
        "preprocessing.image_size=[128,128]",  # Smaller images
        "training.precision=16-mixed"  # Use mixed precision
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
    has_gpu = run_memory_check()
    
    if has_gpu:
        print("\nGPU detected, proceeding with lightweight test...")
    else:
        print("\nNo GPU detected. The test will run on CPU and may be very slow.")
        proceed = input("Do you want to continue anyway? (y/n): ")
        if proceed.lower() != 'y':
            sys.exit(0)
    
    run_pipeline_with_lighter_settings()