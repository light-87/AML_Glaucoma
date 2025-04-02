"""
GPU Memory Check Script

Run this to check if your GPU has enough memory for training.
"""

import torch
import os

def check_gpu_memory():
    """Check available GPU memory and provide recommendations."""
    print("Checking GPU availability and memory...")
    
    if not torch.cuda.is_available():
        print("No GPU detected. Training will use CPU which will be slow.")
        return
    
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    
    # Get GPU memory information
    total_memory = torch.cuda.get_device_properties(0).total_memory
    reserved_memory = torch.cuda.memory_reserved(0)
    allocated_memory = torch.cuda.memory_allocated(0)
    free_memory = total_memory - allocated_memory
    
    # Convert to GB for better readability
    total_gb = total_memory / (1024**3)
    free_gb = free_memory / (1024**3)
    
    print(f"Total GPU memory: {total_gb:.2f} GB")
    print(f"Free GPU memory: {free_gb:.2f} GB")
    
    # Provide recommendations based on available memory
    if free_gb > 10:
        print("Memory Status: EXCELLENT - You have plenty of memory.")
        print("You can use a large batch size (32+) and large model.")
    elif free_gb > 6:
        print("Memory Status: GOOD - You have sufficient memory.")
        print("Recommended batch size: 16-32")
    elif free_gb > 3:
        print("Memory Status: MODERATE - You need to be careful with memory usage.")
        print("Recommended batch size: 8-16")
        print("Consider using a smaller encoder (resnet18 instead of resnet34)")
    else:
        print("Memory Status: LIMITED - Training may be difficult with your current GPU.")
        print("Recommended batch size: 4-8")
        print("Consider using:")
        print("- A smaller encoder (resnet18)")
        print("- Smaller image size (e.g., 128x128)")
        print("- Mixed precision training")
    
    # Additional recommendations for different dataset sizes
    dataset_size = 8070  # From your logs
    print(f"\nWith your dataset size of {dataset_size} samples:")
    
    # Estimate memory required for different batch sizes
    # Very rough estimate: 5MB per image at 224x224 resolution
    memory_per_batch_8 = 8 * 5  # MB
    memory_per_batch_16 = 16 * 5  # MB
    memory_per_batch_32 = 32 * 5  # MB
    
    print(f"Estimated memory for batch size 8: {memory_per_batch_8} MB")
    print(f"Estimated memory for batch size 16: {memory_per_batch_16} MB")
    print(f"Estimated memory for batch size 32: {memory_per_batch_32} MB")
    
    print("\nRecommended configuration for your GPU:")
    if free_gb > 10:
        print("batch_size: 32")
        print("encoder: resnet34 or efficientnet-b4")
        print("image_size: [224, 224]")
    elif free_gb > 6:
        print("batch_size: 16")
        print("encoder: resnet34")
        print("image_size: [224, 224]")
    elif free_gb > 3:
        print("batch_size: 8")
        print("encoder: resnet18")
        print("image_size: [224, 224]")
    else:
        print("batch_size: 4")
        print("encoder: resnet18")
        print("image_size: [128, 128]")
        print("precision: 16-mixed")

if __name__ == "__main__":
    check_gpu_memory()