"""
Improved Dataset Diagnostics Tool

Check the structure of the datasets in the package data directory.
Support various dataset structures including nested directories.
"""

import os
from pathlib import Path

def count_images_recursively(directory):
    """Count all image files recursively in a directory and its subdirectories."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    count = 0
    
    # Check if path exists and is a directory
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return 0
        
    # Walk through all subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                count += 1
    
    return count

def diagnose_datasets(base_dir="."):
    """Check if the expected dataset structure exists in the package data directory."""
    # Only check package data directory
    data_dir = Path(base_dir) / "src" / "glaucoma_detection" / "data"
    
    print(f"Checking dataset structure in: {data_dir}")
    
    if not data_dir.exists():
        print(f"❌ ERROR: Data directory does not exist: {data_dir}")
        return False
    
    datasets = {
        "ORIGA": {
            "required_files": ["OrigaList.csv"],
            "required_dirs": ["Images", "Masks"]
        },
        "REFUGE": {
            "required_dirs": ["train", "val", "test"]
        },
        "G1020": {
            "required_files": ["G1020.csv"],
            "required_dirs": ["Images", "Masks"]
        }
    }
    
    found_datasets = False
    for dataset_name, requirements in datasets.items():
        dataset_path = data_dir / dataset_name
        
        if dataset_path.exists():
            found_datasets = True
            print(f"\n✅ Found {dataset_name} dataset at {dataset_path}")
            
            # Check files
            for required_file in requirements.get("required_files", []):
                file_path = dataset_path / required_file
                if file_path.exists():
                    print(f"  ✅ Found required file: {required_file}")
                else:
                    print(f"  ❌ Missing required file: {required_file}")
            
            # Check directories
            for required_dir in requirements.get("required_dirs", []):
                dir_path = dataset_path / required_dir
                if dir_path.exists() and dir_path.is_dir():
                    print(f"  ✅ Found required directory: {required_dir}")
                    
                    # For REFUGE, check if split directories have contents
                    if dataset_name == "REFUGE" and required_dir in ["train", "val", "test"]:
                        # Check recursively for images
                        image_count = count_images_recursively(dir_path)
                        
                        if image_count > 0:
                            print(f"    ✅ Found {image_count} images in {required_dir} (including subdirectories)")
                        else:
                            # Check for common subdirectories
                            images_subdir = dir_path / "Images"
                            if images_subdir.exists():
                                subdir_count = count_images_recursively(images_subdir)
                                print(f"    ✅ Found {subdir_count} images in {required_dir}/Images")
                            else:
                                print(f"    ❌ No images found in {required_dir} or its subdirectories")
                else:
                    print(f"  ❌ Missing required directory: {required_dir}")
        else:
            print(f"\n❌ {dataset_name} dataset not found at {dataset_path}")
    
    if not found_datasets:
        print("\n❌ No datasets found in the data directory")
    
    print("\nSummary of available data:")
    for dataset_name in datasets:
        dataset_path = data_dir / dataset_name
        if dataset_path.exists():
            if dataset_name in ["ORIGA", "G1020"]:
                # For ORIGA and G1020, check Images directory
                images_dir = dataset_path / "Images"
                if images_dir.exists():
                    image_count = count_images_recursively(images_dir)
                    print(f"- {dataset_name}: {image_count} images")
                else:
                    print(f"- {dataset_name}: Images directory not found")
            elif dataset_name == "REFUGE":
                # For REFUGE, images might be in split directories
                total_images = 0
                for split in ["train", "val", "test"]:
                    split_dir = dataset_path / split
                    if split_dir.exists():
                        split_count = count_images_recursively(split_dir)
                        total_images += split_count
                        print(f"  - {dataset_name}/{split}: {split_count} images")
                print(f"- {dataset_name}: {total_images} total images across all splits")
    
    # Check each dataset for detailed structure
    print("\nDetailed structure analysis:")
    for dataset_name in datasets:
        dataset_path = data_dir / dataset_name
        if dataset_path.exists():
            print(f"\n{dataset_name} Dataset Structure:")
            
            if dataset_name == "REFUGE":
                # Look for possible structure variations in REFUGE
                for split in ["train", "val", "test"]:
                    split_dir = dataset_path / split
                    if split_dir.exists():
                        print(f"  - {split}/")
                        
                        # Check for common subdirectory patterns
                        for subdir in ["Images", "GT", "Masks", "Labels"]:
                            subdir_path = split_dir / subdir
                            if subdir_path.exists():
                                count = count_images_recursively(subdir_path)
                                print(f"    - {subdir}/: {count} images")
                        
                        # Check for direct images in split directory
                        direct_images = len(list(split_dir.glob("*.jpg"))) + len(list(split_dir.glob("*.png")))
                        if direct_images > 0:
                            print(f"    - (directly in {split}/): {direct_images} images")
    
    return found_datasets

if __name__ == "__main__":
    import sys
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    diagnose_datasets(base_dir)