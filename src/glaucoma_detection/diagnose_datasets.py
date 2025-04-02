import os
from pathlib import Path

def diagnose_dataset_structure(base_dir="."):
    """Check if the expected dataset structure exists in either location."""
    # Check both possible data locations
    data_dir = Path(base_dir) / "data"
    package_data_dir = Path(base_dir) / "src" / "glaucoma_detection" / "data"
    
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
    
    print(f"Checking dataset structure...")
    print(f"Location 1: {data_dir}")
    print(f"Location 2: {package_data_dir}")
    
    # Check if either directory exists
    if not data_dir.exists() and not package_data_dir.exists():
        print(f"ERROR: Neither data directory exists!")
        return False
    
    # Check both locations
    locations = {
        "Project data directory": data_dir,
        "Package data directory": package_data_dir
    }
    
    found_datasets = False
    
    for location_name, location_path in locations.items():
        print(f"\nChecking {location_name}: {location_path}")
        
        if not location_path.exists():
            print(f"  - Directory does not exist")
            continue
            
        location_has_datasets = False
        
        for dataset_name, requirements in datasets.items():
            dataset_path = location_path / dataset_name
            
            if dataset_path.exists():
                location_has_datasets = True
                found_datasets = True
                print(f"  - Found {dataset_name} at {dataset_path}")
                
                # Check files and directories
                all_valid = True
                
                # Check required files
                for required_file in requirements.get("required_files", []):
                    file_path = dataset_path / required_file
                    if file_path.exists():
                        print(f"    ✅ Found required file: {required_file}")
                    else:
                        print(f"    ❌ Missing required file: {required_file}")
                        all_valid = False
                
                # Check required directories
                for required_dir in requirements.get("required_dirs", []):
                    dir_path = dataset_path / required_dir
                    if dir_path.exists() and dir_path.is_dir():
                        print(f"    ✅ Found required directory: {required_dir}")
                    else:
                        print(f"    ❌ Missing required directory: {required_dir}")
                        all_valid = False
                
                if all_valid:
                    print(f"    ✅ {dataset_name} structure is valid")
                else:
                    print(f"    ❌ {dataset_name} has structural issues")
            else:
                print(f"  - {dataset_name} not found")
        
        if not location_has_datasets:
            print(f"  - No datasets found in this location")
    
    if not found_datasets:
        print("\n❌ No datasets found in any location")
    
    return found_datasets

if __name__ == "__main__":
    diagnose_dataset_structure()