"""
A very simple test script that directly imports the modules to test them.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))

# Now try importing the modules
try:
    from glaucoma_detection.model import create_model
    from glaucoma_detection.data_loader import consolidate_datasets
    print("Successfully imported the modules!")
    
    # Try creating a simple model
    model_config = {
        'architecture': 'unet',
        'encoder': 'resnet34',
        'pretrained': True,
        'in_channels': 3,
        'num_classes': 1
    }
    
    model = create_model(model_config)
    print(f"Successfully created a model: {type(model)}")
    
    print("Basic module tests passed!")
except Exception as e:
    print(f"Error importing or testing the modules: {e}")
    import traceback
    print(traceback.format_exc())