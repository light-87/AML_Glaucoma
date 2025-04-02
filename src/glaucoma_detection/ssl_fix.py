"""
SSL Certificate Fix for PyTorch Model Downloads

This script disables SSL certificate verification for model downloads.
IMPORTANT: This is for development/testing only, not recommended for production.
"""

import ssl
import torch
from functools import wraps
import urllib.request
import os

def fix_ssl_for_torch_downloads():
    """Apply SSL fix for PyTorch model downloads."""
    print("Applying SSL certificate verification bypass...")
    
    # Create unverified context
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Patch torch.hub.load_state_dict_from_url
    original_load = torch.hub.load_state_dict_from_url
    
    @wraps(original_load)
    def patched_load(*args, **kwargs):
        try:
            return original_load(*args, **kwargs)
        except Exception as e:
            print(f"Error during model download: {e}")
            print("Attempting direct download with SSL verification disabled...")
            url = args[0]
            cached_file = args[1] if len(args) > 1 else kwargs.get('model_dir', None)
            
            if cached_file is None:
                # Determine default cache location
                torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.cache/torch'))
                hub_dir = os.path.join(torch_home, 'hub')
                model_dir = os.path.join(hub_dir, 'checkpoints')
                os.makedirs(model_dir, exist_ok=True)
                filename = os.path.basename(url)
                cached_file = os.path.join(model_dir, filename)
            
            # Download with SSL verification disabled
            print(f"Downloading {url} to {cached_file}")
            with urllib.request.urlopen(url) as source, open(cached_file, "wb") as output:
                output.write(source.read())
            
            print(f"Download complete. Saved to {cached_file}")
            return torch.load(cached_file, map_location=kwargs.get('map_location', None))
    
    # Replace the original function with our patched version
    torch.hub.load_state_dict_from_url = patched_load
    
    print("SSL fix applied. Model downloads should now work.")

if __name__ == "__main__":
    fix_ssl_for_torch_downloads()
    print("You can now run your pipeline without SSL certificate issues.")