"""
Configuration Module

This module uses Hydra for configuration management.
"""

from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path
import os

# Define default configuration structure
@hydra.main(config_path="conf", config_name="config", version_base=None)
def get_config(cfg: DictConfig) -> DictConfig:
    """Get configuration with Hydra."""
    return cfg

# Default paths setup
def setup_paths(cfg: DictConfig) -> DictConfig:
    """Set up and create necessary directories."""
    # Set base paths
    os.makedirs(cfg.paths.data_dir, exist_ok=True)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    os.makedirs(cfg.paths.model_dir, exist_ok=True)
    os.makedirs(cfg.paths.log_dir, exist_ok=True)
    return cfg

def load_config(config_path: str = None) -> DictConfig:
    """Load config from path or use default."""
    if config_path and os.path.exists(config_path):
        return OmegaConf.load(config_path)
    # Return default config
    cfg = OmegaConf.create()
    # Add your default configuration here
    return cfg