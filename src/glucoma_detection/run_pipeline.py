"""
Pipeline Coordinator

Coordinates the execution of the glaucoma detection pipeline using modern libraries.
"""

import os
import logging
import datetime
import uuid
from pathlib import Path
import typer
import pandas as pd
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from typing import List, Optional, Dict, Any, Union, Tuple
import wandb

# Import refactored modules
from data_loader import extract_zip, consolidate_datasets
from data_cleaner import clean_dataset
from preprocessor import GlaucomaDataModule
from model import create_model, save_model, load_model
from trainer import GlaucomaSegmentationModel, train_model
from evaluator import SegmentationEvaluator

# Create Typer app
app = typer.Typer(help="Glaucoma Detection Pipeline")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for the pipeline."""
    # Print config
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Get paths
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create run ID and metadata
    run_id = str(uuid.uuid4())
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    logger.info(f"Starting pipeline run with ID: {run_id}")
    
    # Execute requested steps
    steps = cfg.pipeline.steps
    
    # Extract data if requested
    if "extract" in steps:
        if cfg.data.zip_file and os.path.exists(cfg.data.zip_file):
            extract_zip(cfg.data.zip_file, cfg.paths.data_dir)
    
    # Load data if requested
    if "load" in steps:
        df = consolidate_datasets(cfg.paths.data_dir)
        consolidated_csv = output_dir / "consolidated_glaucoma_dataset.csv"
        df.to_csv(consolidated_csv, index=False)
        logger.info(f"Saved consolidated dataset to {consolidated_csv}")
    else:
        # Try to load existing consolidated dataset
        consolidated_csv = output_dir / "consolidated_glaucoma_dataset.csv"
        if consolidated_csv.exists():
            df = pd.read_csv(consolidated_csv)
            logger.info(f"Loaded consolidated dataset from {consolidated_csv}")
        else:
            logger.error("No consolidated dataset found. Run the 'load' step first.")
            return
    
    # Clean data if requested
    if "clean" in steps:
        cleaned_df = clean_dataset(df, random_state=cfg.data.random_state)
        cleaned_csv = output_dir / "cleaned_glaucoma_dataset.csv"
        cleaned_df.to_csv(cleaned_csv, index=False)
        logger.info(f"Saved cleaned dataset to {cleaned_csv}")
    else:
        # Try to load existing cleaned dataset
        cleaned_csv = output_dir / "cleaned_glaucoma_dataset.csv"
        if cleaned_csv.exists():
            cleaned_df = pd.read_csv(cleaned_csv)
            logger.info(f"Loaded cleaned dataset from {cleaned_csv}")
        else:
            logger.warning("No cleaned dataset found. Using consolidated dataset.")
            cleaned_df = df
    
    # Create data module
    data_module = GlaucomaDataModule(
        data_df=cleaned_df,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        target_size=tuple(cfg.preprocessing.image_size),
        augment_train=cfg.preprocessing.augmentation.enabled,
        mode=cfg.preprocessing.mode,
        val_split=cfg.data.split_config.val_ratio,
        test_split=cfg.data.split_config.test_ratio,
        random_state=cfg.data.random_state
    )
    
    # Preprocess and split data
    data_module.setup()
    
    # Save train, val, test splits
    if "preprocess" in steps:
        data_module.train_df.to_csv(output_dir / "train_dataset.csv", index=False)
        data_module.val_df.to_csv(output_dir / "val_dataset.csv", index=False)
        data_module.test_df.to_csv(output_dir / "test_dataset.csv", index=False)
        logger.info(f"Saved data splits to {output_dir}")
    
    # Train model if requested
    if "train" in steps:
        # Initialize wandb if enabled
        if cfg.logging.use_wandb:
            wandb.init(
                project=cfg.logging.wandb_project,
                name=f"run_{run_timestamp}",
                config=OmegaConf.to_container(cfg, resolve=True)
            )
        
        # Create model
        model = create_model(cfg.model)
        
        # Train model
        lightning_model, trainer = train_model(
            model=model,
            data_module=data_module,
            config=cfg.training,
            output_dir=output_dir
        )
        
        # Close wandb
        if cfg.logging.use_wandb:
            wandb.finish()
    
    # Evaluate model if requested
    if "evaluate" in steps:
        # Load best model if exists
        model_path = output_dir / "models" / "glaucoma_model-best.ckpt"
        if not model_path.exists():
            model_path = list((output_dir / "models").glob("*.ckpt"))
            if model_path:
                model_path = model_path[0]
            else:
                logger.error("No model checkpoint found. Train model first.")
                return
        
        # Load model
        model = create_model(cfg.model)
        lightning_model = GlaucomaSegmentationModel.load_from_checkpoint(
            checkpoint_path=str(model_path),
            model=model,
            config=cfg.training
        )
        
        # Initialize evaluator
        evaluator = SegmentationEvaluator(
            model=lightning_model,
            output_dir=output_dir / "evaluation"
        )
        
        # Evaluate model
        test_loader = data_module.test_dataloader()
        results = evaluator.evaluate(
            test_loader=test_loader,
            device="cuda" if torch.cuda.is_available() and cfg.training.use_gpu else "cpu"
        )
        
        logger.info(f"Evaluation results: {results}")
    
    # Update run tracking
    run_info = {
        "id": run_id,
        "timestamp": run_timestamp,
        "steps": steps,
        "config": OmegaConf.to_container(cfg, resolve=True)
    }
    
    # Save run info
    import json
    with open(output_dir / f"run_{run_id}.json", 'w') as f:
        json.dump(run_info, f, indent=2)
    
    # Update notebook.md
    notebook_file = output_dir / "notebook.md"
    if not notebook_file.exists():
        with open(notebook_file, 'w') as f:
            f.write("# Pipeline Run Tracking\n\n")
            f.write("| Run ID | Timestamp | Steps | Description |\n")
            f.write("|--------|-----------|-------|-------------|\n")
    
    with open(notebook_file, 'a') as f:
        description = cfg.pipeline.description if hasattr(cfg.pipeline, "description") else ""
        steps_str = ", ".join(steps)
        f.write(f"| {run_id} | {run_timestamp} | {steps_str} | {description} |\n")
    
    logger.info(f"Pipeline run {run_id} completed")

@app.command()
def run(
    config: str = typer.Option(None, "--config", "-c", help="Path to config file"),
    steps: str = typer.Option("extract,load,clean,preprocess,train,evaluate", "--steps", "-s", 
                              help="Comma-separated list of steps to run"),
    output_dir: str = typer.Option("output", "--output-dir", "-o", help="Output directory"),
    data_dir: str = typer.Option(None, "--data-dir", "-d", help="Data directory"),
    zip_file: str = typer.Option(None, "--zip-file", "-z", help="Path to ZIP file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force rerun of all steps"),
    description: str = typer.Option("", "--description", help="Run description"),
    wandb: bool = typer.Option(False, "--wandb", help="Enable Weights & Biases logging")
):
    """Run the pipeline with specified steps."""
    # Override configuration with command line arguments
    overrides = []
    if steps:
        overrides.append(f"pipeline.steps=[{','.join(steps.split(','))}]")
    if output_dir:
        overrides.append(f"paths.output_dir={output_dir}")
    if data_dir:
        overrides.append(f"paths.data_dir={data_dir}")
    if zip_file:
        overrides.append(f"data.zip_file={zip_file}")
    if force:
        overrides.append("pipeline.force=true")
    if description:
        overrides.append(f"pipeline.description='{description}'")
    if wandb:
        overrides.append("logging.use_wandb=true")
    
    # Load config if provided
    if config:
        overrides.append(f"--config-path={os.path.dirname(config)}")
        overrides.append(f"--config-name={os.path.basename(config).split('.')[0]}")
    
    # Call hydra with overrides
    main_with_overrides = hydra.main(config_path="conf", config_name="config", version_base=None)(main)
    main_with_overrides(overrides)

if __name__ == "__main__":
    app()