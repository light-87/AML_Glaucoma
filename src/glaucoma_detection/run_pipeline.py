"""
Simplified Pipeline Coordinator

Coordinates the glaucoma detection pipeline with fixed config path.
"""

import os
import logging
import datetime
import pandas as pd
import torch
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

# Import project modules
from glaucoma_detection.path_utils import PathManager
from glaucoma_detection.data_loader import consolidate_datasets, save_consolidated_dataset
from glaucoma_detection.data_cleaner import clean_dataset
from glaucoma_detection.preprocessor import GlaucomaDataModule
from glaucoma_detection.model import create_model, save_model
from glaucoma_detection.trainer import train_model
from glaucoma_detection.evaluator import SegmentationEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pipeline")

# Add at the beginning of the file
import torch
torch.set_float32_matmul_precision('medium')

# IMPORTANT: Use the correct relative path to your config directory
# Assuming conf is in the project root directory: AML_Glaucoma/conf
@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def run_pipeline(cfg: DictConfig):
    """Run the glaucoma detection pipeline.
    
    Args:
        cfg: Configuration from Hydra
    """
    start_time = datetime.datetime.now()
    run_timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Starting pipeline run at {run_timestamp}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Initialize path manager
    path_manager = PathManager()
    logger.info(f"Data directory: {path_manager.data_dir}")
    
    # Create output directories
    output_dir = Path(cfg.paths.output_dir)
    run_dir = output_dir / f"run_{run_timestamp}"
    run_dir.mkdir(exist_ok=True, parents=True)
    
    # Save configuration
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Saved configuration to {config_path}")
    
    # Determine steps to run
    steps = cfg.pipeline.steps if hasattr(cfg.pipeline, "steps") else ["load", "clean", "preprocess", "train", "evaluate"]
    logger.info(f"Pipeline steps: {steps}")
    
    # Step 1: Load data
    if "load" in steps:
        logger.info("Step 1: Loading data")
        
        # Load datasets from the package data directory
        data_df = consolidate_datasets(path_manager.data_dir)
        
        if data_df.empty:
            logger.error("No data loaded from datasets")
            return
        
        # Save consolidated dataset
        data_path = run_dir / "consolidated_data.csv"
        save_consolidated_dataset(data_df, data_path)
    else:
        # Try to load existing data
        data_path = run_dir / "consolidated_data.csv"
        if not data_path.exists():
            data_path = output_dir / "consolidated_data.csv"
        
        if data_path.exists():
            data_df = pd.read_csv(data_path)
            logger.info(f"Loaded existing data with {len(data_df)} samples")
        else:
            logger.error("No data found and 'load' step not included")
            return
    
    # Step 2: Clean data
    if "clean" in steps:
        logger.info("Step 2: Cleaning data")
        
        cleaned_df = clean_dataset(data_df, random_state=cfg.data.random_state)
        
        # Save cleaned dataset
        cleaned_path = run_dir / "cleaned_data.csv"
        cleaned_df.to_csv(cleaned_path, index=False)
        logger.info(f"Saved cleaned data with {len(cleaned_df)} samples")
    else:
        # Try to load existing cleaned data
        cleaned_path = run_dir / "cleaned_data.csv"
        if not cleaned_path.exists():
            cleaned_path = output_dir / "cleaned_data.csv"
        
        if cleaned_path.exists():
            cleaned_df = pd.read_csv(cleaned_path)
            logger.info(f"Loaded existing cleaned data with {len(cleaned_df)} samples")
        else:
            logger.info("Using uncleaned data")
            cleaned_df = data_df
    
    # Step 3: Preprocess data
    if "preprocess" in steps:
        logger.info("Step 3: Preprocessing data")
        
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
        
        # Set up data module
        data_module.setup()
        
        # Save splits
        data_module.train_df.to_csv(run_dir / "train_data.csv", index=False)
        data_module.val_df.to_csv(run_dir / "val_data.csv", index=False)
        data_module.test_df.to_csv(run_dir / "test_data.csv", index=False)
        
        logger.info(f"Created data splits: Train: {len(data_module.train_df)}, " 
                   f"Val: {len(data_module.val_df)}, Test: {len(data_module.test_df)}")
    else:
        # Load existing splits
        train_path = run_dir / "train_data.csv"
        val_path = run_dir / "val_data.csv"
        test_path = run_dir / "test_data.csv"
        
        if not all(p.exists() for p in [train_path, val_path, test_path]):
            # Try previous runs
            train_path = output_dir / "train_data.csv"
            val_path = output_dir / "val_data.csv"
            test_path = output_dir / "test_data.csv"
        
        if all(p.exists() for p in [train_path, val_path, test_path]):
            # Load existing splits
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
            
            # Create data module with existing splits
            data_module = GlaucomaDataModule(
                data_df=cleaned_df,
                batch_size=cfg.training.batch_size,
                num_workers=cfg.training.num_workers,
                target_size=tuple(cfg.preprocessing.image_size),
                augment_train=cfg.preprocessing.augmentation.enabled,
                mode=cfg.preprocessing.mode,
                use_existing_splits=True,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df
            )
            data_module.setup()
            logger.info(f"Using existing splits: Train: {len(train_df)}, " 
                       f"Val: {len(val_df)}, Test: {len(test_df)}")
        else:
            # Create new splits
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
            data_module.setup()
            logger.info("Created new data splits (no existing splits found)")
    
    # Step 4: Train model
    model = None
    if "train" in steps:
        logger.info("Step 4: Training model")
        
        # Initialize wandb if enabled
        if cfg.logging.use_wandb:
            import wandb
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
            output_dir=str(run_dir)
        )
        
        # Save trained model
        model_path = run_dir / "model.pt"
        save_model(lightning_model.model, str(model_path))
        logger.info(f"Saved trained model to {model_path}")
        
        # Close wandb if used
        if cfg.logging.use_wandb:
            wandb.finish()
    else:
        # Try to load existing model
        model_path = run_dir / "model.pt"
        if not model_path.exists():
            model_path = output_dir / "model.pt"
        
        if model_path.exists():
            from glaucoma_detection.model import load_model
            model, _ = load_model(str(model_path), cfg.model)
            logger.info(f"Loaded existing model from {model_path}")
        else:
            logger.warning("No model found and 'train' step not included")
    
    # Step 5: Evaluate model
    if "evaluate" in steps and model is not None:
        logger.info("Step 5: Evaluating model")
        
        # Create evaluation directory
        eval_dir = run_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        
        # Create evaluator
        from glaucoma_detection.trainer import GlaucomaSegmentationModel
        lightning_model = GlaucomaSegmentationModel(model, cfg.training)
        evaluator = SegmentationEvaluator(
            model=lightning_model,
            output_dir=str(eval_dir)
        )
        
        # Run evaluation
        device = "cuda" if torch.cuda.is_available() and cfg.training.use_gpu else "cpu"
        test_loader = data_module.test_dataloader()
        results = evaluator.evaluate(test_loader=test_loader, device=device)
        
        logger.info(f"Evaluation results: {results}")
    
    # Calculate run time
    end_time = datetime.datetime.now()
    run_time = end_time - start_time
    logger.info(f"Pipeline run completed in {run_time}")
    
    # Update notebook.md
    notebook_path = output_dir / "notebook.md"
    if not notebook_path.exists():
        with open(notebook_path, "w") as f:
            f.write("# Glaucoma Detection Pipeline Run Log\n\n")
            f.write("| Timestamp | Steps | Duration | Results |\n")
            f.write("|-----------|-------|----------|--------|\n")
    
    with open(notebook_path, "a") as f:
        steps_str = ", ".join(steps)
        results_str = "Completed" if "evaluate" not in steps else f"Dice: {results.get('dice', 'N/A'):.4f}"
        f.write(f"| {run_timestamp} | {steps_str} | {run_time} | {results_str} |\n")

if __name__ == "__main__":
    # Configure Hydra
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Show full error traces
    
    # Run the pipeline
    run_pipeline()