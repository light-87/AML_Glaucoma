"""
Enhanced Pipeline Coordinator

Improved pipeline coordination with robust error handling, validation, and memory efficiency.
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
import traceback
import sys
import gc

# Import enhanced project modules
from glaucoma_detection.logger import get_logger, ERROR_CODES, log_exception_handler
from glaucoma_detection.path_utils import get_path_manager, PathManager
from glaucoma_detection.data_validator import DataValidator
from glaucoma_detection.config_validator import validate_config
from glaucoma_detection.memory_efficient_loader import create_memory_efficient_data_loader, memory_stats

# Import original project modules
from glaucoma_detection.data_loader import extract_zip, consolidate_datasets, save_consolidated_dataset
from glaucoma_detection.data_cleaner import clean_dataset
from glaucoma_detection.preprocessor import GlaucomaDataModule
from glaucoma_detection.model import create_model, save_model, load_model
from glaucoma_detection.trainer import GlaucomaSegmentationModel, train_model
from glaucoma_detection.evaluator import SegmentationEvaluator

# Create Typer app
app = typer.Typer(help="Enhanced Glaucoma Detection Pipeline")

# Initialize logger
logger = get_logger("pipeline")

# Define the main function to be used with Hydra
@hydra.main(config_path="../conf", config_name="config", version_base=None)
@log_exception_handler
def run_with_hydra(cfg: DictConfig) -> None:
    """Main entry point for the pipeline using Hydra with enhanced error handling.
    
    Args:
        cfg: Configuration from Hydra
    """
    # Start timing
    start_time = datetime.datetime.now()
    
    # Print raw config for debugging
    logger.info("Raw configuration received:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Validate configuration
    try:
        # Convert OmegaConf to dict for validation
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        validated_cfg = validate_config(cfg_dict)
        
        # Convert back to OmegaConf
        cfg = OmegaConf.create(validated_cfg)
        logger.info("Configuration successfully validated")
    except Exception as e:
        logger.critical(f"Configuration validation failed: {str(e)}", error_code=ERROR_CODES['CONFIG_ERROR'])
        logger.info("Attempting to continue with original configuration...")
    
    # Create run ID and metadata
    run_id = str(uuid.uuid4())
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Initialize path manager
    path_manager = get_path_manager(cfg.paths.base_dir)
    
    # Set up output directories
    output_dir = Path(cfg.paths.output_dir)
    run_output_dir = output_dir / f"run_{run_timestamp}_{run_id[:8]}"
    logs_dir = run_output_dir / "logs"
    models_dir = run_output_dir / "models"
    eval_dir = run_output_dir / "evaluation"
    
    # Create directories
    for dir_path in [run_output_dir, logs_dir, models_dir, eval_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Add file handler to logger for this run
    file_handler = logging.FileHandler(logs_dir / "pipeline.log")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.logger.addHandler(file_handler)
    
    # Log run information
    logger.info(f"Starting pipeline run {run_id} at {run_timestamp}")
    logger.info(f"Output directory: {run_output_dir}")
    
    # Save configuration
    config_path = run_output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Configuration saved to {config_path}")
    
    # Log initial memory usage
    mem_stats = memory_stats()
    logger.info(f"Initial memory usage: {mem_stats['rss_mb']:.2f} MB, System available: {mem_stats['system_available_gb']:.2f} GB")
    
    # Execute requested steps
    steps = cfg.pipeline.steps
    step_results = {}
    
    try:
        # STEP 1: Extract data if requested
        if "extract" in steps:
            logger.info("STEP 1: EXTRACTING DATA")
            
            if cfg.data.get('zip_file') and os.path.exists(cfg.data.zip_file):
                extract_dir = extract_zip(cfg.data.zip_file, cfg.paths.data_dir)
                logger.info(f"Extracted data to {extract_dir}")
                step_results['extract'] = {'status': 'success', 'output_dir': extract_dir}
            else:
                logger.warning("No zip file specified or file does not exist, skipping extraction")
                step_results['extract'] = {'status': 'skipped'}
        
        # STEP 2: Load data if requested
        consolidated_df = None
        if "load" in steps:
            logger.info("STEP 2: LOADING DATA")
            
            # Validate dataset structure before loading
            validator = DataValidator(path_manager)
            dataset_validation = {}
            for dataset_name in ["ORIGA", "REFUGE", "G1020"]:
                validation_result = validator.validate_dataset_structure(dataset_name)
                dataset_validation[dataset_name] = validation_result
                
                if validation_result['valid']:
                    logger.info(f"Dataset {dataset_name} structure validated successfully")
                else:
                    logger.warning(f"Dataset {dataset_name} validation issues: {validation_result['issues']}")
            
            # Load data
            consolidated_df = consolidate_datasets(cfg.paths.data_dir)
            
            if consolidated_df.empty:
                logger.error("Failed to load any data", error_code=ERROR_CODES['DATA_LOAD_ERROR'])
                step_results['load'] = {'status': 'failed', 'error': 'No data loaded'}
                raise ValueError("No data loaded from datasets")
            
            # Save consolidated dataset
            consolidated_csv = run_output_dir / "consolidated_dataset.csv"
            save_consolidated_dataset(consolidated_df, str(consolidated_csv))
            logger.info(f"Saved consolidated dataset with {len(consolidated_df)} samples to {consolidated_csv}")
            
            # Validate the loaded dataset
            df_validation = validator.validate_dataframe(consolidated_df, mode=cfg.preprocessing.mode)
            
            if not df_validation['valid']:
                logger.warning(f"Dataset validation issues: {df_validation['issues']}")
                if len(df_validation['issues']) > 0:
                    logger.warning("Attempting to continue despite validation issues")
            
            # Generate and save validation report
            validation_report = validator.generate_validation_report(
                consolidated_df, 
                output_path=run_output_dir / "dataset_validation.json"
            )
            
            step_results['load'] = {
                'status': 'success', 
                'samples': len(consolidated_df),
                'validation': df_validation['valid']
            }
        else:
            # Try to load existing consolidated dataset
            consolidated_csv = output_dir / "consolidated_dataset.csv"
            if consolidated_csv.exists():
                consolidated_df = pd.read_csv(consolidated_csv)
                logger.info(f"Loaded existing consolidated dataset from {consolidated_csv}")
                step_results['load'] = {'status': 'loaded_existing', 'samples': len(consolidated_df)}
            else:
                logger.error("No consolidated dataset found and 'load' step not included", 
                            error_code=ERROR_CODES['DATA_LOAD_ERROR'])
                step_results['load'] = {'status': 'failed', 'error': 'No dataset available'}
                raise FileNotFoundError(f"No consolidated dataset found at {consolidated_csv}")
        
        # STEP 3: Clean data if requested
        cleaned_df = None
        if "clean" in steps:
            logger.info("STEP 3: CLEANING DATA")
            
            if consolidated_df is None:
                logger.error("Cannot clean data: No consolidated dataset available", 
                            error_code=ERROR_CODES['DATA_CLEAN_ERROR'])
                step_results['clean'] = {'status': 'failed', 'error': 'No dataset available'}
                raise ValueError("No consolidated dataset available for cleaning")
            
            # Clean the dataset
            cleaned_df = clean_dataset(consolidated_df, random_state=cfg.data.random_state)
            
            # Save cleaned dataset
            cleaned_csv = run_output_dir / "cleaned_dataset.csv"
            cleaned_df.to_csv(cleaned_csv, index=False)
            logger.info(f"Saved cleaned dataset with {len(cleaned_df)} samples to {cleaned_csv}")
            
            # Validate the cleaned dataset
            validator = DataValidator(path_manager)
            df_validation = validator.validate_dataframe(cleaned_df, mode=cfg.preprocessing.mode)
            
            if not df_validation['valid']:
                logger.warning(f"Cleaned dataset validation issues: {df_validation['issues']}")
            
            step_results['clean'] = {
                'status': 'success', 
                'samples_before': len(consolidated_df),
                'samples_after': len(cleaned_df),
                'validation': df_validation['valid']
            }
        else:
            # Try to load existing cleaned dataset
            cleaned_csv = output_dir / "cleaned_dataset.csv"
            if cleaned_csv.exists():
                cleaned_df = pd.read_csv(cleaned_csv)
                logger.info(f"Loaded existing cleaned dataset from {cleaned_csv}")
                step_results['clean'] = {'status': 'loaded_existing', 'samples': len(cleaned_df)}
            else:
                logger.info("No cleaned dataset found, using consolidated dataset")
                cleaned_df = consolidated_df
                step_results['clean'] = {'status': 'using_consolidated'}
        
        # STEP 4: Preprocess data
        if "preprocess" in steps:
            logger.info("STEP 4: PREPROCESSING DATA")
            
            if cleaned_df is None:
                logger.error("Cannot preprocess data: No dataset available", 
                            error_code=ERROR_CODES['PREPROCESSING_ERROR'])
                step_results['preprocess'] = {'status': 'failed', 'error': 'No dataset available'}
                raise ValueError("No dataset available for preprocessing")
            
            # Use memory-efficient data loading for large datasets
            if len(cleaned_df) > 1000:
                logger.info("Using memory-efficient data loading for large dataset")
                
                # Create cache directory
                cache_dir = run_output_dir / "cache"
                cache_dir.mkdir(exist_ok=True)
                
                # Create data module with memory-efficient loading
                data_module = GlaucomaDataModule(
                    data_df=cleaned_df,
                    batch_size=cfg.training.batch_size,
                    num_workers=cfg.training.num_workers,
                    target_size=tuple(cfg.preprocessing.image_size),
                    augment_train=cfg.preprocessing.augmentation.enabled,
                    mode=cfg.preprocessing.mode,
                    val_split=cfg.data.split_config.val_ratio,
                    test_split=cfg.data.split_config.test_ratio,
                    random_state=cfg.data.random_state,
                    use_memory_efficient=True,  # Enable memory-efficient loading
                    cache_dir=str(cache_dir)    # Use disk caching
                )
            else:
                # Use standard data module for smaller datasets
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
            
            # Save train, val, test splits
            data_module.train_df.to_csv(run_output_dir / "train_dataset.csv", index=False)
            data_module.val_df.to_csv(run_output_dir / "val_dataset.csv", index=False)
            data_module.test_df.to_csv(run_output_dir / "test_dataset.csv", index=False)
            
            logger.info(f"Dataset split: Train: {len(data_module.train_df)}, "
                       f"Val: {len(data_module.val_df)}, Test: {len(data_module.test_df)}")
            
            step_results['preprocess'] = {
                'status': 'success',
                'train_samples': len(data_module.train_df),
                'val_samples': len(data_module.val_df),
                'test_samples': len(data_module.test_df),
                'image_size': cfg.preprocessing.image_size
            }
        else:
            # Create data module from existing splits
            logger.info("Creating data module from existing splits")
            
            # Check for split files
            train_csv = output_dir / "train_dataset.csv"
            val_csv = output_dir / "val_dataset.csv"
            test_csv = output_dir / "test_dataset.csv"
            
            if all(f.exists() for f in [train_csv, val_csv, test_csv]):
                try:
                    # Load existing splits
                    train_df = pd.read_csv(train_csv)
                    val_df = pd.read_csv(val_csv)
                    test_df = pd.read_csv(test_csv)
                    
                    # Create data module
                    data_module = GlaucomaDataModule(
                        data_df=cleaned_df,  # Pass the full dataset
                        batch_size=cfg.training.batch_size,
                        num_workers=cfg.training.num_workers,
                        target_size=tuple(cfg.preprocessing.image_size),
                        augment_train=cfg.preprocessing.augmentation.enabled,
                        mode=cfg.preprocessing.mode,
                        use_existing_splits=True,  # Use existing splits
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df
                    )
                    
                    # Set up data module
                    data_module.setup("fit")
                    
                    logger.info(f"Using existing splits: Train: {len(train_df)}, "
                               f"Val: {len(val_df)}, Test: {len(test_df)}")
                    
                    step_results['preprocess'] = {
                        'status': 'loaded_existing',
                        'train_samples': len(train_df),
                        'val_samples': len(val_df),
                        'test_samples': len(test_df)
                    }
                except Exception as e:
                    logger.error(f"Error loading existing splits: {str(e)}", 
                                error_code=ERROR_CODES['PREPROCESSING_ERROR'])
                    
                    # Fall back to creating new splits
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
                    
                    step_results['preprocess'] = {
                        'status': 'fallback_created',
                        'train_samples': len(data_module.train_df),
                        'val_samples': len(data_module.val_df),
                        'test_samples': len(data_module.test_df)
                    }
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
                
                # Set up data module
                data_module.setup()
                
                step_results['preprocess'] = {
                    'status': 'created_new',
                    'train_samples': len(data_module.train_df),
                    'val_samples': len(data_module.val_df),
                    'test_samples': len(data_module.test_df)
                }
        
        # STEP 5: Train model if requested
        if "train" in steps:
            logger.info("STEP 5: TRAINING MODEL")
            
            # Initialize wandb if enabled
            if cfg.logging.use_wandb:
                import wandb
                wandb.init(
                    project=cfg.logging.wandb_project,
                    name=f"run_{run_timestamp}_{run_id[:8]}",
                    config=OmegaConf.to_container(cfg, resolve=True),
                    dir=str(run_output_dir)
                )
            
            # Create model
            logger.info(f"Creating model: {cfg.model.architecture} with {cfg.model.encoder} encoder")
            model = create_model(cfg.model)
            
            # Train model
            logger.info(f"Starting training for {cfg.training.epochs} epochs")
            lightning_model, trainer = train_model(
                model=model,
                data_module=data_module,
                config=cfg.training,
                output_dir=str(run_output_dir)
            )
            
            # Save final model
            final_model_path = models_dir / "final_model.pt"
            save_model(lightning_model.model, str(final_model_path), config=cfg.model)
            logger.info(f"Saved final model to {final_model_path}")
            
            # Close wandb
            if cfg.logging.use_wandb:
                wandb.finish()
            
            # Collect garbage to free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            step_results['train'] = {
                'status': 'success',
                'epochs': cfg.training.epochs,
                'final_model_path': str(final_model_path)
            }
        else:
            step_results['train'] = {'status': 'skipped'}
        
        # STEP 6: Evaluate model if requested
        if "evaluate" in steps:
            logger.info("STEP 6: EVALUATING MODEL")
            
            # Check if training was done in this run
            if step_results.get('train', {}).get('status') == 'success':
                model_path = step_results['train']['final_model_path']
                logger.info(f"Using model from this run: {model_path}")
            else:
                # Look for existing model
                model_path = models_dir / "final_model.pt"
                if not model_path.exists():
                    # Try finding any model checkpoint
                    checkpoints = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.ckpt"))
                    if checkpoints:
                        model_path = checkpoints[0]
                        logger.info(f"Using found model checkpoint: {model_path}")
                    else:
                        logger.error("No model found for evaluation", error_code=ERROR_CODES['MODEL_ERROR'])
                        step_results['evaluate'] = {'status': 'failed', 'error': 'No model found'}
                        raise FileNotFoundError("No model found for evaluation")
            
            # Load model
            try:
                if str(model_path).endswith('.pt'):
                    # Load from our saved format
                    model, _ = load_model(str(model_path), cfg.model)
                    lightning_model = GlaucomaSegmentationModel(model, cfg.training)
                else:
                    # Load from Lightning checkpoint
                    model = create_model(cfg.model)
                    lightning_model = GlaucomaSegmentationModel.load_from_checkpoint(
                        checkpoint_path=str(model_path),
                        model=model,
                        config=cfg.training
                    )
                
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}", error_code=ERROR_CODES['MODEL_ERROR'])
                step_results['evaluate'] = {'status': 'failed', 'error': f'Error loading model: {str(e)}'}
                raise
            
            # Initialize evaluator
            evaluator = SegmentationEvaluator(
                model=lightning_model,
                output_dir=str(eval_dir)
            )
            
            # Evaluate model
            test_loader = data_module.test_dataloader()
            results = evaluator.evaluate(
                test_loader=test_loader,
                device="cuda" if torch.cuda.is_available() and cfg.training.use_gpu else "cpu"
            )
            
            logger.info(f"Evaluation results: {results}")
            
            step_results['evaluate'] = {
                'status': 'success',
                'metrics': results,
                'visualization_dir': str(eval_dir / 'visualizations')
            }
        else:
            step_results['evaluate'] = {'status': 'skipped'}
        
        # Calculate run time
        end_time = datetime.datetime.now()
        run_time = end_time - start_time
        
        # Update run information
        run_info = {
            "id": run_id,
            "timestamp": run_timestamp,
            "steps": steps,
            "step_results": step_results,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "run_time": str(run_time),
            "status": "completed"
        }
        
        # Save run info
        import json
        with open(run_output_dir / "run_info.json", 'w') as f:
            json.dump(run_info, f, indent=2, default=str)
        
        # Update notebook.md
        notebook_file = output_dir / "notebook.md"
        if not notebook_file.exists():
            with open(notebook_file, 'w') as f:
                f.write("# Pipeline Run Tracking\n\n")
                f.write("| Run ID | Timestamp | Steps | Status | Description | Duration |\n")
                f.write("|--------|-----------|-------|--------|-------------|----------|\n")
        
        with open(notebook_file, 'a') as f:
            description = cfg.pipeline.description if hasattr(cfg.pipeline, "description") else ""
            steps_str = ", ".join(steps)
            f.write(f"| {run_id[:8]} | {run_timestamp} | {steps_str} | completed | {description} | {run_time} |\n")
        
        logger.info(f"Pipeline run {run_id} completed successfully in {run_time}")
        
    except Exception as e:
        # Handle any unhandled exceptions
        end_time = datetime.datetime.now()
        run_time = end_time - start_time
        
        logger.critical(
            f"Pipeline failed with error: {str(e)}", 
            error_code=ERROR_CODES['UNKNOWN_ERROR'],
            exc_info=True
        )
        
        # Save run info with error
        run_info = {
            "id": run_id,
            "timestamp": run_timestamp,
            "steps": steps,
            "step_results": step_results,
            "config": OmegaConf.to_container(cfg, resolve=True) if cfg else {},
            "run_time": str(run_time),
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        
        # Create output directory if it doesn't exist yet
        run_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save run info
        import json
        with open(run_output_dir / "run_info.json", 'w') as f:
            json.dump(run_info, f, indent=2, default=str)
        
        # Update notebook.md
        notebook_file = output_dir / "notebook.md"
        if not notebook_file.exists():
            with open(notebook_file, 'w') as f:
                f.write("# Pipeline Run Tracking\n\n")
                f.write("| Run ID | Timestamp | Steps | Status | Description | Duration |\n")
                f.write("|--------|-----------|-------|--------|-------------|----------|\n")
        
        with open(notebook_file, 'a') as f:
            description = cfg.pipeline.description if hasattr(cfg.pipeline, "description") else ""
            steps_str = ", ".join(steps)
            f.write(f"| {run_id[:8]} | {run_timestamp} | {steps_str} | failed | {description} | {run_time} |\n")
        
        # Re-raise the exception for Hydra
        raise

# Define Typer CLI command that uses Hydra under the hood
@app.command()
def run(
    config_path: str = typer.Option(None, "--config", "-c", help="Path to config file"),
    steps: str = typer.Option("extract,load,clean,preprocess,train,evaluate", "--steps", "-s", 
                             help="Comma-separated list of steps to run"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    data_dir: str = typer.Option(None, "--data-dir", "-d", help="Data directory"),
    zip_file: str = typer.Option(None, "--zip-file", "-z", help="Path to ZIP file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force rerun of all steps"),
    description: str = typer.Option("", "--description", help="Run description"),
    wandb_logging: bool = typer.Option(False, "--wandb", help="Enable Weights & Biases logging"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode with detailed logging")
):
    """Run the pipeline with specified steps and enhanced error handling."""
    # Configure logging level
    if debug:
        logger.logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    # Build command-line arguments for hydra
    argv = []
    
    # Add explicit overrides
    if steps:
        step_list = steps.split(',')
        argv.append(f"pipeline.steps={step_list}")
    
    if output_dir:
        argv.append(f"paths.output_dir={output_dir}")
    
    if data_dir:
        argv.append(f"paths.data_dir={data_dir}")
    
    if zip_file:
        argv.append(f"data.zip_file={zip_file}")
    
    if force:
        argv.append("pipeline.force=true")
    
    if description:
        argv.append(f"pipeline.description='{description}'")
    
    if wandb_logging:
        argv.append("logging.use_wandb=true")
    
    # Handle custom config file
    if config_path:
        # Use the provided config file
        argv.append(f"--config-path={os.path.dirname(config_path)}")
        argv.append(f"--config-name={os.path.basename(config_path).split('.')[0]}")
    
    # Call hydra's main function with our arguments
    try:
        run_with_hydra(argv)
    except Exception as e:
        logger.critical(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)

# Add additional CLI commands
@app.command()
def validate_data(
    data_dir: str = typer.Option("./data", "--data-dir", "-d", help="Data directory"),
    output_file: str = typer.Option("./validation_report.json", "--output", "-o", help="Output file")
):
    """Validate the datasets and generate a report."""
    try:
        # Initialize path manager
        path_manager = get_path_manager(os.path.dirname(data_dir))
        
        # Initialize validator
        validator = DataValidator(path_manager)
        
        # Validate each dataset
        results = {}
        for dataset_name in ["ORIGA", "REFUGE", "G1020"]:
            logger.info(f"Validating {dataset_name} dataset structure")
            results[dataset_name] = validator.validate_dataset_structure(dataset_name)
        
        # Load consolidated dataset if available
        consolidated_path = os.path.join(os.path.dirname(data_dir), "output", "consolidated_dataset.csv")
        if os.path.exists(consolidated_path):
            logger.info(f"Loading consolidated dataset from {consolidated_path}")
            df = pd.read_csv(consolidated_path)
            
            # Validate dataframe
            results["consolidated"] = validator.validate_dataframe(df)
            
            # Generate full validation report
            report = validator.generate_validation_report(df, output_path=output_file)
            logger.info(f"Validation report saved to {output_file}")
        else:
            # Save basic results
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Basic validation results saved to {output_file}")
        
        logger.info("Data validation completed")
        
    except Exception as e:
        logger.critical(f"Data validation failed: {e}", exc_info=True)
        sys.exit(1)

@app.command()
def generate_docs(
    output_dir: str = typer.Option("./docs", "--output-dir", "-o", help="Output directory")
):
    """Generate configuration documentation."""
    try:
        from glaucoma_detection.config_validator import ConfigValidator
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize config validator
        validator = ConfigValidator()
        
        # Generate Markdown documentation
        md_path = output_dir / "config_schema.md"
        validator.generate_schema_documentation(md_path)
        logger.info(f"Configuration schema documentation saved to {md_path}")
        
        # Export schema as JSON
        json_path = output_dir / "config_schema.json"
        validator.export_schema_as_json(json_path)
        logger.info(f"Configuration schema exported to {json_path}")
        
        # Generate example configuration
        example_config = validator.fill_defaults({})
        example_config_path = output_dir / "example_config.yaml"
        
        with open(example_config_path, 'w') as f:
            import yaml
            yaml.dump(example_config, f, default_flow_style=False)
        
        logger.info(f"Example configuration saved to {example_config_path}")
        logger.info("Documentation generation completed")
        
    except Exception as e:
        logger.critical(f"Documentation generation failed: {e}", exc_info=True)
        sys.exit(1)

@app.command()
def verify_environment():
    """Verify the environment and dependencies."""
    try:
        logger.info("Verifying environment and dependencies")
        
        # Check Python version
        import platform
        python_version = platform.python_version()
        logger.info(f"Python version: {python_version}")
        
        # Check PyTorch version and CUDA availability
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "Not available"
        
        logger.info(f"PyTorch version: {torch_version}")
        logger.info(f"CUDA available: {cuda_available}")
        logger.info(f"CUDA version: {cuda_version}")
        
        # Check other key dependencies - FIXED VERSION
        dependencies = [
            ("pandas", "pandas"),
            ("numpy", "numpy"),
            ("sklearn", "sklearn"),
            ("cv2", "cv2"),
            ("albumentations", "albumentations"),
            ("segmentation_models_pytorch", "segmentation_models_pytorch"),
            ("pytorch_lightning", "pytorch_lightning"),
            ("hydra", "hydra"),
            ("omegaconf", "omegaconf"),
            ("typer", "typer"),
            ("wandb", "wandb")
        ]
        
        for name, module_name in dependencies:
            try:
                module = __import__(module_name)
                version = getattr(module, "__version__", "unknown")
                logger.info(f"{name} version: {version}")
            except ImportError:
                logger.warning(f"{name} is not installed")
        
        # Check memory
        mem = memory_stats()
        logger.info(f"Memory usage: {mem['rss_mb']:.2f} MB")
        logger.info(f"Available system memory: {mem['system_available_gb']:.2f} GB")
        
        # Check path setup
        path_manager = get_path_manager()
        logger.info(f"Base directory: {path_manager.base_dir}")
        logger.info(f"Data directory: {path_manager.data_dir}")
        logger.info(f"Output directory: {path_manager.output_dir}")
        
        # Check dataset availability
        dataset_info = path_manager.get_dataset_info()
        for dataset_name, info in dataset_info.items():
            status = "Available" if info["exists"] else "Not found"
            logger.info(f"{dataset_name} dataset: {status}")
            if info["exists"]:
                logger.info(f"  - Files: {info['num_files']}")
                logger.info(f"  - Path: {info['path']}")
        
        logger.info("Environment verification completed successfully")
        
    except Exception as e:
        logger.critical(f"Environment verification failed: {e}", exc_info=True)
        sys.exit(1)

@app.command()
def clean_cache(
    force: bool = typer.Option(False, "--force", "-f", help="Force remove without confirmation")
):
    """Clean cache files and temporary artifacts."""
    try:
        logger.info("Cleaning cache files and temporary artifacts")
        
        # Initialize path manager
        path_manager = get_path_manager()
        
        # Define paths to clean
        cache_paths = [
            path_manager.output_dir / "cache",
            path_manager.base_dir / "__pycache__",
            path_manager.base_dir / ".hydra"
        ]
        
        # Add any run-specific cache directories
        for run_dir in path_manager.output_dir.glob("run_*"):
            cache_dir = run_dir / "cache"
            if cache_dir.exists():
                cache_paths.append(cache_dir)
        
        # Count total cache size
        total_size = 0
        for path in cache_paths:
            if path.exists():
                if path.is_dir():
                    for p in path.glob("**/*"):
                        if p.is_file():
                            total_size += p.stat().st_size
                else:
                    total_size += path.stat().st_size
        
        # Convert to MB
        total_size_mb = total_size / (1024 * 1024)
        logger.info(f"Found {len(cache_paths)} cache directories with total size: {total_size_mb:.2f} MB")
        
        # Get confirmation if not forced
        if not force:
            import typer
            proceed = typer.confirm(f"Remove {len(cache_paths)} cache directories ({total_size_mb:.2f} MB)?")
            if not proceed:
                logger.info("Cache cleaning aborted")
                return
        
        # Remove cache directories
        removed_count = 0
        for path in cache_paths:
            if path.exists():
                if path.is_dir():
                    import shutil
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed_count += 1
                logger.info(f"Removed: {path}")
        
        logger.info(f"Cleaned {removed_count} cache directories, freed {total_size_mb:.2f} MB")
        
    except Exception as e:
        logger.critical(f"Cache cleaning failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Set Hydra environment variables
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Show full error traces
    
    # Run the Typer app
    app()