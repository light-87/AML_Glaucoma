import os
import sys
import torch
import pandas as pd
from pathlib import Path

# Add the current directory to sys.path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import pipeline modules
from data_loader import consolidate_datasets, save_consolidated_dataset
from data_cleaner import clean_dataset
from preprocessor import preprocess_dataset, create_dataset_splits, create_dataloaders
from model import create_model, get_optimizer, get_loss_function
from trainer import train_model
from utils import setup_logger, create_directory

# Set up logging
logger = setup_logger('test_script')

def test_model_run(data_path, output_dir, epochs=2, batch_size=8):
    """
    Run a test of the model pipeline for specified number of epochs.
    
    Parameters:
    -----------
    data_path : str
        Path to the data directory
    output_dir : str
        Directory to save outputs
    epochs : int, optional
        Number of epochs to train, by default 2
    batch_size : int, optional
        Batch size for training, by default 8
    """
    # Create output directory
    create_directory(output_dir)
    
    logger.info(f"Starting test run with data from {data_path}")
    logger.info(f"Output will be saved to {output_dir}")
    
    # Step 1: Load and consolidate data
    logger.info("Step 1: Loading and consolidating data...")
    consolidated_csv = os.path.join(output_dir, 'consolidated_glaucoma_dataset.csv')
    
    if os.path.exists(consolidated_csv):
        logger.info(f"Loading existing consolidated dataset from {consolidated_csv}")
        consolidated_df = pd.read_csv(consolidated_csv)
    else:
        logger.info(f"Consolidating datasets from {data_path}")
        consolidated_df = consolidate_datasets(data_path)
        save_consolidated_dataset(consolidated_df, consolidated_csv)
    
    # Step 2: Clean data
    logger.info("Step 2: Cleaning data...")
    cleaned_csv = os.path.join(output_dir, 'cleaned_glaucoma_dataset.csv')
    
    if os.path.exists(cleaned_csv):
        logger.info(f"Loading existing cleaned dataset from {cleaned_csv}")
        cleaned_df = pd.read_csv(cleaned_csv)
    else:
        logger.info("Cleaning dataset...")
        cleaned_df = clean_dataset(consolidated_df)
        cleaned_df.to_csv(cleaned_csv, index=False)
    
    # Step 3: Preprocess data
    logger.info("Step 3: Preprocessing data...")
    preprocessed_csv = os.path.join(output_dir, 'preprocessed_glaucoma_dataset.csv')
    train_csv = os.path.join(output_dir, 'train_dataset.csv')
    val_csv = os.path.join(output_dir, 'val_dataset.csv')
    test_csv = os.path.join(output_dir, 'test_dataset.csv')
    
    if (os.path.exists(preprocessed_csv) and os.path.exists(train_csv) 
        and os.path.exists(val_csv) and os.path.exists(test_csv)):
        logger.info("Loading existing preprocessed datasets...")
        preprocessed_df = pd.read_csv(preprocessed_csv)
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
    else:
        logger.info("Preprocessing dataset and creating splits...")
        preprocessed_df = preprocess_dataset(cleaned_df, output_dir)
        train_df, val_df, test_df = create_dataset_splits(
            preprocessed_df,
            val_size=0.15,
            test_size=0.15,
            random_state=42
        )
        
        # Save splits
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)
    
    # Step 4: Create data loaders
    logger.info("Step 4: Creating data loaders...")
    train_loader, val_loader, _ = create_dataloaders(
        train_df, 
        val_df, 
        batch_size=batch_size,
        num_workers=2,  # Reduced for testing
        mode='segmentation'
    )
    
    # Step 5: Create model
    logger.info("Step 5: Creating model...")
    model = create_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Step 6: Train for a few epochs
    logger.info(f"Step 6: Training model for {epochs} epochs...")
    checkpoint_dir = os.path.join(output_dir, 'models')
    create_directory(checkpoint_dir)
    
    # Configure training with reduced settings for testing
    training_config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss_function': 'combined',
        'lr_scheduler': {
            'enabled': False
        },
        'early_stopping': {
            'enabled': False
        },
        'checkpointing': {
            'enabled': True,
            'save_best_only': True
        },
        'model_name': 'glaucoma_test_model'
    }
    
    # Train model
    model, history, best_epoch = train_model(
        train_loader, 
        val_loader, 
        model=model,
        config=training_config,
        checkpoint_dir=checkpoint_dir,
        device=device
    )
    
    logger.info(f"Test run completed. Best epoch: {best_epoch + 1}")
    logger.info(f"Training history: {history}")
    
    return model, history, best_epoch

if __name__ == "__main__":
    # Data and output paths
    DATA_PATH = r"C:\Users\vaibh\Desktop\Surrey\AML\data"
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_test')
    
    # Run test with 2 epochs and smaller batch size
    test_model_run(DATA_PATH, OUTPUT_DIR, epochs=2, batch_size=8)