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

def preprocess_dataset(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Preprocess the dataset."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a copy of the input dataframe
    preprocessed_df = df.copy()
    
    # Validate image paths
    preprocessed_df = preprocessed_df[preprocessed_df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    
    # Add mode column if not present
    if 'mode' not in preprocessed_df.columns:
        # If mask_path exists, use segmentation, otherwise classification
        preprocessed_df['mode'] = 'classification'
        if 'mask_path' in preprocessed_df.columns:
            mask_exists = preprocessed_df['mask_path'].apply(lambda x: x is not None and os.path.exists(x))
            preprocessed_df.loc[mask_exists, 'mode'] = 'segmentation'
    
    # Set default target size
    target_size = (224, 224)
    
    # Save metadata about preprocessing
    metadata = {
        'num_samples': len(preprocessed_df),
        'target_size': target_size,
        'preprocessing_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save metadata to file
    with open(os.path.join(output_dir, 'preprocessing_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return preprocessed_df

def create_dataset_splits(df: pd.DataFrame, 
                         val_size: float = 0.15, 
                         test_size: float = 0.15, 
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train, validation, and test splits from the dataset."""
    # Check if split column already exists
    if 'split' in df.columns:
        # Use existing splits
        train_df = df[df['split'] == 'train'].copy()
        val_df = df[df['split'] == 'val'].copy()
        test_df = df[df['split'] == 'test'].copy()
        
        logger.info(f"Using existing splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df
    
    # Create new splits
    from sklearn.model_selection import train_test_split
    
    # First, split off test set
    if 'label' in df.columns and not df['label'].isna().any():
        # Use stratified split if labels are available
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size,
            random_state=random_state,
            stratify=df['label']
        )
        
        # Adjust val_size for the remaining data
        val_size_adjusted = val_size / (1 - test_size)
        
        # Split remaining data into train and validation
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val_df['label'] if 'label' in train_val_df.columns else None
        )
    else:
        # Use regular split if no labels
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size,
            random_state=random_state
        )
        
        # Adjust val_size for the remaining data
        val_size_adjusted = val_size / (1 - test_size)
        
        # Split remaining data into train and validation
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state
        )
    
    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    logger.info(f"Created new splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df

def create_dataloaders(train_df: pd.DataFrame,
                      val_df: pd.DataFrame, 
                      test_df: pd.DataFrame = None,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      target_size: Tuple[int, int] = (224, 224),
                      augment_train: bool = True,
                      mode: str = 'segmentation') -> Tuple:
    """Create data loaders for training, validation, and testing."""
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = GlaucomaDataset(
        data=train_df,
        target_size=target_size,
        augment=augment_train,
        mode=mode
    )
    
    val_dataset = GlaucomaDataset(
        data=val_df,
        target_size=target_size,
        augment=False,  # No augmentation for validation
        mode=mode
    )
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create test loader if test data is provided
    test_loader = None
    if test_df is not None and len(test_df) > 0:
        test_dataset = GlaucomaDataset(
            data=test_df,
            target_size=target_size,
            augment=False,  # No augmentation for testing
            mode=mode
        )
        
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader

def get_optimizer(model, config):
    """Get the optimizer based on configuration."""
    import torch.optim as optim
    
    optimizer_type = config.get('optimizer', 'adam').lower()
    lr = config.get('learning_rate', 0.001)
    
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr)
    else:
        logger.warning(f"Unknown optimizer: {optimizer_type}. Using Adam.")
        return optim.Adam(model.parameters(), lr=lr)

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
    # Update the training_config dict in your test_model_run function
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
            'enabled': False,
            'patience': 10,  # Add this line
            'min_delta': 0.001  # Also add this parameter
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