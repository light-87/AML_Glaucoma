"""
Training Module

This module handles the training process for the glaucoma detection models.
It implements functions for training, validation, and early stopping.

Functions:
- train_epoch(): Train model for one epoch
- validate(): Validate model on validation set
- train_model(): Train model for multiple epochs with early stopping
"""

import os
import time
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import from other modules
from model import (
    create_model, get_optimizer, get_loss_function, 
    evaluate_metrics, save_model, load_model
)
from config import TRAINING_CONFIG, MODEL_CONFIG
from utils import setup_logger, create_directory

# Set up logging
logger = setup_logger('trainer')

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """
    Train model for one epoch.
    
    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model
    dataloader : torch.utils.data.DataLoader
        Training data loader
    optimizer : torch.optim.Optimizer
        Optimizer
    loss_fn : function
        Loss function
    device : str
        Device to train on ('cuda' or 'cpu')
        
    Returns:
    --------
    tuple
        (epoch_loss, metrics)
    """
    model.train()
    running_loss = 0.0
    
    # Store predictions and targets for metrics calculation
    all_predictions = []
    all_targets = []
    
    # Progress bar
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        # Move data to device
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = loss_fn(outputs, masks)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())
        
        # Store predictions and targets for metrics calculation
        all_predictions.append(outputs.detach().cpu())
        all_targets.append(masks.detach().cpu())
    
    # Calculate average loss
    epoch_loss = running_loss / len(dataloader)
    
    # Calculate metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = evaluate_metrics(all_predictions, all_targets)
    
    return epoch_loss, metrics

def validate(model, dataloader, loss_fn, device):
    """
    Validate model on validation set.
    
    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model
    dataloader : torch.utils.data.DataLoader
        Validation data loader
    loss_fn : function
        Loss function
    device : str
        Device to validate on ('cuda' or 'cpu')
        
    Returns:
    --------
    tuple
        (val_loss, metrics)
    """
    model.eval()
    running_loss = 0.0
    
    # Store predictions and targets for metrics calculation
    all_predictions = []
    all_targets = []
    
    # No gradient calculation for validation
    with torch.no_grad():
        # Progress bar
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            # Move data to device
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = loss_fn(outputs, masks)
            
            # Update running loss
            running_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())
            
            # Store predictions and targets for metrics calculation
            all_predictions.append(outputs.detach().cpu())
            all_targets.append(masks.detach().cpu())
    
    # Calculate average loss
    val_loss = running_loss / len(dataloader)
    
    # Calculate metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = evaluate_metrics(all_predictions, all_targets)
    
    return val_loss, metrics

def train_model(train_loader, val_loader, model=None, config=None, 
               checkpoint_dir=None, device=None):
    """
    Train model for multiple epochs with early stopping.
    
    Parameters:
    -----------
    train_loader : torch.utils.data.DataLoader
        Training data loader
    val_loader : torch.utils.data.DataLoader
        Validation data loader
    model : torch.nn.Module, optional
        PyTorch model, by default None
    config : dict, optional
        Training configuration, by default None
    checkpoint_dir : str, optional
        Directory to save checkpoints, by default None
    device : str, optional
        Device to train on ('cuda' or 'cpu'), by default None
        
    Returns:
    --------
    tuple
        (trained_model, history, best_epoch)
    """
    # Set default configuration
    if config is None:
        config = TRAINING_CONFIG
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
    
    # Set checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    
    # Create checkpoint directory if it doesn't exist
    create_directory(checkpoint_dir)
    
    # Create model if not provided
    if model is None:
        model = create_model()
    
    # Move model to device
    model = model.to(device)
    
    # Get optimizer
    optimizer = get_optimizer(
        model, 
        lr=config['learning_rate'],
        optimizer_type=config['optimizer']
    )
    
    # Get loss function
    loss_fn = get_loss_function(config['loss_function'])
    
    # Learning rate scheduler
    scheduler = None
    if config['lr_scheduler']['enabled']:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min' if config['lr_scheduler']['monitor'] == 'val_loss' else 'max',
            factor=config['lr_scheduler']['factor'],
            patience=config['lr_scheduler']['patience'],
            min_lr=config['lr_scheduler']['min_lr'],
            verbose=True
        )
    
    # Initialize variables for training
    start_epoch = 0
    epochs = config['epochs']
    best_val_loss = float('inf')
    best_val_metric = -float('inf')
    best_epoch = 0
    early_stopping_counter = 0
    early_stopping_patience = config['early_stopping']['patience']
    early_stopping_min_delta = config['early_stopping']['min_delta']
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': [],
        'learning_rates': []
    }
    
    # Start training
    logger.info(f"Starting training for {epochs} epochs on {device}")
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        # Train one epoch
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, loss_fn, device
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        history['learning_rates'].append(current_lr)
        
        # Print epoch results
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Train Dice: {train_metrics['dice']:.4f}, "
                    f"Val Dice: {val_metrics['dice']:.4f}, "
                    f"LR: {current_lr:.6f}")
        
        # Update scheduler
        if scheduler is not None:
            if config['lr_scheduler']['monitor'] == 'val_loss':
                scheduler.step(val_loss)
            else:
                scheduler.step(val_metrics['dice'])
        
        # Check if this is the best model based on validation loss
        if val_loss < best_val_loss - early_stopping_min_delta:
            logger.info(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            best_val_loss = val_loss
            best_epoch = epoch
            early_stopping_counter = 0
            
            # Save best model
            if config['checkpointing']['enabled'] and config['checkpointing']['save_best_only']:
                model_name = config.get('model_name', 'model')
                best_model_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
                save_model(
                    model, 
                    best_model_path, 
                    optimizer,
                    epoch,
                    val_loss,
                    val_metrics
                )
        else:
            early_stopping_counter += 1
            logger.info(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
        
        # Check if alternate metric (Dice) is better
        monitor_metric = 'dice'  # Can be configured
        if val_metrics[monitor_metric] > best_val_metric:
            logger.info(f"Validation {monitor_metric} improved from {best_val_metric:.4f} to {val_metrics[monitor_metric]:.4f}")
            best_val_metric = val_metrics[monitor_metric]
            
            # Save model with best metric
            if config['checkpointing']['enabled']:
                model_name = config.get('model_name', 'model')
                best_metric_path = os.path.join(checkpoint_dir, f"{model_name}_best_{monitor_metric}.pth")
                save_model(
                    model,
                    best_metric_path,
                    optimizer,
                    epoch,
                    val_loss,
                    val_metrics
                )
        
        # Early stopping
        if config['early_stopping']['enabled'] and early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint for every epoch if not save_best_only
        if config['checkpointing']['enabled'] and not config['checkpointing']['save_best_only']:
            model_name = config.get('model_name', 'model')
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch+1}.pth")
            save_model(
                model,
                checkpoint_path,
                optimizer,
                epoch,
                val_loss,
                val_metrics
            )
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Best epoch: {best_epoch+1} with validation loss: {best_val_loss:.4f}")
    
    # Plot training history
    plot_training_history(history, checkpoint_dir)
    
    return model, history, best_epoch

def plot_training_history(history, output_dir):
    """
    Plot training history.
    
    Parameters:
    -----------
    history : dict
        Training history
    output_dir : str
        Directory to save plots
    """
    # Create output directory if it doesn't exist
    create_directory(output_dir)
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_history.png'))
    plt.close()
    
    # Plot Dice coefficient
    train_dice = [metrics['dice'] for metrics in history['train_metrics']]
    val_dice = [metrics['dice'] for metrics in history['val_metrics']]
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_dice, label='Train Dice')
    plt.plot(val_dice, label='Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Training and Validation Dice Coefficient')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'dice_history.png'))
    plt.close()
    
    # Plot learning rates
    plt.figure(figsize=(10, 5))
    plt.plot(history['learning_rates'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'lr_history.png'))
    plt.close()
    
    # Save history as CSV
    import pandas as pd
    
    df = pd.DataFrame({
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_dice': train_dice,
        'val_dice': val_dice,
        'learning_rate': history['learning_rates']
    })
    
    df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)

def main(train_csv, val_csv, config=None, checkpoint_dir=None):
    """
    Main function to train the model.
    
    Parameters:
    -----------
    train_csv : str
        Path to training CSV file
    val_csv : str
        Path to validation CSV file
    config : dict, optional
        Training configuration, by default None
    checkpoint_dir : str, optional
        Directory to save checkpoints, by default None
        
    Returns:
    --------
    tuple
        (trained_model, history, best_epoch)
    """
    import pandas as pd
    from preprocessor import create_dataloaders
    
    # Load training and validation dataframes
    try:
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return None, None, None
    
    # Create data loaders
    train_loader, val_loader, _ = create_dataloaders(
        train_df,
        val_df,
        batch_size=TRAINING_CONFIG['batch_size'],
        num_workers=TRAINING_CONFIG.get('num_workers', 4),
        mode='segmentation'
    )
    
    # Train model
    model, history, best_epoch = train_model(
        train_loader, 
        val_loader, 
        config=config,
        checkpoint_dir=checkpoint_dir
    )
    
    return model, history, best_epoch

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train glaucoma detection model')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to validation CSV file')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    
    args = parser.parse_args()
    
    main(args.train_csv, args.val_csv, checkpoint_dir=args.output_dir)