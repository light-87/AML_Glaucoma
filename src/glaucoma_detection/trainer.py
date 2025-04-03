"""
Modified Trainer Module with Fixed Loss Function

This version fixes the issue with combined loss functions.
"""

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from typing import Dict, Any, Tuple, Optional, List, Union
import logging
import wandb
import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, BinaryAccuracy
from torchmetrics.regression import MeanSquaredError

logger = logging.getLogger(__name__)

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="ShiftScaleRotate is a special case of Affine transform")
warnings.filterwarnings("ignore", message="Consider setting `persistent_workers=True`")
warnings.filterwarnings("ignore", message="The verbose parameter is deprecated")

# Custom Combined Loss Class
class CombinedLoss(nn.Module):
    def __init__(self, mode='binary'):
        super().__init__()
        self.dice_loss = smp.losses.DiceLoss(mode=mode)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, y_pred, y_true):
        # Check if predictions contain any non-zero values
        if torch.sum(y_pred > 0.5) == 0 and torch.rand(1).item() < 0.1:
            print("WARNING: All predictions are below threshold")
        
        # Ensure inputs have correct dimensions
        if y_pred.dim() == 4 and y_pred.size(1) == 1:
            # Reshape for BCE loss which expects [B, C]
            y_pred_bce = y_pred.squeeze(1)
            y_true_bce = y_true.squeeze(1)
        else:
            y_pred_bce = y_pred
            y_true_bce = y_true
            
        dice = self.dice_loss(y_pred, y_true)
        bce = self.bce_loss(y_pred_bce, y_true_bce)
        return dice + bce

class GlaucomaSegmentationModel(pl.LightningModule):
    """PyTorch Lightning module for glaucoma segmentation."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """Initialize the Lightning module."""
        super().__init__()
        self.model = model
        self.config = config
        
        # Save hyperparameters for reproducibility
        self.save_hyperparameters(config)
        
        # Set up loss function
        self.loss_fn = self._get_loss_function()
        
        # Set up metrics
        self.train_metrics = self._get_metrics()
        self.val_metrics = self._get_metrics()
        self.test_metrics = self._get_metrics()
    
    def _get_loss_function(self):
        """Get the loss function based on config."""
        loss_type = self.config.get('loss_function', 'combined')
        
        if loss_type == 'dice':
            return smp.losses.DiceLoss(mode='binary')
        elif loss_type == 'jaccard':
            return smp.losses.JaccardLoss(mode='binary')
        elif loss_type == 'focal':
            return smp.losses.FocalLoss(mode='binary')
        elif loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_type == 'combined':
            # Use our custom combined loss
            return CombinedLoss(mode='binary')
        else:
            logger.warning(f"Unknown loss function: {loss_type}. Using combined loss.")
            return CombinedLoss(mode='binary')
    
    def _get_metrics(self):
        """Get metrics for evaluation."""
        metrics = torchmetrics.MetricCollection({
            'iou': BinaryJaccardIndex(),  # IoU/Jaccard Index
            'dice': BinaryF1Score(),      # Dice coefficient is F1 for binary case
            'accuracy': BinaryAccuracy(),
            'mse': MeanSquaredError()
        })
        return metrics
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer_type = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 0.001)
        
        # Get optimizer
        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=lr)
        else:
            logger.warning(f"Unknown optimizer: {optimizer_type}. Using Adam.")
            optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Configure scheduler if enabled
        if self.config.get('lr_scheduler', {}).get('enabled', False):
            scheduler_config = self.config.get('lr_scheduler', {})
            # Remove verbose=True
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.1),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        return optimizer
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        images, masks = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        
        # Log loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Update and log metrics
        metrics = self.train_metrics(outputs, masks)
        self.log_dict({f'train_{k}': v for k, v in metrics.items()}, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, masks = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        
        # Log loss
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        
        # Update and log metrics
        metrics = self.val_metrics(outputs, masks)
        self.log_dict({f'val_{k}': v for k, v in metrics.items()}, on_epoch=True)
        
        # Log sample images at the end of validation
        if batch_idx == 0 and self.logger:
            self._log_images(images, masks, outputs)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        images, masks = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        
        # Log loss
        self.log('test_loss', loss, on_epoch=True)
        
        # Update and log metrics
        metrics = self.test_metrics(outputs, masks)
        self.log_dict({f'test_{k}': v for k, v in metrics.items()}, on_epoch=True)
        
        return loss
    
    def _log_images(self, images, masks, outputs, num_samples=4):
        """Log sample images to the logger."""
        if isinstance(self.logger, WandbLogger) and wandb.run is not None:
            num_samples = min(num_samples, images.size(0))
            images = images[:num_samples]
            masks = masks[:num_samples]
            outputs = outputs[:num_samples]
            
            # Convert to numpy for visualization
            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            outputs_np = (outputs > 0.5).float().cpu().numpy()
            
            # Log to wandb
            wandb.log({
                "sample_images": [
                    wandb.Image(
                        images_np[i].transpose(1, 2, 0),
                        masks={
                            "ground_truth": {"mask_data": masks_np[i, 0], "class_labels": {0: "background", 1: "glaucoma"}},
                            "prediction": {"mask_data": outputs_np[i, 0], "class_labels": {0: "background", 1: "glaucoma"}}
                        }
                    )
                    for i in range(num_samples)
                ]
            })

def setup_training(
    model: nn.Module,
    data_module: pl.LightningDataModule,
    config: Dict[str, Any],
    output_dir: str
) -> pl.Trainer:
    """Set up training with PyTorch Lightning."""
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join(output_dir, 'models')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up callbacks
    callbacks = []
    
    # Model checkpoint
    if config.get('checkpointing', {}).get('enabled', True):
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{config.get('model_name', 'glaucoma_model')}" + "-{epoch:02d}-{val_loss:.4f}",
            monitor=config.get('checkpointing', {}).get('monitor', 'val_loss'),
            mode=config.get('checkpointing', {}).get('mode', 'min'),
            save_top_k=config.get('checkpointing', {}).get('save_top_k', 3),
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
    
    # Early stopping
    if config.get('early_stopping', {}).get('enabled', True):
        early_stopping_callback = EarlyStopping(
            monitor=config.get('early_stopping', {}).get('monitor', 'val_loss'),
            patience=config.get('early_stopping', {}).get('patience', 10),
            min_delta=config.get('early_stopping', {}).get('min_delta', 0.001),
            verbose=True,
            mode=config.get('early_stopping', {}).get('mode', 'min')
        )
        callbacks.append(early_stopping_callback)
    
    # Learning rate monitor
    if config.get('lr_scheduler', {}).get('enabled', False):
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
    
    # Set up loggers
    loggers = []
    
    # Weights & Biases logger
    if config.get('logging', {}).get('use_wandb', False):
        wandb_logger = WandbLogger(
            project=config.get('logging', {}).get('wandb_project', 'glaucoma-detection'),
            name=config.get('logging', {}).get('run_name', None),
            save_dir=os.path.join(output_dir, 'logs'),
            log_model=True
        )
        loggers.append(wandb_logger)
    
    # CSV logger
    csv_logger = CSVLogger(
        save_dir=os.path.join(output_dir, 'logs'),
        name="csv_logs"
    )
    loggers.append(csv_logger)
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=config.get('epochs', 50),
        accelerator='gpu' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu',
        devices=config.get('gpu_ids', [0]) if torch.cuda.is_available() and config.get('use_gpu', True) else None,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        deterministic=True,
        precision=config.get('precision', '32-true'),
        gradient_clip_val=config.get('gradient_clip_val', 0.0),
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1)
    )
    
    return trainer

def train_model(
    model: nn.Module,
    data_module: pl.LightningDataModule,
    config: Dict[str, Any],
    output_dir: str
) -> Tuple[pl.LightningModule, pl.Trainer]:
    """Train model with PyTorch Lightning."""
    # Create lightning module
    lightning_model = GlaucomaSegmentationModel(model, config)
    
    # Set up trainer
    trainer = setup_training(model, data_module, config, output_dir)
    
    # Train model
    logger.info("Starting model training...")
    trainer.fit(lightning_model, data_module)
    
    # Test model if test data is available
    if hasattr(data_module, 'test_dataloader') and data_module.test_dataloader() is not None:
        logger.info("Testing model...")
        trainer.test(lightning_model, data_module)
    
    logger.info("Training completed")
    return lightning_model, trainer