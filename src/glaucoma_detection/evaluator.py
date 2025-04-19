"""
Evaluation Module

Standardized evaluation using PyTorch Lightning and torchmetrics.
"""

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, BinaryAccuracy, BinaryPrecision, BinaryRecall
from torchmetrics.classification import BinaryConfusionMatrix, BinaryROC, BinaryPrecisionRecallCurve
from typing import Dict, Any, Tuple, Optional, List, Union
import cv2
from pathlib import Path
import wandb

logger = logging.getLogger(__name__)

class SegmentationEvaluator:
    """Evaluate segmentation models."""
    
    def __init__(self, model: pl.LightningModule, output_dir: str):
        """Initialize evaluator."""
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up metrics
        self.metrics = torchmetrics.MetricCollection({
            'dice': BinaryJaccardIndex(),
            'f1': BinaryF1Score(),
            'accuracy': BinaryAccuracy(),
            'precision': BinaryPrecision(),
            'recall': BinaryRecall()
        })
        
        # Set up ROC curve
        self.roc = BinaryROC(thresholds=100)
        
        # Set up PR curve
        self.pr_curve = BinaryPrecisionRecallCurve(thresholds=100)
        
        # Set up confusion matrix
        self.confusion_matrix = BinaryConfusionMatrix()
        
        # Put model in evaluation mode
        self.model.eval()
    
    def evaluate(self, test_loader: torch.utils.data.DataLoader, device: str = 'cuda') -> Dict[str, Any]:
        """Evaluate model on test data."""
        # Move model to device
        self.model = self.model.to(device)
        
        # Store predictions and targets
        all_preds = []
        all_targets = []
        all_images = []
        
        # Process batches
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc='Evaluating')):
                # Move data to device
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Store predictions and targets
                all_preds.append(outputs.cpu())
                
                # Ensure masks are binary for metrics calculation
                binary_masks = (masks > 0.5).float()
                all_targets.append(binary_masks.cpu())
                
                all_images.append(images.cpu())
        
        # Concatenate results
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_images = torch.cat(all_images, dim=0)
        
        # Apply threshold to predictions for binary evaluation
        binary_preds = (all_preds > self.threshold).float()
        
        # Calculate metrics using binary predictions and targets
        metrics_result = self.metrics(binary_preds, all_targets)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = self.roc(all_preds, all_targets)
        roc_auc = torch.trapz(tpr, fpr).item()
        
        # Calculate PR curve
        precision, recall, pr_thresholds = self.pr_curve(all_preds, all_targets)
        pr_auc = torch.trapz(precision, recall).item()
        
        # Calculate confusion matrix
        cm = self.confusion_matrix(binary_preds, all_targets)
        tn, fp, fn, tp = cm.flatten().tolist()
        
        # Compile results
        results = {
            **{k: v.item() for k, v in metrics_result.items()},
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
        }
        
        # Generate visualizations
        self._generate_visualizations(all_images, binary_preds, all_targets, fpr, tpr, precision, recall)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _generate_visualizations(self, images, predictions, targets, fpr, tpr, precision, recall):
        """Generate and save visualizations."""
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Sample indices for visualization
        num_samples = min(5, images.size(0))
        indices = np.random.choice(images.size(0), num_samples, replace=False)
        
        # Generate sample predictions visualizations
        self._generate_sample_visualizations(images, predictions, targets, indices, viz_dir)
        
        # Plot ROC curve
        self._plot_roc_curve(fpr, tpr, viz_dir)
        
        # Plot PR curve
        self._plot_pr_curve(precision, recall, viz_dir)
        
        # Log to wandb if available
        if wandb.run is not None:
            self._log_to_wandb(images, predictions, targets, indices, fpr, tpr, precision, recall)
    
    def _generate_sample_visualizations(self, images, predictions, targets, indices, output_dir):
        """Generate visualizations for sample predictions."""
        # Convert to numpy for visualization
        images_np = images.numpy()
        preds_np = predictions.numpy()
        targets_np = targets.numpy()
        
        # Apply threshold to get binary predictions
        preds_binary = (preds_np > 0.5).astype(np.float32)
        
        for i, idx in enumerate(indices):
            # Get sample
            image = images_np[idx].transpose(1, 2, 0)
            pred = preds_binary[idx, 0]
            target = targets_np[idx, 0]
            
            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean
            image = np.clip(image, 0, 1)
            
            # Create figure with subplots
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot original image
            ax[0].imshow(image)
            ax[0].set_title('Original Image')
            ax[0].axis('off')
            
            # Plot ground truth mask
            ax[1].imshow(target, cmap='gray')
            ax[1].set_title('Ground Truth')
            ax[1].axis('off')
            
            # Plot prediction
            ax[2].imshow(pred, cmap='gray')
            ax[2].set_title('Prediction')
            ax[2].axis('off')
            
            # Save figure
            plt.savefig(output_dir / f'sample_{idx}.png', bbox_inches='tight', dpi=100)
            plt.close()
            
            # Create overlay visualization
            self._create_overlay_visualization(image, pred, target, idx, output_dir)
    
    def _create_overlay_visualization(self, image, pred, target, idx, output_dir):
        """Create overlay visualization."""
        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Create RGB masks
        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        target_rgb = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
        
        # Set colors
        pred_rgb[pred > 0] = [255, 0, 0]  # Red for prediction
        target_rgb[target > 0] = [0, 255, 0]  # Green for ground truth
        
        # Create combined mask
        combined_rgb = np.zeros_like(pred_rgb)
        combined_rgb[pred > 0] = [255, 0, 0]  # Red for prediction only
        combined_rgb[target > 0] = [0, 255, 0]  # Green for ground truth only
        combined_rgb[(pred > 0) & (target > 0)] = [255, 255, 0]  # Yellow for overlap
        
        # Create overlay
        alpha = 0.5
        combined_overlay = cv2.addWeighted(img_uint8, 1-alpha, combined_rgb, alpha, 0)
        
        # Save overlay
        plt.figure(figsize=(8, 8))
        plt.imshow(combined_overlay)
        plt.title('Overlay: Green=GT, Red=Pred, Yellow=Overlap')
        plt.axis('off')
        plt.savefig(output_dir / f'overlay_{idx}.png', bbox_inches='tight', dpi=100)
        plt.close()
    
    def _plot_roc_curve(self, fpr, tpr, output_dir):
        """Plot and save ROC curve."""
        plt.figure(figsize=(10, 8))
        plt.plot(fpr.numpy(), tpr.numpy(), lw=2)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.grid(True)
        plt.savefig(output_dir / 'roc_curve.png', bbox_inches='tight', dpi=100)
        plt.close()
    
    def _plot_pr_curve(self, precision, recall, output_dir):
        """Plot and save precision-recall curve."""
        plt.figure(figsize=(10, 8))
        plt.plot(recall.numpy(), precision.numpy(), lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.savefig(output_dir / 'pr_curve.png', bbox_inches='tight', dpi=100)
        plt.close()
    
    def _log_to_wandb(self, images, predictions, targets, indices, fpr, tpr, precision, recall):
        """Log results to Weights & Biases."""
        # Convert to numpy for visualization
        images_np = images.numpy()
        preds_np = predictions.numpy()
        targets_np = targets.numpy()
        
        # Apply threshold to get binary predictions
        preds_binary = (preds_np > 0.5).astype(np.float32)
        
        # Log sample images
        wandb_images = []
        for i, idx in enumerate(indices):
            # Get sample
            image = images_np[idx].transpose(1, 2, 0)
            pred = preds_binary[idx, 0]
            target = targets_np[idx, 0]
            
            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean
            image = np.clip(image, 0, 1)
            
            # Add to wandb images
            wandb_images.append(wandb.Image(
                image,
                masks={
                    "ground_truth": {"mask_data": target, "class_labels": {0: "background", 1: "glaucoma"}},
                    "prediction": {"mask_data": pred, "class_labels": {0: "background", 1: "glaucoma"}}
                }
            ))
        
        # Log to wandb
        wandb.log({
            "conf_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=targets.flatten().numpy(), 
                preds=predictions.flatten().round().numpy(),
                class_names=["background", "glaucoma"]
            )
        })

        # Log precision-recall curve
        wandb.log({
            "pr_curve": wandb.plot.pr_curve(
                y_true=targets.flatten().numpy(),
                y_probas=predictions.flatten().numpy(),
                labels=["glaucoma"],
                classes_to_plot=[1]
            )
        })
    
    def _save_results(self, results):
        """Save evaluation results to CSV and JSON."""
        # Save as CSV
        pd.DataFrame([results]).to_csv(self.output_dir / 'metrics.csv', index=False)
        
        # Save as JSON
        import json
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Create summary text file
        with open(self.output_dir / 'summary.txt', 'w') as f:
            f.write("Evaluation Results\n")
            f.write("=================\n\n")
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")
            
# Add to evaluator.py or trainer.py
def debug_predictions(images, masks, predictions, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(5, len(images))):
        # Convert tensors to numpy
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        mask = masks[i].cpu().numpy()[0]
        pred = predictions[i].cpu().numpy()[0]
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title('Image')
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f'Ground Truth (unique: {np.unique(mask)})')
        axes[2].imshow(pred, cmap='gray')
        axes[2].set_title(f'Prediction (unique: {np.unique(pred)})')
        
        plt.savefig(os.path.join(save_dir, f'debug_sample_{i}.png'))
        plt.close()

# In evaluator.py
def visualize_predictions(self, images, masks, preds, save_dir, num_samples=10):
    """Generate visualizations of predictions vs ground truth."""
    os.makedirs(save_dir, exist_ok=True)
    
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    
    for i, idx in enumerate(indices):
        image = images[idx].cpu().numpy().transpose(1, 2, 0)
        mask = masks[idx].cpu().numpy()[0]
        pred = (preds[idx] > 0.5).float().cpu().numpy()[0]
        
        # Normalize image for display
        image = (image - image.min()) / (image.max() - image.min())
        
        # Create overlay
        overlay = np.zeros_like(image)
        overlay[:,:,0] = pred * 0.7  # Red channel - prediction
        overlay[:,:,1] = mask * 0.7  # Green channel - ground truth
        
        # Final image with transparency
        result = image * 0.7 + overlay * 0.3
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(image)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(pred, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(result)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.savefig(os.path.join(save_dir, f'result_{idx}.png'), dpi=150, bbox_inches='tight')
        plt.close()

def create_overlay_visualization(image, pred, target):
    """Create overlay visualization for wandb."""
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image * std + mean
    img = np.clip(img, 0, 1)
    
    # Create RGB masks
    overlay = img.copy()
    overlay[pred > 0.5, 0] += 0.5  # Red for prediction
    overlay[target > 0.5, 1] += 0.5  # Green for ground truth
    overlay = np.clip(overlay, 0, 1)
    
    return (overlay * 255).astype(np.uint8)

def _log_detailed_metrics_to_wandb(self, results, all_preds, all_targets, all_images, indices=None):
    """Log detailed per-sample metrics to wandb."""
    if wandb.run is None:
        return
    
    if indices is None:
        indices = np.random.choice(len(all_images), min(20, len(all_images)), replace=False)
    
    # Create a table with detailed metrics
    columns = ["image_idx", "dice", "iou", "accuracy", "image", "ground_truth", "prediction"]
    table_data = []
    
    for idx in indices:
        # Calculate metrics for this sample
        pred = all_preds[idx:idx+1]
        target = all_targets[idx:idx+1]
        image = all_images[idx]
        
        # For each sample, calculate individual metrics
        sample_dice = dice_coefficient(pred, target).item()
        sample_iou = jaccard_index(pred, target).item()
        sample_acc = binary_accuracy(pred, target).item()
        
        # Add to table
        table_data.append([
            idx,
            sample_dice,
            sample_iou,
            sample_acc,
            wandb.Image(image.numpy().transpose(1, 2, 0)),
            wandb.Image(target[0].numpy()),
            wandb.Image(pred[0].numpy())
        ])
    
    # Log the table
    table = wandb.Table(data=table_data, columns=columns)
    wandb.log({"detailed_metrics": table})