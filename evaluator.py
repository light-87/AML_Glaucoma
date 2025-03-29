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
                all_targets.append(masks.cpu())
                all_images.append(images.cpu())
        
        # Concatenate results
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_images = torch.cat(all_images, dim=0)
        
        # Calculate metrics
        metrics_result = self.metrics(all_preds, all_targets)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = self.roc(all_preds, all_targets)
        roc_auc = torch.trapz(tpr, fpr).item()
        
        # Calculate PR curve
        precision, recall, pr_thresholds = self.pr_curve(all_preds, all_targets)
        pr_auc = torch.trapz(precision, recall).item()
        
        # Calculate confusion matrix
        cm = self.confusion_matrix(all_preds, all_targets)
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
        self._generate_visualizations(all_images, all_preds, all_targets, fpr, tpr, precision, recall)
        
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
            "sample_predictions": wandb_images,
            "roc_curve": wandb.plot.roc_curve(
                y_true=targets.flatten().numpy(),
                y_probas=predictions.flatten().numpy()
            ),
            "pr_curve": wandb.plot.pr_curve(
                y_true=targets.flatten().numpy(), 
                y_probas=predictions.flatten().numpy()
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