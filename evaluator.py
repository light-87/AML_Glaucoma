"""
Evaluation Module

This module handles the evaluation of glaucoma detection models on test data.
It calculates performance metrics and generates visualizations.

Functions:
- evaluate_model(): Evaluate model on test set
- calculate_performance_metrics(): Calculate detailed segmentation metrics
- visualize_predictions(): Generate visualizations of model predictions
- save_evaluation_results(): Save evaluation results to disk
"""

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_curve,
    roc_curve, auc, precision_score, recall_score, f1_score
)
import cv2
from datetime import datetime

# Import from other modules
from model import load_model
from preprocessor import GlaucomaDataset
from utils import setup_logger, create_directory

# Set up logging
logger = setup_logger('evaluator')

def evaluate_model(model, test_loader, device, threshold=0.5):
    """
    Evaluate model on test set.
    
    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model
    test_loader : torch.utils.data.DataLoader
        Test data loader
    device : str
        Device to evaluate on ('cuda' or 'cpu')
    threshold : float, optional
        Threshold for binary predictions, by default 0.5
        
    Returns:
    --------
    tuple
        (all_predictions, all_targets, all_images, evaluation_results)
    """
    model.eval()
    
    # Store predictions, targets, and images
    all_predictions = []
    all_targets = []
    all_images = []
    
    # No gradient calculation for evaluation
    with torch.no_grad():
        # Progress bar
        progress_bar = tqdm(test_loader, desc="Evaluating")
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            # Move data to device
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            
            # Forward pass
            outputs = model(images)
            
            # Store predictions, targets, and images
            all_predictions.append(outputs.detach().cpu())
            all_targets.append(masks.detach().cpu())
            all_images.append(images.detach().cpu())
    
    # Concatenate batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_images = torch.cat(all_images, dim=0)
    
    # Calculate performance metrics
    evaluation_results = calculate_performance_metrics(
        all_predictions, all_targets, threshold
    )
    
    return all_predictions, all_targets, all_images, evaluation_results

def calculate_performance_metrics(predictions, targets, threshold=0.5):
    """
    Calculate detailed segmentation metrics.
    
    Parameters:
    -----------
    predictions : torch.Tensor
        Model predictions
    targets : torch.Tensor
        Ground truth targets
    threshold : float, optional
        Threshold for binary predictions, by default 0.5
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Convert to numpy for calculations
    pred_np = predictions.numpy()
    target_np = targets.numpy()
    
    # Apply threshold for binary predictions
    pred_binary = (pred_np > threshold).astype(np.float32)
    
    # Calculate pixel-wise metrics
    pixel_accuracy = accuracy_score(
        target_np.flatten(), pred_binary.flatten()
    )
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        target_np.flatten(), pred_binary.flatten(), labels=[0, 1]
    ).ravel()
    
    # Precision, recall, and F1-score
    precision = precision_score(
        target_np.flatten(), pred_binary.flatten(), zero_division=0
    )
    recall = recall_score(
        target_np.flatten(), pred_binary.flatten(), zero_division=0
    )
    f1 = f1_score(
        target_np.flatten(), pred_binary.flatten(), zero_division=0
    )
    
    # Dice coefficient
    dice = 2 * tp / (2 * tp + fp + fn)
    
    # Intersection over Union (IoU)
    iou = tp / (tp + fp + fn)
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(target_np.flatten(), pred_np.flatten())
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve and AUC
    precision_curve, recall_curve, _ = precision_recall_curve(target_np.flatten(), pred_np.flatten())
    pr_auc = auc(recall_curve, precision_curve)
    
    # Compile metrics
    metrics = {
        'accuracy': pixel_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'dice_coefficient': dice,
        'iou': iou,
        'specificity': specificity,
        'sensitivity': recall,  # Sensitivity is the same as recall
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }
    
    return metrics

def visualize_predictions(images, predictions, targets, indices=None, threshold=0.5, max_samples=10):
    """
    Generate visualizations of model predictions.
    
    Parameters:
    -----------
    images : torch.Tensor
        Input images
    predictions : torch.Tensor
        Model predictions
    targets : torch.Tensor
        Ground truth targets
    indices : list, optional
        List of indices to visualize, by default None (random selection)
    threshold : float, optional
        Threshold for binary predictions, by default 0.5
    max_samples : int, optional
        Maximum number of samples to visualize, by default 10
        
    Returns:
    --------
    list
        List of figure objects
    """
    # Convert to numpy
    images_np = images.numpy()
    preds_np = predictions.numpy()
    targets_np = targets.numpy()
    
    # Apply threshold for binary predictions
    preds_binary = (preds_np > threshold).astype(np.float32)
    
    # Select indices if not provided
    if indices is None:
        total_samples = images_np.shape[0]
        num_samples = min(max_samples, total_samples)
        indices = np.random.choice(total_samples, num_samples, replace=False)
    
    # Create visualizations
    figures = []
    
    for i, idx in enumerate(indices):
        # Get sample
        image = images_np[idx].transpose(1, 2, 0)  # CHW to HWC
        pred = preds_binary[idx, 0]  # Remove channel dimension
        target = targets_np[idx, 0]  # Remove channel dimension
        
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot target mask
        axes[1].imshow(target, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        # Plot prediction mask
        axes[2].imshow(pred, cmap='gray')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
        
        # Add overall title
        fig.suptitle(f'Sample {idx}', fontsize=16)
        plt.tight_layout()
        
        figures.append(fig)
    
    return figures

def generate_overlay_visualizations(images, predictions, targets, indices=None, threshold=0.5, max_samples=10):
    """
    Generate overlay visualizations of model predictions on original images.
    
    Parameters:
    -----------
    images : torch.Tensor
        Input images
    predictions : torch.Tensor
        Model predictions
    targets : torch.Tensor
        Ground truth targets
    indices : list, optional
        List of indices to visualize, by default None (random selection)
    threshold : float, optional
        Threshold for binary predictions, by default 0.5
    max_samples : int, optional
        Maximum number of samples to visualize, by default 10
        
    Returns:
    --------
    list
        List of figure objects
    """
    # Convert to numpy
    images_np = images.numpy()
    preds_np = predictions.numpy()
    targets_np = targets.numpy()
    
    # Apply threshold for binary predictions
    preds_binary = (preds_np > threshold).astype(np.float32)
    
    # Select indices if not provided
    if indices is None:
        total_samples = images_np.shape[0]
        num_samples = min(max_samples, total_samples)
        indices = np.random.choice(total_samples, num_samples, replace=False)
    
    # Create visualizations
    figures = []
    
    for i, idx in enumerate(indices):
        # Get sample
        image = images_np[idx].transpose(1, 2, 0)  # CHW to HWC
        pred = preds_binary[idx, 0]  # Remove channel dimension
        target = targets_np[idx, 0]  # Remove channel dimension
        
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        # Convert to uint8 for visualization
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Create RGB masks for overlay
        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        target_rgb = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
        
        # Set colors: red for prediction, green for target
        pred_rgb[pred > 0] = [255, 0, 0]  # Red for prediction
        target_rgb[target > 0] = [0, 255, 0]  # Green for target
        
        # Create overlays
        alpha = 0.5  # Transparency
        pred_overlay = cv2.addWeighted(image_uint8, 1, pred_rgb, alpha, 0)
        target_overlay = cv2.addWeighted(image_uint8, 1, target_rgb, alpha, 0)
        
        # Create combined overlay
        combined_rgb = np.zeros_like(pred_rgb)
        combined_rgb[pred > 0] = [255, 0, 0]  # Red for prediction only
        combined_rgb[target > 0] = [0, 255, 0]  # Green for target only
        combined_rgb[(pred > 0) & (target > 0)] = [255, 255, 0]  # Yellow for overlap
        combined_overlay = cv2.addWeighted(image_uint8, 1, combined_rgb, alpha, 0)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Plot target overlay
        axes[0, 1].imshow(target_overlay)
        axes[0, 1].set_title('Ground Truth Overlay')
        axes[0, 1].axis('off')
        
        # Plot prediction overlay
        axes[1, 0].imshow(pred_overlay)
        axes[1, 0].set_title('Prediction Overlay')
        axes[1, 0].axis('off')
        
        # Plot combined overlay
        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Combined Overlay\nGreen: GT, Red: Pred, Yellow: Overlap')
        axes[1, 1].axis('off')
        
        # Add overall title
        fig.suptitle(f'Sample {idx}', fontsize=16)
        plt.tight_layout()
        
        figures.append(fig)
    
    return figures

def plot_roc_curve(predictions, targets):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    predictions : torch.Tensor
        Model predictions
    targets : torch.Tensor
        Ground truth targets
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with ROC curve
    """
    # Convert to numpy
    pred_np = predictions.numpy().flatten()
    target_np = targets.numpy().flatten()
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(target_np, pred_np)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True)
    
    return fig

def plot_precision_recall_curve(predictions, targets):
    """
    Plot precision-recall curve.
    
    Parameters:
    -----------
    predictions : torch.Tensor
        Model predictions
    targets : torch.Tensor
        Ground truth targets
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with precision-recall curve
    """
    # Convert to numpy
    pred_np = predictions.numpy().flatten()
    target_np = targets.numpy().flatten()
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(target_np, pred_np)
    pr_auc = auc(recall, precision)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot precision-recall curve
    ax.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(True)
    
    return fig

def save_evaluation_results(metrics, figures, output_dir, prefix=''):
    """
    Save evaluation results to disk.
    
    Parameters:
    -----------
    metrics : dict
        Evaluation metrics
    figures : list
        List of figures
    output_dir : str
        Directory to save results
    prefix : str, optional
        Prefix for filenames, by default ''
        
    Returns:
    --------
    str
        Path to saved results
    """
    # Create output directory if it doesn't exist
    create_directory(output_dir)
    
    # Create subfolder with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(output_dir, f'{prefix}evaluation_{timestamp}')
    create_directory(results_dir)
    
    # Save metrics as CSV and JSON
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)
    
    # Save as JSON for better readability
    import json
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save figures
    for i, fig in enumerate(figures):
        fig.savefig(os.path.join(results_dir, f'figure_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Create a summary text file
    with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
        f.write(f'Evaluation Results ({timestamp})\n')
        f.write('=' * 40 + '\n\n')
        
        f.write('Key Metrics:\n')
        f.write(f'  Dice Coefficient: {metrics["dice_coefficient"]:.4f}\n')
        f.write(f'  IoU: {metrics["iou"]:.4f}\n')
        f.write(f'  Accuracy: {metrics["accuracy"]:.4f}\n')
        f.write(f'  Precision: {metrics["precision"]:.4f}\n')
        f.write(f'  Recall: {metrics["recall"]:.4f}\n')
        f.write(f'  F1 Score: {metrics["f1_score"]:.4f}\n')
        f.write(f'  AUC-ROC: {metrics["roc_auc"]:.4f}\n')
        f.write(f'  AUC-PR: {metrics["pr_auc"]:.4f}\n\n')
        
        f.write('Confusion Matrix:\n')
        f.write(f'  True Positives: {metrics["true_positives"]}\n')
        f.write(f'  False Positives: {metrics["false_positives"]}\n')
        f.write(f'  True Negatives: {metrics["true_negatives"]}\n')
        f.write(f'  False Negatives: {metrics["false_negatives"]}\n')
    
    logger.info(f'Evaluation results saved to {results_dir}')
    return results_dir

def evaluate_segmentation(model_path, test_loader, output_dir, device=None):
    """
    Evaluate segmentation model on test set.
    
    Parameters:
    -----------
    model_path : str
        Path to model checkpoint
    test_loader : torch.utils.data.DataLoader
        Test data loader
    output_dir : str
        Directory to save results
    device : str, optional
        Device to evaluate on ('cuda' or 'cpu'), by default None
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, _, _ = load_model(model_path, device=device)
    
    if model is None:
        logger.error(f"Failed to load model from {model_path}")
        return None
    
    # Move model to device
    model = model.to(device)
    
    # Evaluate model
    logger.info(f"Evaluating model on {device}")
    predictions, targets, images, metrics = evaluate_model(model, test_loader, device)
    
    # Generate visualizations
    logger.info("Generating visualizations")
    figures = []
    
    # Basic visualizations
    basic_visualization_figures = visualize_predictions(
        images, predictions, targets, max_samples=5
    )
    figures.extend(basic_visualization_figures)
    
    # Overlay visualizations
    overlay_figures = generate_overlay_visualizations(
        images, predictions, targets, max_samples=5
    )
    figures.extend(overlay_figures)
    
    # ROC curve
    roc_figure = plot_roc_curve(predictions, targets)
    figures.append(roc_figure)
    
    # Precision-recall curve
    pr_figure = plot_precision_recall_curve(predictions, targets)
    figures.append(pr_figure)
    
    # Save results
    results_dir = save_evaluation_results(metrics, figures, output_dir)
    
    return metrics

def main(model_path, test_csv, output_dir=None):
    """
    Main function to evaluate the model.
    
    Parameters:
    -----------
    model_path : str
        Path to model checkpoint
    test_csv : str
        Path to test CSV file
    output_dir : str, optional
        Directory to save results, by default None
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    import pandas as pd
    from torch.utils.data import DataLoader
    from preprocessor import GlaucomaDataset
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    
    # Load test dataframe
    try:
        logger.info(f"Loading test data from {test_csv}")
        test_df = pd.read_csv(test_csv)
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return None
    
    # Create test dataset
    test_dataset = GlaucomaDataset(
        test_df,
        target_size=(224, 224),
        augment=False,
        mode='segmentation'
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate model
    metrics = evaluate_segmentation(model_path, test_loader, output_dir)
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate glaucoma detection model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save results')
    
    args = parser.parse_args()
    
    main(args.model_path, args.test_csv, args.output_dir)