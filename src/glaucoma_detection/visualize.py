"""
Visualization Module

Comprehensive visualization tools for machine learning pipeline results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

class VisualizationManager:
    """Generate and manage visualizations for the pipeline."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """Initialize with output directory.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Set default matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Configure custom color maps
        self.setup_custom_colormaps()
    
    def setup_custom_colormaps(self):
        """Set up custom colormaps for visualizations."""
        # Custom colormap for segmentation overlays
        self.glaucoma_cmap = LinearSegmentedColormap.from_list(
            'glaucoma', 
            [(0, 'black'), (0.5, 'blue'), (0.7, 'cyan'), (0.9, 'yellow'), (1, 'red')]
        )
    
    def plot_training_history(self, history_file: Union[str, Path]) -> Optional[Path]:
        """Plot training metrics history from CSV logs.
        
        Args:
            history_file: Path to training history CSV file
            
        Returns:
            Path to saved visualization file
        """
        history_path = Path(history_file)
        if not history_path.exists():
            logger.error(f"Training history file not found: {history_path}")
            return None
        
        # Load training history
        try:
            history = pd.read_csv(history_path)
        except Exception as e:
            logger.error(f"Error loading training history: {e}")
            return None
        
        # Determine which metrics to plot
        metrics = []
        for col in history.columns:
            # Skip non-metric columns
            if col in ['epoch', 'step']:
                continue
            
            # Check if there's a validation version of this metric
            base_metric = col.replace('train_', '').replace('val_', '')
            if f"train_{base_metric}" in history.columns and f"val_{base_metric}" in history.columns:
                metrics.append(base_metric)
        
        # Create subplots for each metric
        n_metrics = len(metrics)
        if n_metrics == 0:
            logger.warning("No training metrics found in history file")
            return None
        
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics), dpi=100)
        if n_metrics == 1:
            axes = [axes]  # Make sure axes is always a list
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Get train and val columns
            train_col = f"train_{metric}"
            val_col = f"val_{metric}"
            
            # Plot if available
            if train_col in history.columns:
                history.plot(x='epoch', y=train_col, ax=ax, label=f'Training {metric}', color='blue')
            if val_col in history.columns:
                history.plot(x='epoch', y=val_col, ax=ax, label=f'Validation {metric}', color='orange')
            
            ax.set_title(f'{metric.capitalize()} over time')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True)
        
        # Save figure
        output_path = self.viz_dir / "training_history.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        
        logger.info(f"Training history visualization saved to {output_path}")
        return output_path
    
    def plot_evaluation_metrics(self, metrics_file: Union[str, Path]) -> Optional[Path]:
        """Plot evaluation metrics as a bar chart.
        
        Args:
            metrics_file: Path to metrics JSON file
            
        Returns:
            Path to saved visualization file
        """
        metrics_path = Path(metrics_file)
        if not metrics_path.exists():
            logger.error(f"Metrics file not found: {metrics_path}")
            return None
        
        # Load metrics
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metrics file: {e}")
            return None
        
        # Filter numeric metrics for plotting
        plot_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key not in ['threshold', 'specificity']:
                plot_metrics[key] = value
        
        if not plot_metrics:
            logger.warning("No numeric metrics found in metrics file")
            return None
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        
        # Sort metrics by value for better visualization
        sorted_metrics = {k: v for k, v in sorted(plot_metrics.items(), key=lambda item: item[1], reverse=True)}
        
        # Plot
        ax.bar(sorted_metrics.keys(), sorted_metrics.values(), color=sns.color_palette("viridis", len(sorted_metrics)))
        
        # Add value labels on bars
        for i, (key, value) in enumerate(sorted_metrics.items()):
            ax.text(i, value + 0.01, f'{value:.3f}', ha='center', fontsize=10)
        
        ax.set_title('Evaluation Metrics', fontsize=14)
        ax.set_ylim(0, 1.1)  # Assuming metrics are in 0-1 range
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Metric', fontsize=12)
        
        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        output_path = self.viz_dir / "evaluation_metrics.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        
        logger.info(f"Evaluation metrics visualization saved to {output_path}")
        return output_path
    
    def plot_confusion_matrix(self, 
                            metrics_file: Union[str, Path], 
                            class_names: List[str] = ['Non-Glaucoma', 'Glaucoma']) -> Optional[Path]:
        """Plot confusion matrix from evaluation metrics.
        
        Args:
            metrics_file: Path to metrics JSON file
            class_names: Names of the classes
            
        Returns:
            Path to saved visualization file
        """
        metrics_path = Path(metrics_file)
        if not metrics_path.exists():
            logger.error(f"Metrics file not found: {metrics_path}")
            return None
        
        # Load metrics
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metrics file: {e}")
            return None
        
        # Extract confusion matrix values
        try:
            tn = metrics.get('true_negatives', 0)
            fp = metrics.get('false_positives', 0)
            fn = metrics.get('false_negatives', 0)
            tp = metrics.get('true_positives', 0)
            
            # Create confusion matrix
            cm = np.array([[tn, fp], [fn, tp]])
        except Exception as e:
            logger.error(f"Error extracting confusion matrix values: {e}")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        # Add labels
        ax.set_title('Confusion Matrix', fontsize=14)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        
        # Calculate accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        fig.text(0.5, 0.01, f'Accuracy: {accuracy:.3f}', ha='center', fontsize=12)
        
        # Save figure
        output_path = self.viz_dir / "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        
        logger.info(f"Confusion matrix visualization saved to {output_path}")
        return output_path
    
    def plot_roc_curve(self, metrics_file: Union[str, Path]) -> Optional[Path]:
        """Plot ROC curve from evaluation metrics.
        
        Args:
            metrics_file: Path to metrics JSON file or directory with roc_curve.npz
            
        Returns:
            Path to saved visualization file
        """
        metrics_path = Path(metrics_file)
        if not metrics_path.exists():
            logger.error(f"Metrics file/directory not found: {metrics_path}")
            return None
        
        # Try to find ROC curve data
        roc_data_path = None
        if metrics_path.is_dir():
            potential_path = metrics_path / "roc_curve.npz"
            if potential_path.exists():
                roc_data_path = potential_path
        
        # If no ROC data, try to load from metrics JSON
        if roc_data_path is None:
            if metrics_path.is_file() and metrics_path.suffix == '.json':
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    # Check if metrics contains ROC AUC value
                    if 'roc_auc' in metrics:
                        # Create a simple ROC curve with just the AUC value
                        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
                        
                        # Plot diagonal line (random classifier)
                        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                        
                        # Create a rough ROC curve visualization
                        fpr = np.linspace(0, 1, 100)
                        tpr = fpr ** (1 / (2 * metrics['roc_auc'] - 1)) if metrics['roc_auc'] > 0.5 else fpr
                        
                        ax.plot(fpr, tpr, label=f'Model (AUC = {metrics["roc_auc"]:.3f})')
                        
                        ax.set_title('ROC Curve (Approximated)', fontsize=14)
                        ax.set_xlabel('False Positive Rate', fontsize=12)
                        ax.set_ylabel('True Positive Rate', fontsize=12)
                        ax.legend(loc='lower right')
                        ax.grid(True, alpha=0.3)
                        
                        # Save figure
                        output_path = self.viz_dir / "roc_curve.png"
                        plt.tight_layout()
                        plt.savefig(output_path)
                        plt.close(fig)
                        
                        logger.info(f"Approximated ROC curve saved to {output_path}")
                        return output_path
                except Exception as e:
                    logger.error(f"Error loading metrics for ROC curve: {e}")
                    return None
            
            logger.warning("No ROC curve data found")
            return None
        
        # Load ROC data
        try:
            roc_data = np.load(roc_data_path)
            fpr = roc_data['fpr']
            tpr = roc_data['tpr']
            thresholds = roc_data['thresholds'] if 'thresholds' in roc_data else None
        except Exception as e:
            logger.error(f"Error loading ROC curve data: {e}")
            return None
        
        # Calculate AUC (Area Under Curve)
        auc = np.trapz(tpr, fpr)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        # Add labels and settings
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        ax.legend(loc='lower right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        output_path = self.viz_dir / "roc_curve.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        
        logger.info(f"ROC curve saved to {output_path}")
        return output_path
    
    def plot_pr_curve(self, metrics_file: Union[str, Path]) -> Optional[Path]:
        """Plot Precision-Recall curve from evaluation metrics.
        
        Args:
            metrics_file: Path to metrics JSON file or directory with pr_curve.npz
            
        Returns:
            Path to saved visualization file
        """
        metrics_path = Path(metrics_file)
        if not metrics_path.exists():
            logger.error(f"Metrics file/directory not found: {metrics_path}")
            return None
        
        # Try to find PR curve data
        pr_data_path = None
        if metrics_path.is_dir():
            potential_path = metrics_path / "pr_curve.npz"
            if potential_path.exists():
                pr_data_path = potential_path
        
        # If no PR data, try to load from metrics JSON
        if pr_data_path is None:
            if metrics_path.is_file() and metrics_path.suffix == '.json':
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    # Check if metrics contains PR AUC value
                    if 'pr_auc' in metrics and 'precision' in metrics and 'recall' in metrics:
                            # Create a simple PR curve visualization
                        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
                        
                        # Generate approximate precision-recall curve
                        recall = np.linspace(0, 1, 100)
                        precision = np.ones_like(recall) * metrics['precision']
                        precision = np.maximum(precision - (recall ** 2), 0)
                        
                        ax.plot(recall, precision, 
                                label=f'PR curve (AUC = {metrics["pr_auc"]:.3f})')
                        
                        # Add baseline (random classifier)
                        if 'true_positives' in metrics and 'false_positives' in metrics:
                            tp = metrics['true_positives']
                            fp = metrics['false_positives']
                            fn = metrics['false_negatives']
                            baseline = tp / (tp + fn) if (tp + fn) > 0 else 0
                            ax.axhline(y=baseline, color='r', linestyle='--', 
                                      label=f'Baseline ({baseline:.3f})')
                        
                        ax.set_title('Precision-Recall Curve (Approximated)', fontsize=14)
                        ax.set_xlabel('Recall', fontsize=12)
                        ax.set_ylabel('Precision', fontsize=12)
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.legend(loc='best')
                        ax.grid(True, alpha=0.3)
                        
                        # Save figure
                        output_path = self.viz_dir / "pr_curve.png"
                        plt.tight_layout()
                        plt.savefig(output_path)
                        plt.close(fig)
                        
                        logger.info(f"Approximated PR curve saved to {output_path}")
                        return output_path
                except Exception as e:
                    logger.error(f"Error loading metrics for PR curve: {e}")
                    return None
            
            logger.warning("No PR curve data found")
            return None
        
        # Load PR data
        try:
            pr_data = np.load(pr_data_path)
            precision = pr_data['precision']
            recall = pr_data['recall']
            thresholds = pr_data['thresholds'] if 'thresholds' in pr_data else None
        except Exception as e:
            logger.error(f"Error loading PR curve data: {e}")
            return None
        
        # Calculate AUC (Area Under Curve)
        auc = np.trapz(precision, recall)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        
        # Plot PR curve
        ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR curve (AUC = {auc:.3f})')
        
        # Add labels and settings
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14)
        ax.legend(loc='best')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        output_path = self.viz_dir / "pr_curve.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        
        logger.info(f"PR curve saved to {output_path}")
        return output_path
    
    def create_segmentation_overlay(self, 
                                  image: np.ndarray, 
                                  mask: np.ndarray, 
                                  prediction: np.ndarray,
                                  alpha: float = 0.5,
                                  save_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """Create an overlay of segmentation mask on the original image.
        
        Args:
            image: Original image (h, w, 3)
            mask: Ground truth mask (h, w)
            prediction: Prediction mask (h, w)
            alpha: Transparency of overlay
            save_path: Path to save visualization
            
        Returns:
            Overlay image
        """
        # Ensure masks are binary
        mask_binary = mask > 0.5
        pred_binary = prediction > 0.5
        
        # Create RGB overlays
        h, w = mask_binary.shape[:2]
        mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        pred_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Set colors
        mask_rgb[mask_binary] = [0, 255, 0]  # Green for ground truth
        pred_rgb[pred_binary] = [255, 0, 0]  # Red for prediction
        
        # Create overlay colors
        overlay[mask_binary & ~pred_binary] = [0, 255, 0]  # Green for ground truth only
        overlay[~mask_binary & pred_binary] = [255, 0, 0]  # Red for prediction only
        overlay[mask_binary & pred_binary] = [255, 255, 0]  # Yellow for overlap
        
        # Ensure image is uint8 and has 3 channels
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        if len(image.shape) == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        
        # Create overlay
        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(save_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        
        return result
    
    def plot_sample_predictions(self, 
                             images: np.ndarray, 
                             masks: np.ndarray, 
                             predictions: np.ndarray,
                             num_samples: int = 5,
                             save_dir: Optional[Union[str, Path]] = None) -> List[Path]:
        """Plot sample predictions with ground truth and overlay.
        
        Args:
            images: Batch of images (b, h, w, 3) or (b, 3, h, w)
            masks: Batch of masks (b, h, w) or (b, 1, h, w)
            predictions: Batch of predictions (b, h, w) or (b, 1, h, w)
            num_samples: Number of samples to plot
            save_dir: Directory to save visualizations
            
        Returns:
            List of paths to saved visualizations
        """
        if save_dir is None:
            save_dir = self.viz_dir / "sample_predictions"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Convert torch tensors to numpy if needed
        if hasattr(images, 'cpu'):
            images = images.cpu().numpy()
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(predictions, 'cpu'):
            predictions = predictions.cpu().numpy()
        
        # Handle different input formats
        if images.shape[1] == 3 and len(images.shape) == 4:  # (b, 3, h, w)
            images = images.transpose(0, 2, 3, 1)
        
        if len(masks.shape) == 4 and masks.shape[1] == 1:  # (b, 1, h, w)
            masks = masks.squeeze(1)
        
        if len(predictions.shape) == 4 and predictions.shape[1] == 1:  # (b, 1, h, w)
            predictions = predictions.squeeze(1)
        
        # Normalize images if needed
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        
        # Get indices for visualization
        num_samples = min(num_samples, len(images))
        indices = np.random.choice(len(images), num_samples, replace=False)
        
        output_paths = []
        
        # Create visualizations for each sample
        for i, idx in enumerate(indices):
            image = images[idx]
            mask = masks[idx]
            pred = predictions[idx]
            
            # Create figure
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Plot original image
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Plot ground truth mask
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Plot prediction
            axes[2].imshow(pred, cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            # Plot overlay
            overlay = self.create_segmentation_overlay(image, mask, pred)
            axes[3].imshow(overlay)
            axes[3].set_title('Overlay')
            axes[3].axis('off')
            
            # Save figure
            output_path = save_dir / f"sample_{i}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close(fig)
            
            output_paths.append(output_path)
            
            # Also save just the overlay for easier viewing
            overlay_path = save_dir / f"overlay_{i}.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            output_paths.append(overlay_path)
        
        logger.info(f"Saved {len(output_paths)} sample prediction visualizations to {save_dir}")
        return output_paths
    
    def generate_run_summary(self, 
                          config: Dict[str, Any],
                          metrics: Dict[str, Any],
                          training_history: Optional[pd.DataFrame] = None,
                          sample_images: Optional[List[str]] = None) -> Path:
        """Generate a summary HTML report for the run.
        
        Args:
            config: Run configuration
            metrics: Evaluation metrics
            training_history: Training history dataframe
            sample_images: List of sample image paths
            
        Returns:
            Path to saved HTML report
        """
        # Create HTML content
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>Glaucoma Detection Run Summary</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }",
            "        h1, h2, h3 { color: #2c3e50; }",
            "        .container { max-width: 1200px; margin: 0 auto; }",
            "        .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }",
            "        .metrics-table { border-collapse: collapse; width: 100%; }",
            "        .metrics-table td, .metrics-table th { border: 1px solid #ddd; padding: 8px; }",
            "        .metrics-table tr:nth-child(even) { background-color: #f2f2f2; }",
            "        .metrics-table th { padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: #2c3e50; color: white; }",
            "        .image-container { display: flex; flex-wrap: wrap; gap: 10px; }",
            "        .image-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }",
            "        .chart-container { margin-top: 20px; text-align: center; }",
            "        pre { background-color: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <div class='container'>",
            f"        <h1>Glaucoma Detection Run Summary</h1>",
            f"        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        ]
        
        # Add configuration section
        html_content.extend([
            "        <div class='section'>",
            "            <h2>Configuration</h2>",
            "            <pre>",
            json.dumps(config, indent=4),
            "            </pre>",
            "        </div>"
        ])
        
        # Add metrics section
        html_content.extend([
            "        <div class='section'>",
            "            <h2>Evaluation Metrics</h2>",
            "            <table class='metrics-table'>",
            "                <tr><th>Metric</th><th>Value</th></tr>"
        ])
        
        # Add each metric to the table
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                html_content.append(f"                <tr><td>{key}</td><td>{value:.4f}</td></tr>")
        
        html_content.append("            </table>")
        
        # Add metric visualizations
        html_content.extend([
            "            <div class='chart-container'>",
            "                <h3>Performance Metrics</h3>",
            f"                <img src='{self.viz_dir.relative_to(self.output_dir) / 'evaluation_metrics.png'}' alt='Evaluation Metrics' />",
            "            </div>",
            "            <div class='chart-container'>",
            "                <h3>Confusion Matrix</h3>",
            f"                <img src='{self.viz_dir.relative_to(self.output_dir) / 'confusion_matrix.png'}' alt='Confusion Matrix' />",
            "            </div>",
            "            <div class='chart-container'>",
            "                <h3>ROC Curve</h3>",
            f"                <img src='{self.viz_dir.relative_to(self.output_dir) / 'roc_curve.png'}' alt='ROC Curve' />",
            "            </div>",
            "            <div class='chart-container'>",
            "                <h3>Precision-Recall Curve</h3>",
            f"                <img src='{self.viz_dir.relative_to(self.output_dir) / 'pr_curve.png'}' alt='PR Curve' />",
            "            </div>",
            "        </div>"
        ])

        # Add closing tags
        html_content.append("</body>")
        html_content.append("</html>")

        # Save HTML content to file
        html_file_path = self.viz_dir / "run_summary.html"
        with open(html_file_path, 'w') as f:
            f.write('\n'.join(html_content))

        logger.info(f"Run summary saved to {html_file_path}")
        return html_file_path