# Configuration Schema Documentation

## paths

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| base_dir | str | Yes | - | Base directory for the project |

## data

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| zip_file | str | No | - | Path to ZIP file if extraction is needed |
| random_state | int | No | 42 | Random seed for reproducibility |
| **split_config** | object | - | - | Nested configuration section |

### split_config

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| train_ratio | float | No | 0.7 | Ratio of training data |
| val_ratio | float | No | 0.15 | Ratio of validation data |
| test_ratio | float | No | 0.15 | Ratio of test data |


## model

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| architecture | str | Yes | - | Model architecture to use (Options: ['unet', 'unet++', 'deeplabv3', 'fpn']) |
| encoder | str | Yes | - | Backbone encoder for the model |
| pretrained | bool | No | True | Whether to use pretrained weights |
| in_channels | int | No | 3 | Number of input channels |
| num_classes | int | No | 1 | Number of output classes |

## preprocessing

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| image_size | list | Yes | - | Target image size (width, height) |
| image_channels | int | No | 3 | Number of image channels |
| normalization | str | No | imagenet | Normalization method (Options: ['imagenet', 'instance', 'pixel', 'none']) |
| mode | str | No | segmentation | Mode of operation (Options: ['segmentation', 'classification']) |
| **augmentation** | object | - | - | Nested configuration section |

### augmentation

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| enabled | bool | No | True | Whether to use data augmentation |
| rotation_range | float | No | 15 | Rotation range for augmentation |
| width_shift_range | float | No | 0.1 | Width shift range for augmentation |
| height_shift_range | float | No | 0.1 | Height shift range for augmentation |
| shear_range | float | No | 0.1 | Shear range for augmentation |
| zoom_range | float | No | 0.1 | Zoom range for augmentation |
| horizontal_flip | bool | No | True | Whether to use horizontal flip |
| vertical_flip | bool | No | False | Whether to use vertical flip |


## training

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| epochs | int | Yes | - | Number of training epochs |
| batch_size | int | Yes | - | Batch size for training |
| num_workers | int | No | 4 | Number of workers for data loading |
| learning_rate | float | No | 0.001 | Learning rate |
| optimizer | str | No | adam | Optimizer to use (Options: ['adam', 'sgd', 'adamw']) |
| loss_function | str | No | combined | Loss function to use (Options: ['combined', 'dice', 'bce', 'focal', 'jaccard']) |
| precision | str | No | 32-true | Precision for training (Options: ['16-mixed', '32-true']) |
| use_gpu | bool | No | True | Whether to use GPU for training |
| gpu_ids | list | No | [0] | List of GPU IDs to use |
| gradient_clip_val | float | No | 0.0 | Gradient clipping value |
| accumulate_grad_batches | int | No | 1 | Number of batches to accumulate gradients |
| **lr_scheduler** | object | - | - | Nested configuration section |

### lr_scheduler

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| enabled | bool | No | True | Whether to use learning rate scheduler |
| factor | float | No | 0.1 | Factor by which to reduce learning rate |
| patience | int | No | 5 | Patience for learning rate scheduler |
| min_lr | float | No | 1e-06 | Minimum learning rate |
| monitor | str | No | val_loss | Metric to monitor for scheduler |

| **early_stopping** | object | - | - | Nested configuration section |

### early_stopping

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| enabled | bool | No | True | Whether to use early stopping |
| patience | int | No | 10 | Patience for early stopping |
| monitor | str | No | val_loss | Metric to monitor for early stopping |
| min_delta | float | No | 0.001 | Minimum change to qualify as improvement |
| mode | str | No | min | Mode for early stopping (Options: ['min', 'max']) |

| **checkpointing** | object | - | - | Nested configuration section |

### checkpointing

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| enabled | bool | No | True | Whether to use checkpointing |
| save_top_k | int | No | 3 | Number of best models to save |
| monitor | str | No | val_loss | Metric to monitor for checkpointing |
| mode | str | No | min | Mode for checkpointing (Options: ['min', 'max']) |

| use_class_weights | bool | No | True | Whether to use class weights for imbalanced data |

## evaluation

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| metrics | list | No | ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1'] | Metrics to calculate during evaluation |
| threshold | float | No | 0.5 | Threshold for binary segmentation |
| num_samples | int | No | 5 | Number of sample visualizations |
| **visualization** | object | - | - | Nested configuration section |

### visualization

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| enabled | bool | No | True | Whether to generate visualizations |
| plot_wrong_predictions | bool | No | True | Whether to plot wrong predictions |
| plot_gradcam | bool | No | True | Whether to plot GradCAM visualizations |


## logging

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| use_wandb | bool | No | False | Whether to use Weights & Biases for logging |
| wandb_project | str | No | glaucoma-detection | Weights & Biases project name |
| log_every_n_steps | int | No | 10 | How often to log metrics |

## pipeline

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| steps | list | No | ['extract', 'load', 'clean', 'preprocess', 'train', 'evaluate'] | Pipeline steps to execute |
| force | bool | No | False | Whether to force rerun of steps |
| description | str | No | Default pipeline run | Description of the pipeline run |

