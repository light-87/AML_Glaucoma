# Glaucoma Detection Pipeline

A modular machine learning pipeline for automated glaucoma detection from retinal fundus images, using modern libraries and best practices.

## Project Overview

This project implements a complete machine learning pipeline for glaucoma detection from retinal fundus images. The pipeline is built with modern libraries like PyTorch Lightning, Hydra, and Segmentation Models PyTorch, making it modular, maintainable, and easy to experiment with.

### Datasets

The pipeline works with three glaucoma datasets:

1. **ORIGA** - Singapore Chinese Eye Study dataset containing fundus images with glaucoma diagnosis
2. **REFUGE** - Retinal Fundus Glaucoma Challenge dataset with expert annotations
3. **G1020** - A large collection of retinal images with associated metadata

### Key Features

- **Modern Architecture**: Built with PyTorch Lightning, Hydra, and Segmentation Models PyTorch
- **Memory Efficiency**: Optimized data loading with caching mechanisms for handling large datasets
- **Experiment Tracking**: Seamless integration with Weights & Biases for experiment monitoring
- **Modular Design**: Well-organized, maintainable codebase with clear separation of concerns
- **Flexible Configuration**: Hydra-based configuration system for easy experimentation
- **Reproducibility**: Controlled random seeds, configuration versioning, and explicit dependencies
- **User-Friendly CLI**: Simple command-line interface for easy execution
- **Standardized Metrics**: Using torchmetrics for consistent evaluation
- **Comprehensive Visualization**: Automatic generation of evaluation visualizations
- **GPU Optimization**: Automatic GPU memory check and optimization suggestions

## Installation

### Prerequisites

- Python 3.7+
- pip or conda package manager
- CUDA-compatible GPU (recommended but optional)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/glaucoma-detection.git
cd glaucoma-detection
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

For development:

```bash
pip install -e .
```

With development dependencies:

```bash
pip install -e ".[dev]"
```

## Data Setup

### Dataset Directory Structure

The pipeline expects the following structure:

```
data_dir/
├── ORIGA/
│   ├── Images/           # Contains fundus images (.jpg)
│   ├── Masks/            # Contains mask images (.png)
│   └── OrigaList.csv     # Metadata file with labels
├── REFUGE/
│   ├── train/
│   │   ├── Images/
│   │   └── Masks/
│   ├── val/
│   │   ├── Images/
│   │   └── Masks/
│   └── test/
│       ├── Images/
│       └── Masks/
└── G1020/
    ├── Images/           # Contains fundus images (.jpg)
    ├── Masks/            # Contains mask images (.png)
    └── G1020.csv         # Metadata file with labels
```

### Downloading and Extracting Datasets

If you have a ZIP file containing the datasets, you can extract it using:

```bash
python -m glaucoma_detection.run_pipeline --zip-file /path/to/fundus_dataset.zip --steps extract
```

### Diagnosing Dataset Structure

To verify your dataset structure is correct:

```bash
python -m glaucoma_detection.diagnose_datasets
```

This tool will check for the presence of required files and subdirectories and provide a detailed report.

## Usage

### Quick Start

To run the complete pipeline with default settings:

```bash
python -m glaucoma_detection.run_pipeline
```

### Memory Check

Before running the pipeline, you can check if your GPU has sufficient memory:

```bash
python -m glaucoma_detection.memory_check
```

This will analyze your GPU's available memory and provide recommendations for batch size, image size, and other memory-related parameters.

### Running with Lighter Settings

For systems with limited GPU memory:

```bash
python -m glaucoma_detection.test_pipeline
```

This runs the pipeline with optimized settings for lower memory usage.

### Running Specific Pipeline Steps

You can run specific steps of the pipeline:

```bash
python -m glaucoma_detection.run_pipeline --steps load,clean,preprocess,train
```

Available steps:
- `extract`: Extract datasets from ZIP file
- `load`: Load data into a consolidated format
- `clean`: Clean and preprocess the data
- `preprocess`: Create data loaders and splits
- `train`: Train the segmentation model
- `evaluate`: Evaluate model performance on test data

### Forcing Rerun of Steps

To force rerunning specific steps:

```bash
python -m glaucoma_detection.run_pipeline --steps train,evaluate --force
```

### SSL Fix for Downloads

If you encounter SSL certificate issues with model downloads:

```bash
python -m glaucoma_detection.run_with_fix
```

## Configuration System

The pipeline uses Hydra for configuration. The main configuration file is `conf/config.yaml`, but you can override settings in multiple ways.

### Configuration Files

Main configuration files:
- `conf/config.yaml`: Main configuration
- `conf/data/default.yaml`: Dataset settings
- `conf/model/unet.yaml`: Model architecture settings
- `conf/preprocessing/default.yaml`: Preprocessing and augmentation settings
- `conf/training/default.yaml`: Training parameters
- `conf/evaluation/default.yaml`: Evaluation metrics and visualization
- `conf/logging/default.yaml`: Logging and experiment tracking

### Overriding Configuration

1. **Command line arguments**:

```bash
python -m glaucoma_detection.run_pipeline paths.data_dir=/custom/data/path model.encoder=resnet18
```

2. **Environment variables**:

```bash
export DATA_DIR=/custom/data/path
python -m glaucoma_detection.run_pipeline
```

3. **Configuration files**:

```bash
python -m glaucoma_detection.run_pipeline --config path/to/config.yaml
```

### Common Configuration Examples

#### Changing model architecture

```bash
python -m glaucoma_detection.run_pipeline model.architecture=unet++ model.encoder=efficientnet-b3
```

Available architectures: `unet`, `unet++`, `deeplabv3`, `fpn`

Available encoders: `resnet18`, `resnet34`, `resnet50`, `efficientnet-b0` through `efficientnet-b7`, etc.

#### Adjusting image size and batch size

```bash
python -m glaucoma_detection.run_pipeline preprocessing.image_size=[224,224] training.batch_size=16
```

#### Changing training parameters

```bash
python -m glaucoma_detection.run_pipeline training.epochs=20 training.learning_rate=0.0005 training.optimizer=adamw
```

#### Enabling Weights & Biases logging

```bash
python -m glaucoma_detection.run_pipeline logging.use_wandb=true logging.wandb_project=my-glaucoma-project
```

#### Using mixed precision training (for memory efficiency)

```bash
python -m glaucoma_detection.run_pipeline training.precision=16-mixed
```

#### Memory-efficient loading for large datasets

```bash
python -m glaucoma_detection.run_pipeline preprocessor.use_memory_efficient=true
```

## Pipeline Modules

### 1. Data Loading & Extraction

The data loading module (`data_loader.py`) handles:
- Extracting ZIP files containing datasets
- Loading ORIGA, REFUGE, and G1020 datasets
- Consolidating data from multiple sources
- Saving the combined dataset to CSV

Key functions:
- `extract_zip(zip_file, output_dir)`: Extracts a ZIP file to the specified directory
- `load_origa(dataset_path)`, `load_refuge(dataset_path)`, `load_g1020(dataset_path)`: Load individual datasets
- `consolidate_datasets(data_dir)`: Combine datasets into a single DataFrame

### 2. Data Cleaning

The data cleaning module (`data_cleaner.py`):
- Standardizes dataset splits (train/val/test)
- Handles missing values
- Performs data validation
- Ensures consistent labeling

Key functions:
- `standardize_dataset_splits(df, random_state)`: Create consistent train/validation/test splits
- `handle_missing_values(df)`: Clean missing or invalid entries
- `validate_dataset(df)`: Perform basic quality checks on the dataset
- `clean_dataset(df, random_state)`: Main function to clean the entire dataset

### 3. Preprocessing

The preprocessing module (`preprocessor.py`):
- Creates datasets and dataloaders
- Implements data augmentation with Albumentations
- Normalizes images
- Standardizes image sizes

Key components:
- `GlaucomaDataset`: PyTorch dataset for glaucoma images
- `GlaucomaDataModule`: PyTorch Lightning datamodule for managing data splits
- Memory-efficient loading via `memory_efficient_loader.py` for large datasets

Data augmentation techniques:
- Random rotation, flipping, scaling
- Brightness and contrast adjustments
- Gaussian noise and blur

### 4. Model Architecture

The model module (`model.py`):
- Uses Segmentation Models PyTorch for state-of-the-art architectures
- Supports multiple backbones (ResNet, EfficientNet, etc.)
- Provides pre-trained model initialization
- Implements various loss functions

Supported architectures:
- UNet
- UNet++
- DeepLabV3
- FPN (Feature Pyramid Network)

Key functions:
- `create_model(model_config)`: Create a model based on configuration
- `get_loss_function(loss_type)`: Get the appropriate loss function
- `save_model(model, filepath)`: Save model checkpoint
- `load_model(filepath, model_config)`: Load model from checkpoint

### 5. Training

The training module (`trainer.py`):
- Uses PyTorch Lightning for clean training loops
- Implements early stopping and checkpointing
- Tracks metrics during training
- Supports GPU acceleration

Key components:
- `GlaucomaSegmentationModel`: Lightning module for glaucoma segmentation
- `CombinedLoss`: Custom loss function combining Dice and BCE
- `setup_training(model, data_module, config, output_dir)`: Configure training
- `train_model(model, data_module, config, output_dir)`: Train the model

### 6. Evaluation

The evaluation module (`evaluator.py`):
- Calculates standardized metrics (Dice, IoU, accuracy, etc.)
- Generates visualizations of model predictions
- Creates ROC and PR curves
- Produces evaluation reports

Key components:
- `SegmentationEvaluator`: Main evaluation class
- Visualization functions for predictions, overlays, and curves

## Memory Optimization

The pipeline includes several features for memory optimization:

### Memory-Efficient Data Loading

For large datasets, use the memory-efficient data loader:

```bash
python -m glaucoma_detection.run_pipeline preprocessor.use_memory_efficient=true
```

This uses:
- Batch prefetching
- Image caching with reference counting
- Disk-based caching for large datasets
- Optimized OpenCV loading

### Mixed Precision Training

Enable mixed precision training to reduce memory usage:

```bash
python -m glaucoma_detection.run_pipeline training.precision=16-mixed
```

### Reducing Image Size

For systems with limited memory:

```bash
python -m glaucoma_detection.run_pipeline preprocessing.image_size=[128,128]
```

### Reducing Batch Size

For systems with limited memory:

```bash
python -m glaucoma_detection.run_pipeline training.batch_size=8
```

### Lightweight Models

Use smaller encoder backbones:

```bash
python -m glaucoma_detection.run_pipeline model.encoder=resnet18
```

## Experiment Tracking

The pipeline supports Weights & Biases for experiment tracking.

### Enabling Weights & Biases

```bash
python -m glaucoma_detection.run_pipeline logging.use_wandb=true
```

### Tracked Metrics

During training and evaluation, the pipeline tracks:
- Loss values (training and validation)
- Dice coefficient
- IoU (Intersection over Union)
- Accuracy, precision, recall, F1-score
- ROC and PR curves
- Sample image visualizations

### Viewing Run History

The pipeline maintains a log of runs in `output/notebook.md`, recording:
- Run timestamp
- Steps executed
- Run duration
- Key metrics

## Troubleshooting

### Dataset Issues

Use the dataset diagnostics tool to check your data setup:

```bash
python -m glaucoma_detection.diagnose_datasets
```

### GPU Memory Issues

Check GPU memory and get recommendations:

```bash
python -m glaucoma_detection.memory_check
```

### SSL Certificate Issues

If you encounter SSL certificate issues when downloading pretrained models:

```bash
python -m glaucoma_detection.ssl_fix
```

Or run the pipeline with SSL fixes applied:

```bash
python -m glaucoma_detection.run_with_fix
```

### Common Issues and Solutions

1. **"CUDA out of memory" error**:
   - Reduce batch size: `training.batch_size=8`
   - Use smaller images: `preprocessing.image_size=[128,128]`
   - Use mixed precision: `training.precision=16-mixed`
   - Use a smaller encoder: `model.encoder=resnet18`

2. **Slow data loading**:
   - Increase number of workers: `training.num_workers=8`
   - Enable memory-efficient loading: `preprocessor.use_memory_efficient=true`

3. **Model not converging**:
   - Try different loss functions: `training.loss_function=dice` or `training.loss_function=focal`
   - Adjust learning rate: `training.learning_rate=0.0001`
   - Increase epochs: `training.epochs=50`

4. **Missing or incomplete datasets**:
   - Check dataset paths: `paths.data_dir=/correct/path/to/data`
   - Run dataset diagnostics: `python -m glaucoma_detection.diagnose_datasets`

## Configuration Reference

### Main Configuration (`conf/config.yaml`)

```yaml
defaults:
  - data: default
  - model: unet
  - preprocessing: default
  - training: default
  - evaluation: default
  - logging: default
  - _self_

paths:
  base_dir: ${oc.env:BASE_DIR,${hydra:runtime.cwd}}
  data_dir: ${oc.env:DATA_DIR,${paths.base_dir}/data}
  output_dir: ${oc.env:OUTPUT_DIR,${paths.base_dir}/output}
  model_dir: ${paths.output_dir}/models
  log_dir: ${paths.output_dir}/logs

pipeline:
  steps: [extract, load, clean, preprocess, train, evaluate]
  force: false
  description: "Default pipeline run"
```

### Data Configuration (`conf/data/default.yaml`)

```yaml
zip_file: null  # Path to ZIP file if extraction is needed
random_state: 42

datasets:
  ORIGA:
    path: ${paths.data_dir}/ORIGA
    metadata_file: OrigaList.csv
    image_dir: Images
    mask_dir: Masks
  REFUGE:
    path: ${paths.data_dir}/REFUGE
    splits: [train, val, test]
    metadata_file: index.json
    image_dir: Images
    mask_dir: Masks
  G1020:
    path: ${paths.data_dir}/G1020
    metadata_file: G1020.csv
    image_dir: Images
    mask_dir: Masks

split_config:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

### Model Configuration (`conf/model/unet.yaml`)

```yaml
architecture: unet         # Options: unet, unet++, deeplabv3, fpn
encoder: resnet34          # Backbone encoder
pretrained: true           # Use pretrained weights
in_channels: 3             # RGB input
num_classes: 1             # Binary segmentation
```

### Preprocessing Configuration (`conf/preprocessing/default.yaml`)

```yaml
image_size: [224, 224]     # Width, Height
image_channels: 3          # RGB
normalization: imagenet    # 'imagenet', 'instance', 'pixel', 'none'
use_square_images: true    # Use square cropped images if available
use_cropped_images: true   # Use cropped optic disc images if available
mode: segmentation         # 'segmentation' or 'classification'

augmentation:
  enabled: true
  rotation_range: 15
  width_shift_range: 0.1
  height_shift_range: 0.1
  shear_range: 0.1
  zoom_range: 0.1
  horizontal_flip: true
  vertical_flip: false
  fill_mode: nearest
```

### Training Configuration (`conf/training/default.yaml`)

```yaml
epochs: 50
batch_size: 32
num_workers: 4
learning_rate: 0.001
optimizer: adam            # 'adam', 'sgd', 'adamw'
loss_function: combined    # 'combined', 'dice', 'bce', 'focal', 'jaccard'
precision: 32-true         # '16-mixed', '32-true'
use_gpu: true
gpu_ids: [0]
gradient_clip_val: 0.0
accumulate_grad_batches: 1

lr_scheduler:
  enabled: true
  factor: 0.1
  patience: 5
  min_lr: 0.000001
  monitor: val_loss

early_stopping:
  enabled: true
  patience: 10
  monitor: val_loss
  min_delta: 0.001
  mode: min

checkpointing:
  enabled: true
  save_top_k: 3
  monitor: val_loss
  mode: min

use_class_weights: true
```

### Evaluation Configuration (`conf/evaluation/default.yaml`)

```yaml
metrics: [dice, iou, accuracy, precision, recall, f1]
threshold: 0.5
num_samples: 5  # Number of sample visualizations

visualization:
  enabled: true
  plot_wrong_predictions: true
  plot_gradcam: true
```

### Logging Configuration (`conf/logging/default.yaml`)

```yaml
use_wandb: false
wandb_project: glaucoma-detection
log_every_n_steps: 10
```

## Development

### Code Style

This project follows best practices for Python code:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking
- pytest for testing

To run all code quality checks:

```bash
black .
isort .
flake8
mypy .
pytest
```

### Testing

The project includes comprehensive testing utilities in `test_utils.py`:
- `TestPipelineBase`: Base class for pipeline tests
- `create_test_environment`: Create a test environment
- `generate_test_dataset`: Generate synthetic test data
- `create_test_config`: Create a test configuration

To run tests:

```bash
pytest
```

### Adding a New Model Architecture

1. Add the architecture to the `create_model` function in `model.py`:

```python
elif architecture.lower() == 'new_architecture':
    model = smp.NewArchitecture(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation='sigmoid' if classes == 1 else 'softmax'
    )
```

2. Add the architecture to the schema in `config_validator.py`

3. Create a new configuration file in `conf/model/new_architecture.yaml`

### Adding a New Dataset

1. Create a loader function in `data_loader.py`:

```python
def load_new_dataset(dataset_path):
    """Load the new dataset."""
    # Implementation here
    return df
```

2. Update the dataset configuration in `conf/data/default.yaml`

## Acknowledgements

This project incorporates components from:
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Albumentations](https://github.com/albumentations-team/albumentations)
- [Hydra](https://github.com/facebookresearch/hydra)

---