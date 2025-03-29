# Glaucoma Detection Pipeline

A modular machine learning pipeline for glaucoma detection from fundus images, using modern libraries and best practices.

## Project Overview

This project implements a complete machine learning pipeline for glaucoma detection from retinal fundus images. The pipeline is built with modern libraries like PyTorch Lightning, Hydra, and Segmentation Models PyTorch, making it modular, maintainable, and easy to experiment with.

### Datasets

The pipeline works with three glaucoma datasets:

1. **ORIGA** - Singapore Chinese Eye Study
2. **REFUGE** - Retinal Fundus Glaucoma Challenge
3. **G1020** - A large collection of retinal images

## Features

- **Modern Architecture**: Built with PyTorch Lightning, Hydra, and Segmentation Models PyTorch
- **Experiment Tracking**: Integration with Weights & Biases for experiment tracking
- **Modular Design**: Well-organized, maintainable codebase with clear separation of concerns
- **Flexible Configuration**: Hydra-based configuration system for easy experimentation
- **Reproducibility**: Random seeds, configuration versioning, and explicit dependencies
- **User-Friendly CLI**: Command-line interface with Typer for easy execution
- **Standardized Metrics**: Using torchmetrics for consistent evaluation
- **Performance**: Optimized data loading and processing pipelines

## Installation

### Prerequisites

- Python 3.7+
- pip or conda

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
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

### Data Setup

1. Download the datasets:
   - Place the datasets in the `data` directory, or specify a custom path
   - The pipeline expects the following structure:
     ```
     data_dir/
     ├── ORIGA/
     │   ├── Images/
     │   ├── Masks/
     │   └── OrigaList.csv
     ├── REFUGE/
     │   ├── train/
     │   ├── val/
     │   ├── test/
     └── G1020/
         ├── Images/
         ├── Masks/
         └── G1020.csv
     ```

2. Alternatively, download a ZIP file containing all datasets and extract it:

```bash
python -m glaucoma_detection.run_pipeline --zip-file /path/to/fundus_dataset.zip --steps extract
```

## Usage

### Command Line Interface

The pipeline can be run using the command line interface:

```bash
# Run the complete pipeline
python -m glaucoma_detection.run_pipeline

# Run specific steps
python -m glaucoma_detection.run_pipeline --steps extract,load,clean

# Use custom configuration
python -m glaucoma_detection.run_pipeline --config path/to/config.yaml

# Force rerun steps
python -m glaucoma_detection.run_pipeline --steps train,evaluate --force

# Enable wandb logging
python -m glaucoma_detection.run_pipeline --wandb
```

### Configuration

The pipeline uses Hydra for configuration. The main configuration file is `conf/config.yaml`, but you can override settings using:

1. Command line arguments:
   ```bash
   python -m glaucoma_detection.run_pipeline paths.data_dir=/custom/data/path model=unet++
   ```

2. Configuration files:
   ```bash
   python -m glaucoma_detection.run_pipeline --config path/to/config.yaml
   ```

3. Environment variables:
   ```bash
   export DATA_DIR=/custom/data/path
   python -m glaucoma_detection.run_pipeline
   ```

## Pipeline Modules

### 1. Data Loading & Extraction

The data loading module handles:
- Extracting ZIP files
- Consolidating data from multiple datasets
- Standardizing data formats

### 2. Data Cleaning

The data cleaning module:
- Standardizes dataset splits
- Handles missing values
- Removes redundant columns

### 3. Preprocessing

The preprocessing module:
- Implements data augmentation with Albumentations
- Normalizes images
- Creates datasets and dataloaders

### 4. Model Architecture

The model module:
- Uses Segmentation Models PyTorch for state-of-the-art models
- Supports various backbones (ResNet, EfficientNet, etc.)
- Provides pre-trained models for transfer learning

### 5. Training

The training module:
- Uses PyTorch Lightning for clean training loops
- Implements early stopping and checkpointing
- Tracks metrics during training

### 6. Evaluation

The evaluation module:
- Calculates standardized metrics with torchmetrics
- Generates visualizations of model predictions
- Produces evaluation reports

## Development

### Code Style

This project follows best practices for Python code:
- Black for code formatting
- isort for import sorting
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

### Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the tests
5. Submit a pull request


~vaibhav