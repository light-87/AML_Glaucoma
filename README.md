# Glaucoma Detection Pipeline

A modular machine learning pipeline for glaucoma detection from fundus images.

## Project Overview

This project implements a complete machine learning pipeline for glaucoma detection from retinal fundus images. The pipeline includes data loading, cleaning, preprocessing, model training, and evaluation modules. The architecture is designed to be modular, maintainable, and follows best practices for machine learning projects.

### Datasets

The pipeline is designed to work with three glaucoma datasets:

1. **ORIGA** - Singapore Chinese Eye Study
2. **REFUGE** - Retinal Fundus Glaucoma Challenge
3. **G1020** - A large collection of retinal images

## Installation

### Prerequisites

- Python 3.7+
- pip

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

### Data Setup

1. Download the datasets:
   - Place the datasets in the `/content` directory if using Google Colab, or specify a custom path
   - The pipeline expects the following structure:
     ```
     base_path/
     ├── ORIGA/
     │   ├── Images/
     │   ├── Masks/
     │   ├── OrigaList.csv
     │   └── ...
     ├── REFUGE/
     │   ├── train/
     │   ├── val/
     │   ├── test/
     │   └── ...
     └── G1020/
         ├── Images/
         ├── Masks/
         ├── G1020.csv
         └── ...
     ```

2. Alternatively, download the ZIP file containing all datasets and extract it:

```bash
python run_pipeline.py --zip_file /path/to/fundus_dataset.zip --steps extract
```

## Directory Structure

```
glaucoma-detection/
├── data_loader.py        # Data extraction and loading module
├── data_cleaner.py       # Data cleaning and standardization
├── preprocessor.py       # Data preprocessing and augmentation
├── model.py              # Model architecture definition
├── trainer.py            # Training loop implementation
├── evaluator.py          # Model evaluation and metrics
├── utils.py              # Utility functions
├── config.py             # Configuration parameters
├── run_pipeline.py       # Pipeline coordinator
├── README.md             # Project documentation
├── requirements.txt      # Package dependencies
└── output/               # Generated outputs
    ├── consolidated_glaucoma_dataset.csv
    ├── cleaned_glaucoma_dataset.csv
    ├── preprocessed_glaucoma_dataset.csv
    ├── notebook.md       # Run tracking log
    ├── run_*.json        # Run metadata
    └── visualizations/   # Generated visualizations
```

## Pipeline Modules

### 1. Data Loading & Extraction (`data_loader.py`)

Handles the initial data acquisition, extraction, and consolidation:
- Extracts ZIP files
- Combines data into CSV format
- Loads data from ORIGA, REFUGE, and G1020 datasets

```python
from data_loader import consolidate_datasets, save_consolidated_dataset

# Consolidate datasets
df = consolidate_datasets('/path/to/datasets')
save_consolidated_dataset(df, 'consolidated_glaucoma_dataset.csv')
```

### 2. Data Cleaning (`data_cleaner.py`)

Cleans and standardizes the raw data:
- Standardizes dataset splits across all sources
- Handles missing values
- Removes redundant columns
- Validates data structure

```python
from data_cleaner import clean_dataset

# Clean the dataset
cleaned_df = clean_dataset(df)
```

### 3. Preprocessing (`preprocessor.py`)

Transforms data into model-ready format:
- Implements data augmentation
- Normalizes images
- Splits data into training/validation/test sets
- Handles class imbalance

### 4. Model Architecture (`model.py`)

Defines the neural network architecture:
- Supports various backbone models (ResNet50, VGG16, etc.)
- Creates custom model with configurable parameters
- Handles transfer learning

### 5. Training (`trainer.py`)

Implements the training process:
- Training loop with configurable hyperparameters
- Learning rate scheduling
- Early stopping
- Checkpointing

### 6. Evaluation (`evaluator.py`)

Evaluates model performance:
- Calculates performance metrics
- Generates confusion matrices
- Plots ROC curves
- Visualizes model predictions

### 7. Utilities (`utils.py`)

Provides common utility functions:
- File handling
- Logging
- Dataset statistics
- Path validation

### 8. Configuration (`config.py`)

Centralizes all configurable parameters:
- Data paths and settings
- Preprocessing parameters
- Model architecture settings
- Training hyperparameters
- Evaluation metrics

### 9. Pipeline Coordinator (`run_pipeline.py`)

Orchestrates the entire workflow:
- Runs all pipeline steps sequentially
- Implements conditional execution
- Tracks experiment runs
- Manages command-line interface

## Run Tracking System

The pipeline includes an automatic run tracking system that logs each execution:

- Each run gets a unique ID and timestamp
- Metadata is saved in a JSON file
- A `notebook.md` file summarizes all runs
- Performance metrics are recorded

## Usage

### Running the Pipeline

To run the complete pipeline:

```bash
python run_pipeline.py --base_path /path/to/datasets --output_dir output --description "Complete pipeline run"
```

### Running Specific Steps

To run specific steps of the pipeline:

```bash
python run_pipeline.py --steps extract,load,clean --base_path /path/to/datasets
```

Available steps:
- `extract`: Extract datasets from ZIP files
- `load`: Load and consolidate datasets
- `clean`: Clean and standardize the data
- `preprocess`: Preprocess the data for model training
- `train`: Train the model
- `evaluate`: Evaluate model performance

### Force Rerun

To force rerun of steps even if output files exist:

```bash
python run_pipeline.py --steps load,clean --force
```

### Configuration

To use a custom configuration:

```bash
python run_pipeline.py --config custom_config.json
```

## Example Workflow

1. Download and extract the datasets:
   ```bash
   python run_pipeline.py --zip_file /path/to/fundus_dataset.zip --steps extract
   ```

2. Load and clean the data:
   ```bash
   python run_pipeline.py --steps load,clean
   ```

3. Preprocess the data and train the model:
   ```bash
   python run_pipeline.py --steps preprocess,train
   ```

4. Evaluate the model:
   ```bash
   python run_pipeline.py --steps evaluate
   ```

## Contributing

Contributions to improve the pipeline are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add your feature'`)
5. Push to the branch (`git push origin feature/your-feature`)
6. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The ORIGA, REFUGE, and G1020 datasets for providing valuable data for glaucoma research
- Contributors and researchers in the field of glaucoma detection