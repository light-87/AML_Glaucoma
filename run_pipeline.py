"""
Pipeline Coordinator

This module coordinates the execution of the machine learning pipeline,
managing data loading, cleaning, preprocessing, model training, and evaluation.

Usage:
    python run_pipeline.py --base_path /path/to/data --steps extract,load,clean,preprocess,train,evaluate
"""

import os
import argparse
import pandas as pd
import logging
import datetime
import json
import uuid
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Import pipeline modules
from data_loader import extract_zip, consolidate_datasets, save_consolidated_dataset
from data_cleaner import clean_dataset
from preprocessor import preprocess_dataset, create_dataset_splits, create_dataloaders, GlaucomaDataset
from model import create_model, save_model, load_model
from trainer import train_model
from evaluator import evaluate_segmentation
from utils import setup_logger, create_directory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Class to coordinate the execution of the ML pipeline."""
    
    def __init__(self, base_path, output_dir='output', user_name=None, force_rerun=False):
        """
        Initialize the pipeline runner.
        
        Parameters:
        -----------
        base_path : str
            Base directory containing all datasets
        output_dir : str, optional
            Directory to save output files, by default 'output'
        user_name : str, optional
            User name for run tracking, by default None
        force_rerun : bool, optional
            Force rerun of all steps even if output files exist, by default False
        """
        self.base_path = base_path
        self.output_dir = output_dir
        self.user_name = user_name or os.environ.get('USER', 'unknown_user')
        self.force_rerun = force_rerun
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize run metadata
        self.run_id = str(uuid.uuid4())
        self.run_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_metadata = {
            'run_id': self.run_id,
            'timestamp': self.run_timestamp,
            'user': self.user_name,
            'base_path': base_path,
            'output_dir': output_dir,
            'modules_executed': [],
            'steps_completed': {},
            'description': '',
            'performance_metrics': {}
        }
        
        logger.info(f"Initialized pipeline with run ID: {self.run_id}")
    
    def _should_run_step(self, step_name, output_file):
        """
        Check if a step should be run based on the existence of output files.
        
        Parameters:
        -----------
        step_name : str
            Name of the step
        output_file : str
            Path to the expected output file
            
        Returns:
        --------
        bool
            True if the step should be run, False otherwise
        """
        if self.force_rerun:
            logger.info(f"Force rerun enabled. Running step: {step_name}")
            return True
        
        if not os.path.exists(output_file):
            logger.info(f"Output file not found. Running step: {step_name}")
            return True
        
        logger.info(f"Output file exists. Skipping step: {step_name}")
        return False
    
    def extract_data(self, zip_file=None):
        """
        Extract data from ZIP files if needed.
        
        Parameters:
        -----------
        zip_file : str, optional
            Path to the ZIP file to extract, by default None
            
        Returns:
        --------
        bool
            True if extraction was performed, False otherwise
        """
        if zip_file is None:
            # Look for ZIP files in the base path
            zip_files = [f for f in os.listdir(self.base_path) if f.endswith('.zip')]
            
            if not zip_files:
                logger.info("No ZIP files found to extract")
                return False
            
            # Use the first ZIP file found
            zip_file = os.path.join(self.base_path, zip_files[0])
        
        # Check if extraction is needed (if directories exist)
        extracted_dirs = ['ORIGA', 'REFUGE', 'G1020']
        existing_dirs = [d for d in extracted_dirs if os.path.exists(os.path.join(self.base_path, d))]
        
        if len(existing_dirs) == len(extracted_dirs) and not self.force_rerun:
            logger.info("All expected directories already exist. Skipping extraction.")
            return False
        
        # Extract the ZIP file
        try:
            extract_zip(zip_file, self.base_path)
            logger.info(f"Extracted {zip_file} to {self.base_path}")
            
            # Update run metadata
            self.run_metadata['steps_completed']['extract_data'] = {
                'zip_file': zip_file,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return True
        except Exception as e:
            logger.error(f"Error extracting ZIP file: {e}")
            return False
    
    def load_data(self):
        """
        Load and consolidate data from all datasets.
        
        Returns:
        --------
        pandas.DataFrame or None
            Consolidated DataFrame if successful, None otherwise
        """
        # Check if this step should be run
        consolidated_csv = os.path.join(self.output_dir, 'consolidated_glaucoma_dataset.csv')
        
        if not self._should_run_step('load_data', consolidated_csv):
            # Load the existing consolidated dataset
            return pd.read_csv(consolidated_csv)
        
        try:
            # Consolidate datasets
            all_data = consolidate_datasets(self.base_path)
            
            # Save consolidated dataset
            save_consolidated_dataset(all_data, consolidated_csv)
            
            # Update run metadata
            self.run_metadata['steps_completed']['load_data'] = {
                'output_file': consolidated_csv,
                'num_records': len(all_data),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.run_metadata['modules_executed'].append('data_loader')
            
            return all_data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def clean_data(self, df=None):
        """
        Clean the consolidated dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame, optional
            DataFrame to clean, by default None
            
        Returns:
        --------
        pandas.DataFrame or None
            Cleaned DataFrame if successful, None otherwise
        """
        # Check if this step should be run
        cleaned_csv = os.path.join(self.output_dir, 'cleaned_glaucoma_dataset.csv')
        
        if not self._should_run_step('clean_data', cleaned_csv):
            # Load the existing cleaned dataset
            return pd.read_csv(cleaned_csv)
        
        try:
            # If DataFrame is not provided, try to load the consolidated dataset
            if df is None:
                consolidated_csv = os.path.join(self.output_dir, 'consolidated_glaucoma_dataset.csv')
                if os.path.exists(consolidated_csv):
                    df = pd.read_csv(consolidated_csv)
                else:
                    logger.error(f"Consolidated dataset not found: {consolidated_csv}")
                    return None
            
            # Clean the dataset
            cleaned_df = clean_dataset(df)
            
            # Save cleaned dataset
            cleaned_df.to_csv(cleaned_csv, index=False)
            logger.info(f"Cleaned dataset saved to {cleaned_csv}")
            
            # Update run metadata
            self.run_metadata['steps_completed']['clean_data'] = {
                'output_file': cleaned_csv,
                'num_records': len(cleaned_df),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.run_metadata['modules_executed'].append('data_cleaner')
            
            return cleaned_df
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return None
    
    def preprocess_data(self, df=None):
        """
        Preprocess the cleaned dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame, optional
            DataFrame to preprocess, by default None
            
        Returns:
        --------
        tuple
            (preprocessed_df, train_df, val_df, test_df) or None on failure
        """
        # Check if this step should be run
        preprocessed_csv = os.path.join(self.output_dir, 'preprocessed_glaucoma_dataset.csv')
        
        if not self._should_run_step('preprocess_data', preprocessed_csv):
            # Load the existing preprocessed dataset and splits
            preprocessed_df = pd.read_csv(preprocessed_csv)
            train_df = pd.read_csv(os.path.join(self.output_dir, 'train_dataset.csv'))
            val_df = pd.read_csv(os.path.join(self.output_dir, 'val_dataset.csv'))
            test_df = pd.read_csv(os.path.join(self.output_dir, 'test_dataset.csv'))
            return preprocessed_df, train_df, val_df, test_df
        
        try:
            # If DataFrame is not provided, try to load the cleaned dataset
            if df is None:
                cleaned_csv = os.path.join(self.output_dir, 'cleaned_glaucoma_dataset.csv')
                if os.path.exists(cleaned_csv):
                    df = pd.read_csv(cleaned_csv)
                else:
                    logger.error(f"Cleaned dataset not found: {cleaned_csv}")
                    return None
            
            # Preprocess the dataset
            preprocessed_df = preprocess_dataset(df, self.output_dir, save_images=True)
            
            # Create dataset splits
            train_df, val_df, test_df = create_dataset_splits(
                preprocessed_df,
                val_size=0.15,
                test_size=0.15,
                random_state=42
            )
            
            # Save splits
            train_df.to_csv(os.path.join(self.output_dir, 'train_dataset.csv'), index=False)
            val_df.to_csv(os.path.join(self.output_dir, 'val_dataset.csv'), index=False)
            test_df.to_csv(os.path.join(self.output_dir, 'test_dataset.csv'), index=False)
            
            # Update run metadata
            self.run_metadata['steps_completed']['preprocess_data'] = {
                'output_file': preprocessed_csv,
                'num_records': len(preprocessed_df),
                'num_train': len(train_df),
                'num_val': len(val_df),
                'num_test': len(test_df),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.run_metadata['modules_executed'].append('preprocessor')
            
            return preprocessed_df, train_df, val_df, test_df
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return None
    
    def train_model(self, train_df=None, val_df=None):
        """
        Train the model.
        
        Parameters:
        -----------
        train_df : pandas.DataFrame, optional
            Training data, by default None
        val_df : pandas.DataFrame, optional
            Validation data, by default None
            
        Returns:
        --------
        tuple
            (model, history, best_epoch) or None on failure
        """
        # Check if this step should be run
        checkpoint_dir = os.path.join(self.output_dir, 'models')
        model_path = os.path.join(checkpoint_dir, 'glaucoma_model_best.pth')
        
        # Create checkpoint directory if it doesn't exist
        create_directory(checkpoint_dir)
        
        if not self._should_run_step('train_model', model_path):
            logger.info(f"Model already exists at {model_path}. Skipping training.")
            return True
        
        try:
            # If DataFrames are not provided, try to load them
            if train_df is None or val_df is None:
                train_csv = os.path.join(self.output_dir, 'train_dataset.csv')
                val_csv = os.path.join(self.output_dir, 'val_dataset.csv')
                
                if not os.path.exists(train_csv) or not os.path.exists(val_csv):
                    logger.error(f"Training or validation data not found")
                    return None
                
                train_df = pd.read_csv(train_csv)
                val_df = pd.read_csv(val_csv)
            
            # Create model
            model = create_model()
            
            # Create data loaders
            train_loader, val_loader, _ = create_dataloaders(
                train_df, 
                val_df, 
                batch_size=32, 
                num_workers=4
            )
            
            # Train model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Training model on {device}")
            
            model, history, best_epoch = train_model(
                train_loader, 
                val_loader, 
                model=model, 
                checkpoint_dir=checkpoint_dir,
                device=device
            )
            
            # Update run metadata
            self.run_metadata['steps_completed']['train_model'] = {
                'model_path': model_path,
                'best_epoch': best_epoch,
                'device': str(device),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.run_metadata['modules_executed'].append('trainer')
            
            return model, history, best_epoch
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None
    
    def evaluate_model(self, test_df=None):
        """
        Evaluate the trained model.
        
        Parameters:
        -----------
        test_df : pandas.DataFrame, optional
            Test data, by default None
            
        Returns:
        --------
        dict
            Evaluation metrics or None on failure
        """
        # Model and results paths
        model_path = os.path.join(self.output_dir, 'models', 'glaucoma_model_best.pth')
        results_dir = os.path.join(self.output_dir, 'evaluation')
        
        # Check if model exists
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return None
        
        # Create results directory if it doesn't exist
        create_directory(results_dir)
        
        try:
            # If DataFrame is not provided, try to load it
            if test_df is None:
                test_csv = os.path.join(self.output_dir, 'test_dataset.csv')
                
                if not os.path.exists(test_csv):
                    logger.error(f"Test data not found: {test_csv}")
                    return None
                
                test_df = pd.read_csv(test_csv)
            
            # Create data loader
            test_dataset = GlaucomaDataset(
                test_df,
                target_size=(224, 224),
                augment=False,
                mode='segmentation'
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=8,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # Evaluate model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            metrics = evaluate_segmentation(model_path, test_loader, results_dir, device)
            
            # Update run metadata
            if metrics:
                self.run_metadata['steps_completed']['evaluate_model'] = {
                    'metrics': {
                        'dice': metrics['dice_coefficient'],
                        'iou': metrics['iou'],
                        'accuracy': metrics['accuracy'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1_score': metrics['f1_score']
                    },
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self.run_metadata['modules_executed'].append('evaluator')
                self.run_metadata['performance_metrics'] = {
                    'dice': metrics['dice_coefficient'],
                    'iou': metrics['iou'],
                    'accuracy': metrics['accuracy']
                }
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None
    
    def update_run_tracking(self, description=None):
        """
        Update the run tracking file.
        
        Parameters:
        -----------
        description : str, optional
            Description of the run, by default None
        """
        # Update run description if provided
        if description:
            self.run_metadata['description'] = description
        
        # Save run metadata to JSON file
        run_json_file = os.path.join(self.output_dir, f"run_{self.run_id}.json")
        with open(run_json_file, 'w') as f:
            json.dump(self.run_metadata, f, indent=2)
        
        # Update notebook.md
        notebook_file = os.path.join(self.output_dir, 'notebook.md')
        
        # Create file with header if it doesn't exist
        if not os.path.exists(notebook_file):
            with open(notebook_file, 'w') as f:
                f.write("# Pipeline Run Tracking\n\n")
                f.write("| Run ID | Timestamp | User | Description | Modules | Metrics |\n")
                f.write("|--------|-----------|------|-------------|---------|----------|\n")
        
        # Append run information
        with open(notebook_file, 'a') as f:
            modules = ', '.join(self.run_metadata['modules_executed'])
            metrics = ', '.join([f"{k}: {v}" for k, v in self.run_metadata['performance_metrics'].items()])
            
            f.write(f"| {self.run_id} | {self.run_timestamp} | {self.user_name} | ")
            f.write(f"{self.run_metadata['description']} | {modules} | {metrics} |\n")
        
        logger.info(f"Run tracking updated: {run_json_file}")
        logger.info(f"Run notebook updated: {notebook_file}")

def main():
    """
    Main function to run the pipeline.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run the glaucoma detection ML pipeline')
    parser.add_argument('--base_path', type=str, default='/content',
                        help='Base directory containing all datasets')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output files')
    parser.add_argument('--user', type=str, default=None,
                        help='User name for run tracking')
    parser.add_argument('--description', type=str, default='',
                        help='Description of the pipeline run')
    parser.add_argument('--zip_file', type=str, default=None,
                        help='Path to the ZIP file to extract')
    parser.add_argument('--steps', type=str, default='extract,load,clean,preprocess,train,evaluate',
                        help='Comma-separated list of steps to run')
    parser.add_argument('--force', action='store_true',
                        help='Force rerun of all steps even if output files exist')
    
    args = parser.parse_args()
    
    # Initialize pipeline runner
    pipeline = PipelineRunner(
        base_path=args.base_path,
        output_dir=args.output_dir,
        user_name=args.user,
        force_rerun=args.force
    )
    
    # Run the specified steps
    steps = args.steps.split(',')
    step_results = {}
    
    # Extract data
    if 'extract' in steps:
        step_results['extract'] = pipeline.extract_data(args.zip_file)
    
    # Load data
    if 'load' in steps:
        step_results['load'] = pipeline.load_data()
    
    # Clean data
    if 'clean' in steps:
        df = step_results.get('load', None)
        step_results['clean'] = pipeline.clean_data(df)
    
    # Preprocess data
    if 'preprocess' in steps:
        df = step_results.get('clean', None)
        step_results['preprocess'] = pipeline.preprocess_data(df)
    
    # Train model
    if 'train' in steps:
        preprocessed_results = step_results.get('preprocess', None)
        if preprocessed_results:
            train_df = preprocessed_results[1]  # train_df is the second item in the tuple
            val_df = preprocessed_results[2]    # val_df is the third item in the tuple
            step_results['train'] = pipeline.train_model(train_df, val_df)
        else:
            step_results['train'] = pipeline.train_model()
    
    # Evaluate model
    if 'evaluate' in steps:
        preprocessed_results = step_results.get('preprocess', None)
        if preprocessed_results:
            test_df = preprocessed_results[3]   # test_df is the fourth item in the tuple
            step_results['evaluate'] = pipeline.evaluate_model(test_df)
        else:
            step_results['evaluate'] = pipeline.evaluate_model()
    
    # Update run tracking
    pipeline.update_run_tracking(args.description)
    
    # Print summary
    logger.info("Pipeline execution completed")
    logger.info(f"Steps run: {steps}")
    for step, result in step_results.items():
        if step == 'extract':
            logger.info(f"  - {step}: {'Success' if result else 'Skipped or Failed'}")
        else:
            logger.info(f"  - {step}: {'Success' if result is not None else 'Failed'}")
    
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Run ID: {pipeline.run_id}")

if __name__ == "__main__":
    main()