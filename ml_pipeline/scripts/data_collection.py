"""
Data Collection Script for ML Pipeline
Generates synthetic data for classification and stores it with DVC/S3 integration
"""
import os
import pandas as pd
import numpy as np
import yaml
import boto3
import logging
from sklearn.datasets import make_classification
from pathlib import Path
from typing import Optional
import mlflow
import mlflow.sklearn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def generate_synthetic_data(config: dict) -> pd.DataFrame:
    """Generate synthetic classification dataset"""
    params = config['data_sources']['primary']['parameters']
    
    X, y = make_classification(
        n_samples=params['n_samples'],
        n_features=params['n_features'],
        n_informative=params['n_informative'],
        n_redundant=params['n_redundant'],
        n_classes=params['n_classes'],
        random_state=params['random_state']
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    logger.info(f"Generated dataset with shape: {df.shape}")
    return df

def upload_to_s3(df: pd.DataFrame, config: dict) -> Optional[str]:
    """Upload dataset to S3"""
    s3_config = config['data_sources']['s3_backup']
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=s3_config['region'])
    
    # Save to local temp file first
    temp_file = '/tmp/dataset.csv'
    df.to_csv(temp_file, index=False)
    
    # Upload to S3
    s3_key = f"{s3_config['prefix']}dataset.csv"
    try:
        s3_client.upload_file(temp_file, s3_config['bucket'], s3_key)
        s3_uri = f"s3://{s3_config['bucket']}/{s3_key}"
        logger.info(f"Dataset uploaded to S3: {s3_uri}")
        return s3_uri
    except Exception as e:
        logger.warning(f"Failed to upload to S3: {e}")
        return None

def validate_data(df: pd.DataFrame, config: dict) -> dict:
    """Perform data quality validation"""
    validation_results = {
        'n_samples': len(df),
        'n_features': len(df.columns) - 1,  # Exclude target
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'target_distribution': df['target'].value_counts().to_dict()
    }
    
    # Check quality thresholds
    quality_checks = config['data_validation']['quality_checks']
    validation_results['quality_passed'] = True
    
    if df.isnull().sum().sum() / len(df) > quality_checks[0]['missing_values_threshold']:
        validation_results['quality_passed'] = False
        logger.warning("Data quality check failed: too many missing values")
    
    if df.duplicated().sum() / len(df) > quality_checks[1]['duplicate_threshold']:
        validation_results['quality_passed'] = False
        logger.warning("Data quality check failed: too many duplicates")
    
    return validation_results

def main():
    """Main data collection pipeline"""
    # Load configuration
    config = load_config('config/data_config.yaml')
    
    # Start MLflow run
    mlflow.set_experiment("data-collection")
    
    with mlflow.start_run(run_name="data_collection"):
        # Generate synthetic data
        df = generate_synthetic_data(config)
        
        # Validate data
        validation_results = validate_data(df, config)
        
        # Log metrics to MLflow
        mlflow.log_params(config['data_sources']['primary']['parameters'])
        mlflow.log_metrics({
            'n_samples': validation_results['n_samples'],
            'n_features': validation_results['n_features'],
            'missing_values': validation_results['missing_values'],
            'duplicates': validation_results['duplicates']
        })
        
        # Log data quality results
        for class_label, count in validation_results['target_distribution'].items():
            mlflow.log_metric(f'class_{class_label}_count', count)
        
        # Upload to S3 (optional)
        s3_uri = upload_to_s3(df, config)
        if s3_uri:
            mlflow.log_param('s3_location', s3_uri)
        
        # Create output directory
        output_dir = Path('ml_pipeline/data/raw')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dataset
        output_path = output_dir / 'dataset.csv'
        df.to_csv(output_path, index=False)
        
        # Log dataset as artifact
        mlflow.log_artifact(str(output_path), 'raw_data')
        
        logger.info(f"Dataset saved to: {output_path}")
        logger.info(f"Data validation passed: {validation_results['quality_passed']}")

if __name__ == "__main__":
    main()