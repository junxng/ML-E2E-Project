"""
Data Preprocessing Script for ML Pipeline
Handles feature engineering, scaling, and train/test split with MLFlow tracking
"""
import pandas as pd
import numpy as np
import yaml
import logging
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset from CSV file"""
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df

def handle_missing_values(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Handle missing values in the dataset"""
    missing_config = config['preprocessing']['feature_engineering']['encoding']['numerical']
    
    if missing_config['handle_missing'] == 'drop':
        df_clean = df.dropna()
    else:
        # Use imputation
        strategy_map = {
            'mean': 'mean',
            'median': 'median',
            'mode': 'most_frequent'
        }
        strategy = strategy_map.get(missing_config['handle_missing'], 'median')
        
        # Separate numeric and categorical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_columns:
            numeric_columns.remove('target')
        
        imputer = SimpleImputer(strategy=strategy)
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        df_clean = df
    
    logger.info(f"After handling missing values, shape: {df_clean.shape}")
    return df_clean

def remove_outliers(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Remove outliers using IQR method"""
    outlier_method = config['preprocessing']['feature_engineering']['encoding']['numerical']['outlier_treatment']
    
    if outlier_method == 'iqr':
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_columns:
            numeric_columns.remove('target')
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    logger.info(f"After outlier removal, shape: {df.shape}")
    return df

def feature_scaling(X: pd.DataFrame, config: dict, fit_scaler: bool = True) -> Tuple[pd.DataFrame, Any]:
    """Apply feature scaling to the dataset"""
    scaling_config = config['preprocessing']['feature_engineering']['scaling']
    scaling_method = scaling_config['method']
    
    # Select scaler
    scaler_map = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    
    scaler = scaler_map.get(scaling_method, StandardScaler())
    
    # Exclude specified columns from scaling
    exclude_cols = scaling_config.get('exclude_columns', [])
    columns_to_scale = [col for col in X.columns if col not in exclude_cols]
    
    if fit_scaler:
        X_scaled = X.copy()
        X_scaled[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])
    else:
        X_scaled = X.copy()
        X_scaled[columns_to_scale] = scaler.transform(X[columns_to_scale])
    
    logger.info(f"Applied {scaling_method} scaling to {len(columns_to_scale)} features")
    return X_scaled, scaler

def feature_selection(X: pd.DataFrame, y: pd.Series, config: dict) -> Tuple[pd.DataFrame, Any]:
    """Apply feature selection"""
    selection_config = config['preprocessing']['feature_engineering']['feature_selection']
    method = selection_config['method']
    
    if method == 'selectkbest':
        k = selection_config['k']
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = pd.DataFrame(
            selector.fit_transform(X, y),
            columns=X.columns[selector.get_support()],
            index=X.index
        )
    elif method == 'rfe':
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        k = selection_config['k']
        selector = RFE(estimator=estimator, n_features_to_select=k)
        X_selected = pd.DataFrame(
            selector.fit_transform(X, y),
            columns=X.columns[selector.get_support()],
            index=X.index
        )
    else:
        # No feature selection
        X_selected = X.copy()
        selector = None
    
    logger.info(f"Feature selection: {X.shape[1]} -> {X_selected.shape[1]} features")
    return X_selected, selector

def split_data(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets"""
    split_config = config['preprocessing']['train_test_split']
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    stratify = y if split_config['stratify'] else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_config['test_size'],
        random_state=split_config['random_state'],
        stratify=stratify
    )
    
    logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def save_artifacts(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                  y_train: pd.Series, y_test: pd.Series,
                  scaler: Any, selector: Any) -> None:
    """Save processed data and preprocessing artifacts"""
    # Create output directory
    output_dir = Path('ml_pipeline/data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train/test data
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_df.to_csv(output_dir / 'train.csv', index=False)
    
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df.to_csv(output_dir / 'test.csv', index=False)
    
    # Save preprocessing artifacts
    preprocessing_artifacts = {
        'scaler': scaler,
        'feature_selector': selector,
        'feature_names': list(X_train.columns)
    }
    
    joblib.dump(preprocessing_artifacts, output_dir / 'features.pkl')
    
    logger.info(f"Saved processed data and artifacts to {output_dir}")

def main():
    """Main data preprocessing pipeline"""
    # Load configuration
    config = load_config('config/preprocessing_config.yaml')
    
    # Start MLflow run
    mlflow.set_experiment("data-preprocessing")
    
    with mlflow.start_run(run_name="data_preprocessing"):
        # Load raw data
        df = load_data('ml_pipeline/data/raw/dataset.csv')
        
        # Log initial data info
        mlflow.log_metrics({
            'raw_samples': len(df),
            'raw_features': len(df.columns) - 1,
            'raw_missing_values': df.isnull().sum().sum()
        })
        
        # Data quality checks
        if config['data_quality']['remove_duplicates']:
            initial_shape = df.shape[0]
            df = df.drop_duplicates()
            duplicates_removed = initial_shape - df.shape[0]
            mlflow.log_metric('duplicates_removed', duplicates_removed)
        
        # Handle missing values
        df = handle_missing_values(df, config)
        
        # Remove outliers
        df = remove_outliers(df, config)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(df, config)
        
        # Feature scaling
        X_train_scaled, scaler = feature_scaling(X_train, config, fit_scaler=True)
        X_test_scaled, _ = feature_scaling(X_test, config, fit_scaler=False)
        
        # Feature selection
        X_train_selected, selector = feature_selection(X_train_scaled, y_train, config)
        X_test_selected = pd.DataFrame(
            selector.transform(X_test_scaled) if selector else X_test_scaled,
            columns=X_train_selected.columns,
            index=X_test_scaled.index
        )
        
        # Log preprocessing metrics
        mlflow.log_metrics({
            'final_train_samples': len(X_train_selected),
            'final_test_samples': len(X_test_selected),
            'final_features': len(X_train_selected.columns),
            'class_0_train': (y_train == 0).sum(),
            'class_1_train': (y_train == 1).sum(),
            'class_2_train': (y_train == 2).sum() if (y_train == 2).any() else 0
        })
        
        # Log configuration
        mlflow.log_params({
            'test_size': config['preprocessing']['train_test_split']['test_size'],
            'scaling_method': config['preprocessing']['feature_engineering']['scaling']['method'],
            'feature_selection_method': config['preprocessing']['feature_engineering']['feature_selection']['method'],
            'selected_features_k': config['preprocessing']['feature_engineering']['feature_selection']['k']
        })
        
        # Save artifacts
        save_artifacts(X_train_selected, X_test_selected, y_train, y_test, scaler, selector)
        
        # Log artifacts to MLflow
        mlflow.log_artifacts('ml_pipeline/data/processed', 'processed_data')
        
        logger.info("Data preprocessing completed successfully")

if __name__ == "__main__":
    main()