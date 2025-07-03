"""
Model Training Script for ML Pipeline
Trains ML models with hyperparameter tuning and MLFlow experiment tracking
"""
import pandas as pd
import numpy as np
import yaml
import logging
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load processed training and test data"""
    train_df = pd.read_csv('ml_pipeline/data/processed/train.csv')
    test_df = pd.read_csv('ml_pipeline/data/processed/test.csv')
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    logger.info(f"Loaded train data: {X_train.shape}, test data: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def get_model(algorithm: str, hyperparams: dict) -> Any:
    """Initialize model based on algorithm and hyperparameters"""
    models = {
        'random_forest': RandomForestClassifier,
        'logistic_regression': LogisticRegression
    }
    
    if algorithm not in models:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    model_class = models[algorithm]
    model = model_class(**hyperparams)
    
    return model

def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series, 
                         algorithm: str, config: dict) -> Tuple[Any, Dict]:
    """Perform hyperparameter tuning using GridSearch or RandomizedSearch"""
    training_config = config['training']
    optimization_method = training_config['optimization']['method']
    
    # Define hyperparameter grids
    param_grids = {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'logistic_regression': {
            'C': [0.1, 1.0, 10.0, 100.0],
            'max_iter': [100, 500, 1000],
            'solver': ['liblinear', 'lbfgs']
        }
    }
    
    if algorithm not in param_grids:
        # Use default hyperparameters
        hyperparams = config['model']['hyperparameters'][algorithm]
        model = get_model(algorithm, hyperparams)
        model.fit(X_train, y_train)
        return model, hyperparams
    
    # Get base model
    base_hyperparams = config['model']['hyperparameters'][algorithm]
    base_model = get_model(algorithm, base_hyperparams)
    
    # Setup search
    param_grid = param_grids[algorithm]
    cv_folds = training_config['optimization']['cv_folds']
    scoring = training_config['optimization']['scoring']
    
    if optimization_method == 'grid_search':
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
    elif optimization_method == 'random_search':
        n_iter = training_config['optimization']['n_iter']
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    else:
        # No optimization, use default params
        model = base_model
        model.fit(X_train, y_train)
        return model, base_hyperparams
    
    # Perform search
    logger.info(f"Starting {optimization_method} for {algorithm}")
    search.fit(X_train, y_train)
    
    # Log search results
    logger.info(f"Best score: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1_score': f1_score(y_test, y_pred, average='macro')
    }
    
    # ROC AUC for multiclass
    try:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    except Exception as e:
        logger.warning(f"Could not calculate ROC AUC: {e}")
        metrics['roc_auc'] = 0.0
    
    return metrics

def cross_validate_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series, 
                        config: dict) -> Dict[str, float]:
    """Perform cross-validation"""
    cv_config = config['preprocessing']['validation']['cross_validation']
    cv_folds = cv_config['cv_folds']
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1_macro')
    
    cv_metrics = {
        'cv_mean_score': cv_scores.mean(),
        'cv_std_score': cv_scores.std(),
        'cv_min_score': cv_scores.min(),
        'cv_max_score': cv_scores.max()
    }
    
    return cv_metrics

def save_model_artifacts(model: Any, metrics: Dict, best_params: Dict, 
                        X_train: pd.DataFrame) -> None:
    """Save model and metrics artifacts"""
    # Create output directory
    output_dir = Path('ml_pipeline/models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / 'model.pkl'
    joblib.dump(model, model_path)
    
    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save best parameters
    params_path = output_dir / 'best_params.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    logger.info(f"Model artifacts saved to {output_dir}")

def register_model_mlflow(model: Any, model_name: str, X_train: pd.DataFrame, 
                         metrics: Dict, stage: str = "staging") -> str:
    """Register model in MLFlow Model Registry"""
    # Create model signature
    signature = infer_signature(X_train, model.predict(X_train))
    
    # Log model to MLflow
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name=model_name
    )
    
    # Get the model version
    model_version = model_info.registered_model_version
    
    # Transition model to specified stage
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=stage
    )
    
    logger.info(f"Model registered as {model_name} v{model_version} in {stage} stage")
    return model_version

def main():
    """Main model training pipeline"""
    # Load configuration
    config = load_config('config/model_config.yaml')
    
    # Set MLflow experiment
    experiment_name = config['mlflow']['experiment_name']
    mlflow.set_experiment(experiment_name)
    
    # Set tracking URI (if different from default)
    if 'tracking_uri' in config['mlflow']:
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    with mlflow.start_run(run_name="model_training"):
        # Load data
        X_train, X_test, y_train, y_test = load_data()
        
        # Get model configuration
        algorithm = config['model']['algorithm']
        
        # Log basic info
        mlflow.log_params({
            'algorithm': algorithm,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': len(X_train.columns)
        })
        
        # Hyperparameter tuning
        model, best_params = hyperparameter_tuning(X_train, y_train, algorithm, config)
        
        # Log best hyperparameters
        mlflow.log_params(best_params)
        
        # Cross-validation
        cv_metrics = cross_validate_model(model, X_train, y_train, config)
        mlflow.log_metrics(cv_metrics)
        
        # Final evaluation
        test_metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(test_metrics)
        
        # Combine all metrics
        all_metrics = {**cv_metrics, **test_metrics}
        
        # Save artifacts locally
        save_model_artifacts(model, all_metrics, best_params, X_train)
        
        # Log artifacts to MLflow
        mlflow.log_artifacts('ml_pipeline/models', 'model_artifacts')
        
        # Register model in MLflow Model Registry
        deployment_config = config['deployment']
        model_version = register_model_mlflow(
            model=model,
            model_name=deployment_config['model_name'],
            X_train=X_train,
            metrics=all_metrics,
            stage=deployment_config['model_stage']
        )
        
        # Log model version
        mlflow.log_param('model_version', model_version)
        
        # Log final results
        logger.info(f"Training completed. Test F1-score: {test_metrics['f1_score']:.4f}")
        logger.info(f"Model version: {model_version}")
        
        return model, all_metrics

if __name__ == "__main__":
    main()