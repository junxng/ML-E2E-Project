"""
Model Evaluation Script for ML Pipeline
Comprehensive model evaluation with visualizations and MLFlow tracking
"""
import pandas as pd
import numpy as np
import yaml
import logging
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import mlflow
import mlflow.sklearn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_model_and_data() -> Tuple[Any, pd.DataFrame, pd.Series, list]:
    """Load trained model and test data"""
    # Load model
    model = joblib.load('ml_pipeline/models/model.pkl')
    
    # Load test data
    test_df = pd.read_csv('ml_pipeline/data/processed/test.csv')
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    # Load feature names
    features_artifacts = joblib.load('ml_pipeline/data/processed/features.pkl')
    feature_names = features_artifacts['feature_names']
    
    logger.info(f"Loaded model and test data: {X_test.shape}")
    return model, X_test, y_test, feature_names

def calculate_detailed_metrics(model: Any, X_test: pd.DataFrame, 
                             y_test: pd.Series) -> Tuple[Dict[str, Any], Any, Any, Any]:
    """Calculate comprehensive evaluation metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Basic metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'precision_micro': precision_score(y_test, y_pred, average='micro'),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'recall_micro': recall_score(y_test, y_pred, average='micro'),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_micro': f1_score(y_test, y_pred, average='micro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted')
    }
    
    # Per-class metrics
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Add per-class metrics
    for class_label in np.unique(y_test):
        class_str = str(class_label)
        if class_str in class_report:
            metrics[f'precision_class_{class_label}'] = class_report[class_str]['precision']
            metrics[f'recall_class_{class_label}'] = class_report[class_str]['recall']
            metrics[f'f1_class_{class_label}'] = class_report[class_str]['f1-score']
    
    return metrics, y_pred, y_pred_proba, class_report

def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, 
                         output_dir: Path) -> str:
    """Generate and save confusion matrix plot"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plot_path = output_dir / 'confusion_matrix.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {plot_path}")
    return str(plot_path)

def plot_feature_importance(model: Any, feature_names: list, 
                          output_dir: Path) -> Optional[str]:
    """Generate and save feature importance plot"""
    # Check if model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        
        plot_path = output_dir / 'feature_importance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {plot_path}")
        return str(plot_path)
    else:
        logger.warning("Model does not have feature_importances_ attribute")
        return None

def plot_roc_curves(y_test: pd.Series, y_pred_proba: np.ndarray, 
                   output_dir: Path) -> str:
    """Generate and save ROC curves for multiclass classification"""
    n_classes = len(np.unique(y_test))
    
    if n_classes == 2:
        # Binary classification
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
    else:
        # Multiclass classification
        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
        
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multiclass')
        plt.legend()
    
    plot_path = output_dir / 'roc_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC curves saved to {plot_path}")
    return str(plot_path)

def plot_precision_recall_curves(y_test: pd.Series, y_pred_proba: np.ndarray, 
                                output_dir: Path) -> str:
    """Generate and save Precision-Recall curves"""
    n_classes = len(np.unique(y_test))
    
    if n_classes == 2:
        # Binary classification
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
        avg_precision = average_precision_score(y_test, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'AP = {avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        
    else:
        # Multiclass classification
        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
        
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            avg_precision = average_precision_score(y_test_binarized[:, i], y_pred_proba[:, i])
            plt.plot(recall, precision, label=f'Class {i} (AP = {avg_precision:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Multiclass')
        plt.legend()
    
    plot_path = output_dir / 'precision_recall_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Precision-Recall curves saved to {plot_path}")
    return str(plot_path)

def save_evaluation_results(metrics: Dict, class_report: Dict, 
                          output_dir: Path) -> None:
    """Save detailed evaluation results"""
    # Save metrics
    metrics_path = output_dir / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save classification report
    report_path = output_dir / 'classification_report.json'
    with open(report_path, 'w') as f:
        json.dump(class_report, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_dir}")

def main():
    """Main model evaluation pipeline"""
    # Load configuration
    config = load_config('config/model_config.yaml')
    
    # Set MLflow experiment
    mlflow.set_experiment("model-evaluation")
    
    # Set tracking URI (if different from default)
    if 'tracking_uri' in config['mlflow']:
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    with mlflow.start_run(run_name="model_evaluation"):
        # Load model and data
        model, X_test, y_test, feature_names = load_model_and_data()
        
        # Create output directory
        output_dir = Path('ml_pipeline/models')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate detailed metrics
        metrics, y_pred, y_pred_proba, class_report = calculate_detailed_metrics(
            model, X_test, y_test
        )
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        # Log model info
        mlflow.log_params({
            'model_type': type(model).__name__,
            'n_test_samples': len(X_test),
            'n_features': len(feature_names),
            'n_classes': len(np.unique(y_test))
        })
        
        # Generate and save plots
        plots_generated = []
        
        # Confusion Matrix
        cm_path = plot_confusion_matrix(y_test, y_pred, output_dir)
        if cm_path:
            plots_generated.append(cm_path)
            mlflow.log_artifact(cm_path, 'plots')
        
        # Feature Importance
        fi_path = plot_feature_importance(model, feature_names, output_dir)
        if fi_path:
            plots_generated.append(fi_path)
            mlflow.log_artifact(fi_path, 'plots')
        
        # ROC Curves
        roc_path = plot_roc_curves(y_test, y_pred_proba, output_dir)
        if roc_path:
            plots_generated.append(roc_path)
            mlflow.log_artifact(roc_path, 'plots')
        
        # Precision-Recall Curves
        pr_path = plot_precision_recall_curves(y_test, y_pred_proba, output_dir)
        if pr_path:
            plots_generated.append(pr_path)
            mlflow.log_artifact(pr_path, 'plots')
        
        # Save evaluation results
        save_evaluation_results(metrics, class_report, output_dir)
        
        # Log all artifacts
        mlflow.log_artifacts(str(output_dir), 'evaluation_artifacts')
        
        # Log summary
        logger.info("Model Evaluation Summary:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
        logger.info(f"Precision (Macro): {metrics['precision_macro']:.4f}")
        logger.info(f"Recall (Macro): {metrics['recall_macro']:.4f}")
        
        return metrics, plots_generated

if __name__ == "__main__":
    main()