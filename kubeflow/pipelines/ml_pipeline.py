"""
Kubeflow Pipeline for End-to-End ML Workflow
Orchestrates data collection, preprocessing, training, evaluation, and deployment
"""
import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics
from typing import NamedTuple

# Component definitions
@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas", "numpy", "scikit-learn", "mlflow", "boto3", "pyyaml"
    ]
)
def data_collection_component(
    config_path: str,
    output_dataset: Output[Dataset]
) -> NamedTuple('DataCollectionOutput', [('n_samples', int), ('n_features', int)]):
    """Data collection component for Kubeflow pipeline"""
    import pandas as pd
    import numpy as np
    import yaml
    import logging
    from sklearn.datasets import make_classification
    import mlflow
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Generate synthetic data
    params = config['data_sources']['primary']['parameters']
    X, y = make_classification(
        n_samples=params['n_samples'],
        n_features=params['n_features'],
        n_informative=params['n_informative'],
        n_redundant=params['n_redundant'],
        n_classes=params['n_classes'],
        random_state=params['random_state']
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save dataset
    df.to_csv(output_dataset.path, index=False)
    
    # Return metadata
    from collections import namedtuple
    DataCollectionOutput = namedtuple('DataCollectionOutput', ['n_samples', 'n_features'])
    return DataCollectionOutput(n_samples=len(df), n_features=len(feature_names))

@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas", "numpy", "scikit-learn", "mlflow", "joblib", "pyyaml"
    ]
)
def data_preprocessing_component(
    input_dataset: Input[Dataset],
    config_path: str,
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
    preprocessor: Output[Model]
) -> NamedTuple('PreprocessingOutput', [('train_samples', int), ('test_samples', int), ('final_features', int)]):
    """Data preprocessing component for Kubeflow pipeline"""
    import pandas as pd
    import numpy as np
    import yaml
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Load data and config
    df = pd.read_csv(input_dataset.path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Split data
    X = df.drop('target', axis=1)
    y = df['target']
    
    split_config = config['preprocessing']['train_test_split']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_config['test_size'],
        random_state=split_config['random_state'],
        stratify=y if split_config['stratify'] else None
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=15)
    X_train_selected = pd.DataFrame(
        selector.fit_transform(X_train_scaled, y_train),
        columns=X_train_scaled.columns[selector.get_support()],
        index=X_train_scaled.index
    )
    X_test_selected = pd.DataFrame(
        selector.transform(X_test_scaled),
        columns=X_train_selected.columns,
        index=X_test_scaled.index
    )
    
    # Save processed data
    train_df = X_train_selected.copy()
    train_df['target'] = y_train
    train_df.to_csv(train_dataset.path, index=False)
    
    test_df = X_test_selected.copy()
    test_df['target'] = y_test
    test_df.to_csv(test_dataset.path, index=False)
    
    # Save preprocessor
    preprocessing_artifacts = {
        'scaler': scaler,
        'feature_selector': selector,
        'feature_names': list(X_train_selected.columns)
    }
    joblib.dump(preprocessing_artifacts, preprocessor.path)
    
    # Return metadata
    from collections import namedtuple
    PreprocessingOutput = namedtuple('PreprocessingOutput', ['train_samples', 'test_samples', 'final_features'])
    return PreprocessingOutput(
        train_samples=len(X_train_selected),
        test_samples=len(X_test_selected),
        final_features=len(X_train_selected.columns)
    )

@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas", "numpy", "scikit-learn", "mlflow", "joblib", "pyyaml"
    ]
)
def model_training_component(
    train_dataset: Input[Dataset],
    config_path: str,
    trained_model: Output[Model],
    model_metrics: Output[Metrics]
) -> NamedTuple('TrainingOutput', [('accuracy', float), ('f1_score', float)]):
    """Model training component for Kubeflow pipeline"""
    import pandas as pd
    import numpy as np
    import yaml
    import joblib
    import json
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, f1_score
    import mlflow
    
    # Load data and config
    train_df = pd.read_csv(train_dataset.path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    
    # Initialize model
    model_config = config['model']['hyperparameters']['random_forest']
    model = RandomForestClassifier(**model_config)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
    
    # Calculate metrics
    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred, average='macro')
    
    # Save model
    joblib.dump(model, trained_model.path)
    
    # Save metrics
    metrics_dict = {
        'accuracy': accuracy,
        'f1_score': f1,
        'cv_mean_score': cv_scores.mean(),
        'cv_std_score': cv_scores.std()
    }
    
    with open(model_metrics.path, 'w') as f:
        json.dump(metrics_dict, f)
    
    # Return metadata
    from collections import namedtuple
    TrainingOutput = namedtuple('TrainingOutput', ['accuracy', 'f1_score'])
    return TrainingOutput(accuracy=accuracy, f1_score=f1)

@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "joblib"
    ]
)
def model_evaluation_component(
    test_dataset: Input[Dataset],
    trained_model: Input[Model],
    evaluation_metrics: Output[Metrics]
) -> NamedTuple('EvaluationOutput', [('test_accuracy', float), ('test_f1_score', float)]):
    """Model evaluation component for Kubeflow pipeline"""
    import pandas as pd
    import numpy as np
    import joblib
    import json
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    # Load data and model
    test_df = pd.read_csv(test_dataset.path)
    model = joblib.load(trained_model.path)
    
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    test_precision = precision_score(y_test, y_pred, average='macro')
    test_recall = recall_score(y_test, y_pred, average='macro')
    
    # Save evaluation metrics
    evaluation_results = {
        'test_accuracy': test_accuracy,
        'test_f1_score': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'n_test_samples': len(X_test)
    }
    
    with open(evaluation_metrics.path, 'w') as f:
        json.dump(evaluation_results, f)
    
    # Return metadata
    from collections import namedtuple
    EvaluationOutput = namedtuple('EvaluationOutput', ['test_accuracy', 'test_f1_score'])
    return EvaluationOutput(test_accuracy=test_accuracy, test_f1_score=test_f1)

@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "mlflow", "boto3", "joblib"
    ]
)
def model_deployment_component(
    trained_model: Input[Model],
    model_name: str,
    model_stage: str = "staging"
) -> str:
    """Model deployment component for Kubeflow pipeline"""
    import mlflow
    import mlflow.sklearn
    import joblib
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://mlflow-service:5000")
    
    # Load model
    model = joblib.load(trained_model.path)
    
    # Log and register model
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        # Transition to specified stage
        client = mlflow.tracking.MlflowClient()
        model_version = model_info.registered_model_version
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=model_stage
        )
        
        return f"Model {model_name} v{model_version} deployed to {model_stage}"

# Pipeline definition
@pipeline(
    name="ml-e2e-pipeline",
    description="End-to-End ML Pipeline with MLflow and DVC",
    pipeline_root="s3://ml-e2e-project-artifacts/kubeflow-pipelines/"
)
def ml_e2e_pipeline(
    data_config_path: str = "/config/data_config.yaml",
    preprocessing_config_path: str = "/config/preprocessing_config.yaml",
    model_config_path: str = "/config/model_config.yaml",
    model_name: str = "classification_model",
    model_stage: str = "staging"
):
    """
    End-to-end ML pipeline that includes:
    1. Data collection
    2. Data preprocessing  
    3. Model training
    4. Model evaluation
    5. Model deployment
    """
    
    # Step 1: Data Collection
    data_collection_task = data_collection_component(
        config_path=data_config_path
    )
    
    # Step 2: Data Preprocessing
    preprocessing_task = data_preprocessing_component(
        input_dataset=data_collection_task.outputs['output_dataset'],
        config_path=preprocessing_config_path
    )
    
    # Step 3: Model Training
    training_task = model_training_component(
        train_dataset=preprocessing_task.outputs['train_dataset'],
        config_path=model_config_path
    )
    
    # Step 4: Model Evaluation
    evaluation_task = model_evaluation_component(
        test_dataset=preprocessing_task.outputs['test_dataset'],
        trained_model=training_task.outputs['trained_model']
    )
    
    # Step 5: Model Deployment (conditional on evaluation results)
    with dsl.Condition(evaluation_task.outputs['test_f1_score'] > 0.7):
        deployment_task = model_deployment_component(
            trained_model=training_task.outputs['trained_model'],
            model_name=model_name,
            model_stage=model_stage
        )
    
    # Set resource requirements
    data_collection_task.set_cpu_request("100m").set_memory_request("256Mi")
    preprocessing_task.set_cpu_request("200m").set_memory_request("512Mi")
    training_task.set_cpu_request("500m").set_memory_request("1Gi")
    evaluation_task.set_cpu_request("200m").set_memory_request("512Mi")
    deployment_task.set_cpu_request("100m").set_memory_request("256Mi")

# Compile pipeline
if __name__ == "__main__":
    import kfp.compiler as compiler
    
    compiler.Compiler().compile(
        pipeline_func=ml_e2e_pipeline,
        package_path="ml_e2e_pipeline.yaml"
    )
    print("Pipeline compiled successfully!")