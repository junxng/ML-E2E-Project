# End-to-End ML Project with MLFlow, DVC, and Kubernetes

A comprehensive machine learning project demonstrating end-to-end MLOps practices with experiment tracking, data versioning, and production deployment.

## 🏗️ Architecture Overview

This project implements a complete ML pipeline with the following components:

- **Data Pipeline**: DVC for data versioning and pipeline orchestration
- **Experiment Tracking**: MLFlow for experiment management and model registry
- **Model Training**: Scikit-learn with hyperparameter optimization
- **Deployment**: Kubernetes on AWS EKS with auto-scaling
- **Orchestration**: Kubeflow for pipeline automation
- **Infrastructure**: AWS (EKS, S3, ECR, CloudFormation)

## 📁 Project Structure

```
ml-e2e-project/
├── ml_pipeline/                 # Core ML pipeline
│   ├── scripts/                 # Pipeline scripts
│   │   ├── data_collection.py   # Data generation and collection
│   │   ├── data_preprocessing.py # Feature engineering
│   │   ├── train_model.py       # Model training
│   │   └── evaluate_model.py    # Model evaluation
│   ├── data/                    # Data storage
│   │   ├── raw/                 # Raw data
│   │   └── processed/           # Processed data
│   └── models/                  # Trained models
├── config/                      # Configuration files
│   ├── data_config.yaml         # Data configuration
│   ├── preprocessing_config.yaml # Preprocessing settings
│   └── model_config.yaml        # Model configuration
├── k8s/                         # Kubernetes manifests
│   └── manifests/               # Deployment files
├── kubeflow/                    # Kubeflow pipeline definitions
│   └── pipelines/               # ML pipeline definitions
├── aws/                         # AWS infrastructure
│   └── cloudformation/          # CloudFormation templates
├── app/                         # FastAPI serving application
├── scripts/                     # Deployment scripts
├── dvc.yaml                     # DVC pipeline definition
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Local development
└── pyproject.toml              # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker
- AWS CLI (configured)
- kubectl
- Helm
- DVC

### Local Development

1. **Clone and setup**:
```bash
git clone <repository-url>
cd ml-e2e-project
pip install -e .
```

2. **Run locally with Docker Compose**:
```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key

# Start services
docker-compose up -d
```

3. **Access services**:
- MLFlow UI: http://localhost:5000
- Model API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Production Deployment on AWS

1. **Deploy infrastructure and services**:
```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

2. **Monitor deployment**:
```bash
kubectl get pods -n ml-pipeline
kubectl logs -f deployment/mlflow-server -n ml-pipeline
```

## 🔧 Configuration

### Data Configuration (`config/data_config.yaml`)

```yaml
data_sources:
  primary:
    type: "synthetic"
    source: "sklearn_classification"
    parameters:
      n_samples: 10000
      n_features: 20
      n_classes: 3
```

### Model Configuration (`config/model_config.yaml`)

```yaml
model:
  algorithm: "random_forest"
  hyperparameters:
    random_forest:
      n_estimators: 100
      max_depth: 10
      random_state: 42
```

## 🔄 ML Pipeline

### DVC Pipeline Stages

1. **Data Collection**: Generate synthetic dataset
2. **Data Preprocessing**: Feature engineering and scaling
3. **Model Training**: Train with hyperparameter tuning
4. **Model Evaluation**: Comprehensive evaluation with plots

### Running the Pipeline

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro data_collection

# View pipeline status
dvc status

# Show metrics
dvc metrics show
```

## 📊 Experiment Tracking

### MLFlow Integration

- **Experiments**: Organized by pipeline stage
- **Parameters**: All hyperparameters logged
- **Metrics**: Performance metrics tracked
- **Artifacts**: Models and plots stored
- **Model Registry**: Versioned model management

### Example Usage

```python
import mlflow

# Set experiment
mlflow.set_experiment("model-training")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({"n_estimators": 100, "max_depth": 10})
    
    # Train model
    model = train_model()
    
    # Log metrics
    mlflow.log_metrics({"accuracy": 0.95, "f1_score": 0.93})
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

## 🚀 Model Serving

### FastAPI Service

The model serving API provides:

- **Single Prediction**: `/predict`
- **Batch Prediction**: `/predict/batch`
- **Health Check**: `/health`
- **Model Info**: `/model/info`
- **Metrics**: `/metrics` (Prometheus format)

### Example API Usage

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [1.2, -0.5, 0.8, ...]}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"features": [[1.2, -0.5, ...], [0.8, 1.1, ...]]}
)
print(response.json())
```

## ☸️ Kubernetes Deployment

### Components

- **MLFlow Server**: Experiment tracking with PostgreSQL backend
- **Model Serving**: FastAPI application with auto-scaling
- **PostgreSQL**: Database for MLFlow metadata
- **Ingress**: Load balancer for external access

### Monitoring

```bash
# Check pod status
kubectl get pods -n ml-pipeline

# View logs
kubectl logs -f deployment/model-serving -n ml-pipeline

# Port forward for local access
kubectl port-forward svc/mlflow-service 5000:5000 -n ml-pipeline
```

## 🔧 Kubeflow Integration

### Pipeline Components

The project includes Kubeflow pipeline components for:

- Data collection
- Data preprocessing
- Model training
- Model evaluation
- Model deployment

### Running Kubeflow Pipeline

```python
import kfp

# Compile pipeline
kfp.compiler.Compiler().compile(ml_e2e_pipeline, 'ml_pipeline.yaml')

# Create client
client = kfp.Client()

# Run pipeline
run = client.run_pipeline(
    experiment_id='ml-e2e-experiment',
    job_name='ml-pipeline-run',
    pipeline_package_path='ml_pipeline.yaml'
)
```

## 📈 Monitoring and Observability

### Metrics Collection

- **Prometheus**: System and application metrics
- **Grafana**: Visualization dashboards
- **MLFlow**: ML-specific metrics and artifacts

### Health Checks

All services include comprehensive health checks:

```bash
# Check service health
curl http://localhost:8000/health

# Check Prometheus metrics
curl http://localhost:8000/metrics
```

## 🔒 Security

### AWS IAM Roles

- **EKS Cluster Role**: Manages EKS cluster
- **Node Group Role**: EC2 instances permissions
- **ML Service Role**: S3 and ECR access

### Kubernetes Security

- **Namespaces**: Isolation between components
- **RBAC**: Role-based access control
- **Secrets**: Secure credential management

## 🐛 Troubleshooting

### Common Issues

1. **MLFlow Connection Issues**:
```bash
kubectl describe pod mlflow-server -n ml-pipeline
kubectl logs mlflow-server -n ml-pipeline
```

2. **Model Loading Errors**:
```bash
kubectl logs deployment/model-serving -n ml-pipeline
```

3. **DVC Pipeline Failures**:
```bash
dvc status
dvc dag
```

### Debug Commands

```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n ml-pipeline

# Describe problematic resources
kubectl describe deployment model-serving -n ml-pipeline

# Check events
kubectl get events -n ml-pipeline --sort-by='.lastTimestamp'
```

## 📚 Additional Resources

- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Kubeflow Documentation](https://www.kubeflow.org/docs/)
- [Kubernetes Documentation](https://kubernetes.io/docs/home/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:

- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation links above
