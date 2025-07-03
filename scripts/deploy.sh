#!/bin/bash

# End-to-End ML Project Deployment Script
# This script deploys the ML pipeline to AWS EKS with Kubeflow

set -e

# Configuration
STACK_NAME="ml-e2e-project"
REGION="us-east-1"
CLUSTER_NAME="ml-e2e-cluster"
NAMESPACE="ml-pipeline"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Utility functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if AWS CLI is installed and configured
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed"
        exit 1
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_error "Helm is not installed"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

deploy_infrastructure() {
    log_info "Deploying AWS infrastructure..."
    
    aws cloudformation deploy \
        --template-file aws/cloudformation/ml-infrastructure.yaml \
        --stack-name $STACK_NAME \
        --parameter-overrides ClusterName=$CLUSTER_NAME \
        --capabilities CAPABILITY_IAM \
        --region $REGION
    
    if [ $? -eq 0 ]; then
        log_info "Infrastructure deployed successfully"
    else
        log_error "Infrastructure deployment failed"
        exit 1
    fi
}

configure_kubectl() {
    log_info "Configuring kubectl for EKS cluster..."
    
    aws eks update-kubeconfig \
        --region $REGION \
        --name $CLUSTER_NAME
    
    # Verify connection
    kubectl get nodes
    
    if [ $? -eq 0 ]; then
        log_info "kubectl configured successfully"
    else
        log_error "kubectl configuration failed"
        exit 1
    fi
}

build_and_push_docker_image() {
    log_info "Building and pushing Docker image..."
    
    # Get ECR repository URI
    ECR_URI=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --region $REGION \
        --query 'Stacks[0].Outputs[?OutputKey==`ECRRepositoryURI`].OutputValue' \
        --output text)
    
    # Login to ECR
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI
    
    # Build image
    docker build -t ml-e2e-project .
    
    # Tag image
    docker tag ml-e2e-project:latest $ECR_URI:latest
    
    # Push image
    docker push $ECR_URI:latest
    
    log_info "Docker image pushed to ECR: $ECR_URI:latest"
}

create_namespace() {
    log_info "Creating Kubernetes namespace..."
    
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    log_info "Namespace $NAMESPACE created/updated"
}

deploy_kubernetes_manifests() {
    log_info "Deploying Kubernetes manifests..."
    
    # Update image references in manifests
    ECR_URI=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --region $REGION \
        --query 'Stacks[0].Outputs[?OutputKey==`ECRRepositoryURI`].OutputValue' \
        --output text)
    
    # Replace placeholder in model serving deployment
    sed -i.bak "s|ml-e2e-project:latest|$ECR_URI:latest|g" k8s/manifests/model-serving-deployment.yaml
    
    # Apply manifests
    kubectl apply -f k8s/manifests/ -n $NAMESPACE
    
    log_info "Kubernetes manifests deployed"
}

install_kubeflow() {
    log_info "Installing Kubeflow..."
    
    # Check if Kubeflow is already installed
    if kubectl get namespace kubeflow &> /dev/null; then
        log_warn "Kubeflow namespace already exists, skipping installation"
        return
    fi
    
    # Install Kubeflow (simplified installation)
    # Note: In production, you would use the full Kubeflow manifests
    kubectl apply -k "github.com/kubeflow/manifests/example?ref=v1.8.0"
    
    # Wait for Kubeflow to be ready
    log_info "Waiting for Kubeflow to be ready..."
    kubectl wait --for=condition=ready pod -l app=ml-pipeline -n kubeflow --timeout=300s
    
    log_info "Kubeflow installed successfully"
}

deploy_dvc_pipeline() {
    log_info "Setting up DVC pipeline..."
    
    # Initialize DVC if not already initialized
    if [ ! -f .dvc/config ]; then
        dvc init --no-scm
    fi
    
    # Configure DVC remote to S3
    S3_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --region $REGION \
        --query 'Stacks[0].Outputs[?OutputKey==`MLDataBucketName`].OutputValue' \
        --output text)
    
    dvc remote add -d s3remote s3://$S3_BUCKET/dvc-cache
    dvc remote modify s3remote region $REGION
    
    log_info "DVC pipeline configured with S3 remote: s3://$S3_BUCKET/dvc-cache"
}

run_initial_pipeline() {
    log_info "Running initial ML pipeline..."
    
    # Create a Kubernetes job to run the initial pipeline
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: initial-ml-pipeline
  namespace: $NAMESPACE
spec:
  template:
    spec:
      containers:
      - name: ml-pipeline
        image: $ECR_URI:latest
        command: ["python", "-m", "dvc", "repro"]
        env:
        - name: AWS_DEFAULT_REGION
          value: "$REGION"
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-service:5000"
      restartPolicy: Never
  backoffLimit: 3
EOF
    
    # Wait for job completion
    kubectl wait --for=condition=complete job/initial-ml-pipeline -n $NAMESPACE --timeout=1800s
    
    log_info "Initial ML pipeline completed"
}

display_endpoints() {
    log_info "Deployment completed! Here are the service endpoints:"
    
    # Get MLflow service endpoint
    MLFLOW_ENDPOINT=$(kubectl get service mlflow-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    if [ -z "$MLFLOW_ENDPOINT" ]; then
        MLFLOW_ENDPOINT="localhost:5000 (port-forward required)"
    fi
    
    # Get model serving endpoint
    MODEL_ENDPOINT=$(kubectl get service model-serving-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    if [ -z "$MODEL_ENDPOINT" ]; then
        MODEL_ENDPOINT="localhost:8000 (port-forward required)"
    fi
    
    echo "=========================="
    echo "Service Endpoints:"
    echo "=========================="
    echo "MLflow Tracking Server: http://$MLFLOW_ENDPOINT"
    echo "Model Serving API: http://$MODEL_ENDPOINT"
    echo "Kubeflow Dashboard: http://localhost:8080 (port-forward required)"
    echo ""
    echo "Port forwarding commands:"
    echo "kubectl port-forward svc/mlflow-service 5000:5000 -n $NAMESPACE"
    echo "kubectl port-forward svc/model-serving-service 8000:80 -n $NAMESPACE"
    echo "kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow"
    echo "=========================="
}

main() {
    log_info "Starting ML E2E Project Deployment"
    
    check_prerequisites
    deploy_infrastructure
    configure_kubectl
    build_and_push_docker_image
    create_namespace
    deploy_kubernetes_manifests
    install_kubeflow
    deploy_dvc_pipeline
    run_initial_pipeline
    display_endpoints
    
    log_info "Deployment completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --stack-name)
            STACK_NAME="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --cluster-name)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--stack-name STACK_NAME] [--region REGION] [--cluster-name CLUSTER_NAME]"
            exit 0
            ;;
        *)
            log_error "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Run main function
main