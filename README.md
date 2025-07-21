# MLOps: Deploying ML Model using Kubeflow Pipeline, KServe & Kubernetes
In this video, we’ll walk you through building a powerful machine learning model using Kubeflow and deploying it seamlessly to KServe with InferenceService!

## Requirements
1. Docker Desktop (on Mac & Windows) or Docker Engine (on Linux)
2. Kubectl
3. Minikube
4. Python Virtual Environment (venv) - optional
5. kfp
6. AWS IAM User (with S3 privileges)
6. AWS S3 Bucket
7. Custom Domain Name (DNS)
8. Helm


## Launch Ubuntu EC2 Instance
t3.2xlarge with 8 cpus, 32 GiB Memory, 80 GiB Storage -> $0.33 per hour

```sh
chmod 600 <keypair>
ssh -i <keypair> ubuntu@<PublicIP>
```

## Open the following Ports in Inbound Rules for Smooth Operation
1. Port 22:                ssh
2. Port 80, 443:           http & https
3. Port 8443:              Kubernetes API
4. Port 8080:              Kubeflow Dashboard
5. Port 30000-32767:       Kubernetes NodePort Service
6. Port 5000, 8081, 9000:  KServe Model Serving
7. Port 31390:             KServe Inference
8. Port 31380:             Kubeflow Ingress Gateway


## Update the System
```sh
sudo apt update && apt upgrade -y
```

## Install & Activate Docker
```sh
sudo apt-get update && sudo apt-get install docker.io -y
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

## Install Kubectl 
```sh
sudo snap install kubectl --classic
kubectl version --client  # Verify installation
```

## Create or Check Python Environment & Activate it
```sh
python3 --version
pip --version
sudo apt install python3-pip -y
sudo apt install python3.12-venv -y
python3 -m venv path/to/venv
path/to/venv/bin/python --version
path/to/venv/bin/pip --version

source path/to/venv/bin/activate
```

## Install Minikube 
```sh
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64
minikube version
```

## Start Minikube
```sh
minikube start --cpus=4 --memory=10240 --driver=docker
kubectl get nodes
kubectl cluster-info
```

## Install Kubeflow Pipelines SDK (kfp)
```sh
which kfp
path/to/venv/bin/pip install kfp
```


## Install & Set up Kubeflow - This may take about 15-30 min depending on your System
```sh
export PIPELINE_VERSION=2.4.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"
```

```sh
kubectl get all -n kubeflow
```

## Access Kubeflow Pipeline UI - Use a dedicated terminal

#### (1) If you're using your local machine
```sh
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```


#### (2) If you're using Amazon EC2 Instance - Use a dedicated terminal
```sh
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80 &
```
#### Run this ssh tunnel coommand in a separate terminal
```sh
cd Downloads
ssh -i <keypair> -L 8080:localhost:8080 -N ubuntu@<Public_ip>
```


#### On your Browser
```sh
localhost:8080
```


## Generate Yaml & Build the Pipeline Script
```sh
source path/to/venv/bin/activate

touch pipeline.py
path/to/venv/bin/python pipeline.py
path/to/venv/bin/kfp pipeline create -p IrisProject pipeline.yaml
```


## Set Up AWS User and Access Keys  
1. IAM -> Create User -> Attach Policy directly (AmazonS3FullAccess) -> Create User
2. Click on Newly Created User -> Create Access Keys -> CLI -> Create Access Keys -> Download .csv file -> Done
3. Configure AWS User on local terminal with Access Key ID & Secret Key ID using "aws configure"


## Create S3 Bucket
Create S3 Bucket on AWS (or use an already existing bucket)
S3 -> Create bucket -> General purpose -> Name: kubeflow-bucket-iquant01 -> Keep all default selections -> Create bucket


## On Kubeflow UI 
Create on Pipeline name -> Create run -> Provide Details -> etc


## On Kubeflow UI (Cont'd)
Provide all the details including 

Access key id, Secret access key, S3 bucket name, S3 key
Start the pipeline. 


## On Amazon S3 Bucket
Inspect the bucket for the trained ML model (.joblib file)

## Install Helm 

```sh
sudo snap install helm --classic
```


## Install and Verify Istio, Cert Manager, Knative & KServe
```sh
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.14/hack/quick_install.sh" | bash
```

```sh
kubectl get pods -n kserve
kubectl get pods -n istio-system
kubectl get pods -n knative-serving
```


## Expose KServe via Minikube's ingress
```sh
minikube addons enable ingress
```


## Verify Ingress Controller
```sh
kubectl get pods -n ingress-nginx
```


## Expose External IP for Minikube (Like a Load Balancer) - Run this command in a dedicated terminal
```sh
minikube tunnel
```

## If EXTERNAL-IP shows "pending", 
## Export Istio Ingress Gateway's ExternalIP & Port 
```sh
minikube ip
```

```sh
kubectl get svc istio-ingressgateway -n istio-system
```

```sh
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
```


## Create "kserve-test" Namespace
```sh
kubectl create namespace kserve-test
```


## Create InferenceService 
```sh
kubectl apply -n kserve-test -f - <<EOF
  apiVersion: "serving.kserve.io/v1beta1"
  kind: "InferenceService"
  metadata:
    name: "sklearn-iris"
  spec:
    predictor:
      model:
        modelFormat:
          name: sklearn
        storageUri: "gs://kfserving-examples/models/sklearn/1.0/model"
EOF
```


## Check the InferenceService (May show READY: Unknown or False)
```sh
kubectl get inferenceservices sklearn-iris -n kserve-test
```


## Check Istio  Ingress Gateway
```sh
kubectl get svc istio-ingressgateway -n istio-system
```


## InferenceService should be Ready Now (READY: True)
```sh
kubectl get isvc sklearn-iris -n kserve-test
```


## Create an Datafile for Inference: iris-input.json
```sh
cat <<EOF > "./iris-input.json"
{
  "instances": [
    [6.8,  2.8,  4.8,  1.4],
    [6.0,  3.4,  4.5,  1.6]
  ]
}
EOF
```

## Set the URL of the Model in the InferenceService as an Environment Variable SERVICE_HOST & 
## Use CURL Command to draw inference from the ML Model to get Prediction
```sh
SERVICE_HOSTNAME=$(kubectl get inferenceservice sklearn-iris -n kserve-test -o jsonpath='{.status.url}' | cut -d "/" -f 3)

curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" "http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/sklearn-iris:predict" -d @./iris-input.json
```




## Clean Up

Ctrl + C -> To EXIT minikube tunnel

#### Delete the model
```sh
kubectl delete isvc sklearn-iris -n kserve-test
```

#### Stop and Delete Minikube
```sh
minikube stop 
minikube delete --all
```