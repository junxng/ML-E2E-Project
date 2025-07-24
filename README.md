# MLOps: Deploying ML Model using Kubeflow Pipeline, KServe & Kubernetes
In this project, I’ll walk you through building a powerful machine learning model using Kubeflow and deploying it seamlessly to KServe with InferenceService!


## Requirements
1. Docker Engine
2. ```kubectl```
3. ```minikube```
4. Python Virtual Environment (```uv```) - optional
5. ```kfp``` library
6. AWS IAM User (with S3 privileges)
6. AWS S3 Bucket
7. Custom Domain Name (DNS)
8. ```helm```


## Launch Ubuntu EC2 Instance
```t3.2xlarge``` with 8 CPUs, 32 GiB Memory, 80 GiB Storage => $0.33 per hour

```sh
chmod 400 'default.pem' # Key pair
```

```sh
ssh -i 'default.pem' ubuntu@ec2-34-239-159-220.compute-1.amazonaws.com # Public DNS
```


## Open the following Ports in Inbound Rules for Smooth Operation

```sh
# Get your security group ID
SECURITY_GROUP_ID=$(aws ec2 describe-instances --instance-ids 'i-0b4e2420c7ca6c085' --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text)
```

1. Port 22: SSH

```sh
# aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 22 --cidr 0.0.0.0/0 => # Already done it by default
```

2. Port 80, 443: HTTP & HTTPS

```sh
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 80 --cidr 0.0.0.0/0
```

```sh
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 443 --cidr 0.0.0.0/0
```

3. Port 8443: Kubernetes API

```sh
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 8443 --cidr 0.0.0.0/0
```

4. Port 8080: Kubeflow Dashboard

```sh
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 8080 --cidr 0.0.0.0/0
```

5. Port 30000-32767: Kubernetes NodePort Service

```sh
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 30000-32767 --cidr 0.0.0.0/0
```

6. Port 5000, 8081, 9000: KServe Model Serving

```sh
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 5000 --cidr 0.0.0.0/0
```

```sh
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 8081 --cidr 0.0.0.0/0
```

```sh
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 9000 --cidr 0.0.0.0/0
```

7. Port 31390: KServe Inference

```sh
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 31390 --cidr 0.0.0.0/0
```

8. Port 31380: Kubeflow Ingress Gateway
```sh
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 31380 --cidr 0.0.0.0/0
```


## Update the System, Install & Activate Docker
```sh
sudo apt update && sudo apt upgrade -y && sudo apt-get install ca-certificates curl
```

```sh
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

```sh
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo '${UBUNTU_CODENAME:-$VERSION_CODENAME}') stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

```sh
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```


## Install Kubectl 

```sh
sudo snap install kubectl --classic
kubectl version --client  # Verify installation
```


## Create Virtual Environment & Activate it

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```sh
uv --version
uv venv --python 3.12
source .venv/bin/activate
uv init
```


## Install Minikube 

```sh
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
```

```sh
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64
```


## Start Minikube

```sh
minikube start --cpus=4 --memory=10240 --driver=docker
```

```sh
kubectl get nodes
kubectl cluster-info
```


## Install Kubeflow Pipelines SDK (kfp)

```sh
which kfp
uv add kfp
```


	@@ -125,25 +190,28 @@ kubectl get all -n kubeflow
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


## Generate Yaml & Build the Pipeline Script on your EC2 instance

```sh
source .venv/bin/activate

	@@ -154,44 +222,62 @@ kfp pipeline create -p IrisProject pipeline.yaml


## Set Up AWS User and Access Keys  

1. IAM -> Create User -> Attach Policy directly (AmazonS3FullAccess) -> Create User

```sh
aws iam create-user --user-name kubeflow-s3-user
```

```sh
aws iam attach-user-policy --user-name kubeflow-s3-user --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

```sh
aws iam get-user --user-name kubeflow-s3-user
```

2. Click on Newly Created User -> Create Access Keys -> CLI -> Create Access Keys -> Download .csv file -> Done

```sh
aws iam create-access-key --user-name kubeflow-s3-user
```

3. Configure AWS User on local terminal with Access Key ID & Secret Key ID using "aws configure"

```sh
aws configure --profile kubeflow-s3-user
```


## Create S3 Bucket

Create S3 Bucket on AWS (or use an already existing bucket)
S3 -> Create bucket -> General purpose -> Name: kubeflow-bucket-iquant01 -> Keep all default selections -> Create bucket

```sh
aws s3 mb s3://kubeflow-bucket-dungnq49 --profile kubeflow-s3-user
```


## On Kubeflow UI 

Create on Pipeline name -> Create run -> Provide Details -> etc
+ Run name: iris-classification-run
+ Description: First run of iris classification pipeline
+ Experiment: Default (or create new experiment)


## On Kubeflow UI (Cont'd)

Provide all the details including Access Key ID, Secret Access Key, S3 Bucket Name & S3 key => Start the pipeline


## On Amazon S3 Bucket

Inspect the bucket for the trained ML model (.joblib file)


## Install Helm 

```sh
	@@ -200,6 +286,7 @@ sudo snap install helm --classic


## Install and Verify Istio, Cert Manager, Knative & KServe

```sh
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.14/hack/quick_install.sh" | bash
```
	@@ -212,24 +299,28 @@ kubectl get pods -n knative-serving


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


## Export Istio Ingress Gateway's ExternalIP & Port 

```sh
minikube ip
```
	@@ -240,17 +331,22 @@ kubectl get svc istio-ingressgateway -n istio-system

```sh
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
```

```sh
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
	@@ -268,24 +364,28 @@ EOF


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
	@@ -297,28 +397,36 @@ cat <<EOF > "./iris-input.json"
EOF
```


## Set the URL of the Model in the InferenceService as an Environment Variable SERVICE_HOST & 

```sh
SERVICE_HOSTNAME=$(kubectl get inferenceservice sklearn-iris -n kserve-test -o jsonpath='{.status.url}' | cut -d "/" -f 3)
```


## Use CURL Command to draw inference from the ML Model to get Prediction

```sh
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
