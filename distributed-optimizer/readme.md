# Comet Optimizer and Kubernetes

An example for using Comet Optimizer on a local Kubernetes instance. 

For information about the Comet Optimizer see [Docs](https://www.comet.ml/docs/python-sdk/Optimizer/).

## Setup

Install Docker Desktop, information can be found on the [Docker Website](https://www.docker.com/products/docker-desktop).

Install minikube

```bash
brew install minikube
```

Set your API key, Workspace, and Project Name in job-initialize-comet-optimizer.yaml and job-optimizer.yaml. You can do this by replacing the environment variables placeholders with your values, or by running the following commands in your terminal. 

```bash
COMET_API_KEY="<YOUR API KEY>" 
COMET_WORKSPACE="<YOUR WORKSPACE NAME>" 
COMET_PROJECT_NAME="<YOUR PROJECT NAME>" 

sed -i '' -e "s/REPLACE_WITH_YOUR_API_KEY/$COMET_API_KEY/g" job-initialize-comet-optimizer.yaml &&
sed -i '' -e "s/REPLACE_WITH_YOUR_API_KEY/$COMET_API_KEY/g" job-optimizer.yaml

sed -i '' -e "s/REPLACE_WITH_YOUR_WORKSPACE/$COMET_WORKSPACE/g" job-initialize-comet-optimizer.yaml &&
sed -i '' -e "s/REPLACE_WITH_YOUR_WORKSPACE/$COMET_WORKSPACE/g" job-optimizer.yaml

sed -i '' -e "s/REPLACE_WITH_YOUR_PROJECT_NAME/$COMET_PROJECT_NAME/g" job-initialize-comet-optimizer.yaml &&
sed -i '' -e "s/REPLACE_WITH_YOUR_PROJECT_NAME/$COMET_PROJECT_NAME/g" job-optimizer.yaml
```

## To Run

Start minikube
```bash
minikube start
```
Enable the local use of docker in minikube
```bash
eval $(minikube docker-env).
```
Build the docker image
```bash
docker build ./ -t comet-optimizer
```
Open a dashboard in your web browser to view job status and logs 
```bash
minikube dashboard
```

Run shell script to:
* Intialize Comet Optimizer
* Run Optimization sweeps
```bash
sh run.sh
```

## Customizing this template

1. Add your model training functions in run_optimizer.py. 
2. Change the number of experiments running in parallel by updating "parallelism" value in job-optimizer.yaml (line 6)

