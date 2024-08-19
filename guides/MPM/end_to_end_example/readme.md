# MPM example scripts

The MPM examples are all based on the same Credit Scoring examples, the goal of the model is to identify users that are likely to default on their loan.

This folder contains three different set of scripts that showcase MPM:
* `data_processing`: Script that processes the raw data and creates a new CSV file with the model's features
* `training`: Script that trains a machine learning model and uploads it to Comet's Model Registry
* `serving`: FastAPI inference server that downloads a model from Comet's Model Registry who's predictions are logged to MPM

## Setup
In order to run these demo scripts you will need to set these environment variables:
```bash
export COMET_API_KEY="<Comet API Key>"
export COMET_WORKSPACE="<Comet workspace to log data to>"
export COMET_PROJECT_NAME="<Comet project name>"
export COMET_MODEL_REGISTRY_NAME="<Comet model registry name>"

export COMET_URL_OVERRIDE="<EM endpoint, similar format to https://www.comet.com/clientlib/>"
export COMET_URL="<MPM ingestion endpoint, similar format to https://www.comet.com/>"
```

You will also need to install the Python libraries in `requirements.txt`

## Data processing

For this demo, we will be using a simple credit scoring dataset available in the `data_processing` folder.

The proprocessing set is quite simple in this demo but showcases how you can use Comet's Artifacts features to track all your data processing steps.

The code can be run using:
```
cd data_processing
python data_processing.py
```

## Training
For this demo we train a LightGBM model that we then upload to the model registry.

The code can be run using:
```
cd training
python model_training.py
```

## Serving
**Dependency**: In order to use this inference server, you will need to first train a model and upload it to the model registry using the training scripts.

The inference server is built using FastAPI and demonstrates how to use both the model registry to store models as well as MPM to log predictions.

The code can be run using:
```
cd serving
uvicorn main:app --reload
```

Once the code has been run, an inference server will be available under `http://localhost:8000` and has the following endpoints:
* `http://localhost:8000/`: returns the string `FastAPI inference service` and indicates the inference server is running
* `http://localhost:8000/health_check`: Simple health check to make sure the server is running and accepting requests
* `http://localhost:8000/prediction`: Make a prediction and log it to MPM
* `http://localhost:8000/create_demo_data`: Creates 10,000 predictions over a one week period to populate MPM dashboards

**Note:** It can take a few minutes for the data to appear in the debugger tab in the MPM UI.
