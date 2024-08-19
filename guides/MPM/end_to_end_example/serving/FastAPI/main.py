import shutil
import uuid
import time
import os
import requests
import random

import pandas as pd
from datetime import datetime, timedelta

import lightgbm as lgb
from fastapi import FastAPI
import comet_ml

from comet_mpm import CometMPM

# Hardcoded variables
MODEL_REGISTRY_NAME = os.environ["COMET_MODEL_REGISTRY_NAME"]

def load_mock_data():
    # Load dummy data for demo purposes
    mock_data = pd.read_csv('./demo_data.csv')
    for c in ['EmpStatus', 'OtherCC', 'ResStatus']:
        mock_data[c] = mock_data[c].astype('category')
    
    return mock_data

def download_and_run_model():
    # Load the model from the model registry
    api = comet_ml.API()
    existing_models = api.get_registry_model_versions(workspace=os.environ['COMET_WORKSPACE'],
                                                    registry_name=MODEL_REGISTRY_NAME)
    model_version = max(existing_models)

    api.download_registry_model(workspace=os.environ['COMET_WORKSPACE'],
                                registry_name=MODEL_REGISTRY_NAME,
                                version=model_version, output_path='./model/')
    model = lgb.Booster(model_file='./model/model.txt', params={'objective': 'binary'})
    shutil.rmtree('./model')

    return model, model_version

mock_data = load_mock_data()
model, model_version = download_and_run_model()

# Start webserver and define API endpoints
app = FastAPI()
MPM = CometMPM(
    workspace_name = os.environ['COMET_WORKSPACE'],
    model_name = MODEL_REGISTRY_NAME,
    model_version = model_version, asyncio=True,
    max_batch_time=5
)

@app.on_event("startup")
async def startup_event():
    MPM.connect()

@app.get("/")
def root_path():
    return "FastAPI inference service"


@app.get("/health_check")
def run_health_check():
    return {"res": True}


@app.get("/prediction")
async def run_prediction_random():
    # random single prediction
    data_sample = mock_data.sample(1).drop(columns=['probdefault'], inplace=False)
    features = data_sample.to_dict('records')[0]
    
    prediction_probability = max(min(float(model.predict(data_sample)[0]), 1.0), 0.0)
    prediction_value = bool(prediction_probability > 0.5)

    # Log prediction to Comet
    await MPM.log_event(
        prediction_id=str(uuid.uuid4()),
        input_features=features,
        output_features={
            "value": prediction_value,
            "probability": prediction_probability
        },
        output_value=prediction_value,
        output_probability=prediction_probability
    )

    return {"value": prediction_value, "probablity": prediction_probability}


@app.get("/create_demo_data")
async def run_prediction_random(nb_predictions=10000):
    events = []
    for i in range(nb_predictions):
        # random single prediction
        data_sample = mock_data.sample(1).drop(columns=['probdefault'], inplace=False)
        prediction_probability = max(min(float(model.predict(data_sample)[0]), 1.0), 0.0)
        prediction_value = bool(prediction_probability > 0.5)
        timestamp = int(time.time())
        
        # Processing
        features = data_sample.to_dict('records')[0]
        prediction = {
                "value": prediction_value,
                "probability": prediction_probability
            }
        
        day_of_week = datetime.utcfromtimestamp(timestamp).weekday()
        # Add some interesting features based on the time of the week
        ## Monday and Tuesday we will introduce missing data
        if day_of_week in [0, 1]:
            datetime_timestamp = datetime.utcfromtimestamp(timestamp)
            start_of_week = (datetime_timestamp - timedelta(days=day_of_week)).replace(hour=0, minute=0, second=0, microsecond=0)
            time_since_start_of_week = timestamp - start_of_week.timestamp()
            total_number_seconds = 2 * 24 * 60 * 60
            
            features['CustIncome'] = features['CustIncome'] * (1 - time_since_start_of_week / total_number_seconds)

        ## Thursday and Friday we will introduce data drift
        if day_of_week in [3, 4]:
            if random.random() < 0.1:
                del features['CustAge']
                prediction = {}
        
        prediction_id = str(uuid.uuid4())
        events += [
            {   
                "prediction_id": prediction_id,
                "timestamp": timestamp,
                "features": features,
                "prediction": prediction
            }
        ]

        # Log prediction to Comet
        await MPM.log_event(
            prediction_id=prediction_id,
            input_features=features,
            output_features=prediction
        )
    
    return events

@app.on_event("shutdown")
async def shutdown_mpm():
    await MPM.join()