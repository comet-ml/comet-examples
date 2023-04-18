# coding: utf-8
import comet_ml
from comet_ml.integration.sklearn import log_model

import cloudpickle
from sklearn import datasets, ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

MODEL_NAME = "my-sklearn-model"
MODEL_VERSION = "1.0.1"

# Login to comet and create an Experiment

comet_ml.init()

experiment = comet_ml.Experiment(
    project_name="comet-example-scikit-learn-model-saving-example"
)

# Prepare data

X, y = datasets.fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model

model = ensemble.RandomForestRegressor().fit(X_train_scaled, y_train)

# Save model to Comet
log_model(experiment, model, MODEL_NAME, persistence_module=cloudpickle)
# experiment.register_model(MODEL_NAME, version=MODEL_VERSION)

# Upload everything
experiment.end()

# # Load model from Comet Model Registry
# loaded_model = load_model(f"registry://{WORKSPACE}/{MODEL_NAME}:{MODEL_VERSION}")

# print("LOADED", loaded_model)
