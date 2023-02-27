# coding: utf-8
import os
import tempfile

import comet_ml

import cloudpickle
from sklearn import datasets, ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

MODEL_NAME = "my-sklearn-model"
MODEL_VERSION = "1.0.1"
WORKSPACE = os.environ["COMET_WORKSPACE"]

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

# Helper function to save model


def sklearn_log_model(experiment, model, model_name, pickle_module):
    import pickle

    # Save model
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        pickle_module.dump(model, fp, protocol=pickle.DEFAULT_PROTOCOL)

    # Log model to Comet
    experiment.log_model(model_name, fp.name, file_name="comet-sklearn-model.pkl")


# Save model and register it

sklearn_log_model(experiment, model, MODEL_NAME, cloudpickle)

experiment.register_model(MODEL_NAME, version=MODEL_VERSION)

# Upload everything
experiment.end()

# Helper function to load model from the registry


def sklearn_load_model_from_registry(workspace, model_name, version, pickle_module):
    import tempfile
    from pathlib import Path

    from comet_ml import API

    api = API()

    tmpdir = tempfile.mkdtemp()
    api.download_registry_model(
        workspace, model_name, version=version, output_path=tmpdir
    )

    with open(Path(tmpdir) / "comet-sklearn-model.pkl", "rb") as fp:
        return pickle_module.load(fp)


# Load model
loaded_model = sklearn_load_model_from_registry(
    WORKSPACE, MODEL_NAME, MODEL_VERSION, cloudpickle
)

print("LOADED", loaded_model)
