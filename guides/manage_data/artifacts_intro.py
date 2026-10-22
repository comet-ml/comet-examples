# coding: utf-8
import os

import comet_ml
import pandas as pd
from joblib import dump
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# WHY: comet_ml.login() reads COMET_API_KEY from the environment - never hardcode it.
comet_ml.login(project_name="comet-example-artifacts-intro")

DATASETS_DIR = "./datasets"
ARTIFACTS_DIR = "./artifacts"
MODEL_PATH = "./linear-model.pkl"

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def train_and_log(experiment, train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    y_train = train_df.pop("target").values
    X_train = train_df.values
    y_test = test_df.pop("target").values
    X_test = test_df.values

    model = LinearRegression()
    model.fit(X_train, y_train)

    experiment.log_metric("train-score", model.score(X_train, y_train))
    experiment.log_metric("test-score", model.score(X_test, y_test))
    return model


# Prepare the raw dataset
dataset = fetch_california_housing()
X, y = dataset.data, dataset.target
featurecols = dataset.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

train_df = pd.DataFrame(X_train, columns=featurecols)
test_df = pd.DataFrame(X_test, columns=featurecols)
train_df["target"] = y_train
test_df["target"] = y_test

train_df.to_csv(f"{DATASETS_DIR}/train.csv", index=False)
test_df.to_csv(f"{DATASETS_DIR}/test.csv", index=False)

# 1. Log the raw dataset as an Artifact
experiment = comet_ml.start()
experiment.add_tag("upload")

artifact = comet_ml.Artifact(
    name="california",
    artifact_type="dataset",
    aliases=["raw"],
    metadata={"task": "regression"},
)
for split in ["train", "test"]:
    artifact.add(
        f"{DATASETS_DIR}/{split}.csv",
        metadata={"dataset_stage": "raw", "dataset_split": split},
    )

experiment.log_artifact(artifact)
print("Logged raw dataset:", experiment.url)
experiment.end()

# 2. Fetch the Artifact by alias, train a model, log the model as an Artifact
experiment = comet_ml.start()
experiment.add_tag("train")

logged_artifact = experiment.get_artifact("california", version_or_alias="raw")
logged_artifact.download(ARTIFACTS_DIR, overwrite_strategy="PRESERVE")

model = train_and_log(
    experiment, f"{ARTIFACTS_DIR}/train.csv", f"{ARTIFACTS_DIR}/test.csv"
)
dump(model, MODEL_PATH)

model_artifact = comet_ml.Artifact(
    "housing-model", artifact_type="model", aliases=["baseline"]
)
model_artifact.add(MODEL_PATH)
experiment.log_artifact(model_artifact)
print("Trained on raw dataset:", experiment.url)
experiment.end()

# 3. Scale the features and log a new version of the dataset Artifact
experiment = comet_ml.start()
experiment.add_tag("upload")

X_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

train_scaled_df = pd.DataFrame(X_scaler.transform(X_train), columns=featurecols)
test_scaled_df = pd.DataFrame(X_scaler.transform(X_test), columns=featurecols)
train_scaled_df["target"] = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
test_scaled_df["target"] = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

train_scaled_df.to_csv(f"{DATASETS_DIR}/train-scaled.csv", index=False)
test_scaled_df.to_csv(f"{DATASETS_DIR}/test-scaled.csv", index=False)

scaled_artifact = comet_ml.Artifact(
    name="california",
    artifact_type="dataset",
    aliases=["standard-scaled"],
    metadata={"task": "regression"},
)
for split in ["train", "test"]:
    scaled_artifact.add(
        f"{DATASETS_DIR}/{split}-scaled.csv",
        metadata={"dataset_stage": "standard-scaled", "dataset_split": split},
    )

experiment.log_artifact(scaled_artifact)
print("Logged scaled dataset:", experiment.url)
experiment.end()

# 4. Train on the new version, fetched by its alias
experiment = comet_ml.start()
experiment.add_tag("train")

logged_artifact = experiment.get_artifact(
    "california", version_or_alias="standard-scaled"
)
logged_artifact.download(ARTIFACTS_DIR, overwrite_strategy="PRESERVE")

train_and_log(
    experiment,
    f"{ARTIFACTS_DIR}/train-scaled.csv",
    f"{ARTIFACTS_DIR}/test-scaled.csv",
)
print("Trained on scaled dataset:", experiment.url)
experiment.end()
