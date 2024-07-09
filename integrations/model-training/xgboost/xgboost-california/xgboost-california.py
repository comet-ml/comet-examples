#!/usr/bin/env python
# coding: utf-8

# Import Comet
from comet_ml import Experiment, login

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import xgboost as xgb

# Login to Comet if needed
login()

experiment = Experiment(project_name="comet-example-xgboost-california")

# Load and configure california housing dataset
california = fetch_california_housing()
data = pd.DataFrame(california.data)
data.columns = california.feature_names
data["Price"] = california.target
X, y = data.iloc[:, :-1], data.iloc[:, -1]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# Define hyperparameters for model
param = {
    "objective": "reg:squarederror",
    "colsample_bytree": 0.3,
    "learning_rate": 0.1,
    "max_depth": 5,
    "alpha": 10,
    "n_estimators": 10,
}

# Initialize XGBoost Regressor
xg_reg = xgb.XGBRegressor(eval_metric="rmse", **param)

# Train model
xg_reg.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
)
