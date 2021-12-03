#!/usr/bin/env python
# coding: utf-8
#### Import Comet ####
from comet_ml import Experiment

import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

experiment = Experiment()

#### Load and configure boston housing dataset ####
boston = load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data["Price"] = boston.target
X, y = data.iloc[:, :-1], data.iloc[:, -1]

#### Split data into train and test sets ####
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

#### Define hyperparameters for model ####
param = {
    "objective": "reg:squarederror",
    "colsample_bytree": 0.3,
    "learning_rate": 0.1,
    "max_depth": 5,
    "alpha": 10,
    "n_estimators": 10,
}

#### Initialize XGBoost Regressor ####
xg_reg = xgb.XGBRegressor(**param)

#### Train model ####
xg_reg.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric="rmse",
)
