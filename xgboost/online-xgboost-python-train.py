#!/usr/bin/env python
# coding: utf-8
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2020 Comet ML INC
#  This file can not be copied and/or distributed without the express
#  permission of Comet ML Inc.
# *******************************************************

import os.path

from comet_ml import Experiment

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import xgboost as xgb


# Thanks to Chenglong Chen for providing this in the forum
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1.0 / (y[ind] ** 2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return rmspe


def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return "rmspe", rmspe


# Gather some features
def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), "Open"] = 1
    # Use some properties directly
    features.extend(
        [
            "Store",
            "CompetitionDistance",
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
            "Promo",
            "Promo2",
            "Promo2SinceWeek",
            "Promo2SinceYear",
        ]
    )

    # add some more with a bit of preprocessing
    features.append("SchoolHoliday")
    data["SchoolHoliday"] = data["SchoolHoliday"].astype(float)
    #
    # features.append('StateHoliday')
    # data.loc[data['StateHoliday'] == 'a', 'StateHoliday'] = '1'
    # data.loc[data['StateHoliday'] == 'b', 'StateHoliday'] = '2'
    # data.loc[data['StateHoliday'] == 'c', 'StateHoliday'] = '3'
    # data['StateHoliday'] = data['StateHoliday'].astype(float)

    features.append("DayOfWeek")
    features.append("month")
    features.append("day")
    features.append("year")
    data["year"] = data.Date.apply(lambda x: x.split("-")[0])
    data["year"] = data["year"].astype(float)
    data["month"] = data.Date.apply(lambda x: x.split("-")[1])
    data["month"] = data["month"].astype(float)
    data["day"] = data.Date.apply(lambda x: x.split("-")[2])
    data["day"] = data["day"].astype(float)

    features.append("StoreType")
    data.loc[data["StoreType"] == "a", "StoreType"] = "1"
    data.loc[data["StoreType"] == "b", "StoreType"] = "2"
    data.loc[data["StoreType"] == "c", "StoreType"] = "3"
    data.loc[data["StoreType"] == "d", "StoreType"] = "4"
    data["StoreType"] = data["StoreType"].astype(float)

    features.append("Assortment")
    data.loc[data["Assortment"] == "a", "Assortment"] = "1"
    data.loc[data["Assortment"] == "b", "Assortment"] = "2"
    data.loc[data["Assortment"] == "c", "Assortment"] = "3"
    data["Assortment"] = data["Assortment"].astype(float)


CURRENT_DIR = os.path.dirname(__file__)


print("Load the training, test and store data using pandas")
train = pd.read_csv(os.path.join(CURRENT_DIR, "data", "train.csv"))
test = pd.read_csv(os.path.join(CURRENT_DIR, "data", "test.csv"))
store = pd.read_csv(os.path.join(CURRENT_DIR, "data", "store.csv"))

print("Assume store open, if not provided")
test.fillna(1, inplace=True)

print(
    "Consider only open stores for training. Closed stores wont count into the score."
)
train = train[train["Open"] != 0]

print("Join with store")
train = pd.merge(train, store, on="Store")
test = pd.merge(test, store, on="Store")

features = []

print("augment features")
build_features(features, train)
build_features([], test)
print(features)

params = {
    "objective": "reg:linear",
    "eta": 0.3,
    "max_depth": 8,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "silent": 1,
}
num_trees = 50

print("Train a XGBoost model")
val_size = 100000
# train = train.sort(['Date'])
print(train.tail(1)["Date"])
X_train, X_test = train_test_split(train, test_size=0.01)
# X_train, X_test = train.head(len(train) - val_size), train.tail(val_size)
dtrain = xgb.DMatrix(X_train[features], np.log(X_train["Sales"] + 1))
dvalid = xgb.DMatrix(X_test[features], np.log(X_test["Sales"] + 1))
dtest = xgb.DMatrix(test[features])

watchlist = [(dvalid, "eval"), (dtrain, "train")]

# Experiment 1: everything as normal, using .train():
experiment = Experiment()
experiment.add_tag("metrics")
results = {}
gbm = xgb.train(
    params,
    dtrain,
    num_trees,
    evals=watchlist,
    early_stopping_rounds=50,
    feval=rmspe_xg,
    verbose_eval=True,
    evals_result=results,
)
experiment.end()

# Experiment 2: no results (thus no metrics), using .train():
experiment = Experiment()
experiment.add_tag("no metrics")
gbm = xgb.train(
    params,
    dtrain,
    num_trees,
    evals=watchlist,
    early_stopping_rounds=50,
    feval=rmspe_xg,
    verbose_eval=True,
)
experiment.end()

# Experiment 3: results, but no metrics because we told it no, using .train():
experiment = Experiment(auto_metric_logging=False)
experiment.add_tag("no metrics")
results = {}
gbm = xgb.train(
    params,
    dtrain,
    num_trees,
    evals=watchlist,
    early_stopping_rounds=50,
    feval=rmspe_xg,
    verbose_eval=True,
    evals_result=results,
)
experiment.end()

# print("Validating")
# train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
# indices = train_probs < 0
# train_probs[indices] = 0
# error = rmspe(np.exp(train_probs) - 1, X_test['Sales'].values)
# print('error', error)

# print("Make predictions on the test set")
# test_probs = gbm.predict(xgb.DMatrix(test[features]))
# indices = test_probs < 0
# test_probs[indices] = 0
# submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(test_probs) - 1})
# submission.to_csv("xgboost_kscript_submission.csv", index=False)
