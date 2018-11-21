# coding: utf-8
# import comet_ml in the top of your file
from comet_ml import Experiment
import os

# Setting the API key (saved as environment variable)
experiment = Experiment(
    #api_key="YOUR API KEY",
    # or
    api_key=os.environ.get("COMET_API_KEY"),
    project_name='comet-examples')

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


random_state = 42

cancer = load_breast_cancer()
print("cancer.keys(): {}".format(cancer.keys()))
print("Shape of cancer data: {}\n".format(cancer.data.shape))
print("Sample counts per class:\n{}".format(
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("\nFeature names:\n{}".format(cancer.feature_names))

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data,
    cancer.target,
    stratify=cancer.target,
    random_state=random_state)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression()

param_grid = {'C': [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100]}

clf = GridSearchCV(logreg,
                   param_grid=param_grid,
                   cv=10,
                   n_jobs=-1)

clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

print("\nResults\nConfusion matrix \n {}".format(
    confusion_matrix(y_test, y_pred)))

f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

params = {"random_state": random_state,
          "model_type": "logreg",
          "scaler": "standard scaler",
          "param_grid": str(param_grid),
          "stratify": True
          }
metrics = {"f1": f1,
           "recall": recall,
           "precision": precision
           }

experiment.log_dataset_hash(X_train_scaled)
experiment.log_multiple_params(params)
experiment.log_multiple_metrics(metrics)
