# Metaflow integration with Comet.ml

Comet integrates with [Metaflow](https://metaflow.org/).

Metaflow is a human-friendly Python/R library that helps scientists and engineers build and manage real-life data science projects. Metaflow was originally developed at Netflix to boost productivity of data scientists who work on a wide variety of projects from classical statistics to state-of-the-art deep learning.

## Documentation

For more information on using and configuring Metaflow integration, please see: https://www.comet.ml/docs/v2/integrations/third-party-tools/metaflow/

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example

This example is a Metaflow example training two models on the MNIST database of handwritten digits. It fits two Scikit-Learn estimators, a [GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) one and a [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) one. For each estimator, a confusion matrix is generated with the test dataset.

```bash
python helloworld.py run
```

> This example uses a small version of the MNist dataset [available on Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv). If you want to train with the full dataset, connect to Kaggle and download the full file from there.
