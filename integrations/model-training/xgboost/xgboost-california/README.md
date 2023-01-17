# XGBoost integration with Comet.ml

[XGBoost](https://github.com/dmlc/xgboost) is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Kubernetes, Hadoop, SGE, MPI, Dask) and can solve problems beyond billions of examples.

Instrument xgboost with Comet to start managing experiments, create dataset versions and track hyperparameters for faster and easier reproducibility and collaboration.


## Documentation

For more information on using and configuring the xgboost integration, see: https://www.comet.com/docs/v2/integrations/ml-frameworks/xgboost/

## See it

Take a look at this [public Comet Project](https://www.comet.com/examples/comet-example-xgboost-california/).

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example

This example showcase a simple regression on the California Housing dataset.


```bash
python xgboost-california.py
```
