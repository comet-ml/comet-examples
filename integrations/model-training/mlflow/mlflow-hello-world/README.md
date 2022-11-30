# MLflow integration with Comet.ml

[MLflow](https://github.com/mlflow/mlflow/) is a platform to streamline machine learning development, including tracking experiments, packaging code into reproducible runs, and sharing and deploying models. MLflow offers a set of lightweight APIs that can be used with any existing machine learning application or library (TensorFlow, PyTorch, XGBoost, etc), wherever you currently run ML code (e.g. in notebooks, standalone applications or the cloud).

## Documentation

For more information on using and configuring the MLflow integration, see: https://www.comet.com/docs/v2/integrations/ml-frameworks/mlflow/#configure-comet-for-mlflow

## See it

Take a look at this [public Comet Project](https://www.comet.com/examples/comet-example-mlflow-hello-world/).

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example

This example is based on the following [MLflow tutorial](https://github.com/mlflow/mlflow/blob/master/examples/keras/train.py).

```bash
python mlflow-hello-world.py
```

# Comet-for-MLFlow

If you have previous MLFlow runs that you would like to visualize in Comet.ml, please see:

https://githib.com/comet-ml/comet-for-mlflow
