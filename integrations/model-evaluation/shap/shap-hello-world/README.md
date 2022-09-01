# SHAP integration with Comet.ml

Comet integrates with [SHAP](https://github.com/slundberg/shap).

SHAP or SHAPley Additive exPlanations is a visualization tool that can be used for making a machine learning model more explainable by visualizing its output. It can be used for explaining the prediction of any model by computing the contribution of each feature to the prediction.

## Documentation

For more information on using and configuring Metaflow integration, please see: https://www.comet.com/docs/v2/integrations/ml-frameworks/shap/

## See it

Take a look at this [public Comet Project](https://www.comet.com/examples/comet-example-shap-hello-world).

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example

This example is based on the Deep learning example from [SHAP Readme](https://github.com/slundberg/shap#deep-learning-example-with-gradientexplainer-tensorflowkeraspytorch-models).


```bash
python shap-hello-world.py run
```