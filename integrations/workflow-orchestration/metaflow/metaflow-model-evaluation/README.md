# Model Evaluation Flow with Metaflow and Comet

Comet integrates with [Metaflow](https://metaflow.org/).

Metaflow is a human-friendly Python/R library that helps scientists and engineers build and manage real-life data science projects. Metaflow was originally developed at Netflix to boost productivity of data scientists who work on a wide variety of projects from classical statistics to state-of-the-art deep learning.

## Documentation

For more information on using and configuring Metaflow integration, please see: https://www.comet.ml/docs/v2/integrations/third-party-tools/metaflow/

## See it

[Here is an example project](https://www.comet.com/examples/comet-example-metaflow-model-evaluation/view/Erns9fTvjSvl7nLabBJoydPxg/panels) with the results of a Metaflow run.

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

Set Comet Credentials

```shell
export COMET_API_KEY=<Your Comet API Key>
export COMET_WORKSPACE=<Your Comet Workspace>
```

## Run the example

In this guide, we will demonstrate how to use Comet's Metaflow integration to build a simple model evaluation flow.

```shell
python metaflow_model_evaluation.py run --max-workers 1 --n_samples 100
```

Our flow consists of two steps.

### 1. An evaluation step

In this step, we will evaluate models from the [timm](https://timm.fast.ai/) library on the [Imagenet Sketches dataset.](https://huggingface.co/datasets/imagenet_sketch)

For each model under consideration, we are going to create an experiment, stream a fixed number of examples from the dataset, and log the resulting model evaluation data to Comet.

This data includes:

1. The Pretained Model Name
2. A [Classification Report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report) for the models performance on the dataset examples
3. Flow related parameters. These are global parameters for our Flow that are autologged by Comet.

### 2. A model registration step

Once we have logged the performance of each model, we will register the model with the highest macro average recall across all classes to the [Comet Model Registry](https://www.comet.com/site/products/machine-learning-model-versioning/).

