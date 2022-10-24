# Model Evaluation Flow with Metaflow and Comet

In this guide, we will demonstrate how to use Comet's Metaflow integration to build a simple model evaluation flow.

Our flow consists of two steps.

## 1. An evaluation step

In this step, we will evaluate models from the [timm](https://timm.fast.ai/) library on the [Imagenet Sketches dataset.](https://huggingface.co/datasets/imagenet_sketch)

For each model under consideration, we are going to create an experiment, stream a fixed number of examples from the dataset, and log the resulting model evaluation data to Comet.

This data includes:

1. The Pretained Model Name
2. A [Classification Report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report) for the models performance on the dataset examples
3. Flow related parameters. These are global parameters for our Flow that are autologged by Comet.

## 2. A model registration step

Once we have logged the performance of each model, we will register the model with the highest macro average recall across all classes to the [Comet Model Registry](https://www.comet.com/site/products/machine-learning-model-versioning/).

## Example Project

[Here is an example project](https://www.comet.com/team-comet-ml/comet-example-metaflow-model-evaluation/view/oq9fv1aJFAzkJJXhqQHcmS62D/panels) with the results of a Metaflow run.