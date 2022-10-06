# Pytorch Tensorboard integration with Comet.ml

[PyTorch](https://pytorch.org/) is a popular open source machine learning framework based on the Torch library, used for applications such as computer vision and natural language processing.

PyTorch enables fast, flexible experimentation and efficient production through a user-friendly front-end, distributed training, and ecosystem of tools and libraries.

TensorBoard is a visualization toolkit for machine learning experimentation. TensorBoard allows tracking and visualizing metrics such as loss and accuracy, visualizing the model graph, viewing histograms, displaying images and much more.

Pytorch now includes native Tensorboard support to let you log PyTorch models, metrics and images.

Instrument PyTorch's Tensorboard with Comet to start managing experiments, create dataset versions and track hyperparameters for faster and easier reproducibility and collaboration.

## Documentation

For more information on using and configuring the PyTorch integration, see: https://www.comet.com/docs/v2/integrations/ml-frameworks/pytorch/

## See it

Take a look at this [public Comet Project]().

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example

This example is based on the [Pytorch tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html). The code trains a CNN to classify clothing using the Fashion-MNIST dataset.

```bash
python pytorch-tensorboard-example.py
```
