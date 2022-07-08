# Pytorch integration with Comet.ml

Comet integrates with [PyTorch](https://pytorch.org/).

PyTorch is an open source machine learning framework based on the Torch library, used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab.

PyTorch enables fast, flexible experimentation and efficient production through a user-friendly front-end, distributed training, and ecosystem of tools and libraries.

## Documentation

For more information on using and configuring Metaflow integration, please see: https://www.comet.com/docs/v2/integrations/ml-frameworks/pytorch/

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example

This example is based on the tutorial from [Yunjey](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py) this code trains an RNN to detect hand writted digits from the MNIST dataset.


```bash
python pytorch-mnist-example.py run
```