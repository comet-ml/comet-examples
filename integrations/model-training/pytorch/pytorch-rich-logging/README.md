# Pytorch integration with Comet.ml

[PyTorch](https://pytorch.org/) is a popular open source machine learning framework based on the Torch library, used for applications such as computer vision and natural language processing.

PyTorch enables fast, flexible experimentation and efficient production through a user-friendly front-end, distributed training, and ecosystem of tools and libraries.

Instrument PyTorch with Comet to start managing experiments, create dataset versions and track hyperparameters for faster and easier reproducibility and collaboration.

## Documentation

For more information on using and configuring the PyTorch integration, see: https://www.comet.com/docs/v2/integrations/ml-frameworks/pytorch/

## See it

Take a look at this [public Comet Project](https://www.comet.com/examples/comet-example-pytorch-rich-logging/).

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example

This example is based on the tutorial from [Yunjey](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py). The code trains an RNN to detect hand-written digits from the MNIST dataset.


```bash
python pytorch-rich-logging.py run
```
