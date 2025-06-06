# Pytorch integration with Comet.ml

[PyTorch](https://pytorch.org/) is a popular open source machine learning framework based on the Torch library, used for applications such as computer vision and natural language processing.

PyTorch enables fast, flexible experimentation and efficient production through a user-friendly front-end, distributed training, and ecosystem of tools and libraries.

Instrument PyTorch with Comet to start managing experiments, create dataset versions and track hyperparameters for faster and easier reproducibility and collaboration.

## Documentation

For more information on using and configuring the PyTorch integration, see: [https://www.comet.com/docs/v2/integrations/ml-frameworks/pytorch/](https://www.comet.com/docs/v2/integrations/ml-frameworks/pytorch/?utm_source=comet-examples&utm_medium=referral&utm_campaign=github_repo_2023&utm_content=pytorch)

## See it

Take a look at this [public Comet Project](https://www.comet.com/examples/comet-example-pytorch-fsdp2/view/new/panels).

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example

This example is based on the [official FSDP2 example](https://github.com/pytorch/examples/tree/main/distributed/FSDP2) training a toy transformer model.

```bash
torchrun --nproc_per_node 2 train.py
```

* For 1st time, it creates a "checkpoints" folder and saves state dicts there
* For 2nd time, it loads from previous checkpoints

To enable mixed precision

```bash
torchrun --nproc_per_node 2 train.py --mixed-precision
```
