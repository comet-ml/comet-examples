# Composer integration with Comet.ml

[Composer](https://github.com/mosaicml/composer) is an open-source deep learning training library by [MosaicML](https://www.mosaicml.com/). Built on top of PyTorch, the Composer library makes it easier to implement distributed training workflows on large-scale clusters.

Instrument Composer with Comet to start managing experiments, create dataset versions and track hyperparameters for faster and easier reproducibility and collaboration.

## See it

Take a look at this [public Comet Project](https://www.comet.com/examples/comet-example-pytorch-mnist?utm_source=comet-examples&utm_medium=referral&utm_campaign=github_repo_2023&utm_content=pytorch).

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example

This example is based on the [offical Getting Started example](https://colab.research.google.com/github/mosaicml/composer/blob/master/examples/getting_started.ipynb#scrollTo=7a7HokeLUFLO). The code trains an Resnet to detect classes from the Cifar-10 dataset.


```bash
python mosaicml-getting-started.py
```
