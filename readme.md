<img src="https://www.comet.com/images/logo_comet_light.png" width="350" alt="Drawing" style="width: 350px;"/>

## Comet for Machine Learning Experiment Management
**Our Misson:** Comet is doing for ML what GitHub did for code. We allow data science teams to automagically track their datasets, code changes, experimentation history and production models creating efficiency, transparency, and reproducibility. 

We all strive to be data driven and yet every day valuable experiment results are lost and forgotten. Comet provides a dead simple way of fixing that. It works with any workflow, any ML task, any machine, and any piece of code.

## Examples Repository

This repository contains examples of using Comet in many Machine Learning Python libraries, including fastai, torch, sklearn, chainer, caffe, keras, tensorflow, mxnet, Jupyter notebooks, and with just pre Python.

If you don't see something you need, just let us know! See contact methods below.

## Documentation
[![PyPI version](https://badge.fury.io/py/comet-ml.svg)](https://badge.fury.io/py/comet-ml)

Full documentation and additional training examples are available on http://www.comet.com/docs/v2

## Installation

- [Sign up for free!](https://www.comet.com/signup)

- **Install Comet from PyPI:**

```sh
pip install comet_ml
```
Comet Python SDK is compatible with: __Python 3.5-3.13__.

## Tutorials + Examples

- [fastai](https://github.com/comet-ml/comet-examples/tree/master/integrations/model-training/fastai/)
- [keras](https://github.com/comet-ml/comet-examples/tree/master/keras)
- [pytorch](https://github.com/comet-ml/comet-examples/tree/master/pytorch)
- [scikit](https://github.com/comet-ml/comet-examples/tree/master/integrations/model-training/scikit-learn)
- [tensorflow](https://github.com/comet-ml/comet-examples/tree/master/tensorflow)

## Support 
Have questions? We have answers - 
- Email us at <info@comet.com>
- For the fastest response, ping us on [Slack](https://chat.comet.com/)

**Want to request a feature?** 
We take feature requests through github at: https://github.com/comet-ml/issue-tracking

## Feature Spotlight
Check out new product features and updates through our [Release Notes](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/releases/). Also check out our [blog](https://www.comet.com/site/blog/).

## Using pyenv+poetry to install dependencies

```sh
pyenv install 3.12.6
pyenv local 3.12.6

poetry config virtualenvs.create true
poetry config virtualenvs.in-project true

poetry install
```
