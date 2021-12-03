# Comet and Chainer models

An example for using Comet Experiment Management with Chainer models. 

This example trains a simple deep NN on the MNIST dataset using the Chainer library, and logs a series confusion matrix images. 

## Setup

Install dependencies

```bash
pip install -r requirements.txt
```

Set your API key

```
export COMET_API_KEY=<Your API Key>
```

## To Run

```
python train-example.py
```

## Example Experiment
You can find an example of a completed run in this [Experiment](https://www.comet.ml/team-comet-ml/chainer/)
