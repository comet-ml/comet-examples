# Comet and Chainer models

An example for using Comet Experiment Management with Keras models. 

This example trains a simple deep NN on the MNIST dataset using the Keras library. There are 3 scripts. 
1. "keras-mnist-artifact-load" loads the datasets used to train and test the experiment into Comet.
2. "keras-mnist" is a simple experiment that showcases Comets auto-logging features
3. "keras-mnist-rich" logs additiona metrics, parameters, images, histograms, and an interactive confusion matrix via the Comet SDK. 

Find out more about how you can customize Comet.ml in our documentation: https://www.comet.ml/docs/

## Setup

Install dependencies

```bash
pip install -r requirements.txt
```

Replace experiment parameters in script

## To Run

```
python keras-mnist.py
```

## Example Experiment
You can find an example of a completed run in this [Experiment](https://www.comet.ml/team-comet-ml/chainer/)
