# Comet and MXNet models

An example for using Comet Experiment Management with MXNet models using the CIFAR10 dataset. 

## Setup

Install dependencies

```bash
pip install -r requirements.txt
```

Set your API key and workspace name

```
export COMET_API_KEY=<Your API Key>
export COMET_WORKSPACE=<Your Workspace Name>
```

## To Run

Specify arguments including the model, batch size, dropout, gpu usage etc. See lines 30-65 in mxnet_cifar10.py for all arguments. 

```
python mxnet_cifar10.py --model cifar_resnet20_v1
```

## Example Experiment
You can find an example of a completed run for the classification model in this [Experiment](https://www.comet.ml/team-comet-ml/mxnet-comet-tutorial/view/keLYGFlti8CiCNSD4sOhFxeWl/panels)

For further information, refer to the [Comet Blog](https://www.comet.ml/site/implementing-resnet-with-mxnet-gluon-and-comet-ml-for-image-classification/).


