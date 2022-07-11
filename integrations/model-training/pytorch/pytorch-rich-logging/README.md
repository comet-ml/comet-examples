# Pytorch integration with Comet.ml

Comet integrates with [PyTorch](https://pytorch.org/).

PyTorch is an open source machine learning framework based on the Torch library, used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab.

PyTorch enables fast, flexible experimentation and efficient production through a user-friendly front-end, distributed training, and ecosystem of tools and libraries.

## Documentation

For more information on using and configuring Metaflow integration, please see: https://www.comet.com/docs/v2/integrations/ml-frameworks/pytorch/

## See it live

You can see how this example looks like by taking a look at this [public Comet Project](https://www.comet.com/examples/comet-example-pytorch-rich-logging/).

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example

In order to run the Horovod example in a distributed manner, you will need the following:

1. At least 2 machines 1 gpu each.
2. Ensure that the master node has ssh access to the worker machines without requiring a password.
3. Horovod and Comet.ml installed on all machines.
4. This script must be present in the same directory on all machines.

To the run the example:

```
horovodrun --gloo -np <Number of Nodes * Number of GPUS> -H <server1_address:num_gpus>,<server2_address:num_gpus>,<server3_address:num_gpus> python pytorch-horovod-example.py
```

>If you're curious about learning more on parallelized training in Pytorch, checkout our report [here](https://www.comet.ml/team-comet-ml/parallelism/reports/advanced-ml-parallelism).