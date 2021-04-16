# comet-pytorch-example

#### Using Comet.ml to track PyTorch experiments

The following code snippets shows how to use PyTorch with Comet.ml. Based on the tutorial from [Yunjey](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py) this code trains an RNN to detect hand writted digits from the MNIST dataset.

By initializing the `Experiment()` object, Comet.ml will log stdout and source code. To log hyper-parameters, metrics and visualizations, we add a few function calls such as `experiment.log_metric()` and `experiment.log_parameters()`.

Find out more about how you can customize Comet.ml in our documentation: https://www.comet.ml/docs/

##### Using Comet.ml with Pytorch Parallel Data training

This directory contains an example how to track an experiment when using Pytorch DDP for Data Parallelism across multiple machines. The example files are named `comet-pytorch-ddp-mnist-example.py` and `comet-pytorch-ddp-cifar10.py`. You can launch either of them in the following way:

* You will need X machines, each of them should have Pytorch and Comet.ml installed. Also make sure the Comet API Key is either configured on the system or update the script to hard-code it. They need to be able to talk to each-other on the unprivileged port range (1024-65535) in addition to the master port. You will need to find the IP of the "master" node that is reachable by all of the other machines, `MASTER_ADDR`. Make sure all of the machines have the same number of GPUS, `Y`.
* On the master node, run the script like that: `python comet-pytorch-ddp-mnist-example.py --nodes X --gpus Y --nr 0 --master_addr MASTER_ADDR --master_port 8892`
* On the next node, run the script like this: `python comet-pytorch-ddp-mnist-example.py --nodes X --gpus Y --nr 1 --master_addr MASTER_ADDR --master_port 8892`
* On the next node, run the script like this: `python comet-pytorch-ddp-mnist-example.py --nodes X --gpus Y --nr 2 --master_addr MASTER_ADDR --master_port 8892`
* ...
* On the last node, run the script like this: `python comet-pytorch-ddp-mnist-example.py --nodes X --gpus Y --nr X-1 --master_addr MASTER_ADDR --master_port 8892`

For example, with 2 machines, each having 2 gpus and the master node IP being `192.168.1.1`, to run the distributed training:
* On the master node: `python comet-pytorch-ddp-mnist-example.py --nodes 2 --gpus 2 --nr 0 --master_addr 192.168.1.1 --master_port 8892`
* On the other node: `python comet-pytorch-ddp-mnist-example.py --nodes 2 --gpus 2 --nr 1 --master_addr 192.168.1.1 --master_port 8892`

##### Using Comet.ml with Horovod and Pytorch
In order to run the Horovod example in a distributed manner, you will need the following
```bash
1. At least 2 machines 1 gpu each.
2. Ensure that the master node has ssh access to the worker machines without requiring a password
3. Horovod and Comet.ml installed on all machines
4. This script must be present in the same directory on all machines.
```
To the run the example 

```
horovodrun --gloo -np <Number of Nodes * Number of GPUS> -H <server1_address:num_gpus>,<server2_address:num_gpus>,<server3_address:num_gpus> python comet-pytorch-horovod-example.py
```