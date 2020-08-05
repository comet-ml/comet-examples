# comet-pytorch-example

#### Using Comet.ml to track PyTorch experiments

The following code snippets shows how to use PyTorch with Comet.ml. Based on the tutorial from [Yunjey](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py) this code trains an RNN to detect hand writted digits from the MNIST dataset.

By initializing the `Experiment()` object, Comet.ml will log stdout and source code. To log hyper-parameters, metrics and visualizations, we add a few function calls such as `experiment.log_metric()` and `experiment.log_parameters()`.

Find out more about how you can customize Comet.ml in our documentation: https://www.comet.ml/docs/

##### Using Comet.ml with Pytorch Parallel Data training

This directory contains an example how to track an experiment when using Pytorch DDP for Data Parallelism across multiple machines. The example file is named `comet-pytorch-ddp-mnist-example.py` and you can launch it the following way:

* You will need X machines, each of them should have Pytorch and Comet.ml. Also make sure the Comet API Key is either configured on the system or update the script to hard-code it. They need to be able to talk to each-other on the unprivileged port range (1024-65535) in addition to the master port. You will need to find the IP of the "master" node that is reachable by all of the other machines, `MASTER_ADDR`. Make sure all of the machines have the same number of GPUS, `Y`.
* On the master node, run the script like that: `python comet-pytorch-ddp-mnist-example.py --nodes X --gpus Y --nr 0 --master_addr MASTER_ADDR --master_port 8892`
* On the next node, run the script like this: `python comet-pytorch-ddp-mnist-example.py --nodes X --gpus Y --nr 1 --master_addr MASTER_ADDR --master_port 8892`
* On the next node, run the script like this: `python comet-pytorch-ddp-mnist-example.py --nodes X --gpus Y --nr 2 --master_addr MASTER_ADDR --master_port 8892`
* ...
* On the last node, run the script like this: `python comet-pytorch-ddp-mnist-example.py --nodes X --gpus Y --nr X-1 --master_addr MASTER_ADDR --master_port 8892`

For example, with 2 machines, each having 2 gpus and the master node IP being `192.168.1.1`, to run the distributed training:
* On the master node: `python comet-pytorch-ddp-mnist-example.py --nodes 2 --gpus 2 --nr 0 --master_addr 192.168.1.1 --master_port 8892`
* On the other node: `python comet-pytorch-ddp-mnist-example.py --nodes 2 --gpus 2 --nr 1 --master_addr 192.168.1.1 --master_port 8892`