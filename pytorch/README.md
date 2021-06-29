# comet-pytorch-example

## Using Comet.ml to track PyTorch experiments

The following code snippets shows how to use PyTorch with Comet.ml. Based on the tutorial from [Yunjey](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py) this code trains an RNN to detect hand writted digits from the MNIST dataset.

By initializing the `Experiment()` object, Comet.ml will log stdout and source code. To log hyper-parameters, metrics and visualizations, we add a few function calls such as `experiment.log_metric()` and `experiment.log_parameters()`.

Find out more about how you can customize Comet.ml in our documentation: https://www.comet.ml/docs/

## Using Comet.ml with Pytorch Distributed Data Parallel (DDP) training

### Setup
All machines on the network need to be able to talk to each-other on the unprivileged port range (1024-65535) in addition to the master port. 

You will need to define the IP, `master_addr`, of the "master" node that is reachable by all of the other machines and the port, `master_port`, on which the workers will connect. Make sure all of the machines have the same number of GPUS.

### Logging Metrics from Multiple Machines 
To capture model metrics and system metrics (GPU/CPU Usage, RAM etc) from each machine while running distributed training, we recommend creating an Experiment object per GPU process, and grouping these experiments under a user provided run ID. 

An example project can be found [here.](https://www.comet.ml/team-comet-ml/pytorch-ddp-cifar10/view/Tzf2pUfV5BWOVa36eWoW0HOO1)

In order to reproduce the project, you will need to run the `comet-pytorch-ddp-cifar10.py` example. The example can be run in the following ways

- Single Machine, Multiple GPUs
- Multiple Machines, Multiple GPUs

##### Running the Example
We will run the example based on the assumption that we have 2 machines, with a single GPU each. We will use `192.168.1.1` as our `master_addr` and `8892` as our `master_port`.  

On the master node, start the script with the following command

```
python comet-pytorch-ddp-cifar10.py \
--nodes 2 \
--gpus 1 \
--node_rank 0 \
--master_addr 192.168.1.1 \
--master_port 8892 \
--epochs 5 \
--replica_batch_size 32 \
--run_id <your run name>
```

On the worker node run 

```
python comet-pytorch-ddp-cifar10.py \
--nodes 2 \
--gpus 1 \
--node_rank 1 \
--master_addr 192.168.1.1 \
--master_port 8892 \
--epochs 5 \
--replica_batch_size 32 \
--run_id <your run name>
```

The command line arguments are: 

```
nodes: The number of available compute nodes

gpus: The number of GPUs available on each machine

node_rank: The ranking of the machine within the nodes. It starts at 0

replica_batch_size: The batch size allocated to a single GPU process

run_id: A user provided string that allows us to group the experiments from a single run. 
```

As you add machines and GPUs, you will have to run the same command on each machine while incrementing the `node_rank`. For example in the case of N machines, we would run the script on each machine up until `node_rank = N-1`

### Logging Metrics from Multiple Machines as a single Experiment

If you would like to log the metrics from each worker as a single experiment, you will need to run the `comet-pytorch-ddp-mnist-single-experiment.py` example. Keep in mind, logging system metrics (CPU/GPU Usage, RAM, etc) from mutiple workers as a single experiment is not currently supported. We recommend using an Experiment per GPU process instead.            

An example project can be found [here.](https://www.comet.ml/team-comet-ml/pytorch-ddp-mnist-single/view/new)

##### Running the Example
We will run the example based on the assumption that we have 2 machines, with a single GPU each. We will use `192.168.1.1` as our `master_addr` and `8892` as our `master_port`.  

On the master node, start the script with the following command 

```
python comet-pytorch-ddp-mnist-single-experiment.py \
--nodes 2 \
--gpus 1 \
--master_addr 192.168.1.1 \
--master_port 8892 \
--node_rank 0 \
--local_rank 0 \ 
--run_id <your run name>
```

In this case, the sha256 hash of the `run_id` string will be used to create an experiment key for the Experiment that will be used to log the metrics from each worker.   

On the worker node run

```
python comet-pytorch-ddp-mnist-single-experiment.py \
--nodes 2 \
--gpus 1 \
--master_addr 192.168.1.1 \
--master_port 8892 \
--node_rank 1 \
--local_rank 0 \ 
--run_id <your run name>
```

The command line arguments are: 

```
nodes: The number of available compute nodes

gpus: The number of GPUs available on each machine

node_rank: The ranking of the machine within the nodes. It starts at 0

local_rank: The rank of the process within the current node

run_id: A user provided string 
```

### Using Comet.ml with Horovod and Pytorch
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

>If you're curious about learning more on parallelized training in Pytorch, checkout our report [here](https://www.comet.ml/team-comet-ml/parallelism/reports/advanced-ml-parallelism
)