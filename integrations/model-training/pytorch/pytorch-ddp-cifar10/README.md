# Using Comet.ml with Pytorch Distributed Data Parallel (DDP) training

## Setup
All machines on the network need to be able to talk to each-other on the unprivileged port range (1024-65535) in addition to the master port.

You will need to define the IP, `master_addr`, of the "master" node that is reachable by all of the other machines and the port, `master_port`, on which the workers will connect. Make sure all of the machines have the same number of GPUS.

## Logging Metrics from Multiple Machines
To capture model metrics and system metrics (GPU/CPU Usage, RAM etc) from each machine while running distributed training, we recommend creating an Experiment object per GPU process, and grouping these experiments under a user provided run ID.

An example project can be found [here.](https://www.comet.ml/team-comet-ml/pytorch-ddp-cifar10/view/Tzf2pUfV5BWOVa36eWoW0HOO1)

In order to reproduce the project, you will need to run the `pytorch-ddp-cifar10.py` example. The example can be run in the following ways

- Single Machine, Multiple GPUs
- Multiple Machines, Multiple GPUs

## Running the Example
We will run the example based on the assumption that we have 2 machines, with a single GPU each. We will use `192.168.1.1` as our `master_addr` and `8892` as our `master_port`.

On the master node, start the script with the following command

```
python pytorch-ddp-cifar10.py \
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
python pytorch-ddp-cifar10.py \
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