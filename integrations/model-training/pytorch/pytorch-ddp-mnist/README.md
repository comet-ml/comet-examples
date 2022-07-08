
### Logging Metrics from Multiple Machines as a single Experiment

If you would like to log the metrics from each worker as a single experiment, you will need to run the `pytorch-ddp-mnist-single-experiment.py` example. Keep in mind, logging system metrics (CPU/GPU Usage, RAM, etc) from mutiple workers as a single experiment is not currently supported. We recommend using an Experiment per GPU process instead.

An example project can be found [here.](https://www.comet.ml/team-comet-ml/pytorch-ddp-mnist-single/view/new)

##### Running the Example
We will run the example based on the assumption that we have 2 machines, with a single GPU each. We will use `192.168.1.1` as our `master_addr` and `8892` as our `master_port`.

On the master node, start the script with the following command

```
python pytorch-ddp-mnist-single-experiment.py \
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
python pytorch-ddp-mnist-single-experiment.py \
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