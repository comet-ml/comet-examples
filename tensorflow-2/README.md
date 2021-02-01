#### Getting started with Comet
   * get an api key from https://www.comet.ml
   * install Comet:
```
    pip3 install comet_ml
    pip install comet_ml
```

#### Set API Key as an environment variable
```
export COMET_API_KEY=<your-api-key>
```
Learn more about [configuring Comet](https://www.comet.ml/docs/python-sdk/advanced/)

#### Running Distributed Training
The distributed training examples will require a multi-GPU machine.     

Start the MultiWorkerMirrored strategy by running the script.
```
python comet-tensorflow-distributed-multiworker-mirrored-strategy.py
```

When running the parameter server strategy example, we recommend allocating a single GPU to each process. This can be done by setting the `CUDA_VISIBLE_DEVICES` envrionment variable to the appropriate GPU ID. For example, `export CUDA_VISIBLE_DEVICES=0` will only allow the process to access GPU ID 0.

Once you have done this, the following commands will start a parameter server on `localhost:8000` and two workers on ports `localhost:8001` and `localhost:8002`.  

Start the Parameter Server
```
python comet-tensorflow-distributed-paramter-server-strategy.py --worker_hosts localhost:8001,localhost:8002 --ps_hosts localhost:8000 --run_id 0 --task_type ps --task_index 0
```
Start the workers in different terminals 

**Worker 0**
```
python comet-tf1-distributed-paramter-server-strategy.py --worker_hosts localhost:8001,localhost:8002 --ps_hosts localhost:8000 --run_id 0 --task_type worker --task_index 0
```

**Worker 1**
```
python comet-tf1-distributed-paramter-server-strategy.py --worker_hosts localhost:8001,localhost:8002 --ps_hosts localhost:8000 --run_id 0 --task_type worker --task_index 1
```

Finally, in a new terminal, start the chief process by running
```
python comet-tf1-distributed-paramter-server-strategy.py --worker_hosts localhost:8001,localhost:8002 --ps_hosts localhost:8000 --run_id 0 --task_type chief --task_index 0
```
The chief process will launch the [coordinator](https://www.tensorflow.org/tutorials/distribute/parameter_server_training), that will distributed the data across workers, and synchronize the model weights in the parameter server. Find out more about distributed strategies [here](https://www.tensorflow.org/guide/distributed_training)

