# comet-tensorflow-example
   * update your Comet installation
```
pip install --no-cache-dir --upgrade comet_ml
pip3 install --no-cache-dir --upgrade comet_ml
    
```

## Getting started with Comet
   * get an api key from https://www.comet.ml
   * install Comet:
```
    pip3 install comet_ml
    pip install comet_ml
```

   * import Comet:  [code](https://github.com/comet-ml/comet-quickstart-guide/blob/master/tensorflow/comet_tensorflow_example.py#L11)
```
#make sure comet_ml is the first import (before all other Machine learning lib)
from comet_ml import Experiment
```
   * create an Experiment: [code](https://github.com/comet-ml/comet-quickstart-guide/blob/master/tensorflow/comet_tensorflow_example.py#L45)
```
# initiate an Experiment with your api key from https://www.comet.ml
experiment = Experiment(api_key="YOUR-API-KEY", project_name='my project')
```
+ report hyper params: [code](https://github.com/comet-ml/comet-quickstart-guide/blob/master/tensorflow/comet_tensorflow_example.py#L46)
```
hyper_params = {"learning_rate": 0.5, "steps": 100000, "batch_size": 50}
experiment.log_parameters(hyper_params)
```
+ report dataset hash: [code](https://github.com/comet-ml/comet-quickstart-guide/blob/master/tensorflow/comet_tensorflow_example.py#L47)
```
mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data/", one_hot=True)
experiment.log_dataset_hash(mnist)
```
+ report loss, accuracy and steps: [code](https://github.com/comet-ml/comet-quickstart-guide/blob/master/tensorflow/comet_tensorflow_example.py#L53-L64)
```
        for i in range(hyper_params["steps"]):
            batch = mnist.train.next_batch(hyper_params["batch_size"])
            experiment.log_step(i)
            # Compute train accuracy every 10 steps
            if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                experiment.log_metric("acc", train_accuracy)

            # Update weights (back propagation)
            loss = train_step.run(feed_dict={x: batch[0], y_: batch[1]})
            experiment.log_loss(loss)
```

   * run your code as usual and view results 

   * see full code example at: [link](https://github.com/comet-ml/comet-quickstart-guide/blob/master/tensorflow/comet_tensorflow_example.py)

## Running Distributed Training

The distributed training examples will require a multi-GPU machine and have been tested with `tensorflow-gpu==1.15.4` and `tensorflow-extimator=1.15.1`.

### Mirrored Worker strategy example

You can start the MirroredWorker strategy example with the following command, it will automatically uses all available GPU and you only need to launch the command once:

```
python comet-tf1-distributed-mirrored-strategy.py
```

### MultiWorkerMirrored Estimator strategy example

To start the MultiWorkerMirrored strategy with TF Estimator we will need to start a chief process, an evaluator process and a worker process. We will also need to supply a `run_id` for the training run so that metrics from each process can be logged to a single experiment. The `run_id` is a string that is hashed to compute the Experiment ID. We also recommend allocating a single GPU to each process used in this example. This can be done by setting the `CUDA_VISIBLE_DEVICES` envrionment variable to the appropriate GPU ID. For example, `export CUDA_VISIBLE_DEVICES=0` will only allow the process to access GPU ID 0. 

**Note:** You will need to start the evaluator process before starting the chief and worker process. 

The following command will start a evaluator process on `localhost:8002` with `task_index == 0`. 

```
python comet-tf1-distributed-estimator-multiworker-mirrored-strategy.py --chief_host localhost:8000 --worker_hosts localhost:8001 --eval_hosts localhost:8002 --task_index 0 --task_type evaluator --run_id <your run id>
```

The following command will start a chief process on `localhost:8000` with `task_index == 0`. 

```
python comet-tf1-distributed-estimator-multiworker-mirrored-strategy.py --chief_host localhost:8000 --worker_hosts localhost:8001 --eval_hosts localhost:8003 --task_index 0 --task_type chief --run_id <your run id>

```
The following command will start a worker process on `localhost:8001` with `task_index == 0`. 

```
python comet-tf1-distributed-estimator-multiworker-mirrored-strategy.py --chief_host localhost:8000 --worker_hosts localhost:8001 --eval_hosts localhost:8003 --task_index 0 --task_type worker --run_id <your run id>
```

### Parameter Server Strategy example

When running the TF1 parameter server strategy example, we recommend allocating a single GPU to each process. This can be done by setting the `CUDA_VISIBLE_DEVICES` envrionment variable to the appropriate GPU ID. For example, `export CUDA_VISIBLE_DEVICES=0` will only allow the process to access GPU ID 0.

Once you have done this, the following commands will start a parameter server on `localhost:8000` and two workers on ports `localhost:8001` and `localhost:8002`

**Start the Parameter Server**
```
python comet-tf1-distributed-parameter-server-strategy.py --worker_hosts localhost:8001,localhost:8002 --ps_hosts localhost:8000 --run_id 0 --task_type ps --task_index 0
```

Start the workers in different terminals 

**Worker 0**
```
python comet-tf1-distributed-parameter-server-strategy.py --worker_hosts localhost:8001,localhost:8002 --ps_hosts localhost:8000 --run_id 0 --task_type worker --task_index 0
```

**Worker 1**
```
python comet-tf1-distributed-parameter-server-strategy.py --worker_hosts localhost:8001,localhost:8002 --ps_hosts localhost:8000 --run_id 0 --task_type worker --task_index 1
```

