# comet-tensorflow-example

#### Commet is still in beta version with new features adding up each day
   * update your Comet installation
```
pip install --no-cache-dir --upgrade comet_ml
pip3 install --no-cache-dir --upgrade comet_ml
    
```

#### Getting started with Comet
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
experiment.log_multiple_params(hyper_params)
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

   * run your code as usual an view results on https://www.comet.ml/view/yourApiKey

   * see full code example at: [link](https://github.com/comet-ml/comet-quickstart-guide/blob/master/tensorflow/comet_tensorflow_example.py)
