# comet-pytorch-example

#### Using Comet.ml to track PyTorch experiments

The following code snippets shows how to use PyTorch with Comet.ml. Based on the tutorial from [Yunjey](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py) this code trains an RNN to detect hand writted digits from the MNIST dataset.

By initializing the `Experiment()` object, Comet.ml will log stdout and source code. To log hyper-parameters, metrics and visualizations, we add a few function calls such as `experiment.log_metric()` and `experiment.log_multiple_params()`.

Find out more about how you can customize Comet.ml in our documentation: https://www.comet.ml/docs/
