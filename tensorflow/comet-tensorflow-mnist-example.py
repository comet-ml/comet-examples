"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import comet_ml at the top of your file
from comet_ml import Experiment

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def get_data():
    mnist = input_data.read_data_sets(
        "/tmp/tensorflow/mnist/input_data/", one_hot=True)
    return mnist


def build_model_graph(hyper_params):
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(
        hyper_params['learning_rate']).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    return train_step, cross_entropy, accuracy, x, y, y_


def train(hyper_params):
    mnist = get_data()

    # Get graph definition, tensors and ops
    train_step, cross_entropy, accuracy, x, y, y_ = build_model_graph(
        hyper_params)

    # log parameters to Comet.ml
    import os
    # Setting the API key (saved as environment variable)
    exp = Experiment(
        #api_key="YOUR API KEY",
        # or
        api_key=os.environ.get("COMET_API_KEY"),
        project_name='comet-examples')
    exp.log_multiple_params(hyper_params)
    exp.log_dataset_hash(mnist)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        exp.set_model_graph(sess.graph)

        for i in range(hyper_params["steps"]):
            batch = mnist.train.next_batch(hyper_params["batch_size"])
            exp.set_step(i)
            # Compute train accuracy every 10 steps
            if i % 10 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={x: batch[0], y_: batch[1]})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                exp.log_metric("acc", train_accuracy)

            # Update weights (back propagation)
            loss = train_step.run(feed_dict={x: batch[0], y_: batch[1]})
            exp.log_metric("loss", loss)

        ### Finished Training ###

        # Compute test accuracy
        acc = accuracy.eval(
            feed_dict={x: mnist.test.images, y_: mnist.test.labels})

        print('test accuracy %g' % acc)


if __name__ == '__main__':
    hyper_params = {"learning_rate": 0.5, "steps": 100000, "batch_size": 50}
train(hyper_params)
