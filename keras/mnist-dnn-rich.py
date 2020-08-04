from __future__ import print_function
import logging
import utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from comet_ml import Experiment

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout


params = {
    'dropout': 0.2,
    'batch-size': 64,
    'epochs': 5,
    'layer-1-size': 128,
    'layer-2-size': 128,
    'initial-lr': 1e-2,
    'decay-steps': 2000,
    'decay-rate': 0.9,
    'optimizer': 'adam'
}

def main():
    
    mnist = tf.keras.datasets.mnist
    
    num_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)


    train(x_train, y_train, x_test, y_test)

def build_model_graph(exp, input_shape=(784,)):
    
    model = Sequential([
      Flatten(input_shape=(784, )),
      Dense(exp.get_parameter('layer-1-size'), activation='relu'),
      Dense(exp.get_parameter('layer-2-size'), activation='relu'),
      Dropout(exp.get_parameter('dropout')),
      Dense(10)
    ])
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=exp.get_parameter('initial-lr'),
        decay_steps=exp.get_parameter('decay-steps'),
        decay_rate=exp.get_parameter('decay-rate'))

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer='adam', 
                  loss=loss_fn,
                  metrics=['accuracy'])

    return model


def train(x_train, y_train, x_test, y_test):
    
    exp = Experiment(api_key="ERPBfa6mmwJzQnk61oiqLOCie", project_name="perception", workspace="development")
    
    # log custom hyperparameters
    exp.log_parameters(params)
    
    exp.log_metric('custom_metric', 0.97)

    
    
    # Define model
    model = build_model_graph(exp)

    model.fit(
        x_train,
        y_train,
        batch_size=exp.get_parameter('batch-size'),
        epochs=exp.get_parameter('epochs'),
        validation_data=(x_test, y_test),
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    logging.info("Score %s", score)

    
    # Finalize model includes the following calls
    # exp.log_confusion_matrix()
    # exp.log_image()
    # exp.log_histogram_3d()
    # exp.add_tag()
    # exp.log_model()
    utils.finalize_model(model, x_train, y_train, x_test, y_test, exp)
    

if __name__ == "__main__":
    main()
