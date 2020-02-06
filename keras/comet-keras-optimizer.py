# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.ml
#  Copyright (C) 2019 Comet ML INC
#  This file can not be copied and/or distributed without
#   the express permission of Comet ML Inc.
# *******************************************************

"""
Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
"""
from __future__ import print_function

from os.path import dirname, join
from comet_ml import Optimizer  # isort:skip

import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop


def main():

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
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    config = {
        "algorithm": "bayes",
        "name": "Optimize MNIST Network",
        "spec": {"maxCombo": 10, "objective": "minimize", "metric": "loss"},
        "parameters": {"first_layer_units": {"type": "integer", "min": 1, "max": 1000}},
        "trials": 1,
    }

    opt = Optimizer(config)

    for experiment in opt.get_experiments():
        flu = experiment.get_parameter("first_layer_units")
        loss = fit(experiment, x_train, y_train, x_test, y_test, 3, 120, flu)
        # Reverse the score for minimization
        experiment.log_metric("loss", loss)


def build_model_graph(first_layer_units):
    model = Sequential()
    model.add(Dense(first_layer_units, activation="sigmoid", input_shape=(784,)))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(10, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"]
    )

    return model


def fit(
    experiment, x_train, y_train, x_test, y_test, epoch, batch_size, first_layer_units
):
    current_dir = dirname(__file__)
    experiment.log_image(join(current_dir, "logo_comet_dark.png"))

    experiment.log_dataset_hash(x_train)
    experiment.log_parameter("first_layer_units", first_layer_units)

    # Define model
    model = build_model_graph(first_layer_units)

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epoch,
        validation_data=(x_test, y_test),
    )
    score = model.evaluate(x_test, y_test, verbose=0)[1]

    return score


if __name__ == "__main__":
    main()
