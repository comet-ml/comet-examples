"""Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
"""

from __future__ import print_function

import sys

from comet_ml import (
    Experiment,
    Optimizer,
    NoMoreSuggestionsAvailable,
)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
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

    ## Gets API from config or environment:
    opt = Optimizer()
    pcs_content = """
first_layer_units integer [1,1000] [2]
"""
    # opt.set_params(pcs_content)
    opt.set_params(pcs_content)

    while True:
        try:
            sug = opt.get_suggestion()
        except NoMoreSuggestionsAvailable:
            break
        print("SUG", sug, sug.__dict__)
        flu = sug["first_layer_units"]
        print("FLU", repr(flu))
        score = train(x_train, y_train, x_test, y_test, 3, 120, flu)
        print("Score", score, sug.__dict__)
        # Reverse the score for minimization
        sug.report_score("score", score)


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


def train(x_train, y_train, x_test, y_test, epoch, batch_size, first_layer_units):
    ## Gets API from config or environment:
    experiment = Experiment(project_name="opt-prod-III")


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
