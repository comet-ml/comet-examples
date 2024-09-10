# coding: utf-8

import logging
from pathlib import Path

import comet_ml

from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

import tensorflow as tf

params = {
    "dropout": 0.2,
    "batch-size": 64,
    "epochs": 5,
    "layer-1-size": 128,
    "layer-2-size": 128,
    "optimizer": "adam",
}

# Login to Comet if needed
comet_ml.login()


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


def build_model_graph(experiment, input_shape=(784,)):

    model = Sequential(
        [
            Flatten(input_shape=(784,)),
            Dense(experiment.get_parameter("layer-1-size"), activation="relu"),
            Dense(experiment.get_parameter("layer-2-size"), activation="relu"),
            Dropout(experiment.get_parameter("dropout")),
            Dense(10),
        ]
    )

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    return model


def finalize_model(experiment, model, x_train, y_train, x_test, y_test):
    def test_index_to_example(index):
        img = x_test[index].reshape(28, 28)
        # log the data to Comet, whether it's log_image, log_text, log_audio, ...
        data = experiment.log_image(img, name="test_%d.png" % index)

        if data is None:
            return None

        return {"sample": str(index), "assetId": data["imageId"]}

    # Add tags
    experiment.add_tag("keras-mnist-ddn")

    # Confusion Matrix
    preds = model.predict(x_test)

    experiment.log_confusion_matrix(
        y_test, preds, index_to_example_function=test_index_to_example
    )

    # Log Histograms
    for layer in model.layers:
        if layer.get_weights() != []:
            x = layer.get_weights()
            for _, lst in enumerate(x):
                experiment.log_histogram_3d(lst, name=layer.name, step=_)

    # Log Model
    Path("models/").mkdir(exist_ok=True)
    model.save("models/mnist-nn.h5")
    experiment.log_model("mnist-neural-net", "models/mnist-nn.h5")


def train(x_train, y_train, x_test, y_test):

    experiment = comet_ml.Experiment(project_name="comet-example-keras-mnist-dnn")

    # Log custom hyperparameters
    experiment.log_parameters(params)

    # Define model
    model = build_model_graph(experiment)

    model.fit(
        x_train,
        y_train,
        batch_size=experiment.get_parameter("batch-size"),
        epochs=experiment.get_parameter("epochs"),
        validation_data=(x_test, y_test),
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    logging.info("Score %s", score)

    finalize_model(experiment, model, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
