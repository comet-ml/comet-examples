# coding: utf-8
import os

import comet_ml

# You can use 'tensorflow', 'torch' or 'jax' as backend. Make sure to set the
# environment variable before importing.
os.environ["KERAS_BACKEND"] = "tensorflow"


import mlflow.keras  # noqa: E402
import numpy as np  # noqa: E402

import keras  # noqa: E402

# Login to Comet if necessary
comet_ml.login(project_name="comet-example-mlflow-hello-world")

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
x_train[0].shape

# Build model
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)


def initialize_model():
    return keras.Sequential(
        [
            keras.Input(shape=INPUT_SHAPE),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )


model = initialize_model()
model.summary()

# Train model

BATCH_SIZE = 64  # adjust this based on the memory of your machine
EPOCHS = 3

model = initialize_model()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

run = mlflow.start_run()
model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=[mlflow.keras.MlflowCallback(run)],
)

mlflow.keras.log_model(model, "model", registered_model_name="Test Model")

mlflow.end_run()
