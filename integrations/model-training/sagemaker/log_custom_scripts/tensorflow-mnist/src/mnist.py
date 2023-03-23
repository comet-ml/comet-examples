import argparse
import os

import comet_ml
import numpy as np
from PIL import Image

import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from keras.models import Sequential
from keras.optimizers import SGD

if __name__ == "__main__":
    experiment = comet_ml.Experiment(auto_histogram_gradient_logging=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--training", type=str, default=os.environ["SM_CHANNEL_TRAINING"]
    )
    parser.add_argument(
        "--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"]
    )

    args, _ = parser.parse_known_args()

    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    model_dir = args.model_dir
    training_dir = args.training
    validation_dir = args.validation

    x_train = np.load(os.path.join(training_dir, "training.npz"))["image"]
    y_train = np.load(os.path.join(training_dir, "training.npz"))["label"]
    x_val = np.load(os.path.join(validation_dir, "validation.npz"))["image"]
    y_val = np.load(os.path.join(validation_dir, "validation.npz"))["label"]

    # input image dimensions
    img_rows, img_cols = 28, 28

    # Tensorflow needs image channels last, e.g. (batch size, width, height, channels)
    K.set_image_data_format("channels_last")
    print(K.image_data_format())

    if K.image_data_format() == "channels_first":
        print("Incorrect configuration: Tensorflow needs channels_last")
    else:
        # channels last
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        batch_norm_axis = -1

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_val.shape[0], "test samples")

    # Normalize pixel values
    x_train = x_train.astype("float32")
    x_val = x_val.astype("float32")
    x_train /= 255
    x_val /= 255

    # Convert class vectors to binary class matrices
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    model = Sequential()

    # 1st convolution block
    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", input_shape=input_shape))
    model.add(BatchNormalization(axis=batch_norm_axis))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # 2nd convolution block
    model.add(Conv2D(128, kernel_size=(3, 3), padding="valid"))
    model.add(BatchNormalization(axis=batch_norm_axis))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # Fully connected block
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(num_classes, activation="softmax"))

    print(model.summary())

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        epochs=epochs,
        verbose=1,
    )

    eval_loss, eval_accuracy = model.evaluate(x_val, y_val, verbose=0)
    print("Validation loss    :", eval_loss)
    print("Validation accuracy:", eval_accuracy)
    experiment.log_metrics({"eval_loss": eval_loss, "eval_accuracy": eval_accuracy})

    def index_to_example(index):
        image_array = x_val[index].reshape(img_rows, img_cols)
        image = Image.fromarray(np.uint8(image_array * 255))

        image_name = "confusion-matrix-%05d.png" % index
        results = experiment.log_image(image, name=image_name)
        # Return sample, assetId (index is added automatically)
        return {"sample": image_name, "assetId": results["imageId"]}

    def log_predictions(model, x, labels, num_samples):
        predictions = model.predict(x)
        CLASS_LABELS = [f"class_{i}" for i in range(10)]

        experiment.log_confusion_matrix(
            labels[:num_samples],
            predictions[:num_samples],
            labels=CLASS_LABELS,
            index_to_example_function=index_to_example,
            title="Confusion Matrix: Evaluation",
            file_name="confusion-matrix-eval.json",
        )

    def log_model(model_path):
        experiment.log_model("mnist-tf-classifier", model_path)

    log_predictions(model, x_val, y_val, 100)
    model.save(model_dir)
    log_model(model_dir)
