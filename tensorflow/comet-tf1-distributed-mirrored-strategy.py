# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# Copyright (C) 2021 Comet ML INC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import comet_ml

# Import TensorFlow
import tensorflow as tf

# Helper libraries
import numpy as np
import os

PROJECT_NAME = "tf1-mirrored"
experiment = comet_ml.Experiment(log_code=True, project_name=PROJECT_NAME)
print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Adding a dimension to the array -> new shape == (28, 28, 1)
# We are doing this because the first layer in our model is a convolutional
# layer and it requires a 4D input (batch_size, height, width, channels).
# batch_size dimension will be added later on.
train_images = train_images[..., None]
test_images = test_images[..., None]

# Getting the images in [0, 1] range.
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

train_labels = train_labels.astype("int64")
test_labels = test_labels.astype("int64")

# If the list of devices is not specified in the
# `tf.distribute.MirroredStrategy` constructor, it will be auto-detected.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

BUFFER_SIZE = len(train_images)

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 10
STEPS_PER_EPOCH = 100


with strategy.scope():
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
    )
    train_ds = strategy.experimental_distribute_dataset(train_dataset)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(
        BATCH_SIZE
    )
    test_ds = strategy.experimental_distribute_dataset(test_dataset)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    optimizer = tf.train.GradientDescentOptimizer(0.001)

    def train_step(dist_inputs):
        def step_fn(inputs):
            images, labels = inputs
            logits = model(images)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels
            )
            loss = loss = tf.reduce_sum(cross_entropy) * (1.0 / BATCH_SIZE)
            train_op = optimizer.minimize(loss)
            with tf.control_dependencies([train_op]):
                return tf.identity(loss)

        per_replica_losses = strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
        mean_loss = strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )
        return mean_loss

    train_iterator = train_ds.make_initializable_iterator()
    iterator_init = train_iterator.initializer
    var_init = tf.global_variables_initializer()
    loss = train_step(next(train_iterator))

    with tf.train.MonitoredTrainingSession() as sess:
        sess.run([var_init])
        for epoch in range(EPOCHS):
            sess.run([iterator_init])
            for step in range(STEPS_PER_EPOCH):
                current_loss = sess.run(loss)
                if step % 10 == 0:
                    print(
                        "Epoch {} Step {} Loss {:.4f}".format(epoch, step, current_loss)
                    )
                    experiment.log_metric(
                        "loss", current_loss, step=(STEPS_PER_EPOCH * epoch) + step
                    )
