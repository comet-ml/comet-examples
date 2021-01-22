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
import tensorflow as tf

import numpy as np
import os, json

N_REPLICAS = len(tf.config.experimental.get_visible_devices(device_type="GPU"))

print("Number of available devices: ", N_REPLICAS)

BUFFER_SIZE = 10000
BATCH_SIZE = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE * N_REPLICAS
LEARNING_RATE = 1e-5
EPOCHS = 10

experiment = comet_ml.Experiment(log_code=True)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images[..., None]
test_images = test_images[..., None]

train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

train_labels = train_labels.astype("int64")
test_labels = test_labels.astype("int64")


def input_fn(mode, input_context=None):
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    if input_context:
        train_dataset = train_dataset.shard(
            input_context.num_input_pipelines, input_context.input_pipeline_id
        )
    return (
        train_dataset.map(scale)
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(GLOBAL_BATCH_SIZE)
        .repeat(EPOCHS)
    )


def model_fn(features, labels, mode):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    logits = model(features, training=False)
    metrics_dict = {"accuracy": tf.metrics.accuracy(labels, tf.argmax(logits, axis=-1))}

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"logits": logits}
        return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels
    )
    loss = tf.reduce_sum(cross_entropy) * (1.0 / GLOBAL_BATCH_SIZE)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics_dict)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(
            loss, tf.compat.v1.train.get_or_create_global_step()
        ),
    )


strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
config = tf.estimator.RunConfig(
    experimental_distribute=tf.contrib.distribute.DistributeConfig(
        train_distribute=strategy,
    ),
    protocol="grpc",
)

classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir="/tmp/multiworker", config=config
)
tf.estimator.train_and_evaluate(
    classifier,
    train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
    eval_spec=tf.estimator.EvalSpec(
        input_fn=input_fn, start_delay_secs=5, throttle_secs=10
    ),
)
