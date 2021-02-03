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

import os
import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


os.environ["GRPC_FAIL_FAST"] = "use_caller"

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images[..., None]
test_images = test_images[..., None]

train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

BUFFER_SIZE = len(train_images)

EPOCHS = 10
BATCH_SIZE_PER_REPLICA = 64


def build_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )

    return model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int)
    parser.add_argument("--worker_hosts")
    parser.add_argument("--ps_hosts")
    parser.add_argument("--task_index", type=int)
    parser.add_argument("--task_type", type=str)

    return parser.parse_args()


def main():
    args = get_args()

    run_id = args.run_id
    task_index = args.task_index

    ps_hosts = args.ps_hosts.split(",")
    worker_hosts = args.worker_hosts.split(",")

    cluster_dict = {
        "cluster": {"worker": worker_hosts, "ps": ps_hosts},
        "task": {"type": args.task_type, "index": task_index},
    }
    os.environ["TF_CONFIG"] = json.dumps(cluster_dict)

    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    if cluster_resolver.task_type in ("worker", "ps"):
        server = tf.distribute.Server(
            cluster_resolver.cluster_spec(),
            job_name=cluster_resolver.task_type,
            task_index=cluster_resolver.task_id,
            protocol=cluster_resolver.rpc_layer or "grpc",
            start=True,
        )
        server.join()

    experiment = comet_ml.Experiment(log_code=True)
    experiment.log_other("run_id", run_id)

    variable_partitioner = tf.distribute.experimental.partitioners.FixedShardsPartitioner(
        num_shards=len(ps_hosts)
    )

    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver, variable_partitioner=variable_partitioner
    )
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    with strategy.scope():
        model = build_model()

        optimizer = keras.optimizers.RMSprop(learning_rate=0.1)
        criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )

        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )

    @tf.function
    def step_fn(dataset_inputs):
        def train_step(inputs):
            images, labels = next(inputs)
            with tf.GradientTape() as tape:
                pred = model(images, training=True)
                per_example_loss = criterion(labels, pred)
                loss = tf.nn.compute_average_loss(per_example_loss)

                gradients = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
            train_accuracy.update_state(labels, actual_pred)

            return loss

        losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

    def dataset_fn(_):
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            .shuffle(BUFFER_SIZE)
            .batch(GLOBAL_BATCH_SIZE)
            .repeat()
        )

        return train_dataset

    @tf.function
    def per_worker_dataset_fn():
        return strategy.distribute_datasets_from_function(dataset_fn)

    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)
    per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
    per_worker_iterator = iter(per_worker_dataset)

    steps_per_epoch = len(train_images) // BATCH_SIZE_PER_REPLICA

    for i in range(EPOCHS):
        train_accuracy.reset_states()

        total_loss = 0
        for _ in range(steps_per_epoch):
            loss = coordinator.schedule(step_fn, args=(per_worker_iterator,))
            total_loss += loss.fetch()

        experiment.log_metric("loss", total_loss, epoch=i)

        # Wait at epoch boundaries.
        coordinator.join()
        print(
            "Finished epoch %d, accuracy is %f." % (i, train_accuracy.result().numpy())
        )


if __name__ == "__main__":
    main()
