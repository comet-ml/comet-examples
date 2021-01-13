"""
Script to run distributed data parallel training in Tensorflow using ParameterServerStrategy

"""
import comet_ml

import os
import argparse
import json
import numpy as np
import tensorflow as tf

print(tf.__version__)
PROJECT_NAME = "tf1-distributed"

os.environ["GRPC_FAIL_FAST"] = "use_caller"

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images[..., None]
test_images = test_images[..., None]

train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

train_labels = train_labels.astype("int64")
test_labels = test_labels.astype("int64")

PS_HOSTS = ["localhost:8000"]
WORKER_HOSTS = ["localhost:8001", "localhost:8002"]

BUFFER_SIZE = len(train_images)

EPOCHS = 10
STEPS_PER_EPOCH = 100
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
    parser.add_argument("--task_index", type=int)
    parser.add_argument("--task_type", type=str)

    return parser.parse_args()


def main():
    args = get_args()

    run_id = args.run_id
    task_index = args.task_index

    cluster_dict = {
        "cluster": {"worker": WORKER_HOSTS, "ps": PS_HOSTS},
        "task": {"type": args.task_type, "index": task_index},
    }
    os.environ["TF_CONFIG"] = json.dumps(cluster_dict)

    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)

    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    server = tf.distribute.Server(
        cluster_resolver.cluster_spec(),
        job_name=cluster_resolver.task_type,
        task_index=cluster_resolver.task_id,
        protocol=cluster_resolver.rpc_layer or "grpc",
        start=True,
    )
    if cluster_resolver.task_type == "ps":
        server.join()

    experiment = comet_ml.Experiment(log_code=True, project_name=PROJECT_NAME)
    experiment.log_other("run_id", run_id)

    with strategy.scope():
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            .shuffle(BUFFER_SIZE)
            .batch(GLOBAL_BATCH_SIZE)
            .repeat()
        )
        train_ds = strategy.experimental_distribute_dataset(train_dataset)

        model = build_model()
        experiment.set_model_graph(tf.compat.v1.get_default_graph())

        optimizer = tf.train.GradientDescentOptimizer(0.001)

        def train_step(dist_inputs):
            def step_fn(inputs):
                images, labels = inputs
                logits = model(images)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels
                )
                loss = tf.reduce_sum(cross_entropy) * (1.0 / GLOBAL_BATCH_SIZE)
                train_op = optimizer.minimize(loss)
                with tf.control_dependencies([train_op]):
                    return tf.identity(loss)

            per_replica_losses = strategy.experimental_run_v2(
                step_fn, args=(dist_inputs,)
            )
            mean_loss = strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
            )
            return mean_loss

        train_iterator = train_ds.make_initializable_iterator()
        iterator_init = train_iterator.initializer
        var_init = tf.global_variables_initializer()
        loss = train_step(next(train_iterator))

        with tf.train.MonitoredTrainingSession(server.target) as sess:
            sess.run([var_init])
            for epoch in range(EPOCHS):
                sess.run([iterator_init])
                for step in range(STEPS_PER_EPOCH):
                    current_loss = sess.run(loss)
                    if step % 10 == 0:
                        print(
                            "Epoch {} Step {} Loss {:.4f}".format(
                                epoch, step, current_loss
                            )
                        )
                        experiment.log_metric(
                            "loss", current_loss, step=(STEPS_PER_EPOCH * epoch) + step
                        )


if __name__ == "__main__":
    main()
