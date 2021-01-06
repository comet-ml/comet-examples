import os
import argparse
import json
import multiprocessing
import random
import portpicker
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers.experimental.preprocessing as kpl

os.environ["GRPC_FAIL_FAST"] = "use_caller"


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images[..., None]
test_images = test_images[..., None]

train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

BUFFER_SIZE = len(train_images)

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
    parser.add_argument("worker_index", type=int)
    parser.add_argument("run_id", type=int)
    parser.add_argument("ps_hosts", type=str)
    parser.add_argument("worker_hosts", type=str)
    parser.add_argument("task_type", type=str)

    return parser.parse_args()


def main(_):
    args = get_args()

    ps_hosts = args.ps_hosts.split(",")
    worker_hosts = args.worker_hosts.split(",")

    num_ps = len(ps_hosts)
    num_workers = len(worker_hosts)

    run_id = args.run_id
    worker_index = args.worker_index

    os.environ["TF_CONFIG"] = json.dumps(
        {
            "cluster": {"worker": worker_hosts, "ps": ps_hosts},
            "task": {"type": args.task_type, "index": worker_index},
        }
    )

    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    variable_partitioner = tf.distribute.experimental.partitioners.FixedShardsPartitioner(
        num_shards=num_ps
    )
    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver, variable_partitioner=variable_partitioner
    )

    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        .shuffle(BUFFER_SIZE)
        .batch(GLOBAL_BATCH_SIZE)
    )

    with strategy.scope():
        model = build_model()

        optimizer = keras.optimizers.RMSprop(learning_rate=0.1)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )

        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(
                per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE
            )

        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )

    def train_step(inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = compute_loss(labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_accuracy.update_state(labels, predictions)

        return loss

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )

    def dataset_fn():
        return train_dataset

    @tf.function
    def per_worker_dataset_fn():
        return strategy.distribute_datasets_from_function(dataset_fn)

    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)
    per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
    per_worker_iterator = iter(per_worker_dataset)

    num_epochs = args.num_epochs
    steps_per_epoch = len(train_images) / BATCH_SIZE_PER_REPLICA

    for i in range(num_epochs):
        train_accuracy.reset_states()

        total_loss = 0.0
        for _ in range(steps_per_epoch):
            loss = coordinator.schedule(
                distributed_train_step, args=(per_worker_iterator,)
            )
            total_loss += loss.fetch()

        # Wait at epoch boundaries.
        coordinator.join()
        print("Finished epoch %d, accuracy is %f." % (i, accuracy.result().numpy()))


if __name__ == "__main__":
    main()
