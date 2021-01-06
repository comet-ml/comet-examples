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


def feature_and_label_gen(num_examples=200):
    examples = {"features": [], "label": []}
    for _ in range(num_examples):
        features = random.sample(feature_vocab, 3)
        label = ["yes"] if "avenger" in features else ["no"]
        examples["features"].append(features)
        examples["label"].append(label)
    return examples


def dataset_fn(_):
    examples = feature_and_label_gen()
    raw_dataset = tf.data.Dataset.from_tensor_slices(examples)

    train_dataset = (
        raw_dataset.map(
            lambda x: (
                {"features": feature_preprocess_stage(x["features"])},
                label_preprocess_stage(x["label"]),
            )
        )
        .shuffle(200)
        .batch(32)
        .repeat()
    )
    return train_dataset


def build_model():
    # Create the model. The input needs to be compatible with KPLs.
    model_input = keras.layers.Input(shape=(3,), dtype=tf.int64, name="model_input")

    emb_layer = keras.layers.Embedding(
        input_dim=len(feature_lookup_layer.get_vocabulary()), output_dim=20
    )
    emb_output = tf.reduce_mean(emb_layer(model_input), axis=1)
    dense_output = keras.layers.Dense(units=1, activation="sigmoid")(emb_output)
    model = keras.Model({"features": model_input}, dense_output)

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

    with strategy.scope():
        model = build_model()

        optimizer = keras.optimizers.RMSprop(learning_rate=0.1)
        accuracy = keras.metrics.Accuracy()

    @tf.function
    def step_fn(iterator):
        # Experiment Object here to record the worker metrics
        def replica_fn(iterator):
            batch_data, labels = next(iterator)
            with tf.GradientTape() as tape:
                pred = model(batch_data, training=True)
                per_example_loss = keras.losses.BinaryCrossentropy(
                    reduction=tf.keras.losses.Reduction.NONE
                )(labels, pred)
                loss = tf.nn.compute_average_loss(per_example_loss)
                gradients = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
            accuracy.update_state(labels, actual_pred)

            return loss

        losses = strategy.run(replica_fn, args=(iterator,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

    @tf.function
    def per_worker_dataset_fn():
        return strategy.distribute_datasets_from_function(dataset_fn)

    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)
    per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
    per_worker_iterator = iter(per_worker_dataset)

    num_epochs = args.num_epochs
    steps_per_epoch = args.steps_per_epoch
    for i in range(num_epochs):
        accuracy.reset_states()
        for _ in range(steps_per_epoch):
            coordinator.schedule(step_fn, args=(per_worker_iterator,))

        # Wait at epoch boundaries.
        coordinator.join()
        print("Finished epoch %d, accuracy is %f." % (i, accuracy.result().numpy()))


if __name__ == "__main__":
    main()
