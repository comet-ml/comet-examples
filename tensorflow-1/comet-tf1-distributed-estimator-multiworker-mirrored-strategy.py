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

import argparse
import hashlib
import numpy as np
import os, json

from hooks import CometSessionHook

PROJECT_NAME = 'tf-estimator-multiworker'
BUFFER_SIZE = 60000
BATCH_SIZE = 8

LEARNING_RATE = 1e-5
EPOCHS = 2

tf.logging.set_verbosity(tf.logging.INFO)

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images[..., None]
test_images = test_images[..., None]

train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

train_labels = train_labels.astype("int64")
test_labels = test_labels.astype("int64")


def input_fn(mode, input_context=None):
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    dataset = train_dataset if mode == tf.estimator.ModeKeys.TRAIN else test_dataset

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    if input_context:
        dataset = dataset.shard(
            input_context.num_input_pipelines, input_context.input_pipeline_id
        )
    return (
        dataset.map(scale)
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .repeat(EPOCHS)
    )


def model_fn(features, labels, mode, params):
    global_batch_size = BATCH_SIZE * params["n_workers"]
  
    experiment = get_experiment(params["run_id"], exists=True)
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

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"logits": logits}
        return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels, reduction=tf.losses.Reduction.NONE
    )
    loss = tf.reduce_sum(cross_entropy) * (1.0 / (global_batch_size))
    comet_hook = CometSessionHook(
        experiment,
        tensors={f"loss:{params['task_type']}/{params['task_index']}" : loss},
        parameters=None,
        every_n_iter=1
    )
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, evaluation_hooks=[comet_hook])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        training_hooks=[comet_hook],
        train_op=optimizer.minimize(
            loss, tf.compat.v1.train.get_or_create_global_step()
        ),
    )


def get_experiment(run_id, exists=False):
    experiment_id = hashlib.md5(run_id.encode('utf-8')).hexdigest()    
    os.environ['COMET_EXPERIMENT_KEY'] = experiment_id
    
    if exists:
        return comet_ml.ExistingExperiment(project_name=PROJECT_NAME)

    return comet_ml.Experiment(project_name=PROJECT_NAME)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id")
    parser.add_argument("--task_type")
    parser.add_argument("--task_index", type=int)
    parser.add_argument("--chief_host")
    parser.add_argument("--worker_hosts")
    parser.add_argument("--eval_hosts")

    return parser.parse_args()

args = get_args()

worker_hosts = args.worker_hosts.split(",")
eval_hosts = args.eval_hosts.split(",")
n_workers = len(worker_hosts)

cluster_dict = {
    "cluster": {"chief": [args.chief_host], "worker": worker_hosts, "evaluator": eval_hosts},
    "task": {"type": args.task_type, "index": args.task_index},
}
os.environ["TF_CONFIG"] = json.dumps(cluster_dict)

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
config = tf.estimator.RunConfig(
    experimental_distribute=tf.contrib.distribute.DistributeConfig(
        train_distribute=strategy,
    ),
    protocol="grpc",
)

if args.task_type == "chief":
    experiment = get_experiment(args.run_id)
    experiment.log_code("./hooks.py")

else:
    experiment = get_experiment(args.run_id, exists=True)

classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir="/tmp/multiworker", config=config, params={
        "task_index": args.task_index, 
        "n_workers": n_workers, 
        "task_type": args.task_type,
        "task_index": args.task_index, 
        "run_id": args.run_id
        } 
)
tf.estimator.train_and_evaluate(
    classifier,
    train_spec=tf.estimator.TrainSpec(input_fn=input_fn, max_steps=5000),
    eval_spec=tf.estimator.EvalSpec(input_fn=input_fn, throttle_secs=10, start_delay_secs=10)
)
