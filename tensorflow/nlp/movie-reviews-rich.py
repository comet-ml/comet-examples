from __future__ import print_function
from utils import finalize_model
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from comet_ml import Experiment
                        
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds


# import matplotlib.pyplot as plt

def main():
    
    exp = Experiment(
        project_name="movie-reviews",
        auto_histogram_weight_logging=True
    )


    params = {
        'layer-1-size': 16,
        'epochs': 10,
        'batch-size':512,
        'dropout': 0.15,
    }
    
    exp.log_parameters(params)
    
    # Load data
    train_data, test_data = tfds.load(name="imdb_reviews", 
                                      split=["train", "test"], 
                                      batch_size=-1, 
                                      as_supervised=True)
    train_examples, train_labels = tfds.as_numpy(train_data)
    test_examples, test_labels = tfds.as_numpy(test_data)
    
    x_val = train_examples[:10000]
    partial_x_train = train_examples[10000:]
    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]
    
    # Load model
    model = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    hub_layer = hub.KerasLayer(model, 
                               output_shape=[20], 
                               input_shape=[], 
                               dtype=tf.string, 
                               trainable=True)
    hub_layer(train_examples[:3])
    
    # Build model
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(exp.get_parameter('layer-1-size'), activation='relu'))
    model.add(tf.keras.layers.Dropout(exp.get_parameter('dropout')))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam',
                  loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])

    # Train model
    model.fit(partial_x_train,
              partial_y_train,
              epochs=exp.get_parameter('epochs'),
              batch_size=exp.get_parameter('batch-size'),
              validation_data=(x_val, y_val),
              verbose=1)
    
    # log any custom metric
    
    exp.log_metric('custom_metric', 0.98)
    
    # log a dataset hash
    exp.log_dataset_hash(partial_x_train)

    # finalize_model invokes:
    #     * exp.log_confusion_matrix()
    #     * exp.log_text()
    #     * exp.log_model()
    finalize_model(model, test_examples, test_labels, exp)
    


if __name__ == "__main__":
    main()