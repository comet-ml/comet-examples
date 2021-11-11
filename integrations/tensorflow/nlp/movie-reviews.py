from __future__ import print_function
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from comet_ml import Experiment

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

def main():
    
    exp = Experiment(project_name="movie-reviews")
    
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
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam',
                  loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])

    # Train model
    model.fit(partial_x_train,
              partial_y_train,
              epochs=25,
              batch_size=512,
              validation_data=(x_val, y_val),
              verbose=1)

if __name__ == "__main__":
    main()