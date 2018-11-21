# coding: utf-8
'''
Example adapted from https://github.com/keras-team/keras/tree/master/examples
'''
'''Trains a Bidirectional LSTM on the IMDB sentiment classification task.
'''
# import comet_ml in the top of your file(before all other Machine learning libs)
from comet_ml import Experiment
import os
# Setting the API key (saved as environment variable)
exp = Experiment(
    #api_key="YOUR API KEY",
    # or
    api_key=os.environ.get("COMET_API_KEY"),
    project_name='comet-examples')

import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from keras.callbacks import EarlyStopping, ModelCheckpoint

max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
skip_top = 30  # top occuring words to skip
maxlen = 100
batch_size = 32
epochs = 1  # try 4
num_nodes = 8
dropout = 0.6

params = {"num_nodes": num_nodes,
          "model_type": "Bidirectional LSTM",
          "dropout": dropout,
          "auto_param_logging": True,
          "skip_top": skip_top,
          "maxlen": maxlen,
          "batch_size": batch_size,
          "max_features": max_features,
          "epochs": epochs
          }

# log params to Comet.ml
exp.log_multiple_params(params)

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features,
                                                      skip_top=skip_top,
                                                      seed=42)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(num_nodes)))
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print(model.summary())
print('Training\n...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=[x_test, y_test],
          callbacks=[EarlyStopping(
              monitor='val_loss', min_delta=1e-4, patience=3, verbose=1, mode='auto')]
          )
