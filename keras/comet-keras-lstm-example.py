# coding: utf-8
'''Trains an LSTM model on the IMDB sentiment classification task.
Example adapted from https://github.com/keras-team/keras/tree/master/examples
'''
# import comet_ml in the top of your file(before all other Machine learning libs)
from comet_ml import Experiment

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os
# Setting the API key (saved as environment variable)
exp = Experiment(
    #api_key="YOUR API KEY",
    # or
    api_key=os.environ.get("COMET_API_KEY"),
    project_name='comet-examples')

params = {"num_nodes": 128,
          "model_type": "LSTM",
          "dropout": 0.4,
          "dropout_recurrent": 0.4,
          "num_words": 20000,
          "maxlen": 90,  # cut texts after this number of words
          "skip_top": 10,
          "batch_size": 32,
          "epochs": 15
          }
exp.log_multiple_params(params)


print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=params["num_words"],
                                                      skip_top=params["skip_top"],
                                                      seed=42
                                                      )
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=params["maxlen"])
x_test = sequence.pad_sequences(x_test, maxlen=params["maxlen"])
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')


model = Sequential()
model.add(Embedding(params["num_words"], 128))
model.add(LSTM(params["num_nodes"],
               dropout=params['dropout'],
               recurrent_dropout=params['dropout_recurrent']))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
print('Training...')
model.fit(x_train, y_train,
          batch_size=params["batch_size"],
          epochs=params["epochs"],
          validation_data=(x_test, y_test),
          callbacks=[EarlyStopping(monitor='loss', min_delta=1e-3, patience=2, verbose=1, mode='auto')])
score, acc = model.evaluate(x_test, y_test,
                            batch_size=params["batch_size"])
model.save('imdb_lstm_final.h5')
print('Test score:', score)
print('Test accuracy:', acc)
