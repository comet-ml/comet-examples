# coding: utf-8
# import comet_ml in the top of your file(before all other Machine learning libs)
'''Example adapted from https://github.com/keras-team/keras/tree/master/examples'''
from comet_ml import Experiment

import os
# Setting the API key (saved as environment variable)
exp = Experiment(
    #api_key="YOUR API KEY",
    # or
    api_key=os.environ.get("COMET_API_KEY"),
    project_name='comet-examples')


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb

# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 2

params = {"layer1_kernel_size": kernel_size,
          "max_features": max_features,
          "maxlen": maxlen,
          "embedding_size": embedding_size,
          "layer1_filters": filters,
          "dropout": 0.25,
          "layer1": "Conv1D",
          "layer2": "LSTM",
          "layer2_nodes": lstm_output_size,
          "layer1_pool_size": pool_size,
          "epochs": epochs,
          "batch_size": batch_size
          }
# log params to Comet.ml
exp.log_multiple_params(params)

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features, skip_top=50)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# always a good idea to print summary so it gets logged to the output tab on your experiment
print(model.summary())

print('Training...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
