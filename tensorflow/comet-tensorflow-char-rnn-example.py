# coding: utf-8

"""
Example adapted from Vanilla Char-RNN using TensorFlow by Vinh Khuc (@knvinh) https://gist.github.com/vinhkhuc/7ec5bf797308279dc587.
Adapted from Karpathy's min-char-rnn.py
https://gist.github.com/karpathy/d4dee566867f8291f086
Requires tensorflow>=1.0
BSD License
"""
# import comet_ml at the top of your file
from comet_ml import Experiment

# create an experiment with your api key
import os
# Setting the API key (saved as environment variable)
experiment = Experiment(
    #api_key="YOUR API KEY",
    # or
    api_key=os.environ.get("COMET_API_KEY"),
    project_name='comet-examples')

import random
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops
import os

import re

seed_value = 42
tf.set_random_seed(seed_value)
random.seed(seed_value)

# Generative Model using Dracula by Bram Stoker

book_path = os.path.join('data', 'dracula.txt')

with open(book_path, encoding='utf_8') as f:
    my_text = f.read()  # will read until it reaches an error

print('Length of data is {}, type of data is {}\n'.format(
    len(my_text), type(my_text)))


def simple_clean(text):
    text = text.lower()
    text = re.sub(r'[\r+\n+\*"]', ' ', text)
    text = re.sub(r'[\(__\)]', ' ', text)  # parentheticals
    text = re.sub(r'([0-9]+:[0-9]+)', '#', text)  # time in o'clock
    text = re.sub(r'[0-9]+', '#', text)  # all numbers
    text = re.sub(r'[pa]\.\s*m\.', ' ', text)  # a.m. or p.m.
    text = re.sub(r'-+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)  # renormalize spaces
    return text


data = simple_clean(my_text)

# compile index

chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print('Data has %d characters, %d unique.\n' % (data_size, vocab_size))


char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

print(ix_to_char)
print(char_to_ix)

# let's now try to delete some of these special chars from the text and reindex
special = {3, 4, 5, 6, 9, 12, 13, 15, 42, 43, 44}


def more_clean(dictionary, keys, text):
    new_text = text[:]
    for key in keys:
        new_text = re.sub(re.escape(dictionary.get(key)), " ", new_text)
    new_text = re.sub(r'\s+', " ", new_text)
    return new_text


data = more_clean(ix_to_char, special, data)

chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print('Data has %d characters, %d unique.\n' % (data_size, vocab_size))

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

tf.reset_default_graph()

# Hyper-parameters, initialization
hidden_size = 300  # number of hidden neurons
seq_length = 45   # number of steps to unroll


# log your  parameters to Comet.ml!
params = {"len(chars)": len(chars),
          "text": "Dracula",
          "hidden_size": hidden_size,
          "seq_length": seq_length
          }

experiment.log_multiple_params(params)

inputs = tf.placeholder(shape=[None, vocab_size],
                        dtype=tf.float32, name="inputs")
targets = tf.placeholder(
    shape=[None, vocab_size], dtype=tf.float32, name="targets")
init_state = tf.placeholder(
    shape=[1, hidden_size], dtype=tf.float32, name="state")

initializer = tf.random_normal_initializer(stddev=0.1)
bias_initializer = tf.random_normal_initializer(mean=0.5, stddev=0.5)

# Model building preliminaries
# RNN built out explicitly, w/o using predefined TF functions!!
with tf.variable_scope("RNN") as scope:
    hs_t = init_state
    ys = []
    for t, xs_t in enumerate(tf.split(inputs, seq_length, axis=0)):
        if t > 0:
            scope.reuse_variables()  # Reuse variables
        Wxh = tf.get_variable(
            "Wxh", [vocab_size, hidden_size], initializer=initializer)
        Whh = tf.get_variable(
            "Whh", [hidden_size, hidden_size], initializer=initializer)
        Why = tf.get_variable(
            "Why", [hidden_size, vocab_size], initializer=initializer)
        bh = tf.get_variable("bh", [hidden_size], initializer=bias_initializer)
        by = tf.get_variable("by", [vocab_size], initializer=bias_initializer)

        hs_t = tf.tanh(tf.matmul(xs_t, Wxh) + tf.matmul(hs_t, Whh) + bh)
        ys_t = tf.matmul(hs_t, Why) + by
        ys.append(ys_t)

hprev = hs_t
output_softmax = tf.nn.softmax(ys[-1])  # Get softmax for sampling

outputs = tf.concat(ys, axis=0)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=targets, logits=outputs))

# Minimizer
minimizer = tf.train.AdamOptimizer(epsilon=0.1)

grads_and_vars = minimizer.compute_gradients(loss)

# Gradient clipping
grad_clipping = tf.constant(5.0, name="grad_clipping")
clipped_grads_and_vars = []
for grad, var in grads_and_vars:
    clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
    clipped_grads_and_vars.append((clipped_grad, var))

# Gradient updates
updates = minimizer.apply_gradients(clipped_grads_and_vars)


def one_hot(v):
    return np.eye(vocab_size)[v]


# begin training
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# log model graph
experiment.set_model_graph(sess.graph)
# Initial values
MAXITERS = 500000
n, p = 0, 0
hprev_val = np.zeros([1, hidden_size])

while (n < MAXITERS):
    # Initialize
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev_val = np.zeros([1, hidden_size])
        p = 0  # reset

    # Prepare inputs
    input_vals = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    target_vals = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

    input_vals = one_hot(input_vals)
    target_vals = one_hot(target_vals)

    hprev_val, loss_val, _ = sess.run([hprev, loss, updates],
                                      feed_dict={inputs: input_vals,
                                                 targets: target_vals,
                                                 init_state: hprev_val})
    # log the loss to Comet.ml
    experiment.log_metric("loss", loss_val, step=n)

    if n % 500 == 0:
        # Log Progress

        print('iter: %d, p: %d, loss: %f' % (n, p, loss_val))

        # Do sampling
        sample_length = 200
        start_ix = random.randint(0, len(data) - seq_length)
        sample_seq_ix = [char_to_ix[ch]
                         for ch in data[start_ix:start_ix + seq_length]]
        ixes = []
        sample_prev_state_val = np.copy(hprev_val)

        for t in range(sample_length):
            sample_input_vals = one_hot(sample_seq_ix)
            sample_output_softmax_val, sample_prev_state_val = sess.run([output_softmax, hprev],
                                                                        feed_dict={inputs: sample_input_vals, init_state: sample_prev_state_val})

            ix = np.random.choice(
                range(vocab_size), p=sample_output_softmax_val.ravel())
            ixes.append(ix)
            sample_seq_ix = sample_seq_ix[1:] + [ix]

        txt = ''.join(ix_to_char[ix] for ix in ixes)
        print('----\n %s \n----\n' % (txt,))

    p += seq_length
    n += 1
