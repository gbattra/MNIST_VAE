import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

n_pixels = 28*28
X = tf.placeholder(tf.float32, shape=([None, n_pixels]))


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def FC_layer(X, W, b):
    return tf.matmul(X, W) + b


latent_dim = 20
h_dim = 500

# ENCODER
# layer 1
W_enc = weight_variable([n_pixels, h_dim], 'W_enc')
b_enc = bias_variable([h_dim], 'b_enc')
h_enc = tf.nn.tanh(FC_layer(X, W_enc, b_enc))

# layer 2
# mean
W_mu = weight_variable([h_dim, latent_dim], 'W_mu')
b_mu = bias_variable([latent_dim], 'b_mu')
mu = FC_layer(h_enc, W_mu, b_mu)
# stdev
W_logstd = weight_variable([h_dim, latent_dim], 'W_logstd')
b_logstd = bias_variable([latent_dim], 'b_logstd')
logstd = FC_layer(h_enc, W_logstd, b_logstd)

noise = tf.random_normal([1, latent_dim])

z = mu + tf.multiply(noise, tf.exp(0.5*logstd))

# DECODER
# layer 1
W_dec = weight_variable([latent_dim, h_dim], 'W_dec')
b_dec = bias_variable([h_dim], 'b_dec')
h_dec = tf.nn.tanh(FC_layer(z, W_dec, b_dec))

# layer 2
W_rcnst = weight_variable([h_dim, n_pixels], 'W_rcnst')
b_rcnst = bias_variable([n_pixels], 'b_rcnst')
rcnst = tf.nn.sigmoid(FC_layer(h_dec, W_rcnst, b_rcnst))

# LOSS FUNCTION
log_likelihood = tf.reduce_sum(X * tf.log(rcnst + 1e-9) + (1 - X) * tf.log(1 - rcnst + 1e-9), reduction_indices=1)
KL_term = -0.5 * tf.reduce_sum(1 + 2 * logstd - tf.pow(mu, 2) - tf.exp(2 * logstd), reduction_indices=1)
var_low_bound = tf.reduce_mean(log_likelihood - KL_term)
optimizer = tf.train.AdadeltaOptimizer().minimize(-var_low_bound)

# start session and train
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

saver = tf.train.Saver()
num_iters = 600000
rec_interval = 1000
var_low_bounds = []
log_likelihoods = []
KL_terms = []

# train
for i in range(num_iters):
    x_batch = np.round(mnist.train.next_batch(200)[0])
    sess.run(optimizer, feed_dict={X: x_batch})
    if (i % rec_interval == 0):
        vlb = var_low_bound.eval(feed_dict={X: x_batch})
        print('Iter: {}, Loss: {}'.format(i, vlb))
        var_low_bounds.append(vlb)
        log_likelihoods.append(np.mean(log_likelihood.eval(feed_dict={X: x_batch})))
        KL_terms.append(np.mean(KL_term.eval(feed_dict={X: x_batch})))

num_pairs = 10
image_indices = np.random.randint(0, 200, num_pairs)
for pair in range(num_pairs):
    x = np.reshape(mnist.test.images[image_indices[pair]], (1,  n_pixels))
    plt.figure()
    x_image = np.reshape(x, (28, 28))
    plt.subplot(121)
    plt.imshow(x_image)
    x_rcnst = rcnst.eval(feed_dict={X: x})
    x_rcnst_image = (np.reshape(x_rcnst, (28, 28)))
    plt.subplot(122)
    plt.imshow(x_rcnst_image)
    plt.show()
