# based on https://www.tensorflow.org/get_started/mnist/pros

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

import tensorflow as tf
#sess = tf.InteractiveSession()


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Start tf session
sess = tf.Session()

# First Convolutional Layer
## first layer will consist of convolution, followed by max pooling
## The convolution will compute 32 features for each 5x5 patch.
## Its weight tensor will have a shape of [5, 5, 1, 32].
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

sess.run(tf.global_variables_initializer())

print ('b_conv1 = ')
print(b_conv1)
print sess.run(b_conv1)
print



