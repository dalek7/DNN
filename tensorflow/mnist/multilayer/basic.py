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


# Convolution and Pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Start tf session
sess = tf.Session()


# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# First Convolutional Layer
## first layer will consist of convolution, followed by max pooling
## The convolution will compute 32 features for each 5x5 patch.
## Its weight tensor will have a shape of [5, 5, 1, 32].
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


## reshape x to a 4d tensor
x_image = tf.reshape(x, [-1,28,28,1])
print("x_image = ", x_image) # shape=(?, 28, 28, 1)

## We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
## The max_pool_2x2 method will reduce the image size to 14x14.
h_pool1 = max_pool_2x2(h_conv1)
print('h_conv1 shape = ')
print( h_conv1.get_shape())  # 28x28x32
print('h_pool1 shape = ')
print(h_pool1.get_shape()) # 14x14x32
print


# Second Convolutional Layer
## The second layer will have 64 features for each 5x5 patch.

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) ## the image size will be reduced to 7x7

print 5*5*32*64
print('W_conv2 shape = ')
print( W_conv2.get_shape())
print('b_conv2 shape = ')
print( b_conv2.get_shape())

print('h_conv2 shape = ')
print(h_conv2.get_shape()) # 14x14x32
print('h_pool2 shape = ')
print(h_pool2.get_shape()) # 14x14x32
print

sess.run(tf.global_variables_initializer())

print ('b_conv1 = ')
print(b_conv1)
print sess.run(b_conv1)
print



