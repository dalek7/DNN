import tensorflow as tf
import numpy as np

'''
# tf.unstack
Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
  For example, given a tensor of shape `(A, B, C, D)`;

  If `axis == 0` then the i'th tensor in `output` is the slice `value[i, :, :, :]`
    and each tensor in `output` will have shape `(B, C, D)`. (Note that the
    dimension unpacked along is gone, unlike `split`).

  If `axis == 1` then the i'th tensor in `output` is the slice `value[:, i, :, :]`
    and each tensor in `output` will have shape `(A, C, D)`.
  Etc.
  
'''

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps

x = tf.placeholder("float", [None, n_steps, n_input])
print(x.shape) # (?, 28, 28)

# (X, axis=1, num=num_units)
x2 = tf.unstack(x, n_steps, 1)
# x3 = tf.unstack(x, n_steps, 0) # Input shape axis 0 must equal 28, got shape [1,28,28]
x3 = tf.unstack(x, n_steps, 2)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_in = mnist.test.images[9:10] #  (1, 784)
    print("x_in.shape : ", np.array(x_in).shape)

    x_in2 = x_in.reshape((1, n_steps, n_input)) # (1, 28, 28)

    print("x_in.shape (reshaped): ", np.array(x_in2).shape)

    # 지정한 축을 기준으로 다시 구성
    # 따라서 아래 두개는 같은 값을 가질 수 밖에
    x2_ = sess.run(x2, feed_dict={x: x_in2})
    x3_ = sess.run(x3, feed_dict={x: x_in2})

    print(np.array(x2_).shape) # (28, 1, 28)
    print(np.array(x3_).shape) # (28, 1, 28)