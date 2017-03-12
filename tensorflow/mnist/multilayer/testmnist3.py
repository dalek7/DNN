from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

# Placeholders
## The input images x will consist of a 2d tensor of floating point numbers.
x = tf.placeholder(tf.float32, shape=[None, 784])

## The target output classes y_ will also consist of a 2d tensor,
## where each row is a one-hot 10-dimensional vector indicating which digit class (zero through nine) the corresponding MNIST image belongs to.
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
## Before Variables can be used within a session, they must be initialized using that session.
sess.run(tf.global_variables_initializer())

# Predicted Class and Loss Function
y = tf.matmul(x,W) + b

## tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the model's unnormalized model prediction and sums across all classes
## tf.reduce_mean takes the average over these sums.
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Train the Model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})


# Evaluate the Model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
