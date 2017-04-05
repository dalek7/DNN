# based on https://www.tensorflow.org/get_started/mnist/beginners
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

# Load data
X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels

print( 'mnist.train.images.size = %d' % X_train.size)
print( 'mnist.train.labels.size = %d' % Y_train.size)

print X_train.size /28/28, Y_train.size

# Get the next 64 images array and labels
batch_X, batch_Y = mnist.train.next_batch(64)
#print batch_X.size, batch_Y.size
print ('batch_X.size/28/28= %d, batch_Y.size/10= %d' % (batch_X.size/28/28, batch_Y.size/10))


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for iter in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  #print('iter %d' % iter)
  if iter % 100 == 0:
      print('out/w_%s.txt' % iter)
      f = open('out/w_%s.txt' % iter, 'w')
      W1 = sess.run(W)
      #print W1.size / 10
      for i in range(W1.size/10): # inspect w for only '0'
        f.write("\t".join(map(lambda x: str(x), W1[i])))
      f.close()

''' This takes too much time !
      for i in range(784):
          Wi = sess.run (W[i])
          #print "\t".join(map(lambda x: str(x), Wi))
          f.write("\t".join(map(lambda x: str(x), Wi)))
          f.write('\n')
'''


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('accuracy=')
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
