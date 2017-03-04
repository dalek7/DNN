# https://www.tensorflow.org/get_started/mnist/beginners
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Load data
X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels

print X_train.size /28/28, Y_train.size

# Get the next 64 images array and labels
batch_X, batch_Y = mnist.train.next_batch(64)
print batch_X.size, batch_Y.size
print batch_X.size/64, batch_Y.size/64
print batch_X.size/28/28, batch_Y.size
