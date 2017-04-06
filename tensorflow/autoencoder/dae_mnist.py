import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#%matplotlib inline
print ("PACKAGES LOADED")

import os
import errno

# http://stackoverflow.com/a/5032238
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
			
make_sure_path_exists('nets')
print ("System Setup DONE")

mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data/', one_hot=True)
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels
print ("MNIST LOADED")


# NETOWRK PARAMETERS
n_input    = 784
n_hidden_1 = 256
n_hidden_2 = 256
n_output   = 784

# PLACEHOLDERS
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])
dropout_keep_prob = tf.placeholder("float")

# WEIGHTS
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# MODEL
def dae(_X, _weights, _biases, _keep_prob):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    layer_1out = tf.nn.dropout(layer_1, _keep_prob)
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1out, _weights['h2']), _biases['b2']))
    layer_2out = tf.nn.dropout(layer_2, _keep_prob)
    return tf.nn.sigmoid(tf.matmul(layer_2out, _weights['out']) + _biases['out'])

# MODEL AS A FUNCTION
recon = dae(x, weights, biases, dropout_keep_prob)
print ("NETOWRK READY")

# COST
cost = tf.reduce_mean(tf.pow(recon-y, 2))
# OPTIMIZER
optm = tf.train.AdamOptimizer(0.01).minimize(cost)
# INITIALIZER
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
print ("FUNCTIONS READY")

savedir = "nets/"
saver   = tf.train.Saver(max_to_keep=1)
print ("SAVER READY")

# TRAIN
TRAIN_FLAG = 1
epochs     = 100#50
batch_size = 100
disp_step  = 10

sess = tf.Session()
sess.run(init)
if TRAIN_FLAG:
    print ("START OPTIMIZATION")
    for epoch in range(epochs):
        print ("epoch = %d/%d" % (epoch, epochs) )
        num_batch  = int(mnist.train.num_examples/batch_size)
        total_cost = 0.
        for i in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs_noisy = batch_xs + 0.3*np.random.randn(batch_size, 784)
            feeds = {x: batch_xs_noisy, y: batch_xs, dropout_keep_prob: 1.}
            sess.run(optm, feed_dict=feeds)
            total_cost += sess.run(cost, feed_dict=feeds)
        # DISPLAY
        #if epoch % disp_step == 0:
        if epoch == epochs-1:
		
            print ("Epoch %02d/%02d average cost: %.6f"
                   % (epoch, epochs, total_cost/num_batch))
            # PLOT
            randidx  = np.random.randint(testimg.shape[0], size=1)
            print ("displaying idx = %d" % randidx);

            testvec  = testimg[randidx, :]
            noisyvec = testvec + 0.3*np.random.randn(1, 784)
            outvec   = sess.run(recon, feed_dict={x: testvec, dropout_keep_prob: 1.})
            outimg   = np.reshape(outvec, (28, 28))
			
            # Plot
            plt.matshow(np.reshape(testvec, (28, 28)), cmap=plt.get_cmap('gray'))
            plt.title("[" + str(epoch) + "] Original Image")
            plt.colorbar()
            plt.show()
            plt.matshow(np.reshape(noisyvec, (28, 28)), cmap=plt.get_cmap('gray'))
            plt.title("[" + str(epoch) + "] Input Image")
            plt.colorbar()
            plt.show()
            plt.matshow(outimg, cmap=plt.get_cmap('gray'))
            plt.title("[" + str(epoch) + "] Reconstructed Image")
            plt.colorbar()
            plt.show()
        # SAVE
        saver.save(sess, savedir + 'dae.ckpt', global_step=epoch)
print ("OPTIMIZATION FINISHED")
