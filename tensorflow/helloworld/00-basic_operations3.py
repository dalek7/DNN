# TensorFlow Basic Example
import numpy as np
import tensorflow as tf
sess = tf.Session()
x           = tf.placeholder("float", [1, 3])
w           = tf.Variable(tf.random_normal([3, 3]), name='w')
y           = tf.matmul(x, w)
relu_out    = tf.nn.relu(y)
#sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer())
print 'w = '
print sess.run(w)

print 'x = '
print x
print np.array([[1.0, 2.0, 3.0]])

print
print 'y ='
print sess.run(y, feed_dict={x:np.array([[1.0, 2.0, 3.0]])})
print
print 'relu_out ='
print sess.run(relu_out, feed_dict={x:np.array([[1.0, 2.0, 3.0]])})

writer = tf.summary.FileWriter('/tmp/tf_logs/relu1',sess.graph)


sess.close()


#$ tensorboard --logdir=/tmp/tf_logs/relu1 
