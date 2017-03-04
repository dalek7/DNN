import tensorflow as tf
sess = tf.Session()
x = tf.placeholder("float", [1, 3])
w = tf.Variable(tf.random_normal([3, 3]), name='w')
y = tf.matmul(x, w)
relu_out = tf.nn.relu(y)
print x
print w
print relu_out

writer = tf.summary.FileWriter('/tmp/tf_logs/relu',sess.graph)


#$ tensorboard --logdir=/tmp/tf_logs/relu
