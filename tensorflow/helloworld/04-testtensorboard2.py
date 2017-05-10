import tensorflow as tf

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

add = tf.add(X, Y)
mul = tf.multiply(X, Y)
squared1 = tf.multiply(X, X)

# step 1
add_hist    = tf.summary.scalar("add_scalar", add)
mul_hist    = tf.summary.scalar("mul_scalar", mul)
sq_hist     = tf.summary.scalar("square_scalar", squared1)

# step 2
merged =tf.summary.merge_all()


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # step 3
    writer = tf.summary.FileWriter("/tmp/tensorboard/sample_2", sess.graph)

    for step in range(100):
        # step 4
        summary = sess.run(merged, feed_dict={X: step * 1.0, Y: 2.0})
        writer.add_summary(summary, step)

# step 5
# tensorboard --logdir=/tmp/tensorboard/sample_2
