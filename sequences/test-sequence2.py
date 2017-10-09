
# https://youtu.be/39_P23TqUnw?t=11m41s
import tensorflow as tf
tf.set_random_seed(777)  # reproducibility

y_data      = tf.constant([[1,1,1]])

prediction1  = tf.constant([[[0.3,0.7],[0.3,0.7],[0.3,0.7]]], dtype=tf.float32)
prediction2  = tf.constant([[[0.1,0.9],[0.1,0.9],[0.1,0.9]]], dtype=tf.float32)


weight = tf.constant([[1,1,1]], dtype=tf.float32)

sequence_loss1 = tf.contrib.seq2seq.sequence_loss(logits=prediction1, targets=y_data, weights = weight)
sequence_loss2 = tf.contrib.seq2seq.sequence_loss(logits=prediction2, targets=y_data, weights = weight)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Loss1:", sequence_loss1.eval())
    print("Loss2:", sequence_loss2.eval())

'''
    ('Loss1:', 0.51301527)
    ('Loss2:', 0.37110069)
'''
