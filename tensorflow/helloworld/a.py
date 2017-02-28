import tensorflow as tf
print tf.__version__

state = tf.Variable(0, name="counter")
print state

one = tf.constant(2)
#print one
new_value = tf.add(state, one)
#print new_value

update = tf.assign(state, new_value)

#init_op = tf.initialize_all_variables() # deprecated
init_op = tf.global_variables_initializer() # Returns an Op that initializes global variables.
with tf.Session() as sess:
  sess.run(init_op)
  print(sess.run(state))
  print "========="

  for _ in range(5):
    sess.run(update)
    print(sess.run(state))
