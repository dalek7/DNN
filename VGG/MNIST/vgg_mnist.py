
# coding: utf-8

# # Deep Convolutional MNIST Classifier using Tensorflow

# This is a convolutional nerual network with 11 conv-layer and 3 fc-layer base on VGG19, I removed 8 conv-layer and 3 pooling layer to make sure all datas fits in my GPU memory. The project is based on github project tensorflow-vgg19 [https://github.com/machrisaa/tensorflow-vgg]

# ### Initialization
# MNIST dataset is included in Tensorflow as an example dataset. I choise MNIST because it's relatively smaller and spending less time to train. It's good for practicing CNN and Tensorflow

# In[1]:


#get_ipython().magic(u'load_ext autoreload')
#get_ipython().magic(u'autoreload 2')
#get_ipython().magic(u'matplotlib inline')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import utils
import vgg19_trainable2 as vgg19
from tensorflow.examples.tutorials.mnist import input_data
import time
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[2]:


sess = tf.Session()
batch_size = 300

images = tf.placeholder(tf.float32, [None, 28, 28, 1])
true_out = tf.placeholder(tf.float32, [None, 10])
train_mode = tf.placeholder(tf.bool)

vgg = vgg19.Vgg19()
vgg.build(images, train_mode)

print(vgg.get_var_count())


# ### MNIST dataset
# The MNIST dataset is a hand writen digit classification dataset.

# In[3]:


# test classification
sess.run(tf.global_variables_initializer())

batch = mnist.train.next_batch(batch_size)
batch_img = batch[0].reshape((-1,28,28,1))
batch_lbl = batch[1]

print(batch_img.shape, batch_lbl.shape)

print (np.argmax(batch_lbl[0]))
print (np.argmax(batch_lbl[1]))
print (np.argmax(batch_lbl[2]))
print (np.argmax(batch_lbl[3]))


plt.figure()
plt.imshow(batch_img[0,:,:,0])
plt.figure()
plt.imshow(batch_img[1,:,:,0])
plt.figure()
plt.imshow(batch_img[2,:,:,0])
plt.figure()
plt.imshow(batch_img[3,:,:,0])


# ### Define loss function and training process

# In[4]:


cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
train = tf.train.AdamOptimizer(0.001).minimize(cost)

correct_prediction = tf.equal(tf.argmax(vgg.prob, 1), tf.argmax(true_out, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ### Accuracy Before Training
# After randomization, all weights of the network is set to random value, so the prediction should be totally random. Because we have 10 digits in this dataset, the accuracy of prediction of random guess should be around 10%. In this case, the initialization has 14.4% of accuracy.

# In[5]:


sess.run(tf.global_variables_initializer())

vbatch = mnist.validation.next_batch(500)
vbatch_img = vbatch[0].reshape((-1,28,28,1))
vbatch_lbl = vbatch[1]

print ('accuracy: ', sess.run(accuracy, feed_dict={images: vbatch_img, true_out: vbatch_lbl, train_mode: False}))


# In[ ]:


print(vgg.conv1_1)
print(vgg.conv1_2)
print(vgg.pool1)
print()
print(vgg.conv2_1)
print(vgg.conv2_2)
print(vgg.pool2)
print()
print(vgg.conv3_1)
print(vgg.conv3_2)
print(vgg.conv3_3)
print(vgg.conv3_4)
#print(vgg.pool3)


# ### Training
# Within 100 iteration, the accuracy increase to 94.4% on validation data set.

# In[ ]:


velapsed=[]

for i in range(1000):
    batch = mnist.train.next_batch(batch_size)
    batch_img = batch[0].reshape((-1,28,28,1))
    batch_lbl = batch[1]
    t = time.time()
    sess.run(train, feed_dict={images: batch_img, true_out: batch_lbl, train_mode: True})
    elapsed = time.time() - t
    velapsed.append(elapsed)
    if i % 50 == 0:
        print( 'iteration: ', i)
        vbatch = mnist.validation.next_batch(500)
        vbatch_img = vbatch[0].reshape((-1,28,28,1))
        vbatch_lbl = vbatch[1]
        print ('accuracy (validation): ', sess.run(accuracy, feed_dict={images: vbatch_img, true_out: vbatch_lbl, train_mode: False}))


# In[ ]:
tmean = sum(velapsed) / float(len(velapsed))
print('Mean elapsed time : {}'.format(tmean))

# 0.4610021071434021 (CPU)
# 0.0115001633167 (GPU : 3-titan xp)

# ### Validation
# After 1000 iterations, the accuracy on validation dataset increase to 98.75%. 

# In[ ]:


vbatch = mnist.validation.next_batch(2000)
vbatch_img = vbatch[0].reshape((-1,28,28,1))
vbatch_lbl = vbatch[1]
print(sess.run(accuracy, feed_dict={images: vbatch_img, true_out: vbatch_lbl, train_mode: False}))


# In[ ]:


vbatch = mnist.train.next_batch(2000)
vbatch_img = vbatch[0].reshape((-1,28,28,1))
vbatch_lbl = vbatch[1]
print(sess.run(accuracy, feed_dict={images: vbatch_img, true_out: vbatch_lbl, train_mode: False}))

