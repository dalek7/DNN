# based on https://github.com/boyw165/tensorflow-vgg

import numpy as np
import tensorflow as tf

import vgg16
import utils

from skimage import io
import matplotlib.pyplot as plt

fn1 ="./test_data/tiger.jpeg"
fn2 ="./test_data/puzzle.jpeg"
fn2 ="./test_data/6201041_sd.jpg"
fn2 ="./test_data/water1.jpg"


i1 = io.imread(fn1)
i2 = io.imread(fn2)

f, axarr = plt.subplots(1,2, sharey=True, figsize=(11,5))
axarr[0].imshow(i1)
axarr[1].imshow(i2)

img1 = utils.load_image(fn1)
img2 = utils.load_image(fn2)

# Just in case of four-channel images
img1 = img1[:, :, :3]
img2 = img2[:, :, :3]

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)

images = tf.placeholder("float", [2, 224, 224, 3])

# wget https://www.dropbox.com/s/8a8rei66f72um4i/vgg16.npy
vgg = vgg16.Vgg16('../weight/vgg16.npy')

with tf.name_scope("content_vgg"):
    vgg.build(images)

with tf.Session() as sess:
    feed_dict = {images: batch}
    prob = sess.run(vgg.prob, feed_dict=feed_dict)

print(prob)
print
utils.print_prob(prob[0], './synset.txt')
print
utils.print_prob(prob[1], './synset.txt')


plt.show()