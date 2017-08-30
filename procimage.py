from __future__ import print_function
# import csv
import numpy as np
import tensorflow as tf
import random
import urllib2
from matplotlib import pyplot as plt


url = ("file:///devlink2/data/imagenet/flower/img11911632.jpg")
image_string = urllib2.urlopen(url).read()

# Decode string into matrix with intensity values
image = tf.image.decode_jpeg(image_string, channels=3)
image = tf.expand_dims(image, 0)
image = tf.image.resize_bilinear(image, [10, 10])[0]

def extract(indxs):
	global image
	return [image[indxs], image[indxs+1]]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

np_image = sess.run(image)

plt.figure()
plt.imshow(np_image.astype(np.uint8))
plt.suptitle(url, fontsize=14, fontweight='bold')
# plt.axis('off')
plt.show()

print('done!')