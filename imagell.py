from __future__ import print_function
# import csv
import numpy as np
import tensorflow as tf
import random
import urllib2
from matplotlib import pyplot as plt


url = ("file:///devlink2/data/imagenet/flower/img11911632.jpg")
img_w = 12
img_h = 8
c_num_centroids = 7
c_kernel_size = 3*3*3

t_data = tf.Variable(tf.zeros([3, img_h+2, img_w+2], dtype=tf.float32))

def extract(elem):
	# tf.assign(t_output[i], t_data[i])
	global t_data
	iis = [t_data[k, elem[0]+j, elem[1]+i] for i in range(-1,2) for j in range(-1,2) for k in range(3)]
	return iis

image_string = urllib2.urlopen(url).read()

picindxs = [[j, i] for i in range(1,img_w+1) for j in range(1,img_h+1)]

# Decode string into matrix with intensity values
image = tf.image.decode_jpeg(image_string, channels=3)
image = tf.expand_dims(image, 0)
image = tf.image.resize_bilinear(image, [img_h+2, img_w+2])[0]
op_data_set = tf.assign(t_data, tf.transpose(image, perm=[2,0,1]))

elems = tf.constant(picindxs, dtype=tf.int32, name='elems')
structure = [tf.float32 for i in range(c_kernel_size)]
db_raw = tf.map_fn(extract, elems, dtype=structure)
db = tf.transpose(tf.stack(db_raw), perm=[1,0])
db_norm = tf.nn.l2_normalize(db, dim=1, name='db_norm')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

sess.run(op_data_set)
rdb = sess.run(db_norm)

np_image = sess.run(image)

plt.figure()
plt.imshow(np_image.astype(np.uint8))
plt.suptitle(url, fontsize=14, fontweight='bold')
# plt.axis('off')
plt.show()

print('done!')