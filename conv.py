"""
This module extracts conv chunks from an image file. Unlike many others such as imagell.py cluster.py and knnimage.py
it does not use MapFn but rather extracts and moves slices

It is designed as an alternative building block for the knn by conv experiments in this directory

"""
from __future__ import print_function
# import csv
import os
import numpy as np
import tensorflow as tf
import random
import urllib2
from matplotlib import pyplot as plt
from tensorflow.python import debug as tf_debug


# fdir = ("file:///devlink2/data/imagenet/flower/")
fdir = ("/devlink2/data/imagenet/")
img_w = 12 # 12
img_h = 8 # 8
c_num_convs = img_h*img_w
c_num_centroids = 7
c_kernel_size = 3*3*3
c_num_ks = 32
c_class_dir_limit = 10
c_min_file_size = 1000
c_num_files_per_batch = 200
# how much to offset the reciprocal for vote counting. So the first is 1/c_knn_offset, second 1/(c_knn_offset+1) etc.
# the higher the value the less rank actually matters
c_knn_offset = 5.0
c_num_classes = 2
cb_limit_classes = True


t_data = tf.Variable(tf.zeros([3, img_h+2, img_w+2], dtype=tf.float32), name='t_data')

# image_string = urllib2.urlopen(url).read()
classnames = []
for ifn, fn in enumerate(os.listdir(fdir)):
	ffn = os.path.join(fdir, fn)
	if os.path.isdir(ffn):
		classnames.append(fn)

all_file_list = []
num_files_in_class = []
for classname in classnames:
	file_list = []
	dir_name = os.path.join(fdir, classname)
	for ifn, fn in enumerate(os.listdir(dir_name)):
		if ifn >= c_class_dir_limit:
			break
		ffn = os.path.join(dir_name, fn)
		if not os.path.isfile(ffn) or os.path.getsize(ffn) < c_min_file_size:
			continue
		file_list.append(fn)
	all_file_list.append(file_list)
	num_files_in_class.append(ifn)

num_classes = len(classnames)
if cb_limit_classes:
	num_classes = min(num_classes, c_num_classes)

def get_file_strings(num):
	strings = []
	classes = []
	for inum in range(num):
		iclass = random.randint(0, num_classes-1)
		ifile = random.randint(0, num_files_in_class[iclass] - 1)
		ffn = os.path.join(fdir, classnames[iclass], all_file_list[iclass][ifile])
		with open(ffn, mode='rb') as f:
			strings.append(f.read())
		classes.append(iclass)
	return strings, classes

# Decode string into matrix with intensity values
ph_image_string = tf.placeholder(dtype=tf.string, shape=(), name='ph_image_string')
image = tf.image.decode_jpeg(ph_image_string, channels=3)
image = tf.expand_dims(image, 0)
image = tf.image.resize_bilinear(image, [img_h+2, img_w+2])[0]
op_data_set = tf.assign(t_data, tf.transpose(image, perm=[2,0,1]), name='op_data_set')

# A simple loop to create an array of images each one offset width and height
# creates a list of 9 tensors each Shape=[3, img_h, img_w].
# note the original was 2 pixels more in each of the flat dims
convarr = []
for r in range(3):
	for c in range(3):
		convarr.append(t_data[:, r:r+img_h, c:c+img_w])

# stack the array. Shape=[9, 3, img_h, img_w]
t_stacked = tf.stack(convarr, name='t_stacked')
# Transpose so the data actually moves i.e. the flat underlying representation has data in a different order.
# Shape=[img_h, img_w, 9, 3, ]
t_convs = tf.transpose(t_stacked, perm=[2, 3, 0, 1], name='t_convs')
# reshape for what we want in the database or query images.
# shape=[c_num_convs, c_kernel_size]
t_chunks = tf.reshape(t_convs, [img_h*img_w, c_kernel_size], name='t_chunks')
# l2 normalize. Each chunk now has square of properties sum to 1. This captures the structure rather than ans values
db_norm = tf.nn.l2_normalize(t_chunks, dim=1, name='db_norm')

v_db_norm = tf.Variable(tf.zeros(shape=[c_num_convs*c_num_files_per_batch,c_kernel_size], dtype=tf.float32),  name='v_db_norm')
# place holder for the database, copied in from a numpy array built up from the convs of each image.
ph_db_norm = tf.placeholder(dtype=tf.float32, shape=[c_num_convs*c_num_files_per_batch,c_kernel_size],  name='ph_db_norm')
#o= op to put placeholder value into database
op_db_norm_set = tf.assign(v_db_norm, ph_db_norm)


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# sess = tf.Session()
merged = tf.summary.merge_all()
summaries_dir = '/tmp'
train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                      sess.graph)

sess.run(tf.global_variables_initializer())
# sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="curses")

# qtrings, qclass = get_file_strings(1)
# sess.run(op_data_set, feed_dict={ph_image_string: qtrings[0]})
# r_convs, r_data, r_stacked, r_chunks = sess.run([t_convs, t_data, t_stacked, t_chunks])

strings, classes = get_file_strings(c_num_files_per_batch)
db_els = np.zeros([c_num_convs*c_num_files_per_batch, c_kernel_size], dtype=np.float32)
db_labels = np.repeat(classes, c_num_convs)
for ibatch in range(c_num_files_per_batch):
	sess.run(op_data_set, feed_dict={ph_image_string:strings[ibatch]})
	db_els[ibatch*c_num_convs:(ibatch+1)*c_num_convs] = sess.run(db_norm)
	if (ibatch % (c_num_files_per_batch / 100) == 0):
		print('num files in db:', ibatch)

sess.run(op_db_norm_set, feed_dict={ph_db_norm:db_els })
print('db created.')

# np_image = sess.run(image)
#
# plt.figure()
# plt.imshow(np_image.astype(np.uint8))
# plt.suptitle(ffn, fontsize=14, fontweight='bold')
# # plt.axis('off')
# plt.show()

print('done!')