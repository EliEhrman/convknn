"""
This module creates a db of the next level up in feature detection from chunks to super chunks

It takes as input the source images and a transform matrix.

The source images are converted into 27D convolutional chunks. The 3-channel image is coverted to a 2D image
where each pixel is represented by 27 values (the chunk).

Next the transform matrix is used to convert these 27 values to 3.

Next stride (currently 2) is applied so that every other pixel in each direction is discarded. (Sampling. In future implementations
there might be value in averaging or finding the strongest value in the 2*2 instead of discarding. However, currently
I don't know what averaging would mean nor is their a concept of the strongest.) We now create 3*3 super-chunks out of
the 3 values at each pixels over the 3*3 box. The cuhunk database is created as before.


"""
from __future__ import print_function
# import csv
import os
import numpy as np
import tensorflow as tf
import random
import urllib2
# from matplotlib import pyplot as plt
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.tensorboard.plugins import projector
import math


fdir = ("/devlink2/data/imagenet/")
WFile = '/devlink2/tempdata/convknn/dimred1.ckpt'
VisDir = '/devlink2/tempdata/convknn/vis'

c_img_w = 40
c_img_h = 40

img_w = c_img_w - 2 # 12 # 64
img_h = c_img_h - 2 # 8 # 64

# c_num_convs = img_h*img_w
c_num_centroids = 7 # 300
c_kernel_size = 3*3*3
c_num_ks = 32 # 32
c_class_dir_limit = 2000
c_min_file_size = 1000
c_num_files_per_batch = 2000 # 2000
# how much to offset the reciprocal for vote counting. So that the first is 1/c_knn_offset, second 1/(c_knn_offset+1) etc.
# the higher the value the less rank actually matters
c_knn_offset = 5.0
c_num_classes = 2
cb_limit_classes = True
c_num_rands = 5
c_num_iters = 10000
c_target_dim = 3

w_levels = []
h_levels = []
t_data_levels = []
w_levels.append(c_img_w - 2)
h_levels.append(c_img_h - 2)
t_data_levels.append(tf.Variable(tf.zeros([3, h_levels[0] + 2, w_levels[0] + 2], dtype=tf.float32), name='t_data_levels'))
max_level = 0

for level in range(1, 5):
	w = (w_levels[level-1] / 2) - 2
	h = (h_levels[level-1] / 2) - 2
	if w <= 0 or h <= 0:
		break
	w_levels.append(w)
	h_levels.append(h)
	t_data_levels.append(tf.Variable(tf.zeros([3, h_levels[level] + 2, w_levels[level] + 2], dtype=tf.float32), name='t_data_levels'))
	max_level = level

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
	iifn = 0
	for fn in os.listdir(dir_name):
		if iifn >= c_class_dir_limit:
			break
		ffn = os.path.join(dir_name, fn)
		if not os.path.isfile(ffn) or os.path.getsize(ffn) < c_min_file_size:
			continue
		file_list.append(fn)
		iifn += 1
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

t_data = tf.Variable(tf.zeros([3, img_h+2, img_w+2], dtype=tf.float32), name='t_data')

# Decode string into matrix with intensity values
ph_image_string = tf.placeholder(dtype=tf.string, shape=(), name='ph_image_string')
image = tf.image.decode_jpeg(ph_image_string, channels=3)
image = tf.expand_dims(image, 0)
image = tf.image.resize_bilinear(image, [img_h+2, img_w+2])[0]
op_data_set = tf.assign(t_data, tf.transpose(image, perm=[2,0,1]), name='op_data_set')

def make_level_conv(level):
	convarr = []
	for r in range(3):
		for c in range(3):
			convarr.append(t_data_levels[level][:, r:r + h_levels[level], c:c + w_levels[level]])
	# stack the array. Shape=[9, 3, h_levels[level], w_levels[level]]
	t_stacked = tf.stack(convarr, name='t_stacked')
	return t_stacked


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
# reshape so that the 3 channel image becomes a 2D image with a 27D vector at every pixel
# For now the shape is not really that of an image, we combine dims 0 and 1 so that mul can work later
# shape=[img_h * img_w, c_kernel_size]
t_chunks_img_unnorm = tf.reshape(t_convs, [img_h * img_w, c_kernel_size], name='t_chunks_img_unnorm')
# l2 normalize. Each chunk now has square of properties sum to 1. This captures the structure rather than ans values
t_chunks_img = tf.nn.l2_normalize(t_chunks_img_unnorm, dim=1, name='t_chunks_img')

# W converts the 27D values to 3D
W = tf.Variable(tf.random_normal([c_kernel_size, c_target_dim], 0.01/float(c_target_dim*c_kernel_size)), name='W')
# convert image to micro-chunks. Bring the shape back to that of an image
# Shape=[img_h, img_w, c_target_dim]
t_mchunk_img = tf.reshape(tf.matmul(t_chunks_img, W), shape=[img_h, img_w, c_target_dim], name='t_mchunk_img')
# Down sample the image by discarding 75% of the information
# Shape=[img_h/2, img_w/2, c_target_dim]
t_sampled_img = tf.strided_slice(t_mchunk_img, begin=[0,0,0], end=[img_h, img_w, c_target_dim], strides=[2,2,1])

num_convs = h_levels[1]*w_levels[1]

op_data_level_set = tf.assign(t_data_levels[1], tf.transpose(t_sampled_img, perm=[2,0,1]), name='op_data_level_set')
# Shape=[h_levels[1]*w_levels[1], c_kernel_size]
t_img_super_chunks = tf.nn.l2_normalize(tf.reshape(	tf.transpose(make_level_conv(1), perm=[2, 3, 0, 1]),
													shape=[h_levels[1] * w_levels[1], c_kernel_size]),
										dim=1, name='t_img_super_chunks')

# variable (and stop for call graph) holding the db that knn queries will run against.
# set by assign from op_db_norm_set
v_db_norm = tf.Variable(tf.zeros(shape=[num_convs*c_num_files_per_batch,c_kernel_size], dtype=tf.float32),  name='v_db_norm')
# place holder for the database, copied in from a numpy array built up from the convs of each image.
ph_db_norm = tf.placeholder(dtype=tf.float32, shape=[num_convs*c_num_files_per_batch,c_kernel_size],  name='ph_db_norm')
# op to put placeholder value into database
op_db_norm_set = tf.assign(v_db_norm, ph_db_norm, name='op_db_norm_set')

# variable for lables
v_db_labels = tf.Variable(tf.zeros(shape=[num_convs*c_num_files_per_batch], dtype=tf.int32), name='v_db_labels')
# place holder for the labels which are just the classes of the image repeated along all the convs of each image
ph_db_labels = tf.placeholder(dtype=tf.int32, shape=[num_convs*c_num_files_per_batch], name='ph_db_labels')
# op to put placeholder value into label variable
op_db_labels_set = tf.assign(v_db_labels, ph_db_labels, name='op_db_labels_set')

# variable for query
v_q = tf.Variable(tf.zeros(shape=[num_convs,c_kernel_size], dtype=tf.float32), name='v_q')
# place holder for the query which is an array of all of its convs, each of shape=[c_kernel_size]
ph_q = tf.placeholder(dtype=tf.float32, shape=[num_convs,c_kernel_size], name='ph_q')
# op to put placeholder value into query variable
op_q_set = tf.assign(v_q, ph_q, name='op_q_set')


# CDs of each conv of the query image against all the convs of all the images. Shape=[num_convs,num_convs*c_num_files_per_batch]
t_all_CDs = tf.matmul(v_q, v_db_norm, transpose_b=True, name='t_all_CDs')
# k closest indexes from the db for EACH of the convs of the query image.  Shape=[num_convs, c_num_ks]
t_best_CDs, t_best_CD_idxs = tf.nn.top_k(t_all_CDs, c_num_ks, sorted=True, name='t_best_CD_idxs')
# convert the indexes of the database to class integers, where each integer represents a differenct class.
# This is done by using the same indexes to access the labels. Shape=[num_convs, c_num_ks]
t_knn_classes = tf.gather(v_db_labels, t_best_CD_idxs)
# convert the former into one-hot
t_knn_class_oh = tf.one_hot(t_knn_classes, num_classes, on_value=1.0, off_value=0.0, name='t_knn_class_oh')
# the basic rank tensor which requires broadcasting before use. Shape=[c_num_ks]
t_rank_raw = tf.add(tf.range(c_num_ks, dtype=tf.float32), c_knn_offset, name='rank_knn')
t_rank_sum = tf.reduce_sum(tf.reciprocal(t_rank_raw), name='t_rank_sum')
# Now produce a tensor we can use to weight the knn results. Shape = [num_ks, num_classes]. All we've done is take the [2, 3, 4 ...]
# vector and turned it into [[2, 2..], [3, 3...], [4, 4...] ...] so that the divide won't complain
# it's a sort of explicit broadcast needed for shapes that the automatic broadcast can't handle
# Shape = [c_num_ks, num_classes]
t_rank = tf.tile(tf.expand_dims(t_rank_raw, 1), [1, num_classes], name='t_rank')
# weight the truth values by the rank of the result. i.e. closer neighbors. We are calculating an average truth value.
# What we have calculated here is the truth value for each element of the batch as predicted by the knn. Shape=[batch_size, num_classes]
t_knn_truth = tf.divide(tf.reduce_sum(tf.divide(t_knn_class_oh, t_rank), axis=1, name='mean_truth_by_idx'),
						t_rank_sum, name='t_knn_truth')
# The result of the vote is now just finding who has the biggest weighted average. Shape=[num_convs]
t_knn_winner = tf.cast(tf.argmax(t_knn_truth, axis=1), dtype=tf.int32, name='t_knn_winner')
# broadcast the winner so that we can compare to the original t_knn_classes, in order to create a count. Shape=[num_convs, c_num_ks]
t_knn_winner_broadcast = tf.tile(tf.expand_dims(t_knn_winner, -1), [1, c_num_ks], name='t_knn_winner_broadcast')
# place 1's where the elements are equal. Shape=[num_convs, c_num_ks]
# This represents how many actually voted for the winneer rather than the winner himself
# The importance of this is that if someone has, say, an absolute majority he gets a strong signal. If he scraped by
# where all classes are about equal, we don't think that is significant
t_knn_winner_count = tf.where(tf.equal(t_knn_winner_broadcast, t_knn_classes),
							tf.zeros([num_convs, c_num_ks], dtype=tf.float32),
							tf.ones([num_convs, c_num_ks], dtype=tf.float32), name='t_knn_winner_count')
# Shape=[num_convs]
t_knn_imortance_factor = tf.reduce_sum(t_knn_winner_count, axis=1, name='t_knn_imortance_factor')
# Shape=[num_convs, num_classes]
t_knn_signal = tf.squeeze(tf.matmul(tf.expand_dims(t_knn_imortance_factor, 0),
									tf.one_hot(t_knn_winner, num_classes, on_value=1.0, off_value=0.0)),
						  name='t_knn_signal')
t_knn_predict = tf.argmax(t_knn_signal, axis=0, name='t_knn_predict')


saver = tf.train.Saver({'dimred':W})
vis_save = tf.train.Saver()

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
merged = tf.summary.merge_all()
summaries_dir = '/tmp'
train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                      sess.graph)
vis_writer = tf.summary.FileWriter(VisDir)
# config = projector.ProjectorConfig()
# embedding = config.embeddings.add()
# embedding.tensor_name = v_centroids.name

sess.run(tf.global_variables_initializer())
# sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="curses")

saver.restore(sess, WFile)

strings, classes = get_file_strings(c_num_files_per_batch)
db_els = np.zeros([num_convs*c_num_files_per_batch, c_kernel_size], dtype=np.float32)
db_labels = np.repeat(classes, num_convs)
for ibatch in range(c_num_files_per_batch):
	sess.run(op_data_set, feed_dict={ph_image_string:strings[ibatch]})
	sess.run(op_data_level_set)
	db_els[ibatch*num_convs:(ibatch+1)*num_convs] = sess.run(t_img_super_chunks)
	if (ibatch % (c_num_files_per_batch / 100) == 0):
		print('num files in db:', ibatch)

r_db_norm = sess.run(op_db_norm_set, feed_dict={ph_db_norm:db_els })
r_db_labels = sess.run(op_db_labels_set, feed_dict={ph_db_labels:db_labels })

print('db created.')

num_hits = 0.0
num_tests = 100
for itest in range(num_tests):
	qtrings, qclass = get_file_strings(1)
	sess.run(op_data_set, feed_dict={ph_image_string: qtrings[0]})
	sess.run(op_data_level_set)
	q_els = sess.run(t_img_super_chunks)
	sess.run(op_q_set, feed_dict={ph_q:q_els})
	r_pred = sess.run(t_knn_predict)
	# r_knn_classes, r_knn_truth, r_knn_winner, r_knn_predict, r_knn_signal, r_best_CD_idxs, r_best_CDs, r_q = \
	# 	sess.run([t_knn_classes, t_knn_truth, t_knn_winner, t_knn_predict, t_knn_signal, t_best_CD_idxs, t_best_CDs, v_q])
	if r_pred == qclass[0]:
		num_hits += 1.0
print('success rate:', num_hits / float(num_tests))

strings, classes = get_file_strings(1)
sess.run(op_data_set, feed_dict={ph_image_string: strings[0]})
sess.run(op_data_level_set)
r_img_super_chunks = sess.run(t_img_super_chunks)


print('done!')