"""
This module many jpg images, decodes them, extracts the convolution chunks (27 elements)
and creates a knn database out of them. Each image has a label based on which dir it came from

It then takes a number more images and applies knn to each of its conv chunks, and does a weighted average
of the results. It compares the predicted class to the real class to get an accuracy score.

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
img_w = 24 # 12
img_h = 24 # 8
c_num_convs = img_h*img_w
c_num_centroids = 7
c_kernel_size = 3*3*3
c_num_ks = 32
c_class_dir_limit = 10
c_min_file_size = 1000
c_num_files_per_batch = 500
# how much to offset the reciprocal for vote counting. So the first is 1/c_knn_offset, second 1/(c_knn_offset+1) etc.
# the higher the value the less rank actually matters
c_knn_offset = 5.0
c_num_classes = 2
cb_limit_classes = True

t_data = tf.Variable(tf.zeros([3, img_h+2, img_w+2], dtype=tf.float32), name='t_data')

def extract(elem):
	# tf.assign(t_output[i], t_data[i])
	global t_data
	iis = [t_data[k, elem[0]+j, elem[1]+i] for i in range(-1,2) for j in range(-1,2) for k in range(3)]
	return iis

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
	num_classes = c_num_classes

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


picindxs = [[j, i] for i in range(1,img_w+1) for j in range(1,img_h+1)]

# Decode string into matrix with intensity values
ph_image_string = tf.placeholder(dtype=tf.string, shape=(), name='ph_image_string')
image = tf.image.decode_jpeg(ph_image_string, channels=3)
image = tf.expand_dims(image, 0)
image = tf.image.resize_bilinear(image, [img_h+2, img_w+2])[0]
op_data_set = tf.assign(t_data, tf.transpose(image, perm=[2,0,1]), name='op_data_set')

elems = tf.Variable(tf.constant(picindxs, dtype=tf.int32, name='elems'))
structure = [tf.float32 for i in range(c_kernel_size)]
db_raw = tf.map_fn(extract, elems, dtype=structure, name='db_raw')
# Shape=[c_num_convs, c_kernel_size]
db = tf.transpose(tf.stack(db_raw), perm=[1,0], name='db')
db_norm = tf.nn.l2_normalize(db, dim=1, name='db_norm')
# Shape=[c_num_convs, c_kernel_size]
v_db_norm = tf.Variable(tf.zeros([c_num_convs, c_kernel_size], dtype=tf.float32), name='v_db_norm')
op_db_norm =  tf.assign(v_db_norm, tf.nn.l2_normalize(db, dim=1), name='op_db_norm')

# place holder for the database, copied in from a numpy array built up from the convs of each image.
ph_db_norm = tf.placeholder(dtype=tf.float32, shape=[c_num_convs*c_num_files_per_batch,c_kernel_size],  name='ph_db_norm')
# place holder for query which is the convs of a single image
ph_q = tf.placeholder(dtype=tf.float32, shape=[c_num_convs,c_kernel_size], name='ph_q')
# place holder for the labels which are just the classes of the image repeated along all the convs of each image
ph_db_labels = tf.placeholder(dtype=tf.int32, shape=[c_num_convs*c_num_files_per_batch], name='ph_db_labels')
# CDs of each conv of the query image against all the convs of all the images. Shape=[c_num_convs,c_num_convs*c_num_files_per_batch]
t_all_CDs = tf.matmul(ph_q, ph_db_norm, transpose_b=True, name='t_all_CDs')
# k closest indexes from the db for EACH of the convs of the query image.  Shape=[c_num_convs, c_num_ks]
t_best_CDs, t_best_CD_idxs = tf.nn.top_k(t_all_CDs, c_num_ks, sorted=True, name='t_best_CD_idxs')
# convert the indexes of the database to class integers, where each integer represents a differenct class.
# This is done by using the same indexes to access the labels. Shape=[c_num_convs, c_num_ks]
t_knn_classes = tf.gather(ph_db_labels, t_best_CD_idxs)
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
# The result of the vote is now just finding who has the biggest weighted average. Shape=[c_num_convs]
t_knn_winner = tf.cast(tf.argmax(t_knn_truth, axis=1), dtype=tf.int32, name='t_knn_winner')
# broadcast the winner so that we can compare to the original t_knn_classes, in order to create a count. Shape=[c_num_convs, c_num_ks]
t_knn_winner_broadcast = tf.tile(tf.expand_dims(t_knn_winner, -1), [1, c_num_ks], name='t_knn_winner_broadcast')
# place 1's where the elements are equal. Shape=[c_num_convs, c_num_ks]
# This represents how many actually voted for the winneer rather than the winner himself
# The importance of this is that if someone has, say, an absolute majority he gets a strong signal. If he scraped by
# where all classes are about equal, we don't think that is significant
t_knn_winner_count = tf.where(tf.equal(t_knn_winner_broadcast, t_knn_classes),
							tf.zeros([c_num_convs, c_num_ks], dtype=tf.float32),
							tf.ones([c_num_convs, c_num_ks], dtype=tf.float32), name='t_knn_winner_count')
# Shape=[c_num_convs]
t_knn_imortance_factor = tf.reduce_sum(t_knn_winner_count, axis=1, name='t_knn_imortance_factor')
# Shape=[c_num_convs, num_classes]
t_knn_signal = tf.squeeze(tf.matmul(tf.expand_dims(t_knn_imortance_factor, 0),
									tf.one_hot(t_knn_winner, num_classes, on_value=1.0, off_value=0.0)),
						  name='t_knn_signal')
t_knn_predict = tf.argmax(t_knn_signal, axis=0, name='t_knn_predict')

#
#
# ph_query
# t_all_CDs = tf.matmul(v_centroids, v_db_norm, transpose_b=True, name='t_all_CDs')
# t_best_CDs, t_best_CD_idxs = tf.nn.top_k(t_all_CDs, c_num_ks, sorted=True, name='t_best_CD_idxs')
# t_closest_idxs = tf.argmax(t_all_CDs, axis=0, name='t_closest_idxs')


# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
merged = tf.summary.merge_all()
summaries_dir = '/tmp'
train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                      sess.graph)

sess.run(tf.global_variables_initializer())
# sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="curses")

strings, classes = get_file_strings(c_num_files_per_batch)
db_els = np.zeros([c_num_convs*c_num_files_per_batch, c_kernel_size], dtype=np.float32)
db_labels = np.repeat(classes, c_num_convs)
for ibatch in range(c_num_files_per_batch):
	sess.run(op_data_set, feed_dict={ph_image_string:strings[ibatch]})
	db_els[ibatch*c_num_convs:(ibatch+1)*c_num_convs] = sess.run(db_norm)
	if (ibatch % (c_num_files_per_batch / 100) == 0):
		print('num files in db:', ibatch)

print('db created.')

num_hits = 0.0
num_tests = 100
for itest in range(num_tests):
	qtrings, qclass = get_file_strings(1)
	sess.run(op_data_set, feed_dict={ph_image_string: qtrings[0]})
	q_els = sess.run(db_norm)
	r_pred = sess.run(t_knn_predict, feed_dict={ph_db_norm:db_els, ph_q:q_els, ph_db_labels:db_labels})
	if r_pred == qclass:
		num_hits += 1.0
print('success rate:', num_hits / float(num_tests))
# r_knn_classes, r_knn_truth, r_knn_winner, r_knn_predict, r_knn_signal = \
# 	sess.run([t_knn_classes, t_knn_truth, t_knn_winner, t_knn_predict, t_knn_signal],
# 			 feed_dict={ph_db_norm:db_els, ph_q:q_els, ph_db_labels:db_labels})


# np_image = sess.run(image)
#
# plt.figure()
# plt.imshow(np_image.astype(np.uint8))
# plt.suptitle(ffn, fontsize=14, fontweight='bold')
# # plt.axis('off')
# plt.show()

print('done!')