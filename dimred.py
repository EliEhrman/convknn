"""
This module reduces the dimensions of a 27 value conv chunk by learning a matrix that transforms it to 3 values under the
constraint of a loss function that tries to keep the cosine distance between the pre-transformed identical to the posttransformed

"""
from __future__ import print_function
import csv
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
LogDir = '/devlink2/tempdata/convknn/vis/'
LogFile = 'dimred1.ckpt'
tsv_path = 'metadata.tsv'

img_w = 12 # 12 # 64
img_h = 8 # 8 # 64
c_num_convs = img_h*img_w
c_num_centroids = 10 # 300
c_kernel_size = 3*3*3
c_num_ks = 32 # 32
c_class_dir_limit = 200
c_min_file_size = 1000
c_num_files_per_batch = 100 # 2000
# how much to offset the reciprocal for vote counting. So that the first is 1/c_knn_offset, second 1/(c_knn_offset+1) etc.
# the higher the value the less rank actually matters
c_knn_offset = 5.0
c_num_classes = 10
cb_limit_classes = True
c_num_rands = 5
c_num_iters = 10000
c_target_dim = 3
c_kmeans_iters = 10 # 100
c_learn_rate = 0.05

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
	num_files_in_class.append(iifn)

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

# variable (and stop for call graph) holding the db that knn queries will run against.
# set by assign from op_db_norm_set
v_db_norm = tf.Variable(tf.zeros(shape=[c_num_convs*c_num_files_per_batch,c_kernel_size], dtype=tf.float32),  name='v_db_norm')
# place holder for the database, copied in from a numpy array built up from the convs of each image.
ph_db_norm = tf.placeholder(dtype=tf.float32, shape=[c_num_convs*c_num_files_per_batch,c_kernel_size],  name='ph_db_norm')
# op to put placeholder value into database
op_db_norm_set = tf.assign(v_db_norm, ph_db_norm, name='op_db_norm_set')

# variable for lables
v_db_labels = tf.Variable(tf.zeros(shape=[c_num_convs*c_num_files_per_batch], dtype=tf.int32), name='v_db_labels')
# place holder for the labels which are just the classes of the image repeated along all the convs of each image
ph_db_labels = tf.placeholder(dtype=tf.int32, shape=[c_num_convs*c_num_files_per_batch], name='ph_db_labels')
# op to put placeholder value into database
op_db_labels_set = tf.assign(v_db_labels, ph_db_labels, name='op_db_labels_set')

# The goal is to cluster the convolution vectors so that we can perform dimension reduction
# KMeans implementation
# Intitialize the centroids indicies. Shape=[num_centroids]
t_centroids_idxs_init = tf.random_uniform([c_num_centroids], 0, c_num_convs-1, dtype=tf.int32, name='t_centroids_idxs_init')
# Get the centroids variable ready. Must persist between loops. Shape=[c_num_convs*c_num_files_per_batch, c_kernel_size]
v_centroids = tf.Variable(tf.zeros([c_num_centroids, c_kernel_size], dtype=tf.float32), name='v_centroids')
# Create actual centroids as seeds. Shape=[num_centroids, c_kernel_size]
op_centroids_init = tf.assign(v_centroids, tf.gather(v_db_norm, t_centroids_idxs_init, name='op_centroids_init'))
# Do cosine distances for all centroids on all elements of the db. Shape [c_num_centroids, c_num_convs*c_num_files_per_batch]
t_all_CDs = tf.matmul(v_centroids, v_db_norm, transpose_b=True, name='t_all_CDs')
# Do top_k. Shape = [c_num_centroids, c_num_ks]
## t_best_CDs, t_best_CD_idxs = tf.nn.top_k(t_all_CDs, c_num_ks, sorted=True, name='t_best_CD_idxs')
# For each element in the matrix, find the centroid that's closest. Shape=[c_num_convs*c_num_files_per_batch]
v_closest_idxs = tf.Variable(tf.zeros(shape=[c_num_convs*c_num_files_per_batch], dtype=tf.int64), name='v_closest_idxs')
op_closest_idxs_set = tf.assign(v_closest_idxs, tf.argmax(t_all_CDs, axis=0), name='op_closest_idxs_set')
l_new_centroids = []
l_votes_count = []
for icent in range(c_num_centroids):
	# Create an array of True if the closest index was this centroid
	# Shape=[c_num_centroids]
	t_vote_for_this = tf.equal(v_closest_idxs, icent, name='t_vote_for_this')
	# Count the number of trues in the vote_for_tis tensor
	# Shape=()
	t_vote_count = tf.reduce_sum(tf.cast(t_vote_for_this, tf.float32), name='t_vote_count')
	# Create the cluster. Use the True positions to put in the values from the v_db_norm and put zeros elsewhere.
	# This means that instead of a short list of the vectors in this cluster we use the full size with zeros for non-members
	# Shape=[c_num_convs*c_num_files_per_batch, c_kernel_size]
	t_this_cluster = tf.where(t_vote_for_this, v_db_norm, tf.zeros([c_num_convs*c_num_files_per_batch, c_kernel_size]), name='t_this_cluster')
	# Sum the values for each property to get the aveage property
	# Shape=[c_kernel_size]
	t_cluster_sum = tf.reduce_sum(t_this_cluster, axis=0, name='t_cluster_sum')
	# Shape=[c_kernel_size]
	t_avg = tf.cond(t_vote_count > 0.0,
					lambda: tf.divide(t_cluster_sum, t_vote_count),
					lambda: tf.zeros([c_kernel_size]),
					name='t_avg')
	l_new_centroids.append(t_avg)
	l_votes_count.append(t_vote_count)
# Do random centroids again. This time for filling in
t_centroids_idxs = tf.random_uniform([c_num_centroids], 0, c_num_convs-1, dtype=tf.int32, name='t_centroids_idxs')
#Shape = [c_num_centroids, c_kernel_size]
t_new_centroids = tf.stack(l_new_centroids, name='t_new_centroids')
# First time around I forgot that I must normalize the centroids as required for shperical k-means. Avg, as above, will not produce a normalized result
t_new_centroids_norm = tf.nn.l2_normalize(t_new_centroids, dim=1, name='t_new_centroids_norm')
#Shape=[c_num_centroids]
t_votes_count = tf.stack(l_votes_count, name='t_votes_count')
# take the new random idxs and gather new centroids from the db. Only used in case count == 0. Shape=[num_centroids, c_kernel_size]
t_centroids_from_idxs = tf.gather(v_db_norm, t_centroids_idxs, name='t_centroids_from_idxs')
# Assign back to the original v_centroids so that we can go for another round
op_centroids_update = tf.assign(v_centroids, tf.where(tf.greater(t_votes_count, 0.0), t_new_centroids_norm,
													  t_centroids_from_idxs, name='centroids_where'),
								name='op_centroids_update')

# The following section of code is designed to evaluate the cluster quality, specifically the average distance of a conv fragment from
# its centroid.
# t_closest_idxs is an index for each element in the database, specifying which cluster it belongs to. So we use that to
# replicate the centroid of that cluster to the locations alligned with each member of the database
# Shape=[c_num_convs*c_num_files_per_batch, c_kernel_size]
t_centroid_broadcast = tf.gather(v_centroids, v_closest_idxs, name='t_centroid_broadcast')
# element-wise multiplication of each property and the sum down the properties. It is reallt just a CD but we aren't using matmul
# Shape=[c_num_convs*c_num_files_per_batch]
t_cent_dist = tf.reduce_sum(tf.multiply(v_db_norm, t_centroid_broadcast), axis=1, name='t_cent_dist')
# Extract a single number representing the kmeans error. This is the mean of the distances from closest centers. Shape=()
t_kmeans_err = tf.reduce_mean(t_cent_dist, name='t_kmeans_err')

# The follwing code is the call graph for dimension reduction
# We take two random vectors from the centroids, create a matrix of cosine distances between them
t_r1 = tf.random_uniform([c_num_rands], 0, c_num_centroids-1, dtype=tf.int32, name='t_r1')
t_r2 = tf.random_uniform([c_num_rands], 0, c_num_centroids-1, dtype=tf.int32, name='t_r2')
# W converts the 27D values to 3D
W = tf.Variable(tf.random_normal([c_kernel_size, c_target_dim], 0.01/float(c_target_dim*c_kernel_size)), name='W')
# gather to get the randomly selected centroids themselves
# Shape=[c_num_rands, c_kernel_size]
t_l1 = tf.gather(v_centroids, t_r1, name='t_l1')
t_l2 = tf.gather(v_centroids, t_r2, name='t_l2')
# Calculate the cosine distance between each of one vector and each of the other
# Shape=[c_num_rands, c_num_rands]
t_ldist = tf.matmul(t_l1, t_l2, transpose_b=True, name='t_ldist')
# transform each set of centroids using W and calculate the distances between the same points we did before but on 3D
t_t1 = tf.matmul(t_l1, W, name='t_t1')
t_t2 = tf.matmul(t_l2, W, name='t_t2')
t_tdist = tf.matmul(t_t1, t_t2, transpose_b=True, name='t_tdist')
# Define the error as the difference between the pre-transformed and post-transfomed
t_err = tf.reduce_mean(tf.square(t_ldist-t_tdist), name='t_err')
# create a trainer
train_step = tf.train.AdagradOptimizer(c_learn_rate).minimize(t_err)

saver = tf.train.Saver({'dimred':W, 'v_centroids':v_centroids})
# saver = tf.train.Saver()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# sess = tf.Session()
merged = tf.summary.merge_all()
summaries_dir = LogDir
summary_writer = tf.summary.FileWriter(LogDir,
                                      sess.graph)
# Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = v_centroids.name
# embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

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
sess.run(op_db_labels_set, feed_dict={ph_db_labels:db_labels })

print('db created.')

# start by creating random centroids
sess.run(op_centroids_init)
# calculate the closest centroid for each db record
sess.run(op_closest_idxs_set)
print('r_votes_count =', sess.run(t_votes_count))
print('r_kmeans_err =', sess.run(t_kmeans_err))
for ikmeans in range(c_kmeans_iters):
	print('run num:', ikmeans)
	r_centroids = sess.run(op_centroids_update)
	print('r_votes_count =', sess.run(t_votes_count))
	print('r_kmeans_err =', sess.run(t_kmeans_err))
	sess.run(op_closest_idxs_set)


learn_phase = 0
for step in range(c_num_iters+1):
	if step % (c_num_iters / 100) == 0:
		errval1 = math.sqrt(sess.run(t_err))
		sess.run([train_step])
		errval2 = math.sqrt(sess.run(t_err))
		print('step:', step, ', err1:', errval1, ', err2:', errval2)
	else:
		sess.run([train_step])

save_path = saver.save(sess, os.path.join(LogDir, LogFile))
print("Model saved in file: %s" % save_path)
projector.visualize_embeddings(summary_writer, config)

print('done!')