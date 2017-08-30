"""
This module takes a single jpg image, decodes it, extracts the convolution items (27 elements)
and performs kmeans on it
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
fdir = ("/devlink2/data/imagenet/flower/")
img_w = 360 # 12
img_h = 240 # 8
c_num_convs = img_h*img_w
c_num_centroids = 70
c_kernel_size = 3*3*3
c_num_ks = 3

t_data = tf.Variable(tf.zeros([3, img_h+2, img_w+2], dtype=tf.float32), name='t_data')

def extract(elem):
	# tf.assign(t_output[i], t_data[i])
	global t_data
	iis = [t_data[k, elem[0]+j, elem[1]+i] for i in range(-1,2) for j in range(-1,2) for k in range(3)]
	return iis

# image_string = urllib2.urlopen(url).read()
for ifn, fn in enumerate(os.listdir(fdir)):
	ffn = os.path.join(fdir, fn)
	if os.path.isfile(ffn):
		with open(ffn, mode='rb') as f:
			image_string = f.read()
			if ifn >= 0:
				break

picindxs = [[j, i] for i in range(1,img_w+1) for j in range(1,img_h+1)]

# Decode string into matrix with intensity values
image = tf.image.decode_jpeg(image_string, channels=3)
image = tf.expand_dims(image, 0)
image = tf.image.resize_bilinear(image, [img_h+2, img_w+2])[0]
op_data_set = tf.assign(t_data, tf.transpose(image, perm=[2,0,1]), name='op_data_set')

elems = tf.constant(picindxs, dtype=tf.int32, name='elems')
structure = [tf.float32 for i in range(c_kernel_size)]
db_raw = tf.map_fn(extract, elems, dtype=structure, name='db_raw')
# Shape=[c_num_convs, c_kernel_size]
db = tf.transpose(tf.stack(db_raw), perm=[1,0], name='db')
# Shape=[c_num_convs, c_kernel_size]
v_db_norm = tf.Variable(tf.zeros([c_num_convs, c_kernel_size], dtype=tf.float32), name='v_db_norm')
op_db_norm =  tf.assign(v_db_norm, tf.nn.l2_normalize(db, dim=1), name='op_db_norm')

# The goal is to cluster the convolution vectors so that we can perform dimension reduction
# KMeans implementation
# Intitialize the centroids indicies. Shape=[num_centroids]
t_centroids_idxs_init = tf.random_uniform([c_num_centroids], 0, c_num_convs-1, dtype=tf.int32, name='t_centroids_idxs_init')
# Get the centroids variable ready. Must persist between loops
v_centroids = tf.Variable(tf.zeros([c_num_centroids, c_kernel_size], dtype=tf.float32), name='v_centroids')
# Create actual centroids as seeds. Shape=[num_centroids, c_kernel_size]
op_centroids_init = tf.assign(v_centroids, tf.gather(v_db_norm, t_centroids_idxs_init, name='op_centroids_init'))
# Do cosine distances for all centroids on all elements of the db. Shape [c_num_centroids, c_num_convs]
t_all_CDs = tf.matmul(v_centroids, v_db_norm, transpose_b=True, name='t_all_CDs')
# Do top_k. Shape = [c_num_centroids, c_num_ks]
## t_best_CDs, t_best_CD_idxs = tf.nn.top_k(t_all_CDs, c_num_ks, sorted=True, name='t_best_CD_idxs')
# For each element in the matrix, find the centroid that's closest. Shape=[c_num_convs]
t_closest_idxs = tf.argmax(t_all_CDs, axis=0, name='t_closest_idxs')
l_new_centroids = []
l_votes_count = []
for icent in range(c_num_centroids):
	# Create an array of True if the closest index was this centroid
	# Shape=[c_num_centroids]
	t_vote_for_this = tf.equal(t_closest_idxs, icent, name='t_vote_for_this')
	# Count the number of trues in the vote_for_tis tensor
	# Shape=()
	t_vote_count = tf.reduce_sum(tf.cast(t_vote_for_this, tf.float32), name='t_vote_count')
	# Create the cluster. Use the True positions to put in the values from the v_db_norm and put zeros elsewhere.
	# This means that instead of a short list of the vectors in this cluster we use the full size with zeros for non-members
	# Shape=[c_num_convs, c_kernel_size]
	t_this_cluster = tf.where(t_vote_for_this, v_db_norm, tf.zeros([c_num_convs, c_kernel_size]), name='t_this_cluster')
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
# First time around I forgot that I must normalize the centroids as required for shperical k-means. Avg, as above, will not produce a normalized result
t_new_centroids = tf.nn.l2_normalize(tf.stack(l_new_centroids), dim=1, name='t_new_centroids')
#Shape=[c_num_centroids]
t_votes_count = tf.stack(l_votes_count, name='t_votes_count')
# take the new random idxs and gather new centroids from the db. Only used in case count == 0. Shape=[num_centroids, c_kernel_size]
t_centroids_from_idxs = tf.gather(v_db_norm, t_centroids_idxs, name='t_centroids_from_idxs')
# Assign back to the original v_centroids so that we can go for another round
op_centroids_update = tf.assign(v_centroids, tf.where(tf.greater(t_votes_count, 0.0), t_new_centroids,
													  t_centroids_from_idxs, name='centroids_where'),
								name='op_centroids_update')

# The following section of code is designed to evaluate the cluster quality, specifically the average distance of a conv fragment from
# its centroid.
# t_closest_idxs is an index for each element in the database, specifying which cluster it belongs to. So we use that to
# replicate the centroid of that cluster to the locations alligned with each memebr of the database
# Shape=[c_num_convs, c_kernel_size]
t_centroid_broadcast = tf.gather(v_centroids, t_closest_idxs, name='t_centroid_broadcast')
# element-wise multiplication of each property and the sum down the properties. It is reallt just a CD but we aren't using matmul
# Shape=[c_num_convs]
t_cent_dist = tf.reduce_sum(tf.multiply(v_db_norm, t_centroid_broadcast), axis=1, name='t_cent_dist')
# Extract a single number representing the kmeans error. This is the mean of the distances from closest centers
t_kmeans_err = tf.reduce_mean(t_cent_dist, name='t_kmeans_err')

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# sess = tf.Session()
merged = tf.summary.merge_all()
summaries_dir = '/tmp'
train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                      sess.graph)

sess.run(tf.global_variables_initializer())
# sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="curses")

sess.run(op_data_set)
rdb = sess.run(op_db_norm)
rdb = sess.run(op_db_norm)

sess.run(op_centroids_init)
for ikmeans in range(12):
	print('r_votes_count =', sess.run(t_votes_count))
	print('r_kmeans_err =', sess.run(t_kmeans_err))
	r_centroids = sess.run(op_centroids_update)

np_image = sess.run(image)

plt.figure()
plt.imshow(np_image.astype(np.uint8))
plt.suptitle(ffn, fontsize=14, fontweight='bold')
# plt.axis('off')
plt.show()

print('done!')