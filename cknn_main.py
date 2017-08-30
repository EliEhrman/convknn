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
import sys, getopt

import cknn_init
import cknn_db
import cknn_kmeans
import cknn_control
import cknn_knn

from cknn_init import w_levels
from cknn_init import h_levels
# from cknn_init import max_level
from cknn_init import c_num_files_per_batch
from cknn_init import c_num_ks
from cknn_init import c_kernel_size
from cknn_init import c_num_files_for_queries
from cknn_init import c_num_batches
from cknn_init import c_num_centroids
from cknn_init import c_kmeans_iters
from cknn_init import c_num_db_segs
from cknn_init import cb_test_by_knn
from cknn_init import c_signif_centroid_thresh

flg_debug = False

try:
	opts, args = getopt.getopt(sys.argv[1:], "d")
except getopt.GetoptError:
	print('cknn_main.py [-d]')
	sys.exit(2)
for opt, arg in opts:
	if opt == '-d':
		flg_debug = True

cknn_init.do_init()

op_data_set_levels = []
ph_image_string_levels = []
t_chunks_img_levels = []
level_up_set_levels = []

W, Ws, op_Ws_read_levels, op_Ws_set_levels = cknn_db.create_Ws(cknn_init.max_level)


for level in range(cknn_init.max_level):
	# create the call graph that creates an image given a string and assigns it to the correct t_data on the levels table
	# The op assigns the data to the table, the ph is the means of getting the file data which is the input argument
	op_data_set, ph_image_string = cknn_db.make_image_cg(level)
	# Store the ph and ops in a level-indexed array.
	op_data_set_levels.append(op_data_set)
	ph_image_string_levels.append(ph_image_string)
	# takes the normalized chunks which are flat (Shpae = [num_chunks, kernel_size], makes an image of them
	# reduces the dim so that you have a 1-d of target_size (3) vectors and down-samples creating a new image
	if level > 0:
		level_up_set_levels.append(cknn_db.level_up(level-1, t_chunks_img_levels[level-1], W))
	# creates the convolution chunks. (This is just the call graph (cg) not the execution)
	# creates cg nodes that build a convoultion chunk of 3x3x3 by shifting and slicing the original array
	# This cg does not put the data into the t_data table
	# it is therefore not really a call graph in its own right but a cg subsection
	t_chunks_img_levels.append(cknn_db.make_db(level))

qstrings, qclasses = cknn_init.get_file_strings(c_num_files_for_queries)

# function created to insert stops in the debugger. Create a stop/breakpoint by inserting a line like
# 				sess.run(t_for_stop)
def stop_reached(datum, tensor):
	if datum.node_name == 't_for_stop': # and tensor > 100:
		return True
	return False

# saver = tf.train.Saver({'dimred':W})
for level in range(cknn_init.max_level):
	num_convs = 0
	for red_level in range(level, cknn_init.max_level):
		# calculate the total number of conv chunks at this level and add to total
		num_convs += w_levels[red_level] * h_levels[red_level]

	t_for_stop = tf.constant(5.0, name='t_for_stop')

	ph_db_norm, op_db_norm_set, v_db_norm = cknn_db.set_db(num_convs, c_num_files_per_batch)
	ph_db_labels, op_db_labels_set, v_db_labels = cknn_db.set_labels(num_convs, c_num_files_per_batch)
	ph_q, op_q_set, v_q = cknn_db.set_q(num_convs)

	v_centroids, op_centroids_init \
		= cknn_kmeans.make_per_batch_init_cg(num_convs, v_db_norm, c_num_centroids)

	# Get the centroids variable ready. Must persist between loops. Shape=[c_num_convs*c_num_files_per_batch, c_kernel_size]
	v_all_centroids = tf.Variable(tf.zeros([c_num_centroids * c_num_batches, c_kernel_size], dtype=tf.float32), name='v_centroids')
	ph_all_centroids = tf.placeholder(dtype=tf.float32, shape=[c_num_centroids * c_num_batches, c_kernel_size], name='ph_all_centroids')
	op_all_centroids_set = tf.assign(v_all_centroids, ph_all_centroids, name='op_all_centroids_set')

	ph_random_centroids = tf.placeholder(dtype=tf.float32, shape=[c_num_centroids * c_num_batches, c_kernel_size], name='ph_random_centroids')
	ph_centroid_sums  = tf.placeholder(dtype=tf.float32, shape=[c_num_centroids * c_num_batches, c_kernel_size], name='ph_centroid_sums')
	ph_count_sums  = tf.placeholder(dtype=tf.float32, shape=[c_num_centroids * c_num_batches, c_kernel_size], name='ph_count_sums')
	t_new_centroids = tf.where(ph_count_sums > 0.0, ph_centroid_sums/ph_count_sums, ph_random_centroids)
	op_all_centroids_norm_set = tf.assign(v_all_centroids, tf.nn.l2_normalize(t_new_centroids, dim=1), name='op_all_centroids_norm_set')
	# op_all_centroids_norm_set = tf.assign(v_all_centroids, tf.nn.l2_normalize(ph_all_centroids, dim=1), name='op_all_centroids_norm_set')

	# op_closest_idxs_set = cknn_kmeans.make_closest_idxs_cg(num_convs, v_db_norm, num_centroids)
	# cg to create the closest_idxs for one segment of one batch of the v_db
	ph_i_db_seg, t_closest_idxs_seg, ph_closest_idxs, op_closest_idxs_set, v_closest_idxs \
		= cknn_kmeans.make_closest_idxs_cg(	v_db_norm, v_all_centroids,
											num_db_seg_entries = c_num_files_per_batch * num_convs / c_num_db_segs,
											num_tot_db_entries = c_num_files_per_batch * num_convs)
	# Create cg that calculates the votes for just one centroid, must be fed the index of the centroid to calculate for
	ph_i_centroid, t_avg, t_vote_count, t_vote_for_this \
		= cknn_kmeans.vote_for_centroid_cg(	v_db_norm, v_closest_idxs,
											num_tot_db_entries = num_convs * c_num_files_per_batch)

	t_kmeans_err, op_centroids_update, ph_new_centroids, ph_votes_count, t_votes_count \
		= cknn_kmeans.update_centroids_cg(	v_db_norm, v_all_centroids, v_closest_idxs,
											num_tot_db_entries = num_convs * c_num_batches,
											num_centroids = c_num_centroids * c_num_batches)

	# t_kmeans_err, op_centroids_update, t_votes_count \
	# 	= cknn_kmeans.make_kmeans_cg(num_convs, v_db_norm, c_num_centroids * c_num_batches, v_all_centroids)

	t_err, train_step = cknn_db.make_dim_reduce_cg(v_all_centroids, W, num_centroids=c_num_centroids*c_num_batches)

	t_knn_prediction = cknn_knn.make_knn_cg(v_q, v_db_norm, v_db_labels, cknn_init.num_classes, num_convs, c_num_ks)

	sess = tf.Session()
	if flg_debug:
		sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="curses")
		sess.add_tensor_filter("stop_reached", stop_reached)
	merged = tf.summary.merge_all()
	summaries_dir = '/tmp'
	train_writer = tf.summary.FileWriter(summaries_dir + '/train',
										  sess.graph)
	sess.run(tf.global_variables_initializer())

	# Sets the dimension reduction matrix for this whole run to be the one created in the last run
	# In the first run (level = 0) W is not used
	if level > 0:
		sess.run(op_Ws_set_levels[level-1])

	# We loop once over the batch creating instances of v_db_norm just to create the initial random centroids
	batch_iclasses, batch_ifiles = cknn_init.get_file_lists(c_num_batches, c_num_files_per_batch)
	nd_all_controids = np.zeros([c_num_centroids * c_num_batches, c_kernel_size], dtype=np.float32)
	for ibatch in range(c_num_batches):
		db_labels = cknn_control.create_db(	sess, num_convs, op_data_set_levels, ph_image_string_levels,
											t_chunks_img_levels, op_db_norm_set, ph_db_norm, level,
											level_up_set_levels, batch_iclasses[ibatch], batch_ifiles[ibatch])
		# assign the labels to the Tensor Variable. Not din in the create_db because labels are not always needed
		sess.run(op_db_labels_set, feed_dict={ph_db_labels:db_labels })
		nd_all_controids[ibatch*c_num_centroids:(ibatch+1)*c_num_centroids] = sess.run(op_centroids_init)
		print('building initial db. ibatch=', ibatch)
	sess.run(op_all_centroids_set, feed_dict={ph_all_centroids:nd_all_controids})

	for iter_kmeans in range(c_kmeans_iters):
		l_centroid_avgs = []
		l_centroid_counts = []
		l_kmeans_err = []
		for ibatch in range(c_num_batches):
			db_labels = cknn_control.create_db(	sess, num_convs, op_data_set_levels, ph_image_string_levels,
												t_chunks_img_levels, op_db_norm_set, ph_db_norm, level,
												level_up_set_levels, batch_iclasses[ibatch], batch_ifiles[ibatch])
			# assign the labels to the Tensor Variable. Not din in the create_db because labels are not always needed
			sess.run(op_db_labels_set, feed_dict={ph_db_labels:db_labels })
			for iseg in range(c_num_db_segs):
				n1 = sess.run(t_closest_idxs_seg, feed_dict={ph_i_db_seg:iseg})
				if iseg == 0:
					nd_closest_idxs = n1
				else:
					nd_closest_idxs = np.concatenate([nd_closest_idxs, n1], axis=0)
			sess.run(op_closest_idxs_set, feed_dict={ph_closest_idxs:nd_closest_idxs})
			nd_new_centroids = np.ndarray(dtype = np.float32, shape = [c_num_centroids * c_num_batches, c_kernel_size])
			nd_votes_count = np.ndarray(dtype = np.float32, shape = [c_num_centroids * c_num_batches])
			for icent in range(c_num_centroids * c_num_batches):
				r_cent_avg, r_cent_vote_count = sess.run([t_avg, t_vote_count], feed_dict={ph_i_centroid:icent})
				nd_new_centroids[icent, : ]  = r_cent_avg
				nd_votes_count[icent] = r_cent_vote_count
			r_votes_count, r_centroids, r_kmeans_err \
				= sess.run(	[t_votes_count, op_centroids_update, t_kmeans_err],
							feed_dict={ph_new_centroids:nd_new_centroids, ph_votes_count:nd_votes_count})
			l_centroid_avgs.append(r_centroids)
			l_centroid_counts.append(r_votes_count)
			l_kmeans_err.append(r_kmeans_err)
			print('building kmeans db. ibatch=', ibatch)
		np_centroid_avgs = np.stack(l_centroid_avgs)
		np_centroid_counts = np.stack(l_centroid_counts)
		np_count_sums = np.tile(np.expand_dims(np.sum(np_centroid_counts, axis=0), axis=-1), reps=[1, c_kernel_size])
		np_br_centroid_counts = np.tile(np.expand_dims(np_centroid_counts, axis=-1), reps=[1, c_kernel_size])
		np_centroid_facs = np.multiply(np_centroid_avgs, np_br_centroid_counts)
		np_centroid_sums = np.sum(np_centroid_facs, axis=0)
		# np_default_avgs = np.squeeze(np.take(np_centroid_avgs, np.random.random_integers(0, c_num_batches-1, size=1), axis=0))
		# np_non_zero = np_count_sums>0
		# n1 = np.take(np_centroid_sums, )
		# np_new_centroids = np_centroid_sums[np_non_zero] / np_count_sums[np_non_zero]
		# np_new_centroids = np.where(np_count_sums>0, np_centroid_sums / np_count_sums, np_default_avgs)
		# r_centroids = sess.run(op_all_centroids_norm_set, feed_dict={ph_all_centroids:np_new_centroids})
		np.random.shuffle(nd_all_controids)
		r_centroids = sess.run(	op_all_centroids_norm_set,
								feed_dict={	ph_random_centroids: nd_all_controids,
											ph_centroid_sums: np_centroid_sums,
											ph_count_sums: np_count_sums})
		print('kmeans iter:', iter_kmeans, 'kmeans err:', np.mean(np.stack(l_kmeans_err)))

	l_nd_clusters = []
	l_nd_cluster_labels = []
	l_votes_count = []

	for ibatch in range(c_num_batches):
		db_labels = cknn_control.create_db(sess, num_convs, op_data_set_levels, ph_image_string_levels,
										   t_chunks_img_levels, op_db_norm_set, ph_db_norm, level,
										   level_up_set_levels, batch_iclasses[ibatch], batch_ifiles[ibatch])
		# assign the labels to the Tensor Variable. Not din in the create_db because labels are not always needed
		sess.run(op_db_labels_set, feed_dict={ph_db_labels: db_labels})
		r_db_norm = sess.run(v_db_norm)
		for iseg in range(c_num_db_segs):
			n1 = sess.run(t_closest_idxs_seg, feed_dict={ph_i_db_seg: iseg})
			if iseg == 0:
				nd_closest_idxs = n1
			else:
				nd_closest_idxs = np.concatenate([nd_closest_idxs, n1], axis=0)
		sess.run(op_closest_idxs_set, feed_dict={ph_closest_idxs:nd_closest_idxs})
		for icent in range(c_num_centroids * c_num_batches):
			r_vote_for_this, r_cent_vote_count = sess.run([t_vote_for_this, t_vote_count], feed_dict={ph_i_centroid:icent})
			if ibatch == 0:
				l_nd_clusters.append(r_db_norm[r_vote_for_this])
				l_nd_cluster_labels.append(db_labels[r_vote_for_this])
				l_votes_count.append(r_cent_vote_count)
			else:
				l_nd_clusters[icent] = np.concatenate([l_nd_clusters[icent], r_db_norm[r_vote_for_this]], axis=0)
				l_nd_cluster_labels[icent] = np.concatenate([l_nd_cluster_labels[icent], db_labels[r_vote_for_this]])
				l_votes_count[icent] += r_cent_vote_count
		# nd_new_centroids[icent, : ]  = r_cent_avg
	nd_signif_centroids = np.empty([c_num_centroids * c_num_batches], dtype=np.bool )
	for icent in range(c_num_centroids * c_num_batches):
		nd_cluster_votes = np.bincount(l_nd_cluster_labels[icent])
		winner_idx = np.argmax(nd_cluster_votes)
		winner_count = nd_cluster_votes[winner_idx]
		nd_signif_centroids[icent] = winner_count / l_votes_count[icent] > c_signif_centroid_thresh

	nd_sig_centroids_idxs = np.arange(c_num_centroids * c_num_batches)[nd_signif_centroids]
	nd_lowest_CD = np.zeros([nd_sig_centroids_idxs.size])
	for ii, isig in enumerate(nd_sig_centroids_idxs):
		cluster = l_nd_clusters[isig]
		centroid = r_centroids[isig]
		cluster_CDs = np.dot(cluster, np.transpose(centroid ))
		nd_lowest_CD[ii] = np.amin(cluster_CDs)

	# Get query image, create conv chunks
	# for each chunk, for each sig centroid, find out if the CD is above the min membership CD
	qstrings, qclasses = cknn_init.get_file_strings(1)
	q_els = cknn_control.create_db_els(	sess, num_convs, op_data_set_levels,
										ph_image_string_levels, t_chunks_img_levels,
										level, level_up_set_levels, qstrings[0])
	nd_q_CDs = np.dot(r_centroids, np.transpose(q_els))

	# cknn_control.create_clusters(sess, op_centroids_init, op_closest_idxs_set, t_votes_count, t_kmeans_err, op_centroids_update)

	# Assign (copy) the level W to the single W expected in the learning of a dim reduction transformation
	sess.run(op_Ws_set_levels[level])
	# Execute the learning
	cknn_control.reduce_cluster_dims(sess, t_err, train_step)
	# copy the data back to the level table
	sess.run(op_Ws_read_levels[level])

	if cb_test_by_knn:
		q_els = np.zeros([num_convs, c_kernel_size], dtype=np.float32)
		# q_labels = np.repeat(qclasses, num_convs)
		num_hits = 0.0
		for ibatch in range(c_num_files_for_queries):
			q_els = cknn_control.create_db_els(	sess, num_convs, op_data_set_levels,
												ph_image_string_levels, t_chunks_img_levels,
												level, level_up_set_levels, qstrings[ibatch])
			sess.run(op_q_set, feed_dict={ph_q:q_els})
			r_pred = sess.run(t_knn_prediction)
			if r_pred == qclasses[ibatch]:
				num_hits += 1.0

		print('success rate for level:', level, 'is', num_hits / float(c_num_files_for_queries))

	# save_path = saver.save(sess, os.path.join(LogDir, LogFile))
	# print("Model saved in file: %s" % save_path)


	sess.close()


