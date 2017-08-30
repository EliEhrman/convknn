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

import cknn_init
from cknn_init import w_levels
from cknn_init import h_levels
# from cknn_init import max_level
from cknn_init import c_num_files_per_batch
from cknn_init import c_kernel_size
from cknn_init import c_kmeans_iters
from cknn_init import c_dimred_num_iters

def create_db_els(	sess, num_convs, op_data_set_levels, ph_image_string_levels, t_chunks_img_levels,
					target_level, level_up_set_levels, string_el):
	els = np.zeros([num_convs, c_kernel_size], dtype=np.float32)
	start = 0
	for level in range(cknn_init.max_level-target_level):
		sess.run(op_data_set_levels[level], feed_dict={ph_image_string_levels[level]:string_el})
		for target in range(target_level):
			sess.run(level_up_set_levels[target+level])
		end = start + w_levels[level+target_level] * h_levels[level+target_level]
		els[start:end] = sess.run(t_chunks_img_levels[level+target_level])
		start = end
	return els

def create_db(	sess, num_convs, op_data_set_levels, ph_image_string_levels, t_chunks_img_levels, op_db_norm_set,
				ph_db_norm, target_level, level_up_set_levels, iclasses, ifiles):
	strings, classes = cknn_init.get_file_strings_by_list(iclasses, ifiles)
	db_els = np.zeros([num_convs*c_num_files_per_batch, c_kernel_size], dtype=np.float32)
	db_labels = np.repeat(classes, num_convs).astype(np.int32)
	for ibatch in range(c_num_files_per_batch):
		db_els[ibatch*num_convs:(ibatch+1)*num_convs] \
			= create_db_els(	sess, num_convs, op_data_set_levels,
								ph_image_string_levels, t_chunks_img_levels,
								target_level, level_up_set_levels, strings[ibatch])
	r_db_norm = sess.run(op_db_norm_set, feed_dict={ph_db_norm:db_els })

	# print('db created.')
	return db_labels

"""
def create_db(	sess, num_convs, op_data_set_levels, ph_image_string_levels, t_chunks_img_levels, op_db_norm_set,
				ph_db_norm, target_level, level_up_set_levels):
	strings, classes = cknn_init.get_file_strings(c_num_files_per_batch)
	db_els = np.zeros([num_convs*c_num_files_per_batch, c_kernel_size], dtype=np.float32)
	db_labels = np.repeat(classes, num_convs)
	for ibatch in range(c_num_files_per_batch):
		start = ibatch * num_convs
		for level in range(cknn_init.max_level-target_level):
			sess.run(op_data_set_levels[level], feed_dict={ph_image_string_levels[level]:strings[ibatch]})
			for target in range(target_level):
				sess.run(level_up_set_levels[target+level])
			end = start + w_levels[level+target_level] * h_levels[level+target_level]
			db_els[start:end] = sess.run(t_chunks_img_levels[level+target_level])
			start = end
			# sess.run(op_data_level_set_levels[level])

	r_db_norm = sess.run(op_db_norm_set, feed_dict={ph_db_norm:db_els })

	print('db created.')
"""

def create_clusters(sess, op_centroids_init, op_closest_idxs_set, t_votes_count, t_kmeans_err, op_centroids_update):
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

def reduce_cluster_dims(sess, t_err, train_step):
	learn_phase = 0
	for step in range(c_dimred_num_iters+1):
		if step % (c_dimred_num_iters / 100) == 0:
			errval1 = math.sqrt(sess.run(t_err))
			sess.run([train_step])
			errval2 = math.sqrt(sess.run(t_err))
			print('step:', step, ', err1:', errval1, ', err2:', errval2)
		else:
			sess.run([train_step])



