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
import cknn_db
import cknn_kmeans
import cknn_control

from cknn_init import w_levels
from cknn_init import h_levels
from cknn_init import t_data_levels
# from cknn_init import max_level
from cknn_init import c_num_files_per_batch
from cknn_init import c_kernel_size
from cknn_init import c_kmeans_iters
from cknn_init import c_dimred_num_iters

cknn_init.do_init()

op_data_set_levels = []
ph_image_string_levels = []
op_data_level_set_levels = []
t_chunks_img_levels = []

num_convs = 0
for level in range(cknn_init.max_level):
	op_data_set, ph_image_string = cknn_db.make_image_cg(level)
	op_data_set_levels.append(op_data_set)
	ph_image_string_levels.append(ph_image_string)
	num_convs += w_levels[level] * h_levels[level]
	t_chunks_img_levels.append(cknn_db.make_db(level))

ph_db_norm, op_db_norm_set, v_db_norm = cknn_db.set_db(num_convs, c_num_files_per_batch)

t_kmeans_err, v_centroids, op_closest_idxs_set, op_centroids_init, op_centroids_update, t_votes_count \
	= cknn_kmeans.make_kmeans_cg(num_convs, v_db_norm)

t_err, train_step, W = cknn_db.make_dim_reduce_cg(v_centroids)

sess = tf.Session()
merged = tf.summary.merge_all()
summaries_dir = '/tmp'
train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                      sess.graph)
sess.run(tf.global_variables_initializer())

cknn_control.create_db(	sess, num_convs, op_data_set_levels, ph_image_string_levels,
						t_chunks_img_levels, op_db_norm_set, ph_db_norm)

cknn_control.create_clusters(sess, op_centroids_init, op_closest_idxs_set, t_votes_count, t_kmeans_err, op_centroids_update)

cknn_control.reduce_cluster_dims(sess, t_err, train_step)

sess.close()


