from __future__ import print_function
import tensorflow as tf

import cknn_init
from cknn_init import c_knn_offset

def make_knn_cg(v_q, v_db_norm, v_db_labels, num_classes, num_convs, num_ks):
	# CDs of each conv of the query image against all the convs of all the images. Shape=[num_convs,num_convs*c_num_files_per_batch]
	t_all_CDs = tf.matmul(v_q, v_db_norm, transpose_b=True, name='t_all_CDs')
	# k closest indexes from the db for EACH of the convs of the query image.  Shape=[num_convs, num_ks]
	t_best_CDs, t_best_CD_idxs = tf.nn.top_k(t_all_CDs, num_ks, sorted=True, name='t_best_CD_idxs')
	# convert the indexes of the database to class integers, where each integer represents a differenct class.
	# This is done by using the same indexes to access the labels. Shape=[num_convs, num_ks]
	t_knn_classes = tf.gather(v_db_labels, t_best_CD_idxs)
	# convert the former into one-hot
	t_knn_class_oh = tf.one_hot(t_knn_classes, num_classes, on_value=1.0, off_value=0.0, name='t_knn_class_oh')
	# the basic rank tensor which requires broadcasting before use. Shape=[num_ks]
	t_rank_raw = tf.add(tf.range(num_ks, dtype=tf.float32), c_knn_offset, name='rank_knn')
	t_rank_sum = tf.reduce_sum(tf.reciprocal(t_rank_raw), name='t_rank_sum')
	# Now produce a tensor we can use to weight the knn results. Shape = [num_ks, num_classes]. All we've done is take the [2, 3, 4 ...]
	# vector and turned it into [[2, 2..], [3, 3...], [4, 4...] ...] so that the divide won't complain
	# it's a sort of explicit broadcast needed for shapes that the automatic broadcast can't handle
	# Shape = [num_ks, num_classes]
	t_rank = tf.tile(tf.expand_dims(t_rank_raw, 1), [1, num_classes], name='t_rank')
	# weight the truth values by the rank of the result. i.e. closer neighbors. We are calculating an average truth value.
	# What we have calculated here is the truth value for each element of the batch as predicted by the knn. Shape=[batch_size, num_classes]
	t_knn_truth = tf.divide(tf.reduce_sum(tf.divide(t_knn_class_oh, t_rank), axis=1, name='mean_truth_by_idx'),
							t_rank_sum, name='t_knn_truth')
	# The result of the vote is now just finding who has the biggest weighted average. Shape=[num_convs]
	t_knn_winner = tf.cast(tf.argmax(t_knn_truth, axis=1), dtype=tf.int32, name='t_knn_winner')
	# broadcast the winner so that we can compare to the original t_knn_classes, in order to create a count. Shape=[num_convs, num_ks]
	t_knn_winner_broadcast = tf.tile(tf.expand_dims(t_knn_winner, -1), [1, num_ks], name='t_knn_winner_broadcast')
	# place 1's where the elements are equal. Shape=[num_convs, num_ks]
	# This represents how many actually voted for the winneer rather than the winner himself
	# The importance of this is that if someone has, say, an absolute majority he gets a strong signal. If he scraped by
	# where all classes are about equal, we don't think that is significant
	t_knn_winner_count = tf.where(tf.equal(t_knn_winner_broadcast, t_knn_classes),
								  tf.zeros([num_convs, num_ks], dtype=tf.float32),
								  tf.ones([num_convs, num_ks], dtype=tf.float32), name='t_knn_winner_count')
	# Shape=[num_convs]
	t_knn_imortance_factor = tf.reduce_sum(t_knn_winner_count, axis=1, name='t_knn_imortance_factor')
	# Shape=[num_convs, num_classes]
	t_knn_signal = tf.squeeze(tf.matmul(tf.expand_dims(t_knn_imortance_factor, 0),
										tf.one_hot(t_knn_winner, num_classes, on_value=1.0, off_value=0.0)),
							  name='t_knn_signal')
	return tf.argmax(t_knn_signal, axis=0, name='t_knn_prediction')
