from __future__ import print_function
import tensorflow as tf

import cknn_init
from cknn_init import c_kernel_size
# from cknn_init import c_num_centroids
from cknn_init import c_num_files_per_batch
from cknn_init import c_num_db_segs


def make_per_batch_init_cg(num_convs, v_db_norm, num_centroids):
	# The goal is to cluster the convolution vectors so that we can perform dimension reduction
	# KMeans implementation
	# Intitialize the centroids indicies. Shape=[num_centroids]
	t_centroids_idxs_init = tf.random_uniform([num_centroids], 0, num_convs - 1, dtype=tf.int32,
											  name='t_centroids_idxs_init')
	# Get the centroids variable ready. Must persist between loops. Shape=[c_num_convs*c_num_files_per_batch, c_kernel_size]
	v_centroids = tf.Variable(tf.zeros([num_centroids, c_kernel_size], dtype=tf.float32), name='v_centroids')
	# Create actual centroids as seeds. Shape=[num_centroids, c_kernel_size]
	op_centroids_init = tf.assign(v_centroids, tf.gather(v_db_norm, t_centroids_idxs_init, name='op_centroids_init'))

	return v_centroids, op_centroids_init


def make_closest_idxs_cg(v_db_norm, v_all_centroids, num_db_seg_entries, num_tot_db_entries):
	ph_i_db_seg = tf.placeholder(dtype=tf.int32, shape=(), name='ph_i_db_seg')
	# Create actual centroids as seeds. Shape=[num_centroids, c_kernel_size]
	# op_centroids_init = tf.assign(v_centroids, tf.gather(v_db_norm, t_centroids_idxs_init, name='op_centroids_init'))
	# Do cosine distances for all centroids on all elements of the db. Shape [num_centroids, num_db_seg_entries]
	t_all_CDs = tf.matmul(v_all_centroids, v_db_norm[ph_i_db_seg*num_db_seg_entries:(ph_i_db_seg+1)*num_db_seg_entries, :], transpose_b=True, name='t_all_CDs')
	# For each entry in the chunk database, find the centroid that's closest.
	# Basically, we are finding which centroid had the highest cosine distance for each entry of the chunk db
	# This holds the index to the centroid which we can then use to create an average among the entries that voted for it
	# Shape=[num_db_seg_entries]
	t_closest_idxs_seg = tf.argmax(t_all_CDs, axis=0, name='t_closest_idxs_seg')
	# unconnected piece of cg building. Create a way of assigning np complete array back into the tensor Variable
	# code remains here because it would  be nice to replace with an in-graph assignment like TensorArray
	ph_closest_idxs = tf.placeholder(	shape=[num_tot_db_entries], dtype=tf.int32,
										name='ph_closest_idxs')
	v_closest_idxs = tf.Variable(tf.zeros(shape=[num_tot_db_entries], dtype=tf.int32),
								 name='v_closest_idxs')
	op_closest_idxs_set = tf.assign(v_closest_idxs, ph_closest_idxs, name='op_closest_idxs_set')

	return ph_i_db_seg, t_closest_idxs_seg, ph_closest_idxs, op_closest_idxs_set, v_closest_idxs

# Find the vote count and create an average for a single centroid
def vote_for_centroid_cg(v_db_norm, v_closest_idxs, num_tot_db_entries):
	# create placehoder to tell the call graph which iteration, i.e. which centroid we are working on
	ph_i_centroid = tf.placeholder(dtype=tf.int32, shape=(), name='ph_i_centroid')
	# Create an array of True if the closest index was this centroid
	# Shape=[num_centroids]
	t_vote_for_this = tf.equal(v_closest_idxs, ph_i_centroid, name='t_vote_for_this')
	# Count the number of trues in the vote_for_tis tensor
	# Shape=()
	t_vote_count = tf.reduce_sum(tf.cast(t_vote_for_this, tf.float32), name='t_vote_count')
	# Create the cluster. Use the True positions to put in the values from the v_db_norm and put zeros elsewhere.
	# This means that instead of a short list of the vectors in this cluster we use the full size with zeros for non-members
	# Shape=[num_tot_db_entries, c_kernel_size]
	t_this_cluster = tf.where(t_vote_for_this, v_db_norm,
							  tf.zeros([num_tot_db_entries, c_kernel_size]), name='t_this_cluster')
	# Sum the values for each property to get the aveage property
	# Shape=[c_kernel_size]
	t_cluster_sum = tf.reduce_sum(t_this_cluster, axis=0, name='t_cluster_sum')
	# Shape=[c_kernel_size]
	t_avg = tf.cond(t_vote_count > 0.0,
					lambda: tf.divide(t_cluster_sum, t_vote_count),
					lambda: tf.zeros([c_kernel_size]),
					name='t_avg')

	return ph_i_centroid, t_avg, t_vote_count, t_vote_for_this


def update_centroids_cg(v_db_norm, v_all_centroids, v_closest_idxs, num_tot_db_entries, num_centroids):
	ph_new_centroids = tf.placeholder(dtype=tf.float32, shape=[num_centroids, c_kernel_size], name='ph_new_centroids')
	ph_votes_count = tf.placeholder(dtype=tf.float32, shape=[num_centroids], name='ph_votes_count')
	# Do random centroids again. This time for filling in
	t_centroids_idxs = tf.random_uniform([num_centroids], 0, num_tot_db_entries - 1, dtype=tf.int32, name='t_centroids_idxs')
	# Shape = [num_centroids, c_kernel_size]
	# First time around I forgot that I must normalize the centroids as required for shperical k-means. Avg, as above, will not produce a normalized result
	t_new_centroids_norm = tf.nn.l2_normalize(ph_new_centroids, dim=1, name='t_new_centroids_norm')
	# Shape=[num_centroids]
	t_votes_count = ph_votes_count
	# take the new random idxs and gather new centroids from the db. Only used in case count == 0. Shape=[num_centroids, c_kernel_size]
	t_centroids_from_idxs = tf.gather(v_db_norm, t_centroids_idxs, name='t_centroids_from_idxs')
	# Assign back to the original v_centroids so that we can go for another round
	op_centroids_update = tf.assign(v_all_centroids, tf.where(tf.greater(ph_votes_count, 0.0), t_new_centroids_norm,
														  t_centroids_from_idxs, name='centroids_where'),
									name='op_centroids_update')

	# The following section of code is designed to evaluate the cluster quality, specifically the average distance of a conv fragment from
	# its centroid.
	# t_closest_idxs is an index for each element in the database, specifying which cluster it belongs to. So we use that to
	# replicate the centroid of that cluster to the locations alligned with each member of the database
	# Shape=[num_tot_db_entries, c_kernel_size]
	t_centroid_broadcast = tf.gather(v_all_centroids, v_closest_idxs, name='t_centroid_broadcast')
	# element-wise multiplication of each property and the sum down the properties. It is reallt just a CD but we aren't using matmul
	# Shape=[num_tot_db_entries]
	t_cent_dist = tf.reduce_sum(tf.multiply(v_db_norm, t_centroid_broadcast), axis=1, name='t_cent_dist')
	# Extract a single number representing the kmeans error. This is the mean of the distances from closest centers. Shape=()
	t_kmeans_err = tf.reduce_mean(t_cent_dist, name='t_kmeans_err')
	return t_kmeans_err, op_centroids_update, ph_new_centroids, ph_votes_count, t_votes_count



# Create centroids from the v_db_norm and iterate a few times. Assunmes only c_nuum
def make_kmeans_cg(num_convs, v_db_norm, num_centroids, v_all_centroids):
	# The goal is to cluster the convolution vectors so that we can perform dimension reduction
	# KMeans implementation
	# Intitialize the centroids indicies. Shape=[num_centroids]
	t_centroids_idxs_init = tf.random_uniform([num_centroids], 0, num_convs - 1, dtype=tf.int32,
											  name='t_centroids_idxs_init')
	# Get the centroids variable ready. Must persist between loops. Shape=[c_num_convs*c_num_files_per_batch, c_kernel_size]
	# v_centroids = tf.Variable(tf.zeros([num_centroids, c_kernel_size], dtype=tf.float32), name='v_centroids')
	# Create actual centroids as seeds. Shape=[num_centroids, c_kernel_size]
	# op_centroids_init = tf.assign(v_centroids, tf.gather(v_db_norm, t_centroids_idxs_init, name='op_centroids_init'))
	# Do cosine distances for all centroids on all elements of the db. Shape [num_centroids, c_num_convs*c_num_files_per_batch]
	t_all_CDs = tf.matmul(v_all_centroids, v_db_norm, transpose_b=True, name='t_all_CDs')
	# Do top_k. Shape = [num_centroids, c_num_ks]
	## t_best_CDs, t_best_CD_idxs = tf.nn.top_k(t_all_CDs, c_num_ks, sorted=True, name='t_best_CD_idxs')
	# For each element in the matrix, find the centroid that's closest. Shape=[c_num_convs*c_num_files_per_batch]
	v_closest_idxs = tf.Variable(tf.zeros(shape=[num_convs * c_num_files_per_batch], dtype=tf.int64),
								 name='v_closest_idxs')
	op_closest_idxs_set = tf.assign(v_closest_idxs, tf.argmax(t_all_CDs, axis=0), name='op_closest_idxs_set')
	l_new_centroids = []
	l_votes_count = []
	for icent in range(num_centroids):
		# Create an array of True if the closest index was this centroid
		# Shape=[num_centroids]
		t_vote_for_this = tf.equal(v_closest_idxs, icent, name='t_vote_for_this')
		# Count the number of trues in the vote_for_tis tensor
		# Shape=()
		t_vote_count = tf.reduce_sum(tf.cast(t_vote_for_this, tf.float32), name='t_vote_count')
		# Create the cluster. Use the True positions to put in the values from the v_db_norm and put zeros elsewhere.
		# This means that instead of a short list of the vectors in this cluster we use the full size with zeros for non-members
		# Shape=[c_num_convs*c_num_files_per_batch, c_kernel_size]
		t_this_cluster = tf.where(t_vote_for_this, v_db_norm,
								  tf.zeros([num_convs * c_num_files_per_batch, c_kernel_size]), name='t_this_cluster')
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
	t_centroids_idxs = tf.random_uniform([num_centroids], 0, num_convs - 1, dtype=tf.int32, name='t_centroids_idxs')
	# Shape = [num_centroids, c_kernel_size]
	t_new_centroids = tf.stack(l_new_centroids, name='t_new_centroids')
	# First time around I forgot that I must normalize the centroids as required for shperical k-means. Avg, as above, will not produce a normalized result
	t_new_centroids_norm = tf.nn.l2_normalize(t_new_centroids, dim=1, name='t_new_centroids_norm')
	# Shape=[num_centroids]
	t_votes_count = tf.stack(l_votes_count, name='t_votes_count')
	# take the new random idxs and gather new centroids from the db. Only used in case count == 0. Shape=[num_centroids, c_kernel_size]
	t_centroids_from_idxs = tf.gather(v_db_norm, t_centroids_idxs, name='t_centroids_from_idxs')
	# Assign back to the original v_centroids so that we can go for another round
	op_centroids_update = tf.assign(v_all_centroids, tf.where(tf.greater(t_votes_count, 0.0), t_new_centroids_norm,
														  t_centroids_from_idxs, name='centroids_where'),
									name='op_centroids_update')

	# The following section of code is designed to evaluate the cluster quality, specifically the average distance of a conv fragment from
	# its centroid.
	# t_closest_idxs is an index for each element in the database, specifying which cluster it belongs to. So we use that to
	# replicate the centroid of that cluster to the locations alligned with each member of the database
	# Shape=[c_num_convs*c_num_files_per_batch, c_kernel_size]
	t_centroid_broadcast = tf.gather(v_all_centroids, v_closest_idxs, name='t_centroid_broadcast')
	# element-wise multiplication of each property and the sum down the properties. It is reallt just a CD but we aren't using matmul
	# Shape=[c_num_convs*c_num_files_per_batch]
	t_cent_dist = tf.reduce_sum(tf.multiply(v_db_norm, t_centroid_broadcast), axis=1, name='t_cent_dist')
	# Extract a single number representing the kmeans error. This is the mean of the distances from closest centers. Shape=()
	t_kmeans_err = tf.reduce_mean(t_cent_dist, name='t_kmeans_err')
	return t_kmeans_err, op_centroids_update, t_votes_count
