from __future__ import print_function
import tensorflow as tf

import cknn_init
from cknn_init import w_levels
from cknn_init import h_levels
from cknn_init import t_data_levels
from cknn_init import c_kernel_size
from cknn_init import c_target_dim
from cknn_init import c_num_centroids
from cknn_init import c_dimred_num_rands
from cknn_init import c_dimred_learn_rate


# max_levels instances of this function/cg are called/created
def make_image_cg(level):
	# Decode string into matrix with intensity values
	ph_image_string = tf.placeholder(dtype=tf.string, shape=(), name='ph_image_string')
	image = tf.image.decode_jpeg(ph_image_string, channels=3)
	image = tf.expand_dims(image, 0)
	image = tf.image.resize_bilinear(image, [h_levels[level]+2, w_levels[level]+2])[0]
	return tf.assign(t_data_levels[level], image, name='op_data_set'), ph_image_string

# max_levels instances of this function/cg are called/created
def make_level_conv(level):
	convarr = []
	for r in range(3):
		for c in range(3):
			convarr.append(t_data_levels[level][r:r + h_levels[level], c:c + w_levels[level], : ])
	# stack the array. Shape=[h_levels[level], w_levels[level], 3, 9]
	t_stacked = tf.stack(convarr, axis=3, name='t_stacked')
	return t_stacked

# max_levels instances of this function/cg are called/created
def make_db(level):
	# reshape so that the 3 channel image becomes a 2D image with a 27D vector at every pixel
	# For now the shape is not really that of an image, we combine dims 0 and 1 so that mul can work later
	# shape=[img_h * img_w, c_kernel_size]
	t_chunks_img_unnorm = tf.reshape(make_level_conv(level), [h_levels[level] * w_levels[level], c_kernel_size], name='t_chunks_img_unnorm')
	# l2 normalize. Each chunk now has square of properties sum to 1. This captures the structure rather than ans values
	return tf.nn.l2_normalize(t_chunks_img_unnorm, dim=1, name='t_chunks_img')

# level here is src level not dest level
# You have to be careful here with the W that is passed in. So if you pass in W,
# when you create a call graph it is bound to a specific tensor
# the python variable can switch which tensor it is referring to
# So you must either pass in the specific W for that level or assign to a single W before executing this call graph
# max_levels instances of this function/cg are called/created
def level_up(level, t_chunks_img, W):
	# # W converts the 27D values to 3D
	# W = tf.Variable(tf.random_normal([c_kernel_size, c_target_dim], 0.01 / float(c_target_dim * c_kernel_size)),
	# 				name='W')
	# convert image to micro-chunks. Bring the shape back to that of an image
	# Shape=[img_h, img_w, c_target_dim]
	t_mchunk_img = tf.reshape(tf.matmul(t_chunks_img, W), shape=[h_levels[level], w_levels[level], c_target_dim],
							  name='t_mchunk_img')
	# Down sample the image by discarding 75% of the information
	# Shape=[img_h/2, img_w/2, c_target_dim]
	t_sampled_img = tf.strided_slice(t_mchunk_img, begin=[0, 0, 0], end=[h_levels[level], w_levels[level], c_target_dim],
									 strides=[2, 2, 1], name='t_sampled_img')

	# num_convs = h_levels[1] * w_levels[1]

	return tf.assign(	t_data_levels[level+1], t_sampled_img,
						name='level_up_set')


def set_db(num_convs, num_files_per_batch):
	# variable (and stop for call graph) holding the db that knn queries will run against.
	# set by assign from op_db_norm_set
	v_db_norm = tf.Variable(tf.zeros(shape=[num_convs * num_files_per_batch, c_kernel_size], dtype=tf.float32),
							name='v_db_norm')
	# place holder for the database, copied in from a numpy array built up from the convs of each image.
	ph_db_norm = tf.placeholder(dtype=tf.float32, shape=[num_convs * num_files_per_batch, c_kernel_size],
								name='ph_db_norm')
	# op to put placeholder value into database
	op_db_norm_set = tf.assign(v_db_norm, ph_db_norm, name='op_db_norm_set')

	return ph_db_norm, op_db_norm_set, v_db_norm

def set_labels(num_convs, num_files_per_batch):
	# variable for lables
	v_db_labels = tf.Variable(tf.zeros(shape=[num_convs * num_files_per_batch], dtype=tf.int32), name='v_db_labels')
	# place holder for the labels which are just the classes of the image repeated along all the convs of each image
	ph_db_labels = tf.placeholder(dtype=tf.int32, shape=[num_convs * num_files_per_batch], name='ph_db_labels')
	# op to put placeholder value into label variable
	op_db_labels_set = tf.assign(v_db_labels, ph_db_labels, name='op_db_labels_set')

	return ph_db_labels, op_db_labels_set, v_db_labels

def set_q(num_convs):
	# variable for query
	v_q = tf.Variable(tf.zeros(shape=[num_convs, c_kernel_size], dtype=tf.float32), name='v_q')
	# place holder for the query which is an array of all of its convs, each of shape=[c_kernel_size]
	ph_q = tf.placeholder(dtype=tf.float32, shape=[num_convs, c_kernel_size], name='ph_q')
	# op to put placeholder value into query variable
	op_q_set = tf.assign(v_q, ph_q, name='op_q_set')

	return ph_q, op_q_set, v_q

# W converts the 27D values to 3D. Shape=[c_kernel_size, c_target_size]
# centroids look like conv chunks (Shape=[c_kernel_size]) but there are only c_num_centroids of them
# there is only one instance of this function/cg created, so W is shared
def make_dim_reduce_cg(v_centroids, W, num_centroids):
	# The follwing code is the call graph for dimension reduction
	# We take two random vectors from the centroids, create a matrix of cosine distances between them
	t_r1 = tf.random_uniform([c_dimred_num_rands], 0, num_centroids - 1, dtype=tf.int32, name='t_r1')
	t_r2 = tf.random_uniform([c_dimred_num_rands], 0, num_centroids - 1, dtype=tf.int32, name='t_r2')
	# gather to get the randomly selected centroids themselves
	# Shape=[c_num_rands, c_kernel_size]
	t_l1 = tf.gather(v_centroids, t_r1, name='t_l1')
	t_l2 = tf.gather(v_centroids, t_r2, name='t_l2')
	# Calculate the cosine distance between each of one vector and each of the other
	# Shape=[c_num_rands, c_num_rands]
	t_ldist = tf.matmul(t_l1, t_l2, transpose_b=True, name='t_ldist')
	# transform each  set of centroids using W and calculate the distances between the same points we did before but on 3D
	t_t1 = tf.matmul(t_l1, W, name='t_t1')
	t_t2 = tf.matmul(t_l2, W, name='t_t2')
	t_tdist = tf.matmul(t_t1, t_t2, transpose_b=True, name='t_tdist')
	# Define the error as the difference between the pre-transformed and post-transfomed
	t_err = tf.reduce_mean(tf.square(t_ldist - t_tdist), name='t_err')
	# create a trainer
	train_step = tf.train.AdagradOptimizer(c_dimred_learn_rate).minimize(t_err)
	return t_err, train_step

def create_Ws(num_levels):
	W = tf.Variable(tf.random_normal([c_kernel_size, c_target_dim], 0.01 / float(c_target_dim * c_kernel_size)),
					name='W')
	Ws = []
	op_Ws_set_levels = []
	op_Ws_read_levels = []
	for level in range(num_levels):
		levelW = tf.Variable(	tf.random_normal(	[c_kernel_size, c_target_dim],
													0.01 / float(c_target_dim * c_kernel_size)),
								name='W')
		op_Ws_set_levels.append(tf.assign(levelW, W, name='op_Ws_set'))
		op_Ws_read_levels.append(tf.assign(W, levelW, name='op_Ws_read'))
		Ws.append(levelW)
	return W, Ws, op_Ws_read_levels, op_Ws_set_levels

