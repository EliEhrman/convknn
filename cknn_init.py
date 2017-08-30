from __future__ import print_function
import os
import tensorflow as tf
import random

fdir = ("/devlink2/data/imagenet/")

c_img_w = 12
c_img_h = 8

img_w = c_img_w - 2 # 12 # 64
img_h = c_img_h - 2 # 8 # 64

c_kernel_size = 3*3*3
c_class_dir_limit = 2000
c_min_file_size = 1000
c_num_files_per_batch = 4 # 2000 # should be a muliple of c_num_db_segs
c_num_files_for_queries = 100
# how much to offset the reciprocal for vote counting. So that the first is 1/c_knn_offset, second 1/(c_knn_offset+1) etc.
# the higher the value the less rank actually matters
c_target_dim = 3

w_levels = []
h_levels = []
t_data_levels = []
classnames = []
all_file_list = []
num_files_in_class = []
num_classes = 0
max_level = 0
# how small are we interested in still working with an image
# image will also be shrunk only up to one level above this
c_min_image_dim = 0
c_num_centroids = 7 # 300
c_kmeans_iters = 4 # 100
c_dimred_learn_rate = 0.05
# How many random samples to use for dim reduce
c_dimred_num_rands = 5
c_dimred_num_iters = 1000
# knn params
c_num_ks = 3 # 32
c_knn_offset = 5.0
c_num_classes = 2
cb_limit_classes = False
# batch processing
c_num_batches = 3
# breaking db into segmest
c_num_db_segs = 2 # should factor into c_num_files_per_batch

cb_test_by_knn = False
c_signif_centroid_thresh = 0.4

def do_init():
	global w_levels, h_levels, t_data_levels, classnames, all_file_list, num_files_in_class, num_classes, max_level

	w_levels.append(c_img_w - 2)
	h_levels.append(c_img_h - 2)
	t_data_levels.append(
		tf.Variable(tf.zeros([h_levels[0] + 2, w_levels[0] + 2, 3], dtype=tf.float32), name='t_data_levels'))

	for level in range(1, 5):
		# each level holds an image where the start image is a the 2D shape of the picture and 3 channels for rgb
		# each subsequent level is an image where the 3 channels are a dimension reduction of the 3x3x3 conv chunk to 3
		# however since we are dealing with the edge we lose the outer row of pixels at each level (1 on each side = 2)
		# When we subsample we need to do integer div by 2 but rounded UP! So we add 1 first.
		# We THEN substract by 2 because we will lose the outer row for the conv AGAIN
		w = ((w_levels[level - 1] + 1) / 2) - 2
		h = ((h_levels[level - 1] + 1) / 2) - 2
		if w <= c_min_image_dim or h <= c_min_image_dim:
			break
		w_levels.append(w)
		h_levels.append(h)
		t_data_levels.append(tf.Variable(tf.zeros([h_levels[level] + 2, w_levels[level] + 2, 3], dtype=tf.float32),
										 name='t_data_levels'))
		max_level = level

	# image_string = urllib2.urlopen(url).read()
	for ifn, fn in enumerate(os.listdir(fdir)):
		ffn = os.path.join(fdir, fn)
		if os.path.isdir(ffn):
			classnames.append(fn)

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

"""
def get_file_strings(num):
	global classnames, all_file_list, num_files_in_class, num_classes
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
"""

def get_file_lists(num_batches, num_in_batch):
	global classnames, all_file_list, num_files_in_class, num_classes
	batch_iclasses = []
	batch_ifiles = []
	for ibatch in range(num_batches):
		iclasses = []
		ifiles = []
		for inum in range(num_in_batch):
			iclass = random.randint(0, num_classes-1)
			iclasses.append(iclass)
			ifiles.append(random.randint(0, num_files_in_class[iclass] - 1))
		batch_iclasses.append(iclasses)
		batch_ifiles.append(ifiles)
	return  batch_iclasses, batch_ifiles

def get_file_strings_by_list(iclasses, ifiles):
	global classnames, all_file_list, num_files_in_class, num_classes
	strings = []
	classes = []
	for ii, iclass in enumerate( iclasses):
		ifile = ifiles[ii]
		ffn = os.path.join(fdir, classnames[iclass], all_file_list[iclass][ifile])
		with open(ffn, mode='rb') as f:
			strings.append(f.read())
		classes.append(iclass)
	return strings, classes

def get_file_strings(num):
	global classnames, all_file_list, num_files_in_class, num_classes
	iclasses, ifiles = get_file_lists(1, num)
	return get_file_strings_by_list(iclasses[0], ifiles[0])
