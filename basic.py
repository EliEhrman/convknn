from __future__ import print_function
# import csv
import numpy as np
import tensorflow as tf
import random
from tensorflow.python import debug as tf_debug


# print( range(10))

picindxs = [[j, i] for i in range(1,9) for j in range(1,9)]
rangepic = [[[i + 100*k for i in range(10*j, 10*(j+1))] for j in range(10)] for k in range(3)]

t_data = tf.constant(rangepic, name='t_data')
t_output = tf.Variable(173, dtype=tf.int32)

def newval(current_input):
	global t_output
	tf.assign(t_output, current_input)
	return current_input

def wbody(elem):
	# tf.assign(t_output[i], t_data[i])
	global t_data
	iis = [t_data[k, elem[0]+j, elem[1]+i] for i in range(-1,2) for j in range(-1,2) for k in range(3)]
	return iis
	# return [elem[0], elem[1]]

def wtest(elem):
	# tf.assign(t_output[i], t_data[i])
	global t_data
	iis = [rangepic[k][elem[0]+j][elem[1]+i] for i in range(-1,2) for j in range(-1,2) for k in range(3)]
	return iis

# rbd = [wtest(elem) for elem in picindxs]


elems = tf.constant(picindxs, dtype=tf.int32, name='elems')
structure = [tf.int32 for i in range(27)]
alternates = tf.map_fn(wbody, elems, dtype=structure)
db = tf.transpose(tf.stack(alternates), perm=[1,0])

"""
i = tf.constant(0)
c = lambda i: tf.less(i, 10)
b = lambda i: tf.add(i, 1)
r = tf.while_loop(c, b, [i])

t_data = tf.constant(range(10), name='t_data')
t_len = tf.shape(t_data, name='t_len')
i = tf.constant(0)
c = lambda counter: tf.less(counter, 10)
body = lambda i: tf.assign(t_output[i], t_data[i])
# loop_op = tf.while_loop(c, wbody, [i])
scan_op = tf.scan(newval, t_data, initializer=i)
"""
sess = tf.Session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="curses")

sess.run(tf.global_variables_initializer())

print(sess.run(db))
print(sess.run(t_output))