# Define the vgg16 style model
# Author: Yuliang Zou
#         ylzou@umich.edu
# Date:   2017-02-19

import tensorflow as tf
import numpy as np
from util import bilinear_upsample_weights
import ipdb

"""Define a base class, containing some useful layer functions"""
class Network(object):
	def __init__(self, inputs):
		self.inputs = []
		self.layers = {}
		self.outputs = {}

	"""Extract parameters from ckpt file to npy file"""
	def extract(self, data_path, session, saver):
		raise NotImplementedError('Must be subclassed.')

	"""Load pre-trained model from numpy data_dict"""
	def load(self, data_dict, session, ignore_missing=True):
		fc_shapes = {'fc6':(7,7,512,4096), 'fc7':(1,1,4096,4096)}
		fc_scopes = {'fc6':'conv6', 'fc7':'conv7'}
		for key in data_dict:
			# Special cases: fc6 and fc7
			if key == 'fc6' or key == 'fc7':
				w = np.reshape(data_dict[key]['weights'], fc_shapes[key])
				b = data_dict[key]['biases']
				with tf.variable_scope(fc_scopes[key], reuse=True):
					var1 = tf.get_variable('weights')
					session.run(var1.assign(w))
					print "Assign pretrain model weights to " + fc_scopes[key]
					var2 = tf.get_variable('biases')
					session.run(var2.assign(b))
					print "Assign pretrain model biases to " + fc_scopes[key]
				continue

			with tf.variable_scope(key, reuse=True):
				for subkey in data_dict[key]:
					try:
						var = tf.get_variable(subkey)
						session.run(var.assign(data_dict[key][subkey]))
						print "Assign pretrain model " + subkey + " to " + key
					except ValueError:
						print "Ignore " + key
						if not ignore_missing:
							raise

	"""Get outputs given key names"""
	def get_output(self, key):
		if key not in self.outputs:
			raise KeyError
		return self.outputs[key]

	"""Get parameters given key names"""
	def get_param(self, key):
		if key not in self.layers:
			raise KeyError
		return self.layers[key]['weights'], self.layers[key]['biases']

	"""Add conv part of vgg16"""
	def add_conv(self, inputs, num_classes, stage='TRAIN'):
		# Dropout is different for training and testing
		if stage == 'TRAIN':
			keep_prob = 0.5
		elif stage == 'TEST':
			keep_prob = 1
		else:
			raise ValueError

		# Conv1
		with tf.variable_scope('conv1_1') as scope:
			w_conv1_1 = tf.get_variable('weights', [3, 3, 3, 64], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv1_1 = tf.get_variable('biases', [64], 
				initializer=tf.constant_initializer(0))
			z_conv1_1 = tf.nn.conv2d(inputs, w_conv1_1, strides=[1, 1, 1, 1], 
				padding='SAME') + b_conv1_1
			a_conv1_1 = tf.nn.relu(z_conv1_1)

		with tf.variable_scope('conv1_2') as scope:
			w_conv1_2 = tf.get_variable('weights', [3, 3, 64, 64], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv1_2 = tf.get_variable('biases', [64], 
				initializer=tf.constant_initializer(0))
			z_conv1_2 = tf.nn.conv2d(a_conv1_1, w_conv1_2, strides=[1, 1, 1, 1], 
				padding='SAME') + b_conv1_2
			a_conv1_2 = tf.nn.relu(z_conv1_2)
		
		pool1 = tf.nn.max_pool(a_conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='VALID', name='pool1')

		# Conv2
		with tf.variable_scope('conv2_1') as scope:
			w_conv2_1 = tf.get_variable('weights', [3, 3, 64, 128], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv2_1 = tf.get_variable('biases', [128], 
				initializer=tf.constant_initializer(0))
			z_conv2_1 = tf.nn.conv2d(pool1, w_conv2_1, strides=[1, 1, 1, 1], 
				padding='SAME') + b_conv2_1
			a_conv2_1 = tf.nn.relu(z_conv2_1)

		with tf.variable_scope('conv2_2') as scope:
			w_conv2_2 = tf.get_variable('weights', [3, 3, 128, 128], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv2_2 = tf.get_variable('biases', [128], 
				initializer=tf.constant_initializer(0))
			z_conv2_2 = tf.nn.conv2d(a_conv2_1, w_conv2_2, strides=[1, 1, 1, 1], 
				padding='SAME') + b_conv2_2
			a_conv2_2 = tf.nn.relu(z_conv2_2)

		pool2 = tf.nn.max_pool(a_conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='VALID', name='pool2')

		# Conv3
		with tf.variable_scope('conv3_1') as scope:
			w_conv3_1 = tf.get_variable('weights', [3, 3, 128, 256],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv3_1 = tf.get_variable('biases', [256],
				initializer=tf.constant_initializer(0))
			z_conv3_1 = tf.nn.conv2d(pool2, w_conv3_1, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv3_1
			a_conv3_1 = tf.nn.relu(z_conv3_1)

		with tf.variable_scope('conv3_2') as scope:
			w_conv3_2 = tf.get_variable('weights', [3, 3, 256, 256],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv3_2 = tf.get_variable('biases', [256],
				initializer=tf.constant_initializer(0))
			z_conv3_2 = tf.nn.conv2d(a_conv3_1, w_conv3_2, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv3_2
			a_conv3_2 = tf.nn.relu(z_conv3_2)

		with tf.variable_scope('conv3_3') as scope:
			w_conv3_3 = tf.get_variable('weights', [3, 3, 256, 256],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv3_3 = tf.get_variable('biases', [256],
				initializer=tf.constant_initializer(0))
			z_conv3_3 = tf.nn.conv2d(a_conv3_2, w_conv3_3, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv3_3
			a_conv3_3 = tf.nn.relu(z_conv3_3)

		pool3 = tf.nn.max_pool(a_conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='VALID', name='pool3')

		# Conv4
		with tf.variable_scope('conv4_1') as scope:
			w_conv4_1 = tf.get_variable('weights', [3, 3, 256, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv4_1 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv4_1 = tf.nn.conv2d(pool3, w_conv4_1, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv4_1
			a_conv4_1 = tf.nn.relu(z_conv4_1)

		with tf.variable_scope('conv4_2') as scope:
			w_conv4_2 = tf.get_variable('weights', [3, 3, 512, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv4_2 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv4_2 = tf.nn.conv2d(a_conv4_1, w_conv4_2, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv4_2
			a_conv4_2 = tf.nn.relu(z_conv4_2)

		with tf.variable_scope('conv4_3') as scope:
			w_conv4_3 = tf.get_variable('weights', [3, 3, 512, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv4_3 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv4_3 = tf.nn.conv2d(a_conv4_2, w_conv4_3, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv4_3
			a_conv4_3 = tf.nn.relu(z_conv4_3)

		pool4 = tf.nn.max_pool(a_conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='VALID', name='pool4')

		# Conv5
		with tf.variable_scope('conv5_1') as scope:
			w_conv5_1 = tf.get_variable('weights', [3, 3, 512, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv5_1 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv5_1 = tf.nn.conv2d(pool4, w_conv5_1, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv5_1
			a_conv5_1 = tf.nn.relu(z_conv5_1)

		with tf.variable_scope('conv5_2') as scope:
			w_conv5_2 = tf.get_variable('weights', [3, 3, 512, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv5_2 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv5_2 = tf.nn.conv2d(a_conv5_1, w_conv5_2, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv5_2
			a_conv5_2 = tf.nn.relu(z_conv5_2)

		with tf.variable_scope('conv5_3') as scope:
			w_conv5_3 = tf.get_variable('weights', [3, 3, 512, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv5_3 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv5_3 = tf.nn.conv2d(a_conv5_2, w_conv5_3, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv5_3
			a_conv5_3 = tf.nn.relu(z_conv5_3)

		pool5 = tf.nn.max_pool(a_conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='VALID', name='pool5')

		# Transform fully-connected layers to convolutional layers
		with tf.variable_scope('conv6') as scope:
			w_conv6 = tf.get_variable('weights', [7, 7, 512, 4096],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv6 = tf.get_variable('biases', [4096],
				initializer=tf.constant_initializer(0))
			z_conv6 = tf.nn.conv2d(pool5, w_conv6, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv6
			a_conv6 = tf.nn.relu(z_conv6)
			d_conv6 = tf.nn.dropout(a_conv6, keep_prob)

		with tf.variable_scope('conv7') as scope:
			w_conv7 = tf.get_variable('weights', [1, 1, 4096, 4096],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv7 = tf.get_variable('biases', [4096],
				initializer=tf.constant_initializer(0))
			z_conv7 = tf.nn.conv2d(d_conv6, w_conv7, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv7
			a_conv7 = tf.nn.relu(z_conv7)
			d_conv7 = tf.nn.dropout(a_conv7, keep_prob)

		# Replace the original classifier layer
		with tf.variable_scope('conv8') as scope:
			w_conv8 = tf.get_variable('weights', [1, 1, 4096, num_classes],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv8 = tf.get_variable('biases', [num_classes],
				initializer=tf.constant_initializer(0))
			z_conv8 = tf.nn.conv2d(d_conv7, w_conv8, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv8

		# Add to store dicts
		self.outputs['conv1_1'] = a_conv1_1
		self.outputs['conv1_2'] = a_conv1_2
		self.outputs['pool1']   = pool1
		self.outputs['conv2_1'] = a_conv2_1
		self.outputs['conv2_2'] = a_conv2_2
		self.outputs['pool2']   = pool2
		self.outputs['conv3_1'] = a_conv3_1
		self.outputs['conv3_2'] = a_conv3_2
		self.outputs['conv3_3'] = a_conv3_3
		self.outputs['pool3']   = pool3
		self.outputs['conv4_1'] = a_conv4_1
		self.outputs['conv4_2'] = a_conv4_2
		self.outputs['conv4_3'] = a_conv4_3
		self.outputs['pool4']   = pool4
		self.outputs['conv5_1'] = a_conv5_1
		self.outputs['conv5_2'] = a_conv5_2
		self.outputs['conv5_3'] = a_conv5_3
		self.outputs['pool5']   = pool5
		self.outputs['conv6']   = d_conv6
		self.outputs['conv7']   = d_conv7
		self.outputs['conv8']   = z_conv8

		self.layers['conv1_1'] = {'weights':w_conv1_1, 'biases':b_conv1_1}
		self.layers['conv1_2'] = {'weights':w_conv1_2, 'biases':b_conv1_2}
		self.layers['conv2_1'] = {'weights':w_conv2_1, 'biases':b_conv2_1}
		self.layers['conv2_2'] = {'weights':w_conv2_2, 'biases':b_conv2_2}
		self.layers['conv3_1'] = {'weights':w_conv3_1, 'biases':b_conv3_1}
		self.layers['conv3_2'] = {'weights':w_conv3_2, 'biases':b_conv3_2}
		self.layers['conv3_3'] = {'weights':w_conv3_3, 'biases':b_conv3_3}
		self.layers['conv4_1'] = {'weights':w_conv4_1, 'biases':b_conv4_1}
		self.layers['conv4_2'] = {'weights':w_conv4_2, 'biases':b_conv4_2}
		self.layers['conv4_3'] = {'weights':w_conv4_3, 'biases':b_conv4_3}
		self.layers['conv5_1'] = {'weights':w_conv5_1, 'biases':b_conv5_1}
		self.layers['conv5_2'] = {'weights':w_conv5_2, 'biases':b_conv5_2}
		self.layers['conv5_3'] = {'weights':w_conv5_3, 'biases':b_conv5_3}
		self.layers['conv6']   = {'weights':w_conv6, 'biases':b_conv6}
		self.layers['conv7']   = {'weights':w_conv7, 'biases':b_conv7}
		self.layers['conv8']   = {'weights':w_conv8, 'biases':b_conv8}


"""Baseline model"""
class FCN32(Network):
	def __init__(self, config):
		self.num_classes = config['num_classes']
		self.batch_num = config['batch_num']
		self.max_size = config['max_size']
		self.weight_decay = config['weight_decay']
		self.base_lr = config['base_lr']
		self.momentum = config['momentum']

		self.img  = tf.placeholder(tf.float32, 
			[self.batch_num, self.max_size[0], self.max_size[1], 3])
		self.seg  = tf.placeholder(tf.int32, 
			[self.batch_num, self.max_size[0], self.max_size[1], 1])
		self.mask = tf.placeholder(tf.float32, 
			[self.batch_num, self.max_size[0], self.max_size[1], 1])

		self.layers = {}
		self.outputs = {}
		self.set_up()

	def set_up(self):
		self.add_conv(self.img, self.num_classes)
		self.add_deconv(bilinear=True)
		self.add_loss_op()
		self.add_weight_decay()
		self.add_train_op()

	"""Add the deconv(upsampling) layer to get dense prediction"""
	def add_deconv(self, bilinear=False):
		conv8 = self.get_output('conv8')

		with tf.variable_scope('deconv') as scope:
			# Learn from scratch
			if not bilinear:
				w_deconv = tf.get_variable('weights', [64, 64, self.num_classes, self.num_classes],
					initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			# Using fiexed bilinearing upsampling filter
			else:
				w_deconv = tf.get_variable('weights', trainable=True, 
					initializer=bilinear_upsample_weights(32, self.num_classes))

			b_deconv = tf.get_variable('biases', [self.num_classes],
				initializer=tf.constant_initializer(0))
			z_deconv = tf.nn.conv2d_transpose(conv8, w_deconv, 
				[self.batch_num, self.max_size[0], self.max_size[1], self.num_classes],
				strides=[1,32,32,1], padding='SAME', name='z') + b_deconv

		# Add to store dicts
		self.outputs['deconv'] = z_deconv
		self.layers['deconv']  = {'weights':w_deconv, 'biases':b_deconv}

	"""Add pixelwise softmax loss"""
	def add_loss_op(self):
		pred = self.get_output('deconv')
		pred_reshape = tf.reshape(pred, [-1, self.num_classes])
		gt_reshape = tf.reshape(self.seg, [-1])

		loss_reshape = tf.nn.sparse_softmax_cross_entropy_with_logits(pred_reshape, gt_reshape)
		loss = tf.reshape(loss_reshape, [self.batch_num, self.max_size[0], self.max_size[1], 1])
		loss_valid = tf.reduce_sum(loss * self.mask, (1,2,3))

		valid_pixels = tf.reduce_sum(self.mask, (1,2,3))
		loss_avg = tf.reduce_mean(loss_valid / valid_pixels)

		self.loss = loss_avg

	"""Add weight decay"""
	def add_weight_decay(self):
		for key in self.layers:
			w = self.layers[key]['weights']
			self.loss += self.weight_decay * tf.nn.l2_loss(w)

	"""Set up training optimization"""
	def add_train_op(self):
		self.train_op = tf.train.MomentumOptimizer(self.base_lr, 
			self.momentum).minimize(self.loss)


"""A better model"""
class FCN16(FCN32):
	def __init__(self, config):
		FCN32.__init__(self, config)

	def set_up(self):
		self.add_conv(self.img, self.num_classes)
		self.add_shortcut(bilinear=True)
		self.add_deconv(bilinear=True)
		self.add_loss_op()
		self.add_weight_decay()
		self.add_train_op()

	def add_shortcut(self, bilinear=False):
		conv8 = self.get_output('conv8')
		target_size = 2 * int(conv8.get_shape()[1])

		with tf.variable_scope('2x_conv8') as scope:
			# Learn from scratch
			if not bilinear:
				w_deconv = tf.get_variable('weights', [4, 4, self.num_classes, self.num_classes],
					initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			# Using fiexed bilinearing upsampling filter
			else:
				w_deconv = tf.get_variable('weights', trainable=True, 
					initializer=bilinear_upsample_weights(2, self.num_classes))

			b_deconv = tf.get_variable('biases', [self.num_classes],
				initializer=tf.constant_initializer(0))
			z_deconv = tf.nn.conv2d_transpose(conv8, w_deconv, 
				[self.batch_num, target_size, target_size, self.num_classes],
				strides=[1,2,2,1], padding='SAME', name='z') + b_deconv

		pool4 = self.get_output('pool4')

		with tf.variable_scope('pool4_1x1') as scope:
			w_pool4 = tf.get_variable('weights', [1, 1, 512, self.num_classes],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_pool4 = tf.get_variable('biases', [self.num_classes],
				initializer=tf.constant_initializer(0))
			z_pool4 = tf.nn.conv2d(pool4, w_pool4, strides= [1, 1, 1, 1],
				padding='SAME') + b_pool4

		# Element-wise sum
		fusion = z_deconv + z_pool4

		# Add to store dicts
		self.outputs['2x_conv8'] = z_deconv
		self.outputs['pool4_1x1'] = z_pool4
		self.outputs['fusion'] = fusion
		self.layers['2x_conv8']  = {'weights':w_deconv, 'biases':b_deconv}
		self.layers['pool4_1x1'] = {'weights':w_pool4, 'biases':b_pool4}


	"""Add the deconv(upsampling) layer to get dense prediction"""
	def add_deconv(self, bilinear=False):
		fusion = self.get_output('fusion')

		with tf.variable_scope('deconv') as scope:
			# Learn from scratch
			if not bilinear:
				w_deconv = tf.get_variable('weights', [32, 32, self.num_classes, self.num_classes],
					initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			# Using fiexed bilinearing upsampling filter
			else:
				w_deconv = tf.get_variable('weights', trainable=True, 
					initializer=bilinear_upsample_weights(16, self.num_classes))

			b_deconv = tf.get_variable('biases', [self.num_classes],
				initializer=tf.constant_initializer(0))
			z_deconv = tf.nn.conv2d_transpose(fusion, w_deconv, 
				[self.batch_num, self.max_size[0], self.max_size[1], self.num_classes],
				strides=[1,16,16,1], padding='SAME', name='z') + b_deconv

		# Add to store dicts
		self.outputs['deconv'] = z_deconv
		self.layers['deconv']  = {'weights':w_deconv, 'biases':b_deconv}


"""The best model"""
class FCN8(FCN16):
	def __init__(self, config):
		FCN16.__init__(self, config)

	def add_shortcut(self, bilinear=True):
		conv8 = self.get_output('conv8')
		target_size = 2 * int(conv8.get_shape()[1])

		with tf.variable_scope('2x_conv8') as scope:
			# Learn from scratch
			if not bilinear:
				w_deconv = tf.get_variable('weights', [4, 4, self.num_classes, self.num_classes],
					initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			# Using fiexed bilinearing upsampling filter
			else:
				w_deconv = tf.get_variable('weights', trainable=True, 
					initializer=bilinear_upsample_weights(2, self.num_classes))

			b_deconv = tf.get_variable('biases', [self.num_classes],
				initializer=tf.constant_initializer(0))
			z_deconv = tf.nn.conv2d_transpose(conv8, w_deconv, 
				[self.batch_num, target_size, target_size, self.num_classes],
				strides=[1,2,2,1], padding='SAME', name='z') + b_deconv

		pool4 = self.get_output('pool4')

		with tf.variable_scope('pool4_1x1') as scope:
			w_pool4 = tf.get_variable('weights', [1, 1, 512, self.num_classes],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_pool4 = tf.get_variable('biases', [self.num_classes],
				initializer=tf.constant_initializer(0))
			z_pool4 = tf.nn.conv2d(pool4, w_pool4, strides= [1, 1, 1, 1],
				padding='SAME') + b_pool4

		# Element-wise sum
		fusion1 = z_deconv + z_pool4

		## Second fusion stage
		pool3 = self.get_output('pool3')

		with tf.variable_scope('pool3_1x1') as scope:
			w_pool3 = tf.get_variable('weights', [1, 1, 256, self.num_classes],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_pool3 = tf.get_variable('biases', [self.num_classes],
				initializer=tf.constant_initializer(0))
			z_pool3 = tf.nn.conv2d(pool3, w_pool3, strides= [1, 1, 1, 1],
				padding='SAME') + b_pool3

		target_size *= 2

		with tf.variable_scope('2x_fusion') as scope:
			# Learn from scratch
			if not bilinear:
				w_deconv2 = tf.get_variable('weights', [4, 4, self.num_classes, self.num_classes],
					initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			# Using fiexed bilinearing upsampling filter
			else:
				w_deconv2 = tf.get_variable('weights', trainable=True, 
					initializer=bilinear_upsample_weights(2, self.num_classes))

			b_deconv2 = tf.get_variable('biases', [self.num_classes],
				initializer=tf.constant_initializer(0))
			z_deconv2 = tf.nn.conv2d_transpose(fusion1, w_deconv2, 
				[self.batch_num, target_size, target_size, self.num_classes],
				strides=[1,2,2,1], padding='SAME', name='z') + b_deconv2

		fusion2 = z_pool3 + z_deconv2

		# Add to store dicts
		self.outputs['2x_conv8'] = z_deconv
		self.outputs['pool4_1x1'] = z_pool4
		self.outputs['pool3_1x1'] = z_pool3
		self.outputs['2x_fusion'] = z_deconv2
		self.outputs['fusion'] = fusion2
		self.layers['2x_conv8']  = {'weights':w_deconv, 'biases':b_deconv}
		self.layers['pool4_1x1'] = {'weights':w_pool4, 'biases':b_pool4}
		self.layers['pool3_1x1'] = {'weights':w_pool3, 'biases':b_pool3}
		self.layers['2x_fusion'] = {'weights':w_deconv2, 'biases':b_deconv2}


	"""Add the deconv(upsampling) layer to get dense prediction"""
	def add_deconv(self, bilinear=False):
		fusion = self.get_output('fusion')

		with tf.variable_scope('deconv') as scope:
			# Learn from scratch
			if not bilinear:
				w_deconv = tf.get_variable('weights', [16, 16, self.num_classes, self.num_classes],
					initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			# Using fiexed bilinearing upsampling filter
			else:
				w_deconv = tf.get_variable('weights', trainable=True, 
					initializer=bilinear_upsample_weights(16, self.num_classes))

			b_deconv = tf.get_variable('biases', [self.num_classes],
				initializer=tf.constant_initializer(0))
			z_deconv = tf.nn.conv2d_transpose(fusion, w_deconv, 
				[self.batch_num, self.max_size[0], self.max_size[1], self.num_classes],
				strides=[1,8,8,1], padding='SAME', name='z') + b_deconv

		# Add to store dicts
		self.outputs['deconv'] = z_deconv
		self.layers['deconv']  = {'weights':w_deconv, 'biases':b_deconv}


if __name__ == '__main__':
	config = {
	'batch_num':5, 
	'iter':100000, 
	'num_classes':21, 
	'max_size':(640,640),
	'weight_decay': 0.0005,
	'base_lr': 0.005,
	'momentum': 0.9
	}

	#model = FCN32(config)
	#model = FCN16(config)
	model = FCN8(config)

