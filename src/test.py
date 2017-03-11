# Generate segmentation results
# Author: Yuliang Zou
#         ylzou@umich.edu
# Date:   2017-03-07

import numpy as np
import tensorflow as tf
from Model import FCN32_test, FCN16_test, FCN8_test
from Dataloader import Dataloader, Dataloader_test
from util import get_original_size, seg_gray_to_rgb
import cv2
from os import makedirs
from os.path import exists, join
import ipdb


config = {
'batch_num':1, 
'iter':100000, 
'num_classes':21, 
'max_size':(640,640),
'weight_decay': 0.0005,
'base_lr': 0.0001,
'momentum': 0.9
}

if __name__ == '__main__':
	# Specify which set to test
	split = 'val'
	model = FCN8_test(config)
	# Import, since we don't want the random shuffle
	data_loader = Dataloader_test(split, config)

	saver = tf.train.Saver()
	ckpt = '../model/FCN8_adam_iter_10000.ckpt'
	ID = ckpt.split('/')[-1][:-5]

	res_dir = '../result/'
	dump_path = join(res_dir, ID)
	dump_path = join(dump_path, split)
	rgb_path = join(dump_path, 'rgb')
	gray_path = join(dump_path, 'gray')

	if not exists(rgb_path):
		makedirs(rgb_path)
	if not exists(gray_path):
		makedirs(gray_path)

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
		saver.restore(session, ckpt)
		print 'Model restored.'

		# Iterate the whole set once
		for i in range(data_loader.num_images):
			minibatch = data_loader.get_minibatch_at(i)
			feed_dict = {model.img: minibatch[0]}
			pred = session.run(model.get_output('deconv'), feed_dict=feed_dict)

			mask = minibatch[2][0]
			seg  = np.argmax(pred[0], axis=2)

			row, col = minibatch[3]
			seg_valid = np.zeros((row, col))
			seg_valid[:, :] = seg[0:row, 0:col]
			seg_rgb = seg_gray_to_rgb(seg_valid, data_loader.gray_to_rgb)

			im_name = data_loader._seg_at(i).split('/')[-1]
			cv2.imwrite(join(rgb_path, im_name), seg_rgb[:,:,::-1])
			cv2.imwrite(join(gray_path, im_name), seg_valid)

			print str(i) + '/' + str(data_loader.num_images) + ' done!'
