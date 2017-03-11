# Demo
# Author: Yuliang Zou
#         ylzou@umich.edu
# Date:   2017-03-03

import numpy as np
import tensorflow as tf
from Model import FCN32_test, FCN16_test, FCN8_test
from Dataloader import Dataloader, Dataloader_small
import matplotlib.pyplot as plt
import cv2
import ipdb


# BGR mean pixel value
MEAN_PIXEL = np.array([103.939, 116.779, 123.68])

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

config = {
'batch_num':5, 
'iter':100000, 
'num_classes':21, 
'max_size':(640,640),
'weight_decay': 0.0005,
'base_lr': 0.001,
'momentum': 0.9
}

if __name__ == '__main__':
	model = FCN8_test(config)
	data_loader = Dataloader('val', config)

	saver = tf.train.Saver()
	ckpt = '../model/FCN8_adam_iter_10000.ckpt'
	# Extract ckpt into npy, if needed
	# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
		# model.extract(ckpt, session, saver)
	# ipdb.set_trace()

	dump_path = '../demo/'

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
		saver.restore(session, ckpt)
		print 'Model restored.'

		minibatch = data_loader.get_next_minibatch()
		feed_dict = {model.img: minibatch[0],
						model.seg: minibatch[1],
						model.mask: minibatch[2]}
		pred = session.run(model.get_output('deconv'), feed_dict=feed_dict)

		for i in range(config['batch_num']):
			mask = minibatch[2][i]
			seg  = np.argmax(pred[i], axis=2)
			img  = minibatch[0][i]
			gt   = minibatch[1][i][:,:,0]
			f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False)
			ax1.imshow(seg)
			img = img + MEAN_PIXEL
			ax2.imshow(img[:,:,::-1])
			ax3.imshow(gt)
			plt.show()
			cv2.imwrite(dump_path + str(i) + '_seg.png', seg)
			cv2.imwrite(dump_path + str(i) + '_img.png', img)

