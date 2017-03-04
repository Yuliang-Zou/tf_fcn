# Demo
# Author: Yuliang Zou
#         ylzou@umich.edu
# Date:   2017-03-03

import numpy as np
import tensorflow as tf
from Model import FCN32
from Dataloader import Dataloader
import matplotlib.pyplot as plt
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
'batch_num':1, 
'iter':100000, 
'num_classes':21, 
'max_size':(640,640),
'weight_decay': 0.0005,
'base_lr': 0.001,
'momentum': 0.9
}

if __name__ == '__main__':
	model = FCN32(config)
	data_loader = Dataloader('val', config['batch_num'])

	saver = tf.train.Saver()
	ckpt = '../model/FCN32_1e-3_iter_5000.ckpt'

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
			plt.imshow(seg)
			plt.show()
			plt.imshow(minibatch[0][i])
			plt.show()
