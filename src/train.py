# Training code
# Author: Yuliang Zou
#         ylzou@umich.edu
# Date:   2017-02-21

import numpy as np
import tensorflow as tf
from Model import FCN32, FCN16, FCN8
from Dataloader import Dataloader
import ipdb


config = {
'batch_num':5, 
'iter':100000, 
'num_classes':21, 
'max_size':(640,640),
'weight_decay': 0.0001,
'base_lr': 0.001,
'momentum': 0.9
}

if __name__ == '__main__':
	# Load pre-trained model
	model_path = '../model/VGG_imagenet.npy'
	data_dict = np.load(model_path).item()

	# Set up model and data loader
	model = FCN32(config)
	init = tf.initialize_all_variables()

	data_loader = Dataloader('train', config['batch_num'])

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
		session.run(init)
		model.load(data_dict, session)
		saver = tf.train.Saver()

		loss = 0
		for i in xrange(config['iter']):
			minibatch = data_loader.get_next_minibatch()
			feed_dict = {model.img: minibatch[0],
						model.seg: minibatch[1],
						model.mask: minibatch[2]}
			_, temp_loss = session.run([model.train_op, model.loss], feed_dict=feed_dict)
			loss += temp_loss

			# Monitor
			if i % 20 == 0 and i != 0:
				loss /= 20
				print 'Iter: {}'.format(i) + '/{}'.format(config['iter']) + ', loss = ' + str(loss)

				# Learning rate decay
				if loss <= 0.4:
					model.base_lr /= 10
					
				loss = 0

			# Write to saver
			if i % 5000 == 0 and i != 0:
				saver.save(session, '../model/FCN16_adam_iter_'+str(i)+'.ckpt')

