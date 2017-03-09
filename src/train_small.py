# Training code for small size input
# Author: Yuliang Zou
#         ylzou@umich.edu
# Date:   2017-02-21

import numpy as np
import tensorflow as tf
from Model import FCN32, FCN16, FCN8
from Dataloader import Dataloader_small
import ipdb


config = {
'batch_num':20, 
'iter':100000, 
'num_classes':21, 
'max_size':(256,256),
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
	loss_list = []
	f = open('./FCN32_small.txt', 'w')
	DECAY = False    # decay flag
	init = tf.initialize_all_variables()

	data_loader = Dataloader_small('train', config)

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

			loss_list.append(temp_loss)
			f.write(str(temp_loss) + '\n')
			print str(i) + ': ' + str(temp_loss)

			# Learning rate decay
			if len(loss_list) > 100 and not DECAY:
				avg = sum(loss_list[-100::]) / 100.0
				if avg <= 0.4:
					model.base_lr /= 10
					DECAY = True

			# Monitor
			if i % 20 == 0 and i != 0:
				loss /= 20
				print 'Iter: {}'.format(i) + '/{}'.format(config['iter']) + ', loss = ' + str(loss)					
				loss = 0

			# Write to saver
			if i % 5000 == 0 and i != 0:
				saver.save(session, '../model/FCN32_small_adam_iter_'+str(i)+'.ckpt')

	f.close()

