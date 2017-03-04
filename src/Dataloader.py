# Define the data loader for segmentation task
# Author: Yuliang Zou
#         ylzou@umich.edu
# Date:   2017-02-14

import ipdb
import numpy as np
from os.path import join
from util import colormap, prep_im_for_blob

"""
The Dataloader for VOC2011 to load and preprocess input image and segmentation
ground truth. (Only)
"""
class Dataloader(object):
	def __init__(self, split, batch_num):
		# Validate split input
		if split != 'train' and split != 'val' and split != 'trainval' and split != 'test':
			raise Exception('Please enter valid split variable!')

		root = '../data/VOCdevkit/VOC2011/'
		self.img_path = join(root, 'JPEGImages/')
		self.seg_path = join(root, 'SegmentationClass/')
		self.split = split
		img_set = join(root, 'ImageSets/Segmentation/' + split + '.txt')
		with open(img_set) as f:
			self.img_list = f.read().rstrip().split('\n')

		self.num_images = len(self.img_list)
		self.batch_num = batch_num
		self.temp_pointer = 0    # First idx of the current batch
		self._shuffle()

		# Create double side mappings
		self.gray_to_rgb, self.rgb_to_gray = colormap()


	def _shuffle(self):
		self.img_list = np.random.permutation(self.img_list)

	def _img_at(self, i):
		return self.img_path + self.img_list[i] + '.jpg'

	def _seg_at(self, i):
		return self.seg_path + self.img_list[i] + '.png'

	"""Use padding to get same shapes"""
	def get_next_minibatch(self):
		img_blobs = []
		seg_blobs = []
		mask_blobs = []

		for _ in xrange(self.batch_num):
			# Permutate the data again
			if self.temp_pointer == self.num_images:
				self.temp_pointer = 0
				self._shuffle()

			img_name = self._img_at(self.temp_pointer)
			seg_name = self._seg_at(self.temp_pointer)
			img_blob, seg_blob, mask = prep_im_for_blob(img_name, seg_name, self.rgb_to_gray)
			self.temp_pointer += 1

			img_blobs.append(img_blob)
			seg_blobs.append(seg_blob)
			mask_blobs.append(mask)

		img_blobs = np.array(img_blobs)
		seg_blobs = np.array(seg_blobs)
		mask_blobs = np.array(mask_blobs)

		return [img_blobs, seg_blobs, mask_blobs]


if __name__ == '__main__':
	dataloader = Dataloader('train', 5)
	feed_dict = dataloader.get_next_minibatch()

	ipdb.set_trace()