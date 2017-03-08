# Define the data loader for segmentation task
# Author: Yuliang Zou
#         ylzou@umich.edu
# Date:   2017-02-14

import ipdb
import numpy as np
from os.path import join
from util import colormap, prep_im_for_blob, prep_im, prep_run_wrapper
import multiprocessing

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
		process_size = self.batch_num
		# process mini_batch as 5 process, require that the number of 
		# sample in a mini_batch is a multiplying of 5
		for _ in xrange(self.batch_num/process_size):
			# Permutate the data again

			if self.temp_pointer+process_size > self.num_images:
				self.temp_pointer = 0
				self._shuffle()

			temp_range = range(self.temp_pointer, self.temp_pointer+process_size, 1)
			temp_imName = [self._img_at(x) for x in temp_range]
			temp_segName = [self._seg_at(x) for x in temp_range]
			temp_map = [self.rgb_to_gray,]*process_size

			p = multiprocessing.Pool(process_size)

			temp_result = p.map(prep_run_wrapper, zip(temp_imName, temp_segName, temp_map))
			p.close()
			p.join()

			for x in temp_result:
				img_blobs.append(x['im_blob'])
				seg_blobs.append(x['seg_blob'])
				mask_blobs.append(x['mask'])

			self.temp_pointer += process_size


		return [img_blobs, seg_blobs, mask_blobs]

"""No shuffle, fix batch num to 1"""
class Dataloader_test(Dataloader):
	def __init__(self, split):
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
		self.batch_num = 1
		self.temp_pointer = 0    # First idx of the current batch

		# Create double side mappings
		self.gray_to_rgb, self.rgb_to_gray = colormap()

	"""Get minibatch by index"""
	def get_minibatch_at(self, i):
		img_name = self._img_at(i)
		seg_name = self._seg_at(i)
		data = prep_im_for_blob(img_name, seg_name, self.rgb_to_gray)
		img_blob = data['im_blob']
		# seg_blob = data['seg_blob']
		mask = data['mask']

		img_blobs = np.array([img_blob])
		# seg_blobs = np.array([seg_blob])
		mask_blobs = np.array([mask])
		seg_blobs = None
		# mask_blobs = None

		return [img_blobs, seg_blobs, mask_blobs]


"""Small size dataloader"""
class Dataloader_small(Dataloader):
	def __init__(self, split):
		Dataloader.__init__(self, split)

	"""Override"""
	def get_next_minibatch(self):
		img_blobs = []
		seg_blobs = []
		mask_blobs = []
		ori_sizes = []

		process_size = 5
		# process mini_batch as 5 process, require that the number of 
		# sample in a mini_batch is a multiplying of 5
		for _ in xrange(self.batch_num/process_size):
			# Permutate the data again

			if self.temp_pointer+process_size > self.num_images:
				self.temp_pointer = 0
				self._shuffle()

			temp_range = range(self.temp_pointer, self.temp_pointer+process_size, 1)
			temp_imName = [self._img_at(x) for x in temp_range]
			temp_segName = [self._seg_at(x) for x in temp_range]
			temp_map = [self.rgb_to_gray,]*process_size

			p = multiprocessing.Pool(process_size)

			temp_result = p.map(prep_run_wrapper, zip(temp_imName, temp_segName, temp_map))
			p.close()
			p.join()

			for x in temp_result:
				img_blobs.append(x['im_blob'])
				seg_blobs.append(x['seg_blob'])
				mask_blobs.append(x['mask'])
				ori_sizes.append(x['original_size'])

			self.temp_pointer += process_size


		return [img_blobs, seg_blobs, mask_blobs, ori_sizes]

if __name__ == '__main__':
	# dataloader = Dataloader('train', 10)
	# minibatch = dataloader.get_next_minibatch()
	dataloader = Dataloader_test('train')
	minibatch = dataloader.get_next_minibatch()


	ipdb.set_trace()
