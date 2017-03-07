# Define some util functions
# Author: Yuliang Zou
#         ylzou@umich.edu
# Date:   2017-02-19

import numpy as np
import cv2
import ipdb

# BGR mean pixel value
MEAN_PIXEL = np.array([103.939, 116.779, 123.68])

"""Padding image and segmentation ground truth to (640, 640)"""
def prep_im_for_blob(im_name, seg_name, rgb_to_gray, max_size=(640,640)):
	im = cv2.imread(im_name)    # OpenCV color map default BGR
	im = im - MEAN_PIXEL
	seg = cv2.imread(seg_name)[:,:,::-1]

	row, col, _ = im.shape
	im_blob = np.zeros((max_size[0], max_size[1], 3))
	im_blob[0:row,0:col,:] = im

	seg_blob = np.zeros((max_size[0], max_size[1], 1))
	mask = np.zeros_like(seg_blob)
	for i in xrange(row):
		for j in xrange(col):
			seg_blob[i,j] = rgb_to_gray[tuple(seg[i,j,:])]
			# Discard 255 edge class
			if seg_blob[i,j] != 255:
				mask[i,j] = 1
			else:
				seg_blob[i,j] = 0

	return {'im_blob':im_blob, 'seg_blob':seg_blob, 'mask':mask}

"""Minus mean pixel value"""
def prep_im(im_name):
	im = cv2.imread(im_name)    # OpenCV color map default BGR
	im = np.array([im - MEAN_PIXEL])
	return im

"""Create color mappings, check VOClabelcolormap.m for reference"""
def colormap(N=256):
	# Create double side mappings
	gray_to_rgb = {}
	rgb_to_gray = {}

	for i in range(N):
		temp = i
		r = 0
		g = 0
		b = 0
		for j in range(8):
			r = r | ((temp & 1) << (7-j))
			g = g | (((temp >> 1) & 1) << (7-j))
			b = b | (((temp >> 2) & 1) << (7-j))
			temp = temp >> 3
		gray_to_rgb[i] = (r,g,b)

	for key, val in gray_to_rgb.iteritems():
		rgb_to_gray[val] = key

	return gray_to_rgb, rgb_to_gray

"""For multi-processing dataloader"""
def prep_run_wrapper(args):
	return prep_im_for_blob(*args)

"""Get original size"""
def get_original_size(mask, max_size=(640,640)):
	for i in range(max_size[0]-1, -1, -1):
		if mask[i,0,0] == 1:
			row = i + 1
			break

	for i in range(max_size[1]-1, -1, -1):
		if mask[0,i,0] == 1:
			col = i + 1
			break

	return row, col

"""Transform gray scale segmentation result to rgb format"""
def seg_gray_to_rgb(seg, gray_to_rgb):
	row, col = seg.shape
	rgb = np.zeros((row, col, 3))

	for i in range(row):
		for j in range(col):
			r, g, b = gray_to_rgb[seg[i, j]]
			rgb[i, j, 0] = r
			rgb[i, j, 1] = g
			rgb[i, i, 2] = b

	return rgb


"""
Helper functions for bilinear upsampling
credit: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
"""
def get_kernel_size(factor):
	"""
	Find the kernel size given the desired factor of upsampling.
	"""
	return 2 * factor - factor % 2

def upsample_filt(size):
	"""
	Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
	"""
	factor = (size + 1) // 2
	if size % 2 == 1:
		center = factor - 1
	else:
		center = factor - 0.5
	og = np.ogrid[:size, :size]
	return (1 - abs(og[0] - center) / factor) * \
			(1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, number_of_classes):
	"""
	Create weights matrix for transposed convolution with bilinear filter
	initialization.
	"""
	filter_size = get_kernel_size(factor)

	weights = np.zeros((filter_size,
						filter_size,
						number_of_classes,
						number_of_classes), dtype=np.float32)

	upsample_kernel = upsample_filt(filter_size)

	for i in xrange(number_of_classes): 
		weights[:, :, i, i] = upsample_kernel

	return weights



if __name__ == '__main__':
	root = '../data/VOCdevkit/VOC2011/'
	im_name = root + 'JPEGImages/2007_000033.jpg'
	seg_name = root + 'SegmentationClass/2007_000033.png'
	_, rgb_to_gray = colormap()

	im_blob, seg_blob = prep_im_for_blob(im_name, seg_name, rgb_to_gray)
	import matplotlib.pyplot as plt
	plt.imshow(im_blob)
	plt.show()
	plt.imshow(seg_blob)
	plt.show()
	ipdb.set_trace()
