# Define some util functions
# Author: Yuliang Zou
#         ylzou@umich.edu
# Date:   2017-02-19

import numpy as np
import cv2
import ipdb

# BGR mean pixel value
MEAN_PIXEL = np.array([103.939, 116.779, 123.68])

"""Padding image and segmentation ground truth to (600, 600)"""
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

	return im_blob, seg_blob, mask


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
