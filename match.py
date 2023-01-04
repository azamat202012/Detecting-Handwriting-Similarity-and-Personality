from __future__ import print_function
from math import sqrt

import cv2
import numpy
from skimage.metrics import structural_similarity as ssim

class MatchImages:
	def matchLight(self, imageFirst, imageSecond):
    	# Images
		sampleOfTheImageOne = cv2.imread(cv2.samples.findFile(imageFirst), cv2.IMREAD_GRAYSCALE)
		sampleOfTheImageTwo = cv2.imread(cv2.samples.findFile(imageSecond), cv2.IMREAD_GRAYSCALE)

		# Size of the images 500x300
		image1 = cv2.resize(image1, (800, 500))
		image2 = cv2.resize(image2, (800, 500)) 

		# Procentage of the images
		similarityValue = "{:.2f}".format(ssim(image1, image2) * 100)

		# Return results
		return round(float(similarityValue))
	
	def matchDeep(self, imageFirst, imageSecond):
		# Load images
		sampleOfTheImageOne = cv2.imread(cv2.samples.findFile(imageFirst), cv2.IMREAD_GRAYSCALE)
		sampleOfTheImageTwo = cv2.imread(cv2.samples.findFile(imageSecond), cv2.IMREAD_GRAYSCALE)

		# Check if images is exist or not
		if imageFirst is None or imageSecond is None:
			print("Could not open or find the images!")
			exit(0)

        # File for homography for images
		fs = cv2.FileStorage(cv2.samples.findFile('./H1to3p.xml'), cv2.FILE_STORAGE_READ)
		homography = fs.getFirstTopLevelNode().mat()

		# AKAZE local features to detect and match keypoints on two images.
		akaze = cv2.AKAZE_create()
		kpts1, desc1 = akaze.detectAndCompute(sampleOfTheImageOne, None)
		kpts2, desc2 = akaze.detectAndCompute(sampleOfTheImageTwo, None)

		# Use brute-force matcher to find 2-nn matches
		matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
		nn_matches = matcher.knnMatch(desc1, desc2, 2)

		# Use 2-nn matches and ratio criterion to find correct keypoint matches
		matched1 = []
		matched2 = []
		nn_match_ratio = 0.8 # Nearest neighbor matching ratio
		for m, n in nn_matches:
			if m.distance < nn_match_ratio * n.distance:
				matched1.append(kpts1[m.queryIdx])
				matched2.append(kpts2[m.trainIdx])

		# Check if our matches fit in the homography model
		inliers1 = []
		inliers2 = []
		good_matches = []
		inlier_threshold = 2.5 # Distance threshold to identify inliers with homography check
		for i, m in enumerate(matched1):
			col = numpy.ones((3,1), dtype=numpy.float64)
			col[0:2, 0] = m.pt

			col = numpy.dot(homography, col)
			col /= col[2, 0]
			dist = sqrt(pow(col[0, 0] - matched2[i].pt[0], 2) + pow(col[1, 0] - matched2[i].pt[1], 2))
			
			if dist < inlier_threshold:
				good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
				inliers1.append(matched1[i])
				inliers2.append(matched2[i])
		
		# Return results
		return len(matched1)