import cv2
import numpy
from skimage.metrics import structural_similarity as ssim

class MatchImages:
	def matchLight(self, imageFirst, imageSecond):
    	# read the images
		image1 = cv2.imread(imageFirst)
		image2 = cv2.imread(imageSecond)

		# Color of the images "Gray"
		image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
		image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

		# Size of the images 500x300
		image1 = cv2.resize(image1, (800, 500))
		image2 = cv2.resize(image2, (800, 500)) 

		# Procentage of the images
		similarityValue = "{:.2f}".format(ssim(image1, image2) * 100)
		return round(float(similarityValue))
	
	def matchDeep(self, firstImage, secondImage):
		pass
