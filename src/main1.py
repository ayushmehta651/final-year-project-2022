import os
import cv2
from WordSegmentation import wordSegmentation, prepareImg


def main():
        # read image, prepare it by resizing it to fixed height and converting it to grayscale
	img = prepareImg(cv2.imread('../data/0.png'), 50)
		
	# execute segmentation with given parameters
	# -kernelSize: size of filter kernel (odd integer)
	# -sigma: standard deviation of Gaussian function used for filter kernel
	# -theta: approximated width/height ratio of words, filter function is distorted by this factor
	# - minArea: ignore word candidates smaller than specified area
	res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
		
	# write output to 'out/inputFileName' directory
		# iterate over all segmented words
	print('Segmented into %d words'%len(res))
	for (j, w) in enumerate(res):
		(wordBox, wordImg) = w
		(x, y, w, h) = wordBox
		cv2.imshow('dkdk',wordImg)
		cv2.waitKey(0)
		#cv2.imwrite('../out/%s/%d.png'%(f, j), wordImg) # save word
		#cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
		
		# output summary image with bounding boxes around words
		#cv2.imwrite('../out/%s/summary.png'%f, img)


if __name__ == '__main__':
	main()
