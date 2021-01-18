# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from skimage import io
import matplotlib.pyplot as plt

def eye_histogram(img):
	# construct the argument parser and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-p", "--shape-predictor", required=True,
	# 	help="path to facial landmark predictor")
	# ap.add_argument("-i", "--image", required=True,
	# 	help="path to input image")
	# args = vars(ap.parse_args())
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	# predictor = dlib.shape_predictor(args["shape_predictor"])
	#mandy:
	predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
	# load the input image, resize it, and convert it to grayscale
	# image = cv2.imread(args["image"])
	#mandy:
	image = cv2.imread(img)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# ax = plt.hist(image.ravel(), bins = 256)
	# plt.show()

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# loop over the face parts individually
		# for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			# clone the original image so we can draw on it, then
			# display the name of the face part on the image
			# if (name != "left_eye"):
			# 	break
		clone = image.copy()
		cv2.putText(clone, "right_eye", (36, 42), cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)
		# loop over the subset of facial landmarks, drawing the
		# specific face part
		for (x, y) in shape[36:42]:
			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

			# extract the ROI of the face region as a separate image
			(x, y, w, h) = cv2.boundingRect(np.array([shape[36:42]]))
			roi = image[y:y + h, x:x + w]
			roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
			# show the particular face part
			# cv2.imshow("ROI", roi)
			# cv2.imshow("Image", clone)
			# # image.show()
			# cv2.startWindowThread()
			# cv2.namedWindow("preview")
			# cv2.imshow("preview", image)
			ax = plt.hist(roi.ravel(), bins=256)
			right_unique = np.unique(roi.ravel())
			# print('uniques right:', right_unique)
			# print('num of uniques right:', len(right_unique))

			plt.title('right eye')
			# #show:
			# plt.show()
			# cv2.waitKey(0)
			break

		clone = image.copy()
		cv2.putText(clone, "left_eye", (42, 48), cv2.FONT_HERSHEY_SIMPLEX,
					0.7, (0, 0, 255), 2)

		for (x, y) in shape[42:48]:

			# extract the ROI of the face region as a separate image
			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
			(x, y, w, h) = cv2.boundingRect(np.array([shape[42:48]]))
			roi = image[y:y + h, x:x + w]
			roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
			# show the particular face part
			# cv2.imshow("ROI", roi)
			# cv2.imshow("Image", clone)
			# # image.show()
			# cv2.startWindowThread()
			# cv2.namedWindow("preview")
			# cv2.imshow("preview", image)
			ax = plt.hist(roi.ravel(), bins=256)
			left_unique = np.unique(roi.ravel())
			# print('uniques left:', left_unique)
			# print('num of uniques left:', len(left_unique))

			plt.title('left eye')
			# show:
			# plt.show()
			# cv2.waitKey(0)
			break

		#if there is more than eye_treshold unique pixels in a eye ROI, it means there is a eye (TRUE), otherewise - the eye is closed/glasses (FALSE)
		eye_treshold = 90
		if ((len(left_unique) > eye_treshold) or (len(right_unique) > eye_treshold )):
			return True
		else:
			return False
		# visualize all facial landmarks with a transparent overlay
		# output = face_utils.visualize_facial_landmarks(image, shape)
		# cv2.imshow("Image", output)
		# cv2.waitKey(0)