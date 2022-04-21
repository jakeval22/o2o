import cv2
import numpy as np

npzfile = np.load('calibrationdata.npz')
mtx = npzfile['mtx']
dist = npzfile['dist']
rvecs = npzfile['rvecs']
tvecs = npzfile['tvecs']



cap = cv2.VideoCapture(1) # video capture source camera (Here webcam of laptop)

# reading the input using the camera
result, image = cap.read()

# If image will detected without any error,
# show result
if result:
	# showing result, it take frame name and image
	# output
	h, w = image.shape[:2]
	newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
	undistortedImg = cv2.undistort(image, mtx, dist, None, newCameraMtx)
	cv2.imwrite("track_pic_2.jpg", undistortedImg)
	# If keyboard interrupt occurs, destroy image
	# window
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# If captured image is corrupted, moving to else part
else:
	print("No image detected. Please! try again")