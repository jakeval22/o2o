# importing the module
import cv2
import numpy as np

def roi_transform(image,ul,ur,ll,lr):
    point_matrix = np.float32([ul, ur, ll, lr])
    width = image.shape[0]
    height = image.shape[1]
    new_ul = [0, 0]
    new_ur = [width, 0]
    new_ll = [0, height]
    new_lr = [width, height]
    converted_points = np.float32([new_ul,new_ur,new_ll,new_lr])
    perspective_transform = cv2.getPerspectiveTransform(point_matrix, converted_points)
    img_Output = cv2.warpPerspective(image, perspective_transform, (width, height))
    return img_Output

npzfile = np.load('calibrationdata.npz')
mtx = npzfile['mtx']
dist = npzfile['dist']
rvecs = npzfile['rvecs']
tvecs = npzfile['tvecs']


# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):

	# checking for left mouse clicks
	if event == cv2.EVENT_LBUTTONDOWN:
		# displaying the coordinates
		# on the Shell
		print(x, ' ', y)

		# displaying the coordinates
		# on the image window
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img, str(x) + ',' +
					str(y), (x,y), font,
					1, (255, 0, 0), 2)
		cv2.imshow('image', img)

	# checking for right mouse clicks
	if event==cv2.EVENT_RBUTTONDOWN:

		# displaying the coordinates
		# on the Shell
		print(x, ' ', y)

		# displaying the coordinates
		# on the image window
		font = cv2.FONT_HERSHEY_SIMPLEX
		b = img[y, x, 0]
		g = img[y, x, 1]
		r = img[y, x, 2]
		cv2.putText(img, str(b) + ',' +
					str(g) + ',' + str(r),
					(x,y), font, 1,
					(255, 255, 0), 2)
		cv2.imshow('image', img)



cap = cv2.VideoCapture(1)

result, img = cap.read()

if result:
	img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
	cv2.imshow('image', img)
	h, w = img.shape[:2]
	newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
	undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)
	img = roi_transform(undistortedImg, [159, 154], [481, 154], [0, 480], [640, 480])
	cv2.imshow('undistort', img)
	cv2.setMouseCallback('undistort', click_event)



	cv2.waitKey(0)
	cv2.destroyAllWindows()


