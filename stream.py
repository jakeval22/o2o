# import the opencv library
import cv2
import numpy as np

npzfile = np.load('calibrationdata.npz')
mtx = npzfile['mtx']
dist = npzfile['dist']
rvecs = npzfile['rvecs']
tvecs = npzfile['tvecs']


vid = cv2.VideoCapture(1)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    h, w = frame.shape[:2]
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistortedImg = cv2.undistort(frame, mtx, dist, None, newCameraMtx)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('undistort', undistortedImg)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
