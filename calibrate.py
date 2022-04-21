import cv2
import numpy as np

npzfile = np.load('calibrationdata.npz')
print(npzfile.files)
mtx = npzfile['mtx']
dist = npzfile['dist']
rvecs = npzfile['rvecs']
tvecs = npzfile['tvecs']



# Kalibrasyon verilerini kullanarak görüntüyü düzeltelim
cap = cv2.VideoCapture(1)
while True:
    # Gri renk uzayında frame okuyalım
    ret3, img = cap.read()
    h, w = img.shape[:2]

    '''
    Kalibrasyon ile elde ettiğimiz K ve D verileri ve OpenCV undistort* metotlarını kullanarak 
    bozuk görüntüyü düzeltebiliriz. Bu metotlar parametre olarak K ve D verilerine ihtiyaç duyar, 
    sonuç olarak ise size düzeltilmiş görüntü dönecektir.
    '''
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)

    cv2.imshow('Duzeltilmis Goruntu', undistortedImg)
    cv2.waitKey(0)

cv2.destroyAllWindows()