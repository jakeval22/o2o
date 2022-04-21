import numpy as np
import cv2

cap = cv2.VideoCapture(1)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
count = 0
print("press (s) to save image:")
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("cal_imgs/frame%d.jpg" % count, frame)
            print("image: ", count, " saved")
            count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

out.release()

cv2.destroyAllWindows()