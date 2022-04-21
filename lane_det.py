import cv2
import numpy as np

# call on file to undistort camera
npzfile = np.load('calibrationdata.npz')
mtx = npzfile['mtx']
dist = npzfile['dist']
rvecs = npzfile['rvecs']
tvecs = npzfile['tvecs']


def roi_transform(image, ul, ur, ll, lr):
    """
    function for showing region of interest given the 4 coordinates (upper left, upper right, lower left, lower right)
    """
    point_matrix = np.float32([ul, ur, ll, lr])
    width = image.shape[0]
    height = image.shape[1]
    new_ul = [0, 0]
    new_ur = [width, 0]
    new_ll = [0, height]
    new_lr = [width, height]
    converted_points = np.float32([new_ul, new_ur, new_ll, new_lr])
    perspective_transform = cv2.getPerspectiveTransform(point_matrix, converted_points)
    img_Output = cv2.warpPerspective(image, perspective_transform, (width, height))
    return img_Output


def pipeline(image):
    """
    full pipeline for lane detection and outputs
    """

    cv2.imshow('Original image', image)
    cv2.imwrite('out_img/org_img.jpg', image)

    #filter for blue color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([60, 35, 140])
    upper_blue = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('blue filt', result)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 30, 80)
    cv2.imshow('edge', edge)
    cv2.imwrite('out_img/edge_img.jpg', edge)
    roi_img = roi_transform(edge, [159, 154], [481, 154], [0, 480], [640, 480])
    lane_detx = roi_transform(image, [159, 154], [481, 154], [0, 480], [640, 480])
    lane_dety = np.copy(lane_detx)
    lane_det = np.copy(lane_detx)
    cv2.imshow('ROI', roi_img)
    cv2.imwrite('out_img/roi_img.jpg', roi_img)

    # creates hough lines (estimates straiht lines from edge image
    lines = cv2.HoughLinesP(
        roi_img,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 150,  # Angle resolution in radians
        threshold=80,  # Min number of votes for valid line
        minLineLength=30,  # Min allowed length of line
        maxLineGap=400  # Max allowed gap between line for joining them
    )

    # get size of roi image
    w = roi_img.shape[0]
    h = roi_img.shape[1]

    # begin finding the right vertical and left vertical lane (probably should be called y_lines, oops)
    xmax = 0
    xmin = h
    x_lines_max = []
    x_lines_min = []
    x_lines = []
    # Iterate over points (list of coordinates) for each line
    for points in lines:
        x1, y1, x2, y2 = points[0]
        # if the change in x value is too large, then it is not a vertical line and should not go in x_lines
        if abs(x1 - x2) < 80:
            # if new line vertical line found further to left or right, that line is what we want to output
            if x1 > xmax:
                xmax = x1
                x_lines_max = ([x1, y1, x2, y2])
            elif x1 < xmin:
                xmin = x1
                x_lines_min = ([x1, y1, x2, y2])
    # x_lines becomes the two lines we want
    if x_lines_max:
        x_lines.append(x_lines_max)
    if x_lines_min:
        x_lines.append(x_lines_min)
    for i in range(len(x_lines)):
        # Extracted points nested in the list
        x1 = x_lines[i][0]
        y1 = x_lines[i][1]
        x2 = x_lines[i][2]
        y2 = x_lines[i][3]
        # Draw the lines on image for vertical lane detection
        cv2.line(lane_detx, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("lane detx", lane_detx)
    cv2.imwrite('out_img/lane_detx.jpg', lane_detx)

    # follow the above code but for horizontal lines and called y_lines
    ymax = 0
    ymin = w
    y_lines_max = []
    y_lines_min = []
    y_lines = []
    for points in lines:
        x1, y1, x2, y2 = points[0]
        # if change in y coordinate is larger than 50 than it is not a horizontal line
        if abs(y1 - y2) < 20:
            # horizontal line must be detected above the bottom of the frame where camera undistortion artifact is
            # and not at very top
            if 10 < y1 < 580:
                if y1 > ymax:
                    ymax = y1
                    y_lines_max = ([x1, y1, x2, y2])
                elif y1 < ymin:
                    ymin = y1
                    y_lines_min = ([x1, y1, x2, y2])
    if y_lines_max:
        y_lines.append(y_lines_max)
    if y_lines_min:
        y_lines.append(y_lines_min)
    for i in range(len(y_lines)):
        # Extracted points nested in the list
        x1 = y_lines[i][0]
        y1 = y_lines[i][1]
        x2 = y_lines[i][2]
        y2 = y_lines[i][3]
        cv2.line(lane_dety, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.imshow("lane dety", lane_dety)
    cv2.imwrite('out_img/lane_dety.jpg', lane_dety)

    straight_av = 0
    # if two vertical lines are found then draw a transparent lane detection
    if len(x_lines) == 2:
        overlay = lane_det.copy()
        # x_oc is the average of the ends of the two horizontal lines
        x_oc = int((x_lines[0][2] + x_lines[1][2]) / 2)
        pts = np.array([[x_lines[0][0], x_lines[0][1]], [x_lines[0][2], x_lines[0][3]], [x_lines[1][0], x_lines[1][1]],
                        [x_lines[1][2], x_lines[1][3]]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        overlay = cv2.fillPoly(overlay, pts=[pts], color=(0, 255, 0))
        alpha = 0.4  # Transparency factor.
        # Following line overlays transparent rectangle over the image
        lane_det = cv2.addWeighted(overlay, alpha, lane_det, 1 - alpha, 0)
        cv2.line(lane_det, (240, 640), (x_oc, 0), (255, 255, 0), 2)
        # angle drawn from center to x_oc and outputted for steering
        angle = np.degrees(np.arctan((320 - x_oc) / 640))
        if (x_lines[0][3] & x_lines[1][3]) > 10:
            stright_av = 1

    # if only one vertical line is detected
    if len(x_lines) == 1:
        # the line is on the left side of the frame, want a positive angle of 15 degrees for steering
        if x_lines[0][0] < 320:
            angle = 15
        # otherwise the line is on the right and we want to steer to the left (negative angle)
        else:
            angle = -15

    # if no vertical lines detected then there is no angle information so angle cannot be calculated
    if len(x_lines) == 0:
        angle = 0

    # if there are two horizontal lines then fill in a polygon showing where the horizontal lane is detected
    left_turn_av = 0
    right_turn_av = 0
    if len(y_lines) == 2:
        overlay = lane_det.copy()
        pts = np.array([[y_lines[1][2], y_lines[1][3]], [y_lines[1][0], y_lines[1][1]], [y_lines[0][0], y_lines[0][1]],
                        [y_lines[0][2], y_lines[0][3]]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        overlay = cv2.fillPoly(overlay, pts=[pts], color=(255, 255, 0))
        alpha = 0.4
        lane_det = cv2.addWeighted(overlay, alpha, lane_det, 1 - alpha, 0)
        # if horizontal lane is on left side, left turn available
        if (y_lines[1][0] | y_lines[0][0]) < 20:
            left_turn_av = 1
        # if horizontal lane is on right side, right turn available
        if (y_lines[1][2] | y_lines[0][2]) > 300:
            right_turn_av = 1

    # if vertical lines go past horizontal lines then straight is available

    print("left turn available: ", left_turn_av)
    print("right turn available: ", right_turn_av)
    print("straight available: ", straight_av)
    print("angle: ", angle)
    # has filled in polygons of where lanes are detected
    cv2.imshow("lane det", lane_det)
    cv2.imwrite('out_img/lane_det.jpg', lane_det)


"""
# define a video capture object
vid = cv2.VideoCapture(1)

while (True):
    ret, frame = vid.read()
    h, w = frame.shape[:2]
    # undistort image
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistortedImg = cv2.undistort(frame, mtx, dist, None, newCameraMtx)
    # lane det pipeline
    pipeline(undistortedImg)
    cv2.imwrite('frame.jpg', undistortedImg)
    # press q to end stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
"""


# run below to run lane_det on an image

import cv2
img = cv2.imread('track_pic_3.jpg') # load a dummy image
while(1):
    #img = cv2.resize(img, (640,480), interpolation= cv2.INTER_LINEAR)
    pipeline(img)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break

