import argparse
import os, sys
import shutil
import time
from pathlib import Path
import moviepy.video.io.ImageSequenceClip
import numpy as np
import imageio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm
import time
import can

"""
bustype = 'socketcan'
channel = 'can0'

stop_flag = 0

bus = can.Bus(channel=channel,interface=bustype)
"""
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])




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

def send_message(speed, angle):
    msg = can.Message(arbitration_id=0xc0ffee, data=[speed, angle], is_extended_id=False)
    bus.send(msg)
    print(can.interfaces.socketcan.socketcan.build_can_frame(msg))
    print(f"Speed: {speed}\nAngle: {angle}")
    return

#send_message(40, 30)

def pipeline(image):
    """
    full pipeline for lane detection and outputs
    """

    cv2.imshow('Original image', image)
    cv2.imwrite('out_img/org_img.jpg', image)
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    edge = cv2.Canny(blur, 10, 50)
    cv2.imwrite('out_img/edge_img.jpg', edge)
    #roi_img = roi_transform(edge, [555, 487], [695, 487], [2, 662], [930, 716])
    roi_img = roi_transform(edge)
    #lane_detx = cv2.cvtColor(roi_transform(image, [555, 487], [695, 487], [2, 662], [930, 716]), cv2.COLOR_GRAY2BGR)
    lane_detx = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
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
        minLineLength=100,  # Min allowed length of line
        maxLineGap=100  # Max allowed gap between line for joining them
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
        if abs(y1 - y2) > 200:
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
    print(h,w)
    # follow the above code but for horizontal lines and called y_lines
    ymax = 0
    ymin = w
    y_lines_max = []
    y_lines_min = []
    y_lines = []
    for points in lines:
        x1, y1, x2, y2 = points[0]
        # if change in y coordinate is larger than 50 than it is not a horizontal line
        if abs(y1 - y2) < 50:
            # horizontal line must be detected above the bottom of the frame where camera undistortion artifact is
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
        cv2.line(lane_det, (360, 1280), (x_oc, 0), (255, 255, 0), 2)
        # angle drawn from center to x_oc and outputted for steering
        angle = np.degrees(np.arctan(((h/2) - x_oc) / h))
        print("angle: ", angle)

    # if only one vertical line is detected
    if len(x_lines) == 1:
        # the line is on the left side of the frame, want a positive angle of 15 degrees for steering
        if x_lines[0][0] < 320:
            angle = 15
        # otherwise the line is on the right and we want to steer to the left (negative angle)
        else:
            angle = -15
        print("angle: ", angle)

    # if no vertical lines detected then there is no angle information so angle cannot be calculated
    if len(x_lines) == 0:
        angle = 0
        print("angle: ", angle)

    # if there are two horizontal lines then fill in a polygon showing where the horizontal lane is detected
    if len(y_lines) == 2:
        overlay = lane_det.copy()
        pts = np.array([[y_lines[1][2], y_lines[1][3]], [y_lines[1][0], y_lines[1][1]], [y_lines[0][0], y_lines[0][1]],
                        [y_lines[0][2], y_lines[0][3]]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        overlay = cv2.fillPoly(overlay, pts=[pts], color=(255, 255, 0))
        alpha = 0.4
        lane_det = cv2.addWeighted(overlay, alpha, lane_det, 1 - alpha, 0)

    # has filled in polygons of where lanes are detected
    cv2.imshow("lane det", lane_det)
    cv2.imwrite('out_img/lane_det.jpg', lane_det)
    return lane_det, lane_detx, lane_dety

def detect(cfg,opt):
    id = 0
    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger,opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out,ll_seg_out= model(img)
        t2 = time_synchronized()
        # if i == 0:
        #     print(det_out)
        inf_out, _ = det_out
        inf_time.update(t2-t1,img.size(0))

        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        t4 = time_synchronized()

        nms_time.update(t4-t3,img.size(0))
        det=det_pred[0]

        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")
        save_path1 = str(opt.save_dir +'/'+ 'seg_' + Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)


        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
        ll_seg_mask = connect_lane(ll_seg_mask)

        img_det, seg_img = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

        #lane_det, lane_detx, lane_dety = pipeline(seg_img)
        can_turn_right = 0
        can_turn_left = 0
        can_go_straight = 0
        right_sec = seg_img[500:520, 1180:1280]
        if np.sum(right_sec) > 850*85:
            can_turn_right = 1
        left_sec = seg_img[400:520, 0:100]
        if np.sum(left_sec) > 750*85:
            can_turn_left = 1
        middle_sec = seg_img[400:720, 420:860]
        if np.sum(middle_sec) > 450*85:
            can_go_straight = 1

        stop = 0

        if len(det):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
            for *xyxy,conf,cls in reversed(det):
                class_label = ((torch.tensor(cls).view(1, 1)).view(-1).tolist())
                coords = ((torch.tensor(xyxy).view(1, 4)).view(-1).tolist())
                print(class_label)
                if class_label[0] == 0:
                    if coords[2] > 1100:
                        stop = 1
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
        """
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s',)
        yolo_model.iou = 0.5
        # Inference
        results = yolo_model(img_det)
        labels = results.pandas().xyxy[0]
        class_id = labels['name']
        for i in class_id:
            print("i: ", i)
            if i == 'stop sign':
                stop = 1
        results.show()
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_det, "Can turn left: " + str(can_turn_left), (1000, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_det, "Can turn right: " + str(can_turn_right), (1000, 150), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_det, "Can go straight: " + str(can_go_straight), (980, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_det, "STOP: " + str(stop), (80, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)


        """
        if stop == 1:
            stop_flag = 0
        if stop_flag == 1:
            if stop == 0:
                send_message(0, 30)
                time.sleep(5)
                send_message(40, 30)
                stop_flag = 0

        if can_turn_left == 1:
            send_message(50, 60)

        if can_turn_right == 1:
            if can_turn_left == 0:
                send_message(50, 0)

        if can_go_straight == 1:
            if can_turn_left == 0:
                if can_turn_right == 0:
                    send_message(50,30)
                    
        """

        if dataset.mode == 'images':
            cv2.imwrite(save_path,img_det)
            cv2.imwrite(save_path1,seg_img)

        elif dataset.mode == 'video' :
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h,w,_=img_det.shape
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img_det)
            cv2.imshow('image', img_det)
            cv2.waitKey(1)

        elif dataset.mode == 'stream':
            cv2.imwrite(("inference/output/frame" + str(id) + ".jpg"),img_det)
            cv2.imshow('frame', img_det)
            id += 1
            if cv2.waitKey(1) & 0xFF == ord('a'):
                break




    else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)  # 1 millisecond

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))

    image_folder = 'inference/output'
    image_files = [os.path.join(image_folder, img)
                    for img in os.listdir(image_folder)
                    if img.endswith(".jpg")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=15)
    clip.write_videofile('my_video.mp4')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)
