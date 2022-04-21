# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/take_img_for_cal.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 10, 50)
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
        threshold=100,  # Min number of votes for valid line
        minLineLength=60,  # Min allowed length of line
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
        if abs(y1 - y2) < 50:
            # horizontal line must be detected above the bottom of the frame where camera undistortion artifact is
            if y1 < 580:
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
        cv2.line(lane_det, (240, 640), (x_oc, 0), (255, 255, 0), 2)
        # angle drawn from center to x_oc and outputted for steering
        angle = np.degrees(np.arctan((320 - x_oc) / 640))
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
        can_turn = 1

    if len(y_lines) == 1:
        if can_turn == 1:
            can_turn = 0


    # has filled in polygons of where lanes are detected
    cv2.imshow("lane det", lane_det)
    cv2.imwrite('out_img/lane_det.jpg', lane_det)

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            h, w = im0.shape[:2]
            newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            undistortedImg = cv2.undistort(im0, mtx, dist, None, newCameraMtx)
            cv2.imshow("lane detection", undistortedImg)
            pipeline(undistortedImg)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    print((torch.tensor(cls).view(1, 1)).view(-1).tolist())
                    print((torch.tensor(xyxy).view(1, 4)).view(-1).tolist())
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond


            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
