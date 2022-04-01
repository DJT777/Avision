"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

#import local classes and their functions
from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_deepsort_with_plugins import TrtYOLO

#Deepsort imports
from tracklite.utils.parser import get_config


WINDOW_NAME = 'TrtYOLODemo'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    parser.add_argument('--config_deepsort', type=str, default="/home/dylan/Programming-Projects/tensorrt_demos/tracklite/configs/deep_sort.yaml")
    args = parser.parse_args()

    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """

    #full_screen is set to false by default
    full_scrn = False
    #fps is set at 0 by default
    fps = 0.0
    #create time variable for measuring the frames per second in real time
    tic = time.time()
    #while loop to perform inference
    while True:
        #determine if window is closed or not ????
        #break the loop if window is closed
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        #create img object from a reading of the camera frame
        img = cam.read()
        #break loop if the camera frame is none
        if img is None:
            break
        #create bounding box coordinate, detection confidence, and class id from the detect function of the trt_yolo object.
        img = trt_yolo.detect(img, conf_th)
        #img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    #parse arguments
    args = parse_args()

    # Deepsort Config File For Tracker:
    cfg_file = "./tracklite/configs/deep_sort.yaml"
    cfg = get_config()
    cfg.merge_from_file(cfg_file)

    #raise errors for lack of arguments, such as the category number and the model file
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    #camera object instantiated with arguments
    cam = Camera(args)
    #raise error if cameras is not opened
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    #create list of classes to be detected
    cls_dict = get_cls_dict(args.category_num)
    #instantiate vis object with class_dict passed as an argument
    #BBOXVisualization contains code to draw boxes and assign colors to each class
    vis = BBoxVisualization(cls_dict)
    #instantiate the TtrYOLO object based on the arguments given in the command to start trt_yolo.py
    trt_yolo = TrtYOLO(args.model, cfg, args.category_num, args.letter_box)

    #open a window based on camera height and width
    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)

    #loop and perform detections
    loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
