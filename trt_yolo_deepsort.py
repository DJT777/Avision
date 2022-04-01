"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse

import cv2
import numpy as np

import pycuda.autoinit  # This is needed for initializing CUDA driver

#import local classes and their functions
from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

import matplotlib.pyplot as plt

from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

WINDOW_NAME = 'TrtYOLODemo'


class_names = [c.strip() for c in open('./deep_sort/labels/coco.names').readlines()]

np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

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
    args = parser.parse_args()
    return args

def convert_tlbr_tlwh(bboxes):
    list_of_boxes = []
    test_array = np.zeros(4)
    for box in bboxes:
        box_list = []
        x, y, x2, y2 = box[0], box[1], box[2], box[3]
        w = x2 - x
        h = y2 - y
        box_list.append(x)
        box_list.append(y)
        box_list.append(w)
        box_list.append(h)
        if not np.array_equal(box, [0,0,0,0]):
            list_of_boxes.append(box_list)
    return list_of_boxes




def loop_and_detect(cam, encoder, tracker, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    counter = []
    #full_screen is set to false by default
    full_scrn = False
    #fps is set at 0 by default
    fps = 0.0
    #create time variable for measuring the frames per second in real time
    tic = time.time()
    #while loop to perform inference
    while True:
        t1 = time.time()
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
        boxes, confs, clss = trt_yolo.detect(img, conf_th)

        classes = clss
        names = []
        ##for i in range(len(classes)):
         #   names.append(class_names[int(classes[i])])

        xywh_boxes = convert_tlbr_tlwh(boxes)
        features = encoder(img, xywh_boxes)

        detections = [Detection(bbox, confs, d_clss, feature) for bbox, confs, d_clss, feature in zip(xywh_boxes, confs, clss, features)]
        # Run non-maxima suppression.
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        nms_max_overlap = 0.3
        indices = preprocessing.non_max_suppression(boxs, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        current_count = int(0)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = 'bird' #track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]

            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)), (int(bbox[0]) + (len(class_name)
                                                                                   + len(str(track.track_id))) * 17,
                                                                   int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)

            '''for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(img, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), color, thickness)'''

            height, width, _ = img.shape
            cv2.line(img, (0, int(3 * height / 6 + height / 20)), (width, int(3 * height / 6 + height / 20)),
                     (0, 255, 0), thickness=2)
            cv2.line(img, (0, int(3 * height / 6 - height / 20)), (width, int(3 * height / 6 - height / 20)),
                     (0, 255, 0), thickness=2)

            center_y = int(((bbox[1]) + (bbox[3])) / 2)

            if center_y <= int(3 * height / 6 + height / 20) and center_y >= int(3 * height / 6 - height / 20):
                if class_name == 'bird' or class_name == '1':
                    counter.append(int(track.track_id))
                    current_count += 1

        total_count = len(set(counter))
        cv2.putText(img, "Current Bird Count: " + str(current_count), (0, 80), 0, 1, (0, 0, 255), 2)
        cv2.putText(img, "Total Bird Count: " + str(total_count), (0, 130), 0, 1, (0, 0, 255), 2)

        fps = 1. / (time.time() - t1)
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30), 0, 1, (0, 0, 255), 2)
        cv2.resizeWindow('output', 1024, 768)
        cv2.imshow('output', img)

        #img = vis.draw_bboxes(img, boxes, confs, clss)
        #img = show_fps(img, fps)
        #cv2.imshow(WINDOW_NAME, img)
        # toc = time.time()
        #curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        #fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        #tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():



    #parse arguments
    args = parse_args()
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

    #Create DeepSort Encoder
    model_filename = './deep_sort_yolov3/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 0.3
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)



    #create list of classes to be detected
    cls_dict = get_cls_dict(args.category_num)
    #instantiate vis object with class_dict passed as an argument
    #BBOXVisualization contains code to draw boxes and assign colors to each class
    vis = BBoxVisualization(cls_dict)
    #instantiate the TtrYOLO object based on the arguments given in the command to start trt_yolo.py
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    #open a window based on camera height and width
    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)

    #loop and perform detections
    loop_and_detect(cam, encoder, tracker, trt_yolo, conf_th=0.3, vis=vis)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
