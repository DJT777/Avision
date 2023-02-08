"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse
from datetime import date
from threading import Thread, Lock


import mysql.connector
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO


WINDOW_NAME = 'TrtYOLODemo'

database = mysql.connector.connect(
    host='192.168.1.131',
    user='dylan',
    password='pw',
    database='bird_detections'
)

cursor = database.cursor()

sql_op = 'INSERT INTO birds_no_tracking (DATE, TIME, CONFIDENCE, X1, Y1, X2, Y2, IMAGE) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'

mutex = Lock()

def insert_sql(boxes, confs, clss, today, time, img_blob):
    #mutex.acquire()
    copy_boxes = boxes
    copy_confs = confs
    copy_clss = clss
    copy_today = today
    copy_time = time
    #mutex.release()
    for data_box, data_confs, data_clss in zip(copy_boxes, copy_confs, copy_clss):
     if data_confs > 0:  #
        x1 = data_box[0]
        y1 = data_box[1]
        x2 = data_box[2]
        y2 = data_box[3]
        cursor.execute(sql_op, (copy_today, copy_time, str(data_confs), str(x1), str(y1), str(x2), str(y2), img_blob))
        database.commit()

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


def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    today = date.today()
    today_formatted = today.strftime("%y-%m-%d")
    tic = time.time()

    while True:
        today_formatted = today.strftime("%y-%m-%d")
        time_formatted = time.strftime("%H:%M:%S")
        mutex.acquire()
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        image_blob = cv2.imencode('.jpg', img)[1].tostring()
        if img is None:
            break
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        insert_sql(boxes, confs, clss, today_formatted, time_formatted, image_blob)
        #data_structure_class = data_structure(boxes, confs, clss, today_formatted, time_formatted, image_blob)
        #data_list = []
        #data_list.append(data_structure_class)
        #mutex.release()
        #t = Thread(target=insert_sql, args=(boxes, confs, clss, today_formatted, time_formatted, image_blob))
        #t.start()
        #mutex.acquire()
        img = vis.draw_bboxes(img, boxes, confs, clss)
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
        #mutex.release()

def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)

    loop_and_detect(cam, trt_yolo, conf_th=0.35, vis=vis)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


#Decoding Image:
#nparr = np.fromstring(STRING_FROM_DATABASE, np.uint8)
#img = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
