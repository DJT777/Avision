"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
import mysql.connector
from mysql.connector import pooling
from mysql.connector import Error

import pymysql
from DBUtils.PooledDB import PooledDB

from datetime import date

#import local classes and their functions
from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins_tracklite_mysql import TrtYOLO
from tracklite.utils.parser import get_config
from tracklite.utils.draw import draw_boxes
from threading import Thread, Lock

WINDOW_NAME = 'TrtYOLODemo'


'''mySQLConnectionPool = PooledDB(creator = pymysql,
                               host = '192.168.1.131',
                               user = 'dylan',
                               password = 'pw',
                               database = 'bird_detections',
                               autocommit = True,
                               charset = 'utf8mb4',
                               #cursorclass = pymysql.cursors.DictCursor,
                               blocking = False,
                               maxconnections = 200,
)'''

connection_pool = mysql.connector.pooling.MySQLConnectionPool(pool_name = 'pool',
                                                              pool_size = 32,
                                                              pool_reset_session = True,
                                                              host = '192.168.1.131',
                                                              database = 'bird_detections',
                                                              user = 'dylan',
                                                              autocommit = True,
                                                              password = 'cookies')

'''



database = mysql.connector.connect(
    host='192.168.1.131',
    user='dylan',
    password='cookies',
    database='bird_detections'
)

cursor = database.cursor()'''



sql_op = 'INSERT INTO detections (DATE, TIME, CONFIDENCE, IDENTITY, IMG) VALUES (%s, %s, %s, %s, %s)'

mutex = Lock()

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
    today = date.today()
    #full_screen is set to false by default
    full_scrn = False
    #fps is set at 0 by default
    fps = 0.0
    #create time variable for measuring the frames per second in real time
    tic = time.time()
    #while loop to perform inference
    while True:
        #mutex.acquire()
        today_formatted = today.strftime("%y-%m-%d")
        time_formatted = time.strftime("%H:%M:%S")
        #determine if window is closed or not ????
        #break the loop if window is closed
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        #create img object from a reading of the camera frame
        img = cam.read()
        #image_blob = cv2.imencode('.jpg', img)[1].tostring()
        #break loop if the camera frame is none
        if img is None:
            break
        #create bounding box coordinate, detection confidence, and class id from the detect function of the trt_yolo object.
        img, outputs, scores = trt_yolo.detect(img, conf_th)
        img_raw = cam.read()
        t = Thread(target=mysql_insert, args=(outputs, scores, today_formatted, time_formatted, img_raw))
        t.start()
        #img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            img = draw_boxes(img, bbox_xyxy, identities)
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
        t.join()

def mysql_insert(bboxes, scores, today, time, raw_img):
    if len(bboxes) > 0:
        mySQLConnection = connection_pool.get_connection()
        mySQLCursor = mySQLConnection.cursor()
        list_of_data = []
        bbox_xyxy = bboxes[:, :4]
        identities = bboxes[:, -1]
        for box, identity, score in zip(bbox_xyxy, identities, scores):
            if score > .9 :
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                identity = str(identity)
                confidence = str(score)
                img_cropped = raw_img[y1:y2, x1:x2]
                img_blob = cv2.imencode('.jpg', img_cropped)[1].tostring()
                cv2.imshow("cropped", img_cropped)
                data_tuple = (today, time, confidence, identity, img_blob)
                list_of_data.append(data_tuple)
        if len(list_of_data) > 0:
            mySQLCursor.executemany(sql_op, list_of_data)
        mySQLCursor.close()
        mySQLConnection.close()
    return


def mysql_insert_old( bbox_xyxy, scores, today, time, img_blob):
    #mutex.acquire()
    mySQLConnection = connection_pool.get_connection()
    mySQLCursor = mySQLConnection.cursor()
    copy_bbox = bbox_xyxy
    copy_scores = scores
    copy_today= today
    copy_time = time
    copy_img_blob = img_blob
    #mutex.release()
    if len(copy_bbox) > 0:
        bbox_xyxy = copy_bbox[:, :4]
        identities = copy_bbox[:, -1]
        for box, identity, score in zip(bbox_xyxy, identities, copy_scores):
            if score > .9 :
                x1 = str(box[0])
                y1 = str(box[1])
                x2 = str(box[2])
                y2 = str(box[3])
                identity = str(identity)
                confidence = str(score)
                mySQLCursor.execute(sql_op, (copy_today, copy_time, confidence, identity, x1, y1, x2, y2, copy_img_blob))
                mySQLConnection.commit()
    mySQLCursor.close()
    mySQLConnection.close()
    return
    #cursor.close()
    #conn.close()


def test_insert():
    sql_insert = 'INSERT INTO test_table(TEST_STRING) VALUES (%s)'
    string_to_insert = 'TEST'
    cursor.execute(sql_insert, (string_to_insert))

def main():
    cfg_file = "./tracklite/configs/deep_sort.yaml"
    cfg = get_config()
    cfg = cfg.merge_from_file(cfg_file)
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


#Decoding Image From SQL:
#nparr = np.fromstring(STRING_FROM_DATABASE, np.uint8)
#img = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
