
import pymysql
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import csv

model = tf.keras.models.load_model('//home/dylan/Desktop/backup_oldubuntu/working-dir/saved-models/vgg16-325/model_fine_tuned/')
try:
        connection = pymysql.connect(host='127.0.0.1',
                                             database='nano_detections',
                                             user='dylan',
                                             password='cookies',
                                             )
        #get class list
        class_array = []
        file = open("/home/dylan/Desktop/backup_oldubuntu/datasets/325/class_dict.csv")
        csv_reader = csv.reader(file)
        next(csv_reader)
        for line in csv_reader:
            class_array.append(line[1])
        print("LENGTH OF ARRAY: " + str(len(class_array)))
        print(class_array)
        cursor = connection.cursor()
        #cursor2 = connection.cursor()
        sql_select_query = """select * from detections where IS_CLASSIFIED = 0"""

        # set variable in query
        cursor.execute(sql_select_query)
        # fetch result
        record = cursor.fetchall()
        print(len(record))
        for row in record:
            id = row[0]
            date = row[1]
            time = row[2]
            conf = row[3]
            track_id = row[4]
            img_blob = row[5]
            nparr = np.fromstring(img_blob, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            resized_image = cv2.resize(img, (224 , 224))
            resized_image = resized_image.reshape(1,224,224,3)
            predict_x = model.predict(resized_image)
            #print("Length of prediction array: " + len(predict_x))
            classes_x = np.argmax(predict_x)
            #class_index = classes_x[0]
            #print(class_index)
            predicted_class = class_array[classes_x]
            #predictions = model.predict(resized_image)
            print("CLASS INDEX:" + str(classes_x))
            print("CLASS PREDICTED: " + predicted_class)
            print(predicted_class)
            #cv2.imshow('image', img)
            #cv2.imshow('resized-image', resized_image)
            #cv2.waitKey()
            sql_update_query = "UPDATE detections SET IS_CLASSIFIED = TRUE, SPECIES = %s WHERE PRIMARY_KEY = %s "
            cursor.execute(sql_update_query,  (predicted_class, id))
            connection.commit()

except Exception as e:
    print("Error reading data from MySQL table", e)
finally:
   if connection.open:
    connection.close()
       ##cursor2.close()
    cursor.close()
    print("MySQL connection is closed")