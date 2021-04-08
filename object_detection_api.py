import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pandas as pd
import requests

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# %matplotlib inline

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'infrence_graph'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'labelmap.pbtxt')


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


import cv2

g_bricks = []
b_bricks = []
t_bricks = []
cap = cv2.VideoCapture('5.mp4')
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    Good_bricks = 0
    Bad_Bricks = 0

    while True:
      ret, image_np = cap.read()
       # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
      cv2.line(image_np, (1,400), (1000,400), (0,0,255), 2) # RED Line
      cv2.line(image_np, (1,370), (1000,370), (0,255,0), 1) # GREEN Offset ABOVE
      cv2.line(image_np, (1,430), (1000,430), (0,255,0), 1) # GREEN Offset BELOW
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
    # Visualization of the results of a detection.
      font = cv2.FONT_HERSHEY_COMPLEX_SMALL
      im_width, im_height, _ = image_np.shape
      # print(im_width, im_height)

      for box_offsets, clas in zip(boxes[scores>=0.5], classes[scores>=0.5]):
        ymin, xmin, ymax, xmax = tuple(box_offsets)
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
        
        y_mid = int((top +bottom)//2)
#         print("y_mid>>", y_mid)
        if y_mid >= 300 and y_mid < 350 :
            if clas == 1:
                Good_bricks += 1
            else:
                Bad_Bricks += 1

        # print(f"Number of Good Bricks ::  {Good_bricks}  \nNumber of Bad BRicks :: {Bad_Bricks}")
#         cv2.rectangle(image_np, pt1 = (left, bottom),  pt2=(right, top), color = (255,255,255))
        cv2.putText(image_np,
                    f"Good Bricks ::  {Good_bricks} ||  Bad BRicks :: {Bad_Bricks}", 
                   (10, 50), font, 0.7 , (155,255,175))
        
#         print("")
            
            
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      box_df = pd.DataFrame({"xmin": [np.squeeze(boxes)[0]]})
      box_df.to_csv('box_test.csv')
      print(f"Good:{Good_bricks} \nBad:{Bad_Bricks} \nTotalBricks:{Good_bricks+Bad_Bricks}")
      #\nAccuracy:{scores[0]}")
      r=requests.post('http://192.168.1.192:8080/submitDefect',data=Bad_Bricks)
      print(r.status)
      # print(r.status)
      g_bricks.append(Good_bricks)
      b_bricks.append(Bad_Bricks)
      t_bricks.append(Good_bricks+Bad_Bricks)
      # print('******************************************************************')
      # print(b_bricks)

      dict = {
          "Good Brick": g_bricks,
          "Bad Bricks": b_bricks,
          "Total Bricks": t_bricks
      }
      data = pd.DataFrame.from_dict(dict)
      data.to_csv(r'path/counter.csv')
      # # total = Good_bricks+Bad_Bricks
      # dict = {'Good':Good_bricks,'Bad':Bad_Bricks,}#'total':total}
      # df = pd.DataFrame(dict)
      # df.to_csv('E:/brick_tfod1 - ssd/models-1.13.0/research/counter.csv')

      # cv2.imshow('object detection', cv2.resize(image_np, (800,600)))

      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()

