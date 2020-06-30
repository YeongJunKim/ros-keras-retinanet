#!/usr/bin/python3

import cv2
import PIL
import rospy
from sensor_msgs.msg import Image
print("start import CvBridge")
from cv_bridge import CvBridge, CvBridgeError
print("start import CvType")
from cv_bridge.boost.cv_bridge_boost import getCvType

print("import getCvType")
import keras

import sys
sys.path.insert(0, '/home/colson/workspace/keras-retinanet/')
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


# use this to change which GPU to use
gpu = 0

# set the modified tf session as backend in keras
setup_gpu(gpu)



model_path = os.path.join('..', 'model', 'resnet50_coco_best_v2.1.0.h5')
#model = models.load_model('/home/colson/catkin_ws/src/ros-keras-retinanet/model/resnet50_coco_best_v2.1.0.h5', backbone_name='resnet50')
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

is_running = 0
boxes = 0
scores = 0
labels = 0

cap = cv2.VideoCapture(4)
# Instantiate CvBridge
bridge = CvBridge()
image = 0

def image_callback(msg):
    # print("Received an image!")
    global is_running
    global image

    if is_running == 1:
        return;
    else:
        is_running = 1

    image = bridge.imgmsg_to_cv2(msg, "bgr8")

    # try:
    #     # Convert your ROS Image message to OpenCV2
    #     image = bridge.imgmsg_to_cv2(msg, "bgr8")
    #     #ret, image = cap.read()
    #     #cv2.imwrite('out.jpg', image)
    #     image = np.ascontiguousarray(image)
        
    #     draw = image.copy()
    #     draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    #     image = preprocess_image(image)
    #     #image, scale = resize_image(image)
    #     scale = 1
        
    #     # process image
    #     start = time.time()
    #     boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    #     print("processing time: ", time.time() - start)

    #     # correct for image scale
    #     boxes /= scale

    #     # visualize detections
    #     for box, score, label in zip(boxes[0], scores[0], labels[0]):
    #         # scores are sorted so we can break
    #         if score < 0.5:
    #             break
                
    #         color = label_color(label)
            
    #         b = box.astype(int)
    #         draw_box(draw, b, color=color)
            
    #         caption = "{} {:.3f}".format(labels_to_names[label], score)
    #         draw_caption(draw, b, caption)
            
    #     plt.figure(figsize=(15, 15))
    #     plt.axis('off')
    #     plt.imshow(draw)
    #     plt.show()

    # except CvBridgeError as e:
    #     print(e)
    # except ValueError as e:
    #     print(e)
    # except:
    #     print("Unexpected error:", sys.exc_info()[0])
    # else:
    #     # Save your OpenCV2 image as a jpeg 
    #     #cv2.imwrite('camera_image.jpeg', cv2_img)
    #     print('good')
    # is_running =0


def main():
    print('image_lisnter')
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/usb_cam/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback, queue_size=1)
    image_pub = rospy.Publisher("/output/image_raw", Image)
    
    #model = models.load_model('/home/colson/catkin_ws/src/ros-keras-retinanet/model/resnet50_coco_best_v2.1.0.h5', backbone_name='resnet50')
    model = models.load_model('/home/colson/catkin_ws/src/ros-keras-retinanet/model/resnet_50_pascal_12_inference.h5', backbone_name='resnet50')
    #model = models.load_model('/home/colson/catkin_ws/src/ros-keras-retinanet/model/mobilenet128_1.0_pascal_06_inference.h5', backbone_name='mobilenet128')
    #model = models.load_model('/home/colson/catkin_ws/src/ros-keras-retinanet/model/vgg19_pascal_14_inference.h5', backbone_name='vgg19')
    labels_to_names = {0: 'drone', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

    # Spin until ctrl + c
    #rospy.spin()
    while(1):
        global is_running
        global image

        if is_running == 1:        
    
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            image = preprocess_image(image)
            #image, scale = resize_image(image)
            scale = 1
            
            # process image
            start = time.time()
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            print("processing time: ", time.time() - start)

            # correct for image scale
            boxes /= scale

            # visualize detections
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.15:
                    break
                    
                color = label_color(label)
                
                b = box.astype(int)
                draw_box(draw, b, color=color)

                caption = "{} {:.3f}".format(labels_to_names[label], score)
                draw_caption(draw, b, caption)
            
            # print(draw)
            # print(draw.size)
            # print(draw.shape)

            draw = draw.astype(np.uint8)
            
            msg = CvBridge().cv2_to_imgmsg(draw, encoding="rgb8")

            
            image_pub.publish(msg)

            # draw = cv2.cvtColor(draw, f
            # plt.figure(figsize=(15, 15))
            # plt.axis('off')
            # plt.imshow(draw)
            # plt.show()

            is_running = 0

if __name__ == '__main__':
    main()
