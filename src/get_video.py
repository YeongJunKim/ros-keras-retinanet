#!/usr/bin/python3

import cv2
import PIL
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

def main():
    print('image_resizer')
    rospy.init_node('image_resizer')
    # Define your image topic
    image_topic = "/usb_cam/image_raw"
    # Set up your subscriber and define its callback
    image_pub = rospy.Publisher("/usb_cam/image_raw", Image, queue_size=1)
    
    cap = cv2.VideoCapture(4)
    # cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    
    while not rospy.is_shutdown():
        image = 0
        ret, fram = cap.read()
        if ret:
            gray = cv2.cvtColor(fram, cv2.IMREAD_COLOR)
            #image = cv2.resize(gray, dsize=(640,480))
            image = cv2.resize(gray, dsize=(320,240))
            # cv2.imshow('video', gray)
            # cv2.waitKey(0)
            msg = CvBridge().cv2_to_imgmsg(image, encoding="bgr8")
            image_pub.publish(msg)
        # print("running...")



        is_running = 0

if __name__ == '__main__':
    main()
