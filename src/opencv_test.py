#!/usr/bin/python3

import cv2
import PIL
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from cv_bridge.boost.cv_bridge_boost import getCvType
import sys

sys.path.insert(0, '/home/colson/workspace/keras-retinanet/')




bridge = CvBridge()

def main():
    print('image_lisnter')
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/usb_cam/image_raw"
    # Set up your subscriber and define its callback
    image_pub = rospy.Publisher("/output/image_raw", Image, queue_size=1)
    
    cap = cv2.VideoCapture(2)
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    #if cap.isOpen():
    #    print('width: {}, height : {}'.format(cap.get(3), cap.get(4)))

    # Spin until ctrl + c
    #rospy.spin()
    
    while(1):
        ret, fram = cap.read()
        if ret:
            gray = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
            cv2.imshow('video', gray)
            cv2.waitKey(0)
        print("running...")

        msg = CvBridge().cv2_to_imgmsg(gray)
        image_pub.publish(msg)


        is_running = 0

if __name__ == '__main__':
    main()
