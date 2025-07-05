#!/usr/bin/env python3

import cv2, os
import matplotlib.pyplot as plt
import numpy as np
import rospy, rosbag
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from PIL import Image as PILImage
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import math

seq = 1
last_point = Odometry()
bridge = CvBridge()

class Task:
    
    def __init__(self):
        self.x_world = None
        self.y_world = None
        self.z_world = None

    def depth_callback(self, depth):
        self.z_world = -depth.pose.pose.position.z

    def image_callback(self, image_msg):
        """Process image from rosbag"""
        global seq, last_point

        w, h = image_msg.header.width, image_msg.header.height
        scale_x = w / 3840
        scale_y = h / 2160

        # Scale intrinsic parameters
        fx_scaled = 2481.80467 * scale_x
        fy_scaled = 2484.001002 * scale_y
        cx_scaled = 1891.149796 * scale_x
        cy_scaled = 1079.160663 * scale_y

        camera_matrix = np.array([
            [fx_scaled, 0, cx_scaled],
            [0, fy_scaled, cy_scaled],
            [0, 0, 1]
        ])
        dist = np.array([-0.274309, 0.075813, -0.000209, -0.000607, 0.0])
        cx = int(w/2)
        cy = int(h/2)
        
        # Get optimal camera matrix
        # newcammtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (w,h), 1, (w,h))
        image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
        frame = image
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.circle(frame, (self.cx, self.cy), 5, (255,0,0), 3)

        # SEGMENTATION

        # Yellow interval
        lower_yellow = np.array([20, 98, 115])
        upper_yellow = np.array([50, 255, 255])
        # Thresholding
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_yellow = cv2.GaussianBlur(mask_yellow, (5, 5), 0)
        yellow_patch = cv2.bitwise_and(frame, frame, mask=mask_yellow)
        # Create contours
        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No yellow contours found")
            return
        # Get max contour
        c = max(contours, key=cv2.contourArea)
        # Get centroid
        M = cv2.moments(c)
        if M["m00"] == 0:
            print("Invalid contour moments")
            return
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # print(f'center pixel coordinates: {cX}, {cY}')
        # print(f'center pixel hsv: {hsv[cY, cX]}')
        
        # Draw center
        cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
        cv2.putText(frame, "center", (cX-15, cY-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.circle(frame, (957, 533), 8, (255,102,255), -1) # Camera center

        # Draw contours
        contur = cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
        frame_bridge = bridge.cv2_to_imgmsg(frame, 'bgr8')
        pub.publish(frame_bridge)

        # POSE ESTIMATION
        inv_cam_matrix = np.linalg.inv(camera_matrix)
        centru = np.array([cX, cY, 1]).reshape(-1, 1) # Center of detected object (converted to homogeneous coordinates)
        C = np.array([0, 0, 0]).reshape(-1, 1) # Center point of camera
        # Line on which the projected point lies in camera reference frame
        M_int = np.dot(inv_cam_matrix, centru)
        d = M_int - C
        u, v = cX, cY # Image coordinates
        image_coord = np.array([u, v, 1]).reshape(-1, 1)
        camera_coord = np.dot(inv_cam_matrix, image_coord) # Derived camera coordinates
        # Vehicle coordinates
        zet = 2400  # mm 2380 height + 20 (estimated) camera
        homo_points = image_coord * zet
        world_coord = np.linalg.inv(camera_matrix) @ homo_points
        world_coord = np.squeeze(np.asarray(world_coord / 1000))
        # Create odometry message
        cov = [0.0] * 36
        cov[0] = 6
        cov[7] = 6
        odom = Odometry()
        odom.pose.covariance = cov
        odom.header.frame_id = "world"
        odom.header.stamp = image_msg.header.stamp
        odom.header.seq = seq
        seq += 1
        odom.child_frame_id = "yellow_patch"
        
        self.x_world = world_coord[0]
        self.y_world = world_coord[1]
        odom.pose.pose.position.x = world_coord[0]
        odom.pose.pose.position.y = world_coord[1]
        
        # Change coordinates so that ox points forward and oy points right
        old_x = odom.pose.pose.position.x
        old_y = odom.pose.pose.position.y
        new_x = -old_y
        new_y = old_x
        odom.pose.pose.position.x = new_x
        odom.pose.pose.position.y = new_y

        # Check for outliers when losing color detection
        x_last = last_point.pose.pose.position.x
        y_last = last_point.pose.pose.position.y
        x_current = odom.pose.pose.position.x
        y_current = odom.pose.pose.position.y
        max_distance_jump = 1.5
        
        if seq == 2:  # First real measurement
            last_point = odom
        elif math.sqrt((x_current - x_last)**2 + (y_current - y_last)**2) > max_distance_jump:
            odom.pose.pose.position.x = last_point.pose.pose.position.x
            odom.pose.pose.position.y = last_point.pose.pose.position.y
        else:
            last_point = odom

        real.publish(odom)

# Publishers
pub = rospy.Publisher('colordetection', Image, queue_size=10)
world = rospy.Publisher('/BlueRov2/plane', Odometry, queue_size=10)
real = rospy.Publisher('/BlueRov2/real_coord', Odometry, queue_size=10)

if __name__ == '__main__':
    rospy.init_node('rosbag_processor', anonymous=True)
    task = Task()

    rospy.Subscriber('/camera/image_compressed', Image, task.image_callback)
    # rospy.Subscriber('/camera/camera_wall_time', Float64, task.depth_callback) Not yet implemented
    rospy.Subscriber('/BlueRov2/odom/depth', Odometry, task.depth_callback)
    rospy.spin()