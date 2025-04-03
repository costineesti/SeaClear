#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
import math

# Global variables for sequence and last valid detection.
seq = 1
last_point = Odometry()
bridge = CvBridge()

class Task:
    def __init__(self):
        self.x_world = None
        self.y_world = None
        self.z_world = None

    def image_callback(self, msg):
        global seq, last_point

        # Convert the ROS image to an OpenCV BGR image.
        try:
            image = bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Camera calibration parameters (as in your original code)
        camera_matrix = np.array([[2481.80467, 0.0, 1891.149796,],
                                [0.0, 2484.001002, 1079.160663],
                                [0.0, 0.0, 1.0]
                                ])
        dist = np.array([-0.274309, 0.075813, -0.000209, -0.000607, 0.0])

        w, h = 3840, 2160
        newcammtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (w, h), 1, (w, h))
        # undistort = cv2.undistort(image, camera_matrix, dist, None, newcammtx)
        #dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        # For now we use the original image.
        frame = image
        # frame = undistort

        marker_length = 0.3  # 30 cm
        ### ARUCO MARKER DETECTION AND POSE ESTIMATION ###
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.cornerRefinementWinSize = 11
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR

        # Detect markers in the image.
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

        if ids is not None and len(ids) > 0:
            # Draw the detected markers on the image.
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # For simplicity, we work with the first detected marker.
            marker_corners = corners[0].reshape((4, 2))
            # Compute the center of the marker by averaging its four corners.
            cX = int(np.mean(marker_corners[:, 0]))
            cY = int(np.mean(marker_corners[:, 1]))
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(frame, "Marker Center", (cX - 15, cY - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist)
            # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, newcammtx, dist)

            # Use the first marker's pose.
            rvec = rvecs[0]
            tvec = tvecs[0][0].flatten()
            rmat, _ = cv2.Rodrigues(rvec)
            rmat_inv = rmat.T
            tvec_inv = -np.dot(rmat_inv, tvec.T)
            rvec_inv, _ = cv2.Rodrigues(rmat_inv)

            # tvec provides the marker position in the camera coordinate system (in meters).
            x_cam, y_cam, z_cam = tvec_inv[0], tvec_inv[1], tvec_inv[2]

            new_z = -y_cam
            new_y = x_cam
            new_x = z_cam 

            # Print the marker's world coordinates.
            print("Marker Position (world frame): X = {:.3f}, Y = {:.3f}, Z = {:.3f}".format(new_x, new_y, new_z))

            # Prepare an Odometry message with the estimated marker position.
            odom = Odometry()
            cov = [0.0] * 36
            cov[0] = 6
            cov[7] = 6
            odom.pose.covariance = cov
            odom.header.frame_id = "world"
            odom.header.stamp = msg.header.stamp
            odom.header.seq = seq
            seq += 1
            odom.child_frame_id = "aruco_marker"
            odom.pose.pose.position.x = new_x
            odom.pose.pose.position.y = new_y
            odom.pose.pose.position.z = new_z

            # (Optional) Check for outliers by comparing to the last valid detection.
            x_last = last_point.pose.pose.position.x
            y_last = last_point.pose.pose.position.y
            x_current = odom.pose.pose.position.x
            y_current = odom.pose.pose.position.y
            max_distance_jump = 1.5  # meters
            if seq == 2:  # This will be true for the very first detection.
                last_point = odom
            if math.sqrt((x_current - x_last)**2 + (y_current - y_last)**2) > max_distance_jump:
                odom.pose.pose.position.x = last_point.pose.pose.position.x
                odom.pose.pose.position.y = last_point.pose.pose.position.y
            else:
                last_point = odom

            # Publish the odometry message on your designated topic.
            real.publish(odom)
        else:
            # rospy.loginfo("can't see the aruco marker!")
            pass

        # Publish the image (with drawings) on the 'colordetection' topic.
        try:
            contur_bridge = bridge.cv2_to_imgmsg(frame, "bgr8")
            pub.publish(contur_bridge)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))


# Publishers
pub = rospy.Publisher('colordetection', Image, queue_size=10)
world = rospy.Publisher('/BlueRov2/plane', Odometry, queue_size=10)
real = rospy.Publisher('/BlueRov2/real_coord', Odometry, queue_size=10)

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    task = Task()

    # Subscribe to the image and depth topics.
    rospy.Subscriber('/usb_cam/image_raw', Image, task.image_callback)

    rospy.spin()