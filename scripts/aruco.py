#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
import math
from geometry_msgs.msg import Vector3

# Global variables for sequence and last valid detection.
seq = 1
last_point = Odometry()
bridge = CvBridge()

class ArucoTask:
    def __init__(self, dist, camera_matrix):
        self.dist= dist
        self.camera_matrix = camera_matrix
        
    def detect_aruco_marker(self, frame, camera_matrix, dist, marker_length=0.3):
        """
        Detect ArUco markers and estimate their 3D position
        
        Args:
            frame: OpenCV image
            camera_matrix: Camera calibration matrix
            dist: Distortion coefficients
            marker_length: Size of marker in meters in real life (default 0.3m)
            
        Returns:
            (x, y, z): Coordinates in world frame or None if no marker detected
            frame: Image with detection visualization
            marker_id: ID of the detected marker or None
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.cornerRefinementWinSize = 11
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR

        try:
            # For OpenCV 4.5.0+ (new API)
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            corners, ids, rejected = detector.detectMarkers(frame)
        except AttributeError:
            try:
                # For older OpenCV versions (legacy API)
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    frame, aruco_dict, parameters=aruco_params)
            except Exception as e:
                rospy.logerr(f"ArUco detection failed: {e}")
                corners, ids, rejected = [], None, []

        if ids is None or len(ids) == 0:
            return None, frame, None

        # Draw the detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rospy.loginfo(f"Detected ArUco markers with IDs: {ids.flatten()}")
        
        # Add marker ID text to the image
        for i, marker_id in enumerate(ids.flatten()):
            marker_corners = corners[i].reshape((4, 2))
            cX = int(np.mean(marker_corners[:, 0]))
            cY = int(np.mean(marker_corners[:, 1]))
            cv2.putText(frame, f"ID: {marker_id}", (cX, cY + 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # For simplicity, work with the first detected marker
        marker_corners = corners[0].reshape((4, 2))
        cX = int(np.mean(marker_corners[:, 0]))
        cY = int(np.mean(marker_corners[:, 1]))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
        cv2.putText(frame, "Marker Center", (cX - 15, cY - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Estimate pose
        try:
            # For OpenCV versions with estimatePoseSingleMarkers
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist)
        except AttributeError:
            # For newer OpenCV versions where estimatePoseSingleMarkers is not available
            rvecs = []
            tvecs = []
            objPoints = np.array([[-marker_length/2, marker_length/2, 0],
                                [marker_length/2, marker_length/2, 0],
                                [marker_length/2, -marker_length/2, 0],
                                [-marker_length/2, -marker_length/2, 0]])
            
            for corner in corners:
                retval, rvec, tvec = cv2.solvePnP(objPoints, corner, camera_matrix, dist)
                rvecs.append(rvec)
                tvecs.append(np.array([tvec]).reshape(1,3))

        # Use the first marker's pose
        rvec = rvecs[0]
        tvec = tvecs[0][0].flatten()
        # rmat, _ = cv2.Rodrigues(rvec)
        # rmat_inv = rmat.T
        # tvec_inv = -np.dot(rmat_inv, tvec.T)
        
        # Convert to world coordinates
        # x_cam, y_cam, z_cam = tvec_inv[0], tvec_inv[1], tvec_inv[2]
        x_cam, y_cam, z_cam = tvec[0], tvec[1], tvec[2] # Let the main script handle the camera transformation through TF
        # new_z = -y_cam
        # new_y = x_cam
        # new_x = z_cam
        
        # Log position
        # rospy.loginfo(f"Marker ID {ids[0][0]} Position: X={new_x:.3f}, Y={new_y:.3f}, Z={new_z:.3f}")
        
        # return (new_x, new_y, new_z), frame, ids[0][0], rvec, tvec
        return (x_cam, y_cam, z_cam), frame, ids[0][0], rvec, tvec

    def fetch_camera_woorldCoordinates(self, msg, camera_type='GoPro'):
        global seq, last_point

        # Convert the ROS image to an OpenCV BGR image
        try:
            image = bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        
        # Detect ArUco marker
        frame = image
        coords, frame, marker_id, rvec, tvec = self.detect_aruco_marker(frame, self.camera_matrix, self.dist)

        if coords:
            new_x, new_y, new_z = coords
            rospy.loginfo(f"{camera_type} Marker ID {marker_id} Position: X={new_x:.3f}, Y={new_y:.3f}, Z={new_z:.3f}")
        
        return coords, frame, marker_id, rvec, tvec


if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    task = ArucoTask()
    rospy.Subscriber('/camera/image_compressed', Image, task.image_callback_usbcamera)
    rospy.Subscriber('/gopro/image_raw', Image, task.image_callback_gopro)
    rospy.spin()