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
        rmat, _ = cv2.Rodrigues(rvec)
        rmat_inv = rmat.T
        tvec_inv = -np.dot(rmat_inv, tvec.T)
        
        # Convert to world coordinates
        x_cam, y_cam, z_cam = tvec_inv[0], tvec_inv[1], tvec_inv[2]
        new_z = -y_cam
        new_y = x_cam
        new_x = z_cam
        
        # Log position
        # rospy.loginfo(f"Marker ID {ids[0][0]} Position: X={new_x:.3f}, Y={new_y:.3f}, Z={new_z:.3f}")
        
        return (new_x, new_y, new_z), frame, ids[0][0]

    def image_callback_usbcamera(self, msg):
        global seq, last_point

        # Convert the ROS image to an OpenCV BGR image
        try:
            image = bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Camera calibration parameters
        w, h = msg.width, msg.height
        self.scale_x = w / 3840
        self.scale_y = h / 2160
        fx_scaled = 2481.80467 * self.scale_x
        fy_scaled = 2484.001002 * self.scale_y
        cx_scaled = 1891.149796 * self.scale_x
        cy_scaled = 1079.160663 * self.scale_y

        camera_matrix = np.array([
            [fx_scaled, 0, cx_scaled],
            [0, fy_scaled, cy_scaled],
            [0, 0, 1]
        ])
        dist = np.array([-0.274309, 0.075813, -0.000209, -0.000607, 0.0])
        
        # Detect ArUco marker
        frame = image
        coords, frame, marker_id = self.detect_aruco_marker(frame, camera_matrix, dist)
        
        if coords:
            new_x, new_y, new_z = coords
            rospy.loginfo(f"USBCAMERA Marker ID {marker_id} Position: X={new_x:.3f}, Y={new_y:.3f}, Z={new_z:.3f}")
            
            # Prepare an Odometry message with the estimated marker position
            odom = Odometry()
            cov = [0.0] * 36
            cov[0] = 6
            cov[7] = 6
            odom.pose.covariance = cov
            odom.header.frame_id = "world"
            odom.header.stamp = msg.header.stamp
            odom.header.seq = seq
            seq += 1
            odom.child_frame_id = f"aruco_marker_{marker_id}"
            odom.pose.pose.position.x = new_x
            odom.pose.pose.position.y = new_y
            odom.pose.pose.position.z = new_z

            # Outlier detection
            x_last = last_point.pose.pose.position.x
            y_last = last_point.pose.pose.position.y
            x_current = odom.pose.pose.position.x
            y_current = odom.pose.pose.position.y
            max_distance_jump = 1.5  # meters
            
            if seq == 2:  # First detection
                last_point = odom
            elif math.sqrt((x_current - x_last)**2 + (y_current - y_last)**2) > max_distance_jump:
                odom.pose.pose.position.x = last_point.pose.pose.position.x
                odom.pose.pose.position.y = last_point.pose.pose.position.y
            else:
                last_point = odom

            world_usbcamera.publish(odom)
        
        # Publish the processed image
        try:
            img_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
            pub.publish(img_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    
    def image_callback_gopro(self, msg):
        global seq, last_point

        # Convert the ROS image to an OpenCV BGR image
        try:
            image = bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Camera calibration parameters
        w, h = msg.width, msg.height
        camera_matrix = np.array(((927.091270, 0.000000, 957.570804), 
                                       (0.000000, 919.995427, 533.540912), 
                                       (0.000000, 0.000000, 1.000000)))
        K = np.array(((927.091270, 0.000000, 957.570804, 0.000000), 
                           (0.000000, 919.995427, 533.540912, 0.000000), 
                           (0.000000, 0.000000, 1.000000, 0.000000)))
        dist = np.array((0.05, 0.07, -0.11, 0.05, 0.000000))
        
        # Detect ArUco marker
        frame = image # copy the image to avoid modifying the original
        coords, frame, marker_id = self.detect_aruco_marker(frame, camera_matrix, dist)
        
        if coords:
            new_x, new_y, new_z = coords
            rospy.loginfo(f"GOPRO Marker ID {marker_id} Position: X={new_x:.3f}, Y={new_y:.3f}, Z={new_z:.3f}")
            
            # Prepare an Odometry message with the estimated marker position
            odom = Odometry()
            cov = [0.0] * 36
            cov[0] = 6
            cov[7] = 6
            odom.pose.covariance = cov
            odom.header.frame_id = "world"
            odom.header.stamp = msg.header.stamp
            odom.header.seq = seq
            seq += 1
            odom.child_frame_id = f"aruco_marker_{marker_id}"
            odom.pose.pose.position.x = new_x
            odom.pose.pose.position.y = new_y
            odom.pose.pose.position.z = new_z

            # Outlier detection
            x_last = last_point.pose.pose.position.x
            y_last = last_point.pose.pose.position.y
            x_current = odom.pose.pose.position.x
            y_current = odom.pose.pose.position.y
            max_distance_jump = 1.5  # meters
            
            if seq == 2:  # First detection
                last_point = odom
            elif math.sqrt((x_current - x_last)**2 + (y_current - y_last)**2) > max_distance_jump:
                odom.pose.pose.position.x = last_point.pose.pose.position.x
                odom.pose.pose.position.y = last_point.pose.pose.position.y
            else:
                last_point = odom

            world_gopro.publish(odom)
        
        # Publish the processed image
        try:
            img_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
            pub.publish(img_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")


# Publishers
pub = rospy.Publisher('arucodetection', Image, queue_size=10)
world_gopro = rospy.Publisher('/BlueRov2/aruco_gopro', Odometry, queue_size=10)
world_usbcamera = rospy.Publisher('/BlueRov2/aruco_usbcamera', Odometry, queue_size=10)

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    task = Task()
    rospy.Subscriber('/camera/image_compressed', Image, task.image_callback_usbcamera)
    rospy.Subscriber('/gopro/image_raw', Image, task.image_callback_gopro)
    rospy.spin()