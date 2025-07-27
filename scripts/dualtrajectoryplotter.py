#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from geometry_msgs.msg import PointStamped, TransformStamped, PoseStamped
import tf2_ros, tf.transformations, tf2_geometry_msgs # do not remove this import! will throw an error if removed
from std_msgs.msg import Float64, Float32
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
import math  
from aruco import ArucoTask

seq_gopro = 1
seq_usbcamera = 1
last_point_gopro = Odometry()
last_point_usbcamera = Odometry()
USE_3D_PLOT = False  # Change this to False for 2D plotting

class DualTrajectoryComputer:

    def __init__(self):

                                                # === CONSTANTS GOPRO ===
        self.timestamp_gopro = None
        self.camera_matrix_gopro = np.array(((927.091270, 0.000000, 957.570804), 
                                       (0.000000, 919.995427, 533.540912), 
                                       (0.000000, 0.000000, 1.000000)))
        self.dist_gopro = np.array((0.05, 0.07, -0.11, 0.05, 0.000000))
        self.camera_center_gopro = None
        self.camera_orientation_gopro = None
        self.GET_PARAMS_GOPRO = True

                                                # === CONSTANTS USBCAMERA ===
        self.timestamp_usbcamera = None
        self.camera_matrix_usbcamera = None
        self.dist_usbcamera = np.array([-0.274309, 0.075813, -0.000209, -0.000607, 0.0])
        self.camera_center_usbcamera = None
        self.camera_orientation_usbcamera = None
        self.GET_PARAMS_USBCAMERA = True

                                                # Initialize ROS components

        rospy.init_node('dual_trajectory_computer', anonymous=True)
        # tf2 components
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
    
        # Publishers for RViz
        self.gopro_path_pub = rospy.Publisher('/gopro_trajectory_path', Path, queue_size=10)
        self.usbcamera_path_pub = rospy.Publisher('/usbcamera_trajectory_path', Path, queue_size=10)

        # Initialize paths for RViz visualization
        self.gopro_path = Path()
        self.gopro_path.header.frame_id = "aruco_marker"
        self.usbcamera_path = Path()
        self.usbcamera_path.header.frame_id = "aruco_marker"

        # Publishers for plotjugger or rqt_plot
        self.gopro_x_pub = rospy.Publisher("/gopro_x", Float32, queue_size=1)
        self.gopro_y_pub = rospy.Publisher("/gopro_y", Float32, queue_size=1)
        self.usb_x_pub   = rospy.Publisher("/usb_x",   Float32, queue_size=1)
        self.usb_y_pub   = rospy.Publisher("/usb_y",   Float32, queue_size=1)

        global USE_3D_PLOT

    def add_pose_to_path(self, path_msg, path_pub, x, y, z, timestamp):
        ps = PoseStamped()
        ps.header.frame_id = path_msg.header.frame_id
        ps.header.stamp = timestamp
        ps.pose.position.x = x
        ps.pose.position.y = y
        if USE_3D_PLOT:
            ps.pose.position.z = z
        else:
            ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0

        path_msg.poses.append(ps)
        path_msg.header.stamp = timestamp
        path_pub.publish(path_msg)

    def publish_camera_to_aruco_transform(self, rvec, tvec, camera_frame, timestamp):
        """
        Publish transform: camera_frame -> aruco_marker
        This represents where the marker is relative to the camera.
        """
        if timestamp is None:
            timestamp = rospy.Time.now()
            
        transform = TransformStamped()
        transform.header.stamp = timestamp
        transform.header.frame_id = "aruco_marker"
        transform.child_frame_id = camera_frame
        
        R, _ = cv2.Rodrigues(rvec)
        R_inv = R.T
        t_inv = -R_inv @ tvec.flatten()

        transform.transform.translation.x = float(t_inv[0])
        transform.transform.translation.y = float(t_inv[1])
        transform.transform.translation.z = float(t_inv[2])

        quat = tf.transformations.quaternion_from_matrix(
            np.vstack([np.hstack([R_inv, [[0], [0], [0]]]), [0, 0, 0, 1]])
)
        
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        
        self.static_tf_broadcaster.sendTransform(transform)
        rospy.loginfo(f"Set up transform chain: {camera_frame} -> aruco_marker")

    def transform_point_to_world(self, point_array, source_frame, timestamp):
        """
        Transform a point from the source frame to the aruco_marker frame using tf2.
        """
        try:
            point_stamped = PointStamped()
            point_stamped.header.frame_id = source_frame
            point_stamped.header.stamp = timestamp
            point_stamped.point.x = point_array[0]
            point_stamped.point.y = point_array[1]
            point_stamped.point.z = point_array[2]

            # Check if transform is available
            # if not self.tf_buffer.can_transform(
            #     "aruco_marker", source_frame, rospy.Time(0), rospy.Duration(1.0)
            # ):
            #     rospy.logwarn(f"Transform from {source_frame} to aruco_marker not available")
            #     return point_array

            transformed_point = self.tf_buffer.transform(
                point_stamped, "aruco_marker"
            )

            return np.array([
                transformed_point.point.x,
                transformed_point.point.y,
                transformed_point.point.z
            ])
        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF2 Error: {e}")
            return point_array
        

    def usbcamera_timestamp_callback(self, timestamp):
        self.timestamp_usbcamera = rospy.Time.from_sec(timestamp.data)

    def gopro_timestamp_callback(self, timestamp):
        self.timestamp_gopro = rospy.Time.from_sec(timestamp.data)
    
    def gopro_image_callback(self, image_msg):
        """
        Create aruco instance in order to get the fixed world coordinates of the gopro relative to the aruco marker
        """
        if not self.GET_PARAMS_GOPRO:
            return
        
        if image_msg is not None and self.GET_PARAMS_GOPRO is True:
            aruco_instance_gopro = ArucoTask(self.dist_gopro, self.camera_matrix_gopro)
            coords, frame, marker_id, rvec, tvec = aruco_instance_gopro.fetch_camera_woorldCoordinates(image_msg, camera_type='GoPro')
            # Store values across code
            if coords is not None and marker_id is not None and rvec is not None and tvec is not None:

                timestamp = self.timestamp_gopro
                # timestamp = rospy.Time.now()
                self.publish_camera_to_aruco_transform(rvec, tvec, "gopro_camera", timestamp)
                R_cam_to_world = cv2.Rodrigues(rvec)[0]
                t_cam_to_world = tvec.flatten()

                # Camera center in world coordinates
                self.camera_center_gopro = -R_cam_to_world.T @ t_cam_to_world
                self.camera_orientation_gopro = R_cam_to_world.T

                rospy.loginfo(f"GoPro ArUco calibration complete: Center={self.camera_center_gopro}, Orientation shape={self.camera_orientation_gopro.shape}")
                self.GET_PARAMS_GOPRO = False  # Only fetch once.

    def usbcamera_image_callback(self, image_msg):
        """
        1. Compute the camera matrix based on the width and height of the video.
        2. Create aruco instance in order to get the fixed world coordinates of the usbcamera relative to the aruco marker
        """
        if not self.GET_PARAMS_USBCAMERA:
            return
        
        if image_msg is not None and self.GET_PARAMS_USBCAMERA is True:
            w, h = image_msg.width, image_msg.height
            scale_x = w / 3840
            scale_y = h / 2160
            # Scale intrinsic parameters
            fx_scaled = 2481.80467 * scale_x
            fy_scaled = 2484.001002 * scale_y
            cx_scaled = 1891.149796 * scale_x
            cy_scaled = 1079.160663 * scale_y

            self.camera_matrix_usbcamera = np.array([
                [fx_scaled, 0, cx_scaled],
                [0, fy_scaled, cy_scaled],
                [0, 0, 1]
            ])
            aruco_instance_usbcamera = ArucoTask(self.dist_usbcamera, self.camera_matrix_usbcamera)
            coords, frame, marker_id, rvec, tvec = aruco_instance_usbcamera.fetch_camera_woorldCoordinates(image_msg, camera_type='usb_camera')
            # Store information across the class
            if coords is not None and marker_id is not None and rvec is not None and tvec is not None:

                timestamp = self.timestamp_usbcamera
                # timestamp = rospy.Time.now()
                self.publish_camera_to_aruco_transform(rvec, tvec, "usb_camera", timestamp)
                R_cam_to_world = cv2.Rodrigues(rvec)[0]
                t_cam_to_world = tvec.flatten()

                # Camera center in world coordinates
                self.camera_center_usbcamera = -R_cam_to_world.T @ t_cam_to_world
                self.camera_orientation_usbcamera = R_cam_to_world.T

                rospy.loginfo(f"USB Camera ArUco calibration complete: Center={self.camera_center_usbcamera}, Orientation shape={self.camera_orientation_usbcamera.shape}")
                self.GET_PARAMS_USBCAMERA = False # Only fetch once.

    def pixel_to_3d_point(self, pixel_x, pixel_y, depth, camera_matrix, camera_center, camera_orientation):
        """
        Convert pixel coordinates to 3D point in camera frame.
        See ray and backward projection from the following sources:
        Source: https://costinchitic.co/notes/camera-backward-projection
        https://costinchitic.co/notes/Multiple-View-Geometry-in-Computer-Vision
        """
        # Convert pixel to normalized camera coordinates
        pixel_coord = np.array([pixel_x, pixel_y, 1.0])
        inv_cam_matrix = np.linalg.inv(camera_matrix)
        ray_cam = inv_cam_matrix @ pixel_coord
        
        # The normalized coordinates give the direction, multiply by depth to get 3D point
        plane_z = -depth # Depth is negative because the camera is underwater
        alfa = (plane_z - camera_center[2]) / (camera_orientation @ ray_cam)[2]
        
        # Return intersection point in camera frame
        camera_point_3d = alfa * ray_cam
        return camera_point_3d

    def gopro_coordinates_callback(self, msg):

        timestamp = self.timestamp_gopro
        # timestamp = rospy.Time.now()
        
        # Wait for ArUco calibration to complete
        if self.camera_center_gopro is None or self.camera_orientation_gopro is None:
            return
        
        if timestamp:
            # Get depth information from the IMU attached on the ROV (depth is inside the pool. At the surface, depth = 0)
            depth = msg.point.z if msg.point.z else 0
            ROV_3D_point = self.pixel_to_3d_point(msg.point.x,
                                                  msg.point.y, 
                                                  depth, 
                                                  self.camera_matrix_gopro,
                                                  self.camera_center_gopro,
                                                  self.camera_orientation_gopro
                                                  )
            world_coord = self.transform_point_to_world(ROV_3D_point, "gopro_camera", timestamp)
            # Publish coordinates for plotjugger
            self.gopro_x_pub.publish(world_coord[0])
            self.gopro_y_pub.publish(world_coord[1])
            # Create Odometry message
            global seq_gopro, last_point_gopro
            cov = [0.0] * 36
            cov[0] = 6
            cov[7] = 6
            odom = Odometry()
            odom.pose.covariance = cov
            odom.header.frame_id = "aruco_marker"
            odom.header.stamp = timestamp
            odom.header.seq = seq_gopro
            seq_gopro += 1
            odom.child_frame_id = "gopro_camera"
            
            odom.pose.pose.position.x = world_coord[0]
            odom.pose.pose.position.y = world_coord[1]
            odom.pose.pose.position.z = world_coord[2]
            
            # Outlier detection
            x_last = last_point_gopro.pose.pose.position.x
            y_last = last_point_gopro.pose.pose.position.y
            x_current = odom.pose.pose.position.x
            y_current = odom.pose.pose.position.y
            max_distance_jump = 1.5  # meters
            
            if seq_gopro == 1:
                last_point_gopro = odom
            elif math.sqrt((x_current - x_last)**2 + (y_current - y_last)**2) > max_distance_jump:
                # Outlier detected, use last valid position
                odom.pose.pose.position.x = last_point_gopro.pose.pose.position.x
                odom.pose.pose.position.y = last_point_gopro.pose.pose.position.y
                odom.pose.pose.position.z = last_point_gopro.pose.pose.position.z
            else:
                # Valid detection, update last point
                last_point_gopro = odom
            
            ground_truth.publish(odom)
            
            # Plot to RViz
            self.add_pose_to_path(self.gopro_path, self.gopro_path_pub, *world_coord, timestamp)

    def usbcamera_coordinates_callback(self, msg):

        timestamp = self.timestamp_usbcamera
        # timestamp = rospy.Time.now()

        if self.camera_center_usbcamera is None:
            return # Wait for ArUco calibration to complete
        
        if timestamp:
            # Get depth information from the IMU attached on the ROV (depth is inside the pool. At the surface, depth = 0)
            depth = msg.point.z
            ROV_3D_point = self.pixel_to_3d_point(msg.point.x,
                                                  msg.point.y, 
                                                  depth, 
                                                  self.camera_matrix_usbcamera,
                                                  self.camera_center_usbcamera,
                                                  self.camera_orientation_usbcamera
                                                  )
            world_coord = self.transform_point_to_world(ROV_3D_point, "usb_camera", timestamp)
            
            # Publish coordinates for plotjugger
            self.usb_x_pub.publish(world_coord[0])
            self.usb_y_pub.publish(world_coord[1])

            # Create Odometry message
            global seq_usbcamera, last_point_usbcamera
            odom = Odometry()
            cov = [0.0] * 36
            cov[0] = 6
            cov[7] = 6
            odom.pose.covariance = cov
            odom.header.frame_id = "aruco_marker"
            odom.header.stamp = timestamp
            odom.header.seq = seq_usbcamera
            seq_usbcamera += 1
            odom.child_frame_id = "usb_camera"
            
            # Store world coordinates
            odom.pose.pose.position.x = world_coord[0]
            odom.pose.pose.position.y = world_coord[1]
            odom.pose.pose.position.z = world_coord[2]
            
            # Outlier detection
            x_last = last_point_usbcamera.pose.pose.position.x
            y_last = last_point_usbcamera.pose.pose.position.y
            x_current = odom.pose.pose.position.x
            y_current = odom.pose.pose.position.y
            max_distance_jump = 1.5  # meters
            
            if seq_usbcamera == 1:  # First detection
                last_point_usbcamera = odom
            elif math.sqrt((x_current - x_last)**2 + (y_current - y_last)**2) > max_distance_jump:
                # Outlier detected, use last valid position
                odom.pose.pose.position.x = last_point_usbcamera.pose.pose.position.x
                odom.pose.pose.position.y = last_point_usbcamera.pose.pose.position.y
                odom.pose.pose.position.z = last_point_usbcamera.pose.pose.position.z
            else:
                # Valid detection, update last point
                last_point_usbcamera = odom
            
            estimation.publish(odom)
            
            # Add position to plotter
            self.add_pose_to_path(self.usbcamera_path, self.usbcamera_path_pub, *world_coord, timestamp)

estimation = rospy.Publisher('/BlueRov2/estimation', Odometry, queue_size=10)
ground_truth = rospy.Publisher('/BlueRov2/groundtruth', Odometry, queue_size=10)

def main():
    
    try:
        computer = DualTrajectoryComputer()
        # Subscribers for coordinate data
        rospy.Subscriber('/gopro/wall_time', Float64, computer.gopro_timestamp_callback) # gopro timestamp callback
        rospy.Subscriber('/camera_wall_time', Float64, computer.usbcamera_timestamp_callback) # usbcamera timestamp callback
        rospy.Subscriber('/gopro/image_raw', Image, computer.gopro_image_callback) # gopro image callback
        rospy.Subscriber('/camera/image_compressed', Image, computer.usbcamera_image_callback) # usbcamera image callback
        rospy.Subscriber('/robotdetection/coordinates_gopro', PointStamped, computer.gopro_coordinates_callback) # gopro coordinates callback
        rospy.Subscriber('/robotdetection/coordinates_usbcamera', PointStamped, computer.usbcamera_coordinates_callback) # usbcamera coordinates callback

        rospy.loginfo("Initializing DualTrajectoryComputer...")
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")

if __name__ == '__main__':
    main()