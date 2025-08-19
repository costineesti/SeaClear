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
        self.camera_matrix_gopro = np.array(((1.00314412e+03, 0.000000, 9.54105008e+02),
                                             (0.000000, 9.98774705e+02, 5.42888665e+02),
                                             (0.000000, 0.00000000e+00, 1.000000)))
        self.dist_gopro = np.array([0.00429666, -0.00949222, 0.00217127, -0.00283602, 0.00417207])
        self.camera_center_gopro = None
        self.camera_orientation_gopro = None
        self.GET_PARAMS_GOPRO = True

                                                # === CONSTANTS USBCAMERA ===
        self.timestamp_usbcamera = None
        self.camera_matrix_usbcamera = np.array(((1248.7537961358316, 0.000000, 972.91303286185735),
                                                 (0.000000, 1244.8793353282827, 546.12095519076013),
                                                 (0.000000, 0.000000, 1.000000)))
        self.dist_usbcamera = np.array([-0.29141956539556901, 0.09518440179444243, -0.0022242325091741334, -1.5895164604828018e-05, 0.0])
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
        Publish transform: aruco_marker -> camera_frame
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
        rospy.loginfo(f"Set up transform chain: aruco_marker -> {camera_frame}")

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
            if not self.tf_buffer.can_transform(
                "aruco_marker", source_frame, timestamp, rospy.Duration(1.0)
            ):
                rospy.logwarn(f"Transform from {source_frame} to aruco_marker not available")
                return point_array

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
        Create aruco instance in order to get the fixed world coordinates of the usbcamera relative to the aruco marker
        """
        if not self.GET_PARAMS_USBCAMERA:
            return
        
        if image_msg is not None and self.GET_PARAMS_USBCAMERA is True:
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


    def refract_dir(self, i, n, n1, n2):
        """
        Calculate the refracted direction of a ray given the incident direction, normal, and refractive indices.
        Uses Snell's law to compute the refraction.
        :param i: Incident direction vector of the ray
        :param n: Normal vector at the point of incidence
        :param n1: Refractive index of the medium from which the ray is coming
        :param n2: Refractive index of the medium into which the ray is entering
        """
        i = i / np.linalg.norm(i)
        n = n / np.linalg.norm(n)
        cosi = float(np.clip(np.dot(i, n), -1.0, 1.0)) # limit to [-1, 1].
        etai, etat = n1, n2
        n_use = n
        if cosi > 0.0: # ray and normal are pointing the same way → we are leaving the medium, so swap indices and flip normal
            n_use = -n_use
            etai, etat = etat, etai
            cosi = -cosi
        eta = etai / etat
        k = 1.0 - eta**2 * (1.0 - cosi**2) # Snell's law
        if k < 0.0:
            return None # Total internal reflection
        t = eta * i + (eta * cosi - math.sqrt(k)) * n_use # Refracted direction

        return t / np.linalg.norm(t)
    
    
    def pixel_to_3d_point(self, pixel_x, pixel_y, depth, K, camera_center, camera_orientation, dist_coeffs, water_n=1.333, air_n=1.00029):
        """
        Convert pixel coordinates to 3D point in camera frame.
        See ray and backward projection from the following sources:
        Source: https://costinchitic.wiki/notes/camera-backward-projection
        https://costinchitic.wiki/notes/Multiple-View-Geometry-in-Computer-Vision
        https://costinchitic.wiki/notes/coordinate-frame
        """
        # Convert pixel to normalized camera coordinates
        pts = np.array([[[pixel_x, pixel_y]]], dtype=np.float64)
        x_n, y_n = cv2.undistortPoints(pts, K, dist_coeffs)[0,0]
        dC = np.array([x_n, y_n, 1.0], dtype=np.float64)               # not unit
        d_air_W = camera_orientation @ dC
        o_W = camera_center

        # 2) intersect with water surface z=0
        denom = d_air_W[2]
        if abs(denom) < 1e-9:
            rospy.logwarn("Ray parallel to water surface")
            return None
        s0 = (0.0 - o_W[2]) / denom
        if s0 <= 0:
            rospy.logwarn(f"Ray does not intersect water surface, or camera is below water surface, camera_z = {o_W[2]}")
            return None
        I0_W = o_W + s0 * d_air_W

        # 3) compute refracted direction into water
        # normal from air -> water (z up): n = [0,0,1]
        n_surface_up = np.array([0.0, 0.0, 1.0])
        d_in = d_air_W / np.linalg.norm(d_air_W)
        d_wtr = self.refract_dir(d_in, n_surface_up, air_n, water_n)
        if d_wtr is None: 
            rospy.logwarn("Ray does not refract into water, total internal reflection")
            return None
        X0_W = I0_W

        # 4) intersect refracted ray with plane z = -depth
        denom2 = d_wtr[2]
        if abs(denom2) < 1e-9:
            rospy.logwarn("Ray parallel to water surface after refraction")
            return None
        
        s1 = ((float(depth)) - X0_W[2]) / denom2
        if s1 <= 0:
            rospy.logwarn(f"Ray does not intersect water surface at depth {depth}, or camera is below water surface, camera_z = {X0_W[2]}")
            return None
        P_W = X0_W + s1 * d_wtr

        # 5) return camera-frame point (so your TF step stays unchanged)
        R_CW = camera_orientation.T
        p_C = R_CW @ (P_W - o_W)
        return p_C

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
                                                  self.camera_orientation_gopro,
                                                  self.dist_gopro,
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
            seq_gopro += 1
            
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
                                                  self.camera_orientation_usbcamera,
                                                  self.dist_usbcamera,
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
            seq_usbcamera += 1
            
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
