#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from mpl_toolkits.mplot3d import Axes3D
from geometry_msgs.msg import PointStamped, TransformStamped
import tf2_ros, tf.transformations, tf2_geometry_msgs
import geometry_msgs.msg
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import time
from math import cos, sin
import math
from collections import deque   
from aruco import ArucoTask
from matplotlib.backends.backend_agg import FigureCanvasAgg

class DualTrajectoryPlotter:

    def __init__(self, max_points=100000, plot_3d=True):
        self.max_points = max_points
        self.gopro_positions = []
        self.usbcamera_positions = []
        self.lock = threading.Lock()
        self.fig = None
        self.ax = None
        self.running = True
        self.plot_thread = None
        self.plot_3d = plot_3d  # Boolean: True for 3D, False for 2D

    def start_plotting(self):
        self.plot_image = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(self.plot_image, "Initializing plot...", (250, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                   
        self.plot_thread = threading.Thread(target=self._plot_loop, daemon=True)
        self.plot_thread.start()

    def add_gopro_position(self, x, y, z, timestamp):
        with self.lock:
            if hasattr(timestamp, 'to_sec'):
                timestamp = timestamp.to_sec()
            else:
                timestamp = float(timestamp.data)
            self.gopro_positions.append((float(x), float(y), float(z), timestamp))
            if len(self.gopro_positions) > self.max_points:
                self.gopro_positions.pop(0) # Prevent memory overflow

    def add_usbcamera_position(self, x, y, z, timestamp):
        with self.lock:
            if hasattr(timestamp, 'to_sec'):
                timestamp = timestamp.to_sec()
            else:
                timestamp = float(timestamp.data)
            self.usbcamera_positions.append((float(x), float(y), float(z), timestamp))
            if len(self.usbcamera_positions) > self.max_points:
                self.usbcamera_positions.pop(0) # Prevent memory overflow

    def _plot_loop(self):
        plt.ion() # Interactive mode on
        self.fig = plt.figure(figsize=(10, 8))
        if self.plot_3d:
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')

        self.canvas = FigureCanvasAgg(self.fig)

        while self.running:
            try:
                with self.lock:
                    self._update_plot()
                    time.sleep(0.1) # Reduce CPU usage
                    
                    # Save plot to image
                    self.fig.canvas.draw()
                    
                    # More efficient buffer conversion
                    renderer = self.fig.canvas.get_renderer()
                    raw_data = renderer.tostring_rgb()
                    size = self.fig.canvas.get_width_height()
                    
                    # Direct conversion to BGR
                    img_array = np.frombuffer(raw_data, dtype=np.uint8)
                    img_array = img_array.reshape((size[1], size[0], 3))
                    self.plot_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            except Exception as e:
                rospy.logerr(f"Error in plot loop: {e}")
                time.sleep(1)

    def _update_plot(self):
        self.ax.clear()

        # Check if we have any data to plot
        if not self.gopro_positions and not self.usbcamera_positions:
            self.ax.text(0.5, 0.5, "Waiting for trajectory data...", 
                        ha='center', va='center', transform=self.ax.transAxes)
            return

        try:
            if self.plot_3d:
                if self.gopro_positions:
                    gopro_array = np.array(self.gopro_positions, dtype=float)
                    gopro_x = gopro_array[:, 0]
                    gopro_y = gopro_array[:, 1]
                    gopro_z = gopro_array[:, 2]
                    self.ax.plot(gopro_x, gopro_y, gopro_z, 'g-', alpha=0.8, linewidth=2, label='GoPro (Ground Truth)')
                    self.ax.scatter(gopro_x, gopro_y, gopro_z, c='green', s=30, alpha=0.6)
                    
                    # Highlight most recent point
                    latest = self.gopro_positions[-1]
                    self.ax.scatter([latest[0]], [latest[1]], [latest[2]], 
                                    c='lime', s=100, marker='*', edgecolors='black', linewidth=2)
                
                if self.usbcamera_positions:
                    usbcamera_array = np.array(self.usbcamera_positions, dtype=float)
                    usb_x = usbcamera_array[:, 0]
                    usb_y = usbcamera_array[:, 1]
                    usb_z = usbcamera_array[:, 2]
                    self.ax.plot(usb_x, usb_y, usb_z, 'b-', alpha=0.8, linewidth=2, label='USB Camera (Estimation)')
                    self.ax.scatter(usb_x, usb_y, usb_z, c='blue', s=30, alpha=0.6)
                    
                    # Highlight most recent point
                    latest = self.usbcamera_positions[-1]
                    self.ax.scatter([latest[0]], [latest[1]], [latest[2]], 
                                    c='cyan', s=100, marker='*', edgecolors='black', linewidth=2)
                
            else:
                if self.gopro_positions:
                    gopro_array = np.array(self.gopro_positions, dtype=float)
                    gopro_x = gopro_array[:, 0]
                    gopro_y = gopro_array[:, 1]
                    self.ax.plot(gopro_x, gopro_y, 'g-', alpha=0.8, linewidth=2, label='GoPro (Ground Truth)')
                    self.ax.scatter(gopro_x, gopro_y, c='green', s=30, alpha=0.6)
                    
                    # Highlight most recent point
                    latest = self.gopro_positions[-1]
                    self.ax.scatter([latest[0]], [latest[1]], 
                                    c='lime', s=100, marker='*', edgecolors='black', linewidth=2)
                
                if self.usbcamera_positions:
                    usbcamera_array = np.array(self.usbcamera_positions, dtype=float)
                    usb_x = usbcamera_array[:, 0]
                    usb_y = usbcamera_array[:, 1]
                    self.ax.plot(usb_x, usb_y, 'b-', alpha=0.8, linewidth=2, label='USB Camera (Estimation)')
                    self.ax.scatter(usb_x, usb_y, c='blue', s=30, alpha=0.6)
                    
                    # Highlight most recent point
                    latest = self.usbcamera_positions[-1]
                    self.ax.scatter([latest[0]], [latest[1]], 
                                    c='cyan', s=100, marker='*', edgecolors='black', linewidth=2)
                
            self.ax.grid(True, alpha=0.3)
            self.ax.legend(loc='upper left')
            # For 2D plot, set equal aspect ratio
            if not self.plot_3d:
                self.ax.set_aspect('equal', adjustable='box')
                
            # Add position info text
            if self.gopro_positions:
                latest_gopro = self.gopro_positions[-1]
                gopro_text = f"GoPro: ({latest_gopro[0]:.2f}, {latest_gopro[1]:.2f})"
                self.ax.text(0.02, 0.95, gopro_text, transform=self.ax.transAxes, 
                            color='green', fontweight='bold')
                
            if self.usbcamera_positions:
                latest_usb = self.usbcamera_positions[-1]
                usb_text = f"USB: ({latest_usb[0]:.2f}, {latest_usb[1]:.2f})"
                self.ax.text(0.02, 0.90, usb_text, transform=self.ax.transAxes, 
                            color='blue', fontweight='bold')
        
        except Exception as e:
            rospy.logerr(f"Error in _update_plot: {e}")
            # Display error on plot
            self.ax.text(0.5, 0.5, f"Error plotting: {e}", 
                        ha='center', va='center', transform=self.ax.transAxes,
                        color='red', fontsize=12)

    def get_plot_image(self):
        """Get the current plot as OpenCV image"""
        if hasattr(self, 'plot_image'):
            return self.plot_image.copy()
        else:
            # Return a blank image if plot not ready
            return np.zeros((600, 800, 3), dtype=np.uint8)
        
    def stop(self):
        """Stop the plotting thread"""
        self.running = False
        if self.plot_thread:
            self.plot_thread.join()
        plt.close(self.fig)

seq_gopro = 1
seq_usbcamera = 1
last_point_gopro = Odometry()
last_point_usbcamera = Odometry()
bridge = CvBridge()

class DualTrajectoryComputer:

    def __init__(self):

                                                # === CONSTANTS GOPRO ===
        
        self.timestamp_gopro = None
        self.camera_matrix_gopro = np.array(((927.091270, 0.000000, 957.570804), 
                                       (0.000000, 919.995427, 533.540912), 
                                       (0.000000, 0.000000, 1.000000)))
        self.dist_gopro = np.array((0.05, 0.07, -0.11, 0.05, 0.000000))
        self.camera_center_gopro = None  # 3D position of GoPro in world frame
        self.camera_orientation_gopro = None  # Rotation matrix for GoPro
        self.GET_PARAMS_GOPRO = True

                                                # === CONSTANTS USBCAMERA ===

        self.timestamp_usbcamera = None
        self.camera_matrix_usbcamera = None
        self.dist_usbcamera = np.array([-0.274309, 0.075813, -0.000209, -0.000607, 0.0])
        self.camera_center_usbcamera = None  # 3D position of USB camera in world frame
        self.camera_orientation_usbcamera = None  # Rotation matrix for USB camera
        self.GET_PARAMS_USBCAMERA = True

        # Initialize the plotter
        self.plotter = DualTrajectoryPlotter(plot_3d=False)
        self.plotter.start_plotting()
                                                # Initialize ROS components

        rospy.init_node('dual_trajectory_computer', anonymous=True)
        # tf2 components
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        self.bridge = CvBridge()
        # Add a timer to publish the plot regularly
        rospy.Timer(rospy.Duration(0.1), self.publish_plot)
    
        # Publishers
        self.plot_pub = rospy.Publisher('/trajectory_plot', Image, queue_size=10)

    def publish_camera_to_aruco_transform(self, rvec, tvec, camera_frame, timestamp):
        """
        Publish transform: camera_frame -> aruco_marker
        This represents where the marker is relative to the camera.
        """
        if timestamp is None:
            timestamp = rospy.Time.now()
            
        transform = TransformStamped()
        transform.header.stamp = timestamp
        transform.header.frame_id = camera_frame
        transform.child_frame_id = "aruco_marker"
        
        transform.transform.translation.x = float(tvec[0])
        transform.transform.translation.y = float(tvec[1])
        transform.transform.translation.z = float(tvec[2])
        
        R, _ = cv2.Rodrigues(rvec)
        quat = tf.transformations.quaternion_from_matrix(
            np.vstack([np.hstack([R, [[0], [0], [0]]]), [0, 0, 0, 1]])
        )
        
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(transform)
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

    def publish_plot(self, event):
        if self.plotter:
            plot_img = self.plotter.get_plot_image()
            if plot_img is not None:
                try:
                    img_msg = self.bridge.cv2_to_imgmsg(plot_img, encoding="bgr8")
                    self.plot_pub.publish(img_msg)
                except Exception as e:
                    rospy.logerr(f"Error publishing plot image: {e}")

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

                timestamp = self.timestamp_gopro if self.timestamp_gopro else rospy.Time.now()
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

                timestamp = self.timestamp_usbcamera if self.timestamp_usbcamera else rospy.Time.now()
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
        alfa       = (plane_z - camera_center[2]) / (camera_orientation @ ray_cam)[2]
        
        # Return intersection point in camera frame
        camera_point_3d = alfa * ray_cam
        return camera_point_3d

    def gopro_coordinates_callback(self, msg):

        timestamp = self.timestamp_gopro
        
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
            
            # Add position to plotter
            self.plotter.add_gopro_position(world_coord[0], world_coord[1], world_coord[2], timestamp)
        
    def usbcamera_coordinates_callback(self, msg):

        timestamp = self.timestamp_usbcamera

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
            self.plotter.add_usbcamera_position(world_coord[0], world_coord[1], world_coord[2], timestamp)

estimation = rospy.Publisher('/BlueRov2/estimation', Odometry, queue_size=10)
ground_truth = rospy.Publisher('/BlueRov2/groundtruth', Odometry, queue_size=10)

def main():
    # CONFIGURATION: Set to True for 3D plotting, False for 2D plotting
    USE_3D_PLOT = False  # Change this to False for 2D plotting
    
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