#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from mpl_toolkits.mplot3d import Axes3D
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import time
from math import cos, sin
import math
from collections import deque

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

        while self.running:
            try:
                with self.lock:
                    self._update_plot()
                    
                    # Save plot to image
                    self.fig.canvas.draw()
                    
                    # Get the RGB data from the figure canvas in a way that works with newer Matplotlib
                    from matplotlib.backends.backend_agg import FigureCanvasAgg
                    canvas = FigureCanvasAgg(self.fig)
                    canvas.draw()
                    
                    w, h = self.fig.canvas.get_width_height()
                    # Using buffer_rgba instead of tostring_rgb
                    buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
                    buf.shape = (h, w, 4)
                    # Convert RGBA to RGB
                    buf = buf[:,:,:3]
                    
                    # Convert RGB to BGR for OpenCV
                    self.plot_image = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
                    time.sleep(0.1)

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
        self.x_world_gopro = None
        self.y_world_gopro = None
        self.z_world_gopro = None
        self.timestamp_gopro = None

        self.camera_matrix_gopro = np.array(((927.091270, 0.000000, 957.570804), 
                                       (0.000000, 919.995427, 533.540912), 
                                       (0.000000, 0.000000, 1.000000)))
        self.K_gopro = np.array(((927.091270, 0.000000, 957.570804, 0.000000), 
                           (0.000000, 919.995427, 533.540912, 0.000000), 
                           (0.000000, 0.000000, 1.000000, 0.000000)))
        self.dist_gopro = np.array((0.05, 0.07, -0.11, 0.05, 0.000000))

                                                # === CONSTANTS USBCAMERA ===

        self.x_world_usbcamera = None
        self.y_world_usbcamera = None
        self.z_world_usbcamera = None
        self.scale_x = None
        self.scale_y = None
        self.timestamp_usbcamera = None

        self.dist_usbcamera = np.array([-0.274309, 0.075813, -0.000209, -0.000607, 0.0])

        # Initialize the plotter
        self.plotter = DualTrajectoryPlotter(plot_3d=False)
        self.plotter.start_plotting()
                                                # Initialize ROS components

        rospy.init_node('dual_trajectory_computer', anonymous=True)
        self.bridge = CvBridge()
        # Add a timer to publish the plot regularly
        rospy.Timer(rospy.Duration(0.5), self.publish_plot)  # Publish at 2Hz
    
        # Publishers
        self.plot_pub = rospy.Publisher('/trajectory_plot', Image, queue_size=10)
        
    def publish_plot(self, event):
        if self.plotter:
            plot_img = self.plotter.get_plot_image()
            if plot_img is not None:
                try:
                    img_msg = self.bridge.cv2_to_imgmsg(plot_img, encoding="bgr8")
                    self.plot_pub.publish(img_msg)
                except Exception as e:
                    rospy.logerr(f"Error publishing plot image: {e}")

    def depth_callback(self, depth):
        self.z_world_gopro = -depth.pose.pose.position.z
        self.z_world_usbcamera = -depth.pose.pose.position.z

    def usbcamera_timestamp_callback(self, timestamp):
        self.timestamp_usbcamera = timestamp

    def gopro_timestamp_callback(self, timestamp):
        self.timestamp_gopro = timestamp

    def usbcamera_imagesize(self, image_msg):
        w, h = image_msg.width, image_msg.height
        self.scale_x = w / 3840
        self.scale_y = h / 2160

    def gopro_callback(self, msg):

        timestamp = self.timestamp_gopro

                                                    # POSE ESTIMATION

        inv_cam_matrix = np.linalg.inv(self.camera_matrix_gopro)
        centre = np.array([msg.point.x, msg.point.y, msg.point.z]).reshape(-1, 1)
        centre = np.vstack(centre)
        
        C = np.array([0, 0, 0])
        C = np.vstack(C)
        
        M_int = np.dot(inv_cam_matrix, centre)
        d = M_int - C
        
        roll = 0
        pitch = 0
        yaw = 0
        
        R_roll = np.matrix([[1, 0, 0], [0, cos(roll), sin(roll)], [0, -sin(roll), cos(roll)]])
        R_pitch = np.matrix([[cos(pitch), 0, -sin(pitch)], [0, 1, 0], [sin(pitch), 0, cos(pitch)]])
        R_yaw = np.matrix([[cos(yaw), sin(yaw), 0], [-sin(yaw), cos(yaw), 0], [0, 0, 1]])
        R = R_yaw * R_pitch * R_roll
        
        inv_R = np.linalg.inv(R)
        
        u, v = msg.point.x, msg.point.y
        image_coord = np.array([u, v, 1]).reshape(-1, 1)
        camera_coord = np.dot(inv_cam_matrix, image_coord)
        dirr = camera_coord - C
        dwor = np.dot(inv_R, dirr)
        
        camera_coord = np.append(camera_coord, 1)
        homo_points = image_coord
        depth = self.z_world_gopro
        if depth:
            zet = 2285 - depth
        else:
            zet = 2285 # Distance from GoPro to Water Surface.
        homo_points = homo_points * zet
        world_coord = np.linalg.inv(self.camera_matrix_gopro) @ homo_points
        world_coord = np.squeeze(np.asarray(world_coord / 1000))
        
        # Publish odometry
        global seq_gopro, last_point_gopro
        cov = [0.0] * 36
        cov[0] = 6
        cov[7] = 6
        odom = Odometry()
        odom.pose.covariance = cov
        odom.header.frame_id = "world"
        odom.header.stamp = timestamp
        odom.header.seq = seq_gopro
        seq_gopro += 1
        odom.child_frame_id = "gopro"
        
        self.x_world_gopro = world_coord[0]
        self.y_world_gopro = world_coord[1]
        
        odom.pose.pose.position.x = world_coord[0]
        odom.pose.pose.position.y = world_coord[1]
        
        # Coordinate transformation
        old_x = odom.pose.pose.position.x
        old_y = odom.pose.pose.position.y
        new_x = -old_y
        new_y = old_x
        odom.pose.pose.position.x = new_x
        odom.pose.pose.position.y = new_y
        
        # Outlier detection
        x_last = last_point_gopro.pose.pose.position.x
        y_last = last_point_gopro.pose.pose.position.y
        x_current = odom.pose.pose.position.x
        y_current = odom.pose.pose.position.y
        max_distance_jump = 1.5
        
        if seq_gopro == 1:
            last_point_gopro = odom
        elif math.sqrt((x_current - x_last)**2 + (y_current - y_last)**2) > max_distance_jump:
            odom.pose.pose.position.x = last_point_gopro.pose.pose.position.x
            odom.pose.pose.position.y = last_point_gopro.pose.pose.position.y
        else:
            last_point_gopro = odom
        
        ground_truth.publish(odom)

        # Add position to plotter
        self.plotter.add_gopro_position(self.x_world_gopro, self.y_world_gopro, self.z_world_gopro, timestamp)

    def usbcamera_callback(self, msg):

        timestamp = self.timestamp_usbcamera
        # Scale intrinsic parameters
        fx_scaled = 2481.80467 * self.scale_x
        fy_scaled = 2484.001002 * self.scale_y
        cx_scaled = 1891.149796 * self.scale_x
        cy_scaled = 1079.160663 * self.scale_y

        camera_matrix = np.array([
            [fx_scaled, 0, cx_scaled],
            [0, fy_scaled, cy_scaled],
            [0, 0, 1]
        ])
        dist = self.dist_usbcamera

        cX = msg.point.x
        cY = msg.point.y

        # You need to measure/calibrate these values for your specific setup
        # Camera position relative to world frame (in meters)
        # In our case: Inaltime camera usb = 277[cm], Inaltimea apei = 96 [cm] => h = 1.81 [m]
        #              Distanta camera - punct_ref_piscina = TODO
        #              Piscina(Grilaj): 435 cm x 180 cm (29 x 12 grid. 15 cm fiecare patrat)
        camera_height = 1.81
        camera_translation = np.array([0.0, 0.0, camera_height])  # Camera position in meters
        camera_rotation = np.array([0.0, -0.34906585, 0.0])  # Camera orientation in radians (roll, pitch, yaw)
        robot_dimensions = np.array([0.457, 0.338, 0.254])  # Robot dimensions in meters

        total_depth = camera_height + msg.point.z # Camera to water surface + depth of robot
        pixel_coords = np.array([cX, cY, 1.0]).reshape(-1, 1)
        inv_camera_matrix = np.linalg.inv(camera_matrix)
        camera_coords = np.dot(inv_camera_matrix, pixel_coords)

        world_x = camera_coords[0] * total_depth
        world_y = camera_coords[1] * total_depth
        world_z = -total_depth  # Negative because it's below the camera

        new_x = world_x    # Camera's z becomes world's x
        new_y = world_y   # Camera's x becomes world's y  
        new_z = world_z  # Camera's y becomes world's z (inverted)

        #print("ROV Position (world frame usbcamera): X = {:.3f}, Y = {:.3f}, Z = {:.3f}".format(new_x, new_y, new_z))

        # Create Odometry message (similar to ArUco code)
        global seq_usbcamera, last_point_usbcamera
        odom = Odometry()
        cov = [0.0] * 36
        cov[0] = 6
        cov[7] = 6
        odom.pose.covariance = cov
        odom.header.frame_id = "world"
        odom.header.stamp = timestamp
        odom.header.seq = seq_usbcamera
        seq_usbcamera += 1
        odom.child_frame_id = "usbcamera"
        
        odom.pose.pose.position.x = new_x
        odom.pose.pose.position.y = new_y
        odom.pose.pose.position.z = new_z
        
        # Store world coordinates in class variables
        self.x_world_usbcamera = new_x
        self.y_world_usbcamera = new_y
        self.z_world_usbcamera = new_z
        
        x_last = last_point_usbcamera.pose.pose.position.x
        y_last = last_point_usbcamera.pose.pose.position.y
        x_current = odom.pose.pose.position.x
        y_current = odom.pose.pose.position.y
        max_distance_jump = 1.5  # meters
        
        if seq_usbcamera == 2:  # First detection
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
        self.plotter.add_usbcamera_position(self.x_world_usbcamera, self.y_world_usbcamera, self.z_world_usbcamera, timestamp)

estimation = rospy.Publisher('/BlueRov2/estimation', Odometry, queue_size=10)
ground_truth = rospy.Publisher('/BlueRov2/groundtruth', Odometry, queue_size=10)

def main():
    # CONFIGURATION: Set to True for 3D plotting, False for 2D plotting
    USE_3D_PLOT = False  # Change this to False for 2D plotting
    
    try:
        computer = DualTrajectoryComputer()
        # Subscribers for coordinate data
        rospy.Subscriber('/camera_wall_time', Float64, computer.gopro_timestamp_callback)
        rospy.Subscriber('/gopro/wall_time', Float64, computer.usbcamera_timestamp_callback)
        rospy.Subscriber('/BlueRov2/odom/depth', Odometry, computer.depth_callback)
        rospy.Subscriber('/camera/image_compressed', Image, computer.usbcamera_imagesize)
        rospy.Subscriber('/robotdetection/coordinates_gopro', PointStamped, computer.gopro_callback)
        rospy.Subscriber('/robotdetection/coordinates_usbcamera', PointStamped, computer.usbcamera_callback)

        rospy.loginfo("Initializing DualTrajectoryComputer...")
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")

if __name__ == '__main__':
    main()