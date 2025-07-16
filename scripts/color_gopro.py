#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rospy
from math import cos, sin
from cv_bridge import CvBridge
from sensor_msgs.msg import Image # For plotting
from nav_msgs.msg import Odometry # for odometry
from std_msgs.msg import Float64 # for timestamp
from geometry_msgs.msg import PointStamped
import math
import threading # for plotting thread
import time 
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend. Otherwise error
from mpl_toolkits.mplot3d import Axes3D # for 3D plotting
from matplotlib.backends.backend_agg import FigureCanvasAgg

seq = 1
last_point = Odometry()
bridge = CvBridge()

class PlotVisualizer:
    def __init__(self, max_points=100000, plot_3d=True):
        self.max_points = max_points
        self.positions = []
        self.lock = threading.Lock()
        self.fig = None
        self.ax = None
        self.running = True
        self.plot_thread = None
        self.plot_3d = plot_3d  # Boolean: True for 3D, False for 2D
        
    def start_plotting(self):
        """Start the plotting thread"""
        self.plot_thread = threading.Thread(target=self._plot_loop, daemon=True)
        self.plot_thread.start()
        
    def add_position(self, x, y, z, timestamp):
        """Add a new position to plot"""
        with self.lock:
            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                    rospy.logwarn("Invalid position values (NaN detected)")
                    return
            self.positions.append((float(x), float(y), float(z), timestamp.to_sec()))
            
    def _plot_loop(self):
        """Main plotting loop running in separate thread"""
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=(10, 8))
        
        # Create 2D or 3D subplot based on plot_3d setting
        if self.plot_3d:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)
        
        while self.running:
            try:
                with self.lock:
                    if len(self.positions) > 0:
                        self._update_plot()
                        
                # Save plot as image
                self.fig.canvas.draw()
                canvas = FigureCanvasAgg(self.fig)
                canvas.draw()
                
                w, h = self.fig.canvas.get_width_height()
                # Using buffer_rgba instead of tostring_rgb
                buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
                buf.shape = (h, w, 4)
                # Convert RGBA to RGB
                buf = buf[:,:,:3]
                self.plot_image = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
                
                time.sleep(0.1)  # Update at 10Hz
                
            except Exception as e:
                print(f"Plot error: {e}")
                time.sleep(1)
                
    def _update_plot(self):
        """Update the 2D or 3D plot based on plot_3d setting"""
        self.ax.clear()
        
        if len(self.positions) == 0:
            return
            
        # Extract coordinates and metadata
        positions_array = np.array(self.positions)
        x_coords = positions_array[:, 0]
        y_coords = positions_array[:, 1] 
        z_coords = positions_array[:, 2]
        
        if self.plot_3d:
            # 3D plotting
            # Plot trajectory line
            self.ax.plot(x_coords, y_coords, z_coords, 'b-', alpha=0.6, linewidth=1)
            
            # Plot points
            self.ax.scatter(x_coords, y_coords, z_coords, c='yellow', s=50, alpha=0.8)
            
            # Highlight most recent point
            if len(self.positions) > 0:
                latest = self.positions[-1]
                self.ax.scatter([latest[0]], [latest[1]], [latest[2]], 
                              c='yellow', s=100, marker='*', edgecolors='black', linewidth=2)
            
            # Set labels and title for 3D
            self.ax.set_xlabel('X Position (m)')
            self.ax.set_ylabel('Y Position (m)')
            self.ax.set_zlabel('Z Position (m)')
            self.ax.set_title(f'GoPro ROV Trajectory\n{len(self.positions)} points')
            
            # Set reasonable axis limits for 3D
            if len(self.positions) > 1:
                x_range = max(x_coords) - min(x_coords)
                y_range = max(y_coords) - min(y_coords)
                z_range = max(z_coords) - min(z_coords)
                
                center_x, center_y, center_z = np.mean(x_coords), np.mean(y_coords), np.mean(z_coords)
                max_range = max(x_range, y_range, z_range) / 2 + 1
                
                self.ax.set_xlim(center_x - max_range, center_x + max_range)
                self.ax.set_ylim(center_y - max_range, center_y + max_range)
                self.ax.set_zlim(center_z - max_range, center_z + max_range)
            
            self.ax.grid(True, alpha=0.3)
            
        else:
            # 2D plotting (X-Y plane)
            # Plot trajectory line
            self.ax.plot(x_coords, y_coords, 'b-', alpha=0.8, linewidth=1, label='Trajectory')
            
            # Highlight most recent point
            if len(self.positions) > 0:
                latest = self.positions[-1]
                self.ax.scatter([latest[0]], [latest[1]], 
                              c='yellow', s=100, marker='*', edgecolors='black', linewidth=2, 
                              label='Latest', zorder=5)
            
            # Set labels and title for 2D
            self.ax.set_xlabel('X Position (m)')
            self.ax.set_ylabel('Y Position (m)')
            self.ax.set_title(f'GoPro ROV 2D Trajectory\n{len(self.positions)} points')
            
            # Set equal aspect ratio for 2D
            self.ax.set_aspect('equal', adjustable='box')
            
            # Set reasonable axis limits for 2D
            if len(self.positions) > 1:
                x_range = max(x_coords) - min(x_coords)
                y_range = max(y_coords) - min(y_coords)
                
                center_x, center_y = np.mean(x_coords), np.mean(y_coords)
                max_range = max(x_range, y_range) / 2 + 1
                
                self.ax.set_xlim(center_x - max_range, center_x + max_range)
                self.ax.set_ylim(center_y - max_range, center_y + max_range)

            self.ax.grid(True, alpha=0.3)
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        
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

class Task:
    def __init__(self, plot_3d=True):
        self.x_world = None
        self.y_world = None
        self.z_world = None
        self.timestamp = None
        self.plotter = PlotVisualizer(max_points=100000, plot_3d=plot_3d)
        self.plotter.start_plotting()

        plot_mode = "3D" if plot_3d else "2D"
        print(f"Initialized with {plot_mode} plotting mode")

        # === CONSTANTS ===
        self.camera_matrix = np.array(((927.091270, 0.000000, 957.570804), 
                                       (0.000000, 919.995427, 533.540912), 
                                       (0.000000, 0.000000, 1.000000)))
        self.K = np.array(((927.091270, 0.000000, 957.570804, 0.000000), 
                           (0.000000, 919.995427, 533.540912, 0.000000), 
                           (0.000000, 0.000000, 1.000000, 0.000000)))
        self.dist = np.array((0.05, 0.07, -0.11, 0.05, 0.000000))

        # Publishers
        self.plot_pub = rospy.Publisher('gopro_trajectory', Image, queue_size=10)
        self.real_pub = rospy.Publisher('/BlueRov2/real_coord', Odometry, queue_size=10)
        rospy.Timer(rospy.Duration(0.1), self.publish_plot_image)  # Publish plot image every 0.1 seconds

    def publish_plot_image(self, event):
        """Publish the current plot image to the ROS topic"""
        plot_image = self.plotter.get_plot_image()
        if plot_image is not None:
            try:
                # Convert OpenCV image to ROS Image message
                img_msg = bridge.cv2_to_imgmsg(plot_image, encoding="bgr8")
                if self.timestamp:
                    img_msg.header.stamp = self.timestamp
                else:
                    img_msg.header.stamp = rospy.Time.now()
                    rospy.logwarn("Timestamp not set, using current time")
                self.plot_pub.publish(img_msg)
            except Exception as e:
                rospy.logerr(f"Error publishing plot image: {e}")

        else:
            rospy.logwarn("No plot image available to publish")

    def depth_callback(self, depth):
        self.z_world = -depth.pose.pose.position.z

    def timestamp_callback(self, timestamp):
        self.timestamp = rospy.Time.from_sec(timestamp.data)

    def gopro_callback(self, msg):

        timestamp = self.timestamp
        if timestamp is None:
            rospy.logwarn("Timestamp not set yet, using current time")
            timestamp = rospy.Time.now()  # Use current time if timestamp is not set

                                                    # POSE ESTIMATION

        inv_cam_matrix = np.linalg.inv(self.camera_matrix)
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
        depth = self.z_world
        if depth:
            zet = 2285 - depth
        else:
            zet = 2285 # Distance from GoPro to Water Surface.
        homo_points = homo_points * zet
        world_coord = np.linalg.inv(self.camera_matrix) @ homo_points
        world_coord = np.squeeze(np.asarray(world_coord / 1000))
        
        # Publish odometry
        global seq, last_point
        cov = [0.0] * 36
        cov[0] = 6
        cov[7] = 6
        odom = Odometry()
        odom.pose.covariance = cov
        odom.header.frame_id = "world"
        odom.header.stamp = timestamp
        odom.header.seq = seq
        seq += 1
        odom.child_frame_id = "gopro"
        
        self.x_world = world_coord[0]
        self.y_world = world_coord[1]
        
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
        x_last = last_point.pose.pose.position.x
        y_last = last_point.pose.pose.position.y
        x_current = odom.pose.pose.position.x
        y_current = odom.pose.pose.position.y
        max_distance_jump = 1.5
        
        if seq == 1:
            last_point = odom
        elif math.sqrt((x_current - x_last)**2 + (y_current - y_last)**2) > max_distance_jump:
            odom.pose.pose.position.x = last_point.pose.pose.position.x
            odom.pose.pose.position.y = last_point.pose.pose.position.y
        else:
            last_point = odom

        # Add position to plotter and publish updated odometry
        self.real_pub.publish(odom)
        self.plotter.add_position(odom.pose.pose.position.y, odom.pose.pose.position.x, self.z_world, timestamp)

    def cleanup(self):
        """Clean up resources"""
        self.plotter.stop()


world = rospy.Publisher('/BlueRov2/plane', Odometry, queue_size=10)
real = rospy.Publisher('/BlueRov2/real_coord', Odometry, queue_size=10)

if __name__ == '__main__':
    # CONFIGURATION: Set to True for 3D plotting, False for 2D plotting
    USE_3D_PLOT = True  # Change this to False for 2D plotting
    
    rospy.init_node('listener', anonymous=True)
    task = Task(plot_3d=USE_3D_PLOT)
    
    try:
        rospy.Subscriber('/gopro/wall_time', Float64, task.timestamp_callback)
        rospy.Subscriber('/robotdetection/coordinates_gopro', PointStamped, task.gopro_callback)
        rospy.Subscriber('/BlueRov2/odom/depth', Odometry, task.depth_callback)
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        task.cleanup()