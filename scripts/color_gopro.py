#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from math import cos, sin
from nav_msgs.msg import Odometry
import math
import threading
import time
from collections import deque
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from mpl_toolkits.mplot3d import Axes3D

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
        
    def add_position(self, x, y, z, confidence):
        """Add a new position to plot"""
        with self.lock:
            self.positions.append((x, y, z, confidence, time.time()))
            
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
                self.fig.canvas.flush_events()
                
                # Convert plot to OpenCV image
                buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
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
        confidences = positions_array[:, 3]
        timestamps = positions_array[:, 4]
        
        # Color points by confidence (green = high, red = low)
        colors = []
        for conf in confidences:
            if conf > 70:
                colors.append('green')
            elif conf > 50:
                colors.append('orange') 
            else:
                colors.append('red')
        
        if self.plot_3d:
            # 3D plotting
            # Plot trajectory line
            self.ax.plot(x_coords, y_coords, z_coords, 'b-', alpha=0.6, linewidth=1)
            
            # Plot points
            self.ax.scatter(x_coords, y_coords, z_coords, c=colors, s=50, alpha=0.8)
            
            # Highlight most recent point
            if len(self.positions) > 0:
                latest = self.positions[-1]
                self.ax.scatter([latest[0]], [latest[1]], [latest[2]], 
                              c='yellow', s=100, marker='*', edgecolors='black', linewidth=2)
            
            # Set labels and title for 3D
            self.ax.set_xlabel('X Position (m)')
            self.ax.set_ylabel('Y Position (m)')
            self.ax.set_zlabel('Z Position (m)')
            self.ax.set_title(f'Tank Detection 3D Trajectory\n{len(self.positions)} points')
            
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
            
            # Add grid
            self.ax.grid(True, alpha=0.3)
            
        else:
            # 2D plotting (X-Y plane)
            # Plot trajectory line
            self.ax.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=1, label='Trajectory')
            
            # Plot points with confidence colors
            self.ax.scatter(x_coords, y_coords, c=colors, s=50, alpha=0.8, label='Detections')
            
            # Highlight most recent point
            if len(self.positions) > 0:
                latest = self.positions[-1]
                self.ax.scatter([latest[0]], [latest[1]], 
                              c='yellow', s=100, marker='*', edgecolors='black', linewidth=2, 
                              label='Latest', zorder=5)
            
            # Set labels and title for 2D
            self.ax.set_xlabel('X Position (m)')
            self.ax.set_ylabel('Y Position (m)')
            self.ax.set_title(f'Tank Detection 2D Trajectory (X-Y Plane)\n{len(self.positions)} points')
            
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
            
            # Add grid and legend for 2D
            self.ax.grid(True, alpha=0.3)
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add confidence color info as text
            conf_text = "Colors: Green (>70%), Orange (50-70%), Red (<50%)"
            self.ax.text(0.02, 0.98, conf_text, transform=self.ax.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
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
        self.last_valid_contour = None
        self.detection_confidence = 0
        self.plotter = PlotVisualizer(max_points=200, plot_3d=plot_3d)
        self.plotter.start_plotting()

        # Print current plotting mode
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

        self.lower_yellow1 = np.array([20, 100, 120])
        self.upper_yellow1 = np.array([35, 255, 255])
        self.lower_yellow2 = np.array([15, 80, 100])
        self.upper_yellow2 = np.array([45, 255, 255])

        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def depth_callback(self, depth):
        self.z_world = -depth.pose.pose.position.z

    def is_tank_like_contour(self, contour, relaxed_mode=False):
        """
        Enhanced contour validation to distinguish tank from wire
        Supports relaxed mode for partially occluded tanks
        """
        area = cv2.contourArea(contour)
        
        # Area filtering - more lenient in relaxed mode
        min_area = 800 if relaxed_mode else 1500
        max_area = 50000
        if area < min_area or area > max_area:
            return False, 0
        
        # Bounding rectangle analysis
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # More lenient aspect ratio for occluded tanks
        min_ar = 0.2 if relaxed_mode else 0.3
        max_ar = 4.0 if relaxed_mode else 3.0
        if aspect_ratio < min_ar or aspect_ratio > max_ar:
            return False, 0
        
        # Convex hull and solidity check
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return False, 0
        
        solidity = float(area) / hull_area
        min_solidity = 0.5 if relaxed_mode else 0.75  # Much more lenient for occluded tanks
        if solidity < min_solidity:
            return False, 0
        
        # Perimeter to area ratio - more lenient for broken contours
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False, 0
        
        compactness = (4 * math.pi * area) / (perimeter * perimeter)
        min_compactness = 0.1 if relaxed_mode else 0.2
        if compactness < min_compactness:
            return False, 0
        
        # Extent - more lenient for irregular shapes
        rect_area = w * h
        extent = float(area) / rect_area
        min_extent = 0.25 if relaxed_mode else 0.4
        if extent < min_extent:
            return False, 0
        
        # Calculate confidence score
        confidence = 0
        
        # Aspect ratio score
        ar_score = 1.0 - abs(1.0 - aspect_ratio) / 3.0
        confidence += max(0, ar_score) * 25
        
        # Solidity score
        confidence += solidity * 20
        
        # Compactness score
        confidence += compactness * 20
        
        # Area score
        area_score = min(area / 3000.0, 1.0)
        confidence += area_score * 15
        
        # Bonus for being in relaxed mode but still meeting criteria
        if relaxed_mode:
            confidence += 20
        
        return True, confidence

    def find_best_tank_contour(self, contours):
        """
        Find the most tank-like contour from all candidates
        Uses two-pass approach: strict first, then relaxed for occluded tanks
        """
        candidates = []
        
        # First pass: strict criteria
        for contour in contours:
            is_valid, confidence = self.is_tank_like_contour(contour, relaxed_mode=False)
            if is_valid:
                candidates.append((contour, confidence, False))  # False = not relaxed
        
        # If no good candidates found, try relaxed mode
        if not candidates:
            for contour in contours:
                is_valid, confidence = self.is_tank_like_contour(contour, relaxed_mode=True)
                if is_valid:
                    candidates.append((contour, confidence, True))  # True = relaxed
        
        if not candidates:
            return None, 0, False
        
        # Sort by confidence and return the best one
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_contour, best_confidence, was_relaxed = candidates[0]
        
        # Lower threshold for relaxed mode
        threshold = 40 if was_relaxed else 50
        if best_confidence > threshold:
            return best_contour, best_confidence, was_relaxed
        
        return None, 0, False

    def merge_nearby_contours(self, contours, max_distance=50):
        """
        Merge contours that are close to each other (for fragmented tank detection)
        """
        if len(contours) < 2:
            return contours
        
        merged = []
        used = set()
        
        for i, contour1 in enumerate(contours):
            if i in used:
                continue
                
            # Get centroid of contour1
            M1 = cv2.moments(contour1)
            if M1["m00"] == 0:
                continue
                
            cx1 = int(M1["m10"] / M1["m00"])
            cy1 = int(M1["m01"] / M1["m00"])
            
            # Find nearby contours to merge
            to_merge = [contour1]
            used.add(i)
            
            for j, contour2 in enumerate(contours):
                if j in used or j <= i:
                    continue
                    
                M2 = cv2.moments(contour2)
                if M2["m00"] == 0:
                    continue
                    
                cx2 = int(M2["m10"] / M2["m00"])
                cy2 = int(M2["m01"] / M2["m00"])
                
                # Check distance between centroids
                distance = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                if distance < max_distance:
                    to_merge.append(contour2)
                    used.add(j)
            
            # Merge contours if we found nearby ones
            if len(to_merge) > 1:
                merged_contour = np.vstack(to_merge)
                merged.append(merged_contour)
            else:
                merged.append(contour1)
        
        return merged

    def combine_images(self, camera_image, plot_image):
        """Combine camera image and 3D plot side by side"""
        # Resize plot to match camera image height if needed
        cam_h, cam_w = camera_image.shape[:2]
        plot_h, plot_w = plot_image.shape[:2]
        
        # Resize plot to match camera height
        if plot_h != cam_h:
            aspect_ratio = plot_w / plot_h
            new_width = int(cam_h * aspect_ratio)
            plot_image = cv2.resize(plot_image, (new_width, cam_h))
        
        # Concatenate horizontally
        combined = np.hstack((camera_image, plot_image))
        return combined

    def image_callback(self, msg):
        image = bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = image.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Center pixel reference
        height, width, _ = frame.shape
        cx = int(width / 2)
        cy = int(height / 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), 3)

        # Create masks for yellow detection
        mask1 = cv2.inRange(hsv, self.lower_yellow1, self.upper_yellow1)
        mask2 = cv2.inRange(hsv, self.lower_yellow2, self.upper_yellow2)
        mask_yellow = cv2.bitwise_or(mask1, mask2)

        # Morphological operations
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, self.kernel_medium)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, self.kernel_large)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, self.kernel_small)
        mask_yellow = cv2.GaussianBlur(mask_yellow, (5, 5), 0)
        
        # Find contours
        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Try to merge nearby fragmented contours
        merged_contours = self.merge_nearby_contours(contours, max_distance=10)
        
        # Find best tank contour with relaxed mode support
        best_contour, confidence, was_relaxed = self.find_best_tank_contour(merged_contours)
        
        if best_contour is not None:
            self.last_valid_contour = best_contour
            self.detection_confidence = confidence
            
            # Calculate centroid
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Draw detection results
                detection_mode = "RELAXED" if was_relaxed else "STRICT"
                color = (0, 200, 255) if was_relaxed else (0, 255, 0)  # Orange for relaxed, green for strict
                
                cv2.circle(frame, (cX, cY), 8, color, -1)
                cv2.putText(frame, f"TANK {detection_mode} (conf: {confidence:.1f})", 
                           (cX - 80, cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.drawContours(frame, [best_contour], -1, color, 3)
                
                # Camera center reference
                cv2.circle(frame, (957, 533), 8, (255, 102, 255), -1)
                
                # POSE ESTIMATION (your existing code)
                inv_cam_matrix = np.linalg.inv(self.camera_matrix)
                centru = np.array([cX, cY, 1])
                centru = np.vstack(centru)
                
                C = np.array([0, 0, 0])
                C = np.vstack(C)
                
                M_int = np.dot(inv_cam_matrix, centru)
                d = M_int - C
                
                roll = 0
                pitch = 0
                yaw = 0
                
                R_roll = np.matrix([[1, 0, 0], [0, cos(roll), sin(roll)], [0, -sin(roll), cos(roll)]])
                R_pitch = np.matrix([[cos(pitch), 0, -sin(pitch)], [0, 1, 0], [sin(pitch), 0, cos(pitch)]])
                R_yaw = np.matrix([[cos(yaw), sin(yaw), 0], [-sin(yaw), cos(yaw), 0], [0, 0, 1]])
                R = R_yaw * R_pitch * R_roll
                
                inv_R = np.linalg.inv(R)
                
                u, v = cX, cY
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
                
                # print(f'Tank detected with confidence {confidence:.1f} ({"relaxed" if was_relaxed else "strict"} mode)')
                # print('ROV position: \n', world_coord)
                
                # Publish odometry
                global seq, last_point
                cov = [0.0] * 36
                cov[0] = 6
                cov[7] = 6
                odom = Odometry()
                odom.pose.covariance = cov
                odom.header.frame_id = "world"
                odom.header.stamp = msg.header.stamp
                odom.header.seq = seq
                seq += 1
                odom.child_frame_id = "yellow_patch"
                
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
                max_distance_jump = 0.5
                
                if seq == 1:
                    last_point = odom
                elif math.sqrt((x_current - x_last)**2 + (y_current - y_last)**2) > max_distance_jump:
                    odom.pose.pose.position.x = last_point.pose.pose.position.x
                    odom.pose.pose.position.y = last_point.pose.pose.position.y
                else:
                    last_point = odom
                
                real.publish(odom)

                # Add position to 3D plot
                if len(world_coord) >= 2:
                    z_pos = self.z_world if self.z_world is not None else 0
                    if seq == 1 or math.sqrt((x_current - x_last)**2 + (y_current - y_last)**2) < max_distance_jump:
                        self.plotter.add_position(odom.pose.pose.position.y, odom.pose.pose.position.x, z_pos, confidence)
                
                # Get plot image and combine with camera image
                plot_img = self.plotter.get_plot_image()
                combined_image = self.combine_images(frame, plot_img)
                
                # Publish combined image
                combined_bridge = bridge.cv2_to_imgmsg(combined_image, "bgr8")
                pub.publish(combined_bridge)
        else:
            # No valid tank detected
            print("No valid tank contour found")
            # Draw all contours for debugging
            cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
            cv2.putText(frame, "NO TANK DETECTED", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Still combine with plot for consistency
            plot_img = self.plotter.get_plot_image()
            combined_image = self.combine_images(frame, plot_img)
            
            # Publish debug image with plot
            debug_bridge = bridge.cv2_to_imgmsg(combined_image, "bgr8")
            pub.publish(debug_bridge)

    def cleanup(self):
        """Clean up resources"""
        self.plotter.stop()


pub = rospy.Publisher('colordetection', Image, queue_size=10)
world = rospy.Publisher('/BlueRov2/plane', Odometry, queue_size=10)
real = rospy.Publisher('/BlueRov2/real_coord', Odometry, queue_size=10)

if __name__ == '__main__':
    # CONFIGURATION: Set to True for 3D plotting, False for 2D plotting
    USE_3D_PLOT = False  # Change this to False for 2D plotting
    
    rospy.init_node('listener', anonymous=True)
    task = Task(plot_3d=USE_3D_PLOT)
    
    try:
        rospy.Subscriber('/gopro/image_raw', Image, task.image_callback)
        rospy.Subscriber('/BlueRov2/odom/depth', Odometry, task.depth_callback)
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        task.cleanup()