#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os
import math

class GridOverlayNode:
    def __init__(self):
        rospy.init_node('grid_detection_roi', anonymous=True)
        self.bridge = CvBridge()
        self.debug = True
        self.referencePoint = None # click
        self.generalWidth = None # computed
        self.generalHeight = None # computed
        self.GETPARAMS = True # To only get (width,height) once!
        self.cols = 12 # SET from real life
        self.rows = 29 # SET from real life
        self.angle = 0 # Grid Rotation
        self.video_path = os.path.expanduser('~/Documents/SeaClear/June_BlueROV_Experiments/23Jun_GoPro.MP4')
        self.image_pub = rospy.Publisher('/grid_detection/image_raw', Image, queue_size=10)
        self.rate = rospy.Rate(10)  # 10 Hz

    def select_point(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.referencePoint = (int(x*2), int(y*2)) # Multiplied by 2 because the reference image is shrinked to half it's original size.

    def rescaleFrame(self, frame):
        roi_x_min, roi_y_min, roi_x_max, roi_y_max = 630, 500, 1180, 900 # Empirically. Image from GoPro. IF GRID LOOKS WEIRD, CHANGE THESE VALUES!
        height, width = frame.shape[:2]
        x_min, y_min = max(0, roi_x_min), max(0, roi_y_min)
        x_max, y_max = min(width, roi_x_max), min(height, roi_y_max)
        roi = frame[y_min:y_max, x_min:x_max]
        return roi
    
    def preprocess(self, frame):
        """
        Preprocessing the ROI to get the best possible grid lines.
        Empyrical approach.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=2)
        return edges
    
    def sort_lsd_lines(self, lsd_lines, angle_threshold=10):
        """
        Take the filtered lines and group them into horizontal and vertical.
        Will futher compute (width, height) based on them.
        """
        horizontal_lines=[]
        vertical_lines=[]
        for line in lsd_lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Normalize angle to [-90, 90]
            angle = (angle + 180) % 180
            if angle > 90:
                angle -= 180
            if abs(angle) < angle_threshold:
                horizontal_lines.append(line)
            elif abs(angle - 90) < angle_threshold:
                vertical_lines.append(line)
        
        return horizontal_lines, vertical_lines

    def computeGridLines(self, lsd_lines):
        """
        Take horizontal and vertical lines from the ROI with minimal length and group them.
        """
        min_length = 40 # FILTER NOISE. Only get lines long enough!
        filtered_lines = []
        for line in lsd_lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length > min_length:
                filtered_lines.append(line)
        horizontal_lines, vertical_lines = self.sort_lsd_lines(filtered_lines)
        return horizontal_lines, vertical_lines

    def getGridSquareParameters(self, horizontal, vertical, iterations=10):
        """
        Iterate through ROI to get the most accurate (width,height)[px] possible.
        """
        width_list, height_list = [], []
        for _ in range(iterations):
            _, width = self.simplify_lines(horizontal, 'horizontal')
            _, height = self.simplify_lines(vertical, 'vertical')
            width_list.append(width)
            height_list.append(height)
        return np.mean(width_list), np.mean(height_list)
    
    def simplify_lines(self, lines, axis='horizontal', threshold=10):
        """
        Groups nearby lines (by Y for horizontal, by X for vertical) and averages them.
        Returns simplified line list.
        """
        coords = [(y1 + y2) / 2 if axis == 'horizontal' else (x1 + x2) / 2 for x1, y1, x2, y2 in [line[0] for line in lines]]
        coords = sorted(coords)
        grouped, group = [], []
        for c in coords:
            if not group or abs(c - group[-1]) < threshold:
                group.append(c)
            else:
                grouped.append(int(np.mean(group)))
                group = [c]
        if group:
            grouped.append(int(np.mean(group)))
        diffs = np.diff(np.array(grouped))
        average = np.mean(diffs)
        return grouped, average
    
    """ Not used anymore. Good reference though.
    def draw_angled_rec(self, center, width, height, angle, img):
        Source: https://richardpricejones.medium.com/drawing-a-rectangle-with-a-angle-using-opencv-c9284eae3380
        x0, y0 = center[0], center[1]
        _angle = math.radians(angle)  # better to directly convert to radians
        b = math.cos(_angle) * 0.5
        a = math.sin(_angle) * 0.5
        pt0 = (x0 - a * height - b * width, y0 + b * height - a * width)
        pt1 = (x0 + a * height - b * width, y0 - b * height - a * width)
        pt2 = (2 * x0 - pt0[0], 2 * y0 - pt0[1])
        pt3 = (2 * x0 - pt1[0], 2 * y0 - pt1[1])
        pt0 = (int(round(pt0[0])), int(round(pt0[1])))
        pt1 = (int(round(pt1[0])), int(round(pt1[1])))
        pt2 = (int(round(pt2[0])), int(round(pt2[1])))
        pt3 = (int(round(pt3[0])), int(round(pt3[1])))
        cv2.line(img, pt0, pt1, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(img, pt1, pt2, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(img, pt2, pt3, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(img, pt3, pt0, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    """

    def overlayGrid(self, width, height, referencePoint, img, angle=0):
        """
        Overlay a grid of 12 x 29 on an image starting with the reference point.
        228 squares in total. The amount the grid has in real life.
        """
        cols, rows = self.cols, self.rows
        # Determine grid direction (Click only on top left or bottom right!)
        img_w, img_h = img.shape[:2]
        start_from_top = referencePoint[1] < img_h // 2
        start_from_left = referencePoint[0] < img_w // 2
        if start_from_left:
            x_sign = 1  # move right
        else:
            x_sign = -1  # move left
        if start_from_top:
            y_sign = 1  # move down
        else:
            y_sign = -1  # move up
        # Generate local grid points
        x_coords = x_sign * np.arange(cols + 1) * width
        y_coords = y_sign * np.arange(rows + 1) * height
        xv, yv = np.meshgrid(x_coords, y_coords)  # full grid
        grid_points = np.stack([xv.flatten(), yv.flatten()], axis=1)  # (N, 2)
        angle_rad = math.radians(angle)
        rotation_matrix = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad),  math.cos(angle_rad)]
        ])
        # Apply rotation and translation
        rotated_points = grid_points @ rotation_matrix.T  # (N, 2)
        rotated_points[:, 0] += referencePoint[0] # X
        rotated_points[:, 1] += referencePoint[1] # Y
        transformed_points = np.round(rotated_points).astype(int) # cv2.line takes int as argument.
        # Draw horizontal lines
        for r in range(rows + 1):
            for c in range(cols):
                idx1 = r * (cols + 1) + c
                idx2 = idx1 + 1
                pt1 = tuple(transformed_points[idx1])
                pt2 = tuple(transformed_points[idx2])
                cv2.line(img, pt1, pt2, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        # Draw vertical lines
        for c in range(cols + 1):
            for r in range(rows):
                idx1 = r * (cols + 1) + c
                idx2 = idx1 + (cols + 1)
                pt1 = tuple(transformed_points[idx1])
                pt2 = tuple(transformed_points[idx2])
                cv2.line(img, pt1, pt2, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    def choose_gridAngle(self, frame):
        """
        Allow the user to adjust the grid angle on a fixed frame before starting real processing.
        """
        while True:
            display_frame = np.copy(frame)
            self.overlayGrid(self.generalHeight, self.generalWidth, self.referencePoint, display_frame, angle=self.angle)
            resized_display = cv2.resize(display_frame, None, fx=0.5, fy=0.5)
            cv2.imshow("Adjust Grid Angle (Press A/D to rotate, Enter to confirm)", resized_display)
            key = cv2.waitKey(50) & 0xFF
            if key == ord('a') or key == 81:  # 'a' key or Left Arrow.
                self.angle -= 0.1
            elif key == ord('d') or key == 83:  # 'd' key or Right Arrow.
                self.angle += 0.1
            elif key == 13:  # Enter key
                rospy.loginfo(f"Final grid angle selected: {self.angle} degrees.")
                break
            elif key == ord('q'):  # Emergency exit
                rospy.logwarn("Canceled during grid adjustment.")
                return
        cv2.destroyAllWindows()
        return display_frame
    
    def choose_referencePoint(self):
        """
        Allow the user to choose a global reference point of the virtualized grid.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            rospy.logerr(f"Cannot open video file at {self.video_path}")
            return
        ret, frame = cap.read()
        if not ret:
            rospy.logerr("Failed to read the first frame of the video.")
            return
        cv2.namedWindow("Select Reference Point")
        cv2.setMouseCallback("Select Reference Point", self.select_point)
        while self.referencePoint is None:
            display_frame = cv2.resize(frame, None, fx=0.5, fy=0.5) # Half size for better visualization.
            cv2.imshow("Select Reference Point", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        if self.referencePoint is None:
            rospy.logerr("No reference point selected.")
            return
        rospy.loginfo(f"Using reference point: {self.referencePoint}")
        cap.release()

    def run(self):
        """RUN METHOD"""
        self.choose_referencePoint()
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            rospy.logerr(f"Cannot reopen video file at {self.video_path}")
            return
        
        while not rospy.is_shutdown() and cap.isOpened(): # Main loop
            ret, frame = cap.read()
            if not ret:
                rospy.loginfo("End of video file reached")
                break
                                                            # PREPROCESSING #
            roi_frame = self.rescaleFrame(frame)
            cv2.imshow("ROI Frame", roi_frame) # Show the ROI frame for debugging
            edges = self.preprocess(roi_frame)
                                                            # POSTPROCESSING #
            lsd = cv2.createLineSegmentDetector(0) # Source: https://costinchitic.wiki/notes/Line-Segment-Detector
            # Use cv2.ximgproc.createFastLineDetector() for OpenCV newer than 4.1.0!
            lsd_lines = lsd.detect(edges)[0] # Mainly used to get generalWidth and generalHeight. Reliable method.
            horizontal_lines, vertical_lines = self.computeGridLines(lsd_lines)

            if self.GETPARAMS: # Only execute once
                self.generalWidth, self.generalHeight = self.getGridSquareParameters(horizontal_lines, vertical_lines)
                rospy.loginfo(f"General Grid (width, height): {int(self.generalWidth), int(self.generalHeight)}")
                postprocessing_image = self.choose_gridAngle(frame)
                self.GETPARAMS = False

            if self.debug:
                display_frame = np.copy(frame)
                self.overlayGrid(self.generalHeight, 
                                 self.generalWidth, 
                                 self.referencePoint, 
                                 display_frame, 
                                 angle=self.angle)
                cv2.imwrite('final.jpg', display_frame)
                
            self.rate.sleep()

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    node = GridOverlayNode()
    node.run()