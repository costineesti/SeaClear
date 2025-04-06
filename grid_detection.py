#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os
from collections import deque

def angle_of_line(line):
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    angle = np.arctan2(dy, dx) * 180.0 / np.pi
    return angle

def rescaleFrame(frame):
    #rescale image
    # CONSTANTS FOUND EMPIRICALLY
    roi_x_min = 630
    roi_y_min = 18
    roi_x_max = 1380
    roi_y_max = 1080
    height, width = frame.shape[:2]
    x_min = max(0, roi_x_min)
    y_min = max(0, roi_y_min)
    x_max = min(width, roi_x_max)
    y_max = min(height, roi_y_max)

    roi = frame[y_min:y_max, x_min:x_max]

    return roi, x_min, y_min, x_max, y_max

def mergeNearestLines(lines, threshold=50):
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1-y2) < 50: # Horizontal line
            horizontal_lines.append(y1)
        elif abs(x1-x2) < 10: # Vertical line
            vertical_lines.append(x1)
        
    horizontal_lines = sorted(set(horizontal_lines)) # Build an ordered collection of unique elements
    vertical_lines = sorted(set(vertical_lines))

    def mergeLines(line_positions, threshold):
        merged_lines = []
        current_line = line_positions[0]

        for line in line_positions[1:]:
            if line - current_line <= threshold: # Filter
                continue
            else:
                merged_lines.append(current_line)
                current_line = line

        merged_lines.append(current_line)
        return merged_lines
    
    merged_horizontal_lines = mergeLines(horizontal_lines, threshold)
    merged_vertical_lines = mergeLines(vertical_lines, threshold)

    return merged_horizontal_lines, merged_vertical_lines

# To avoid always finding the lines from scratch.
def average_lines(line_history, history_length=5):
    if not line_history:
        return []
    
    # Pad with last entry if history is shorter than expected
    if len(line_history) < history_length:
        line_history = list(line_history)
        last = line_history[-1]
        for _ in range(history_length - len(line_history)):
            line_history.append(last)

    # Transpose to group same-line indices
    grouped = list(zip(*line_history))
    avg_lines = [int(np.mean(group)) for group in grouped]
    return avg_lines

def main():
    rospy.init_node('grid_detection_roi', anonymous=True)
    bridge = CvBridge()

    video_path = os.path.expanduser('~/Videos/GX010179.MP4')
    cap = cv2.VideoCapture(video_path)
    image_pub = rospy.Publisher('/grid_detection/image_raw', Image, queue_size=10)

    if not cap.isOpened():
        rospy.logerr(f"Cannot open video file at {video_path}")
        return

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            rospy.loginfo("End of video file reached")
            break

                                                # PREPROCESSING #

        roi_frame, x_min, y_min, x_max, y_max = rescaleFrame(frame)
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=2)

        cv2.imwrite('canny.jpg', edges)

                                                # POSTPROCESSING #

        #lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=240)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=240, minLineLength=150, maxLineGap=8)
        horizontal, vertical = mergeNearestLines(lines)
        max_history = 5
        horizontal_history = deque(maxlen=max_history)
        vertical_history = deque(maxlen=max_history)
        horizontal_history.append(horizontal)
        vertical_history.append(vertical)

        stable_horizontal = average_lines(horizontal_history)
        stable_vertical = average_lines(vertical_history)

        img_with_merged_lines = np.copy(frame)
        for y in stable_horizontal:
            cv2.line(img_with_merged_lines, pt1=(x_min, y+y_min), pt2=(x_max, y+y_min), color=(0,255,0), thickness=2)
        for x in stable_vertical:
            cv2.line(img_with_merged_lines, pt1=(x+x_min, y_min), pt2=(x+x_min, y_max), color=(0,255,0), thickness=2)
        cv2.imwrite('hough.jpg',img_with_merged_lines)
            # line flags

        # image_pub.publish(bridge.cv2_to_imgmsg(bw, encoding="mono8"))
        rate.sleep()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()