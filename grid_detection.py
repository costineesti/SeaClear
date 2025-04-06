#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os

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

def draw_coordinate_frame(img, x, y, length=40):
    # X-axis (red)
    cv2.arrowedLine(img, (x, y), (x + length, y), (0, 0, 255), 2, tipLength=0.3)
    # Y-axis (green)
    cv2.arrowedLine(img, (x, y), (x, y - length), (0, 255, 0), 2, tipLength=0.3)

"""
For coordinate frame
"""
def get_intersection_point(h_line, v_line):
    x1, y1, x2, y2 = h_line[0]
    x3, y3, x4, y4 = v_line[0]

    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    det = A1 * B2 - A2 * B1
    if det == 0:
        return None  # lines are parallel
    else:
        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det
        return int(x), int(y)

def index_squares(image, horizontal_lines, vertical_lines):
    index = 1

    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            x = vertical_lines[j]
            y = horizontal_lines[i]

            # Draw the index in blue at the top-left corner
            cv2.putText(image, str(index), 
                        (x + 3, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6,                              
                        (255, 0, 0),                      
                        1, 
                        cv2.LINE_AA)
            index += 1

    return index-1

def simplify_lines(lines, axis='horizontal', threshold=5):
    """
    Groups nearby lines (by Y for horizontal, by X for vertical) and averages them.
    Returns simplified line list.
    """
    coords = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        coord = (y1 + y2) / 2 if axis == 'horizontal' else (x1 + x2) / 2
        coords.append(coord)
    
    coords = sorted(coords)
    grouped = []
    group = []

    for c in coords:
        if not group or abs(c - group[-1]) < threshold:
            group.append(c)
        else:
            grouped.append(int(np.mean(group)))
            group = [c]
    if group:
        grouped.append(int(np.mean(group)))

    return grouped

def sort_lsd_lines(lsd_lines, angle_threshold=10):
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


                                                     # MAIN #

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
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=2)

        cv2.imwrite('canny.jpg', edges)

                                                # POSTPROCESSING #

        lsd = cv2.createLineSegmentDetector(0)
        lsd_lines = lsd.detect(edges)[0]
        # FILTER NOISE
        min_length = 30
        filtered_lines = []

        for line in lsd_lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length > min_length:
                filtered_lines.append(line)
        
        horizontal_lines, vertical_lines = sort_lsd_lines(filtered_lines)

        horizontal_lines_sorted = sorted(horizontal_lines, key=lambda l: (l[0][1] + l[0][3]) / 2)
        vertical_lines_sorted = sorted(vertical_lines, key=lambda l: (l[0][0] + l[0][2]) / 2)

        coord_frame_x, coord_frame_y = get_intersection_point(horizontal_lines_sorted[-1], vertical_lines_sorted[-1])
        horizontal_y = simplify_lines(horizontal_lines, axis='horizontal', threshold=10)
        vertical_x = simplify_lines(vertical_lines, axis='vertical', threshold=10)

        postprocessing_image = np.copy(roi_frame)
        squares_nbr = index_squares(postprocessing_image, horizontal_y, vertical_x)
        draw_coordinate_frame(postprocessing_image, coord_frame_x, coord_frame_y)
        drawn = lsd.drawSegments(postprocessing_image, np.array(filtered_lines))
        cv2.imwrite('lsd.jpg', drawn)

        # publish
        # image_pub.publish(bridge.cv2_to_imgmsg(bw, encoding="mono8"))
        rate.sleep()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()