#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os

referencePoint = None
# Mouse callback function
def select_point(event, x, y, flags, param):
    global referencePoint
    if event == cv2.EVENT_LBUTTONDOWN:
        referencePoint = (int(x*2), int(y*2))
        print(f"Selected reference point: {referencePoint}")

def computeGridLines(lsd_lines):
    # FILTER NOISE
    min_length = 40
    filtered_lines = []

    for line in lsd_lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length > min_length:
            filtered_lines.append(line)
    
    horizontal_lines, vertical_lines = sort_lsd_lines(filtered_lines)

    return horizontal_lines, vertical_lines, filtered_lines

def getGridSquareParameters(horizontal, vertical):
    _, width = simplify_lines(horizontal, 
                              axis='horizontal', 
                              threshold=10)
    
    _, length = simplify_lines(vertical, 
                               axis='vertical', 
                               threshold=10)
    
    return width, length

def rescaleFrame(frame):
    #rescale image
    # CONSTANTS FOUND EMPIRICALLY
    roi_x_min = 630
    roi_y_min = 500
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


def simplify_lines(lines, axis='horizontal', threshold=5):
    """
    Groups nearby lines (by Y for horizontal, by X for vertical) and averages them.
    Returns simplified line list.
    """
    coords = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if axis == 'horizontal':
            coord = (y1 + y2) / 2
        else:
            coord = (x1 + x2) / 2
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

    diffs = np.diff(np.array(grouped))
    average = np.mean(diffs)

    print(f'average: {average}, axis: {axis}')
    return grouped, average

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

    if not cap.isOpened():
        rospy.logerr(f"Cannot open video file at {video_path}")
        return

    # Grab first frame for point selection
    ret, frame = cap.read()
    if not ret:
        rospy.logerr("Failed to read the first frame of the video.")
        return

    # Show frame and wait for click
    cv2.namedWindow("Select Reference Point")
    cv2.setMouseCallback("Select Reference Point", select_point)

    while referencePoint is None:
        display_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        cv2.imshow("Select Reference Point", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    if referencePoint is None:
        rospy.logerr("No reference point selected.")
        return

    rospy.loginfo(f"Using reference point: {referencePoint}")

    # Re-open video to reset to first frame
    cap.release()
    cap = cv2.VideoCapture(video_path)

    rate = rospy.Rate(10)  # 10 Hz
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
        
        horizontal_lines, vertical_lines, filtered_lines = computeGridLines(lsd_lines)
        generalWidth, generalLength = getGridSquareParameters(horizontal_lines, vertical_lines)
        postprocessing_image = np.copy(roi_frame)

        # Get REFERENCE POINT for the generated grid.

        drawn = lsd.drawSegments(postprocessing_image, np.array(filtered_lines))
        cv2.imwrite('lsd.jpg', drawn)

        # publish
        # image_pub.publish(bridge.cv2_to_imgmsg(bw, encoding="mono8"))
        rate.sleep()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()