#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os

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
    roi_y_min = 20
    roi_x_max = 1380
    roi_y_max = 1080
    height, width = frame.shape[:2]
    x_min = max(0, roi_x_min)
    y_min = max(0, roi_y_min)
    x_max = min(width, roi_x_max)
    y_max = min(height, roi_y_max)

    roi = frame[y_min:y_max, x_min:x_max]

    return roi, x_min, y_min

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

        roi_frame, x_min, y_min = rescaleFrame(frame)
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        edges = cv2.Canny(blur, 110, 150, apertureSize=3)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        kernel = np.ones((5,5), np.uint8)
        edges = cv2.erode(edges, kernel, iterations=1)
        cv2.imwrite('canny.jpg', edges)

        lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=240)
        # lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=240, minLineLength=100, maxLineGap=10)
        filter = True

        if filter:
            rho_threshold = 15
            theta_threshold = 0.1

            # How many lines are similar to a given one
            similar_lines = {i : [] for i in range(len(lines))}
            for i in range(len(lines)):
                for j in range(len(lines)):
                    if i == j:
                        continue
                    
                    rho_i, theta_i = lines[i][0]
                    rho_j, theta_j = lines[j][0]
                    if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                        similar_lines[i].append(j)

            # ordering the INDECES of the lines by how many are similar to them.
            indices = [i for i in range(len(lines))]
            indices.sort(key=lambda x : len(similar_lines[x]))

            # line flags is the base for the filtering
            line_flags = len(lines)*[True]
            for i in range(len(lines) - 1):
                if not line_flags[indices[i]]:# if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
                    continue

                for j in range(i + 1, len(lines)): # we are only considering those elements that had less similar line
                    if not line_flags[indices[j]]: # and only if we have not disregarded them already
                        continue

                    rho_i,theta_i = lines[indices[i]][0]
                    rho_j,theta_j = lines[indices[j]][0]
                    if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                        line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now

        print('number of Hough lines:', len(lines))

        filtered_lines = []
        if filter:
            for i in range(len(lines)): # filtering
                if line_flags[i]:
                    filtered_lines.append(lines[i])

            print('Number of filtered lines:', len(filtered_lines))
        else:
            filtered_lines = lines

        for line in filtered_lines:
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(frame,(x1+x_min,y1+y_min),(x2+x_min,y2+y_min),(0,0,255),2)

        cv2.imwrite('hough.jpg',frame)
            # line flags

        # image_pub.publish(bridge.cv2_to_imgmsg(bw, encoding="mono8"))
        rate.sleep()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()