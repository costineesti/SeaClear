#!/usr/bin/env python3
import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge
import os

bag_path = "/home/costin/June_BlueROV_Experiments/usbcamera_13Jun.bag"
output_dir = "/home/costin/frames/"
image_topic = "/camera/image_compressed"
frame_skip = 5  # save every 5th frame

os.makedirs(output_dir, exist_ok=True)
bag = rosbag.Bag(bag_path, "r")
bridge = CvBridge()

count = 0
saved = 0

for topic, msg, t in bag.read_messages(topics=[image_topic]):
    if count % frame_skip == 0:
        image = bridge.imgmsg_to_cv2(msg, "bgr8")
        filename = os.path.join(output_dir, f"frame_usbcam_13Jun{saved:05d}.jpg")
        cv2.imwrite(filename, image)
        saved += 1
    count += 1

bag.close()
print(f"Saved {saved} images in {output_dir}")