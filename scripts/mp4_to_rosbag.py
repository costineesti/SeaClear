import rosbag, rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Float64
import cv2
import time
import os
import re, subprocess
from datetime import datetime

bridge = CvBridge()
video_path = os.path.expanduser('~/Desktop/GX010182.MP4')
bag_path = os.path.expanduser('~/Desktop/GoPro.bag')

def get_video_metadata(video_path):
    output = subprocess.check_output(
                                    ["../gpmf-parser/demo/gpmfdemo", video_path],
                                    stderr=subprocess.STDOUT
                                    ).decode("utf-8", errors="replace")

    fps_match = re.search(r"VIDEO FRAMERATE:\s+([\d.]+) with (\d+) frames", output)
    if fps_match:
        fps = float(fps_match.group(1))
        frame_count = int(fps_match.group(2))
        return fps, frame_count
    else:
        raise RuntimeError("Couldn't parse FPS and frame count from gpmfdemo output.")

def get_stmp_gpsu_pairs(video_path):

    output = subprocess.check_output(
        ["../gpmf-parser/demo/extract_utc", video_path],
        stderr=subprocess.STDOUT
    ).decode("utf-8", errors="replace")

    stmp = re.findall(r"STMP microsecond timestamp: ([\d.]+) s", output)
    gpsu = re.findall(r"GPSU UTC Timedata:\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)", output)

    if len(stmp) != len(gpsu):
        raise RuntimeError("Mismatch between STMP and GPSU entries")

    stmp_gpsu = []
    for s, g in zip(stmp, gpsu):
        t = float(s)
        dt = datetime.strptime(g, "%Y-%m-%d %H:%M:%S.%f")
        unix = dt.timestamp()
        stmp_gpsu.append((t, unix))
    
    return stmp_gpsu

def interpolate_frame_timestamps(frame_count, fps, stmp_gpsu):
    timestamps = [0.0] * frame_count
    frames_per_segment = int(round(fps))

    for i in range(len(stmp_gpsu) - 1):
        stmp0, gpsu0 = stmp_gpsu[i]
        stmp1, gpsu1 = stmp_gpsu[i + 1]
        idx0 = i * frames_per_segment
        idx1 = min((i + 1) * frames_per_segment, frame_count)

        for f in range(idx0, idx1):
            alpha = (f - idx0) / (idx1 - idx0)
            interpolated = stmp0 + alpha * (stmp1 - stmp0)
            timestamps[f] = gpsu0 + (interpolated - stmp0)

    # Handle any leftover frames
    last = (len(stmp_gpsu) - 1) * frames_per_segment
    if last < frame_count:
        dt = 1.0 / fps
        for f in range(last, frame_count):
            timestamps[f] = timestamps[last - 1] + (f - last + 1) * dt

    return timestamps
    

def video_to_rosbag(video_path, bag_path, frame_count, timestamps):
    cap = cv2.VideoCapture(video_path)
    bag = rosbag.Bag(bag_path, 'w')

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"Stopped at frame {i}")
            break

        timestamp = rospy.Time.from_sec(timestamps[i])
        print(f"Writing frame {i} at time {timestamp.to_sec()}")
        header = Header()
        header.stamp = timestamp
        header.frame_id = "gopro"
        img_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img_msg.header = header

        bag.write('/gopro/image_raw', img_msg, t=timestamp)
        # bag.write('/camera_wall_time', Float64(timestamp.to_sec()), timestamp)

    cap.release()
    bag.close()
    print(i)

if __name__ == "__main__":
    fps, frame_count = get_video_metadata(video_path)
    stmp_gpsu = get_stmp_gpsu_pairs(video_path)
    timestamps = interpolate_frame_timestamps(frame_count, fps, stmp_gpsu)
    video_to_rosbag(video_path, bag_path, frame_count, timestamps)