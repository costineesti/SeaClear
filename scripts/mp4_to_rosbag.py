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
    print(output)
    fps_match = re.search(r"VIDEO FRAMERATE:\s+([\d.]+) with (\d+) frames", output)
    if fps_match:
        fps = float(fps_match.group(1))
        frame_count = int(fps_match.group(2))
        return fps, frame_count
    else:
        raise RuntimeError("Couldn't parse FPS and frame count from gpmfdemo output.")

def get_creation_time_unix(video_path):
    output = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        stderr=subprocess.STDOUT
    ).decode("utf-8", errors="replace").strip()
    
    creation_dt = datetime.strptime(output, "%Y-%m-%dT%H:%M:%S.%fZ")
    return creation_dt.timestamp()

def get_stmp_list(video_path):
    output = subprocess.check_output(
        ["../gpmf-parser/demo/extract_utc", video_path],
        stderr=subprocess.STDOUT
    ).decode("utf-8", errors="replace")
    print(output)
    stmp = re.findall(r"STMP microsecond timestamp: ([\d.]+) s", output)
    return [float(s) for s in stmp]

def interpolate_frame_timestamps(frame_count, fps, creation_unix, stmp_list):
    timestamps = [0.0] * frame_count
    frames_per_segment = int(round(fps))

    for i in range(len(stmp_list) - 1):
        stmp0 = stmp_list[i]
        stmp1 = stmp_list[i + 1]
        idx0 = i * frames_per_segment
        idx1 = min((i + 1) * frames_per_segment, frame_count)

        for f in range(idx0, idx1):
            alpha = (f - idx0) / (idx1 - idx0)
            interpolated = stmp0 + alpha * (stmp1 - stmp0)
            timestamps[f] = creation_unix + interpolated

    # Handle tail
    last = (len(stmp_list) - 1) * frames_per_segment
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

        bag.write('/gopro/image_raw', img_msg, t=timestamp) # Image
        wall_time_msg = Float64()
        wall_time_msg.data = timestamps[i]
        bag.write('/gopro/wall_time', wall_time_msg, t=timestamp) # wall time for sync

    cap.release()
    bag.close()
    print(i)

if __name__ == "__main__":
    fps, frame_count = get_video_metadata(video_path)
    creation_unix = get_creation_time_unix(video_path)
    stmp_list = get_stmp_list(video_path)
    timestamps = interpolate_frame_timestamps(frame_count, fps, creation_unix, stmp_list)
    video_to_rosbag(video_path, bag_path, frame_count, timestamps)