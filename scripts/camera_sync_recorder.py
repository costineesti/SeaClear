#!/usr/bin/env python3

"""
ROS Node that:
1. records the stream from the camera (pointed at a real-time unix clock on the monitor)
2. computes the timestamp for each frame using v4l2
3. attaches the two in a rosbag for further comparison
"""

import os
import fcntl
import v4l2
from struct import pack
import time
import rospy
import rosbag
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
import csv

rosbag_path = os.path.expanduser("~/camera_sync_uptime.bag")
csv_path = "csv_timestamps"

class CameraSyncRecorder:

    def __init__(self):
        rospy.init_node('camera_sync_recorder')
        
        # Class parameters
        self.offset = time.time() - time.monotonic() # Computed once at start.
        self.bag = rosbag.Bag(rosbag_path, 'w')
        self.video_device = "/dev/video5"

        # Open CSV file inside the class
        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["system_time", "camera_wall_time", "diff"])  # CSV header

        # Open metadata to record timestamps and setup the buffers once.
        self.stream = os.open(self.video_device, os.O_RDWR)
        self.setup_metadata_buffers()

        # Subscribe to the camera topic
        rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)

        self.frame_count = 0

        rospy.loginfo("Recording video stream with synchronized timestamps...")


    """
    source: https://costinchitic.co/notes/UVC-Video-Stream
    """
    def setup_metadata_buffers(self):
        req = v4l2.v4l2_requestbuffers()
        req.count = 4
        req.type = v4l2.V4L2_BUF_TYPE_META_CAPTURE
        req.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(self.stream, v4l2.VIDIOC_REQBUFS, req)

        self.buf = v4l2.v4l2_buffer()
        self.buf.type = v4l2.V4L2_BUF_TYPE_META_CAPTURE
        self.buf.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(self.stream, v4l2.VIDIOC_QUERYBUF, self.buf)
        fcntl.ioctl(self.stream, v4l2.VIDIOC_QBUF, self.buf)

        buf_type = pack("i", v4l2.V4L2_BUF_TYPE_META_CAPTURE)
        fcntl.ioctl(self.stream, v4l2.VIDIOC_STREAMON, buf_type)
    

    """
    Here I extract and convert the camera timestamp to wall time.
    """
    def get_camera_walltime(self):
        fcntl.ioctl(self.stream, v4l2.VIDIOC_DQBUF, self.buf) # dequeue the buffer to extract info.

        sequence = self.buf.sequence
        timestamp_sec = self.buf.timestamp.secs # same as v4l2, the timeval class
        timestamp_usec = self.buf.timestamp.usecs
        camera_timestamp = timestamp_sec + (timestamp_usec / 1_000_000) 
        
        camera_wall_time = camera_timestamp + self.offset # This is what we want. We compare it to the time on the monitor.
        curr_sys_time = time.time()

        fcntl.ioctl(self.stream, v4l2.VIDIOC_QBUF, self.buf)
        
        return sequence, camera_wall_time, curr_sys_time
    
    """
    Process incoming video frames and attach corresponding (sequence,timestamp)
    """
    def image_callback(self, msg):
        seq, camera_wall_time, curr_time = self.get_camera_walltime()
        # Write to CSV
        self.csv_writer.writerow([curr_time, camera_wall_time, abs(curr_time-camera_wall_time)])
        self.csv_file.flush()  # Ensure data is written immediately
        rospy.loginfo(f"seq {seq}, ts {camera_wall_time}")
        # Save to ROS bag
        self.bag.write('/camera/image_compressed', msg, rospy.Time.from_sec(camera_wall_time))
        self.bag.write('/camera_wall_time', Float64(camera_wall_time), rospy.Time.from_sec(camera_wall_time))

        self.frame_count += 1

        # Stop recording after 20 frames
        # if self.frame_count >= 40:
        #     rospy.loginfo("Reached 20 sequences, shutting down...")
        #     self.shutdown()
        #     rospy.signal_shutdown("Recording complete")

    """ Clean up on shutdown. """
    def shutdown(self):
        fcntl.ioctl(self.stream, v4l2.VIDIOC_STREAMOFF, pack("i", v4l2.V4L2_BUF_TYPE_META_CAPTURE))
        os.close(self.stream)
        self.bag.close()
        self.csv_file.close()  # Close the CSV file properly
        rospy.loginfo(f"Saved recorded data to {rosbag_path} and {csv_path}")

############################################################ MAIN #######################################################################
if __name__ == "__main__":
    try:
        recorder = CameraSyncRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        recorder.shutdown()