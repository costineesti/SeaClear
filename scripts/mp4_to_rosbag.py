import rosbag, rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Float64
import cv2
import time
import os
import re, subprocess
from datetime import datetime
import numpy as np

bridge = CvBridge()
video_path = os.path.expanduser('~/Desktop/GX010182.MP4')
bag_path = os.path.expanduser('~/Desktop/GoPro.bag')

def get_video_metadata(video_path):
    # Get duration and frame count precisely
    duration_cmd = ["ffprobe", "-i", video_path, "-show_entries", "format=duration", 
                   "-v", "quiet", "-of", "csv=p=0"]
    duration = float(subprocess.check_output(duration_cmd).decode().strip())
    # Get frame count
    frame_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
                "-count_packets", "-show_entries", "stream=nb_read_packets", 
                "-of", "csv=p=0", video_path]
    frame_count = int(subprocess.check_output(frame_cmd).decode().strip())
    # Calculate actual average FPS
    avg_fps = frame_count / duration
    
    return avg_fps, frame_count, duration

def get_creation_time_unix(video_path):
    """Get video creation time as UNIX timestamp"""
    output = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        stderr=subprocess.STDOUT
    ).decode("utf-8", errors="replace").strip()
    
    creation_dt = datetime.strptime(output, "%Y-%m-%dT%H:%M:%S.%fZ")
    return creation_dt.timestamp()

def parse_stmp_data(video_path):
    """Parse STMP timestamps from extract_utc output"""
    output = subprocess.check_output(
        ["../gpmf-parser/demo/extract_utc", video_path],
        stderr=subprocess.STDOUT
    ).decode("utf-8", errors="replace")
    
    # Extract STMP BE (big-endian) timestamps - these are more reliable
    stmp_matches = re.findall(r"STMP\[0\] BE: ([\d.]+) s", output)
    stmp_times = [float(s) for s in stmp_matches]
    
    # Also extract timeline information for better accuracy
    timeline_matches = re.findall(r"Timeline Start: ([\d.]+) s \| End: ([\d.]+) s", output)
    timeline_segments = [(float(start), float(end)) for start, end in timeline_matches]
    
    return stmp_times, timeline_segments

def create_precise_timestamps(frame_count, avg_fps, creation_unix, stmp_times, timeline_segments):
    """Create precise timestamps using STMP data and timeline information"""
    timestamps = np.zeros(frame_count)
    
    if not stmp_times or not timeline_segments:
        # Fallback to uniform distribution
        print("Warning: No STMP data found, using uniform frame distribution")
        for i in range(frame_count):
            timestamps[i] = creation_unix + (i / avg_fps)
        return timestamps.tolist()
    
    # Calculate frames per segment based on timeline duration
    total_segments = len(timeline_segments)
    frames_processed = 0
    
    for seg_idx, (start_time, end_time) in enumerate(timeline_segments):
        if seg_idx >= len(stmp_times):
            break
            
        segment_duration = end_time - start_time
        frames_in_segment = max(1, int(round(segment_duration * avg_fps)))
        
        # Don't exceed total frame count
        if frames_processed + frames_in_segment > frame_count:
            frames_in_segment = frame_count - frames_processed
        
        # Use STMP timestamp as the reference point for this segment
        stmp_reference = creation_unix + stmp_times[seg_idx]
        
        # Distribute frames evenly within the segment
        for frame_in_seg in range(frames_in_segment):
            frame_idx = frames_processed + frame_in_seg
            if frame_idx < frame_count:
                # Calculate frame time within segment
                if frames_in_segment > 1:
                    progress = frame_in_seg / (frames_in_segment - 1)
                else:
                    progress = 0.0
                
                frame_offset = progress * segment_duration
                timestamps[frame_idx] = stmp_reference + frame_offset
        
        frames_processed += frames_in_segment
        
        if frames_processed >= frame_count:
            break
    
    # Handle any remaining frames (shouldn't happen with good data)
    if frames_processed < frame_count:
        print(f"Warning: {frame_count - frames_processed} frames not assigned timestamps")
        last_timestamp = timestamps[frames_processed - 1] if frames_processed > 0 else creation_unix
        frame_interval = 1.0 / avg_fps
        
        for i in range(frames_processed, frame_count):
            timestamps[i] = last_timestamp + (i - frames_processed + 1) * frame_interval
    
    return timestamps.tolist()

def video_to_rosbag_with_precise_timing(video_path, bag_path, frame_count, timestamps):
    """Convert video to ROS bag with precise timestamps"""
    cap = cv2.VideoCapture(video_path)
    
    # Verify we can read the video
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    
    bag = rosbag.Bag(bag_path, 'w')
    
    try:
        frames_written = 0
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                print(f"End of video reached at frame {i}")
                break
            
            timestamp = rospy.Time.from_sec(timestamps[i])
            
            # Create header with precise timestamp
            header = Header()
            header.stamp = timestamp
            header.frame_id = "gopro"
            header.seq = i
            
            # Convert to ROS image message
            img_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header = header
            
            # Write to bag
            bag.write('/gopro/image_raw', img_msg, t=timestamp)
            
            # Also write wall time for synchronization reference
            wall_time_msg = Float64()
            wall_time_msg.data = timestamps[i]
            bag.write('/gopro/wall_time', wall_time_msg, t=timestamp)
            
            frames_written += 1
            
            if i % 100 == 0:  # Progress indicator
                print(f"Processed {i}/{frame_count} frames")
        
        print(f"Successfully wrote {frames_written} frames to {bag_path}")
        
    finally:
        cap.release()
        bag.close()

def validate_timestamps(timestamps, avg_fps):
    """Validate timestamp consistency and provide statistics"""
    if len(timestamps) < 2:
        return
    
    intervals = np.diff(timestamps)
    expected_interval = 1.0 / avg_fps
    
    print(f"\nTimestamp Validation:")
    print(f"Expected frame interval: {expected_interval:.6f}s ({avg_fps:.2f} fps)")
    print(f"Actual intervals - Mean: {np.mean(intervals):.6f}s, Std: {np.std(intervals):.6f}s")
    print(f"Min interval: {np.min(intervals):.6f}s, Max interval: {np.max(intervals):.6f}s")
    
    # Check for large deviations
    large_deviations = np.abs(intervals - expected_interval) > (expected_interval * 0.1)
    if np.any(large_deviations):
        print(f"Warning: {np.sum(large_deviations)} frames have intervals >10% off expected")

if __name__ == "__main__":
    print("Starting MP4 to ROS bag conversion with precise timing...")
    avg_fps, frame_count, duration = get_video_metadata(video_path)
    print(f"Video: {frame_count} frames, {duration:.3f}s, {avg_fps:.3f} fps")
    creation_unix = get_creation_time_unix(video_path)
    creation_readable = datetime.fromtimestamp(creation_unix)
    print(f"Creation time: {creation_readable} ({creation_unix})")
    stmp_times, timeline_segments = parse_stmp_data(video_path)
    print(f"Found {len(stmp_times)} STMP timestamps and {len(timeline_segments)} timeline segments")
    timestamps = create_precise_timestamps(frame_count, avg_fps, creation_unix, 
                                         stmp_times, timeline_segments)
    validate_timestamps(timestamps, avg_fps)
    video_to_rosbag_with_precise_timing(video_path, bag_path, frame_count, timestamps)
    print("Conversion complete!")