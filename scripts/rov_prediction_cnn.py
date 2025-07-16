#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge
import torch
from ultralytics import YOLO

class RobotDetectionTask:
    def __init__(self):
        rospy.init_node('robot_detection_node', anonymous=True)
        self.bridge = CvBridge()

        # Load YOLOv8 model
        model_path = "/home/costin/Documents/SeaClear/CNN/best.pt"
        try:
            self.model = YOLO(model_path)
            # Check for available devices and set the device accordingly
            if torch.backends.mps.is_available(): # MACOS
                device = 'mps'
                rospy.loginfo("Using MPS (Metal Performance Shaders)")
            elif torch.cuda.is_available(): # NVIDIA GPU. Best case.
                device = 'cuda'
                rospy.loginfo("Using CUDA")
            else:
                device = 'cpu' # CPU fallback. VM usually runs on CPU
                rospy.loginfo("Using CPU")
            self.model.to(device)
            rospy.loginfo(f"Successfully loaded YOLO model from {model_path} on {device}")
            
        except Exception as e:
            rospy.logerr(f"Failed to load YOLO model: {e}")
            return
        
        # Publishers
        self.detection_pub_gopro = rospy.Publisher('/robotdetection/coordinates_gopro', PointStamped, queue_size=10)
        self.detection_pub_usbcamera = rospy.Publisher('/robotdetection/coordinates_usbcamera', PointStamped, queue_size=10)
        self.image_pub_gopro = rospy.Publisher('/robotdetection/gopro', Image, queue_size=10)
        self.image_pub_usbcamera = rospy.Publisher('/robotdetection/usbcamera', Image, queue_size=10)
        
        self.current_depth = None
        # Detection parameters
        self.confidence_threshold = 0.5
        self.last_detection_time = rospy.Time.now()
        
        rospy.loginfo("Robot detection node initialized successfully")
        rospy.loginfo("Publishing GoPro annotated images to /robotdetection/gopro")
        rospy.loginfo("Publishing USB camera annotated images to /robotdetection/usbcamera")

    def process_image(self, msg, camera_type):
        """Unified image processing method for both cameras (gopro and usbcamera)"""
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Run YOLO inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            annotated_image = results[0].plot() # Use YOLOv8's built-in visualization

            detection_found = False
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    # Get the detection with highest confidence
                    confidences = boxes.conf.cpu().numpy()
                    best_idx = np.argmax(confidences)
                    
                    # Extract bounding box coordinates and compute center coordinates
                    box = boxes.xyxy[best_idx].cpu().numpy()
                    x1, y1, x2, y2 = box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    confidence = confidences[best_idx]
                    
                    self.publish_detection(center_x, center_y, confidence, msg.header.stamp, camera_type)
                    detection_found = True
                
                    self.last_detection_time = rospy.Time.now()
            
            # Add "No Detection" text when no robot is found
            if not detection_found:
                cv2.putText(annotated_image, f"No Robot Detected ({camera_type})", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                rospy.logwarn(f"No robot detected on {camera_type}")
                
            self.publish_annotated_image(annotated_image, msg.header, camera_type)
                    
        except Exception as e:
            rospy.logerr(f"Error in {camera_type} image callback: {e}")

    def image_callback_usbcamera(self, msg):
        self.process_image(msg, "usbcamera")
    
    def image_callback_gopro(self, msg):
        self.process_image(msg, "gopro")
    
    def depth_callback(self, msg):
        self.current_depth = msg.pose.pose.position.z
    
    def publish_detection(self, x, y, confidence, timestamp, camera_type):
        point_msg = PointStamped()
        point_msg.header.stamp = timestamp
        point_msg.header.frame_id = "camera_frame"
        
        point_msg.point.x = float(x)
        point_msg.point.y = float(y)
        point_msg.point.z = self.current_depth if self.current_depth is not None else 0.0

        if self.current_depth is None:
            rospy.logwarn("Current depth is not available, using default value of 0.0")
        
        # Publish to appropriate topic based on camera type
        if camera_type == "gopro":
            self.detection_pub_gopro.publish(point_msg)
        elif camera_type == "usbcamera":
            self.detection_pub_usbcamera.publish(point_msg)
    
    def publish_annotated_image(self, image, header, camera_type):
        """Publish the annotated image to appropriate topic"""
        try:
            img_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            img_msg.header = header
            
            if camera_type == "gopro":
                self.image_pub_gopro.publish(img_msg)
            elif camera_type == "usbcamera":
                self.image_pub_usbcamera.publish(img_msg)
                
        except Exception as e:
            rospy.logerr(f"Error publishing annotated image for {camera_type}: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        rospy.loginfo("Cleaning up robot detection node...")
        cv2.destroyAllWindows()

def main():
    try:
        task = RobotDetectionTask()
        
        rospy.Subscriber('/camera/image_compressed', Image, task.image_callback_usbcamera)
        rospy.Subscriber('/gopro/image_raw', Image, task.image_callback_gopro)
        rospy.Subscriber('/BlueRov2/odom/depth', Odometry, task.depth_callback)
        
        rospy.loginfo("Robot detection node started")
        rospy.spin()
        
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")
    finally:
        if 'task' in locals():
            task.cleanup()

if __name__ == '__main__':
    main()