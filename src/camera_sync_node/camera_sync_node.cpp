#include "camera_sync_node/camera_sync_node.hpp"

/*
ROS NODE THAT:
1. records the stream from the camera (pointed at a real-time unix clock on the monitor)
2. computes the timestamp for each frame using v4l2
3. attaches the two in a rosbag for further comparison
*/

CameraSyncNode::CameraSyncNode(ros::NodeHandle& nh) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double monotonic_time = ts.tv_sec + ts.tv_nsec / 1e9;
    offset_ = ros::Time::now().toSec() - monotonic_time;

    video_stream = open(video_device_.c_str(), O_RDWR);
    if (video_stream < 0) {
       perror("Failed to open video device!");
       exit(1);
    }

    setupMetadataBuffers();
    bag_.open(std::string(getenv("HOME")) + "/camera_cpp.bag", rosbag::bagmode::Write);
    sub_ = nh.subscribe("usb_cam/image_raw", 10, &CameraSyncNode::imageCallback, this);
    frame_count_ = 0;
    ROS_INFO("Recording started ...");
}

CameraSyncNode::~CameraSyncNode() {
    int type = V4L2_BUF_TYPE_META_CAPTURE;
    ioctl(video_stream, VIDIOC_STREAMOFF, &type);
    close(video_stream);
    bag_.close();
}

/*
source: https://costinchitic.co/notes/UVC-Video-Stream
*/
void CameraSyncNode::setupMetadataBuffers() {
    v4l2_requestbuffers req = {};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_META_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    ioctl(video_stream, VIDIOC_REQBUFS, &req);

    memset(&buf_, 0, sizeof(buf_));
    buf_.type = V4L2_BUF_TYPE_META_CAPTURE;
    buf_.memory = V4L2_MEMORY_MMAP;
    ioctl(video_stream, VIDIOC_QUERYBUF, &buf_);
    ioctl(video_stream, VIDIOC_QBUF, &buf_);

    int buf_type = V4L2_BUF_TYPE_META_CAPTURE;
    ioctl(video_stream, VIDIOC_STREAMON, &buf_type);
}

/*
Here I extract and convert the camera timestamp to wall time.
*/
void CameraSyncNode::getCameraWalltime(uint32_t& sequence, double& camera_wall_time, double& curr_sys_time) {
    ioctl(video_stream, VIDIOC_DQBUF, &buf_);

    sequence = buf_.sequence;
    double ts = buf_.timestamp.tv_sec + buf_.timestamp.tv_usec / 1e6;
    camera_wall_time = ts + offset_;
    curr_sys_time = ros::Time::now().toSec();

    ioctl(video_stream, VIDIOC_QBUF, &buf_);
}

/*
Process incoming video frames and attach corresponding (sequence,timestamp)
*/
void CameraSyncNode::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    uint32_t seq;
    double cam_ts, sys_ts;
    getCameraWalltime(seq, cam_ts, sys_ts);

    bag_.write("/camera/image_compressed", ros::Time(cam_ts), *msg);
    std_msgs::Float64 ts_msg;
    ts_msg.data = cam_ts;
    bag_.write("/camera_wall_time", ros::Time(cam_ts), ts_msg);

    ROS_INFO("seq %d, ts %f", seq, cam_ts);

    frame_count_++;
    // if (frame_count_ >= 20) ros::shutdown();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "camera_sync_recorder");
    ros::NodeHandle nh;
    CameraSyncNode recorder(nh);
    ros::spin();
    return 0;
}