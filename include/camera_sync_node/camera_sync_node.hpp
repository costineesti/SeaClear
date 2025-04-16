#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float64.h>
#include <fstream>

class CameraSyncNode {
public:
    CameraSyncNode(ros::NodeHandle& nh);
    ~CameraSyncNode();

private:
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void setupMetadataBuffers();
    void getCameraWalltime(uint32_t& sequence, double& camera_wall_time, double& curr_sys_time);

    int video_stream;
    struct v4l2_buffer buf_;
    double offset_;
    int frame_count_;

    std::string video_device_ = "/dev/video5";
    rosbag::Bag bag_;
    ros::Subscriber sub_;
};