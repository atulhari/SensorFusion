#include <ros/ros.h>
#include <std_msgs/String.h>

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "sfip_estimation_interface");
    ROS_INFO("\033[1;32m---->\033[0m Starting SFIP EstimationInterface (Simplified Version).");
    
    ros::NodeHandle nh("~");
    
    // Create a publisher for status messages
    ros::Publisher status_pub = nh.advertise<std_msgs::String>("status", 10);
    
    // Run at 1 Hz for minimal functionality
    ros::Rate rate(1);
    while (ros::ok()) {
        ros::spinOnce();
        
        // Publish status message
        std_msgs::String msg;
        msg.data = "SFIP EstimationInterface running (SuiteSparse dependencies removed)";
        status_pub.publish(msg);
        
        rate.sleep();
    }
    
    return 0;
}
