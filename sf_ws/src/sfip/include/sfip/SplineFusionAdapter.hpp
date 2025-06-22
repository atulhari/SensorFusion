#pragma once

#include "sfip/SplineFusionCore.hpp"
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include "sfip/Estimate.h"
#include "sfip/Spline.h"
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace sfip {

/**
 * @brief ROS adapter for SplineFusionCore
 * 
 * This class adapts the SplineFusionCore library to be used with ROS.
 * It handles ROS message conversion, subscriptions, and publications.
 */
class SplineFusionAdapter {
public:
    /**
     * @brief Constructor
     * @param nh ROS node handle
     */
    SplineFusionAdapter(ros::NodeHandle& nh);
    
    /**
     * @brief Destructor
     */
    ~SplineFusionAdapter();
    
    /**
     * @brief Run one iteration of the fusion
     * @return True if iteration was successful
     */
    bool run();

private:
    // ROS node handle
    ros::NodeHandle nh_;
    
    // ROS parameters
    std::string imuTopic_;
    std::string poseTopic_;
    std::string platformFrameId_;
    std::string imuFrameId_;
    std::string outputFrameId_;
    double imuSampleCoeff_;
    double poseSampleCoeff_;
    double imuFrequency_;
    double poseFrequency_;
    double outputVisualizationFps_;
    bool gyroUnit_;
    bool accRatio_;
    
    // TF handling
    tf2_ros::Buffer tfBuffer_;
    std::shared_ptr<tf2_ros::TransformListener> tfListener_;
    Eigen::Vector3d staticTransPlatformImu_;
    Eigen::Quaterniond staticQuatPlatformImu_;
    
    // ROS subscribers and publishers
    ros::Subscriber subImu_;
    ros::Subscriber subPoseRaw_;
    ros::Publisher pubImuDs_;
    ros::Publisher pubPoseDs_;
    ros::Publisher pubEstimate_;
    ros::Publisher pubFusedPose_;
    ros::Publisher pubOptOld_;
    ros::Publisher pubOptWindow_;
    
    // Fusion core
    std::shared_ptr<SplineFusionCore> fusionCore_;
    
    // ROS callback functions
    void imuCallback(const sensor_msgs::Imu::ConstPtr& imuMsg);
    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& poseMsg);
    void startCallback(const std_msgs::Int64::ConstPtr& startTimeMsg);
    
    // TF helper functions
    bool getStaticTransform();
    bool transformPoseToImuStaticOnly(const geometry_msgs::PoseStamped& in, geometry_msgs::PoseStamped& out);
    geometry_msgs::PoseStamped transformImuToOutput(const geometry_msgs::PoseStamped& in);
    
    // Utility functions
    bool sampleData(const int64_t tNs, const int64_t lastTNs, const double coeff, const double frequency) const;
    void publishEstimate(const OptimizationResult& result);
    void createFusionParameters(FusionParameters& params);
    void publishVisualization(const std::shared_ptr<SplineState>& splineState, bool isWindowFull);
    void convertSplineToRosMessage(const std::shared_ptr<SplineState>& splineState, sfip::Spline& splineMsg);
    
    // State tracking
    int64_t latestPoseTimestamp_;
    double averageRuntime_;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace sfip
