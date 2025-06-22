#include "sfip/SplineFusionAdapter.hpp"
#include <nav_msgs/Path.h>
#include <std_msgs/Int64.h>

namespace sfip {

SplineFusionAdapter::SplineFusionAdapter(ros::NodeHandle& nh) 
    : nh_(nh),
      latestPoseTimestamp_(0),
      averageRuntime_(0)
{
    // Create TF listener
    tfListener_ = std::make_shared<tf2_ros::TransformListener>(tfBuffer_);
    
    // Read ROS parameters
    nh_.param<std::string>("topic_imu", imuTopic_, "/imu");
    nh_.param<std::string>("topic_pose_input", poseTopic_, "/pose_data");
    nh_.param<std::string>("platform_frame_id", platformFrameId_, "camera_link");
    nh_.param<std::string>("imu_frame", imuFrameId_, "imu_link");
    nh_.param<std::string>("output_frame", outputFrameId_, "map");
    nh_.param<double>("imu_sample_coeff", imuSampleCoeff_, 1.0);
    nh_.param<double>("pose_sample_coeff", poseSampleCoeff_, 1.0);
    nh_.param<double>("imu_frequency", imuFrequency_, 200.0);
    nh_.param<double>("pose_frequency", poseFrequency_, 20.0);
    nh_.param<double>("output_visualization_fps", outputVisualizationFps_, 60.0);
    nh_.param<bool>("gyro_unit", gyroUnit_, false);
    nh_.param<bool>("acc_ratio", accRatio_, false);
    
    // Get the static transform from platform to IMU frame at startup
    if (!getStaticTransform()) {
        ROS_ERROR("[SFA] Failed to get static transform from platform to IMU. This is mandatory.");
        ros::shutdown();
        return;
    }
    
    ROS_INFO_STREAM("[SFA] Using imu_frame='" << imuFrameId_ << "'  platform_frame='" << platformFrameId_ 
                   << "'  output_frame='" << outputFrameId_ << "'.");
    ROS_INFO_STREAM("[SFA] Static transform platform->IMU: t=[" 
                   << staticTransPlatformImu_.x() << ", " 
                   << staticTransPlatformImu_.y() << ", " 
                   << staticTransPlatformImu_.z() << "]"
                   << " q=[" << staticQuatPlatformImu_.w() << ", " 
                   << staticQuatPlatformImu_.x() << ", " 
                   << staticQuatPlatformImu_.y() << ", " 
                   << staticQuatPlatformImu_.z() << "]");
    
    // Setup ROS subscribers and publishers
    subImu_ = nh_.subscribe(imuTopic_, 1000, &SplineFusionAdapter::imuCallback, this);
    subPoseRaw_ = nh_.subscribe(poseTopic_, 1000, &SplineFusionAdapter::poseCallback, this);
    
    pubImuDs_ = nh_.advertise<sensor_msgs::Imu>("imu_ds", 400);
    pubPoseDs_ = nh_.advertise<geometry_msgs::PoseStamped>("pose_ds", 1000);
    pubEstimate_ = nh_.advertise<sfip::Estimate>("est_window", 1000);
    pubFusedPose_ = nh_.advertise<geometry_msgs::PoseStamped>("fused_pose", 10);
    pubOptOld_ = nh_.advertise<nav_msgs::Path>("bspline_optimization_old", 1000);
    pubOptWindow_ = nh_.advertise<nav_msgs::Path>("bspline_optimization_window", 1000);
    
    // Create fusion parameters
    FusionParameters params;
    createFusionParameters(params);
    
    // Create fusion core
    fusionCore_ = std::make_shared<SplineFusionCore>(params);
}

SplineFusionAdapter::~SplineFusionAdapter() {
    // Nothing special to clean up
}

bool SplineFusionAdapter::run() {
    // Check if fusion core is initialized
    if (!fusionCore_->initialize()) {
        // Not initialized yet, need more data
        return false;
    }
    
    // Run optimization
    OptimizationResult result = fusionCore_->runOptimization();
    
    // Publish results
    publishEstimate(result);
    
    return true;
}

void SplineFusionAdapter::imuCallback(const sensor_msgs::Imu::ConstPtr& imuMsg) {
    static int64_t lastImu = 0;
    int64_t tNs = imuMsg->header.stamp.toNSec();
    
    // Validate timestamp is in sequence
    if (lastImu > 0 && tNs <= lastImu) {
        ROS_WARN_STREAM_THROTTLE(1.0, "[SFA] IMU timestamp out of sequence: current=" 
                                << tNs << " last=" << lastImu << ". Skipping.");
        return;
    }
    
    // Pre-filter IMU data
    Eigen::Vector3d accRaw(imuMsg->linear_acceleration.x,
                          imuMsg->linear_acceleration.y,
                          imuMsg->linear_acceleration.z);
    Eigen::Vector3d gyroRaw(imuMsg->angular_velocity.x,
                           imuMsg->angular_velocity.y,
                           imuMsg->angular_velocity.z);
    
    // Apply unit conversions if needed
    if (accRatio_) accRaw *= 9.81;
    if (gyroUnit_) gyroRaw *= M_PI / 180.0;
    
    // Don't accept IMU data older than latest pose
    if (latestPoseTimestamp_ > 0 && tNs < latestPoseTimestamp_) {
        return;
    }
    
    // Subsample data if needed
    if (sampleData(tNs, lastImu, imuSampleCoeff_, imuFrequency_)) {
        lastImu = tNs;
        
        // Convert to internal format
        ImuMeasurement imu(tNs, gyroRaw, accRaw);
        
        // Add to fusion core
        fusionCore_->addImuMeasurement(imu);
        
        // Also publish the downsampled IMU data for other nodes
        sensor_msgs::Imu imuDsMsg;
        imuDsMsg.header = imuMsg->header;
        imuDsMsg.linear_acceleration.x = accRaw[0];
        imuDsMsg.linear_acceleration.y = accRaw[1];
        imuDsMsg.linear_acceleration.z = accRaw[2];
        imuDsMsg.angular_velocity.x = gyroRaw[0];
        imuDsMsg.angular_velocity.y = gyroRaw[1];
        imuDsMsg.angular_velocity.z = gyroRaw[2];
        pubImuDs_.publish(imuDsMsg);
    }
}

void SplineFusionAdapter::poseCallback(const geometry_msgs::PoseStamped::ConstPtr& poseMsg) {
    static int64_t lastPoseInputNs = 0;
    static int64_t lastPoseRawNs = 0;
    
    int64_t inputTNs = poseMsg->header.stamp.toNSec();
    
    // Validate timestamp is in sequence
    if (lastPoseInputNs > 0 && inputTNs <= lastPoseInputNs) {
        ROS_WARN_STREAM_THROTTLE(1.0, "[SFA] Pose timestamp out of sequence: current=" 
                                << inputTNs << " last=" << lastPoseInputNs << ". Skipping.");
        return;
    }
    
    // Check if timestamp is too far in the past
    ros::Time now = ros::Time::now();
    int64_t nowNs = now.toNSec();
    double ageSec = (nowNs - inputTNs) * 1e-9;
    const double MAX_POSE_AGE_SEC = 2.0;
    if (ageSec > MAX_POSE_AGE_SEC) {
        ROS_WARN_STREAM_THROTTLE(5.0, "[SFA] Pose timestamp too old: " << ageSec << " seconds. Skipping.");
        return;
    }
    
    // Transform pose to IMU frame
    geometry_msgs::PoseStamped poseIn = *poseMsg;
    geometry_msgs::PoseStamped poseInImu;
    if (!transformPoseToImuStaticOnly(poseIn, poseInImu)) {
        return;  // Skip if transformation fails
    }
    
    // Convert to internal format
    Eigen::Quaterniond orientation(poseInImu.pose.orientation.w, 
                                  poseInImu.pose.orientation.x,
                                  poseInImu.pose.orientation.y, 
                                  poseInImu.pose.orientation.z);
    Eigen::Vector3d position(poseInImu.pose.position.x, 
                            poseInImu.pose.position.y,
                            poseInImu.pose.position.z);
    
    // Create internal pose measurement
    PoseMeasurement pose(poseInImu.header.stamp.toNSec(), orientation, position);
    
    // Update latest pose timestamp for IMU filtering
    latestPoseTimestamp_ = pose.timestampNanoseconds;
    
    // Subsample data if needed
    if (sampleData(pose.timestampNanoseconds, lastPoseRawNs, poseSampleCoeff_, poseFrequency_)) {
        // Add to fusion core
        fusionCore_->addPoseMeasurement(pose);
        
        // Also publish the downsampled pose data for other nodes
        pubPoseDs_.publish(poseInImu);
        
        lastPoseRawNs = pose.timestampNanoseconds;
    }
    
    lastPoseInputNs = inputTNs;
}

bool SplineFusionAdapter::getStaticTransform() {
    // Try to get the static transform from platform to IMU frame
    int retryCount = 0;
    const int maxRetries = 50;
    const double retryDelay = 0.1;  // seconds
    
    while (retryCount < maxRetries && ros::ok()) {
        try {
            geometry_msgs::TransformStamped tfStamped = tfBuffer_.lookupTransform(
                imuFrameId_, platformFrameId_, ros::Time(0), ros::Duration(retryDelay));
            
            // Extract translation and rotation
            staticTransPlatformImu_.x() = tfStamped.transform.translation.x;
            staticTransPlatformImu_.y() = tfStamped.transform.translation.y;
            staticTransPlatformImu_.z() = tfStamped.transform.translation.z;
            
            staticQuatPlatformImu_.w() = tfStamped.transform.rotation.w;
            staticQuatPlatformImu_.x() = tfStamped.transform.rotation.x;
            staticQuatPlatformImu_.y() = tfStamped.transform.rotation.y;
            staticQuatPlatformImu_.z() = tfStamped.transform.rotation.z;
            staticQuatPlatformImu_.normalize();
            
            ROS_INFO("[SFA] Successfully obtained static transform from platform to IMU frame.");
            return true;
            
        } catch (const tf2::TransformException& ex) {
            retryCount++;
            ROS_WARN_STREAM_THROTTLE(1.0, "[SFA] Attempt " << retryCount << "/" << maxRetries 
                                    << " to get static transform failed: " << ex.what());
            
            if (retryCount < maxRetries) {
                ros::Duration(retryDelay).sleep();
            }
        }
    }
    
    ROS_ERROR("[SFA] Failed to get static transform after %d attempts.", maxRetries);
    return false;
}

bool SplineFusionAdapter::transformPoseToImuStaticOnly(const geometry_msgs::PoseStamped& in, 
                                                     geometry_msgs::PoseStamped& out) {
    // Use the pre-computed static transform only
    
    // If input is already in platform frame, apply static transform directly
    if (in.header.frame_id == platformFrameId_) {
        // Apply static transform: T_imu = T_imu_platform * T_platform_world * pose_in_world
        // Since pose_in is in platform frame: T_imu = T_imu_platform * pose_in_platform
        
        Eigen::Vector3d posIn(in.pose.position.x, in.pose.position.y, in.pose.position.z);
        Eigen::Quaterniond quatIn(in.pose.orientation.w, in.pose.orientation.x, 
                               in.pose.orientation.y, in.pose.orientation.z);
        
        // Transform position: p_imu = R_imu_platform * p_platform + t_imu_platform
        Eigen::Vector3d posOut = staticQuatPlatformImu_ * posIn + staticTransPlatformImu_;
        
        // Transform orientation: q_imu = q_imu_platform * q_platform
        Eigen::Quaterniond quatOut = staticQuatPlatformImu_ * quatIn;
        quatOut.normalize();
        
        out.header = in.header;
        out.header.frame_id = imuFrameId_;
        out.pose.position.x = posOut.x();
        out.pose.position.y = posOut.y();
        out.pose.position.z = posOut.z();
        out.pose.orientation.w = quatOut.w();
        out.pose.orientation.x = quatOut.x();
        out.pose.orientation.y = quatOut.y();
        out.pose.orientation.z = quatOut.z();
        
        return true;
    } else {
        // If not in platform frame, try to get transform to platform frame first
        try {
            geometry_msgs::PoseStamped posePlatform;
            geometry_msgs::TransformStamped tfToPlatform = tfBuffer_.lookupTransform(
                platformFrameId_, in.header.frame_id, ros::Time(0), ros::Duration(0.1));
            tf2::doTransform(in, posePlatform, tfToPlatform);
            
            // Now apply static transform
            return transformPoseToImuStaticOnly(posePlatform, out);
            
        } catch (const tf2::TransformException& ex) {
            ROS_WARN_STREAM_THROTTLE(2.0, "[SFA] transformPoseToImuStaticOnly failed: " << ex.what());
            return false;
        }
    }
}

geometry_msgs::PoseStamped SplineFusionAdapter::transformImuToOutput(const geometry_msgs::PoseStamped& in) {
    geometry_msgs::PoseStamped out = in;
    if (outputFrameId_ == imuFrameId_) return out;
    
    try {
        // Use ros::Time(0) for output transform to avoid extrapolation warnings
        geometry_msgs::TransformStamped tf = tfBuffer_.lookupTransform(
            outputFrameId_, imuFrameId_, ros::Time(0), ros::Duration(0.1));
        tf2::doTransform(in, out, tf);
        out.header.frame_id = outputFrameId_;
        // Preserve the original timestamp for the fused pose
        out.header.stamp = in.header.stamp;
    } catch (const tf2::TransformException& ex) {
        ROS_WARN_STREAM_THROTTLE(2.0, "[SFA] transformImuToOutput failed: " << ex.what());
    }
    
    return out;
}

bool SplineFusionAdapter::sampleData(const int64_t tNs, const int64_t lastTNs,
                                   const double coeff, const double frequency) const {
    if (coeff == 0)
        return false;
    
    int64_t dt = 1e9 / (coeff * frequency);
    
    if (coeff == 1) {
        return true;
    } else if (tNs - lastTNs > dt - 1e5) {
        return true;
    } else {
        return false;
    }
}

void SplineFusionAdapter::publishEstimate(const OptimizationResult& result) {
    // Get spline state from fusion core
    auto splineState = fusionCore_->getSplineState();
    
    // Get window state
    WindowState windowState = fusionCore_->getWindowState();
    
    // Create spline message
    sfip::Spline splineMsg;
    convertSplineToRosMessage(splineState, splineMsg);
    
    // Create estimate message
    sfip::Estimate estMsg;
    estMsg.spline = splineMsg;
    estMsg.if_full_window.data = windowState.isFullSize;
    estMsg.runtime.data = averageRuntime_;
    
    // Update average runtime
    if (result.runtime > 0) {
        static int numWindows = 0;
        averageRuntime_ = (result.runtime + static_cast<double>(numWindows) * averageRuntime_) / 
                         static_cast<double>(numWindows + 1);
        numWindows++;
    }
    
    // Publish estimate
    pubEstimate_.publish(estMsg);
    
    // Publish visualization
    publishVisualization(splineState, windowState.isFullSize);
    
    // Publish latest pose estimate
    if (splineState->getNumKnots() > 0) {
        int64_t latestTime = splineState->maxTimeNanoseconds();
        
        geometry_msgs::PoseStamped fusedPsMsg;
        fusedPsMsg.header.stamp.fromNSec(latestTime);
        fusedPsMsg.header.frame_id = imuFrameId_;
        
        Eigen::Quaterniond latestOrientation;
        splineState->interpolateQuaternion(latestTime, &latestOrientation);
        Eigen::Vector3d latestPosition = splineState->interpolatePosition(latestTime);
        
        fusedPsMsg.pose.position.x = latestPosition.x();
        fusedPsMsg.pose.position.y = latestPosition.y();
        fusedPsMsg.pose.position.z = latestPosition.z();
        fusedPsMsg.pose.orientation.w = latestOrientation.w();
        fusedPsMsg.pose.orientation.x = latestOrientation.x();
        fusedPsMsg.pose.orientation.y = latestOrientation.y();
        fusedPsMsg.pose.orientation.z = latestOrientation.z();
        
        // Transform to output frame if needed
        geometry_msgs::PoseStamped fusedOut = transformImuToOutput(fusedPsMsg);
        pubFusedPose_.publish(fusedOut);
    }
}

void SplineFusionAdapter::createFusionParameters(FusionParameters& params) {
    // Read parameters from ROS parameter server
    bool ifOptG;
    int windowSize, maxIter, controlPointFps;
    double wPosePos, wPoseRot, wAcc, wGyro, wBiasAcc, wBiasGyro;
    double initialLambda, initialLambdaVee;
    std::vector<double> accelVarInv, gyroVarInv, biasAccelVarInv, biasGyroVarInv;
    std::vector<double> gravityInitial;
    
    nh_.param<bool>("if_opt_g", ifOptG, true);
    nh_.param<int>("window_size", windowSize, 10);
    nh_.param<int>("max_iter", maxIter, 10);
    nh_.param<int>("control_point_fps", controlPointFps, 20);
    
    nh_.param<double>("w_pose_pos", wPosePos, 1.0);
    nh_.param<double>("w_pose_rot", wPoseRot, 1.0);
    nh_.param<double>("w_accel", wAcc, 1.0);
    nh_.param<double>("w_gyro", wGyro, 1.0);
    nh_.param<double>("w_bias_accel", wBiasAcc, 1.0);
    nh_.param<double>("w_bias_gyro", wBiasGyro, 1.0);
    
    nh_.param<double>("initial_lambda", initialLambda, 1e-6);
    nh_.param<double>("initial_lambda_vee", initialLambdaVee, 2.0);
    
    nh_.param<std::vector<double>>("accel_var_inv", accelVarInv, {1.0, 1.0, 1.0});
    nh_.param<std::vector<double>>("gyro_var_inv", gyroVarInv, {1.0, 1.0, 1.0});
    nh_.param<std::vector<double>>("bias_accel_var_inv", biasAccelVarInv, {1.0, 1.0, 1.0});
    nh_.param<std::vector<double>>("bias_gyro_var_inv", biasGyroVarInv, {1.0, 1.0, 1.0});
    nh_.param<std::vector<double>>("gravity_initial", gravityInitial, {0.0, 0.0, -9.80665});
    
    // Set parameters
    params.optimizeGravity = ifOptG;
    params.windowSize = windowSize;
    params.maxIterations = maxIter;
    params.controlPointFps = controlPointFps;
    params.knotIntervalNanoseconds = static_cast<int64_t>(1e9 / controlPointFps);
    
    params.weightPosePosition = wPosePos;
    params.weightPoseOrientation = wPoseRot;
    params.weightAccel = wAcc;
    params.weightGyro = wGyro;
    params.weightBiasAccel = wBiasAcc;
    params.weightBiasGyro = wBiasGyro;
    
    params.initialLambda = initialLambda;
    params.initialLambdaVee = initialLambdaVee;
    
    // Set measurement variances if provided
    if (accelVarInv.size() == 3) {
        params.accelVarianceInv << accelVarInv[0], accelVarInv[1], accelVarInv[2];
    }
    
    if (gyroVarInv.size() == 3) {
        params.gyroVarianceInv << gyroVarInv[0], gyroVarInv[1], gyroVarInv[2];
    }
    
    if (biasAccelVarInv.size() == 3) {
        params.biasAccelVarianceInv << biasAccelVarInv[0], biasAccelVarInv[1], biasAccelVarInv[2];
    }
    
    if (biasGyroVarInv.size() == 3) {
        params.biasGyroVarianceInv << biasGyroVarInv[0], biasGyroVarInv[1], biasGyroVarInv[2];
    }
    
    // Set initial gravity if provided
    if (gravityInitial.size() == 3) {
        params.gravity << gravityInitial[0], gravityInitial[1], gravityInitial[2];
    }
    
    // Set pose-only mode based on IMU coefficient
    params.poseOnlyMode = (imuSampleCoeff_ == 0);
    
    // Log configuration
    ROS_INFO_STREAM("[SFA] Created fusion parameters: " 
                  << "poseOnlyMode=" << (params.poseOnlyMode ? "true" : "false")
                  << ", controlPointFps=" << params.controlPointFps
                  << ", windowSize=" << params.windowSize);
}

void SplineFusionAdapter::publishVisualization(const std::shared_ptr<SplineState>& splineState, bool isWindowFull) {
    // TODO: Implement visualization publishing for debugging/visualization
    // This should create path messages from the spline for visualization in RViz
}

void SplineFusionAdapter::convertSplineToRosMessage(const std::shared_ptr<SplineState>& splineState, sfip::Spline& splineMsg) {
    // This is a temporary adapter function that will be removed once the refactoring is complete
    // It simply adapts the new SplineState to the ROS message format for backward compatibility
    splineState->getSplineMsg(reinterpret_cast<void*>(&splineMsg));
}

} // namespace sfip
