#include "sfip/SplineFusionCore.hpp"
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include "sfip/Estimate.h"
#include "sfip/Spline.h"

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "sfip_spline_fusion");
    ROS_INFO("\033[1;32m---->\033[0m Starting SFIP SplineFusion.");
    
    ros::NodeHandle nh("~");
    
    // Create fusion parameters
    sfip::FusionParameters params;
    
    // Read parameters from ROS parameter server
    bool ifOptG;
    int windowSize, maxIter, controlPointFps;
    double wPosePos, wPoseRot, wAcc, wGyro, wBiasAcc, wBiasGyro;
    double initialLambda, initialLambdaVee;
    double imuSampleCoeff;
    std::vector<double> accelVarInv, gyroVarInv, biasAccelVarInv, biasGyroVarInv;
    std::vector<double> gravityInitial;
    
    nh.param<bool>("if_opt_g", ifOptG, true);
    nh.param<int>("window_size", windowSize, 10);
    nh.param<int>("max_iter", maxIter, 10);
    nh.param<int>("control_point_fps", controlPointFps, 20);
    nh.param<double>("imu_sample_coeff", imuSampleCoeff, 1.0);
    
    nh.param<double>("w_pose_pos", wPosePos, 1.0);
    nh.param<double>("w_pose_rot", wPoseRot, 1.0);
    nh.param<double>("w_accel", wAcc, 1.0);
    nh.param<double>("w_gyro", wGyro, 1.0);
    nh.param<double>("w_bias_accel", wBiasAcc, 1.0);
    nh.param<double>("w_bias_gyro", wBiasGyro, 1.0);
    
    nh.param<double>("initial_lambda", initialLambda, 1e-6);
    nh.param<double>("initial_lambda_vee", initialLambdaVee, 2.0);
    
    nh.param<std::vector<double>>("accel_var_inv", accelVarInv, {1.0, 1.0, 1.0});
    nh.param<std::vector<double>>("gyro_var_inv", gyroVarInv, {1.0, 1.0, 1.0});
    nh.param<std::vector<double>>("bias_accel_var_inv", biasAccelVarInv, {1.0, 1.0, 1.0});
    nh.param<std::vector<double>>("bias_gyro_var_inv", biasGyroVarInv, {1.0, 1.0, 1.0});
    nh.param<std::vector<double>>("gravity_initial", gravityInitial, {0.0, 0.0, -9.80665});
    
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
    params.poseOnlyMode = (imuSampleCoeff == 0);
    
    // Create the fusion core
    auto fusionCore = std::make_shared<sfip::SplineFusionCore>(params);
    
    // Create publisher for estimate
    ros::Publisher pubEst = nh.advertise<sfip::Estimate>("est_window", 1000);
    
    // Create publisher for start time
    ros::Publisher pubStartTime = nh.advertise<std_msgs::Int64>("start_time", 1000);
    
    // Run at 1000 Hz
    ros::Rate rate(1000);
    
    while (ros::ok()) {
        ros::spinOnce();
        
        // Check if fusion core is initialized
        if (fusionCore->initialize()) {
            // Run optimization
            sfip::OptimizationResult result = fusionCore->runOptimization();
            
            // Get spline state
            auto splineState = fusionCore->getSplineState();
            
            // Get window state
            sfip::WindowState windowState = fusionCore->getWindowState();
            
            // Create spline message
            sfip::Spline splineMsg;
            splineState->getSplineMsg(reinterpret_cast<void*>(&splineMsg));
            
            // Create estimate message
            sfip::Estimate estMsg;
            estMsg.spline = splineMsg;
            estMsg.if_full_window.data = windowState.isFullSize;
            estMsg.runtime.data = result.runtime;
            
            // Publish estimate
            pubEst.publish(estMsg);
        }
        
        rate.sleep();
    }
    
    return 0;
}
