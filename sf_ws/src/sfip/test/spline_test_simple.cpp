#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>

// Use the actual headers but provide minimal ROS stubs
#include <ros/ros.h>
#include "../include/utils/common_utils.hpp"
#include "../include/SplineState.hpp"
#include "../include/Linearizer.hpp"

class SimpleSplineTest {
private:
    struct TestPose {
        int64_t timestampNanoseconds;
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;
        
        TestPose(int64_t t, const Eigen::Vector3d& p, const Eigen::Quaterniond& q) 
            : timestampNanoseconds(t), position(p), orientation(q) {}
    };
    
    struct TestIMU {
        int64_t timestampNanoseconds;
        Eigen::Vector3d accel;
        Eigen::Vector3d gyro;
        
        TestIMU(int64_t t, const Eigen::Vector3d& a, const Eigen::Vector3d& g) 
            : timestampNanoseconds(t), accel(a), gyro(g) {}
    };

    std::vector<TestPose> ground_truth_poses;
    std::vector<TestPose> noisy_pose_measurements;
    std::vector<TestIMU> synthetic_imu;
    
    SplineState spline_test;
    Parameters param;
    
    // Simplified test parameters
    static constexpr int64_t DURATION_SEC = 5;   // Very short test
    static constexpr int64_t NS_PER_SEC = 1000000000LL;
    static constexpr double POSE_FREQ = 10.0;    // Reduced frequency
    static constexpr double IMU_FREQ = 50.0;     // Reduced frequency
    static constexpr double CONTROL_FREQ = 20.0; // Reduced frequency
    static constexpr int WINDOW_SIZE = 3;        // Very small window
    
    std::mt19937 rng;
    std::normal_distribution<double> pose_noise;
    std::normal_distribution<double> imu_noise;

public:
    SimpleSplineTest() : rng(42), pose_noise(0.0, 0.001), imu_noise(0.0, 0.01) {
        setupParameters();
    }
    
    void setupParameters() {
        // Simple parameters
        param.accel_var_inv << 10.0, 10.0, 10.0;
        param.bias_accel_var_inv << 1.0, 1.0, 1.0;
        param.w_acc = 1.0;
        param.w_bias_acc = 1.0;
        
        param.gyro_var_inv << 10.0, 10.0, 10.0;
        param.bias_gyro_var_inv << 1.0, 1.0, 1.0;
        param.w_gyro = 1.0;
        param.w_bias_gyro = 1.0;
        
        param.w_pose_pos = 1000.0;  // High weight for pose anchoring
        param.w_pose_rot = 1000.0;  // High weight for orientation anchoring
        
        param.gravity << 0.0, 0.0, -9.80665;
        param.if_opt_g = false; // Keep gravity fixed
        
        std::cout << "Simple test parameters:" << std::endl;
        std::cout << "  Motion: X-axis bell curve only, Y=0, Z=0, no rotation" << std::endl;
        std::cout << "  Duration: " << DURATION_SEC << " seconds" << std::endl;
        std::cout << "  Pose freq: " << POSE_FREQ << " Hz, IMU freq: " << IMU_FREQ << " Hz" << std::endl;
    }
    
    void generateSimpleTrajectory() {
        std::cout << "\n=== Generating Simple Bell Curve Trajectory ===" << std::endl;
        
        ground_truth_poses.clear();
        
        int64_t start_time = 0;
        double dt_pose = 1.0 / POSE_FREQ;
        int num_poses = static_cast<int>(DURATION_SEC * POSE_FREQ);
        
        for (int i = 0; i < num_poses; i++) {
            double t_sec = i * dt_pose;
            int64_t t_ns = start_time + static_cast<int64_t>(t_sec * NS_PER_SEC);
            
            // Simple bell curve motion ONLY in X-axis
            double progress = t_sec / DURATION_SEC; // 0 to 1
            double bell_factor = std::exp(-std::pow((progress - 0.5) * 3, 2)); // Bell curve centered at 0.5
            
            Eigen::Vector3d position;
            position.x() = 1.0 * bell_factor;  // Simple bell curve motion in X only
            position.y() = 0.0;                // Always zero
            position.z() = 0.0;                // Always zero
            
            // No rotation - always identity quaternion
            Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
            
            ground_truth_poses.emplace_back(t_ns, position, orientation);
        }
        
        std::cout << "Generated " << ground_truth_poses.size() << " ground truth poses" << std::endl;
        
        // Print all poses to show the bell curve
        std::cout << "Bell curve trajectory:" << std::endl;
        for (int i = 0; i < ground_truth_poses.size(); i++) {
            const auto& pose = ground_truth_poses[i];
            std::cout << "  t=" << std::fixed << std::setprecision(2) << pose.timestampNanoseconds / 1e9 
                      << "s: x=" << std::setprecision(4) << pose.position.x() << std::endl;
        }
    }
    
    void generateSimpleMeasurements() {
        std::cout << "\n=== Generating Simple Measurements ===" << std::endl;
        
        noisy_pose_measurements.clear();
        
        // Add minimal noise to ground truth poses
        for (const auto& gt_pose : ground_truth_poses) {
            Eigen::Vector3d noisy_pos = gt_pose.position;
            noisy_pos.x() += pose_noise(rng);  // Small noise in X
            
            // Keep orientation as identity (no rotation)
            Eigen::Quaterniond noisy_orient = Eigen::Quaterniond::Identity();
            
            noisy_pose_measurements.emplace_back(gt_pose.timestampNanoseconds, noisy_pos, noisy_orient);
        }
        
        std::cout << "Generated " << noisy_pose_measurements.size() << " noisy pose measurements" << std::endl;
        
        // Generate very simple synthetic IMU data
        synthetic_imu.clear();
        double dt_imu = 1.0 / IMU_FREQ;
        int num_imu = static_cast<int>(DURATION_SEC * IMU_FREQ);
        
        for (int i = 0; i < num_imu; i++) {
            double t_sec = i * dt_imu;
            int64_t t_ns = static_cast<int64_t>(t_sec * NS_PER_SEC);
            
            // Simple IMU: Just gravity + small noise
            Eigen::Vector3d accel = param.gravity; // Start with gravity
            
            // Add small noise
            accel.x() += imu_noise(rng);
            accel.y() += imu_noise(rng);
            accel.z() += imu_noise(rng);
            
            // No rotation, so gyro should be zero + noise
            Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
            gyro.x() += imu_noise(rng);
            gyro.y() += imu_noise(rng);
            gyro.z() += imu_noise(rng);
            
            synthetic_imu.emplace_back(t_ns, accel, gyro);
        }
        
        std::cout << "Generated " << synthetic_imu.size() << " synthetic IMU measurements" << std::endl;
    }
    
    void testSimpleOptimization() {
        std::cout << "\n=== Testing Simple Spline Optimization ===" << std::endl;
        
        // Initialize spline
        int64_t dt_ns = static_cast<int64_t>(1e9 / CONTROL_FREQ);
        spline_test.init(dt_ns, 0, 0);
        
        std::cout << "Spline initialized with dt_ns=" << dt_ns << " (control_freq=" << CONTROL_FREQ << " Hz)" << std::endl;
        
        // Initialize with first pose (anchor the spline)
        if (!noisy_pose_measurements.empty()) {
            const auto& first_pose = noisy_pose_measurements[0];
            Eigen::Matrix<double, 6, 1> zero_bias = Eigen::Matrix<double, 6, 1>::Zero();
            
            // Add initial knots for stability
            spline_test.addSingleStateKnot(first_pose.orientation, first_pose.position, zero_bias);
            spline_test.addSingleStateKnot(first_pose.orientation, first_pose.position, zero_bias);
            
            std::cout << "Initialized spline with first pose: pos=[" << first_pose.position.transpose() 
                      << "], quat=[" << first_pose.orientation.w() << ", " << first_pose.orientation.x() 
                      << ", " << first_pose.orientation.y() << ", " << first_pose.orientation.z() << "]" << std::endl;
        }
        
        // Simple test: just try one optimization with all data
        std::cout << "\n--- Single Window Optimization Test ---" << std::endl;
        
        // Convert all measurements to required format
        std::vector<PoseData> all_poses;
        std::vector<ImuData> all_imu;
        
        for (const auto& pose : noisy_pose_measurements) {
            all_poses.emplace_back(pose.timestampNanoseconds, pose.orientation, pose.position);
        }
        
        for (const auto& imu : synthetic_imu) {
            all_imu.emplace_back(imu.timestampNanoseconds, imu.gyro, imu.accel);
        }
        
        std::cout << "Using " << all_poses.size() << " poses, " << all_imu.size() << " IMU measurements" << std::endl;
        
        // Add one more knot to have enough for optimization
        if (spline_test.getNumKnots() < 3) {
            Eigen::Vector3d last_pos = spline_test.getKnotPosition(spline_test.getNumKnots()-1);
            Eigen::Quaterniond last_quat = spline_test.getKnotOrientation(spline_test.getNumKnots()-1);
            Eigen::Matrix<double, 6, 1> zero_bias = Eigen::Matrix<double, 6, 1>::Zero();
            spline_test.addSingleStateKnot(last_quat, last_pos, zero_bias);
            std::cout << "Added knot #" << spline_test.getNumKnots()-1 << std::endl;
        }
        
        // Try optimization
        bool converged = optimizeSimple(all_poses, all_imu);
        std::cout << "Optimization " << (converged ? "converged" : "did not converge") << std::endl;
        
        // Print final results
        if (spline_test.getNumKnots() > 0) {
            std::cout << "\nFinal spline state:" << std::endl;
            for (size_t i = 0; i < spline_test.getNumKnots(); i++) {
                Eigen::Vector3d pos = spline_test.getKnotPosition(i);
                Eigen::Quaterniond quat = spline_test.getKnotOrientation(i);
                std::cout << "  Knot[" << i << "]: pos=[" << pos.transpose() << "], quat=["
                          << quat.w() << ", " << quat.x() << ", " << quat.y() << ", " << quat.z() << "]" << std::endl;
            }
        }
    }
    
    bool optimizeSimple(const std::vector<PoseData>& poses, const std::vector<ImuData>& imu_data) {
        if (spline_test.getNumKnots() < 2) {
            std::cout << "  Not enough knots for optimization" << std::endl;
            return false;
        }
        
        // Setup linearizer
        int num_knots = spline_test.getNumKnots();
        size_t bias_block_offset = Linearizer::POSE_SIZE * num_knots;
        size_t gravity_block_offset = bias_block_offset + Linearizer::BIAS_SIZE * num_knots;
        size_t hess_size = gravity_block_offset;
        if (param.if_opt_g) hess_size += Linearizer::G_SIZE;
        
        bool pose_fixed = false;
        
        std::cout << "  Setup: num_knots=" << num_knots << ", hess_size=" << hess_size << std::endl;
        
        // Very simple optimization - just one iteration for debugging
        std::cout << "  Starting single optimization iteration..." << std::endl;
        
        try {
            Linearizer lopt(bias_block_offset, gravity_block_offset, hess_size,
                           &spline_test, &param, pose_fixed);
            
            // Add pose residuals first (they should be safer)
            if (!poses.empty()) {
                std::cout << "    Adding " << poses.size() << " pose residuals" << std::endl;
                Eigen::aligned_deque<PoseData> pose_deque(poses.begin(), poses.end());
                lopt(pose_deque);
                std::cout << "    Pose residuals added, cost so far: " << lopt.error << std::endl;
            }
            
            // Skip IMU for now to isolate the issue
            std::cout << "    Skipping IMU residuals for this simple test" << std::endl;
            
            std::cout << "    Final cost: " << std::scientific << lopt.error << std::endl;
            
            // Check for numerical issues
            if (!std::isfinite(lopt.error)) {
                std::cout << "    Cost is not finite, stopping" << std::endl;
                return false;
            }
            
            std::cout << "    Optimization completed successfully!" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cout << "    Exception: " << e.what() << std::endl;
            return false;
        } catch (...) {
            std::cout << "    Unknown exception" << std::endl;
            return false;
        }
    }
    
    void run() {
        std::cout << "=== Simple Bell Curve Spline Test ===" << std::endl;
        std::cout << "Motion: Bell curve in X-axis only (Y=0, Z=0, no rotation)" << std::endl;
        
        generateSimpleTrajectory();
        generateSimpleMeasurements();
        testSimpleOptimization();
        
        std::cout << "\n=== Simple Test Complete ===" << std::endl;
    }
};

int main(int argc, char** argv) {
    // Initialize ROS for the test (needed for the headers)
    ros::init(argc, argv, "simple_spline_test");
    ros::NodeHandle nh;
    
    try {
        SimpleSplineTest test;
        test.run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
} 