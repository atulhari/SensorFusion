#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include <ros/ros.h>
#include "../include/utils/common_utils.hpp"
#include "../include/SplineState.hpp"
#include "../include/Linearizer.hpp"

class SimpleBellTest {
private:
    SplineState spline_test;
    Parameters param;
    
    // Very simple test parameters
    static constexpr int64_t DURATION_SEC = 5;
    static constexpr int64_t NS_PER_SEC = 1000000000LL;
    static constexpr double POSE_FREQ = 10.0;
    static constexpr double CONTROL_FREQ = 20.0;

public:
    SimpleBellTest() {
        setupParameters();
    }
    
    void setupParameters() {
        param.accel_var_inv << 10.0, 10.0, 10.0;
        param.bias_accel_var_inv << 1.0, 1.0, 1.0;
        param.w_acc = 1.0;
        param.w_bias_acc = 1.0;
        
        param.gyro_var_inv << 10.0, 10.0, 10.0;
        param.bias_gyro_var_inv << 1.0, 1.0, 1.0;
        param.w_gyro = 1.0;
        param.w_bias_gyro = 1.0;
        
        param.w_pose_pos = 1000.0;
        param.w_pose_rot = 1000.0;
        
        param.gravity << 0.0, 0.0, -9.80665;
        param.if_opt_g = false;
        
        std::cout << "Simple Bell Test Parameters:" << std::endl;
        std::cout << "  Motion: X-axis bell curve only, Y=0, Z=0, no rotation" << std::endl;
        std::cout << "  Duration: " << DURATION_SEC << " seconds" << std::endl;
    }
    
    void generateBellCurveTrajectory() {
        std::cout << "\n=== Generating Bell Curve Trajectory ===" << std::endl;
        
        std::vector<PoseData> poses;
        
        double dt_pose = 1.0 / POSE_FREQ;
        int num_poses = static_cast<int>(DURATION_SEC * POSE_FREQ);
        
        for (int i = 0; i < num_poses; i++) {
            double t_sec = i * dt_pose;
            int64_t t_ns = static_cast<int64_t>(t_sec * NS_PER_SEC);
            
            // Bell curve motion ONLY in X-axis
            double progress = t_sec / DURATION_SEC; // 0 to 1
            double bell_factor = std::exp(-std::pow((progress - 0.5) * 3, 2));
            
            Eigen::Vector3d position;
            position.x() = 1.0 * bell_factor;  // Bell curve in X
            position.y() = 0.0;                // Always zero
            position.z() = 0.0;                // Always zero
            
            // No rotation - always identity
            Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
            
            poses.emplace_back(t_ns, orientation, position);
        }
        
        std::cout << "Generated " << poses.size() << " poses with bell curve trajectory:" << std::endl;
        for (const auto& pose : poses) {
            std::cout << "  t=" << std::fixed << std::setprecision(2) << pose.timestampNanoseconds / 1e9 
                      << "s: x=" << std::setprecision(4) << pose.position.x() << std::endl;
        }
        
        // Test basic spline operations
        testBasicSpline(poses);
    }
    
    void testBasicSpline(const std::vector<PoseData>& poses) {
        std::cout << "\n=== Testing Basic Spline Operations ===" << std::endl;
        
        // Initialize spline
        int64_t dt_ns = static_cast<int64_t>(1e9 / CONTROL_FREQ);
        spline_test.init(dt_ns, 0, 0);
        
        std::cout << "Spline initialized with dt_ns=" << dt_ns << std::endl;
        
        // Add initial knots
        if (!poses.empty()) {
            const auto& first_pose = poses[0];
            Eigen::Matrix<double, 6, 1> zero_bias = Eigen::Matrix<double, 6, 1>::Zero();
            
            // Add 3 knots for basic functionality
            for (int i = 0; i < 3; i++) {
                spline_test.addSingleStateKnot(first_pose.orientation, first_pose.position, zero_bias);
                std::cout << "Added knot " << i << ": pos=[" << first_pose.position.transpose() << "]" << std::endl;
            }
        }
        
        // Test basic operations
        std::cout << "\nTesting basic spline operations:" << std::endl;
        std::cout << "  Number of knots: " << spline_test.getNumKnots() << std::endl;
        std::cout << "  Min time: " << spline_test.minTimeNanoseconds() / 1e9 << "s" << std::endl;
        std::cout << "  Max time: " << spline_test.maxTimeNanoseconds() / 1e9 << "s" << std::endl;
        
        // Test interpolation
        int64_t test_time = spline_test.minTimeNanoseconds() + 
                           (spline_test.maxTimeNanoseconds() - spline_test.minTimeNanoseconds()) / 2;
        
        try {
            Eigen::Vector3d interp_pos = spline_test.interpolatePosition(test_time);
            Eigen::Quaterniond interp_quat;
            spline_test.interpolateQuaternion(test_time, &interp_quat);
            
            std::cout << "  Interpolation at t=" << test_time / 1e9 << "s:" << std::endl;
            std::cout << "    pos=[" << interp_pos.transpose() << "]" << std::endl;
            std::cout << "    quat=[" << interp_quat.w() << ", " << interp_quat.x() 
                      << ", " << interp_quat.y() << ", " << interp_quat.z() << "]" << std::endl;
            
            std::cout << "  Basic spline operations: SUCCESS" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "  Basic spline operations: FAILED - " << e.what() << std::endl;
            return;
        }
        
        // Test very simple optimization (pose-only)
        testSimpleOptimization(poses);
        
        // Test IMU residuals separately
        testIMUResiduals();
    }
    
    void testSimpleOptimization(const std::vector<PoseData>& poses) {
        std::cout << "\n=== Testing Simple Pose-Only Optimization ===" << std::endl;
        
        if (spline_test.getNumKnots() < 2) {
            std::cout << "Not enough knots for optimization" << std::endl;
            return;
        }
        
        try {
            // Setup linearizer for pose-only optimization
            int num_knots = spline_test.getNumKnots();
            size_t bias_block_offset = Linearizer::POSE_SIZE * num_knots;
            size_t gravity_block_offset = bias_block_offset + Linearizer::BIAS_SIZE * num_knots;
            size_t hess_size = gravity_block_offset;
            
            bool pose_fixed = false;
            
            std::cout << "Linearizer setup: num_knots=" << num_knots << ", hess_size=" << hess_size << std::endl;
            
            // Create linearizer
            Linearizer lopt(bias_block_offset, gravity_block_offset, hess_size,
                           &spline_test, &param, pose_fixed);
            
            std::cout << "Linearizer created successfully" << std::endl;
            
            // Add only the first few poses to avoid complexity
            std::vector<PoseData> test_poses;
            for (int i = 0; i < std::min(3, (int)poses.size()); i++) {
                test_poses.push_back(poses[i]);
            }
            
            std::cout << "Adding " << test_poses.size() << " pose residuals..." << std::endl;
            
            // Convert to deque and add residuals
            Eigen::aligned_deque<PoseData> pose_deque(test_poses.begin(), test_poses.end());
            lopt(pose_deque);
            
            std::cout << "Pose residuals added successfully" << std::endl;
            std::cout << "Final cost: " << std::scientific << lopt.error << std::endl;
            
            if (std::isfinite(lopt.error)) {
                std::cout << "Simple optimization: SUCCESS" << std::endl;
            } else {
                std::cout << "Simple optimization: FAILED - non-finite cost" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cout << "Simple optimization: FAILED - " << e.what() << std::endl;
        } catch (...) {
            std::cout << "Simple optimization: FAILED - unknown exception" << std::endl;
        }
    }
    
    void testIMUResiduals() {
        std::cout << "\n=== Testing IMU Residuals ===" << std::endl;
        
        if (spline_test.getNumKnots() < 2) {
            std::cout << "Not enough knots for IMU residuals test" << std::endl;
            return;
        }
        
        try {
            // Create simple synthetic IMU data
            std::vector<ImuData> test_imu;
            
            // Add just one IMU measurement for testing
            int64_t test_time = spline_test.minTimeNanoseconds() + 
                               (spline_test.maxTimeNanoseconds() - spline_test.minTimeNanoseconds()) / 2;
            
            Eigen::Vector3d test_accel = param.gravity; // Just gravity
            test_accel.x() += 0.01; // Small perturbation
            
            Eigen::Vector3d test_gyro = Eigen::Vector3d::Zero(); // No rotation
            
            test_imu.emplace_back(test_time, test_gyro, test_accel);
            
            std::cout << "Created test IMU data: t=" << test_time / 1e9 << "s" << std::endl;
            std::cout << "  accel=[" << test_accel.transpose() << "]" << std::endl;
            std::cout << "  gyro=[" << test_gyro.transpose() << "]" << std::endl;
            
            // Setup linearizer
            int num_knots = spline_test.getNumKnots();
            size_t bias_block_offset = Linearizer::POSE_SIZE * num_knots;
            size_t gravity_block_offset = bias_block_offset + Linearizer::BIAS_SIZE * num_knots;
            size_t hess_size = gravity_block_offset;
            
            bool pose_fixed = false;
            
            std::cout << "Creating linearizer for IMU test..." << std::endl;
            
            Linearizer lopt(bias_block_offset, gravity_block_offset, hess_size,
                           &spline_test, &param, pose_fixed);
            
            std::cout << "Linearizer created, adding IMU residuals..." << std::endl;
            
            // Convert to deque and add IMU residuals
            Eigen::aligned_deque<ImuData> imu_deque(test_imu.begin(), test_imu.end());
            lopt(imu_deque);
            
            std::cout << "IMU residuals added successfully!" << std::endl;
            std::cout << "IMU cost: " << std::scientific << lopt.error << std::endl;
            
            if (std::isfinite(lopt.error)) {
                std::cout << "IMU residuals test: SUCCESS" << std::endl;
            } else {
                std::cout << "IMU residuals test: FAILED - non-finite cost" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cout << "IMU residuals test: FAILED - " << e.what() << std::endl;
        } catch (...) {
            std::cout << "IMU residuals test: FAILED - unknown exception (possible segfault)" << std::endl;
        }
    }
    
    void run() {
        std::cout << "=== Simple Bell Curve Test ===" << std::endl;
        std::cout << "Testing basic spline functionality with simple bell curve motion" << std::endl;
        
        generateBellCurveTrajectory();
        
        std::cout << "\n=== Test Complete ===" << std::endl;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "simple_bell_test");
    ros::NodeHandle nh;
    
    try {
        SimpleBellTest test;
        test.run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
} 