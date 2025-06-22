#include <iostream>
#include <vector>
#include <fstream>

// Include our system components
#include "../include/SplineState.hpp"
#include "../include/Linearizer.hpp"
#include "../include/Residuals.hpp"
#include "../include/utils/common_utils.hpp"

class SimpleIntegrationTest {
public:
    SimpleIntegrationTest() {
        std::cout << "=== SFIP Simple Integration Test ===" << std::endl;
        std::cout << "Testing core system components" << std::endl;
    }

    bool testSplineBasics() {
        std::cout << "\n[Test 1] Testing SplineState basics..." << std::endl;
        
        try {
            SplineState spline;
            int64_t dt_ns = 1e9 / 30; // 30 Hz control points
            int64_t start_time = 1738770124000000000LL; // Example timestamp
            spline.init(dt_ns, 0, start_time);
            
            // Add test knots
            Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
            Eigen::Vector3d pos(1.0, 2.0, 3.0);
            Eigen::Matrix<double, 6, 1> bias = Eigen::Matrix<double, 6, 1>::Zero();
            
            spline.addSingleStateKnot(q, pos, bias);
            spline.addSingleStateKnot(q, pos, bias);
            
            std::cout << "✓ Spline initialized with " << spline.getNumKnots() << " knots" << std::endl;
            std::cout << "✓ Start time: " << spline.minTimeNanoseconds() << std::endl;
            std::cout << "✓ End time: " << spline.maxTimeNanoseconds() << std::endl;
            
            // Test interpolation
            int64_t test_time = start_time + dt_ns / 2;
            Eigen::Vector3d interp_pos = spline.interpolatePosition(test_time);
            Eigen::Quaterniond interp_quat;
            spline.interpolateQuaternion(test_time, &interp_quat);
            
            std::cout << "✓ Interpolated pos: " << interp_pos.transpose() << std::endl;
            std::cout << "✓ Interpolated quat: [" << interp_quat.w() << ", " << interp_quat.x() 
                      << ", " << interp_quat.y() << ", " << interp_quat.z() << "]" << std::endl;
            
            // Test bounds checking
            if (interp_pos.norm() > 0 && std::isfinite(interp_quat.norm())) {
                std::cout << "✓ Interpolation produces finite results" << std::endl;
                return true;
            } else {
                std::cout << "✗ Interpolation produces invalid results" << std::endl;
                return false;
            }
            
        } catch (const std::exception& e) {
            std::cout << "✗ Spline test failed: " << e.what() << std::endl;
            return false;
        }
    }

    bool testResidualComputation() {
        std::cout << "\n[Test 2] Testing residual computation..." << std::endl;
        
        try {
            // Set up spline with test data
            SplineState spline;
            int64_t dt_ns = 1e9 / 30;
            int64_t start_time = 1738770124000000000LL;
            spline.init(dt_ns, 0, start_time);
            
            Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
            Eigen::Vector3d pos(0.0, 0.0, 0.0);
            Eigen::Matrix<double, 6, 1> bias = Eigen::Matrix<double, 6, 1>::Zero();
            
            spline.addSingleStateKnot(q, pos, bias);
            spline.addSingleStateKnot(q, pos, bias);
            
            // Test pose residual
            int64_t test_time = start_time + dt_ns / 4;
            Eigen::Matrix<double, 6, 1> pose_residual = Residuals::poseResidual(
                test_time, &spline, pos, q);
            
            std::cout << "✓ Pose residual computed: " << pose_residual.transpose() << std::endl;
            std::cout << "✓ Pose residual norm: " << pose_residual.norm() << std::endl;
            
            if (pose_residual.norm() > 1e10) {
                std::cout << "⚠ Warning: Very large pose residual: " << pose_residual.norm() << std::endl;
                return false;
            }
            
            // Test IMU residual
            Eigen::Vector3d accel(0.0, 0.0, -9.81); // Stationary accelerometer
            Eigen::Vector3d gyro(0.0, 0.0, 0.0);    // No rotation
            Eigen::Vector3d gravity(0, 0, -9.81);
            
            Eigen::Matrix<double, 6, 1> imu_residual = Residuals::imuResidual(
                test_time, &spline, &accel, &gyro, gravity);
            
            std::cout << "✓ IMU residual computed: " << imu_residual.transpose() << std::endl;
            std::cout << "✓ IMU residual norm: " << imu_residual.norm() << std::endl;
            
            if (imu_residual.norm() > 1e10) {
                std::cout << "⚠ Warning: Very large IMU residual: " << imu_residual.norm() << std::endl;
                return false;
            }
            
            return true;
            
        } catch (const std::exception& e) {
            std::cout << "✗ Residual computation failed: " << e.what() << std::endl;
            return false;
        }
    }

    void runAllTests() {
        std::vector<bool> results;
        std::vector<std::string> test_names = {
            "Spline Basics", "Residual Computation"
        };
        
        results.push_back(testSplineBasics());
        results.push_back(testResidualComputation());
        
        std::cout << "\n=== INTEGRATION TEST REPORT ===" << std::endl;
        int passed = 0;
        for (size_t i = 0; i < results.size(); i++) {
            std::cout << test_names[i] << ": " << (results[i] ? "PASS" : "FAIL") << std::endl;
            if (results[i]) passed++;
        }
        
        std::cout << "\nSummary: " << passed << "/" << results.size() << " tests passed" << std::endl;
        
        if (passed == results.size()) {
            std::cout << "\n✓ Core components working! The crash likely stems from:" << std::endl;
            std::cout << "  1. ROS message handling or TF lookups" << std::endl;
            std::cout << "  2. Timing/initialization sequence" << std::endl;
            std::cout << "  3. Memory management in the main loop" << std::endl;
        } else {
            std::cout << "\n⚠ Some core tests failed. Issue is in fundamental math components." << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    SimpleIntegrationTest test;
    test.runAllTests();
    return 0;
} 