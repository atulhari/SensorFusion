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

class SplineTestStandalone {
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
    
    // Simplified test parameters - shorter duration, simpler motion
    static constexpr int64_t DURATION_SEC = 10;  // Reduced from 100 to 10 seconds
    static constexpr int64_t NS_PER_SEC = 1000000000LL;
    static constexpr double POSE_FREQ = 20.0;  // 20 Hz pose measurements
    static constexpr double IMU_FREQ = 100.0;  // 100 Hz IMU
    static constexpr double CONTROL_FREQ = 60.0; // 60 Hz control points
    static constexpr int WINDOW_SIZE = 5;      // Reduced window size
    
    std::mt19937 rng;
    std::normal_distribution<double> pose_noise;
    std::normal_distribution<double> imu_noise;

public:
    SplineTestStandalone() : rng(42), pose_noise(0.0, 0.001), imu_noise(0.0, 0.01) {
        setupParameters();
    }
    
    void setupParameters() {
        // Setup optimization parameters similar to the actual system
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
        param.if_opt_g = false; // Keep gravity fixed for this test
        
        std::cout << "Simplified test parameters configured:" << std::endl;
        std::cout << "  Pose weights: pos=" << param.w_pose_pos << ", rot=" << param.w_pose_rot << std::endl;
        std::cout << "  IMU weights: acc=" << param.w_acc << ", gyro=" << param.w_gyro << std::endl;
        std::cout << "  Motion: X-axis bell curve only, Y=0, Z=0, no rotation" << std::endl;
    }
    
    void generateGroundTruthTrajectory() {
        std::cout << "\n=== Generating Simplified Ground Truth Trajectory ===" << std::endl;
        
        ground_truth_poses.clear();
        
        int64_t start_time = 0;
        double dt_pose = 1.0 / POSE_FREQ;
        int num_poses = static_cast<int>(DURATION_SEC * POSE_FREQ);
        
        for (int i = 0; i < num_poses; i++) {
            double t_sec = i * dt_pose;
            int64_t t_ns = start_time + static_cast<int64_t>(t_sec * NS_PER_SEC);
            
            // SIMPLIFIED: Bell curve motion ONLY in X-axis
            double progress = t_sec / DURATION_SEC; // 0 to 1
            double bell_factor = std::exp(-std::pow((progress - 0.5) * 4, 2)); // Bell curve centered at 0.5
            
            Eigen::Vector3d position;
            position.x() = 2.0 * bell_factor;  // Bell curve motion in X only
            position.y() = 0.0;                // Always zero
            position.z() = 0.0;                // Always zero
            
            // SIMPLIFIED: No rotation - always identity quaternion
            Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
            
            ground_truth_poses.emplace_back(t_ns, position, orientation);
        }
        
        std::cout << "Generated " << ground_truth_poses.size() << " ground truth poses over " 
                  << DURATION_SEC << " seconds" << std::endl;
        
        // Print first few poses for verification
        for (int i = 0; i < std::min(5, (int)ground_truth_poses.size()); i++) {
            const auto& pose = ground_truth_poses[i];
            std::cout << "  GT[" << i << "]: t=" << pose.timestampNanoseconds / 1e9 << "s, "
                      << "pos=[" << std::fixed << std::setprecision(3)
                      << pose.position.x() << ", " << pose.position.y() << ", " << pose.position.z() << "], "
                      << "quat=[" << pose.orientation.w() << ", " << pose.orientation.x() 
                      << ", " << pose.orientation.y() << ", " << pose.orientation.z() << "]" << std::endl;
        }
        
        // Print some middle and end poses to show the bell curve
        std::cout << "  Bell curve progression:" << std::endl;
        for (int i : {0, num_poses/4, num_poses/2, 3*num_poses/4, num_poses-1}) {
            if (i < ground_truth_poses.size()) {
                const auto& pose = ground_truth_poses[i];
                std::cout << "    t=" << std::fixed << std::setprecision(2) << pose.timestampNanoseconds / 1e9 
                          << "s: x=" << std::setprecision(4) << pose.position.x() << std::endl;
            }
        }
    }
    
    void generateNoisyMeasurements() {
        std::cout << "\n=== Generating Noisy Measurements ===" << std::endl;
        
        noisy_pose_measurements.clear();
        
        // Add minimal noise to ground truth poses
        for (const auto& gt_pose : ground_truth_poses) {
            Eigen::Vector3d noisy_pos = gt_pose.position;
            noisy_pos.x() += pose_noise(rng);  // Small noise in X
            noisy_pos.y() += pose_noise(rng);  // Small noise in Y (should stay ~0)
            noisy_pos.z() += pose_noise(rng);  // Small noise in Z (should stay ~0)
            
            // Keep orientation as identity (no rotation noise)
            Eigen::Quaterniond noisy_orient = Eigen::Quaterniond::Identity();
            
            noisy_pose_measurements.emplace_back(gt_pose.timestampNanoseconds, noisy_pos, noisy_orient);
        }
        
        std::cout << "Generated " << noisy_pose_measurements.size() << " noisy pose measurements" << std::endl;
        
        // Generate simplified synthetic IMU data
        synthetic_imu.clear();
        double dt_imu = 1.0 / IMU_FREQ;
        int num_imu = static_cast<int>(DURATION_SEC * IMU_FREQ);
        
        for (int i = 0; i < num_imu; i++) {
            double t_sec = i * dt_imu;
            int64_t t_ns = static_cast<int64_t>(t_sec * NS_PER_SEC);
            
            // SIMPLIFIED IMU: Just gravity + motion acceleration in X
            Eigen::Vector3d accel = param.gravity; // Start with gravity
            
            // Add X-axis motion acceleration (second derivative of bell curve)
            if (i > 0 && i < num_imu - 1) {
                double progress = t_sec / DURATION_SEC;
                double bell_factor = std::exp(-std::pow((progress - 0.5) * 4, 2));
                
                // Approximate second derivative of bell curve motion
                double d2x_dt2 = 2.0 * bell_factor * (32.0 * std::pow(progress - 0.5, 2) - 16.0) / (DURATION_SEC * DURATION_SEC);
                accel.x() += d2x_dt2;
            }
            
            // Add small noise
            accel.x() += imu_noise(rng);
            accel.y() += imu_noise(rng);
            accel.z() += imu_noise(rng);
            
            // SIMPLIFIED: No rotation, so gyro should be zero
            Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
            gyro.x() += imu_noise(rng);
            gyro.y() += imu_noise(rng);
            gyro.z() += imu_noise(rng);
            
            synthetic_imu.emplace_back(t_ns, accel, gyro);
        }
        
        std::cout << "Generated " << synthetic_imu.size() << " synthetic IMU measurements" << std::endl;
        
        // Print first few IMU measurements
        for (int i = 0; i < std::min(3, (int)synthetic_imu.size()); i++) {
            const auto& imu = synthetic_imu[i];
            std::cout << "  IMU[" << i << "]: t=" << imu.timestampNanoseconds / 1e9 << "s, "
                      << "accel=[" << imu.accel.x() << ", " << imu.accel.y() << ", " << imu.accel.z() << "], "
                      << "gyro=[" << imu.gyro.x() << ", " << imu.gyro.y() << ", " << imu.gyro.z() << "]" << std::endl;
        }
    }
    
    Eigen::Vector3d interpolateGroundTruthPosition(double t_sec) {
        if (t_sec < 0) t_sec = 0;
        if (t_sec > DURATION_SEC) t_sec = DURATION_SEC;
        
        double progress = t_sec / DURATION_SEC;
        double bell_factor = std::exp(-std::pow((progress - 0.5) * 4, 2));
        
        Eigen::Vector3d position;
        position.x() = 2.0 * bell_factor;
        position.y() = 0.0;
        position.z() = 0.0;
        return position;
    }
    
    Eigen::Quaterniond interpolateGroundTruthOrientation(double t_sec) {
        // Always identity - no rotation
        return Eigen::Quaterniond::Identity();
    }
    
    void testSplineOptimization() {
        std::cout << "\n=== Testing Simplified Spline Optimization ===" << std::endl;
        
        // Initialize spline
        int64_t dt_ns = static_cast<int64_t>(1e9 / CONTROL_FREQ);
        spline_test.init(dt_ns, 0, 0);
        
        std::cout << "Spline initialized with dt_ns=" << dt_ns << " (control_freq=" << CONTROL_FREQ << " Hz)" << std::endl;
        
        // Initialize with first pose (anchor the spline)
        if (!noisy_pose_measurements.empty()) {
            const auto& first_pose = noisy_pose_measurements[0];
            Eigen::Matrix<double, 6, 1> zero_bias = Eigen::Matrix<double, 6, 1>::Zero();
            
            // Add two initial knots for stability
            spline_test.addSingleStateKnot(first_pose.orientation, first_pose.position, zero_bias);
            spline_test.addSingleStateKnot(first_pose.orientation, first_pose.position, zero_bias);
            
            std::cout << "Initialized spline with first pose: pos=[" << first_pose.position.transpose() 
                      << "], quat=[" << first_pose.orientation.w() << ", " << first_pose.orientation.x() 
                      << ", " << first_pose.orientation.y() << ", " << first_pose.orientation.z() << "]" << std::endl;
        }
        
        // Process sliding windows
        double window_duration = 1.0; // 1 second windows
        int window_count = 0;
        
        for (double window_start = 0; window_start < DURATION_SEC - window_duration; window_start += window_duration) {
            double window_end = window_start + window_duration;
            
            std::cout << "\n--- Window " << window_count << ": [" << std::fixed << std::setprecision(2) 
                      << window_start << "s, " << window_end << "s] ---" << std::endl;
            
            // Collect measurements in this window
            std::vector<PoseData> window_poses;
            std::vector<ImuData> window_imu;
            
            // Get pose measurements in window
            for (const auto& pose : noisy_pose_measurements) {
                double t_sec = pose.timestampNanoseconds / 1e9;
                if (t_sec >= window_start && t_sec <= window_end) {
                    window_poses.emplace_back(pose.timestampNanoseconds, pose.orientation, pose.position);
                }
            }
            
            // Get IMU measurements in window
            for (const auto& imu : synthetic_imu) {
                double t_sec = imu.timestampNanoseconds / 1e9;
                if (t_sec >= window_start && t_sec <= window_end) {
                    window_imu.emplace_back(imu.timestampNanoseconds, imu.gyro, imu.accel);
                }
            }
            
            std::cout << "Window contains " << window_poses.size() << " poses, " << window_imu.size() << " IMU measurements" << std::endl;
            
            // Add new knot if needed
            int64_t next_knot_time = static_cast<int64_t>((window_start + window_duration) * NS_PER_SEC);
            if (spline_test.getNumKnots() == 0 || next_knot_time > spline_test.maxTimeNanoseconds()) {
                // Predict next state - simple constant velocity model
                Eigen::Quaterniond next_q = Eigen::Quaterniond::Identity(); // No rotation
                Eigen::Vector3d next_p = Eigen::Vector3d::Zero();
                if (spline_test.getNumKnots() > 0) {
                    next_p = spline_test.getKnotPosition(spline_test.getNumKnots()-1);
                }
                Eigen::Matrix<double, 6, 1> next_bias = Eigen::Matrix<double, 6, 1>::Zero();
                
                spline_test.addSingleStateKnot(next_q, next_p, next_bias);
                std::cout << "Added knot #" << spline_test.getNumKnots()-1 << " at time " << next_knot_time / 1e9 << "s" << std::endl;
            }
            
            // Optimize if we have measurements
            if (!window_poses.empty() || !window_imu.empty()) {
                bool converged = optimizeWindow(window_poses, window_imu, window_count);
                std::cout << "Optimization " << (converged ? "converged" : "did not converge") << std::endl;
            }
            
            // Remove old knots to maintain sliding window
            if (spline_test.getNumKnots() > WINDOW_SIZE) {
                spline_test.removeSingleOldState();
                std::cout << "Removed old knot, now have " << spline_test.getNumKnots() << " knots" << std::endl;
            }
            
            // Print current state
            if (spline_test.getNumKnots() > 0) {
                size_t last_idx = spline_test.getNumKnots() - 1;
                Eigen::Vector3d pos = spline_test.getKnotPosition(last_idx);
                Eigen::Quaterniond quat = spline_test.getKnotOrientation(last_idx);
                std::cout << "Latest knot state: pos=[" << pos.transpose() << "], quat=["
                          << quat.w() << ", " << quat.x() << ", " << quat.y() << ", " << quat.z() << "]" << std::endl;
                
                // Compare with ground truth at this time
                double gt_time = window_end;
                Eigen::Vector3d gt_pos = interpolateGroundTruthPosition(gt_time);
                Eigen::Quaterniond gt_quat = interpolateGroundTruthOrientation(gt_time);
                
                Eigen::Vector3d pos_error = pos - gt_pos;
                double orient_error = 2.0 * std::acos(std::abs(quat.dot(gt_quat))); // angle between quaternions
                
                std::cout << "Ground truth: pos=[" << gt_pos.transpose() << "], quat=["
                          << gt_quat.w() << ", " << gt_quat.x() << ", " << gt_quat.y() << ", " << gt_quat.z() << "]" << std::endl;
                std::cout << "Error: pos_norm=" << pos_error.norm() << "m, orient=" << orient_error * 180/M_PI << "deg" << std::endl;
            }
            
            window_count++;
            if (window_count >= 5) break; // Limit to first 5 windows for debugging
        }
    }
    
    bool optimizeWindow(const std::vector<PoseData>& poses, const std::vector<ImuData>& imu_data, int window_id) {
        if (spline_test.getNumKnots() < 2) return false;
        
        // Setup linearizer
        int num_knots = spline_test.getNumKnots();
        size_t bias_block_offset = Linearizer::POSE_SIZE * num_knots;
        size_t gravity_block_offset = bias_block_offset + Linearizer::BIAS_SIZE * num_knots;
        size_t hess_size = gravity_block_offset;
        if (param.if_opt_g) hess_size += Linearizer::G_SIZE;
        
        bool pose_fixed = false; // Don't fix poses in this test
        
        std::cout << "  Optimizing: num_knots=" << num_knots << ", hess_size=" << hess_size << std::endl;
        
        // Run optimization iterations
        double lambda = 1e-6;
        double lambda_vee = 2.0;
        int max_iter = 5;  // Reduced iterations for debugging
        
        for (int iter = 0; iter < max_iter; iter++) {
            std::cout << "    Starting iteration " << iter << std::endl;
            
            try {
                Linearizer lopt(bias_block_offset, gravity_block_offset, hess_size,
                               &spline_test, &param, pose_fixed);
                
                // Add IMU residuals
                if (!imu_data.empty()) {
                    std::cout << "      Adding " << imu_data.size() << " IMU residuals" << std::endl;
                    Eigen::aligned_deque<ImuData> imu_deque(imu_data.begin(), imu_data.end());
                    lopt(imu_deque);
                }
                
                // Add pose residuals
                if (!poses.empty()) {
                    std::cout << "      Adding " << poses.size() << " pose residuals" << std::endl;
                    Eigen::aligned_deque<PoseData> pose_deque(poses.begin(), poses.end());
                    lopt(pose_deque);
                }
                
                std::cout << "    Iter " << iter << ": cost=" << std::scientific << lopt.error 
                          << ", lambda=" << lambda << std::endl;
                
                // Check for numerical issues
                if (!std::isfinite(lopt.error) || lopt.error > 1e10) {
                    std::cout << "    Cost is not finite or too large, stopping" << std::endl;
                    break;
                }
                
                // Check for convergence
                if (iter > 0) {
                    double gradient_norm = lopt.accum.getB().norm();
                    std::cout << "    Gradient norm: " << gradient_norm << std::endl;
                    if (gradient_norm < 1e-6) {
                        std::cout << "    Converged (gradient_norm=" << gradient_norm << ")" << std::endl;
                        return true;
                    }
                }
                
                // Solve for increment
                std::cout << "      Setting up solver" << std::endl;
                lopt.accum.setup_solver();
                Eigen::VectorXd Hdiag = lopt.accum.Hdiagonal();
                Eigen::VectorXd Hdiag_lambda = Hdiag * lambda;
                for (int i = 0; i < Hdiag_lambda.size(); i++) {
                    Hdiag_lambda[i] = std::max(Hdiag_lambda[i], 1e-12);
                }
                
                std::cout << "      Solving for increment" << std::endl;
                Eigen::VectorXd inc = -lopt.accum.solve(&Hdiag_lambda);
                
                std::cout << "      Increment norm: " << inc.norm() << std::endl;
                if (!std::isfinite(inc.norm()) || inc.norm() > 10.0) {
                    std::cout << "    Large or invalid increment: " << inc.norm() << ", stopping" << std::endl;
                    break;
                }
                
                // Apply increment
                std::cout << "      Applying increment" << std::endl;
                applyIncrement(inc);
                
                // Simple acceptance (no step quality check for now)
                lambda = std::max(1e-12, lambda / 1.5);
                
            } catch (const std::exception& e) {
                std::cout << "    Exception in iteration " << iter << ": " << e.what() << std::endl;
                break;
            } catch (...) {
                std::cout << "    Unknown exception in iteration " << iter << std::endl;
                break;
            }
        }
        
        return false; // Didn't converge within max_iter
    }
    
    void applyIncrement(const Eigen::VectorXd& inc) {
        size_t num_knots = spline_test.getNumKnots();
        
        std::cout << "        Applying pose increments for " << num_knots << " knots" << std::endl;
        
        // Apply pose increments
        for (size_t i = 0; i < num_knots; i++) {
            if (inc.size() >= (i+1) * Linearizer::POSE_SIZE) {
                Eigen::Matrix<double, 6, 1> pose_inc = 
                    inc.segment<Linearizer::POSE_SIZE>(i * Linearizer::POSE_SIZE);
                spline_test.applyPoseIncrement(i, pose_inc);
            }
        }
        
        std::cout << "        Applying bias increments" << std::endl;
        
        // Apply bias increments
        size_t bias_offset = Linearizer::POSE_SIZE * num_knots;
        for (size_t i = 0; i < num_knots; i++) {
            if (inc.size() >= bias_offset + (i+1) * Linearizer::BIAS_SIZE) {
                Eigen::Matrix<double, 6, 1> bias_inc = 
                    inc.segment<Linearizer::BIAS_SIZE>(bias_offset + i * Linearizer::BIAS_SIZE);
                spline_test.applyBiasIncrement(i, bias_inc);
            }
        }
        
        std::cout << "        Checking quaternion control points" << std::endl;
        spline_test.checkQuaternionControlPoints();
        std::cout << "        Increment application complete" << std::endl;
    }
    
    void run() {
        std::cout << "=== Simplified Spline Fusion Standalone Test ===" << std::endl;
        std::cout << "Duration: " << DURATION_SEC << " seconds" << std::endl;
        std::cout << "Pose frequency: " << POSE_FREQ << " Hz" << std::endl;
        std::cout << "IMU frequency: " << IMU_FREQ << " Hz" << std::endl;
        std::cout << "Control point frequency: " << CONTROL_FREQ << " Hz" << std::endl;
        std::cout << "Window size: " << WINDOW_SIZE << " knots" << std::endl;
        std::cout << "Motion: Bell curve in X-axis only (Y=0, Z=0, no rotation)" << std::endl;
        
        generateGroundTruthTrajectory();
        generateNoisyMeasurements();
        testSplineOptimization();
        
        std::cout << "\n=== Simplified Test Complete ===" << std::endl;
    }
};

int main(int argc, char** argv) {
    // Initialize ROS for the test (needed for the headers)
    ros::init(argc, argv, "spline_test_standalone");
    ros::NodeHandle nh;
    
    try {
        SplineTestStandalone test;
        test.run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
} 