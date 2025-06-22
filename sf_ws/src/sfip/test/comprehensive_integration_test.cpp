#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <memory>
#include <cmath>

// Include standalone utility definitions
#include "standalone_utils.hpp"

// Include core components
#include "sfip/Types.hpp"
#include "sfip/config.hpp"

// Include core components without ROS dependencies
#include "sfip/SplineState.hpp"
#include "sfip/WindowManager.hpp"
#include "sfip/Optimizer.hpp"

namespace sfip {

// Tests for comprehensive integration of core components using Figure-8 trajectory
class ComprehensiveIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up trajectory generator with default parameters
        std::cout << "Setting up Figure-8 trajectory test data..." << std::endl;
        generateTestData();
    }

    void generateTestData() {
        // Create sample IMU and pose data using patterns similar to figure8_trajectory_test
        // This is simplified for testing core components
        std::mt19937 rng(42);  // Fixed seed for reproducibility
        
        // Create some basic test data over 5 seconds
        const double duration = 5.0;
        const double dt_imu = 0.01;  // 100Hz IMU
        const double dt_pose = 0.1;  // 10Hz pose
        
        // Generate a simple circular trajectory
        for (double t = 0.0; t < duration; t += dt_imu) {
            // Generate IMU data
            int64_t timestamp = static_cast<int64_t>(t * 1e9);
            
            // Simulate circular motion
            double angle = t * 2.0 * M_PI / 5.0;  // One circle every 5 seconds
            double radius = 2.0;
            
            // Accelerometer measures centripetal acceleration plus gravity
            double centripetal_acc = radius * std::pow(2.0 * M_PI / 5.0, 2);
            Eigen::Vector3d accel(
                centripetal_acc * std::cos(angle),
                centripetal_acc * std::sin(angle),
                -9.81
            );
            
            // Add some noise
            std::normal_distribution<double> accel_noise(0.0, 0.05);
            accel += Eigen::Vector3d(
                accel_noise(rng), accel_noise(rng), accel_noise(rng)
            );
            
            // Gyroscope measures angular velocity
            Eigen::Vector3d gyro(0.0, 0.0, 2.0 * M_PI / 5.0);
            
            // Add some noise
            std::normal_distribution<double> gyro_noise(0.0, 0.01);
            gyro += Eigen::Vector3d(
                gyro_noise(rng), gyro_noise(rng), gyro_noise(rng)
            );
            
            // Create and add IMU measurement
            ImuMeasurement imu_data(timestamp, gyro, accel);
            imu_data_.push_back(imu_data);
            
            // Generate pose data at lower frequency
            if (std::fmod(t, dt_pose) < dt_imu) {
                // Position on circle
                Eigen::Vector3d position(
                    radius * std::cos(angle),
                    radius * std::sin(angle),
                    0.0
                );
                
                // Add noise to position
                std::normal_distribution<double> pos_noise(0.0, 0.02);
                position += Eigen::Vector3d(
                    pos_noise(rng), pos_noise(rng), pos_noise(rng)
                );
                
                // Orientation
                Eigen::AngleAxisd rot(angle + M_PI/2.0, Eigen::Vector3d::UnitZ());
                Eigen::Quaterniond orientation(rot);
                
                // Add noise to orientation
                std::normal_distribution<double> orient_noise(0.0, 0.01);
                Eigen::AngleAxisd noise_rot(orient_noise(rng), Eigen::Vector3d::Random().normalized());
                orientation = Eigen::Quaterniond(noise_rot) * orientation;
                orientation.normalize();
                
                // Create and add pose measurement
                PoseMeasurement pose_data(timestamp, orientation, position);
                pose_data_.push_back(pose_data);
            }
        }
        
        // Create measurement queues
        for (const auto& imu : imu_data_) {
            imu_queue_.push_back(imu);
        }
        
        for (const auto& pose : pose_data_) {
            pose_queue_.push_back(pose);
        }
        
        std::cout << "Generated " << imu_data_.size() << " IMU and " 
                 << pose_data_.size() << " pose measurements" << std::endl;
    }

    void setupComponents() {
        // Configure parameters
        fusion_params_.windowSize = 10;
        fusion_params_.controlPointFps = 10;
        fusion_params_.knotIntervalNanoseconds = 1e8;  // 0.1 second (10 Hz)
        fusion_params_.maxIterations = 10;
        fusion_params_.weightPosePosition = 1.0;
        fusion_params_.weightPoseOrientation = 1.0;
        fusion_params_.weightAccel = 1.0;
        fusion_params_.weightGyro = 1.0;
        
        calib_params_.gravity = Eigen::Vector3d(0, 0, -9.81);
        
        // Create components
        spline_state_ = std::make_shared<SplineState>();
        window_manager_ = std::make_unique<WindowManager>(fusion_params_);
        optimizer_ = std::make_unique<Optimizer>(fusion_params_);
    }
    
    // Test data
    std::vector<ImuMeasurement, Eigen::aligned_allocator<ImuMeasurement>> imu_data_;
    std::vector<PoseMeasurement, Eigen::aligned_allocator<PoseMeasurement>> pose_data_;
    
    // System queues
    ImuMeasurementDeque imu_queue_;
    PoseMeasurementDeque pose_queue_;
    
    // System components
    std::shared_ptr<SplineState> spline_state_;
    std::unique_ptr<WindowManager> window_manager_;
    std::unique_ptr<Optimizer> optimizer_;
    
    // Parameters
    FusionParameters fusion_params_;
    CalibrationParameters calib_params_;
};

// Test SplineState initialization and interpolation
TEST_F(ComprehensiveIntegrationTest, SplineStateBasics) {
    std::cout << "\n[Test 1] Testing SplineState basics..." << std::endl;
    
    // Initialize spline with test data
    setupComponents();
    int64_t start_time = pose_queue_.front().timestampNanoseconds;
    
    spline_state_->init(fusion_params_.knotIntervalNanoseconds, 0, start_time);
    
    // Add two knots
    Eigen::Matrix<double, 6, 1> zero_bias = Eigen::Matrix<double, 6, 1>::Zero();
    spline_state_->addSingleStateKnot(pose_queue_.front().orientation, 
                                     pose_queue_.front().position, 
                                     zero_bias);
    
    int64_t second_knot_time = start_time + fusion_params_.knotIntervalNanoseconds;
    spline_state_->addSingleStateKnot(pose_queue_.front().orientation, 
                                     pose_queue_.front().position, 
                                     zero_bias);
    
    // Test interpolation
    int64_t test_time = start_time + fusion_params_.knotIntervalNanoseconds / 2;
    Eigen::Vector3d position = spline_state_->interpolatePosition(test_time);
    Eigen::Quaterniond orientation;
    spline_state_->interpolateQuaternion(test_time, &orientation);
    
    std::cout << "✓ Spline initialized with " << spline_state_->getNumKnots() << " knots" << std::endl;
    std::cout << "✓ Start time: " << spline_state_->minTimeNanoseconds() << std::endl;
    std::cout << "✓ End time: " << spline_state_->maxTimeNanoseconds() << std::endl;
    std::cout << "✓ Interpolated position: " << position.transpose() << std::endl;
    std::cout << "✓ Interpolated orientation: [" << orientation.w() << ", " 
              << orientation.x() << ", " << orientation.y() << ", " << orientation.z() << "]" << std::endl;
    
    // Basic checks
    EXPECT_EQ(spline_state_->getNumKnots(), 2);
    EXPECT_NEAR(position.norm(), pose_queue_.front().position.norm(), 1e-5);
    EXPECT_NEAR(orientation.norm(), 1.0, 1e-5);
}

// Test window manager functionality
TEST_F(ComprehensiveIntegrationTest, WindowManagerFunctionality) {
    std::cout << "\n[Test 2] Testing WindowManager functionality..." << std::endl;
    
    // Initialize components
    setupComponents();
    int64_t start_time = pose_queue_.front().timestampNanoseconds;
    
    // Initialize window manager
    bool init_success = window_manager_->initialize(spline_state_, start_time);
    EXPECT_TRUE(init_success);
    std::cout << "✓ Window manager initialized successfully" << std::endl;
    
    // Add knots and test sliding window
    int knots_added = 0;
    for (int i = 0; i < 15; i++) {
        int64_t knot_time = start_time + i * fusion_params_.knotIntervalNanoseconds;
        
        if (window_manager_->canAddKnot(knot_time)) {
            Eigen::Matrix<double, 6, 1> zero_bias = Eigen::Matrix<double, 6, 1>::Zero();
            Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
            Eigen::Vector3d position(i * 0.1, 0, 0);  // Simple position increment
            
            bool add_success = window_manager_->addKnot(orientation, position, zero_bias);
            if (add_success) {
                knots_added++;
            }
        }
        
        // Periodically slide window
        if (i >= 10) {
            window_manager_->slideWindow();
        }
    }
    
    std::cout << "✓ Added " << knots_added << " knots to window" << std::endl;
    std::cout << "✓ Current spline has " << spline_state_->getNumKnots() << " knots" << std::endl;
    
    // Test measurement window updates
    ImuMeasurementDeque imu_window;
    PoseMeasurementDeque pose_window;
    bool update_success = window_manager_->updateMeasurementWindows(
        imu_window, pose_window, imu_queue_, pose_queue_);
    
    std::cout << "✓ Window update success: " << (update_success ? "true" : "false") << std::endl;
    std::cout << "✓ IMU measurements in window: " << imu_window.size() << std::endl;
    std::cout << "✓ Pose measurements in window: " << pose_window.size() << std::endl;
    
    EXPECT_GT(knots_added, 0);
    EXPECT_GT(imu_window.size(), 0);
    EXPECT_GT(pose_window.size(), 0);
}

// Test optimizer functionality
TEST_F(ComprehensiveIntegrationTest, OptimizerFunctionality) {
    std::cout << "\n[Test 3] Testing Optimizer functionality..." << std::endl;
    
    // Initialize components
    setupComponents();
    int64_t start_time = pose_queue_.front().timestampNanoseconds;
    
    // Initialize window manager and add some knots
    window_manager_->initialize(spline_state_, start_time);
    
    // Add several knots
    for (int i = 0; i < 5; i++) {
        if (window_manager_->canAddKnot(start_time + i * fusion_params_.knotIntervalNanoseconds)) {
            Eigen::Matrix<double, 6, 1> zero_bias = Eigen::Matrix<double, 6, 1>::Zero();
            Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
            Eigen::Vector3d position(i * 0.1, 0, 0);
            
            window_manager_->addKnot(orientation, position, zero_bias);
        }
    }
    
    // Get measurement windows
    ImuMeasurementDeque imu_window;
    PoseMeasurementDeque pose_window;
    window_manager_->updateMeasurementWindows(imu_window, pose_window, imu_queue_, pose_queue_);
    
    // Setup optimizer
    WindowState window_state = window_manager_->getWindowState();
    bool setup_success = optimizer_->setup(spline_state_, calib_params_, window_state);
    EXPECT_TRUE(setup_success);
    
    // Calculate initial error
    double initial_error = optimizer_->calculateError(imu_window, pose_window);
    std::cout << "✓ Initial optimization error: " << initial_error << std::endl;
    
    // Run optimization
    OptimizationState opt_state;
    OptimizationResult result = optimizer_->optimize(imu_window, pose_window, opt_state);
    
    // Report results
    std::cout << "✓ Optimization completed with " << opt_state.iterations << " iterations" << std::endl;
    std::cout << "✓ Initial cost: " << opt_state.previousError << std::endl;
    std::cout << "✓ Final cost: " << opt_state.currentError << std::endl;
    std::cout << "✓ Cost reduction: " << (1.0 - opt_state.currentError / opt_state.previousError) * 100.0 << "%" << std::endl;
    std::cout << "✓ Converged: " << (opt_state.converged ? "Yes" : "No") << std::endl;
    
    EXPECT_TRUE(result.converged);
    EXPECT_GT(opt_state.previousError, opt_state.currentError);
}

// Test the combined pipeline
TEST_F(ComprehensiveIntegrationTest, EndToEndPipeline) {
    std::cout << "\n[Test 4] Testing end-to-end estimation pipeline..." << std::endl;
    
    // Initialize components
    setupComponents();
    int64_t start_time = pose_queue_.front().timestampNanoseconds;
    
    // Initialize window
    window_manager_->initialize(spline_state_, start_time);
    
    // Process measurements in batches to simulate real-time operation
    int num_batches = 5;
    int imu_per_batch = imu_queue_.size() / num_batches;
    int pose_per_batch = pose_queue_.size() / num_batches;
    
    for (int batch = 0; batch < num_batches; batch++) {
        std::cout << "\nProcessing batch " << (batch + 1) << "/" << num_batches << std::endl;
        
        // Process IMU measurements
        for (int i = 0; i < imu_per_batch; i++) {
            if (imu_queue_.empty()) break;
            
            // Process IMU for prediction
            const ImuMeasurement& imu = imu_queue_.front();
            int64_t imu_time = imu.timestampNanoseconds;
            
            // Add knots as needed
            if (window_manager_->canAddKnot(imu_time)) {
                Eigen::Matrix<double, 6, 1> zero_bias = Eigen::Matrix<double, 6, 1>::Zero();
                Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
                Eigen::Vector3d position(0, 0, 0);
                
                // In a real system, we would predict position/orientation
                // For this test, we're just adding simple knots
                window_manager_->addKnot(orientation, position, zero_bias);
            }
            
            imu_queue_.pop_front();
        }
        
        // Process pose measurements
        for (int i = 0; i < pose_per_batch; i++) {
            if (pose_queue_.empty()) break;
            
            // In a real system, we would update the state here
            pose_queue_.pop_front();
        }
        
        // Run optimization
        ImuMeasurementDeque imu_window;
        PoseMeasurementDeque pose_window;
        window_manager_->updateMeasurementWindows(imu_window, pose_window, imu_queue_, pose_queue_);
        
        // Setup optimizer
        WindowState window_state = window_manager_->getWindowState();
        optimizer_->setup(spline_state_, calib_params_, window_state);
        
        // Run optimization
        OptimizationState opt_state;
        OptimizationResult result = optimizer_->optimize(imu_window, pose_window, opt_state);
        
        std::cout << "  Optimization result: " << (result.converged ? "CONVERGED" : "FAILED") << std::endl;
        std::cout << "  Cost reduction: " << (1.0 - opt_state.currentError / opt_state.previousError) * 100.0 << "%" << std::endl;
        
        // Slide window
        window_manager_->slideWindow();
    }
    
    // Final verification
    std::cout << "\n✓ Completed end-to-end pipeline test" << std::endl;
    std::cout << "✓ Final spline has " << spline_state_->getNumKnots() << " knots" << std::endl;
    
    EXPECT_GT(spline_state_->getNumKnots(), 0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "=====================================================\n";
    std::cout << "Comprehensive SFIP Integration Test Suite\n";
    std::cout << "Testing core estimation components with synthetic data\n";
    std::cout << "=====================================================\n";
    
    return RUN_ALL_TESTS();
}

} // namespace sfip
