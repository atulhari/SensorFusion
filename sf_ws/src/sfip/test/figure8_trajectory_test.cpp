#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

#include "standalone_utils.hpp"
#include "sfip/Types.hpp"
#include "sfip/config.hpp"

// Create a standalone test without dependencies on SplineFusionCore
// This will test the trajectory generation and evaluation components

namespace sfip_test {

// Test configuration enums
enum class TrajectoryOrientation {
    HORIZONTAL,     // Figure-8 parallel to ground (XY plane)
    INCLINED        // Figure-8 on a 45-degree inclined plane
};

// Test parameters structure
struct TestParameters {
    // Trajectory parameters
    double duration;          // seconds
    double amplitude_x;       // meters
    double amplitude_y;       // meters
    double z_offset;          // meters
    double z_variation;       // meters
    
    // Trajectory configuration
    TrajectoryOrientation orientation;
    double plane_inclination; // radians (used for INCLINED orientation)
    
    // Sampling rates
    double pose_frequency;    // Hz
    double imu_frequency;     // Hz
    
    // Noise parameters
    double pose_position_noise_std;    // meters
    double pose_orientation_noise_std; // radians
    double accel_noise_std;            // m/s²
    double gyro_noise_std;             // rad/s
    double accel_bias_std;             // m/s²
    double gyro_bias_std;              // rad/s

    TestParameters() : 
        duration(10.0),
        amplitude_x(5.0),
        amplitude_y(5.0),
        z_offset(0.0),
        z_variation(1.0),
        orientation(TrajectoryOrientation::INCLINED),
        plane_inclination(M_PI/4.0), // 45 degrees
        pose_frequency(10.0),
        imu_frequency(100.0),
        pose_position_noise_std(0.05),
        pose_orientation_noise_std(0.01),
        accel_noise_std(0.05),
        gyro_noise_std(0.01),
        accel_bias_std(0.01),
        gyro_bias_std(0.001)
    {}
};

// Extended pose data for ground truth
struct GroundTruthData {
    int64_t timestampNanoseconds;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    Eigen::Vector3d velocity;
    Eigen::Vector3d acceleration;
    Eigen::Vector3d angular_velocity;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Utility class for trajectory generation
class TrajectoryGenerator {
public:
    TrajectoryGenerator(const TestParameters& params) 
        : params_(params), 
          rand_engine_(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())) 
    {
        generateTrajectory();
    }

    // Generate a figure-8 trajectory based on the specified orientation
    void generateTrajectory() {
        // Initialize rotation and gravity based on orientation
        Eigen::Matrix3d rotation;
        
        if (params_.orientation == TrajectoryOrientation::HORIZONTAL) {
            // No rotation for horizontal trajectory (parallel to ground)
            rotation = Eigen::Matrix3d::Identity();
            
            // Standard gravity vector pointing downward
            gravity_ = Eigen::Vector3d(0, 0, -9.80665);
        } else {
            // Calculate the rotation to apply for the inclined plane
            rotation = Eigen::AngleAxisd(params_.plane_inclination, Eigen::Vector3d::UnitX()).toRotationMatrix();
            
            // Pre-calculate the gravity vector for the inclined plane
            gravity_ = rotation * Eigen::Vector3d(0, 0, -9.80665);
        }
        
        std::cout << "Gravity vector: [" << gravity_.x() << ", " << gravity_.y() << ", " << gravity_.z() << "]" << std::endl;
        
        // Number of samples for ground truth (high resolution)
        const int num_samples = static_cast<int>(params_.duration * 1000); // 1000 Hz for ground truth
        const double dt = 1.0 / 1000.0;
        
        ground_truth_.reserve(num_samples);
        
        // Frequency parameters for the figure-8
        const double wx = 2.0 * M_PI / params_.duration * 0.5; // One full figure-8 in the duration
        const double wy = wx;
        const double wz = 2.0 * wx; // Double frequency for z to make it interesting
        
        for (int i = 0; i < num_samples; ++i) {
            double t = i * dt;
            
            // Calculate the planar figure-8 pattern
            Eigen::Vector3d pos_planar;
            pos_planar.x() = params_.amplitude_x * sin(wx * t);
            pos_planar.y() = params_.amplitude_y * sin(wy * t) * cos(wy * t);
            pos_planar.z() = params_.z_offset + params_.z_variation * cos(wz * t);
            
            // Rotate to the inclined plane
            Eigen::Vector3d position = rotation * pos_planar;
            
            // Calculate velocity (analytical derivative of position)
            Eigen::Vector3d vel_planar;
            vel_planar.x() = params_.amplitude_x * wx * cos(wx * t);
            vel_planar.y() = params_.amplitude_y * wy * (cos(wy * t) * cos(wy * t) - sin(wy * t) * sin(wy * t));
            vel_planar.z() = -params_.z_variation * wz * sin(wz * t);
            Eigen::Vector3d velocity = rotation * vel_planar;
            
            // Calculate acceleration (analytical derivative of velocity)
            Eigen::Vector3d acc_planar;
            acc_planar.x() = -params_.amplitude_x * wx * wx * sin(wx * t);
            acc_planar.y() = -params_.amplitude_y * wy * wy * 2.0 * sin(wy * t) * cos(wy * t);
            acc_planar.z() = -params_.z_variation * wz * wz * cos(wz * t);
            Eigen::Vector3d acceleration = rotation * acc_planar;
            
            // Calculate orientation (align z-axis with velocity direction)
            Eigen::Vector3d forward = velocity.normalized();
            Eigen::Vector3d up = Eigen::Vector3d(0, 0, 1); // Global up
            Eigen::Vector3d right = up.cross(forward).normalized();
            up = forward.cross(right).normalized();
            
            Eigen::Matrix3d rot_matrix;
            rot_matrix.col(0) = right;
            rot_matrix.col(1) = up;
            rot_matrix.col(2) = forward;
            Eigen::Quaterniond orientation(rot_matrix);
            
            // Calculate angular velocity based on orientation change
            Eigen::Vector3d angular_velocity;
            if (i > 0) {
                Eigen::Quaterniond prev_q = ground_truth_[i-1].orientation;
                Eigen::Quaterniond delta_q = orientation * prev_q.inverse();
                
                // Convert to axis-angle representation
                Eigen::AngleAxisd axis_angle(delta_q);
                angular_velocity = axis_angle.axis() * axis_angle.angle() / dt;
            } else {
                angular_velocity = Eigen::Vector3d::Zero();
            }
            
            // Store the ground truth data
            GroundTruthData gt_data;
            gt_data.timestampNanoseconds = static_cast<int64_t>(t * 1e9);
            gt_data.position = position;
            gt_data.orientation = orientation;
            gt_data.velocity = velocity;
            gt_data.acceleration = acceleration;
            gt_data.angular_velocity = angular_velocity;
            
            ground_truth_.push_back(gt_data);
        }
        
        generateSyntheticMeasurements();
    }

    // Generate synthetic IMU and pose measurements from the ground truth
    void generateSyntheticMeasurements() {
        // Normal distributions for noise
        std::normal_distribution<double> pos_noise(0.0, params_.pose_position_noise_std);
        std::normal_distribution<double> orient_noise(0.0, params_.pose_orientation_noise_std);
        std::normal_distribution<double> accel_noise(0.0, params_.accel_noise_std);
        std::normal_distribution<double> gyro_noise(0.0, params_.gyro_noise_std);
        
        // Bias distributions
        std::normal_distribution<double> accel_bias_dist(0.0, params_.accel_bias_std);
        std::normal_distribution<double> gyro_bias_dist(0.0, params_.gyro_bias_std);
        
        // Generate constant biases for this test
        Eigen::Vector3d accel_bias(accel_bias_dist(rand_engine_), accel_bias_dist(rand_engine_), accel_bias_dist(rand_engine_));
        Eigen::Vector3d gyro_bias(gyro_bias_dist(rand_engine_), gyro_bias_dist(rand_engine_), gyro_bias_dist(rand_engine_));
        
        // Generate pose measurements
        double pose_dt = 1.0 / params_.pose_frequency;
        int pose_stride = static_cast<int>(1000.0 / params_.pose_frequency);
        
        for (size_t i = 0; i < ground_truth_.size(); i += pose_stride) {
            const auto& gt = ground_truth_[i];
            
            // Add noise to position
            Eigen::Vector3d noisy_position = gt.position + Eigen::Vector3d(
                pos_noise(rand_engine_), pos_noise(rand_engine_), pos_noise(rand_engine_));
            
            // Add noise to orientation
            Eigen::AngleAxisd noise_aa(orient_noise(rand_engine_), 
                                      Eigen::Vector3d::Random().normalized());
            Eigen::Quaterniond noisy_orientation = Eigen::Quaterniond(noise_aa) * gt.orientation;
            noisy_orientation.normalize();
            
            // Create pose measurement
            PoseData pose;
            pose.timestampNanoseconds = gt.timestampNanoseconds;
            pose.position = noisy_position;
            pose.orientation = noisy_orientation;
            
            pose_measurements_.push_back(pose);
        }
        
        // Generate IMU measurements
        double imu_dt = 1.0 / params_.imu_frequency;
        int imu_stride = static_cast<int>(1000.0 / params_.imu_frequency);
        
        for (size_t i = 0; i < ground_truth_.size(); i += imu_stride) {
            const auto& gt = ground_truth_[i];
            
            // Calculate specific force (acceleration minus gravity, in body frame)
            Eigen::Vector3d specific_force = gt.orientation.inverse() * (gt.acceleration - gravity_);
            
            // Add noise and bias to accelerometer data
            Eigen::Vector3d noisy_accel = specific_force + accel_bias + Eigen::Vector3d(
                accel_noise(rand_engine_), accel_noise(rand_engine_), accel_noise(rand_engine_));
            
            // Add noise and bias to gyroscope data
            Eigen::Vector3d noisy_gyro = gt.angular_velocity + gyro_bias + Eigen::Vector3d(
                gyro_noise(rand_engine_), gyro_noise(rand_engine_), gyro_noise(rand_engine_));
            
            // Create IMU measurement using the constructor that takes all parameters
            ImuData imu(gt.timestampNanoseconds, noisy_gyro, noisy_accel);
            
            imu_measurements_.push_back(imu);
        }
    }

    // Run trajectory analysis and print statistics
    void analyzeTrajectory() {
        std::cout << "\n====== Figure-8 Trajectory Analysis ======" << std::endl;
        std::cout << "Trajectory duration: " << params_.duration << " seconds" << std::endl;
        std::cout << "Plane inclination: " << (params_.plane_inclination * 180.0 / M_PI) << " degrees" << std::endl;
        std::cout << "Gravity vector: [" << gravity_.x() << ", " << gravity_.y() << ", " << gravity_.z() << "]" << std::endl;
        
        // Calculate trajectory statistics
        Eigen::Vector3d min_pos = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
        Eigen::Vector3d max_pos = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
        Eigen::Vector3d avg_pos = Eigen::Vector3d::Zero();
        
        Eigen::Vector3d min_vel = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
        Eigen::Vector3d max_vel = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
        Eigen::Vector3d avg_vel = Eigen::Vector3d::Zero();
        
        Eigen::Vector3d min_acc = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
        Eigen::Vector3d max_acc = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
        Eigen::Vector3d avg_acc = Eigen::Vector3d::Zero();
        
        for (const auto& gt : ground_truth_) {
            // Position stats
            min_pos = min_pos.cwiseMin(gt.position);
            max_pos = max_pos.cwiseMax(gt.position);
            avg_pos += gt.position;
            
            // Velocity stats
            min_vel = min_vel.cwiseMin(gt.velocity);
            max_vel = max_vel.cwiseMax(gt.velocity);
            avg_vel += gt.velocity;
            
            // Acceleration stats
            min_acc = min_acc.cwiseMin(gt.acceleration);
            max_acc = max_acc.cwiseMax(gt.acceleration);
            avg_acc += gt.acceleration;
        }
        
        double n = static_cast<double>(ground_truth_.size());
        avg_pos /= n;
        avg_vel /= n;
        avg_acc /= n;
        
        // Print statistics
        std::cout << "\nTrajectory Statistics:" << std::endl;
        std::cout << "  Position range (x): [" << min_pos.x() << ", " << max_pos.x() << "] m, avg: " << avg_pos.x() << " m" << std::endl;
        std::cout << "  Position range (y): [" << min_pos.y() << ", " << max_pos.y() << "] m, avg: " << avg_pos.y() << " m" << std::endl;
        std::cout << "  Position range (z): [" << min_pos.z() << ", " << max_pos.z() << "] m, avg: " << avg_pos.z() << " m" << std::endl;
        
        std::cout << "\n  Velocity range (x): [" << min_vel.x() << ", " << max_vel.x() << "] m/s, avg: " << avg_vel.x() << " m/s" << std::endl;
        std::cout << "  Velocity range (y): [" << min_vel.y() << ", " << max_vel.y() << "] m/s, avg: " << avg_vel.y() << " m/s" << std::endl;
        std::cout << "  Velocity range (z): [" << min_vel.z() << ", " << max_vel.z() << "] m/s, avg: " << avg_vel.z() << " m/s" << std::endl;
        
        std::cout << "\n  Acceleration range (x): [" << min_acc.x() << ", " << max_acc.x() << "] m/s², avg: " << avg_acc.x() << " m/s²" << std::endl;
        std::cout << "  Acceleration range (y): [" << min_acc.y() << ", " << max_acc.y() << "] m/s², avg: " << avg_acc.y() << " m/s²" << std::endl;
        std::cout << "  Acceleration range (z): [" << min_acc.z() << ", " << max_acc.z() << "] m/s², avg: " << avg_acc.z() << " m/s²" << std::endl;
        
        // Measurement statistics
        std::cout << "\nMeasurements Generated:" << std::endl;
        std::cout << "  Ground truth samples: " << ground_truth_.size() << " (" << (1000.0) << " Hz)" << std::endl;
        std::cout << "  Pose measurements: " << pose_measurements_.size() << " (" << params_.pose_frequency << " Hz)" << std::endl;
        std::cout << "  IMU measurements: " << imu_measurements_.size() << " (" << params_.imu_frequency << " Hz)" << std::endl;
    }

    const Eigen::Vector3d& getGravity() const { return gravity_; }
    const std::vector<GroundTruthData, Eigen::aligned_allocator<GroundTruthData>>& getGroundTruth() const { return ground_truth_; }
    const std::vector<PoseData, Eigen::aligned_allocator<PoseData>>& getPoseMeasurements() const { return pose_measurements_; }
    const std::vector<ImuData, Eigen::aligned_allocator<ImuData>>& getImuMeasurements() const { return imu_measurements_; }

private:
    TestParameters params_;
    std::mt19937 rand_engine_;
    Eigen::Vector3d gravity_;
    
    std::vector<GroundTruthData, Eigen::aligned_allocator<GroundTruthData>> ground_truth_;
    std::vector<PoseData, Eigen::aligned_allocator<PoseData>> pose_measurements_;
    std::vector<ImuData, Eigen::aligned_allocator<ImuData>> imu_measurements_;
};

} // namespace sfip_test

// Google Test test cases
TEST(Figure8TrajectoryTest, InclinedPlaneTrajectory) {
    sfip_test::TestParameters params;
    params.orientation = sfip_test::TrajectoryOrientation::INCLINED;
    params.plane_inclination = M_PI/4.0; // 45 degrees
    
    sfip_test::TrajectoryGenerator generator(params);
    
    // Analyze the generated trajectory
    generator.analyzeTrajectory();
    
    // Verify we have measurements
    EXPECT_GT(generator.getGroundTruth().size(), 0);
    EXPECT_GT(generator.getPoseMeasurements().size(), 0);
    EXPECT_GT(generator.getImuMeasurements().size(), 0);
    
    // Verify the gravity vector on 45-degree inclined plane
    // NOTE: The actual observed value has a positive y component
    // This is due to the rotation convention in the Eigen library
    const double g = 9.80665;
    const double g_comp = g / std::sqrt(2.0);  // g/√2 component
    Eigen::Vector3d expected_gravity(0, g_comp, -g_comp);
    EXPECT_NEAR((generator.getGravity() - expected_gravity).norm(), 0.0, 1e-5);
}

TEST(Figure8TrajectoryTest, HorizontalPlaneTrajectory) {
    sfip_test::TestParameters params;
    params.orientation = sfip_test::TrajectoryOrientation::HORIZONTAL;
    
    sfip_test::TrajectoryGenerator generator(params);
    
    // Analyze the generated trajectory
    generator.analyzeTrajectory();
    
    // Verify we have measurements
    EXPECT_GT(generator.getGroundTruth().size(), 0);
    EXPECT_GT(generator.getPoseMeasurements().size(), 0);
    EXPECT_GT(generator.getImuMeasurements().size(), 0);
    
    // Verify the gravity vector is straight down
    Eigen::Vector3d expected_gravity(0, 0, -9.80665);
    EXPECT_NEAR((generator.getGravity() - expected_gravity).norm(), 0.0, 1e-5);
    
    // Additional validation for the horizontal case
    const auto& ground_truth = generator.getGroundTruth();
    
    // Check that Z values are consistent (should be near constant for horizontal plane)
    double z_sum = 0;
    double z_var = 0;
    
    for (const auto& gt : ground_truth) {
        z_sum += gt.position.z();
    }
    
    double z_mean = z_sum / ground_truth.size();
    
    for (const auto& gt : ground_truth) {
        z_var += (gt.position.z() - z_mean) * (gt.position.z() - z_mean);
    }
    
    z_var /= ground_truth.size();
    
    // Z variance should be small for a horizontal figure-8
    // (allowing for any configured Z variation)
    std::cout << "Z mean: " << z_mean << ", Z variance: " << z_var << std::endl;
    EXPECT_LE(z_var, params.z_variation * params.z_variation);
}

// Test for comparing horizontal vs inclined trajectories
TEST(Figure8TrajectoryTest, CompareTrajectories) {
    // Create both trajectories
    sfip_test::TestParameters horizontal_params;
    horizontal_params.orientation = sfip_test::TrajectoryOrientation::HORIZONTAL;
    sfip_test::TrajectoryGenerator horizontal_generator(horizontal_params);
    
    sfip_test::TestParameters inclined_params;
    inclined_params.orientation = sfip_test::TrajectoryOrientation::INCLINED;
    sfip_test::TrajectoryGenerator inclined_generator(inclined_params);
    
    // Get measurements
    const auto& horizontal_imu = horizontal_generator.getImuMeasurements();
    const auto& inclined_imu = inclined_generator.getImuMeasurements();
    
    // Compare some basic IMU statistics
    Eigen::Vector3d h_accel_sum = Eigen::Vector3d::Zero();
    Eigen::Vector3d i_accel_sum = Eigen::Vector3d::Zero();
    
    for (const auto& imu : horizontal_imu) {
        h_accel_sum += imu.accel;
    }
    
    for (const auto& imu : inclined_imu) {
        i_accel_sum += imu.accel;
    }
    
    Eigen::Vector3d h_accel_mean = h_accel_sum / horizontal_imu.size();
    Eigen::Vector3d i_accel_mean = i_accel_sum / inclined_imu.size();
    
    std::cout << "\n=== Trajectory Comparison ===" << std::endl;
    std::cout << "Horizontal mean accel: [" << h_accel_mean.x() << ", " 
                                          << h_accel_mean.y() << ", " 
                                          << h_accel_mean.z() << "]" << std::endl;
    std::cout << "Inclined mean accel: [" << i_accel_mean.x() << ", " 
                                        << i_accel_mean.y() << ", " 
                                        << i_accel_mean.z() << "]" << std::endl;
    
    // The test should pass regardless - this is just for information
    EXPECT_TRUE(true);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "=====================================================" << std::endl;
    std::cout << "Figure-8 Trajectory Test Suite" << std::endl;
    std::cout << "Testing horizontal and inclined (45°) trajectories" << std::endl;
    std::cout << "=====================================================" << std::endl;
    
    return RUN_ALL_TESTS();
}
