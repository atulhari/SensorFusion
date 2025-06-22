#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <deque>
#include <vector>
#include <cstdint>
#include <memory>

namespace sfip {

/**
 * @brief Common data types for the SFIP library
 */

// Forward declarations
class SplineState;
class WindowManager;
class Optimizer;
class DataProcessor;

/**
 * @brief IMU measurement data structure
 */
struct ImuMeasurement {
    int64_t timestampNanoseconds;
    Eigen::Vector3d gyro;  // Angular velocity in rad/s
    Eigen::Vector3d accel; // Linear acceleration in m/s^2

    ImuMeasurement() = default;
    ImuMeasurement(int64_t t, const Eigen::Vector3d& g, const Eigen::Vector3d& a)
        : timestampNanoseconds(t), gyro(g), accel(a) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief Pose measurement data structure
 */
struct PoseMeasurement {
    int64_t timestampNanoseconds;
    Eigen::Quaterniond orientation;
    Eigen::Vector3d position;

    PoseMeasurement() = default;
    PoseMeasurement(int64_t t, const Eigen::Quaterniond& q, const Eigen::Vector3d& p)
        : timestampNanoseconds(t), orientation(q), position(p) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief Calibration parameters
 */
struct CalibrationParameters {
    Eigen::Vector3d gravity; // Gravity vector in world frame

    CalibrationParameters() : gravity(0, 0, -9.80665) {}

    CalibrationParameters& operator=(const CalibrationParameters& other) {
        if (this != &other) {
            gravity = other.gravity;
        }
        return *this;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief Fusion parameters
 */
struct FusionParameters {
    // Optimization parameters
    bool optimizeGravity;
    int maxIterations;
    double initialLambda;
    double initialLambdaVee;
    
    // Window parameters
    int windowSize;
    int64_t knotIntervalNanoseconds;
    int controlPointFps;
    
    // Measurement weights
    double weightPosePosition;
    double weightPoseOrientation;
    double weightAccel;
    double weightGyro;
    double weightBiasAccel;
    double weightBiasGyro;
    
    // Measurement variances (inverse)
    Eigen::Vector3d accelVarianceInv;
    Eigen::Vector3d gyroVarianceInv;
    Eigen::Vector3d biasAccelVarianceInv;
    Eigen::Vector3d biasGyroVarianceInv;
    Eigen::Vector3d positionVarianceInv;

    // Mode selection
    bool poseOnlyMode;

    FusionParameters() :
        optimizeGravity(true),
        maxIterations(10),
        initialLambda(1e-6),
        initialLambdaVee(2.0),
        windowSize(10),
        knotIntervalNanoseconds(0), // Will be calculated from controlPointFps
        controlPointFps(20),
        weightPosePosition(1.0),
        weightPoseOrientation(1.0),
        weightAccel(1.0),
        weightGyro(1.0),
        weightBiasAccel(1.0),
        weightBiasGyro(1.0),
        accelVarianceInv(Eigen::Vector3d::Ones()),
        gyroVarianceInv(Eigen::Vector3d::Ones()),
        biasAccelVarianceInv(Eigen::Vector3d::Ones()),
        biasGyroVarianceInv(Eigen::Vector3d::Ones()),
        positionVarianceInv(Eigen::Vector3d::Ones()),
        poseOnlyMode(false)
    {
        // Calculate knot interval from FPS
        knotIntervalNanoseconds = static_cast<int64_t>(1e9 / controlPointFps);
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief Optimization state
 */
struct OptimizationState {
    double currentError;
    double previousError;
    int iterations;
    bool converged;
    double lambda;
    double lambdaVee;
    
    OptimizationState() :
        currentError(0.0),
        previousError(0.0),
        iterations(0),
        converged(false),
        lambda(1e-6),
        lambdaVee(2.0)
    {}
};

/**
 * @brief Optimization result
 */
struct OptimizationResult {
    bool converged;
    double finalError;
    int iterations;
    double runtime;
    
    OptimizationResult() :
        converged(false),
        finalError(0.0),
        iterations(0),
        runtime(0.0)
    {}
};

/**
 * @brief Window state
 */
struct WindowState {
    bool isFullSize;
    int numKnots;
    int64_t startTimeNanoseconds;
    int64_t endTimeNanoseconds;
    int startIndex;
    
    WindowState() :
        isFullSize(false),
        numKnots(0),
        startTimeNanoseconds(0),
        endTimeNanoseconds(0),
        startIndex(0)
    {}
};

/**
 * @brief Pose estimate output
 */
struct PoseEstimate {
    int64_t timestampNanoseconds;
    Eigen::Quaterniond orientation;
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Vector3d acceleration;
    Eigen::Vector3d angularVelocity;
    
    PoseEstimate() = default;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Utility typedefs for common containers
using ImuMeasurementDeque = std::deque<ImuMeasurement, Eigen::aligned_allocator<ImuMeasurement>>;
using PoseMeasurementDeque = std::deque<PoseMeasurement, Eigen::aligned_allocator<PoseMeasurement>>;
using ImuMeasurementVector = std::vector<ImuMeasurement, Eigen::aligned_allocator<ImuMeasurement>>;
using PoseMeasurementVector = std::vector<PoseMeasurement, Eigen::aligned_allocator<PoseMeasurement>>;

} // namespace sfip
