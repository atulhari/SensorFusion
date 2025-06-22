#include "sfip/DataProcessor.hpp"
#include "sfip/SplineState.hpp"

namespace sfip {

DataProcessor::DataProcessor(const FusionParameters& params)
    : params_(params),
      firstImuIntegration_(true),
      prevAccel_(Eigen::Vector3d::Zero()),
      prevGyro_(Eigen::Vector3d::Zero())
{
    // Initialize with default calibration parameters
    calibParams_ = CalibrationParameters();
}

DataProcessor::~DataProcessor() {
    // Nothing special to clean up
}

void DataProcessor::setSplineState(std::shared_ptr<SplineState> splineState, 
                                 const CalibrationParameters& calibParams) {
    splineState_ = splineState;
    calibParams_ = calibParams;
}

bool DataProcessor::validateImuMeasurement(const ImuMeasurement& measurement) const {
    // Skip if in pose-only mode
    if (params_.poseOnlyMode) {
        return false;
    }
    
    // Check for NaN or Inf values
    if (!std::isfinite(measurement.accel.x()) || 
        !std::isfinite(measurement.accel.y()) || 
        !std::isfinite(measurement.accel.z()) ||
        !std::isfinite(measurement.gyro.x()) || 
        !std::isfinite(measurement.gyro.y()) || 
        !std::isfinite(measurement.gyro.z())) {
        return false;
    }
    
    // Check for unreasonable magnitudes
    const double MAX_ACCEL_NORM = 50.0; // m/s^2
    const double MAX_GYRO_NORM = 10.0;  // rad/s
    
    if (measurement.accel.norm() > MAX_ACCEL_NORM || 
        measurement.gyro.norm() > MAX_GYRO_NORM) {
        return false;
    }
    
    return true;
}

bool DataProcessor::validatePoseMeasurement(const PoseMeasurement& measurement) const {
    // Check for NaN or Inf values in position
    if (!std::isfinite(measurement.position.x()) || 
        !std::isfinite(measurement.position.y()) || 
        !std::isfinite(measurement.position.z())) {
        return false;
    }
    
    // Check quaternion validity
    if (!std::isfinite(measurement.orientation.w()) || 
        !std::isfinite(measurement.orientation.x()) || 
        !std::isfinite(measurement.orientation.y()) || 
        !std::isfinite(measurement.orientation.z())) {
        return false;
    }
    
    // Check if quaternion is normalized
    const double qNorm = measurement.orientation.norm();
    if (std::abs(qNorm - 1.0) > 1e-3) {
        return false;
    }
    
    return true;
}

bool DataProcessor::getImuInterval(int64_t fromTime, int64_t toTime, 
                                 const ImuMeasurementDeque& imuBuffer,
                                 ImuMeasurementVector& result) {
    result.clear();
    
    if (imuBuffer.empty()) {
        return false;
    }
    
    // Get IMU measurements within the time interval
    for (const auto& imu : imuBuffer) {
        if (imu.timestampNanoseconds >= fromTime && imu.timestampNanoseconds <= toTime) {
            result.push_back(imu);
        } else if (imu.timestampNanoseconds > toTime) {
            break;
        }
    }
    
    return !result.empty();
}

bool DataProcessor::integrate(int64_t fromTime, int64_t toTime,
                            const ImuMeasurementVector& imuMeasurements,
                            Eigen::Quaterniond& resultOrientation,
                            Eigen::Vector3d& resultPosition,
                            Eigen::Vector3d* resultVelocity) {
    if (imuMeasurements.empty() || !splineState_) {
        return false;
    }
    
    // Initialize integration state
    Eigen::Matrix3d orientation;
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    
    // Get initial state from spline at fromTime
    splineState_->interpolateQuaternion(fromTime, &resultOrientation);
    orientation = resultOrientation.toRotationMatrix();
    position = splineState_->interpolatePosition(fromTime);
    velocity = splineState_->interpolatePosition<1>(fromTime);
    
    // Reset integration state
    firstImuIntegration_ = true;
    
    // Integrate each IMU measurement
    for (size_t i = 0; i < imuMeasurements.size(); i++) {
        const auto& imu = imuMeasurements[i];
        
        // Calculate time step
        int64_t dt;
        if (i == 0) {
            dt = imu.timestampNanoseconds - fromTime;
        } else {
            dt = imu.timestampNanoseconds - imuMeasurements[i-1].timestampNanoseconds;
        }
        
        if (dt <= 0) {
            // Skip invalid time steps
            continue;
        }
        
        // Perform integration step
        integrateStep(fromTime, dt, imu, orientation, position, velocity);
        
        // Update fromTime for next iteration
        fromTime = imu.timestampNanoseconds;
    }
    
    // Handle potential gap between last IMU and toTime
    if (fromTime < toTime && !imuMeasurements.empty()) {
        int64_t dt = toTime - fromTime;
        if (dt > 0) {
            integrateStep(fromTime, dt, imuMeasurements.back(), orientation, position, velocity);
        }
    }
    
    // Set results
    resultOrientation = Eigen::Quaterniond(orientation);
    resultOrientation.normalize();
    resultPosition = position;
    
    if (resultVelocity) {
        *resultVelocity = velocity;
    }
    
    return true;
}

void DataProcessor::integrateStep(int64_t prevTime, int64_t dt,
                                const ImuMeasurement& imuMeasurement,
                                Eigen::Matrix3d& orientation,
                                Eigen::Vector3d& position,
                                Eigen::Vector3d& velocity) {
    // Convert dt to seconds
    double dtSeconds = dt * NS_TO_S;
    
    // Get IMU data
    Eigen::Vector3d accel = imuMeasurement.accel;
    Eigen::Vector3d gyro = imuMeasurement.gyro;
    
    // Initialize previous values if this is the first integration
    if (firstImuIntegration_) {
        firstImuIntegration_ = false;
        prevAccel_ = accel;
        prevGyro_ = gyro;
    }
    
    // Get bias at this time
    Eigen::Matrix<double, 6, 1> bias;
    if (splineState_) {
        bias = splineState_->interpolateBias(prevTime);
    } else {
        bias.setZero();
    }
    Eigen::Vector3d accelBias = bias.head<3>();
    Eigen::Vector3d gyroBias = bias.tail<3>();
    
    // Apply bias correction
    Eigen::Vector3d accelCorrected = accel - accelBias;
    Eigen::Vector3d gyroCorrected = gyro - gyroBias;
    Eigen::Vector3d prevAccelCorrected = prevAccel_ - accelBias;
    Eigen::Vector3d prevGyroCorrected = prevGyro_ - gyroBias;
    
    // Gravity vector from calibration
    Eigen::Vector3d gravity = calibParams_.gravity;
    
    // Transform acceleration to world frame and subtract gravity
    Eigen::Vector3d accelWorld = orientation * accelCorrected - gravity;
    
    // Average gyro for rotation update
    Eigen::Vector3d avgGyro = 0.5 * (prevGyroCorrected + gyroCorrected);
    
    // Update orientation using small angle approximation
    Eigen::Vector3d rotVec = avgGyro * dtSeconds;
    double rotAngle = rotVec.norm();
    
    if (rotAngle > 1e-10) {
        Eigen::Vector3d rotAxis = rotVec / rotAngle;
        Eigen::AngleAxisd rotation(rotAngle, rotAxis);
        orientation = orientation * rotation.toRotationMatrix();
    }
    
    // Recompute accelWorld with updated orientation
    Eigen::Vector3d accelWorld2 = orientation * accelCorrected - gravity;
    
    // Average acceleration for position/velocity update
    Eigen::Vector3d avgAccel = 0.5 * (accelWorld + accelWorld2);
    
    // Update position and velocity using basic integration
    position += velocity * dtSeconds + 0.5 * avgAccel * dtSeconds * dtSeconds;
    velocity += avgAccel * dtSeconds;
    
    // Store current values for next iteration
    prevAccel_ = accel;
    prevGyro_ = gyro;
}

bool DataProcessor::updateMeasurementWindows(ImuMeasurementDeque& imuWindow,
                                          PoseMeasurementDeque& poseWindow,
                                          ImuMeasurementDeque& imuBuffer,
                                          PoseMeasurementDeque& poseBuffer,
                                          int64_t windowStart,
                                          int64_t windowEnd) {
    // Update IMU window
    if (!imuWindow.empty()) {
        // Remove old measurements
        while (!imuWindow.empty() && imuWindow.front().timestampNanoseconds < windowStart) {
            imuWindow.pop_front();
        }
    }
    
    // Update pose window
    if (!poseWindow.empty()) {
        // Remove old measurements
        while (!poseWindow.empty() && poseWindow.front().timestampNanoseconds < windowStart) {
            poseWindow.pop_front();
        }
    }
    
    // Add new IMU measurements from buffer
    for (const auto& imu : imuBuffer) {
        if (imu.timestampNanoseconds >= windowStart && imu.timestampNanoseconds <= windowEnd) {
            imuWindow.push_back(imu);
        } else if (imu.timestampNanoseconds > windowEnd) {
            break;
        }
    }
    
    // Add new pose measurements from buffer
    for (const auto& pose : poseBuffer) {
        if (pose.timestampNanoseconds >= windowStart && pose.timestampNanoseconds <= windowEnd) {
            poseWindow.push_back(pose);
        } else if (pose.timestampNanoseconds > windowEnd) {
            break;
        }
    }
    
    // Remove used measurements from buffers
    while (!imuBuffer.empty() && imuBuffer.front().timestampNanoseconds <= windowEnd) {
        imuBuffer.pop_front();
    }
    
    while (!poseBuffer.empty() && poseBuffer.front().timestampNanoseconds <= windowEnd) {
        poseBuffer.pop_front();
    }
    
    // Sort windows to ensure time-ordering
    std::sort(imuWindow.begin(), imuWindow.end(), 
             [](const ImuMeasurement& a, const ImuMeasurement& b) {
                 return a.timestampNanoseconds < b.timestampNanoseconds;
             });
    
    std::sort(poseWindow.begin(), poseWindow.end(), 
             [](const PoseMeasurement& a, const PoseMeasurement& b) {
                 return a.timestampNanoseconds < b.timestampNanoseconds;
             });
    
    return true;
}

} // namespace sfip
