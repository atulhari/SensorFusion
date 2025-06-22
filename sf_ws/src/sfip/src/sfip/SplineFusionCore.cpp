// Include the core header first
#include "sfip/SplineFusionCore.hpp"

// Then include complete definitions for all required components
#include "sfip/SplineState.hpp"
#include "sfip/WindowManager.hpp"
#include "sfip/Optimizer.hpp"
#include "sfip/DataProcessor.hpp"
#include "sfip/Types.hpp"

// Additional includes as needed
#include <algorithm>
#include <cmath>

namespace sfip {

SplineFusionCore::SplineFusionCore(const FusionParameters& params)
    : params_(params),
      nextKnotTimeNs_(0),
      lastImuTimeNs_(0),
      initialized_(false)
{
    // Initialize calibration parameters
    calibParams_ = CalibrationParameters();
    
    // Initialize optimization state
    optState_.lambda = params.initialLambda;
    optState_.lambdaVee = params.initialLambdaVee;
    
    // Create components
    splineState_ = std::make_shared<SplineState>();
    windowManager_ = std::make_unique<WindowManager>(params);
    optimizer_ = std::make_unique<Optimizer>(params);
    dataProcessor_ = std::make_unique<DataProcessor>(params);
    
    // Set initial window state
    windowState_.isFullSize = false;
    windowState_.numKnots = 0;
    windowState_.startTimeNanoseconds = 0;
    windowState_.endTimeNanoseconds = 0;
    windowState_.startIndex = 0;
}

SplineFusionCore::~SplineFusionCore() {
    // Nothing special to clean up, smart pointers handle memory
}

bool SplineFusionCore::addImuMeasurement(const ImuMeasurement& measurement) {
    // Skip if in pose-only mode
    if (params_.poseOnlyMode) {
        return false;
    }
    
    // Validate measurement
    if (!dataProcessor_->validateImuMeasurement(measurement)) {
        return false;
    }
    
    // Check if this IMU measurement is newer than any existing ones
    if (!imuBuffer_.empty() && measurement.timestampNanoseconds <= imuBuffer_.back().timestampNanoseconds) {
        return false;
    }
    
    // Add to buffer
    imuBuffer_.push_back(measurement);
    return true;
}

bool SplineFusionCore::addPoseMeasurement(const PoseMeasurement& measurement) {
    // Validate measurement
    if (!dataProcessor_->validatePoseMeasurement(measurement)) {
        return false;
    }
    
    // Check if this pose measurement is newer than any existing ones
    if (!poseBuffer_.empty() && measurement.timestampNanoseconds <= poseBuffer_.back().timestampNanoseconds) {
        return false;
    }
    
    // Add to buffer
    poseBuffer_.push_back(measurement);
    return true;
}

bool SplineFusionCore::initialize() {
    // Check if we have at least one pose measurement
    if (poseBuffer_.empty()) {
        return false;
    }
    
    // Check if already initialized
    if (initialized_) {
        return true;
    }
    
    // Get the start time from the first pose measurement
    int64_t startTimeNs = poseBuffer_.front().timestampNanoseconds;
    
    // Initialize the spline state
    splineState_->init(params_.knotIntervalNanoseconds, 0, startTimeNs);
    
    // Initialize the window manager with the spline state
    if (!windowManager_->initialize(splineState_, startTimeNs)) {
        return false;
    }
    
    // Set the next knot time
    nextKnotTimeNs_ = startTimeNs;
    
    // Add the first two knots with the same pose to have a zero-velocity start
    const Eigen::Quaterniond& q_anchor = poseBuffer_.front().orientation;
    const Eigen::Vector3d& p_anchor = poseBuffer_.front().position;
    Eigen::Matrix<double, 6, 1> bias_ini = Eigen::Matrix<double, 6, 1>::Zero();
    
    // Add initial knot
    if (!windowManager_->addKnot(q_anchor, p_anchor, bias_ini)) {
        return false;
    }
    nextKnotTimeNs_ += params_.knotIntervalNanoseconds;
    
    // Add duplicate knot to maintain zero-velocity start
    if (!windowManager_->addKnot(q_anchor, p_anchor, bias_ini)) {
        return false;
    }
    nextKnotTimeNs_ += params_.knotIntervalNanoseconds;
    
    // Initialize the data processor
    dataProcessor_->setSplineState(splineState_, calibParams_);
    
    // Set up the optimizer
    optimizer_->setup(splineState_, calibParams_, windowManager_->getWindowState());
    
    // Update window state
    windowState_ = windowManager_->getWindowState();
    
    // Mark as initialized
    initialized_ = true;
    
    // If using IMU, compute initial gravity estimate
    if (!params_.poseOnlyMode && !imuBuffer_.empty()) {
        // Average first few IMU measurements to estimate gravity
        Eigen::Vector3d gravitySum = Eigen::Vector3d::Zero();
        size_t count = 0;
        const size_t maxSamples = std::min(imuBuffer_.size(), size_t(100));
        
        for (size_t i = 0; i < maxSamples; i++) {
            gravitySum += imuBuffer_[i].accel;
            count++;
        }
        
        if (count > 0 && gravitySum.norm() > 1.0) {
            Eigen::Vector3d gravityEst = gravitySum.normalized() * 9.80665;
            calibParams_.gravity = gravityEst;
            optimizer_->setCalibrationParameters(calibParams_);
        }
    }
    
    return true;
}

OptimizationResult SplineFusionCore::runOptimization() {
    OptimizationResult result;
    
    // Check if initialized
    if (!initialized_) {
        return result;
    }
    
    // First check if we have any more data to process
    if (!checkInitialization()) {
        return result;
    }
    
    // Update measurement windows to match current spline window
    if (!updateMeasurementWindows()) {
        return result;
    }
    
    // Get current window state
    windowState_ = windowManager_->getWindowState();
    
    // Update optimizer settings
    optimizer_->setup(splineState_, calibParams_, windowState_);
    
    // Run optimization
    result = optimizer_->optimize(imuWindow_, poseWindow_, optState_);
    
    // Update calibration parameters
    calibParams_ = optimizer_->getCalibrationParameters();
    
    // Slide window if full size
    if (windowState_.isFullSize) {
        windowManager_->slideWindow();
        windowState_ = windowManager_->getWindowState();
    }
    
    return result;
}

PoseEstimate SplineFusionCore::query(int64_t timestampNanoseconds) const {
    PoseEstimate estimate;
    estimate.timestampNanoseconds = timestampNanoseconds;
    
    // Check if initialized
    if (!initialized_) {
        return estimate;
    }
    
    // Check if time is within spline range
    int64_t minTime = splineState_->minTimeNanoseconds();
    int64_t maxTime = splineState_->maxTimeNanoseconds();
    
    if (timestampNanoseconds < minTime || timestampNanoseconds > maxTime) {
        // Return empty estimate if out of range
        return estimate;
    }
    
    // Interpolate position and derivatives
    estimate.position = splineState_->interpolatePosition(timestampNanoseconds);
    estimate.velocity = splineState_->interpolatePosition<1>(timestampNanoseconds);
    estimate.acceleration = splineState_->interpolatePosition<2>(timestampNanoseconds);
    
    // Interpolate orientation and angular velocity
    splineState_->interpolateQuaternion(timestampNanoseconds, &estimate.orientation, &estimate.angularVelocity);
    
    return estimate;
}

WindowState SplineFusionCore::getWindowState() const {
    return windowState_;
}

CalibrationParameters SplineFusionCore::getCalibrationParameters() const {
    return calibParams_;
}

void SplineFusionCore::setCalibrationParameters(const CalibrationParameters& params) {
    calibParams_ = params;
    optimizer_->setCalibrationParameters(params);
}

FusionParameters SplineFusionCore::getFusionParameters() const {
    return params_;
}

OptimizationState SplineFusionCore::getOptimizationState() const {
    return optState_;
}

std::shared_ptr<SplineState> SplineFusionCore::getSplineState() const {
    return splineState_;
}

bool SplineFusionCore::checkInitialization() {
    // Find the latest available measurement time
    int64_t latestMeasurementTime = 0;
    
    if (!imuBuffer_.empty()) {
        latestMeasurementTime = std::max(latestMeasurementTime, imuBuffer_.back().timestampNanoseconds);
    }
    
    if (!poseBuffer_.empty()) {
        latestMeasurementTime = std::max(latestMeasurementTime, poseBuffer_.back().timestampNanoseconds);
    }
    
    // Check if we can add a new knot
    if (windowManager_->canAddKnot(latestMeasurementTime)) {
        // Determine the new knot's pose using either IMU integration or extrapolation
        Eigen::Quaterniond newOrientation;
        Eigen::Vector3d newPosition;
        
        if (!params_.poseOnlyMode && !imuBuffer_.empty()) {
            // Use IMU integration
            if (!integrateImu(lastImuTimeNs_, nextKnotTimeNs_, newOrientation, newPosition)) {
                // Fallback to extrapolation if integration fails
                if (!extrapolateNextKnot(newOrientation, newPosition)) {
                    return false;
                }
            }
            // Update last IMU time
            lastImuTimeNs_ = nextKnotTimeNs_;
        } else {
            // Use extrapolation in pose-only mode
            if (!extrapolateNextKnot(newOrientation, newPosition)) {
                return false;
            }
        }
        
        // Create a new bias estimate based on the last knot's bias
        Eigen::Matrix<double, 6, 1> newBias;
        if (splineState_->getNumKnots() > 0) {
            newBias = splineState_->getKnotBias(splineState_->getNumKnots() - 1);
        } else {
            newBias.setZero();
        }
        
        // Add the new knot
        if (windowManager_->addKnot(newOrientation, newPosition, newBias)) {
            // Update next knot time
            nextKnotTimeNs_ += params_.knotIntervalNanoseconds;
            return true;
        }
    }
    
    return false;
}

bool SplineFusionCore::updateMeasurementWindows() {
    // Get window time range
    int64_t windowStart = splineState_->minTimeNanoseconds();
    int64_t windowEnd = splineState_->maxTimeNanoseconds();
    
    // Update windows using the data processor
    return dataProcessor_->updateMeasurementWindows(
        imuWindow_, poseWindow_, imuBuffer_, poseBuffer_, windowStart, windowEnd);
}

bool SplineFusionCore::integrateImu(int64_t fromTime, int64_t toTime, 
                                  Eigen::Quaterniond& resultOrientation,
                                  Eigen::Vector3d& resultPosition) {
    // Get IMU measurements in the interval
    ImuMeasurementVector imuMeasurements;
    if (!dataProcessor_->getImuInterval(fromTime, toTime, imuBuffer_, imuMeasurements)) {
        return false;
    }
    
    // Perform integration
    return dataProcessor_->integrate(fromTime, toTime, imuMeasurements, 
                                   resultOrientation, resultPosition);
}

bool SplineFusionCore::extrapolateNextKnot(Eigen::Quaterniond& resultOrientation,
                                         Eigen::Vector3d& resultPosition) {
    // Check if we have enough knots to extrapolate
    if (splineState_->getNumKnots() < 2) {
        // If not enough knots, return false (caller should handle this case)
        return false;
    }
    
    // Extrapolate based on the current trajectory
    resultOrientation = splineState_->extrapolateKnotOrientation(1);
    resultPosition = splineState_->extrapolateKnotPosition(1);
    
    return true;
}

} // namespace sfip
