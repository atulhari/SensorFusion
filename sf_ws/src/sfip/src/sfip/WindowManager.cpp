#include "sfip/WindowManager.hpp"
#include "sfip/SplineState.hpp"

namespace sfip {

WindowManager::WindowManager(const FusionParameters& params)
    : params_(params),
      initialized_(false),
      nextKnotTimeNs_(0)
{
    state_.isFullSize = false;
    state_.numKnots = 0;
    state_.startTimeNanoseconds = 0;
    state_.endTimeNanoseconds = 0;
    state_.startIndex = 0;
}

WindowManager::~WindowManager() {
    // Nothing special to clean up
}

bool WindowManager::initialize(std::shared_ptr<SplineState> splineState, int64_t startTimeNs) {
    // Store the spline state reference
    splineState_ = splineState;
    
    // Update state
    state_.startTimeNanoseconds = startTimeNs;
    state_.startIndex = 0;
    state_.numKnots = 0;
    state_.endTimeNanoseconds = startTimeNs;
    
    // Set next knot time
    nextKnotTimeNs_ = startTimeNs;
    
    initialized_ = true;
    return true;
}

bool WindowManager::addKnot(const Eigen::Quaterniond& orientation, 
                          const Eigen::Vector3d& position, 
                          const Eigen::Matrix<double, 6, 1>& bias) {
    if (!initialized_ || !splineState_) {
        return false;
    }
    
    // Add the knot to the spline
    splineState_->addSingleStateKnot(orientation, position, bias);
    
    // Update state
    state_.numKnots = splineState_->getNumKnots();
    state_.endTimeNanoseconds = splineState_->maxTimeNanoseconds();
    
    // Check if we've reached full window size
    if (state_.numKnots >= params_.windowSize) {
        state_.isFullSize = true;
    }
    
    return true;
}

bool WindowManager::slideWindow() {
    if (!initialized_ || !splineState_) {
        return false;
    }
    
    // Check if window is full
    if (!state_.isFullSize) {
        return false;
    }
    
    // Remove the oldest knot
    splineState_->removeSingleOldState();
    
    // Update state
    state_.startTimeNanoseconds = splineState_->minTimeNanoseconds();
    state_.endTimeNanoseconds = splineState_->maxTimeNanoseconds();
    state_.numKnots = splineState_->getNumKnots();
    state_.startIndex++;
    
    return true;
}

bool WindowManager::canAddKnot(int64_t latestMeasurementTime) const {
    if (!initialized_ || !splineState_) {
        return false;
    }
    
    // Check if we have enough data to add a new knot
    // We need to ensure we have measurements available after the next knot time
    return latestMeasurementTime > nextKnotTimeNs_;
}

WindowState WindowManager::getWindowState() const {
    return state_;
}

bool WindowManager::isInitialized() const {
    return initialized_;
}

bool WindowManager::isFullSize() const {
    return state_.isFullSize;
}

int64_t WindowManager::getNextKnotTimeNs() const {
    return nextKnotTimeNs_;
}

int64_t WindowManager::getWindowStartTimeNs() const {
    return state_.startTimeNanoseconds;
}

int64_t WindowManager::getWindowEndTimeNs() const {
    return state_.endTimeNanoseconds;
}

bool WindowManager::updateMeasurementWindows(ImuMeasurementDeque& imuWindow,
                                          PoseMeasurementDeque& poseWindow,
                                          ImuMeasurementDeque& imuBuffer,
                                          PoseMeasurementDeque& poseBuffer) {
    if (!initialized_ || !splineState_) {
        return false;
    }
    
    // Get window time range
    int64_t windowStart = splineState_->minTimeNanoseconds();
    int64_t windowEnd = splineState_->maxTimeNanoseconds();
    
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
    size_t added_imu_count = 0;
    for (size_t i = 0; i < imuBuffer.size(); i++) {
        auto& imu = imuBuffer[i];
        if (imu.timestampNanoseconds >= windowStart && imu.timestampNanoseconds <= windowEnd) {
            imuWindow.push_back(imu);
            added_imu_count++;
        } else if (imu.timestampNanoseconds > windowEnd) {
            break;
        }
    }
    
    // Add new pose measurements from buffer
    size_t added_pose_count = 0;
    for (size_t i = 0; i < poseBuffer.size(); i++) {
        auto& pose = poseBuffer[i];
        if (pose.timestampNanoseconds >= windowStart && pose.timestampNanoseconds <= windowEnd) {
            poseWindow.push_back(pose);
            added_pose_count++;
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
