#pragma once

#include "sfip/Types.hpp"
#include <memory>

namespace sfip {

/**
 * @brief Processes and filters sensor measurements
 * 
 * This class handles preprocessing of IMU and pose measurements,
 * including validation, filtering, and integration.
 */
class DataProcessor {
public:
    /**
     * @brief Constructor
     * @param params Fusion parameters for data processing configuration
     */
    DataProcessor(const FusionParameters& params);

    /**
     * @brief Destructor
     */
    ~DataProcessor();

    /**
     * @brief Set the spline state to use for integration
     * @param splineState Pointer to the spline state
     * @param calibParams Calibration parameters
     */
    void setSplineState(std::shared_ptr<SplineState> splineState, 
                       const CalibrationParameters& calibParams);

    /**
     * @brief Validate an IMU measurement
     * @param measurement IMU measurement to validate
     * @return True if measurement is valid
     */
    bool validateImuMeasurement(const ImuMeasurement& measurement) const;

    /**
     * @brief Validate a pose measurement
     * @param measurement Pose measurement to validate
     * @return True if measurement is valid
     */
    bool validatePoseMeasurement(const PoseMeasurement& measurement) const;

    /**
     * @brief Filter IMU measurements within a time interval
     * @param fromTime Start time in nanoseconds
     * @param toTime End time in nanoseconds
     * @param imuBuffer Buffer containing IMU measurements
     * @param result Vector to store filtered measurements
     * @return True if at least one valid measurement was found
     */
    bool getImuInterval(int64_t fromTime, int64_t toTime, 
                       const ImuMeasurementDeque& imuBuffer,
                       ImuMeasurementVector& result);

    /**
     * @brief Integrate IMU measurements to predict pose
     * @param fromTime Start time in nanoseconds
     * @param toTime End time in nanoseconds
     * @param imuMeasurements IMU measurements to integrate
     * @param resultOrientation Resulting orientation after integration
     * @param resultPosition Resulting position after integration
     * @param resultVelocity Resulting velocity after integration (optional)
     * @return True if integration succeeded
     */
    bool integrate(int64_t fromTime, int64_t toTime,
                  const ImuMeasurementVector& imuMeasurements,
                  Eigen::Quaterniond& resultOrientation,
                  Eigen::Vector3d& resultPosition,
                  Eigen::Vector3d* resultVelocity = nullptr);

    /**
     * @brief Perform a single integration step
     * @param prevTime Previous time in nanoseconds
     * @param dt Time step in nanoseconds
     * @param imuMeasurement IMU measurement for this step
     * @param orientation Current orientation (updated in-place)
     * @param position Current position (updated in-place)
     * @param velocity Current velocity (updated in-place)
     */
    void integrateStep(int64_t prevTime, int64_t dt,
                      const ImuMeasurement& imuMeasurement,
                      Eigen::Matrix3d& orientation,
                      Eigen::Vector3d& position,
                      Eigen::Vector3d& velocity);

    /**
     * @brief Update measurement windows to match current spline window
     * @param imuWindow IMU measurements deque to update
     * @param poseWindow Pose measurements deque to update
     * @param imuBuffer Full IMU buffer
     * @param poseBuffer Full pose buffer
     * @param windowStart Window start time in nanoseconds
     * @param windowEnd Window end time in nanoseconds
     * @return True if windows were updated successfully
     */
    bool updateMeasurementWindows(ImuMeasurementDeque& imuWindow,
                                 PoseMeasurementDeque& poseWindow,
                                 ImuMeasurementDeque& imuBuffer,
                                 PoseMeasurementDeque& poseBuffer,
                                 int64_t windowStart,
                                 int64_t windowEnd);

private:
    // References
    std::shared_ptr<SplineState> splineState_;
    FusionParameters params_;
    CalibrationParameters calibParams_;
    
    // IMU integration state
    bool firstImuIntegration_;
    Eigen::Vector3d prevAccel_;
    Eigen::Vector3d prevGyro_;
    
    static constexpr double NS_TO_S = 1e-9;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace sfip
