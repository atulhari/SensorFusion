#pragma once

#include "sfip/Types.hpp"
#include <memory>

namespace sfip {

/**
 * @brief Core class for spline-based sensor fusion
 * 
 * This class implements the core functionality of the spline-based fusion
 * algorithm without any ROS dependencies. It combines IMU and pose measurements
 * to estimate the pose of a platform at high frequency.
 */
class SplineFusionCore {
public:
    /**
     * @brief Constructor
     * @param params Fusion parameters
     */
    SplineFusionCore(const FusionParameters& params);

    /**
     * @brief Destructor
     */
    ~SplineFusionCore();

    /**
     * @brief Add an IMU measurement to the buffer
     * @param measurement IMU measurement
     * @return True if successfully added
     */
    bool addImuMeasurement(const ImuMeasurement& measurement);

    /**
     * @brief Add a pose measurement to the buffer
     * @param measurement Pose measurement
     * @return True if successfully added
     */
    bool addPoseMeasurement(const PoseMeasurement& measurement);

    /**
     * @brief Initialize the spline fusion system
     * @return True if initialization succeeded
     */
    bool initialize();

    /**
     * @brief Run a single optimization iteration
     * @return Result of the optimization
     */
    OptimizationResult runOptimization();

    /**
     * @brief Query the spline state at a specific time
     * @param timestampNanoseconds Query timestamp in nanoseconds
     * @return Estimated pose at the requested time
     */
    PoseEstimate query(int64_t timestampNanoseconds) const;

    /**
     * @brief Get the current window state
     * @return Current window state
     */
    WindowState getWindowState() const;

    /**
     * @brief Get the current calibration parameters
     * @return Current calibration parameters
     */
    CalibrationParameters getCalibrationParameters() const;

    /**
     * @brief Set calibration parameters
     * @param params New calibration parameters
     */
    void setCalibrationParameters(const CalibrationParameters& params);

    /**
     * @brief Get the fusion parameters
     * @return Current fusion parameters
     */
    FusionParameters getFusionParameters() const;

    /**
     * @brief Get the latest optimization state
     * @return Current optimization state
     */
    OptimizationState getOptimizationState() const;

    /**
     * @brief Get raw spline state for visualization or debugging
     * @return Pointer to the internal spline state
     */
    std::shared_ptr<SplineState> getSplineState() const;

private:
    // Internal state
    FusionParameters params_;
    CalibrationParameters calibParams_;
    OptimizationState optState_;
    WindowState windowState_;
    int64_t nextKnotTimeNs_;
    int64_t lastImuTimeNs_;
    bool initialized_;

    // Measurement buffers
    ImuMeasurementDeque imuBuffer_;
    PoseMeasurementDeque poseBuffer_;
    ImuMeasurementDeque imuWindow_;
    PoseMeasurementDeque poseWindow_;

    // Core components
    std::shared_ptr<SplineState> splineState_;
    std::unique_ptr<WindowManager> windowManager_;
    std::unique_ptr<Optimizer> optimizer_;
    std::unique_ptr<DataProcessor> dataProcessor_;

    // Private methods
    bool checkInitialization();
    bool updateMeasurementWindows();
    bool integrateImu(int64_t fromTime, int64_t toTime, 
                     Eigen::Quaterniond& resultOrientation,
                     Eigen::Vector3d& resultPosition);
    bool extrapolateNextKnot(Eigen::Quaterniond& resultOrientation,
                            Eigen::Vector3d& resultPosition);

    // Prevent copying
    SplineFusionCore(const SplineFusionCore&) = delete;
    SplineFusionCore& operator=(const SplineFusionCore&) = delete;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace sfip
