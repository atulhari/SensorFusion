#pragma once

#include "sfip/Types.hpp"
#include <memory>

namespace sfip {

/**
 * @brief Manages the sliding window of control points for the spline
 * 
 * This class handles the logic for maintaining the window of active spline knots,
 * deciding when to marginalize old knots, and managing the transition between the
 * initialization phase and full window operation.
 */
class WindowManager {
public:
    /**
     * @brief Constructor
     * @param params Fusion parameters for window configuration
     */
    WindowManager(const FusionParameters& params);

    /**
     * @brief Destructor
     */
    ~WindowManager();

    /**
     * @brief Initialize the window
     * @param splineState Pointer to the spline state to manage
     * @param startTimeNs Start time for the spline in nanoseconds
     * @return True if initialization succeeded
     */
    bool initialize(std::shared_ptr<SplineState> splineState, int64_t startTimeNs);

    /**
     * @brief Add a new knot to the spline window
     * @param orientation Orientation for the new knot
     * @param position Position for the new knot
     * @param bias Bias for the new knot
     * @return True if knot was added successfully
     */
    bool addKnot(const Eigen::Quaterniond& orientation, 
                const Eigen::Vector3d& position, 
                const Eigen::Matrix<double, 6, 1>& bias);

    /**
     * @brief Slide the window by removing the oldest knot
     * @return True if window was successfully slid
     */
    bool slideWindow();

    /**
     * @brief Check if a new knot can be added based on available data
     * @param latestMeasurementTime Latest measurement timestamp available
     * @return True if a new knot can be added
     */
    bool canAddKnot(int64_t latestMeasurementTime) const;

    /**
     * @brief Get the current window state
     * @return Current window state
     */
    WindowState getWindowState() const;

    /**
     * @brief Check if the window is initialized
     * @return True if window is initialized
     */
    bool isInitialized() const;

    /**
     * @brief Check if the window is at full size
     * @return True if window is at full size
     */
    bool isFullSize() const;

    /**
     * @brief Get the next knot time in nanoseconds
     * @return Next knot time
     */
    int64_t getNextKnotTimeNs() const;

    /**
     * @brief Get the window start time in nanoseconds
     * @return Window start time
     */
    int64_t getWindowStartTimeNs() const;

    /**
     * @brief Get the window end time in nanoseconds
     * @return Window end time
     */
    int64_t getWindowEndTimeNs() const;

    /**
     * @brief Update measurement windows to match current spline window
     * @param imuWindow IMU measurements deque to update
     * @param poseWindow Pose measurements deque to update
     * @param imuBuffer Full IMU buffer
     * @param poseBuffer Full pose buffer
     * @return True if windows were updated successfully
     */
    bool updateMeasurementWindows(ImuMeasurementDeque& imuWindow,
                                 PoseMeasurementDeque& poseWindow,
                                 ImuMeasurementDeque& imuBuffer,
                                 PoseMeasurementDeque& poseBuffer);

private:
    // References
    std::shared_ptr<SplineState> splineState_;
    FusionParameters params_;

    // State
    WindowState state_;
    bool initialized_;
    int64_t nextKnotTimeNs_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace sfip
