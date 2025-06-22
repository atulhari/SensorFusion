#pragma once

#include "sfip/Types.hpp"
#include <memory>

namespace sfip {

/**
 * @brief Handles optimization of the spline control points
 * 
 * This class implements the optimization algorithm to fit the spline to 
 * the provided IMU and pose measurements.
 */
class Optimizer {
public:
    /**
     * @brief Constructor
     * @param params Fusion parameters for optimization configuration
     */
    Optimizer(const FusionParameters& params);

    /**
     * @brief Destructor
     */
    ~Optimizer();

    /**
     * @brief Setup the optimization problem
     * @param splineState Pointer to the spline state to optimize
     * @param calibParams Calibration parameters
     * @param windowState Current window state
     * @return True if setup succeeded
     */
    bool setup(std::shared_ptr<SplineState> splineState, 
              const CalibrationParameters& calibParams,
              const WindowState& windowState);

    /**
     * @brief Run one iteration of the optimization
     * @param imuMeasurements IMU measurements in the current window
     * @param poseMeasurements Pose measurements in the current window
     * @param optState Current optimization state (will be updated)
     * @return Result of the optimization iteration
     */
    OptimizationResult iterate(const ImuMeasurementDeque& imuMeasurements,
                             const PoseMeasurementDeque& poseMeasurements,
                             OptimizationState& optState);

    /**
     * @brief Run the full optimization
     * @param imuMeasurements IMU measurements in the current window
     * @param poseMeasurements Pose measurements in the current window
     * @param optState Optimization state (will be updated)
     * @return Result of the optimization
     */
    OptimizationResult optimize(const ImuMeasurementDeque& imuMeasurements,
                              const PoseMeasurementDeque& poseMeasurements,
                              OptimizationState& optState);

    /**
     * @brief Calculate error without optimization
     * @param imuMeasurements IMU measurements in the current window
     * @param poseMeasurements Pose measurements in the current window
     * @return Error value
     */
    double calculateError(const ImuMeasurementDeque& imuMeasurements,
                         const PoseMeasurementDeque& poseMeasurements);

    /**
     * @brief Get calibration parameters
     * @return Current calibration parameters
     */
    CalibrationParameters getCalibrationParameters() const;

    /**
     * @brief Set calibration parameters
     * @param params New calibration parameters
     */
    void setCalibrationParameters(const CalibrationParameters& params);

private:
    // Core data
    std::shared_ptr<SplineState> splineState_;
    FusionParameters params_;
    CalibrationParameters calibParams_;
    WindowState windowState_;
    
    // Optimization parameters
    size_t biasBlockOffset_;
    size_t gravityBlockOffset_;
    size_t hessianSize_;
    bool poseFixed_;

    // Private methods
    bool updateOptimizerSize();
    bool applyIncrement(Eigen::VectorXd& increment);
    void setupProblem(const ImuMeasurementDeque& imuMeasurements,
                     const PoseMeasurementDeque& poseMeasurements);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace sfip
