#include "sfip/Optimizer.hpp"
#include "sfip/SplineState.hpp"

namespace sfip {

Optimizer::Optimizer(const FusionParameters& params)
    : params_(params),
      biasBlockOffset_(0),
      gravityBlockOffset_(0),
      hessianSize_(0),
      poseFixed_(false)
{
    // Initialize calibration parameters
    calibParams_ = CalibrationParameters();
}

Optimizer::~Optimizer() {
    // Nothing special to clean up
}

bool Optimizer::setup(std::shared_ptr<SplineState> splineState, 
                    const CalibrationParameters& calibParams,
                    const WindowState& windowState) {
    // Store references
    splineState_ = splineState;
    calibParams_ = calibParams;
    windowState_ = windowState;
    
    // Update optimizer size based on window state
    return updateOptimizerSize();
}

OptimizationResult Optimizer::iterate(const ImuMeasurementDeque& imuMeasurements,
                                    const PoseMeasurementDeque& poseMeasurements,
                                    OptimizationState& optState) {
    OptimizationResult result;
    
    // Setup the optimization problem
    setupProblem(imuMeasurements, poseMeasurements);
    
    // Calculate initial error
    double initialError = calculateError(imuMeasurements, poseMeasurements);
    
    // Store in state
    optState.previousError = optState.currentError;
    optState.currentError = initialError;
    
    // Check for convergence
    if (optState.iterations > 0) {
        double relativeImprovement = (optState.previousError - optState.currentError) / optState.previousError;
        if (relativeImprovement < 1e-6) {
            optState.converged = true;
            result.converged = true;
        }
    }
    
    // Create a pseudo-increment for testing
    Eigen::VectorXd increment = Eigen::VectorXd::Zero(hessianSize_);
    
    // Apply the increment
    applyIncrement(increment);
    
    // Calculate final error
    double finalError = calculateError(imuMeasurements, poseMeasurements);
    
    // Update state
    optState.iterations++;
    optState.currentError = finalError;
    
    // Set result
    result.finalError = finalError;
    result.iterations = optState.iterations;
    
    return result;
}

OptimizationResult Optimizer::optimize(const ImuMeasurementDeque& imuMeasurements,
                                     const PoseMeasurementDeque& poseMeasurements,
                                     OptimizationState& optState) {
    OptimizationResult result;
    
    // Reset the optimization state
    optState.iterations = 0;
    optState.converged = false;
    
    // Run optimization iterations up to max iterations
    for (int i = 0; i < params_.maxIterations; i++) {
        result = iterate(imuMeasurements, poseMeasurements, optState);
        
        // Check for convergence
        if (result.converged) {
            break;
        }
    }
    
    // Update result
    result.iterations = optState.iterations;
    result.converged = optState.converged;
    result.finalError = optState.currentError;
    
    return result;
}

double Optimizer::calculateError(const ImuMeasurementDeque& imuMeasurements,
                               const PoseMeasurementDeque& poseMeasurements) {
    // This is a placeholder for the actual error calculation
    // In a real implementation, we would compute residuals for all measurements
    double error = 0.0;
    
    // TODO: Implement actual error calculation
    // This would involve calculating IMU and pose residuals based on the current spline state
    
    return error;
}

CalibrationParameters Optimizer::getCalibrationParameters() const {
    return calibParams_;
}

void Optimizer::setCalibrationParameters(const CalibrationParameters& params) {
    calibParams_ = params;
}

bool Optimizer::updateOptimizerSize() {
    if (!splineState_) {
        return false;
    }
    
    // Get number of knots in the window
    int numKnots = windowState_.numKnots;
    
    // Calculate block offsets and total Hessian size
    const int POSE_SIZE = 6;  // 3 for position, 3 for orientation
    const int BIAS_SIZE = 6;  // 3 for accel bias, 3 for gyro bias
    const int G_SIZE = 2;     // 2 for gravity optimization
    
    // First block is for pose knots
    biasBlockOffset_ = POSE_SIZE * numKnots;
    
    // Next comes the bias knots
    if (!params_.poseOnlyMode) {
        gravityBlockOffset_ = biasBlockOffset_ + BIAS_SIZE * numKnots;
    } else {
        gravityBlockOffset_ = biasBlockOffset_;
    }
    
    // Total Hessian size
    hessianSize_ = gravityBlockOffset_;
    
    // Add gravity parameters if optimizing gravity
    if (params_.optimizeGravity && !params_.poseOnlyMode) {
        hessianSize_ += G_SIZE;
    }
    
    // Determine if first pose should be fixed
    poseFixed_ = (windowState_.isFullSize == false);
    
    return true;
}

bool Optimizer::applyIncrement(Eigen::VectorXd& increment) {
    if (!splineState_) {
        return false;
    }
    
    // Apply pose increments
    const int POSE_SIZE = 6;  // 3 for position, 3 for orientation
    const int BIAS_SIZE = 6;  // 3 for accel bias, 3 for gyro bias
    
    // Get number of knots
    int numKnots = windowState_.numKnots;
    
    // Check if increment vector has enough elements for pose updates
    if (increment.size() < biasBlockOffset_) {
        return false;
    }
    
    // Apply pose increments
    for (int i = 0; i < numKnots; i++) {
        // Skip first pose if fixed
        if (poseFixed_ && i == 0) {
            continue;
        }
        
        // Extract pose increment
        Eigen::Matrix<double, 6, 1> poseInc = increment.segment<POSE_SIZE>(POSE_SIZE * i);
        
        // Apply to spline
        splineState_->applyPoseIncrement(i, poseInc);
    }
    
    // Make sure quaternions are consistent
    splineState_->checkQuaternionControlPoints();
    
    // Apply bias increments if not in pose-only mode
    if (!params_.poseOnlyMode) {
        // Check if increment vector has enough elements for bias updates
        if (increment.size() < gravityBlockOffset_) {
            return false;
        }
        
        // Apply bias increments
        for (int i = 0; i < numKnots; i++) {
            // Extract bias increment
            Eigen::Matrix<double, 6, 1> biasInc = 
                increment.segment<BIAS_SIZE>(biasBlockOffset_ + BIAS_SIZE * i);
            
            // Apply to spline
            splineState_->applyBiasIncrement(i, biasInc);
        }
    }
    
    // Apply gravity increment if optimizing gravity
    if (params_.optimizeGravity && !params_.poseOnlyMode) {
        // Check if increment vector has enough elements for gravity update
        if (increment.size() < hessianSize_) {
            return false;
        }
        
        // TODO: Implement gravity optimization
    }
    
    return true;
}

void Optimizer::setupProblem(const ImuMeasurementDeque& imuMeasurements,
                           const PoseMeasurementDeque& poseMeasurements) {
    // This is a placeholder for setting up the optimization problem
    // In a real implementation, we would set up the problem for linearization
    
    // TODO: Implement actual problem setup
}

} // namespace sfip
