#pragma once

#include "utils/common_utils.hpp"
#include "utils/math_tools.hpp"
#include "Accumulator.hpp"
#include "SplineState.hpp"
#include "Residuals_improved.hpp"

/**
 * @brief Parameters for the optimization process
 */
struct Parameters {
    // Optimization flags
    bool if_opt_g;                 // Whether to optimize gravity direction
    
    // Weights for different measurement types
    double w_pose_pos;             // Weight for position measurements
    double w_pose_rot;             // Weight for rotation measurements
    double w_acc;                  // Weight for accelerometer measurements
    double w_gyro;                 // Weight for gyroscope measurements
    double w_bias_acc;             // Weight for accelerometer bias regularization
    double w_bias_gyro;            // Weight for gyroscope bias regularization

    // Spline configuration
    int control_point_fps;         // Frequency of control points

    // Inverse variance (information) matrices
    Eigen::Vector3d accel_var_inv;         // Inverse of accelerometer variance 
    Eigen::Vector3d gyro_var_inv;          // Inverse of gyroscope variance
    Eigen::Vector3d bias_accel_var_inv;    // Inverse of accelerometer bias variance
    Eigen::Vector3d bias_gyro_var_inv;     // Inverse of gyroscope bias variance
    Eigen::Vector3d pos_var_inv;           // Inverse of position variance
    
    // World frame gravity vector
    Eigen::Vector3d gravity;
    
    // Constructor with default values
    Parameters() : 
        if_opt_g(false),
        w_pose_pos(1.0), 
        w_pose_rot(1.0), 
        w_acc(1.0), 
        w_gyro(1.0),
        w_bias_acc(1.0), 
        w_bias_gyro(1.0), 
        control_point_fps(60),
        gravity(0.0, 0.0, -9.80665) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief Linearizer class for building the optimization problem
 * 
 * This class handles the construction of the Hessian and gradient for
 * nonlinear least squares optimization of the spline state.
 */
struct Linearizer {
    // Size constants for different state vector components
    static const int POSE_SIZE = 6;            // Size of one pose knot (position + orientation)
    static const int POS_SIZE = 3;             // Size of position component
    static const int POS_OFFSET = 0;           // Offset of position within pose
    static const int ROT_SIZE = 3;             // Size of rotation component (as axis-angle)
    static const int ROT_OFFSET = 3;           // Offset of rotation within pose
    static const int ACCEL_BIAS_SIZE = 3;      // Size of accelerometer bias
    static const int GYRO_BIAS_SIZE = 3;       // Size of gyroscope bias
    static const int BIAS_SIZE = ACCEL_BIAS_SIZE + GYRO_BIAS_SIZE; // Size of one bias knot
    static const int G_SIZE = 2;               // Size of gravity direction parameterization
    
    // Offsets within blocks
    static const int ACCEL_BIAS_OFFSET = 0;                // Offset of accel bias in bias block
    static const int GYRO_BIAS_OFFSET = ACCEL_BIAS_SIZE;   // Offset of gyro bias in bias block
    static const int G_OFFSET = 0;                         // Offset of gravity params

    // State for the optimization
    SparseHashAccumulator accum;   // Sparse Hessian and gradient accumulator
    double error;                  // Current error value
    
    // Structure information
    size_t bias_block_offset;      // Offset of bias parameters in state vector
    size_t gravity_block_offset;   // Offset of gravity parameters in state vector
    size_t opt_size;               // Total size of optimization state vector
    
    // Problem definition
    SplineState* spline;           // Pointer to the spline state being optimized
    const Parameters* param;       // Pointer to optimization parameters
    const bool pose_fixed;         // Whether initial pose is fixed

    /**
     * @brief Constructor
     * 
     * @param _bias_block_offset Offset of bias parameters in the state vector
     * @param _gravity_block_offset Offset of gravity parameters in the state vector
     * @param _opt_size Total size of the optimization state vector
     * @param spl Pointer to the spline state being optimized
     * @param par Pointer to optimization parameters
     * @param _pose_fixed Whether the initial pose is fixed
     */
    Linearizer(
        size_t _bias_block_offset, 
        size_t _gravity_block_offset, 
        size_t _opt_size, 
        SplineState* spl,
        const Parameters* par, 
        const bool _pose_fixed
    ) : 
        bias_block_offset(_bias_block_offset), 
        gravity_block_offset(_gravity_block_offset), 
        opt_size(_opt_size),
        spline(spl), 
        param(par), 
        pose_fixed(_pose_fixed)
    {
        // Initialize the accumulator and error
        accum.reset(opt_size);
        error = 0;
    }

    /**
     * @brief Process a batch of IMU measurements to update Hessian and gradient
     * 
     * @param imuMeasurements Vector of IMU measurements
     */
    void operator()(const Eigen::aligned_deque<ImuData>& imuMeasurements) {
        // Check if we can fit blocks in the state vector
        const size_t N = opt_size;
        auto isInBounds = [N](size_t start, size_t blockSize) { 
            return start + blockSize <= N; 
        };

        // Configure weights based on parameters and measurement count
        const size_t numImuMeasurements = imuMeasurements.size();
        const size_t firstFixedKnot = 1;  // Number of knots to fix (typically first knot)
        
        // Scale information matrices by weights and normalize by number of measurements
        Eigen::Vector3d accelVarInv = param->accel_var_inv * param->w_acc / numImuMeasurements;
        Eigen::Vector3d gyroVarInv = param->gyro_var_inv * param->w_gyro / numImuMeasurements;
        
        // Bias regularization weights
        Eigen::Vector3d biasAccelVarInv = param->bias_accel_var_inv * param->w_bias_acc / (numImuMeasurements - 1);
        Eigen::Vector3d biasGyroVarInv = param->bias_gyro_var_inv * param->w_bias_gyro / (numImuMeasurements - 1);
        
        // Process each IMU measurement
        for (const auto& imu : imuMeasurements) {
            // Compute Jacobians for this measurement
            Jacobian36 JaccelPose;   // Jacobian of accelerometer residual w.r.t. pose
            Jacobian33 JgyroPose;    // Jacobian of gyroscope residual w.r.t. pose
            Jacobian JbiasBoth;      // Jacobian of both residuals w.r.t. bias
            Eigen::Matrix<double, 3, 2> JgravityAccel;  // Jacobian of accel residual w.r.t. gravity
            
            // Get timestamp
            int64_t timestamp = imu.timestampNanoseconds;
            
            // Compute residual and Jacobians
            Eigen::Matrix<double, 6, 1> residual = Residuals::imuResidualJacobian(
                timestamp, spline, &imu.accel, &imu.gyro, param->gravity,
                &JaccelPose, &JgyroPose, &JbiasBoth, &JgravityAccel);
            
            // Extract accelerometer and gyroscope residuals
            const Eigen::Vector3d accelResidual = residual.segment<3>(3);
            const Eigen::Vector3d gyroResidual = residual.head<3>();
            
            // Update error term
            error += accelResidual.transpose() * accelVarInv.asDiagonal() * accelResidual;
            error += gyroResidual.transpose() * gyroVarInv.asDiagonal() * gyroResidual;
            
            // Compute blocks for accelerometer residual
            const size_t gravityBlockStart = gravity_block_offset;
            const size_t numPoseJacobians = JaccelPose.d_val_d_knot.size();
            
            // Add Hessian and gradient contributions for each pose knot
            for (size_t i = 0; i < numPoseJacobians; i++) {
                // Compute the row index in the state vector
                size_t rowPose = (JaccelPose.start_idx + i) * POSE_SIZE;
                
                // Skip if this pose is fixed
                if (pose_fixed && rowPose < firstFixedKnot * POSE_SIZE) {
                    continue;
                }
                
                // Skip if out of bounds
                if (!isInBounds(rowPose, POSE_SIZE)) {
                    continue;
                }
                
                // Add gradient contribution from accelerometer residual
                accum.addB<POSE_SIZE>(
                    rowPose, 
                    JaccelPose.d_val_d_knot[i].transpose() * accelVarInv.asDiagonal() * accelResidual
                );
                
                // Add Hessian contributions for each pair of pose knots (i,j) with j<=i
                for (size_t j = 0; j <= i; j++) {
                    size_t colPose = (JaccelPose.start_idx + j) * POSE_SIZE;
                    
                    // Skip if this pose is fixed
                    if (pose_fixed && colPose < firstFixedKnot * POSE_SIZE) {
                        continue;
                    }
                    
                    // Skip if out of bounds
                    if (!isInBounds(colPose, POSE_SIZE)) {
                        continue;
                    }
                    
                    // Add Hessian block for pose-pose interaction
                    accum.addH<POSE_SIZE, POSE_SIZE>(
                        rowPose, colPose,
                        JaccelPose.d_val_d_knot[i].transpose() * 
                        accelVarInv.asDiagonal() * 
                        JaccelPose.d_val_d_knot[j]
                    );
                }
                
                // Add Hessian contributions for pose-bias interactions
                for (size_t j = 0; j < numPoseJacobians; j++) {
                    // Compute the bias column index
                    size_t colBiasAccel = bias_block_offset + 
                                         (JbiasBoth.start_idx + j) * BIAS_SIZE + 
                                         ACCEL_BIAS_OFFSET;
                    
                    // Skip if out of bounds
                    if (!isInBounds(colBiasAccel, ACCEL_BIAS_SIZE) || !isInBounds(rowPose, POSE_SIZE)) {
                        continue;
                    }
                    
                    // Add Hessian block for pose-bias interaction
                    accum.addH<ACCEL_BIAS_SIZE, POSE_SIZE>(
                        colBiasAccel, rowPose,
                        JbiasBoth.d_val_d_knot[j] * accelVarInv.asDiagonal() * JaccelPose.d_val_d_knot[i]
                    );
                }
                
                // Add Hessian contributions for pose-gravity interactions
                if (param->if_opt_g) {
                    if (isInBounds(gravityBlockStart, G_SIZE) && isInBounds(rowPose, POSE_SIZE)) {
                        accum.addH<G_SIZE, POSE_SIZE>(
                            gravityBlockStart, rowPose,
                            JgravityAccel.transpose() * accelVarInv.asDiagonal() * JaccelPose.d_val_d_knot[i]
                        );
                    }
                }
            }
            
            // Add Hessian and gradient contributions for bias terms
            const size_t numBiasJacobians = JbiasBoth.d_val_d_knot.size();
            for (size_t i = 0; i < numBiasJacobians; i++) {
                // Compute the row indices for accelerometer and gyroscope bias
                size_t rowBiasAccel = bias_block_offset + (JbiasBoth.start_idx + i) * BIAS_SIZE + ACCEL_BIAS_OFFSET;
                
                // Add Hessian contributions for bias-bias interactions (accelerometer)
                for (size_t j = 0; j <= i; j++) {
                    size_t colBiasAccel = bias_block_offset + (JbiasBoth.start_idx + j) * BIAS_SIZE + ACCEL_BIAS_OFFSET;
                    
                    if (isInBounds(rowBiasAccel, ACCEL_BIAS_SIZE) && isInBounds(colBiasAccel, ACCEL_BIAS_SIZE)) {
                        Eigen::Matrix3d JTwJ = JbiasBoth.d_val_d_knot[i] * accelVarInv.asDiagonal() * JbiasBoth.d_val_d_knot[j];
                        accum.addH<ACCEL_BIAS_SIZE, ACCEL_BIAS_SIZE>(rowBiasAccel, colBiasAccel, JTwJ);
                    }
                }
                
                // Add gradient contribution for accelerometer bias
                if (isInBounds(rowBiasAccel, ACCEL_BIAS_SIZE)) {
                    Eigen::Vector3d JTwr = JbiasBoth.d_val_d_knot[i] * accelVarInv.asDiagonal() * accelResidual;
                    accum.addB<ACCEL_BIAS_SIZE>(rowBiasAccel, JTwr);
                }
                
                // Add Hessian contributions for bias-gravity interactions
                if (param->if_opt_g) {
                    if (isInBounds(gravityBlockStart, G_SIZE) && isInBounds(rowBiasAccel, ACCEL_BIAS_SIZE)) {
                        accum.addH<G_SIZE, ACCEL_BIAS_SIZE>(
                            gravityBlockStart, rowBiasAccel, 
                            JgravityAccel.transpose() * accelVarInv.asDiagonal() * JbiasBoth.d_val_d_knot[i]
                        );
                    }
                }
            }
            
            // Add Hessian and gradient contributions for gravity parameters
            if (param->if_opt_g) {
                if (isInBounds(gravityBlockStart, G_SIZE)) {
                    // Add Hessian block for gravity-gravity interaction
                    accum.addH<G_SIZE, G_SIZE>(
                        gravityBlockStart, gravityBlockStart, 
                        JgravityAccel.transpose() * accelVarInv.asDiagonal() * JgravityAccel
                    );
                    
                    // Add gradient contribution for gravity
                    accum.addB<G_SIZE>(
                        gravityBlockStart, 
                        JgravityAccel.transpose() * accelVarInv.asDiagonal() * accelResidual
                    );
                }
            }
            
            // Process gyroscope measurements
            const size_t numGyroJacobians = JgyroPose.d_val_d_knot.size();
            
            // Add Hessian and gradient contributions for gyroscope residual
            for (size_t i = 0; i < numGyroJacobians; i++) {
                // Compute the row index for rotation component
                size_t rowRot = (JgyroPose.start_idx + i) * POSE_SIZE + ROT_OFFSET;
                
                // Skip if this pose is fixed
                if (pose_fixed && ((JgyroPose.start_idx + i) * POSE_SIZE) < firstFixedKnot * POSE_SIZE) {
                    continue;
                }
                
                // Skip if out of bounds
                if (!isInBounds(rowRot, ROT_SIZE)) {
                    continue;
                }
                
                // Add gradient contribution from gyroscope residual
                accum.addB<ROT_SIZE>(
                    rowRot,
                    JgyroPose.d_val_d_knot[i].transpose() * gyroVarInv.asDiagonal() * gyroResidual
                );
                
                // Add Hessian contributions for each pair of pose knots (i,j) with j<=i
                for (size_t j = 0; j <= i; j++) {
                    size_t colRot = (JgyroPose.start_idx + j) * POSE_SIZE + ROT_OFFSET;
                    
                    // Skip if this pose is fixed
                    if (pose_fixed && ((JgyroPose.start_idx + j) * POSE_SIZE) < firstFixedKnot * POSE_SIZE) {
                        continue;
                    }
                    
                    // Skip if out of bounds
                    if (!isInBounds(rowRot, ROT_SIZE) || !isInBounds(colRot, ROT_SIZE)) {
                        continue;
                    }
                    
                    // Add Hessian block for rotation-rotation interaction
                    accum.addH<ROT_SIZE, ROT_SIZE>(
                        rowRot, colRot,
                        JgyroPose.d_val_d_knot[i].transpose() * 
                        gyroVarInv.asDiagonal() * 
                        JgyroPose.d_val_d_knot[j]
                    );
                }
                
                // Add Hessian contributions for rotation-bias interactions
                for (size_t j = 0; j < numBiasJacobians; j++) {
                    size_t colBiasGyro = bias_block_offset + (JbiasBoth.start_idx + j) * BIAS_SIZE + GYRO_BIAS_OFFSET;
                    
                    if (isInBounds(colBiasGyro, GYRO_BIAS_SIZE) && isInBounds(rowRot, ROT_SIZE)) {
                        accum.addH<GYRO_BIAS_SIZE, ROT_SIZE>(
                            colBiasGyro, rowRot,
                            JbiasBoth.d_val_d_knot[j] * gyroVarInv.asDiagonal() * JgyroPose.d_val_d_knot[i]
                        );
                    }
                }
            }
            
            // Add Hessian and gradient contributions for gyroscope bias
            for (size_t i = 0; i < numBiasJacobians; i++) {
                // Compute the row index for gyroscope bias
                size_t rowBiasGyro = bias_block_offset + (JbiasBoth.start_idx + i) * BIAS_SIZE + GYRO_BIAS_OFFSET;
                
                // Add Hessian contributions for bias-bias interactions (gyroscope)
                for (size_t j = 0; j <= i; j++) {
                    size_t colBiasGyro = bias_block_offset + (JbiasBoth.start_idx + j) * BIAS_SIZE + GYRO_BIAS_OFFSET;
                    
                    if (isInBounds(rowBiasGyro, GYRO_BIAS_SIZE) && isInBounds(colBiasGyro, GYRO_BIAS_SIZE)) {
                        Eigen::Matrix3d JTwJ = JbiasBoth.d_val_d_knot[i] * gyroVarInv.asDiagonal() * JbiasBoth.d_val_d_knot[j];
                        accum.addH<GYRO_BIAS_SIZE, GYRO_BIAS_SIZE>(rowBiasGyro, colBiasGyro, JTwJ);
                    }
                }
                
                // Add gradient contribution for gyroscope bias
                if (isInBounds(rowBiasGyro, GYRO_BIAS_SIZE)) {
                    Eigen::Vector3d JTwr = JbiasBoth.d_val_d_knot[i] * gyroVarInv.asDiagonal() * gyroResidual;
                    accum.addB<GYRO_BIAS_SIZE>(rowBiasGyro, JTwr);
                }
            }
        }
        
        // Add regularization terms for consecutive bias values
        auto itFirst = imuMeasurements.begin();
        Jacobian biasJacobian0;
        Eigen::Matrix<double, 6, 1> bias0 = spline->interpolateBias((*itFirst).timestampNanoseconds, &biasJacobian0);
        int64_t time0 = (*itFirst).timestampNanoseconds;
        
        // Advance to second measurement
        ++itFirst;
        
        // Number of control points affecting the first bias interpolation
        size_t numJ0 = biasJacobian0.d_val_d_knot.size();
        
        // Process all IMU measurements for bias regularization
        while (itFirst != imuMeasurements.end()) {
            // Get bias and jacobian at current timestamp
            Jacobian biasJacobian1;
            Eigen::Matrix<double, 6, 1> bias1 = spline->interpolateBias((*itFirst).timestampNanoseconds, &biasJacobian1);
            int64_t time1 = (*itFirst).timestampNanoseconds;
            
            // Number of control points affecting this bias interpolation
            size_t numJ1 = biasJacobian1.d_val_d_knot.size();
            
            // Compute bias difference residuals
            Eigen::Vector3d biasAccelDiff = bias1.head<3>() - bias0.head<3>();
            Eigen::Vector3d biasGyroDiff = bias1.tail<3>() - bias0.tail<3>();
            
            // Add error contributions from bias differences
            error += biasAccelDiff.transpose() * biasAccelVarInv.asDiagonal() * biasAccelDiff;
            error += biasGyroDiff.transpose() * biasGyroVarInv.asDiagonal() * biasGyroDiff;
            
            // Compute control point index offset between the two timestamps
            size_t deltaIdx = biasJacobian1.start_idx - biasJacobian0.start_idx;
            deltaIdx = deltaIdx > 4 ? 4 : deltaIdx;
            
            // Maximum number of control points affecting either bias
            size_t maxNumCp = std::max(numJ0, numJ1);
            
            // Prepare combined Jacobian vector
            Eigen::aligned_vector<std::pair<size_t, double>> vJb(maxNumCp + deltaIdx);
            
            // Combine Jacobians from both timestamps
            for (size_t i = 0; i < maxNumCp; i++) {
                bool setIdx = false;
                
                // Add contribution from first bias point
                if (i < numJ0) {
                    vJb[i].first = biasJacobian0.start_idx + i;
                    setIdx = true;
                    vJb[i].second = -biasJacobian0.d_val_d_knot[i];
                }
                
                // Add contribution from second bias point
                if (i >= deltaIdx) {
                    if (!setIdx) {
                        vJb[i].first = biasJacobian1.start_idx + i - deltaIdx;
                    }
                    vJb[i].second += biasJacobian1.d_val_d_knot[i - deltaIdx];
                }
            }
            
            // Handle any remaining control points affecting only the second bias
            for (size_t i = 0; i < deltaIdx; i++) {
                vJb[i + maxNumCp].first = biasJacobian1.start_idx + i + maxNumCp - deltaIdx;
                vJb[i + maxNumCp].second = biasJacobian1.d_val_d_knot[maxNumCp - deltaIdx + i];
            }
            
            // Add Hessian and gradient contributions for bias regularization
            for (size_t i = 0; i < vJb.size(); i++) {
                // Compute block indices
                size_t biasBlockI = bias_block_offset + vJb[i].first * BIAS_SIZE;
                size_t biasAccelI = biasBlockI + ACCEL_BIAS_OFFSET;
                size_t biasGyroI = biasBlockI + GYRO_BIAS_OFFSET;
                
                // Process each pair of bias control points
                for (size_t j = 0; j <= i; j++) {
                    size_t biasBlockJ = bias_block_offset + vJb[j].first * BIAS_SIZE;
                    size_t biasAccelJ = biasBlockJ + ACCEL_BIAS_OFFSET;
                    size_t biasGyroJ = biasBlockJ + GYRO_BIAS_OFFSET;
                    
                    // Common factor for both bias types
                    double JTJ = vJb[i].second * vJb[j].second;
                    
                    // Weight by information matrices
                    Eigen::Matrix3d JTwbaJ = JTJ * biasAccelVarInv.asDiagonal();
                    Eigen::Matrix3d JTwbgJ = JTJ * biasGyroVarInv.asDiagonal();
                    
                    // Add Hessian blocks for accel bias
                    if (isInBounds(biasAccelI, ACCEL_BIAS_SIZE) && isInBounds(biasAccelJ, ACCEL_BIAS_SIZE)) {
                        accum.addH<ACCEL_BIAS_SIZE, ACCEL_BIAS_SIZE>(biasAccelI, biasAccelJ, JTwbaJ);
                    }
                    
                    // Add Hessian blocks for gyro bias
                    if (isInBounds(biasGyroI, GYRO_BIAS_SIZE) && isInBounds(biasGyroJ, GYRO_BIAS_SIZE)) {
                        accum.addH<GYRO_BIAS_SIZE, GYRO_BIAS_SIZE>(biasGyroI, biasGyroJ, JTwbgJ);
                    }
                }
                
                // Add gradient contributions
                if (isInBounds(biasAccelI, ACCEL_BIAS_SIZE)) {
                    accum.addB<ACCEL_BIAS_SIZE>(
                        biasAccelI, 
                        vJb[i].second * biasAccelVarInv.asDiagonal() * biasAccelDiff
                    );
                }
                
                if (isInBounds(biasGyroI, GYRO_BIAS_SIZE)) {
                    accum.addB<GYRO_BIAS_SIZE>(
                        biasGyroI, 
                        vJb[i].second * biasGyroVarInv.asDiagonal() * biasGyroDiff
                    );
                }
            }
            
            // Prepare for next iteration
            bias0 = bias1;
            biasJacobian0 = biasJacobian1;
            time0 = time1;
            numJ0 = numJ1;
            
            // Move to next measurement
            ++itFirst;
        }
    }

    /**
     * @brief Process a batch of pose measurements to update Hessian and gradient
     * 
     * @param poses Vector of pose measurements
     */
    void operator()(const Eigen::aligned_deque<PoseData>& poses) {
        // Skip if there are no pose measurements
        if (poses.empty()) {
            return;
        }

        // Configure weights based on parameters and measurement count
        const double invNumPoses = 1.0 / static_cast<double>(poses.size());
        const double weightPos = param->w_pose_pos * std::sqrt(invNumPoses);
        const double weightRot = param->w_pose_rot * std::sqrt(invNumPoses);

        // Build information matrix
        Eigen::Matrix<double, 6, 6> W = Eigen::Matrix<double, 6, 6>::Zero();
        W.diagonal().head<3>().setConstant(weightPos * weightPos);
        W.diagonal().tail<3>().setConstant(weightRot * weightRot);

        // Number of knots to fix
        constexpr size_t firstFixedKnot = 1;

        // Process each pose measurement
        for (const PoseData& pose : poses) {
            // Compute residual and Jacobian
            Jacobian66 poseJacobian;
            Eigen::Matrix<double, 6, 1> residual = Residuals::poseResidualJacobian(
                pose.timestampNanoseconds, 
                spline, 
                pose.position, 
                pose.orientation,
                &poseJacobian
            );

            // Add error contribution
            error += residual.transpose() * W * residual;

            // Number of control points affecting this pose
            const size_t numPoseJacobians = poseJacobian.d_val_d_knot.size();

            // Add Hessian and gradient contributions
            for (size_t i = 0; i < numPoseJacobians; ++i) {
                // Compute row index
                const size_t rowPose = (poseJacobian.start_idx + i) * POSE_SIZE;
                
                // Skip if out of bounds
                if (rowPose + POSE_SIZE > opt_size) {
                    continue;
                }
                
                // Skip if this pose is fixed
                if (pose_fixed && rowPose < firstFixedKnot * POSE_SIZE) {
                    continue;
                }

                // Add self-term to Hessian (i,i block)
                const Eigen::Matrix<double, 6, 6> JTWJii =
                    poseJacobian.d_val_d_knot[i].transpose() * W * poseJacobian.d_val_d_knot[i];
                accum.addH<POSE_SIZE, POSE
