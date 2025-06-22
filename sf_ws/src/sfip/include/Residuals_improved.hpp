#pragma once

#include "utils/math_tools.hpp"
#include "SplineState.hpp"

/**
 * @brief Class containing static methods for computing residuals and their Jacobians
 * 
 * Residuals represent the difference between predicted and measured values, used 
 * in optimization to minimize estimation error.
 */
class Residuals {
public:
    /**
     * @brief Compute the IMU residual without Jacobians
     * 
     * @param timeNanoseconds Timestamp in nanoseconds
     * @param spline Pointer to the spline state
     * @param accel Measured accelerometer data
     * @param gyro Measured gyroscope data
     * @param gravity Gravity vector in world frame
     * @return Eigen::Matrix<double, 6, 1> Residual vector [gyroResidual; accelResidual]
     */
    static Eigen::Matrix<double, 6, 1> imuResidual(
        int64_t timeNanoseconds, 
        const SplineState* spline,
        const Eigen::Vector3d* accel,  
        const Eigen::Vector3d* gyro, 
        const Eigen::Vector3d& gravity)
    {
        // Get orientation and angular velocity from spline
        Eigen::Quaterniond qInterpolated;
        Eigen::Vector3d rotVel;
        spline->interpolateQuaternion(timeNanoseconds, &qInterpolated, &rotVel);
        
        // Get bias from spline
        Eigen::Matrix<double, 6, 1> bias = spline->interpolateBias(timeNanoseconds);
        
        // Get linear acceleration from spline's second derivative (+ gravity)
        Eigen::Vector3d accelInWorldFrame = spline->interpolatePosition<2>(timeNanoseconds) + gravity;
        
        // Transform world frame acceleration to body frame
        Eigen::Matrix3d rotWorldToBody = qInterpolated.inverse().toRotationMatrix();
        Eigen::Vector3d accelInBodyFrame = rotWorldToBody * accelInWorldFrame;
        
        // Compute residuals: predicted - measured + bias
        Eigen::Vector3d gyroResidual = rotVel - *gyro + bias.tail<3>();
        Eigen::Vector3d accelResidual = accelInBodyFrame - *accel + bias.head<3>();
        
        // Combine into single residual vector
        Eigen::Matrix<double, 6, 1> residual;
        residual.head<3>() = gyroResidual;
        residual.tail<3>() = accelResidual;
        
        return residual;
    }

    /**
     * @brief Compute the IMU residual and its Jacobians
     * 
     * @param timeNanoseconds Timestamp in nanoseconds
     * @param spline Pointer to the spline state
     * @param accel Measured accelerometer data
     * @param gyro Measured gyroscope data
     * @param gravity Gravity vector in world frame
     * @param Jacc Output Jacobian of accelerometer residual w.r.t. control points
     * @param Jgyro Output Jacobian of gyroscope residual w.r.t. control points
     * @param Jbias Optional output Jacobian of residual w.r.t. bias parameters
     * @param Jgravity Optional output Jacobian of accelerometer residual w.r.t. gravity parameters
     * @return Eigen::Matrix<double, 6, 1> Residual vector [gyroResidual; accelResidual]
     */
    static Eigen::Matrix<double, 6, 1> imuResidualJacobian(
        int64_t timeNanoseconds, 
        const SplineState* spline,
        const Eigen::Vector3d* accel,  
        const Eigen::Vector3d* gyro, 
        const Eigen::Vector3d& gravity,
        Jacobian36* Jacc, 
        Jacobian33* Jgyro,
        Jacobian* Jbias = nullptr, 
        Eigen::Matrix<double, 3, 2>* Jgravity = nullptr)
    {
        // Get orientation, angular velocity, and their Jacobians
        Eigen::Quaterniond qInterpolated;
        Eigen::Vector3d rotVel;
        Jacobian43 Jrot;
        
        // Get Jacobians for quaternion and angular velocity
        spline->interpolateQuaternion(timeNanoseconds, &qInterpolated, &rotVel, &Jrot, Jgyro);
        
        // Get linear acceleration and its Jacobian
        Jacobian JlineAcc;
        Eigen::Vector3d accelInWorldFrameWithoutGravity = 
            spline->interpolatePosition<2>(timeNanoseconds, &JlineAcc);
        
        // Get bias and its Jacobian
        Eigen::Matrix<double, 6, 1> bias = 
            spline->interpolateBias(timeNanoseconds, Jbias);
        
        // Add gravity to get total acceleration in world frame
        Eigen::Vector3d accelInWorldFrameWithGravity = accelInWorldFrameWithoutGravity + gravity;
        
        // Transform from world to body frame
        Eigen::Matrix3d rotWorldToBody = qInterpolated.inverse().toRotationMatrix();
        
        // Compute residuals
        Eigen::Vector3d accelResidual = 
            rotWorldToBody * accelInWorldFrameWithGravity - *accel + bias.head<3>();
        Eigen::Vector3d gyroResidual = rotVel - *gyro + bias.tail<3>();
        
        // Combine into single residual vector
        Eigen::Matrix<double, 6, 1> residual;
        residual.head<3>() = gyroResidual;
        residual.tail<3>() = accelResidual;

        // Compute Jacobian of rotation w.r.t quaternion
        Eigen::Matrix<double, 3, 4> rotationJacobian;
        Quater::drot(accelInWorldFrameWithGravity, qInterpolated, rotationJacobian);
        
        // Set up accelerometer Jacobian structure
        Jacc->start_idx = JlineAcc.start_idx;
        int numJacobians = JlineAcc.d_val_d_knot.size();
        Jacc->d_val_d_knot.resize(numJacobians);
        
        // Fill Jacobian blocks for each control point
        for (int i = 0; i < numJacobians; i++) {
            // Initialize to zero
            Jacc->d_val_d_knot[i].setZero();
            
            // Position derivative contribution
            Jacc->d_val_d_knot[i].template topLeftCorner<3, 3>() =
                JlineAcc.d_val_d_knot[i] * rotWorldToBody;
            
            // Orientation derivative contribution
            Jacc->d_val_d_knot[i].template bottomRightCorner<3, 3>() =
                rotationJacobian * Jrot.d_val_d_knot[i];
        }
        
        // Compute gravity Jacobian if requested
        if (Jgravity) {
            (*Jgravity) = rotWorldToBody * Sphere::TangentBasis(gravity);
        }
        
        return residual;
    }

    /**
     * @brief Compute the pose residual without Jacobians
     * 
     * @param timeNanoseconds Timestamp in nanoseconds
     * @param spline Pointer to the spline state
     * @param measuredPosition Measured position
     * @param measuredOrientation Measured orientation
     * @return Eigen::Matrix<double, 6, 1> Residual vector [positionResidual; orientationResidual]
     */
    static Eigen::Matrix<double, 6, 1> poseResidual(
        int64_t timeNanoseconds, 
        const SplineState* spline, 
        const Eigen::Vector3d& measuredPosition,
        const Eigen::Quaterniond& measuredOrientation)
    {
        // Interpolate spline's predicted pose at the measurement timestamp
        Eigen::Vector3d predictedPosition = spline->interpolatePosition(timeNanoseconds);
        Eigen::Quaterniond predictedOrientation;
        spline->interpolateQuaternion(timeNanoseconds, &predictedOrientation);

        // Position residual (prediction - measurement)
        Eigen::Vector3d positionResidual = predictedPosition - measuredPosition;

        // Orientation residual: log(prediction^-1 * measurement)
        // This gives a rotation vector (axis-angle) representation of the error
        Eigen::Quaterniond errorQuaternion = predictedOrientation.inverse() * measuredOrientation;
        
        // Ensure shortest path rotation (non-negative real part)
        if (errorQuaternion.w() < 0.0) {
            errorQuaternion.coeffs() *= -1.0;
        }
        
        // Convert error quaternion to 3D axis-angle vector
        Eigen::Vector3d orientationResidual;
        Quater::log(errorQuaternion, orientationResidual);

        // Combine into a single residual vector
        Eigen::Matrix<double, 6, 1> residual;
        residual.head<3>() = positionResidual;
        residual.tail<3>() = orientationResidual;
        
        return residual;
    }

    /**
     * @brief Compute the pose residual and its Jacobian
     * 
     * @param timeNanoseconds Timestamp in nanoseconds
     * @param spline Pointer to the spline state
     * @param measuredPosition Measured position
     * @param measuredOrientation Measured orientation
     * @param JoutputPose Output Jacobian of the pose residual w.r.t. control points
     * @return Eigen::Matrix<double, 6, 1> Residual vector [positionResidual; orientationResidual]
     */
    static Eigen::Matrix<double, 6, 1> poseResidualJacobian(
        int64_t timeNanoseconds, 
        const SplineState* spline, 
        const Eigen::Vector3d& measuredPosition, 
        const Eigen::Quaterniond& measuredOrientation,
        Jacobian66* JoutputPose)
    {
        // Initialize residual vector
        Eigen::Matrix<double, 6, 1> residual; 

        // 1. Get predicted pose and Jacobians from spline
        Jacobian Jpos;
        Jacobian43 Jquat;
        
        Eigen::Vector3d predictedPosition = spline->interpolatePosition(timeNanoseconds, &Jpos);
        Eigen::Quaterniond predictedOrientation;
        spline->interpolateQuaternion(timeNanoseconds, &predictedOrientation, nullptr, &Jquat, nullptr);

        // 2. Position residual: r_pos = p_pred - p_meas
        Eigen::Vector3d positionResidual = predictedPosition - measuredPosition;
        residual.head<3>() = positionResidual;

        // 3. Orientation residual: r_ort = log(q_meas * q_pred^{-1})
        Eigen::Quaterniond errorQuaternion = predictedOrientation.inverse() * measuredOrientation;
        
        // Ensure shortest path rotation (non-negative real part)
        if (errorQuaternion.w() < 0.0) {
            errorQuaternion.coeffs() *= -1.0;
        }
        
        // Convert error quaternion to 3D axis-angle vector
        Eigen::Vector3d orientationResidual;
        Quater::log(errorQuaternion, orientationResidual);
        residual.tail<3>() = orientationResidual;

        // If Jacobian output requested, compute it
        if (JoutputPose) {
            const int numControlPoints = Jpos.d_val_d_knot.size();
            JoutputPose->d_val_d_knot.resize(numControlPoints);
            JoutputPose->start_idx = Jpos.start_idx;

            // Compute Jacobian of log map at error quaternion
            Eigen::Matrix<double, 3, 4> J_log;
            Quater::dlog(errorQuaternion, orientationResidual, J_log);

            // Compute d(q_err)/d(q_pred)
            Eigen::Matrix4d QL_pred_inv, QR_meas;
            Quater::Qleft(predictedOrientation.inverse(), QL_pred_inv);
            Quater::Qright(measuredOrientation, QR_meas);
            const Eigen::Matrix4d d_qerr__dqpred = -QR_meas * QL_pred_inv;

            // Compute Jacobian for each control point
            for (int i = 0; i < numControlPoints; ++i) {
                JoutputPose->d_val_d_knot[i].setZero();

                // Position Jacobian block: dr_pos/d(knot) = β * I_3x3
                double beta = Jpos.d_val_d_knot[i];
                JoutputPose->d_val_d_knot[i].template block<3, 3>(0, 0) = 
                    beta * Eigen::Matrix3d::Identity();

                // Orientation Jacobian block: dr_ort/d(knot) = J_log * d(q_err)/d(q_pred) * d(q_pred)/d(knot)
                const Eigen::Matrix<double, 4, 3>& Jq_i = Jquat.d_val_d_knot[i];
                JoutputPose->d_val_d_knot[i].template block<3, 3>(3, 3) = 
                    J_log * d_qerr__dqpred * Jq_i;
            }
        }

        return residual;
    }
};
