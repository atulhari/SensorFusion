#pragma once

#include "utils/math_tools.hpp"
#include "SplineState.hpp"

class Residuals
{

public:

    static Eigen::Matrix<double, 6, 1> imuResidual(int64_t timeNanoseconds, const SplineState* spline,
       const Eigen::Vector3d* accel,  const Eigen::Vector3d* gyro, const Eigen::Vector3d& gravity)
    {
        Eigen::Quaterniond qInterpolated;
        Eigen::Vector3d rotVel;
        spline->interpolateQuaternion(timeNanoseconds, &qInterpolated, &rotVel);
        Eigen::Matrix<double,6, 1> bias = spline->interpolateBias(timeNanoseconds);
        Eigen::Vector3d accelInWorldFrame = spline->interpolatePosition<2>(timeNanoseconds) + gravity;
        Eigen::Matrix3d rotWorldToBody = qInterpolated.inverse().toRotationMatrix();
        Eigen::Vector3d accelInBodyFrame = rotWorldToBody * accelInWorldFrame;
        Eigen::Vector3d gyroResidual = rotVel - *gyro + bias.tail<3>();
        Eigen::Vector3d accelResidual = accelInBodyFrame - *accel + bias.head<3>();
        Eigen::Matrix<double, 6, 1> residual;
        residual.head<3>() = gyroResidual;
        residual.tail<3>() = accelResidual;
        return residual;
    }

    static Eigen::Matrix<double, 6, 1> imuResidualJacobian(int64_t timeNanoseconds, const SplineState* spline,
        const Eigen::Vector3d* accel,  const Eigen::Vector3d* gyro, const Eigen::Vector3d& gravity,
        Jacobian36* Jacc, Jacobian33* Jgyro,
        Jacobian* Jbias = nullptr, Eigen::Matrix<double, 3, 2>* Jgravity = nullptr)
    {
        Eigen::Quaterniond qInterpolated;
        Eigen::Vector3d rotVel;
        Jacobian43 Jrot;
        Jacobian JlineAcc;     // Jacobian of spline position w.r.t scalar coeffs for pos knots
        
        spline->interpolateQuaternion(timeNanoseconds, &qInterpolated, &rotVel, &Jrot, Jgyro);
        
        Eigen::Vector3d accelInWorldFrameWithoutGravity = spline->interpolatePosition<2>(timeNanoseconds, &JlineAcc);
        
        Eigen::Matrix<double,6, 1> bias = spline->interpolateBias(timeNanoseconds, Jbias); // Assuming J_bias_wrt_bias_knots_coeffs is also Jacobian type for scalar coeffs
        Eigen::Vector3d accelInWorldFrameWithGravity = accelInWorldFrameWithoutGravity + gravity;
        Eigen::Matrix3d rotWorldToBody = qInterpolated.inverse().toRotationMatrix();
        
        Eigen::Vector3d accelResidual = rotWorldToBody * accelInWorldFrameWithGravity - *accel + bias.head<3>();
        Eigen::Vector3d gyroResidual = rotVel - *gyro + bias.tail<3>();
        
        Eigen::Matrix<double, 6, 1> residual;
        residual.head<3>() = gyroResidual;
        residual.tail<3>() = accelResidual;

        Eigen::Matrix<double, 3, 4> tmp;
        Quater::drot(accelInWorldFrameWithGravity, qInterpolated, tmp);
        Jacc->start_idx = JlineAcc.start_idx;
        int numJ = JlineAcc.d_val_d_knot.size();
        Jacc->d_val_d_knot.resize(numJ);
        for (int i = 0; i < numJ; i++) {
            // Ensure full matrix is initialised to zero before setting blocks
            Jacc->d_val_d_knot[i].setZero();
            Jacc->d_val_d_knot[i].template topLeftCorner<3, 3>() =
                JlineAcc.d_val_d_knot[i] * rotWorldToBody;
            Jacc->d_val_d_knot[i].template bottomRightCorner<3,3>() =
              tmp * Jrot.d_val_d_knot[i];
        }
        if (Jgravity) {
          (*Jgravity) = rotWorldToBody * Sphere::TangentBasis(gravity);
        }
        return residual;
    }


    static Eigen::Matrix<double, 6, 1> poseResidual(
        int64_t timeNanoseconds, 
        const SplineState* spline, 
        const Eigen::Vector3d& measuredPosition,      // Pass by const reference
        const Eigen::Quaterniond& measuredOrientation) // Pass by const reference
    {
        // Interpolate spline's predicted pose at the measurement timestamp
        Eigen::Vector3d predictedPosition = spline->interpolatePosition(timeNanoseconds);
        Eigen::Quaterniond predictedOrientation;
        spline->interpolateQuaternion(timeNanoseconds, &predictedOrientation);

        // Position residual (prediction - measurement)
        Eigen::Vector3d positionResidual = predictedPosition - measuredPosition;

        // Orientation residual (Log(measuredOrientation * predictedOrientation_inverse))
        Eigen::Quaterniond errorQuaternion = predictedOrientation.inverse() * measuredOrientation;
        if (errorQuaternion.w() < 0.0) {
            errorQuaternion.coeffs() *= -1.0; // Ensure scalar part is non-negative for shortest angle
        }
        Eigen::Vector3d orientationResidual;
        Quater::log(errorQuaternion, orientationResidual); // Convert error quaternion to 3D axis-angle vector

        Eigen::Matrix<double, 6, 1> residual;
        residual.head<3>() = positionResidual;
        residual.tail<3>() = orientationResidual;
        
        return residual;
    }

    // Function to calculate pose residual and its Jacobians
    static Eigen::Matrix<double, 6, 1> poseResidualJacobian(
        int64_t timeNanoseconds, 
        const SplineState* spline, 
        const Eigen::Vector3d& measuredPosition, 
        const Eigen::Quaterniond& measuredOrientation,
        Jacobian66* JoutputPose)
    {
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
        if (errorQuaternion.w() < 0.0) {
            errorQuaternion.coeffs() *= -1.0; // Ensure scalar part is non-negative for shortest angle
        }
        Eigen::Vector3d orientationResidual;
        Quater::log(errorQuaternion, orientationResidual);
        residual.tail<3>() = orientationResidual;

        if (JoutputPose) {
            const int K = Jpos.d_val_d_knot.size();
            JoutputPose->d_val_d_knot.resize(K);
            JoutputPose->start_idx = Jpos.start_idx;

            // Compute Jacobian of log map at error quaternion
            Eigen::Matrix<double, 3, 4> J_log;
            Quater::dlog(errorQuaternion, orientationResidual, J_log);

            // Compute d(q_err)/d(q_pred)
            Eigen::Matrix4d QL_pred_inv, QR_meas;
            Quater::Qleft(predictedOrientation.inverse(), QL_pred_inv);
            Quater::Qright(measuredOrientation, QR_meas);
            const Eigen::Matrix4d d_qerr__dqpred = -QR_meas * QL_pred_inv;

            for (int i = 0; i < K; ++i) {
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
