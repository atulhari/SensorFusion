#pragma once

#include "utils/common_utils.hpp"
#include "utils/math_tools.hpp"
#include "sfip/Spline.h"
#include "sfip/Knot.h"

template <class MatT>
struct JacobianStruct {
    size_t start_idx;
    std::vector<MatT> d_val_d_knot;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef JacobianStruct<double> Jacobian;
typedef JacobianStruct<Eigen::Matrix<double, 4, 3>> Jacobian43;
typedef JacobianStruct<Eigen::Matrix3d> Jacobian33;
typedef JacobianStruct<Eigen::Vector3d> Jacobian13;
typedef JacobianStruct<Eigen::Matrix<double, 6, 1>> Jacobian16;
typedef JacobianStruct<Eigen::Matrix<double, 3, 6>> Jacobian36;
typedef JacobianStruct<Eigen::Matrix<double, 6, 6>> Jacobian66;

class SplineState
{

  public:

    SplineState() {};

    void init(int64_t knotIntervalNanoseconds_, int numKnots_, int64_t startTimeNanoseconds_, int startIndex_ = 0)
    {
        // Initialize parameters
        isFirst = true;
        knotIntervalNanoseconds = knotIntervalNanoseconds_;
        startTimeNanoseconds = startTimeNanoseconds_;
        numKnots = numKnots_;
        invDt = 1e9 / knotIntervalNanoseconds;
        startIndex = startIndex_;

        // Precompute powers of invDt for derivatives
        powInvDt[0] = 1.0;
        powInvDt[1] = invDt;
        powInvDt[2] = invDt * invDt;
        powInvDt[3] = powInvDt[2] * invDt;

        // Initialize idle state
        Eigen::Quaterniond q0 = Eigen::Quaterniond::Identity();
        Eigen::Vector3d t0 = Eigen::Vector3d::Zero();
        Eigen::Matrix<double, 6, 1> b0 = Eigen::Matrix<double, 6, 1>::Zero();
        qIdle = {q0, q0, q0};
        tIdle = {t0, t0, t0};
        bIdle = {b0, b0, b0};
    }

    // Manually set a single knot
    void setSingleStateKnot(int knotIndex, Eigen::Quaterniond quaternion, Eigen::Vector3d position, Eigen::Matrix<double, 6, 1> bias)
    {
        tKnots[knotIndex] = position;
        qKnots[knotIndex] = quaternion;
        bKnots[knotIndex] = bias;
    }

    // Update the knots from another SplineState object (Verify requirements)
    void updateKnots(SplineState* other)
    {
        size_t numKnots = other->getNumKnots();
        for (size_t i = 0; i < numKnots; i++) {
            if (i + other->startIndex < numKnots) {
                setSingleStateKnot(i + other->startIndex, other->qKnots[i],other->tKnots[i],other->bKnots[i]);
            } else {
                addSingleStateKnot(other->qKnots[i],other->tKnots[i],other->bKnots[i]);
            }
        }
        qIdle = other->qIdle;
        tIdle = other->tIdle;
        bIdle = other->bIdle;
    }

    // Get the idle position
    Eigen::Vector3d getIdlePosition(int idx)
    {
        return tIdle[idx];
    }

    // Set the idle position
    void setIdles(int idx, Eigen::Vector3d position, Eigen::Quaterniond quaternion, Eigen::Matrix<double, 6, 1> bias)
    {
        tIdle[idx] = position;
        qIdle[idx] = quaternion;
        bIdle[idx] = bias;
    }

    // Update the bias of the idle state
    void updateBiasIdleFirstWindow()
    {
        if (!isFirst) {
            return;
        } else {
            bIdle = {bKnots[0], bKnots[0], bKnots[0]};
        }
    }

    // Add a single new knot to the end of the spline deque
    void addSingleStateKnot(Eigen::Quaterniond quaternion, Eigen::Vector3d position, Eigen::Matrix<double, 6, 1> bias)
    {
        if (numKnots > 1) {
            Eigen::Quaterniond q0 = qKnots[numKnots - 1];
            Eigen::Quaterniond q1 = quaternion;
            double dotProduct = q0.dot(q1);

            // If the new knot is on the opposite side of the unit quaternion, we change the sign of the quaternion to make it the same side
            if (dotProduct < 0) {
               quaternion = Eigen::Quaterniond(-quaternion.w(), -quaternion.x(), -quaternion.y(), -quaternion.z());
            }
        }

        // If the quaternion is not a unit quaternion, we normalize it
        if (abs(quaternion.norm() - 1) > 1e-5) {
            quaternion.normalize();
        }

        // Add the new knot to the end of the spline deque
        qKnots.push_back(quaternion);
        tKnots.push_back(position);
        bKnots.push_back(bias);
        numKnots++;
    }

    // Check if the quaternion control points are on the same side of the unit quaternion and if not, 
    // we change the sign of the quaternion to make it the same side
    // This is done to avoid discontinuities in the spline after interpolation or optimization
    void checkQuaternionControlPoints()
    {
        if (numKnots > 1) {
            for (size_t i = 1; i < numKnots; i++) {
                Eigen::Quaterniond q1 = qKnots[i];
                // keep same hemisphere as previous knot
                if (qKnots[i - 1].dot(q1) < 0) {
                    qKnots[i] = Eigen::Quaterniond(-q1.w(), -q1.x(), -q1.y(), -q1.z());
                }
            }
        }
        // additionally, enforce w>0 on all knots to keep global sign consistency
        for (size_t i = 0; i < numKnots; ++i) {
            if (qKnots[i].w() < 0) {
                qKnots[i].coeffs() *= -1;
            }
        }
    }

    // Remove the oldest knot from the spline deque
    void removeSingleOldState()
    {   
        // Update the idle state, idle[0] is the oldest knot and is removed and 
        // qKnots.front() is the knot that will be moved from the active window to the idle state
        qIdle = {qIdle[1], qIdle[2], qKnots.front()};
        tIdle = {tIdle[1], tIdle[2], tKnots.front()};
        bIdle = {bIdle[1], bIdle[2], bKnots.front()};

        // Remove the oldest knot from the active window
        qKnots.pop_front();
        tKnots.pop_front();
        bKnots.pop_front();

        numKnots--;
        startIndex++;
        startTimeNanoseconds += knotIntervalNanoseconds;
        isFirst = false;
    }

    // Get all the knots from the active spline window
    void getAllStateKnots(Eigen::aligned_deque<Eigen::Vector3d>& knotsTrans,
        Eigen::aligned_deque<Eigen::Quaterniond>& knotsRot,
        Eigen::aligned_deque<Eigen::Matrix<double,6, 1>>& knotsBias)
    {
        knotsTrans = tKnots;
        knotsRot = qKnots;
        knotsBias = bKnots;
    }

    // Set all the knots from another SplineState object
    void setAllKnots(Eigen::aligned_deque<Eigen::Vector3d>& knotsTrans,
        Eigen::aligned_deque<Eigen::Quaterniond>& knotsRot,
        Eigen::aligned_deque<Eigen::Matrix<double,6, 1>>& knotsBias)
    {
        tKnots = knotsTrans;
        qKnots = knotsRot;
        bKnots = knotsBias;

        // Update the bias of the idle state
        updateBiasIdleFirstWindow();
    }

    // Get the absolute time of the knot at knotIndex
    int64_t getKnotTimeNanoseconds(size_t knotIndex) const
    {
        return startTimeNanoseconds + knotIndex * knotIntervalNanoseconds;
    }

    // Get the quaternion of the knot at knotIndex
    Eigen::Quaterniond getKnotOrientation(size_t knotIndex)
    {
        return qKnots[knotIndex];
    }

    // Get the position of the knot at knotIndex
    Eigen::Vector3d getKnotPosition(size_t knotIndex)
    {
        return tKnots[knotIndex];
    }

    // Get the bias of the knot at knotIndex
    Eigen::Matrix<double, 6, 1> getKnotBias(size_t knotIndex)
    {
        return bKnots[knotIndex];
    }

    // Extrapolate the position of the knot at knotIndex to predict ahead of time providing initial guess for the position of next knot
    // We will assume a constant velocty model for the extrapolation
    // Here index 
    Eigen::Vector3d extrapolateKnotPosition(size_t knotIndex)
    {
        Eigen::Quaterniond lastOrientation = qKnots[numKnots - knotIndex - 1];
        Eigen::Quaterniond currentOrientation = qKnots[numKnots - knotIndex];
        Eigen::Vector3d lastPosition = tKnots[numKnots - knotIndex - 1];
        Eigen::Vector3d currentPosition = tKnots[numKnots - knotIndex];
        Eigen::Vector3d relativeTranslation = lastOrientation.inverse() * (currentPosition - lastPosition); // dR = R.inv * (t_i - t_{i-1})
        return currentPosition + currentOrientation * relativeTranslation; // t_i+1 = t_i + R * dR
    }

    // Extrapolate the orientation of the knot at knotIndex to predict ahead of time providing initial guess for the orientation of next knot
    // We will assume a constant angular velocity model for the extrapolation
    Eigen::Quaterniond extrapolateKnotOrientation(size_t knotIndex)
    {
        Eigen::Quaterniond lastOrientation = qKnots[numKnots - knotIndex - 1];
        Eigen::Quaterniond currentOrientation = qKnots[numKnots - knotIndex];
        Eigen::Quaterniond relativeRotation = currentOrientation * lastOrientation.inverse();
        return relativeRotation * currentOrientation;
    }

    // Apply a pose increment to the knot at knotIndex
    void applyPoseIncrement(int knotIndex, const Eigen::Matrix<double, 6, 1> &poseIncrement)
    {
        tKnots[knotIndex] += poseIncrement.head<3>(); // first 3 elements of poseIncrement are the translation increment
        Eigen::Quaterniond qIncrement;
        Quater::exp(poseIncrement.tail<3>(), qIncrement); // last 3 elements of poseIncrement are the rotation increment
        qKnots[knotIndex] *= qIncrement;
    }

    // Apply a bias increment to the knot at knotIndex
    void applyBiasIncrement(int knotIndex, const Eigen::Matrix<double, 6, 1>& biasIncrement)
    {
        bKnots[knotIndex] += biasIncrement; // has 6 elements, first 3 are the acceleration increment, last 3 are the gyro increment
    }

    // Get the maximum time of the spline
    int64_t maxTimeNanoseconds()
    {
        if (numKnots == 1) {
           return startTimeNanoseconds;
        }
        return startTimeNanoseconds + (numKnots - 1) * knotIntervalNanoseconds - 1;
    }

    // Get the minimum time of the spline
    int64_t minTimeNanoseconds()
    {
        return startTimeNanoseconds + knotIntervalNanoseconds * (!isFirst ?  -1 : 0);
    }

    // Get the maximum time of the next spline knots
    int64_t nextMaxTimeNanoseconds()
    {
        return startTimeNanoseconds + numKnots * knotIntervalNanoseconds - 1;
    }

    // Get the number of knots in the spline  (previously numKnots)
    size_t getNumKnots()
    {
        return numKnots;
    }

    template <int Derivative = 0>
    // Interpolate the position of the spline at a given time
    Eigen::Vector3d interpolatePosition (int64_t timeNanoseconds, Jacobian* Jacobian = nullptr) const
    {
        return interpolateEuclidean<Eigen::Vector3d, Derivative>(timeNanoseconds, tIdle, tKnots, Jacobian);
    }

    // Interpolate the bias of the spline at a given time
    Eigen::Matrix<double, 6, 1> interpolateBias (int64_t timeNanoseconds, Jacobian* Jacobian = nullptr) const
    {
        return interpolateEuclidean<Eigen::Matrix<double, 6, 1>>(timeNanoseconds, bIdle, bKnots, Jacobian);
    }

    // Interpolate the quaternion of the spline at a given time
    void interpolateQuaternion(int64_t timeNanoseconds, Eigen::Quaterniond* quaternionOut = nullptr,
        Eigen::Vector3d* angularVelocityOut = nullptr, Jacobian43* quaternionJacobian = nullptr, Jacobian33* angularVelocityJacobian = nullptr) const
    {
        double normalizedTime;
        int64_t startingIndex;
        int indexRight;
        std::array<Eigen::Quaterniond, 4> controlPoints;

        // Prepare the interpolation of the quaternion of the spline at a given time
        prepareInterpolation(timeNanoseconds, qIdle, qKnots, startingIndex, normalizedTime, controlPoints, indexRight);

        Eigen::Vector4d basisVector;
        Eigen::Vector4d coefficients;
        baseCoeffsWithTime<0>(basisVector, normalizedTime);
        coefficients = cumulativeBlendingMatrix * basisVector;

        // Compute the delta of the quaternion, relative rotation between consecutive control points
        Eigen::Quaterniond qDelta[3];
        qDelta[0] = controlPoints[0].inverse() * controlPoints[1];
        qDelta[1] = controlPoints[1].inverse() * controlPoints[2];
        qDelta[2] = controlPoints[2].inverse() * controlPoints[3];

        Eigen::Vector3d tDelta[3];
        Eigen::Vector3d tDeltaScale[3];
        Eigen::Quaterniond qDeltaScale[3];
        Eigen::Quaterniond qInterpolated[4];
        Eigen::Vector3d angularVelocityInterpolated[4];
        Eigen::Vector4d dcoeff;
        if (quaternionJacobian || angularVelocityJacobian) {
            Eigen::Matrix<double, 3, 4> dlog_dq[3];
            Eigen::Matrix<double, 4, 3> dexp_dt[3];
            Quater::dlog(qDelta[0], tDelta[0], dlog_dq[0]);
            Quater::dlog(qDelta[1], tDelta[1], dlog_dq[1]);
            Quater::dlog(qDelta[2], tDelta[2], dlog_dq[2]);
            tDeltaScale[0] = tDelta[0] * coefficients[1];
            tDeltaScale[1] = tDelta[1] * coefficients[2];
            tDeltaScale[2] = tDelta[2] * coefficients[3];
            Quater::dexp(tDeltaScale[0], qDeltaScale[0], dexp_dt[0]);
            Quater::dexp(tDeltaScale[1], qDeltaScale[1], dexp_dt[1]);
            Quater::dexp(tDeltaScale[2], qDeltaScale[2], dexp_dt[2]);
            int size_J = std::min(indexRight + 1, 4);
            Eigen::Matrix4d d_X_d_dj[3];
            Eigen::Matrix<double, 3, 4> d_r_d_dj[3];
            Eigen::Quaterniond q_r_all[4];
            q_r_all[3] = Eigen::Quaterniond::Identity();
            for (int i = 2; i >= 0; i-- ) {
                q_r_all[i] = qDeltaScale[i] * q_r_all[i+1];
            }
            Eigen::Matrix4d Q_l[size_J - 1];
            Eigen::Matrix4d Q_r[size_J - 1];
            for (int i = 0; i < size_J - 1; i++) {
                Quater::Qleft(qDelta[4 - size_J + i], Q_l[i]);
                Quater::Qright(qDelta[4 - size_J + i], Q_r[i]);
            }
            if (quaternionJacobian) {
                qInterpolated[0] = controlPoints[0];
                qInterpolated[1] = qInterpolated[0] * qDeltaScale[0];
                qInterpolated[2] = qInterpolated[1] * qDeltaScale[1];
                qInterpolated[3] = qInterpolated[2] * qDeltaScale[2];
                *quaternionOut = qInterpolated[3];
                Eigen::Matrix4d Q_l_all[3];
                Quater::Qleft(qInterpolated[0], Q_l_all[0]);
                Quater::Qleft(qInterpolated[1], Q_l_all[1]);
                Quater::Qleft(qInterpolated[2], Q_l_all[2]);
                for (int i = 2; i >= 0; i--) {
                    Eigen::Matrix4d Q_r_all;
                    Quater::Qright(q_r_all[i+1], Q_r_all);
                    d_X_d_dj[i].noalias() = coefficients[i + 1] * Q_r_all * Q_l_all[i] * dexp_dt[i] * dlog_dq[i];
                }
                quaternionJacobian->d_val_d_knot.resize(size_J);
                for (int i = 0; i < size_J; i++) {
                    quaternionJacobian->d_val_d_knot[i].setZero();
                }
                for (int i = 0; i < size_J - 1; i++) {
                    quaternionJacobian->d_val_d_knot[i].noalias() -= d_X_d_dj[4 - size_J + i] * Q_r[i].rightCols(3);
                    quaternionJacobian->d_val_d_knot[i + 1].noalias() += d_X_d_dj[4 - size_J + i] * Q_l[i].rightCols(3);
                }
                quaternionJacobian->start_idx = startIndex;
                if (size_J == 4) {
                    Eigen::Matrix4d Q_r_all;
                    Eigen::Matrix4d Q0_left;
                    Quater::Qright(q_r_all[0], Q_r_all);
                    Quater::Qleft(controlPoints[0], Q0_left);
                    quaternionJacobian->d_val_d_knot[0].noalias() += Q_r_all * Q0_left.rightCols(3);
                } else {
                    Eigen::Matrix4d Q_left;
                    Quater::Qleft(qDelta[3 - size_J], Q_left);
                    quaternionJacobian->d_val_d_knot[0].noalias() += d_X_d_dj[3 - size_J] * Q_left.rightCols(3);
                }
            }
            if (angularVelocityJacobian) {
                baseCoeffsWithTime<1>(basisVector, normalizedTime);
                dcoeff = invDt * cumulativeBlendingMatrix * basisVector;
                angularVelocityInterpolated[0].setZero();
                angularVelocityInterpolated[1] = 2 * dcoeff[1] * tDelta[0];
                angularVelocityInterpolated[2] = qDeltaScale[1].inverse() * angularVelocityInterpolated[1] + 2 * dcoeff[2] * tDelta[1];
                angularVelocityInterpolated[3] = qDeltaScale[2].inverse() * angularVelocityInterpolated[2] + 2 * dcoeff[3] * tDelta[2];
                *angularVelocityOut = angularVelocityInterpolated[3];
                Eigen::Matrix<double, 3, 4> drot_dq[3];
                Quater::drot(angularVelocityInterpolated[0], qDeltaScale[0], drot_dq[0]);
                Quater::drot(angularVelocityInterpolated[1], qDeltaScale[1], drot_dq[1]);
                Quater::drot(angularVelocityInterpolated[2], qDeltaScale[2], drot_dq[2]);
                for (int i = 2; i >= 0; i--) {
                    Eigen::Matrix3d d_vel_d_dj = coefficients[i + 1] * drot_dq[i] * dexp_dt[i];
                    d_vel_d_dj.noalias() += 2 * dcoeff[i + 1] * Eigen::Matrix3d::Identity();
                    d_r_d_dj[i].noalias() = q_r_all[i+1].inverse().toRotationMatrix() * d_vel_d_dj * dlog_dq[i];
                }
                angularVelocityJacobian->d_val_d_knot.resize(size_J);
                for (int i = 0; i < size_J; i++) {
                    angularVelocityJacobian->d_val_d_knot[i].setZero();
                }
                for (int i = 0; i < size_J - 1; i++) {
                    angularVelocityJacobian->d_val_d_knot[i].noalias() -= d_r_d_dj[4 - size_J + i] * Q_r[i].rightCols(3);
                    angularVelocityJacobian->d_val_d_knot[i + 1].noalias() += d_r_d_dj[4 - size_J + i] * Q_l[i].rightCols(3);
                }
                angularVelocityJacobian->start_idx = startIndex;
                if (size_J != 4) {
                    Eigen::Matrix4d Q_left;
                    Quater::Qleft(qDelta[4 - size_J - 1], Q_left);
                    angularVelocityJacobian->d_val_d_knot[0].noalias() += d_r_d_dj[3 - size_J] *  Q_left.rightCols(3);
                }
            }
        } else {
            Quater::log(qDelta[0], tDelta[0]);
            Quater::log(qDelta[1], tDelta[1]);
            Quater::log(qDelta[2], tDelta[2]);
            tDeltaScale[0] = tDelta[0] * coefficients[1];
            tDeltaScale[1] = tDelta[1] * coefficients[2];
            tDeltaScale[2] = tDelta[2] * coefficients[3];
            Quater::exp(tDeltaScale[0], qDeltaScale[0]);
            Quater::exp(tDeltaScale[1], qDeltaScale[1]);
            Quater::exp(tDeltaScale[2], qDeltaScale[2]);
            if (quaternionOut) {
                qInterpolated[0] = controlPoints[0];
                qInterpolated[1] = qInterpolated[0] * qDeltaScale[0];
                qInterpolated[2] = qInterpolated[1] * qDeltaScale[1];
                qInterpolated[3] = qInterpolated[2] * qDeltaScale[2];
                qInterpolated[3].normalize();
                *quaternionOut = qInterpolated[3];
            }
            if (angularVelocityOut) {
                baseCoeffsWithTime<1>(basisVector, normalizedTime);
                dcoeff = invDt * cumulativeBlendingMatrix * basisVector;

                angularVelocityInterpolated[0].setZero();
                angularVelocityInterpolated[1] = 2 * dcoeff[1] * tDelta[0];
                angularVelocityInterpolated[2] = qDeltaScale[1].inverse() * angularVelocityInterpolated[1] + 2 * dcoeff[2] * tDelta[1];
                angularVelocityInterpolated[3] = qDeltaScale[2].inverse() * angularVelocityInterpolated[2] + 2 * dcoeff[3] * tDelta[2];
                *angularVelocityOut = angularVelocityInterpolated[3];
            }
        }
    }


    void getSplineMsg(sfip::Spline& spline_msg)
    {
        spline_msg.dt = knotIntervalNanoseconds;
        spline_msg.start_t = startTimeNanoseconds;
        spline_msg.start_idx = startIndex;
        for (size_t i = 0; i < numKnots; i++) {
            sfip::Knot knot_msg;
            knot_msg.position.x = tKnots[i].x();
            knot_msg.position.y = tKnots[i].y();
            knot_msg.position.z = tKnots[i].z();
            knot_msg.orientation.w = qKnots[i].w();
            knot_msg.orientation.x = qKnots[i].x();
            knot_msg.orientation.y = qKnots[i].y();
            knot_msg.orientation.z = qKnots[i].z();
            Eigen::Matrix<double, 6, 1> bias = bKnots[i];
            knot_msg.bias_acc.x = bias[0];
            knot_msg.bias_acc.y = bias[1];
            knot_msg.bias_acc.z = bias[2];
            knot_msg.bias_gyro.x = bias[3];
            knot_msg.bias_gyro.y = bias[4];
            knot_msg.bias_gyro.z = bias[5];
            spline_msg.knots.push_back(knot_msg);
        }
        for (int i = 0; i < 3; i++) {
            sfip::Knot idle_msg;
            idle_msg.position.x = tIdle[i].x();             // Corrected from t_idle
            idle_msg.position.y = tIdle[i].y();             // Corrected from t_idle
            idle_msg.position.z = tIdle[i].z();             // Corrected from t_idle
            idle_msg.orientation.w = qIdle[i].w();          // Corrected from q_idle
            idle_msg.orientation.x = qIdle[i].x();          // Corrected from q_idle
            idle_msg.orientation.y = qIdle[i].y();          // Corrected from q_idle
            idle_msg.orientation.z = qIdle[i].z();          // Corrected from q_idle
            Eigen::Matrix<double, 6, 1> bias = bIdle[i];    // Corrected from b_idle
            idle_msg.bias_acc.x = bias[0];
            idle_msg.bias_acc.y = bias[1];
            idle_msg.bias_acc.z = bias[2];
            idle_msg.bias_gyro.x = bias[3];
            idle_msg.bias_gyro.y = bias[4];
            idle_msg.bias_gyro.z = bias[5];
            spline_msg.idles.push_back(idle_msg);
        }
    }


    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:

    bool isFirst;

    static constexpr double secondsToNanoseconds = 1e9;
    static const Eigen::Matrix4d blendingMatrix;
    static const Eigen::Matrix4d baseCoefficients;
    static const Eigen::Matrix4d cumulativeBlendingMatrix;

    int64_t knotIntervalNanoseconds;
    double invDt;
    std::array<double, 4> powInvDt;
    int numKnots;
    int64_t startIndex;
    int64_t startTimeNanoseconds;

    std::array<Eigen::Quaterniond, 3> qIdle;
    std::array<Eigen::Vector3d, 3> tIdle;
    std::array<Eigen::Matrix<double, 6, 1>, 3> bIdle;
    Eigen::aligned_deque<Eigen::Quaterniond> qKnots;
    Eigen::aligned_deque<Eigen::Vector3d> tKnots;
    Eigen::aligned_deque<Eigen::Matrix<double, 6, 1>> bKnots;

    // Interpolate the position of the spline at a given time
    template <typename _KnotT, int Derivative = 0>
    _KnotT interpolateEuclidean(int64_t timestampNanoseconds, const std::array<_KnotT, 3>& knotsIdle,
                        const Eigen::aligned_deque<_KnotT>& knots, Jacobian* J = nullptr) const
    {
        double normalizedTime;
        int64_t startingIndex;
        int indexRight;
        std::array<_KnotT,4> controlPoints;
        prepareInterpolation(timestampNanoseconds, knotsIdle, knots, startingIndex, normalizedTime, controlPoints, indexRight);
        Eigen::Vector4d basisVector;
        Eigen::Vector4d coefficients;
        baseCoeffsWithTime<Derivative>(basisVector, normalizedTime);
        coefficients = powInvDt[Derivative] * (blendingMatrix * basisVector);
        _KnotT res_out = coefficients[0] * controlPoints[0] + coefficients[1] * controlPoints[1] + coefficients[2] * controlPoints[2] + coefficients[3] * controlPoints[3];
        if (J) {
            int size_J = std::min(indexRight + 1, 4);
            J->d_val_d_knot.resize(size_J);
            for (int i = 0; i < size_J; i++) {
                J->d_val_d_knot[i] = coefficients[4 - size_J + i];
            }
            J->start_idx = startingIndex;
        }
        return res_out;
    }


    // Prepare the interpolation of the spline at a given time
    template<typename _KnotT>
    void prepareInterpolation(int64_t timestampNanoseconds, const std::array<_KnotT, 3>& knotsIdle,
                              const Eigen::aligned_deque<_KnotT>& knots, int64_t& startingIndex, double& normalizedTime,
                              std::array<_KnotT,4>& controlPoints, int& indexRight) const
    {
        int64_t timestampNanosecondsRelative = timestampNanoseconds - startTimeNanoseconds;
        int indexLeft = floor(double(timestampNanosecondsRelative) / double(knotIntervalNanoseconds));
        indexRight = indexLeft + 1;
        startingIndex = std::max(indexLeft - 2, 0);
        for (int i = 0; i < 2 - indexLeft; i++) {
            controlPoints[i] = knotsIdle[i + indexLeft + 1];
        }
        int indexWindow = std::max(0, 2 - indexLeft);
        for (int i = 0; i < std::min(indexLeft + 2, 4); i++) {
            controlPoints[i + indexWindow] = knots[startingIndex + i];
        }
        normalizedTime = (timestampNanoseconds - startTimeNanoseconds - indexLeft * knotIntervalNanoseconds) / double(knotIntervalNanoseconds);
    }

    // Compute the B-spline blending coefficients for a given time
    template <int Derivative, class Derived>
    static void baseCoeffsWithTime(const Eigen::MatrixBase<Derived>& resConst, double normalizedTime)
    {
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 4);
        Eigen::MatrixBase<Derived>& res = const_cast<Eigen::MatrixBase<Derived>&>(resConst);
        res.setZero();
        res[Derivative] = baseCoefficients(Derivative, Derivative);
        double ti = normalizedTime;
        for (int j = Derivative + 1; j < 4; j++) {
            res[j] = baseCoefficients(Derivative, j) * ti;
            ti = ti * normalizedTime;
        }
    }
    
    // Compute the blending matrix for the B-spline of form:
    // [1, 0, 0, 0]
    // [1, 1, 0, 0]
    // [1, 1, 1, 0]
    // [1, 1, 1, 1]
    template <bool _Cumulative = false>
    static Eigen::Matrix4d computeBlendingMatrix()
    {
        Eigen::Matrix4d m;
        m.setZero();
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double sum = 0;
                for (int s = j; s < 4; ++s) {
                    sum += std::pow(-1.0, s - j) * binomialCoefficient(4, s - j) *
                    std::pow(4 - s - 1.0, 4 - 1.0 - i);
                }
                m(j, i) = binomialCoefficient(3, 3 - i) * sum;
            }
        }
        if (_Cumulative) {
            for (int i = 0; i < 4; i++) {
                for (int j = i + 1; j < 4; j++) {
                    m.row(i) += m.row(j);
                }
            }
        }
        uint64_t factorial = 1;
        for (int i = 2; i < 4; ++i) {
            factorial *= i;
        }
        return m / factorial;
    }

    // Compute the binomial coefficient 
    constexpr static inline uint64_t binomialCoefficient(uint64_t n, uint64_t k)
    {
        if (k > n) return 0;
        uint64_t r = 1;
        for (uint64_t d = 1; d <= k; ++d) {
            r *= n--;
            r /= d;
        }
        return r;
    }

    // Compute the base coefficients for the B-spline of form:
    // [1, 0, 0, 0]
    // [1, 1, 0, 0]
    // [1, 1, 1, 0]
    // [1, 1, 1, 1]
    static Eigen::Matrix4d computeBaseCoefficients()
    {
        Eigen::Matrix4d baseCoefficients;
        baseCoefficients.setZero();
        baseCoefficients.row(0).setOnes();
        int order = 3;
        for (int n = 1; n < 4; n++) {
            for (int i = 3 - order; i < 4; i++) {
                baseCoefficients(n, i) = (order - 3 + i) * baseCoefficients(n - 1, i);
            }
            order--;
        }
        return baseCoefficients;
    }
};

const Eigen::Matrix4d SplineState::baseCoefficients = SplineState::computeBaseCoefficients();
const Eigen::Matrix4d SplineState::blendingMatrix = SplineState::computeBlendingMatrix();
const Eigen::Matrix4d SplineState::cumulativeBlendingMatrix = SplineState::computeBlendingMatrix<true>();
