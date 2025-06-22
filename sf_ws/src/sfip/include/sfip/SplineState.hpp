#pragma once

#include "sfip/Types.hpp"
#include <array>
#include <deque>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

namespace sfip {

/**
 * @brief Jacobian structure for optimization
 */
template <class MatT>
struct JacobianStruct {
    size_t startIndex;
    std::vector<MatT, Eigen::aligned_allocator<MatT>> dValueDknot;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Specialized Jacobian types
using Jacobian = JacobianStruct<double>;
using Jacobian43 = JacobianStruct<Eigen::Matrix<double, 4, 3>>;
using Jacobian33 = JacobianStruct<Eigen::Matrix3d>;
using Jacobian13 = JacobianStruct<Eigen::Vector3d>;
using Jacobian16 = JacobianStruct<Eigen::Matrix<double, 6, 1>>;
using Jacobian36 = JacobianStruct<Eigen::Matrix<double, 3, 6>>;
using Jacobian66 = JacobianStruct<Eigen::Matrix<double, 6, 6>>;

/**
 * @brief Quaternion operations for spline interpolation
 */
class Quater {
public:
    /**
     * @brief Compute quaternion logarithm
     */
    static void log(const Eigen::Quaterniond& q, Eigen::Vector3d& result) {
        // Assumes q is a unit quaternion
        double norm = std::sqrt(q.x()*q.x() + q.y()*q.y() + q.z()*q.z());
        if (norm < 1e-10) {
            result.setZero();
            return;
        }
        
        double w = q.w();
        if (w < 0) {
            w = -w;
            norm = -norm;
        }
        
        double scale;
        if (std::abs(w) < 0.9999) {
            double angle = std::acos(w);
            scale = angle / norm;
        } else {
            // Small angle approximation for numerical stability
            scale = 2.0 / (1.0 + w);
        }
        
        result.x() = scale * q.x();
        result.y() = scale * q.y();
        result.z() = scale * q.z();
    }
    
    /**
     * @brief Compute quaternion exponential
     */
    static void exp(const Eigen::Vector3d& v, Eigen::Quaterniond& result) {
        double norm = v.norm();
        if (norm < 1e-10) {
            result.w() = 1.0;
            result.x() = result.y() = result.z() = 0.0;
            return;
        }
        
        double scale = std::sin(norm) / norm;
        result.w() = std::cos(norm);
        result.x() = scale * v.x();
        result.y() = scale * v.y();
        result.z() = scale * v.z();
    }
    
    /**
     * @brief Compute jacobian of log function
     */
    static void dlog(const Eigen::Quaterniond& q, Eigen::Vector3d& v, Eigen::Matrix<double, 3, 4>& J) {
        log(q, v);
        
        const double qw = q.w();
        const double qx = q.x();
        const double qy = q.y();
        const double qz = q.z();
        
        const double vx = v.x();
        const double vy = v.y();
        const double vz = v.z();
        
        // Compute the Jacobian of the log map
        double theta = v.norm();
        if (theta < 1e-6) {
            // Small angle approximation
            J << 0, 1, 0, 0,
                 0, 0, 1, 0,
                 0, 0, 0, 1;
        } else {
            double scale = -2.0 / std::sin(theta);
            double k1 = scale * std::cos(theta) * std::tan(theta/2);
            double k2 = scale;
            
            // Fill the Jacobian matrix
            J(0, 0) = k1 * vx;
            J(0, 1) = k2 * (qw*qw + qx*qx - qy*qy - qz*qz);
            J(0, 2) = k2 * 2 * (qx*qy - qw*qz);
            J(0, 3) = k2 * 2 * (qw*qy + qx*qz);
            
            J(1, 0) = k1 * vy;
            J(1, 1) = k2 * 2 * (qw*qz + qx*qy);
            J(1, 2) = k2 * (qw*qw - qx*qx + qy*qy - qz*qz);
            J(1, 3) = k2 * 2 * (-qw*qx + qy*qz);
            
            J(2, 0) = k1 * vz;
            J(2, 1) = k2 * 2 * (-qw*qy + qx*qz);
            J(2, 2) = k2 * 2 * (qw*qx + qy*qz);
            J(2, 3) = k2 * (qw*qw - qx*qx - qy*qy + qz*qz);
        }
    }
    
    /**
     * @brief Compute jacobian of exp function
     */
    static void dexp(const Eigen::Vector3d& v, Eigen::Quaterniond& q, Eigen::Matrix<double, 4, 3>& J) {
        exp(v, q);
        
        const double vx = v.x();
        const double vy = v.y();
        const double vz = v.z();
        
        double theta = v.norm();
        if (theta < 1e-6) {
            // Small angle approximation
            J << 0, 0, 0,
                 1, 0, 0,
                 0, 1, 0,
                 0, 0, 1;
        } else {
            double scale1 = -std::sin(theta) / theta;
            double scale2 = (std::cos(theta) - std::sin(theta)/theta) / (theta*theta);
            
            // Fill the Jacobian matrix
            J(0, 0) = scale1 * vx;
            J(0, 1) = scale1 * vy;
            J(0, 2) = scale1 * vz;
            
            J(1, 0) = scale2 * vx * vx + std::cos(theta)/theta;
            J(1, 1) = scale2 * vx * vy;
            J(1, 2) = scale2 * vx * vz;
            
            J(2, 0) = scale2 * vy * vx;
            J(2, 1) = scale2 * vy * vy + std::cos(theta)/theta;
            J(2, 2) = scale2 * vy * vz;
            
            J(3, 0) = scale2 * vz * vx;
            J(3, 1) = scale2 * vz * vy;
            J(3, 2) = scale2 * vz * vz + std::cos(theta)/theta;
        }
    }
    
    /**
     * @brief Compute left quaternion multiplication matrix
     */
    static void Qleft(const Eigen::Quaterniond& q, Eigen::Matrix4d& QL) {
        QL << q.w(), -q.x(), -q.y(), -q.z(),
              q.x(),  q.w(), -q.z(),  q.y(),
              q.y(),  q.z(),  q.w(), -q.x(),
              q.z(), -q.y(),  q.x(),  q.w();
    }
    
    /**
     * @brief Compute right quaternion multiplication matrix
     */
    static void Qright(const Eigen::Quaterniond& q, Eigen::Matrix4d& QR) {
        QR << q.w(), -q.x(), -q.y(), -q.z(),
              q.x(),  q.w(),  q.z(), -q.y(),
              q.y(), -q.z(),  q.w(),  q.x(),
              q.z(),  q.y(), -q.x(),  q.w();
    }
    
    /**
     * @brief Compute the jacobian of rotation operation
     */
    static void drot(const Eigen::Vector3d& v, const Eigen::Quaterniond& q, Eigen::Matrix<double, 3, 4>& J) {
        const double qw = q.w();
        const double qx = q.x();
        const double qy = q.y();
        const double qz = q.z();
        
        const double vx = v.x();
        const double vy = v.y();
        const double vz = v.z();
        
        // Fill the Jacobian matrix
        J(0, 0) =  2 * (qw*vx + qy*vz - qz*vy);
        J(0, 1) =  2 * (qx*vx + qy*vy + qz*vz);
        J(0, 2) =  2 * (qy*vx - qx*vy + qw*vz);
        J(0, 3) =  2 * (qz*vx - qw*vy - qx*vz);
        
        J(1, 0) =  2 * (qw*vy + qz*vx - qx*vz);
        J(1, 1) =  2 * (qx*vy - qw*vz - qz*vx);
        J(1, 2) =  2 * (qy*vy + qx*vx + qw*vz);
        J(1, 3) =  2 * (qz*vy - qx*vz + qw*vx);
        
        J(2, 0) =  2 * (qw*vz + qx*vy - qy*vx);
        J(2, 1) =  2 * (qx*vz - qw*vy - qy*vx);
        J(2, 2) =  2 * (qy*vz - qw*vx - qx*vy);
        J(2, 3) =  2 * (qz*vz + qx*vx + qy*vy);
    }
    
    /**
     * @brief Create a delta quaternion from angular velocity
     */
    static Eigen::Quaterniond deltaQ(const Eigen::Vector3d& w) {
        Eigen::Quaterniond dq;
        double angle = w.norm();
        if (angle < 1e-10) {
            dq.w() = 1.0;
            dq.x() = dq.y() = dq.z() = 0.0;
        } else {
            Eigen::Vector3d axis = w / angle;
            dq = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
        }
        return dq;
    }
};

/**
 * @brief Spherical coordinate operations for gravity optimization
 */
class Sphere {
public:
    /**
     * @brief Get tangent basis for spherical coordinates
     */
    static Eigen::Matrix<double, 3, 2> TangentBasis(const Eigen::Vector3d& g) {
        Eigen::Vector3d g_normalized = g.normalized();
        Eigen::Vector3d b1, b2;
        
        // Choose arbitrary orthogonal basis
        if (std::abs(g_normalized.z()) < 0.866) { // 30 degrees from up
            b1 = Eigen::Vector3d::UnitZ().cross(g_normalized).normalized();
        } else {
            b1 = Eigen::Vector3d::UnitX().cross(g_normalized).normalized();
        }
        b2 = g_normalized.cross(b1);
        
        Eigen::Matrix<double, 3, 2> basis;
        basis.col(0) = b1;
        basis.col(1) = b2;
        return basis;
    }
};

/**
 * @brief Represents the state of the B-spline
 * 
 * This class manages the control points and provides interpolation functions
 * for position, orientation, and biases.
 */
class SplineState {
public:
    static constexpr double SECONDS_TO_NANOSECONDS = 1e9;
    
    /**
     * @brief Default constructor
     */
    SplineState()
        : isFirst(true),
          invDt(0),
          numKnots(0),
          startIndex(0),
          startTimeNanoseconds(0)
    {
        // Initialize powInvDt array
        powInvDt[0] = 1.0;
        powInvDt[1] = 0.0;
        powInvDt[2] = 0.0;
        powInvDt[3] = 0.0;
        
        // Initialize idle states
        Eigen::Quaterniond q0 = Eigen::Quaterniond::Identity();
        Eigen::Vector3d t0 = Eigen::Vector3d::Zero();
        Eigen::Matrix<double, 6, 1> b0 = Eigen::Matrix<double, 6, 1>::Zero();
        qIdle = {q0, q0, q0};
        tIdle = {t0, t0, t0};
        bIdle = {b0, b0, b0};
    }

    /**
     * @brief Initialize the spline
     * @param knotIntervalNanoseconds Time between knots in nanoseconds
     * @param numKnots Initial number of knots
     * @param startTimeNanoseconds Start time in nanoseconds
     * @param startIndex Initial index
     */
    void init(int64_t knotIntervalNanoseconds_, int numKnots_, int64_t startTimeNanoseconds_, int startIndex_ = 0) {
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
        
        // Clear any existing knots
        qKnots.clear();
        tKnots.clear();
        bKnots.clear();
    }

    /**
     * @brief Set a single state knot
     * @param knotIndex Index of the knot to set
     * @param quaternion Orientation
     * @param position Position
     * @param bias Bias vector
     */
    void setSingleStateKnot(int knotIndex, Eigen::Quaterniond quaternion, Eigen::Vector3d position, Eigen::Matrix<double, 6, 1> bias) {
        // Ensure knots containers have enough space
        while (tKnots.size() <= static_cast<size_t>(knotIndex)) {
            tKnots.push_back(Eigen::Vector3d::Zero());
            qKnots.push_back(Eigen::Quaterniond::Identity());
            bKnots.push_back(Eigen::Matrix<double, 6, 1>::Zero());
            numKnots = tKnots.size();
        }
        
        tKnots[knotIndex] = position;
        qKnots[knotIndex] = quaternion;
        bKnots[knotIndex] = bias;
    }

    /**
     * @brief Update knots from another SplineState object
     * @param other Source SplineState
     */
    void updateKnots(SplineState* other) {
        if (!other) return;
        
        size_t numOtherKnots = other->getNumKnots();
        for (size_t i = 0; i < numOtherKnots; i++) {
            if (i + other->startIndex < numOtherKnots) {
                setSingleStateKnot(i + other->startIndex, 
                                  other->qKnots[i],
                                  other->tKnots[i],
                                  other->bKnots[i]);
            } else {
                addSingleStateKnot(other->qKnots[i],
                                  other->tKnots[i],
                                  other->bKnots[i]);
            }
        }
        qIdle = other->qIdle;
        tIdle = other->tIdle;
        bIdle = other->bIdle;
    }

    /**
     * @brief Get idle position
     * @param idx Idle index
     * @return Position of the idle knot
     */
    Eigen::Vector3d getIdlePosition(int idx) {
        if (idx >= 0 && idx < 3) {
            return tIdle[idx];
        }
        return Eigen::Vector3d::Zero();
    }

    /**
     * @brief Set idle state
     * @param idx Idle index
     * @param position Position
     * @param quaternion Orientation
     * @param bias Bias
     */
    void setIdles(int idx, Eigen::Vector3d position, Eigen::Quaterniond quaternion, Eigen::Matrix<double, 6, 1> bias) {
        if (idx >= 0 && idx < 3) {
            tIdle[idx] = position;
            qIdle[idx] = quaternion;
            bIdle[idx] = bias;
        }
    }

    /**
     * @brief Update bias of idle state
     */
    void updateBiasIdleFirstWindow() {
        if (!isFirst) {
            return;
        } else {
            if (bKnots.size() > 0) {
                bIdle = {bKnots[0], bKnots[0], bKnots[0]};
            }
        }
    }

    /**
     * @brief Add a new knot to the end of the spline
     * @param quaternion Orientation
     * @param position Position
     * @param bias Bias
     */
    void addSingleStateKnot(Eigen::Quaterniond quaternion, Eigen::Vector3d position, Eigen::Matrix<double, 6, 1> bias) {
        // Check for quaternion consistency with previous knot
        if (numKnots > 1) {
            Eigen::Quaterniond q0 = qKnots[numKnots - 1];
            Eigen::Quaterniond q1 = quaternion;
            double dotProduct = q0.dot(q1);

            // If the new knot is on the opposite side of the unit quaternion, we change the sign
            if (dotProduct < 0) {
               quaternion = Eigen::Quaterniond(-quaternion.w(), -quaternion.x(), -quaternion.y(), -quaternion.z());
            }
        }

        // Normalize quaternion if needed
        if (abs(quaternion.norm() - 1) > 1e-5) {
            quaternion.normalize();
        }

        // Add the new knot
        qKnots.push_back(quaternion);
        tKnots.push_back(position);
        bKnots.push_back(bias);
        numKnots++;
    }

    /**
     * @brief Check quaternion control points for consistency
     */
    void checkQuaternionControlPoints() {
        // Ensure quaternions are on the same hemisphere
        if (numKnots > 1) {
            for (size_t i = 1; i < numKnots; i++) {
                Eigen::Quaterniond q1 = qKnots[i];
                // Keep same hemisphere as previous knot
                if (qKnots[i - 1].dot(q1) < 0) {
                    qKnots[i] = Eigen::Quaterniond(-q1.w(), -q1.x(), -q1.y(), -q1.z());
                }
            }
        }
        
        // Additionally, enforce w>0 on all knots
        for (size_t i = 0; i < numKnots; ++i) {
            if (qKnots[i].w() < 0) {
                qKnots[i].coeffs() *= -1;
            }
        }
    }

    /**
     * @brief Remove the oldest knot
     */
    void removeSingleOldState() {
        if (numKnots <= 0) return;
        
        // Update idle states
        qIdle = {qIdle[1], qIdle[2], qKnots.front()};
        tIdle = {tIdle[1], tIdle[2], tKnots.front()};
        bIdle = {bIdle[1], bIdle[2], bKnots.front()};

        // Remove oldest knot
        qKnots.pop_front();
        tKnots.pop_front();
        bKnots.pop_front();

        numKnots--;
        startIndex++;
        startTimeNanoseconds += knotIntervalNanoseconds;
        isFirst = false;
    }

    /**
     * @brief Get all knots from the active window
     * @param knotsTrans Output deque for positions
     * @param knotsRot Output deque for orientations
     * @param knotsBias Output deque for biases
     */
    void getAllStateKnots(std::deque<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& knotsTrans,
                         std::deque<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>>& knotsRot,
                         std::deque<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>>& knotsBias) {
        knotsTrans = tKnots;
        knotsRot = qKnots;
        knotsBias = bKnots;
    }

    /**
     * @brief Set all knots
     * @param knotsTrans Input deque for positions
     * @param knotsRot Input deque for orientations
     * @param knotsBias Input deque for biases
     */
    void setAllKnots(std::deque<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& knotsTrans,
                    std::deque<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>>& knotsRot,
                    std::deque<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>>& knotsBias) {
        tKnots = knotsTrans;
        qKnots = knotsRot;
        bKnots = knotsBias;
        numKnots = tKnots.size();

        // Update bias idle state
        updateBiasIdleFirstWindow();
    }

    /**
     * @brief Get time of a knot
     * @param knotIndex Index of the knot
     * @return Time in nanoseconds
     */
    int64_t getKnotTimeNanoseconds(size_t knotIndex) const {
        return startTimeNanoseconds + knotIndex * knotIntervalNanoseconds;
    }

    /**
     * @brief Get orientation of a knot
     * @param knotIndex Index of the knot
     * @return Orientation quaternion
     */
    Eigen::Quaterniond getKnotOrientation(size_t knotIndex) {
        if (knotIndex < numKnots) {
            return qKnots[knotIndex];
        }
        return Eigen::Quaterniond::Identity();
    }

    /**
     * @brief Get position of a knot
     * @param knotIndex Index of the knot
     * @return Position vector
     */
    Eigen::Vector3d getKnotPosition(size_t knotIndex) {
        if (knotIndex < numKnots) {
            return tKnots[knotIndex];
        }
        return Eigen::Vector3d::Zero();
    }

    /**
     * @brief Get bias of a knot
     * @param knotIndex Index of the knot
     * @return Bias vector
     */
    Eigen::Matrix<double, 6, 1> getKnotBias(size_t knotIndex) {
        if (knotIndex < numKnots) {
            return bKnots[knotIndex];
        }
        return Eigen::Matrix<double, 6, 1>::Zero();
    }

    /**
     * @brief Extrapolate position of next knot
     * @param knotIndex Index to extrapolate from
     * @return Predicted position
     */
    Eigen::Vector3d extrapolateKnotPosition(size_t knotIndex) {
        if (numKnots < 2 || knotIndex >= numKnots) {
            return Eigen::Vector3d::Zero();
        }
        
        Eigen::Quaterniond lastOrientation = qKnots[numKnots - knotIndex - 1];
        Eigen::Quaterniond currentOrientation = qKnots[numKnots - knotIndex];
        Eigen::Vector3d lastPosition = tKnots[numKnots - knotIndex - 1];
        Eigen::Vector3d currentPosition = tKnots[numKnots - knotIndex];
        
        // Calculate relative translation in last frame
        Eigen::Vector3d relativeTranslation = lastOrientation.inverse() * (currentPosition - lastPosition);
        
        // Apply to current frame
        return currentPosition + currentOrientation * relativeTranslation;
    }

    /**
     * @brief Extrapolate orientation of next knot
     * @param knotIndex Index to extrapolate from
     * @return Predicted orientation
     */
    Eigen::Quaterniond extrapolateKnotOrientation(size_t knotIndex) {
        if (numKnots < 2 || knotIndex >= numKnots) {
            return Eigen::Quaterniond::Identity();
        }
        
        Eigen::Quaterniond lastOrientation = qKnots[numKnots - knotIndex - 1];
        Eigen::Quaterniond currentOrientation = qKnots[numKnots - knotIndex];
        
        // Calculate relative rotation
        Eigen::Quaterniond relativeRotation = currentOrientation * lastOrientation.inverse();
        
        // Apply to current frame
        return relativeRotation * currentOrientation;
    }

    /**
     * @brief Apply pose increment to a knot
     * @param knotIndex Index of the knot
     * @param poseIncrement Pose increment (position and orientation)
     */
    void applyPoseIncrement(int knotIndex, const Eigen::Matrix<double, 6, 1> &poseIncrement) {
        if (knotIndex >= 0 && knotIndex < static_cast<int>(numKnots)) {
            // Apply position increment
            tKnots[knotIndex] += poseIncrement.head<3>();
            
            // Apply rotation increment using exp map
            Eigen::Quaterniond qIncrement;
            Quater::exp(poseIncrement.tail<3>(), qIncrement);
            qKnots[knotIndex] = qKnots[knotIndex] * qIncrement;
            qKnots[knotIndex].normalize();
        }
    }

    /**
     * @brief Apply bias increment to a knot
     * @param knotIndex Index of the knot
     * @param biasIncrement Bias increment
     */
    void applyBiasIncrement(int knotIndex, const Eigen::Matrix<double, 6, 1>& biasIncrement) {
        if (knotIndex >= 0 && knotIndex < static_cast<int>(numKnots)) {
            bKnots[knotIndex] += biasIncrement;
        }
    }

    /**
     * @brief Get maximum time of the spline
     * @return Maximum time in nanoseconds
     */
    int64_t maxTimeNanoseconds() {
        if (numKnots <= 1) {
            return startTimeNanoseconds;
        }
        return startTimeNanoseconds + (numKnots - 1) * knotIntervalNanoseconds - 1;
    }

    /**
     * @brief Get minimum time of the spline
     * @return Minimum time in nanoseconds
     */
    int64_t minTimeNanoseconds() {
        return startTimeNanoseconds + knotIntervalNanoseconds * (!isFirst ? -1 : 0);
    }

    /**
     * @brief Get maximum time of the next spline knots
     * @return Maximum time in nanoseconds
     */
    int64_t nextMaxTimeNanoseconds() {
        return startTimeNanoseconds + numKnots * knotIntervalNanoseconds - 1;
    }
    
    /**
     * @brief Get number of knots
     * @return Number of knots
     */
    size_t getNumKnots() {
        return numKnots;
    }
    
    /**
     * @brief Interpolate position at a given time
     * @param timeNanoseconds Time in nanoseconds
     * @param jacobian Optional Jacobian output
     * @return Interpolated position
     */
    template <int Derivative = 0>
    Eigen::Vector3d interpolatePosition(int64_t timeNanoseconds, Jacobian* jacobian = nullptr) const {
        return interpolateEuclidean<Eigen::Vector3d, Derivative>(timeNanoseconds, tIdle, tKnots, jacobian);
    }
    
    /**
     * @brief Interpolate bias at a given time
     * @param timeNanoseconds Time in nanoseconds
     * @param jacobian Optional Jacobian output
     * @return Interpolated bias
     */
    Eigen::Matrix<double, 6, 1> interpolateBias(int64_t timeNanoseconds, Jacobian* jacobian = nullptr) const {
        return interpolateEuclidean<Eigen::Matrix<double, 6, 1>>(timeNanoseconds, bIdle, bKnots, jacobian);
    }
    
    /**
     * @brief Interpolate quaternion at a given time
     * @param timeNanoseconds Time in nanoseconds
     * @param quaternionOut Optional output for quaternion
     * @param angularVelocityOut Optional output for angular velocity
     * @param quaternionJacobian Optional output for quaternion Jacobian
     * @param angularVelocityJacobian Optional output for angular velocity Jacobian
     */
    void interpolateQuaternion(int64_t timeNanoseconds, Eigen::Quaterniond* quaternionOut = nullptr,
                              Eigen::Vector3d* angularVelocityOut = nullptr, 
                              Jacobian43* quaternionJacobian = nullptr, 
                              Jacobian33* angularVelocityJacobian = nullptr) const {
        double normalizedTime;
        int64_t startingIndex;
        int indexRight;
        std::array<Eigen::Quaterniond, 4> controlPoints;

        // Prepare the interpolation
        prepareInterpolation(timeNanoseconds, qIdle, qKnots, startingIndex, normalizedTime, controlPoints, indexRight);

        Eigen::Vector4d basisVector;
        Eigen::Vector4d coefficients;
        baseCoeffsWithTime<0>(basisVector, normalizedTime);
        coefficients = cumulativeBlendingMatrix * basisVector;

        // Compute the delta quaternions
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
            // Complex path with Jacobian calculation
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
            
            for (int i = 2; i >= 0; i--) {
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
                
                if (quaternionOut) *quaternionOut = qInterpolated[3];
                
                Eigen::Matrix4d Q_l_all[3];
                Quater::Qleft(qInterpolated[0], Q_l_all[0]);
                Quater::Qleft(qInterpolated[1], Q_l_all[1]);
                Quater::Qleft(qInterpolated[2], Q_l_all[2]);
                
                for (int i = 2; i >= 0; i--) {
                    Eigen::Matrix4d Q_r_all;
                    Quater::Qright(q_r_all[i+1], Q_r_all);
                    d_X_d_dj[i].noalias() = coefficients[i + 1] * Q_r_all * Q_l_all[i] * dexp_dt[i] * dlog_dq[i];
                }
                
                quaternionJacobian->dValueDknot.resize(size_J);
                for (int i = 0; i < size_J; i++) {
                    quaternionJacobian->dValueDknot[i].setZero();
                }
                
                for (int i = 0; i < size_J - 1; i++) {
                    quaternionJacobian->dValueDknot[i].noalias() -= d_X_d_dj[4 - size_J + i] * Q_r[i].rightCols(3);
                    quaternionJacobian->dValueDknot[i + 1].noalias() += d_X_d_dj[4 - size_J + i] * Q_l[i].rightCols(3);
                }
                
                quaternionJacobian->startIndex = startingIndex;
                
                if (size_J == 4) {
                    Eigen::Matrix4d Q_r_all;
                    Eigen::Matrix4d Q0_left;
                    Quater::Qright(q_r_all[0], Q_r_all);
                    Quater::Qleft(controlPoints[0], Q0_left);
                    quaternionJacobian->dValueDknot[0].noalias() += Q_r_all * Q0_left.rightCols(3);
                } else {
                    Eigen::Matrix4d Q_left;
                    Quater::Qleft(qDelta[3 - size_J], Q_left);
                    quaternionJacobian->dValueDknot[0].noalias() += d_X_d_dj[3 - size_J] * Q_left.rightCols(3);
                }
            }
            
            if (angularVelocityJacobian) {
                baseCoeffsWithTime<1>(basisVector, normalizedTime);
                dcoeff = invDt * cumulativeBlendingMatrix * basisVector;
                
                angularVelocityInterpolated[0].setZero();
                angularVelocityInterpolated[1] = 2 * dcoeff[1] * tDelta[0];
                angularVelocityInterpolated[2] = qDeltaScale[1].inverse() * angularVelocityInterpolated[1] + 2 * dcoeff[2] * tDelta[1];
                angularVelocityInterpolated[3] = qDeltaScale[2].inverse() * angularVelocityInterpolated[2] + 2 * dcoeff[3] * tDelta[2];
                
                if (angularVelocityOut) *angularVelocityOut = angularVelocityInterpolated[3];
                
                Eigen::Matrix<double, 3, 4> drot_dq[3];
                Quater::drot(angularVelocityInterpolated[0], qDeltaScale[0], drot_dq[0]);
                Quater::drot(angularVelocityInterpolated[1], qDeltaScale[1], drot_dq[1]);
                Quater::drot(angularVelocityInterpolated[2], qDeltaScale[2], drot_dq[2]);
                
                for (int i = 2; i >= 0; i--) {
                    Eigen::Matrix3d d_vel_d_dj = coefficients[i + 1] * drot_dq[i] * dexp_dt[i];
                    d_vel_d_dj.noalias() += 2 * dcoeff[i + 1] * Eigen::Matrix3d::Identity();
                    d_r_d_dj[i].noalias() = q_r_all[i+1].inverse().toRotationMatrix() * d_vel_d_dj * dlog_dq[i];
                }
                
                angularVelocityJacobian->dValueDknot.resize(size_J);
                for (int i = 0; i < size_J; i++) {
                    angularVelocityJacobian->dValueDknot[i].setZero();
                }
                
                for (int i = 0; i < size_J - 1; i++) {
                    angularVelocityJacobian->dValueDknot[i].noalias() -= d_r_d_dj[4 - size_J + i] * Q_r[i].rightCols(3);
                    angularVelocityJacobian->dValueDknot[i + 1].noalias() += d_r_d_dj[4 - size_J + i] * Q_l[i].rightCols(3);
                }
                
                angularVelocityJacobian->startIndex = startingIndex;
                
                if (size_J != 4) {
                    Eigen::Matrix4d Q_left;
                    Quater::Qleft(qDelta[4 - size_J - 1], Q_left);
                    angularVelocityJacobian->dValueDknot[0].noalias() += d_r_d_dj[3 - size_J] * Q_left.rightCols(3);
                }
            }
        } else {
            // Simple path without Jacobian calculation
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
    
    /**
     * @brief Convert spline to ROS message
     * @param splineMsg Output spline message
     */
    template <typename SplineMsg>
    void getSplineMsg(SplineMsg& splineMsg) {
        splineMsg.dt = knotIntervalNanoseconds;
        splineMsg.start_t = startTimeNanoseconds;
        splineMsg.start_idx = startIndex;
        
        // Clear existing knots if any
        splineMsg.knots.clear();
        splineMsg.idles.clear();
        
        // Add active knots
        for (size_t i = 0; i < numKnots; i++) {
            typename SplineMsg::_knots_type::value_type knotMsg;
            
            // Position
            knotMsg.position.x = tKnots[i].x();
            knotMsg.position.y = tKnots[i].y();
            knotMsg.position.z = tKnots[i].z();
            
            // Orientation
            knotMsg.orientation.w = qKnots[i].w();
            knotMsg.orientation.x = qKnots[i].x();
            knotMsg.orientation.y = qKnots[i].y();
            knotMsg.orientation.z = qKnots[i].z();
            
            // Bias
            Eigen::Matrix<double, 6, 1> bias = bKnots[i];
            knotMsg.bias_acc.x = bias[0];
            knotMsg.bias_acc.y = bias[1];
            knotMsg.bias_acc.z = bias[2];
            knotMsg.bias_gyro.x = bias[3];
            knotMsg.bias_gyro.y = bias[4];
            knotMsg.bias_gyro.z = bias[5];
            
            splineMsg.knots.push_back(knotMsg);
        }
        
        // Add idle knots
        for (int i = 0; i < 3; i++) {
            typename SplineMsg::_idles_type::value_type idleMsg;
            
            // Position
            idleMsg.position.x = tIdle[i].x();
            idleMsg.position.y = tIdle[i].y();
            idleMsg.position.z = tIdle[i].z();
            
            // Orientation
            idleMsg.orientation.w = qIdle[i].w();
            idleMsg.orientation.x = qIdle[i].x();
            idleMsg.orientation.y = qIdle[i].y();
            idleMsg.orientation.z = qIdle[i].z();
            
            // Bias
            Eigen::Matrix<double, 6, 1> bias = bIdle[i];
            idleMsg.bias_acc.x = bias[0];
            idleMsg.bias_acc.y = bias[1];
            idleMsg.bias_acc.z = bias[2];
            idleMsg.bias_gyro.x = bias[3];
            idleMsg.bias_gyro.y = bias[4];
            idleMsg.bias_gyro.z = bias[5];
            
            splineMsg.idles.push_back(idleMsg);
        }
    }
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
private:
    bool isFirst;
    
    static const Eigen::Matrix4d blendingMatrix;
    static const Eigen::Matrix4d baseCoefficients;
    static const Eigen::Matrix4d cumulativeBlendingMatrix;

    int64_t knotIntervalNanoseconds;
    double invDt;
    std::array<double, 4> powInvDt;
    size_t numKnots;
    int64_t startIndex;
    int64_t startTimeNanoseconds;

    std::array<Eigen::Quaterniond, 3> qIdle;
    std::array<Eigen::Vector3d, 3> tIdle;
    std::array<Eigen::Matrix<double, 6, 1>, 3> bIdle;
    std::deque<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> qKnots;
    std::deque<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> tKnots;
    std::deque<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bKnots;

    /**
     * @brief Interpolate Euclidean value at a given time
     * @param timeNanoseconds Time in nanoseconds
     * @param knotsIdle Idle knots
     * @param knots Active knots
     * @param jacobian Optional Jacobian output
     * @return Interpolated value
     */
    template <typename _KnotT, int Derivative = 0>
    _KnotT interpolateEuclidean(int64_t timeNanoseconds, 
                               const std::array<_KnotT, 3>& knotsIdle,
                               const std::deque<_KnotT, Eigen::aligned_allocator<_KnotT>>& knots, 
                               Jacobian* jacobian = nullptr) const {
        double normalizedTime;
        int64_t startingIndex;
        int indexRight;
        std::array<_KnotT, 4> controlPoints;
        
        prepareInterpolation(timeNanoseconds, knotsIdle, knots, startingIndex, normalizedTime, controlPoints, indexRight);
        
        Eigen::Vector4d basisVector;
        Eigen::Vector4d coefficients;
        baseCoeffsWithTime<Derivative>(basisVector, normalizedTime);
        coefficients = powInvDt[Derivative] * (blendingMatrix * basisVector);
        
        _KnotT result = coefficients[0] * controlPoints[0] + 
                        coefficients[1] * controlPoints[1] + 
                        coefficients[2] * controlPoints[2] + 
                        coefficients[3] * controlPoints[3];
        
        if (jacobian) {
            int size_J = std::min(indexRight + 1, 4);
            jacobian->dValueDknot.resize(size_J);
            
            for (int i = 0; i < size_J; i++) {
                jacobian->dValueDknot[i] = coefficients[4 - size_J + i];
            }
            
            jacobian->startIndex = startingIndex;
        }
        
        return result;
    }

    /**
     * @brief Prepare interpolation by selecting control points
     * @param timeNanoseconds Time in nanoseconds
     * @param knotsIdle Idle knots
     * @param knots Active knots
     * @param startingIndex Output starting index
     * @param normalizedTime Output normalized time
     * @param controlPoints Output control points
     * @param indexRight Output right index
     */
    template<typename _KnotT>
    void prepareInterpolation(int64_t timeNanoseconds, 
                             const std::array<_KnotT, 3>& knotsIdle,
                             const std::deque<_KnotT, Eigen::aligned_allocator<_KnotT>>& knots, 
                             int64_t& startingIndex, 
                             double& normalizedTime,
                             std::array<_KnotT, 4>& controlPoints, 
                             int& indexRight) const {
        int64_t timestampNanosecondsRelative = timeNanoseconds - startTimeNanoseconds;
        int indexLeft = floor(double(timestampNanosecondsRelative) / double(knotIntervalNanoseconds));
        indexRight = indexLeft + 1;
        startingIndex = std::max(indexLeft - 2, 0LL);
        
        for (int i = 0; i < 2 - indexLeft; i++) {
            controlPoints[i] = knotsIdle[i + indexLeft + 1];
        }
        
        int indexWindow = std::max(0, 2 - indexLeft);
        for (int i = 0; i < std::min(indexLeft + 2, 4); i++) {
            if (startingIndex + i < knots.size()) {
                controlPoints[i + indexWindow] = knots[startingIndex + i];
            } else {
                controlPoints[i + indexWindow] = _KnotT(); // Default value
            }
        }
        
        normalizedTime = (timeNanoseconds - startTimeNanoseconds - indexLeft * knotIntervalNanoseconds) / 
                          double(knotIntervalNanoseconds);
    }

    /**
     * @brief Compute B-spline blending coefficients for a given time
     * @param resConst Output coefficients
     * @param normalizedTime Normalized time
     */
    template <int Derivative, class Derived>
    static void baseCoeffsWithTime(const Eigen::MatrixBase<Derived>& resConst, double normalizedTime) {
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
    
    /**
     * @brief Compute the blending matrix for the B-spline
     * @return Blending matrix
     */
    template <bool _Cumulative = false>
    static Eigen::Matrix4d computeBlendingMatrix() {
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

    /**
     * @brief Compute binomial coefficient (n choose k)
     * @param n Upper parameter
     * @param k Lower parameter
     * @return Binomial coefficient
     */
    static constexpr uint64_t binomialCoefficient(uint64_t n, uint64_t k) {
        if (k > n) return 0;
        uint64_t r = 1;
        for (uint64_t d = 1; d <= k; ++d) {
            r *= n--;
            r /= d;
        }
        return r;
    }

    /**
     * @brief Compute the base coefficients for the B-spline
     * @return Base coefficients
     */
    static Eigen::Matrix4d computeBaseCoefficients() {
        Eigen::Matrix4d baseCoeffs;
        baseCoeffs.setZero();
        baseCoeffs.row(0).setOnes();
        
        int order = 3;
        for (int n = 1; n < 4; n++) {
            for (int i = 3 - order; i < 4; i++) {
                baseCoeffs(n, i) = (order - 3 + i) * baseCoeffs(n - 1, i);
            }
            order--;
        }
        
        return baseCoeffs;
    }
};

// Static member initialization
const Eigen::Matrix4d SplineState::baseCoefficients = SplineState::computeBaseCoefficients();
const Eigen::Matrix4d SplineState::blendingMatrix = SplineState::computeBlendingMatrix();
const Eigen::Matrix4d SplineState::cumulativeBlendingMatrix = SplineState::computeBlendingMatrix<true>();
