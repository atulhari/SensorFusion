#pragma once

#include <deque>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <map>
#include <unordered_map>
#include <vector>

// Minimal ROS-like logging for standalone test
#define ROS_INFO_STREAM(x) std::cout << "[INFO] " << x << std::endl
#define ROS_WARN_STREAM(x) std::cout << "[WARN] " << x << std::endl
#define ROS_ERROR_STREAM(x) std::cout << "[ERROR] " << x << std::endl
#define ROS_DEBUG_STREAM_THROTTLE(rate, x) // disabled for standalone

namespace Eigen {

template <typename T>
using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;

template <typename T>
using aligned_deque = std::deque<T, Eigen::aligned_allocator<T>>;

template <typename K, typename V>
using aligned_map = std::map<K, V, std::less<K>,
                             Eigen::aligned_allocator<std::pair<K const, V>>>;

template <typename K, typename V>
using aligned_unordered_map =
    std::unordered_map<K, V, std::hash<K>, std::equal_to<K>,
                       Eigen::aligned_allocator<std::pair<K const, V>>>;

} // namespace Eigen

struct PoseData {
  int64_t timestampNanoseconds;
  Eigen::Quaterniond orientation;
  Eigen::Vector3d position;
  PoseData() {}
  PoseData(int64_t timestampNanoseconds, const Eigen::Quaterniond &orientation, const Eigen::Vector3d &position)
      : timestampNanoseconds(timestampNanoseconds), orientation(orientation), position(position) {}
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ImuData {
  int64_t timestampNanoseconds;
  Eigen::Vector3d gyro;
  Eigen::Vector3d accel;
  ImuData(int64_t timestampNanoseconds, const Eigen::Vector3d &gyro, const Eigen::Vector3d &accel)
      : timestampNanoseconds(timestampNanoseconds), gyro(gyro), accel(accel) {}
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class CalibParam {
public:
  Eigen::Quaterniond q_platform_imu;
  Eigen::Vector3d t_platform_imu;
  Eigen::Vector3d gravity;

  CalibParam() : q_platform_imu(Eigen::Quaterniond::Identity()), 
                 t_platform_imu(Eigen::Vector3d::Zero()),
                 gravity(0.0, 0.0, -9.80665) {};

  void setCalibParam(const CalibParam &p) {
    q_platform_imu = p.q_platform_imu;
    t_platform_imu = p.t_platform_imu;
    gravity = p.gravity;
  }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Parameters structure (copied from Linearizer.hpp but made standalone)
struct Parameters {
    bool if_opt_g;
    double w_pose_pos; //weight (std-inv) for position
    double w_pose_rot; //weight (std-inv) for rotation
    double w_acc; //weight (std-inv) for accel
    double w_gyro; //weight (std-inv) for gyro
    double w_bias_acc; //weight (std-inv) for accel bias
    double w_bias_gyro; //weight (std-inv) for gyro bias

    int control_point_fps;

    Eigen::Vector3d accel_var_inv, gyro_var_inv;
    Eigen::Vector3d bias_accel_var_inv, bias_gyro_var_inv;
    Eigen::Vector3d pos_var_inv;
    Eigen::Vector3d gravity; // Gravity vector in world frame

    Parameters() : if_opt_g(false), w_pose_pos(1.0), w_pose_rot(1.0), w_acc(1.0), w_gyro(1.0), 
                   w_bias_acc(1.0), w_bias_gyro(1.0), control_point_fps(60),
                   gravity(0.0, 0.0, -9.80665) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}; 