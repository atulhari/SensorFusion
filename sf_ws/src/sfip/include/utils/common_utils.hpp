#pragma once

#include "geometry_msgs/PoseStamped.h"
#include <deque>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <map>
#include <ros/ros.h>
#include <unordered_map>
#include <vector>

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

// struct TOAData {
//     const int64_t time_ns;
//     const int anchor_id;
//     const double data;
//     TOAData(const int64_t s, const int i, const double r) : time_ns(s),
//     anchor_id(i), data(r) {};
// };

// struct TDOAData {
//   const int64_t time_ns;
//   const int idA;
//   const int idB;
//   const double data;
//   TDOAData(const int64_t s, const int idxA, const int idxB, const double r) :
//   time_ns(s), idA(idxA), idB(idxB), data(r) {};
// };

class CalibParam {
public:
  // Eigen::Vector3d offset;  Not sure why this is needed
  Eigen::Quaterniond q_platform_imu;
  Eigen::Vector3d t_platform_imu;
  Eigen::Vector3d gravity;

  CalibParam(){};

  void setCalibParam(const CalibParam &p) {
    // offset = calib_param.offset;
    q_platform_imu = p.q_platform_imu;
    t_platform_imu = p.t_platform_imu;
    gravity = p.gravity;
  }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class CommonUtils {
public:
  template <typename T>
  static T readParam(ros::NodeHandle &nh, std::string name, const T& default_val) {
    T ans;
    if (!nh.getParam(name, ans)) {
      ROS_WARN_STREAM("Failed to load " << name << ", using default value.");
      return default_val;
    }
    return ans;
  }

  template <typename T>
  static T readParam(ros::NodeHandle &nh, std::string name) {
    T ans;
    if (!nh.getParam(name, ans)) {
      ROS_ERROR_STREAM("Failed to load " << name << ". Shutting down.");
      nh.shutdown();
    }
    return ans;
  }

  static geometry_msgs::PoseStamped pose2msg(const int64_t timestampNanoseconds,
                                             const Eigen::Vector3d &position,
                                             const Eigen::Quaterniond &orientation) {
    geometry_msgs::PoseStamped msg;
    msg.header.stamp.fromNSec(timestampNanoseconds);
    msg.header.frame_id = "map";
    msg.pose.position.x = position.x();
    msg.pose.position.y = position.y();
    msg.pose.position.z = position.z();
    msg.pose.orientation.w = orientation.w();
    msg.pose.orientation.x = orientation.x();
    msg.pose.orientation.y = orientation.y();
    msg.pose.orientation.z = orientation.z();
    return msg;
  }
};
