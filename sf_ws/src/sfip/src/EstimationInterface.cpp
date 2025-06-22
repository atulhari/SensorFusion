#include "SplineState.hpp"
// #include "PoseVisualization.hpp"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "geometry_msgs/TransformStamped.h"
#include "nav_msgs/Path.h"
#include "sensor_msgs/Imu.h"
#include "sensor_msgs/PointCloud.h"
#include "sfip/Estimate.h"
#include "sfip/Spline.h"
#include "std_msgs/Int64.h"
#include <fstream>
#include <ros/ros.h>
#include "geometry_msgs/PoseStamped.h"
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

// Make sure common_utils is included for readParam
#include "utils/common_utils.hpp" 

class EstimationInterface {

public:
  EstimationInterface(ros::NodeHandle &nh) {
    // TF buffer & listener must live longer than any lookup calls
    tf_listener_ptr_ = std::make_shared<tf2_ros::TransformListener>(tf_buffer_);
    
    readParamsInterface(nh);
    
    // Get the static transform from platform to IMU frame at startup
    if (!getStaticTransform()) {
      ROS_ERROR("[EI] Failed to get static transform from platform to IMU. This is mandatory.");
      ros::shutdown();
      return;
    }
    
    ROS_INFO_STREAM("[EI] Using imu_frame='" << imu_frame_ << "'  platform_frame='" << platform_frame_id_ << "'  output_frame='" << output_frame_ << "'.");
    ROS_INFO_STREAM("[EI] Static transform platform->IMU: t=[" << static_T_platform_imu_.x() << ", " << static_T_platform_imu_.y() << ", " << static_T_platform_imu_.z() << "]"
                    << " q=[" << static_q_platform_imu_.w() << ", " << static_q_platform_imu_.x() << ", " << static_q_platform_imu_.y() << ", " << static_q_platform_imu_.z() << "]");
    
    sub_start = nh.subscribe("/spline_fusion_node/start_time", 1000,
                             &EstimationInterface::startCallBack, this);
    std::string imu_type = CommonUtils::readParam<std::string>(nh, "topic_imu");
    sub_imu =
        nh.subscribe(imu_type, 400, &EstimationInterface::getImuCallback, this);
    pub_imu = nh.advertise<sensor_msgs::Imu>("imu_ds", 400);
    std::string pose_input_topic = CommonUtils::readParam<std::string>(nh, "topic_pose_input", "/pose_data");
    sub_pose_raw = nh.subscribe(pose_input_topic, 1000,
                            &EstimationInterface::getRawPoseCallback, this);
    pub_pose_ds = nh.advertise<geometry_msgs::PoseStamped>("pose_ds", 1000);
    int control_point_fps =
        CommonUtils::readParam<int>(nh, "control_point_fps");
    dt_ns = 1e9 / control_point_fps;
    sub_est = nh.subscribe("/spline_fusion_node/est_window", 100,
                           &EstimationInterface::getEstCallback, this);
    pub_opt_old =
        nh.advertise<nav_msgs::Path>("bspline_optimization_old", 1000);
    pub_opt_window =
        nh.advertise<nav_msgs::Path>("bspline_optimization_window", 1000);
    opt_old_path.header.frame_id = "map";
    pub_fused_pose_stamped = nh.advertise<geometry_msgs::PoseStamped>("fused_pose", 10);
  }

private:
  // TF
  tf2_ros::Buffer tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_ptr_;
  
  // Static transform from platform to IMU (computed once at startup)
  Eigen::Vector3d static_T_platform_imu_;
  Eigen::Quaterniond static_q_platform_imu_;

  int64_t dt_ns;
  double imu_sample_coeff;
  double pose_sample_coeff;
  double output_visualization_fps;
  double imu_frequency;
  double pose_frequency;
  double average_runtime;
  bool gyro_unit;
  bool acc_ratio;
  
  // Data quality parameters
  double max_imu_time_gap_sec_;
  double max_pose_age_sec_;
  double imu_accel_max_norm_;
  double imu_gyro_max_norm_;
  
  // Shared timestamp tracking for data synchronization
  int64_t latest_pose_timestamp_;
  
  SplineState spline_global;
  Eigen::aligned_vector<PoseData> opt_old;
  Eigen::aligned_vector<PoseData> opt_window;
  ros::Subscriber sub_imu;
  ros::Subscriber sub_pose_raw;
  ros::Subscriber sub_est;
  ros::Subscriber sub_start;
  ros::Publisher pub_imu;
  ros::Publisher pub_pose_ds;
  ros::Publisher pub_opt_old;
  ros::Publisher pub_opt_window;
  ros::Publisher pub_opt_pose;
  ros::Publisher pub_fused_pose_stamped;
  nav_msgs::Path opt_old_path;
  std::string imu_frame_;
  std::string platform_frame_id_;
  std::string output_frame_;

  void readParamsInterface(ros::NodeHandle &nh) {
    imu_sample_coeff = CommonUtils::readParam<double>(nh, "imu_sample_coeff");
    pose_sample_coeff = CommonUtils::readParam<double>(nh, "pose_sample_coeff", 1.0);
    imu_frequency = CommonUtils::readParam<double>(nh, "imu_frequency");
    pose_frequency = CommonUtils::readParam<double>(nh, "pose_frequency", 20.0);
    output_visualization_fps = CommonUtils::readParam<double>(nh, "output_visualization_fps", 60.0);
    gyro_unit = CommonUtils::readParam<bool>(nh, "gyro_unit");
    acc_ratio = CommonUtils::readParam<bool>(nh, "acc_ratio");
    imu_frame_ = CommonUtils::readParam<std::string>(nh, "imu_frame", "imu_link");
    platform_frame_id_ = CommonUtils::readParam<std::string>(nh, "platform_frame_id", "camera_link");
    output_frame_ = CommonUtils::readParam<std::string>(nh, "output_frame", "map");
    
    // Data quality parameters
    max_imu_time_gap_sec_ = CommonUtils::readParam<double>(nh, "max_imu_time_gap_sec", 0.5);
    max_pose_age_sec_ = CommonUtils::readParam<double>(nh, "max_pose_age_sec", 2.0);
    imu_accel_max_norm_ = CommonUtils::readParam<double>(nh, "imu_accel_max_norm", 50.0);
    imu_gyro_max_norm_ = CommonUtils::readParam<double>(nh, "imu_gyro_max_norm", 10.0);
    
    // Initialize timestamp tracking
    latest_pose_timestamp_ = 0;
  }

  void getEstCallback(const sfip::Estimate::ConstPtr &est_msg) {
    sfip::Spline spline_msg = est_msg->spline;
    SplineState spline_w;
    spline_w.init(spline_msg.dt, 0, spline_msg.start_t, spline_msg.start_idx);
    for (const auto knot : spline_msg.knots) {
      Eigen::Vector3d pos(knot.position.x, knot.position.y, knot.position.z);
      Eigen::Quaterniond quat(knot.orientation.w, knot.orientation.x,
                              knot.orientation.y, knot.orientation.z);
      Eigen::Matrix<double, 6, 1> bias;
      bias << knot.bias_acc.x, knot.bias_acc.y, knot.bias_acc.z,
          knot.bias_gyro.x, knot.bias_gyro.y, knot.bias_gyro.z;
      spline_w.addSingleStateKnot(quat, pos, bias);
    }
    for (int i = 0; i < 3; i++) {
      sfip::Knot idle = spline_msg.idles[i];
      Eigen::Vector3d t_idle(idle.position.x, idle.position.y, idle.position.z);
      Eigen::Quaterniond q_idle(idle.orientation.w, idle.orientation.x,
                                idle.orientation.y, idle.orientation.z);
      Eigen::Matrix<double, 6, 1> b_idle;
      b_idle << idle.bias_acc.x, idle.bias_acc.y, idle.bias_acc.z,
          idle.bias_gyro.x, idle.bias_gyro.y, idle.bias_gyro.z;
      spline_w.setIdles(i, t_idle, q_idle, b_idle);
    }
    spline_global.updateKnots(&spline_w);
    
    pubOpt(spline_w, !est_msg->if_full_window.data);
    average_runtime = est_msg->runtime.data;

    if (spline_w.getNumKnots() > 0) { 
        geometry_msgs::PoseStamped fused_ps_msg;
        fused_ps_msg.header.stamp.fromNSec(spline_w.maxTimeNanoseconds());
        fused_ps_msg.header.frame_id = "map"; // Or your desired fixed frame
        
        Eigen::Quaterniond q_latest;
        spline_w.interpolateQuaternion(spline_w.maxTimeNanoseconds(), &q_latest);
        Eigen::Vector3d p_latest = spline_w.interpolatePosition(spline_w.maxTimeNanoseconds());

        fused_ps_msg.pose.position.x = p_latest.x();
        fused_ps_msg.pose.position.y = p_latest.y();
        fused_ps_msg.pose.position.z = p_latest.z();
        fused_ps_msg.pose.orientation.w = q_latest.w();
        fused_ps_msg.pose.orientation.x = q_latest.x();
        fused_ps_msg.pose.orientation.y = q_latest.y();
        fused_ps_msg.pose.orientation.z = q_latest.z();
        geometry_msgs::PoseStamped fused_out = transformImuToOutput(fused_ps_msg);
        pub_fused_pose_stamped.publish(fused_out);
    }
  }

  void pubOpt(SplineState &spline_local, const bool if_window_full) {
    int64_t min_t_ns = spline_local.minTimeNanoseconds();
    int64_t max_t_ns = spline_local.maxTimeNanoseconds();
    
    if (min_t_ns >= max_t_ns) return;

    static int64_t last_published_path_end_time_ns = 0;

    if (!if_window_full) {
      for (const auto& pose_data : opt_window) {
        if (pose_data.timestampNanoseconds < min_t_ns && pose_data.timestampNanoseconds > last_published_path_end_time_ns) {
          opt_old.push_back(pose_data);
          opt_old_path.poses.push_back(
              CommonUtils::pose2msg(pose_data.timestampNanoseconds, pose_data.position, pose_data.orientation));
        }
      }
      if (!opt_window.empty()) {
          last_published_path_end_time_ns = opt_window.back().timestampNanoseconds;
      } else if (min_t_ns > last_published_path_end_time_ns) {
          last_published_path_end_time_ns = min_t_ns;
      }

    } else {
      opt_old.clear();
      opt_old_path.poses.clear();
      last_published_path_end_time_ns = 0;
    }

    opt_window.clear();
    nav_msgs::Path opt_window_path;
    opt_window_path.header.frame_id = "map";
    opt_window_path.header.stamp.fromNSec(min_t_ns);

    double viz_dt_s = 1.0 / output_visualization_fps;
    int64_t viz_dt_ns = static_cast<int64_t>(viz_dt_s * 1e9);

    for (int64_t t_ns = min_t_ns; t_ns <= max_t_ns; t_ns += viz_dt_ns) {
      PoseData pose_from_spline;
      spline_local.interpolateQuaternion(t_ns, &pose_from_spline.orientation);
      pose_from_spline.position = spline_local.interpolatePosition(t_ns);
      pose_from_spline.timestampNanoseconds = t_ns;
      
      opt_window.push_back(pose_from_spline);
      opt_window_path.poses.push_back(
          CommonUtils::pose2msg(t_ns, pose_from_spline.position, pose_from_spline.orientation));
      if (t_ns == min_t_ns && if_window_full) {
          opt_old_path.poses.push_back(CommonUtils::pose2msg(t_ns, pose_from_spline.position, pose_from_spline.orientation));
      }
    }
    if (!opt_window.empty() && opt_window.back().timestampNanoseconds < max_t_ns) {
        PoseData pose_at_max_t;
        spline_local.interpolateQuaternion(max_t_ns, &pose_at_max_t.orientation);
        pose_at_max_t.position = spline_local.interpolatePosition(max_t_ns);
        pose_at_max_t.timestampNanoseconds = max_t_ns;
        opt_window.push_back(pose_at_max_t);
        opt_window_path.poses.push_back(
            CommonUtils::pose2msg(max_t_ns, pose_at_max_t.position, pose_at_max_t.orientation));
    }

    pub_opt_old.publish(opt_old_path);
    pub_opt_window.publish(opt_window_path);
    
    if (!opt_window.empty()) {
        // opt_pose_vis.pubPose(opt_window.back().position, opt_window.back().orientation,
        //                  opt_window.back().timestampNanoseconds);
    }
  }

  bool getStaticTransform() {
    // Try to get the static transform from platform to IMU frame
    int retry_count = 0;
    const int max_retries = 50;
    const double retry_delay = 0.1; // seconds
    
    while (retry_count < max_retries && ros::ok()) {
      try {
        geometry_msgs::TransformStamped tf_stamped = tf_buffer_.lookupTransform(
          imu_frame_, platform_frame_id_, ros::Time(0), ros::Duration(retry_delay));
        
        // Extract translation and rotation
        static_T_platform_imu_.x() = tf_stamped.transform.translation.x;
        static_T_platform_imu_.y() = tf_stamped.transform.translation.y;
        static_T_platform_imu_.z() = tf_stamped.transform.translation.z;
        
        static_q_platform_imu_.w() = tf_stamped.transform.rotation.w;
        static_q_platform_imu_.x() = tf_stamped.transform.rotation.x;
        static_q_platform_imu_.y() = tf_stamped.transform.rotation.y;
        static_q_platform_imu_.z() = tf_stamped.transform.rotation.z;
        static_q_platform_imu_.normalize();
        
        ROS_INFO("[EI] Successfully obtained static transform from platform to IMU frame.");
        return true;
        
      } catch (const tf2::TransformException& ex) {
        retry_count++;
        ROS_WARN_STREAM_THROTTLE(1.0, "[EI] Attempt " << retry_count << "/" << max_retries 
                                << " to get static transform failed: " << ex.what());
        
        if (retry_count < max_retries) {
          ros::Duration(retry_delay).sleep();
        }
      }
    }
    
    ROS_ERROR("[EI] Failed to get static transform after %d attempts. Make sure static_transform_publisher is running.", max_retries);
    return false;
  }

  void getImuCallback(const sensor_msgs::Imu::ConstPtr &imu_msg) {
    static int64_t last_imu = 0;
    int64_t t_ns = imu_msg->header.stamp.toNSec();
    
    // Validate timestamp is in sequence
    if (last_imu > 0 && t_ns <= last_imu) {
        ROS_WARN_STREAM_THROTTLE(1.0, "[EI] IMU timestamp out of sequence: current=" << t_ns << " last=" << last_imu << ". Skipping.");
        return;
    }
    
    // Check for large time gaps between consecutive IMU measurements
    if (last_imu > 0) {
        double dt_sec = (t_ns - last_imu) * 1e-9;
        if (dt_sec > max_imu_time_gap_sec_) {
            ROS_WARN_STREAM_THROTTLE(1.0, "[EI] Large IMU time gap detected: " << dt_sec << " seconds. Skipping.");
            return;
        }
    }
    
    // Check if timestamp is too far in the past
    ros::Time now = ros::Time::now();
    int64_t now_ns = now.toNSec();
    if (now_ns - t_ns > 1e9) {
        ROS_WARN_STREAM_THROTTLE(5.0, "[EI] IMU timestamp too old: " << (now_ns - t_ns) / 1e9 << " seconds. Skipping.");
        return;
    }
    
    // Pre-filter IMU data before unit conversion
    Eigen::Vector3d acc_raw(imu_msg->linear_acceleration.x,
                            imu_msg->linear_acceleration.y,
                            imu_msg->linear_acceleration.z);
    Eigen::Vector3d gyro_raw(imu_msg->angular_velocity.x,
                             imu_msg->angular_velocity.y,
                             imu_msg->angular_velocity.z);
    
    // Apply unit conversions first
    if (acc_ratio) acc_raw *= 9.81;
    if (gyro_unit) gyro_raw *= M_PI / 180.0;
    
    // Check for unreasonable values after unit conversion
    if (acc_raw.norm() > imu_accel_max_norm_) {
        ROS_WARN_STREAM_THROTTLE(1.0, "[EI] IMU acceleration too large: " << acc_raw.norm() << " m/s^2. Skipping.");
        return;
    }
    
    if (gyro_raw.norm() > imu_gyro_max_norm_) {
        ROS_WARN_STREAM_THROTTLE(1.0, "[EI] IMU gyroscope too large: " << gyro_raw.norm() << " rad/s. Skipping.");
        return;
    }
    
    // IMPORTANT: Don't accept IMU data that is older than the latest pose measurement
    // This prevents optimization instability from temporal ordering issues
    if (latest_pose_timestamp_ > 0 && t_ns < latest_pose_timestamp_) {
        ROS_DEBUG_STREAM_THROTTLE(1.0, "[EI] IMU timestamp " << t_ns << " is older than latest pose " << latest_pose_timestamp_ << ". Skipping.");
        return;
    }
    
    if (sampleData(t_ns, last_imu, imu_sample_coeff, imu_frequency)) {
      last_imu = t_ns;
      sensor_msgs::Imu imu_ds_msg;
      imu_ds_msg.header = imu_msg->header;
      imu_ds_msg.linear_acceleration.x = acc_raw[0];
      imu_ds_msg.linear_acceleration.y = acc_raw[1];
      imu_ds_msg.linear_acceleration.z = acc_raw[2];
      imu_ds_msg.angular_velocity.x = gyro_raw[0];
      imu_ds_msg.angular_velocity.y = gyro_raw[1];
      imu_ds_msg.angular_velocity.z = gyro_raw[2];
      pub_imu.publish(imu_ds_msg);
    }
  }

  void getRawPoseCallback(
      const geometry_msgs::PoseStampedConstPtr &pose_msg) {
    static int64_t last_pose_input_ns = 0;
    int64_t input_t_ns = pose_msg->header.stamp.toNSec();
    
    // Validate timestamp is in sequence
    if (last_pose_input_ns > 0 && input_t_ns <= last_pose_input_ns) {
        ROS_WARN_STREAM_THROTTLE(1.0, "[EI] Pose timestamp out of sequence: current=" << input_t_ns << " last=" << last_pose_input_ns << ". Skipping.");
        return;
    }
    
    // Check if timestamp is too far in the past
    ros::Time now = ros::Time::now();
    int64_t now_ns = now.toNSec();
    double age_sec = (now_ns - input_t_ns) * 1e-9;
    if (age_sec > max_pose_age_sec_) {
        ROS_WARN_STREAM_THROTTLE(5.0, "[EI] Pose timestamp too old: " << age_sec << " seconds. Skipping.");
        return;
    }
    
    geometry_msgs::PoseStamped pose_in = *pose_msg;
    geometry_msgs::PoseStamped pose_in_imu;
    if (!transformPoseToImuStaticOnly(pose_in, pose_in_imu))
        return; // skip if transformation fails

    Eigen::Quaterniond q(pose_in_imu.pose.orientation.w, pose_in_imu.pose.orientation.x,
                         pose_in_imu.pose.orientation.y, pose_in_imu.pose.orientation.z);
    Eigen::Vector3d pos(pose_in_imu.pose.position.x, pose_in_imu.pose.position.y,
                        pose_in_imu.pose.position.z);

    // Use the transformed pose timestamp for consistency
    PoseData pose_internal(pose_in_imu.header.stamp.toNSec(), q, pos);
    
    // Update last pose timestamp for IMU filtering
    latest_pose_timestamp_ = pose_internal.timestampNanoseconds;

    static int64_t last_pose_raw_ns = 0;
    if (sampleData(pose_internal.timestampNanoseconds, last_pose_raw_ns, pose_sample_coeff, pose_frequency)) {
        geometry_msgs::PoseStamped pose_stamped_msg = pose_in_imu; // already in IMU frame
        pub_pose_ds.publish(pose_stamped_msg);
        last_pose_raw_ns = pose_internal.timestampNanoseconds;
    }
    
    last_pose_input_ns = input_t_ns;
  }

  bool transformPoseToImuStaticOnly(const geometry_msgs::PoseStamped& in, geometry_msgs::PoseStamped& out)
  {
      // Use the pre-computed static transform only
      // Assume input pose is in platform frame (or can be transformed to platform frame)
      
      // If input is already in platform frame, apply static transform directly
      if (in.header.frame_id == platform_frame_id_) {
          // Apply static transform: T_imu = T_imu_platform * T_platform_world * pose_in_world
          // Since pose_in is in platform frame: T_imu = T_imu_platform * pose_in_platform
          
          Eigen::Vector3d pos_in(in.pose.position.x, in.pose.position.y, in.pose.position.z);
          Eigen::Quaterniond q_in(in.pose.orientation.w, in.pose.orientation.x, 
                                  in.pose.orientation.y, in.pose.orientation.z);
          
          // Transform position: p_imu = R_imu_platform * p_platform + t_imu_platform
          Eigen::Vector3d pos_out = static_q_platform_imu_ * pos_in + static_T_platform_imu_;
          
          // Transform orientation: q_imu = q_imu_platform * q_platform
          Eigen::Quaterniond q_out = static_q_platform_imu_ * q_in;
          q_out.normalize();
          
          out.header = in.header;
          out.header.frame_id = imu_frame_;
          out.pose.position.x = pos_out.x();
          out.pose.position.y = pos_out.y();
          out.pose.position.z = pos_out.z();
          out.pose.orientation.w = q_out.w();
          out.pose.orientation.x = q_out.x();
          out.pose.orientation.y = q_out.y();
          out.pose.orientation.z = q_out.z();
          
          return true;
      } else {
          // If not in platform frame, try to get transform to platform frame first
          try {
              geometry_msgs::PoseStamped pose_platform;
              geometry_msgs::TransformStamped tf_to_platform = tf_buffer_.lookupTransform(
                  platform_frame_id_, in.header.frame_id, ros::Time(0), ros::Duration(0.1));
              tf2::doTransform(in, pose_platform, tf_to_platform);
              
              // Now apply static transform
              return transformPoseToImuStaticOnly(pose_platform, out);
              
          } catch (const tf2::TransformException& ex) {
              ROS_WARN_STREAM_THROTTLE(2.0, "[EI] transformPoseToImuStaticOnly failed to get transform to platform frame: " << ex.what());
              return false;
          }
      }
  }

  bool transformPoseToImu(const geometry_msgs::PoseStamped& in, geometry_msgs::PoseStamped& out)
  {
      try {
          // Use ros::Time(0) to get the latest available transform when playing bags with sim_time
          ros::Time lookup_time = in.header.stamp;
          if (ros::param::param<bool>("/use_sim_time", false)) {
              // In sim time mode with bags, use the exact timestamp but with short timeout
              geometry_msgs::TransformStamped tf = tf_buffer_.lookupTransform(imu_frame_, in.header.frame_id,
                                                                              lookup_time, ros::Duration(0.05));
              tf2::doTransform(in, out, tf);
          } else {
              // In real-time mode, use latest available
              geometry_msgs::TransformStamped tf = tf_buffer_.lookupTransform(imu_frame_, in.header.frame_id,
                                                                              ros::Time(0), ros::Duration(0.1));
              tf2::doTransform(in, out, tf);
              // Keep original timestamp for real-time mode
              out.header.stamp = in.header.stamp;
          }
          out.header.frame_id = imu_frame_;
          return true;
      } catch (const tf2::TransformException& ex) {
          ROS_WARN_STREAM_THROTTLE(2.0, "[EI] transformPoseToImu failed: " << ex.what());
          return false;
      }
  }

  geometry_msgs::PoseStamped transformImuToOutput(const geometry_msgs::PoseStamped& in)
  {
      geometry_msgs::PoseStamped out = in;
      if (output_frame_ == imu_frame_) return out;
      try {
          // Use ros::Time(0) for output transform to avoid extrapolation warnings
          geometry_msgs::TransformStamped tf = tf_buffer_.lookupTransform(output_frame_, imu_frame_,
                                                                          ros::Time(0), ros::Duration(0.1));
          tf2::doTransform(in, out, tf);
          out.header.frame_id = output_frame_;
          // Preserve the original timestamp for the fused pose
          out.header.stamp = in.header.stamp;
      } catch (const tf2::TransformException& ex) {
          ROS_WARN_STREAM_THROTTLE(2.0, "[EI] transformImuToOutput failed: " << ex.what());
      }
      return out;
  }

  bool sampleData(const int64_t t_ns, const int64_t last_t_ns,
                  const double coeff, const double frequency) const {
    if (coeff == 0)
      return false;
    int64_t dt = 1e9 / (coeff * frequency);
    if (coeff == 1) {
      return true;
    } else if (t_ns - last_t_ns > dt - 1e5) {
      return true;
    } else {
      return false;
    }
  }

  void startCallBack(const std_msgs::Int64::ConstPtr &start_time_msg) {
    int64_t bag_start_time = start_time_msg->data;
    spline_global.init(dt_ns, 0, bag_start_time);
  }
};

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "sfuise");
  ROS_INFO("\033[1;32m---->\033[0m Starting EstimationInterface.");
  ros::NodeHandle nh("~");
  EstimationInterface interface(nh);
  ros::Rate rate(1000);
  while (ros::ok()) {
    ros::spinOnce();
    rate.sleep();
  }
  return 0;
}
