#include "Accumulator.hpp"
#include "Linearizer.hpp"
#include "utils/tic_toc.hpp"
#include "sensor_msgs/Imu.h"
#include "sfip/Estimate.h"
#include "sfip/Spline.h"
#include "std_msgs/Int64.h"
#include <ros/ros.h>
#include "geometry_msgs/PoseStamped.h"
#include "sensor_msgs/PointCloud2.h"
#include <algorithm>

class SplineFusion {

public:
  SplineFusion(ros::NodeHandle &nh) {
    average_runtime = 0;
    window_count = 0;
    solver_flag = INITIAL;
    readParameters(nh);
    sub_imu = nh.subscribe("/estimation_interface_node/imu_ds", 1000,
                           &SplineFusion::getImuCallback, this);
    sub_pose = nh.subscribe("/estimation_interface_node/pose_ds", 1000,
                           &SplineFusion::getPoseCallback, this);
    pub_est = nh.advertise<sfip::Estimate>("est_window", 1000);
    pub_start_time = nh.advertise<std_msgs::Int64>("start_time", 1000);
  }

  void run() {
    static int num_window = 0;
    TicToc t_window;
    if (initialization()) {

      // displayControlPoints();
      bool is_converged = optimization();
      ROS_INFO_STREAM_COND(is_converged, "[SF] run: Optimization converged.");
      ROS_INFO_STREAM_COND(!is_converged, "[SF] run: Optimization did NOT converge (max iterations reached).");

      double t_consum = t_window.toc();
      average_runtime = (t_consum + double(num_window) * average_runtime) /
                        double(num_window + 1);
      num_window++;
      if (spline_local.getNumKnots() >= (size_t)window_size) {
        window_count++;
        if (solver_flag == INITIAL) {
          ROS_INFO_STREAM("[SF] run: Solver transitioning from INITIAL to FULLSIZE. num_knots=" << spline_local.getNumKnots() << " window_size=" << window_size);
          solver_flag = FULLSIZE;
        }
      }
      sfip::Spline spline_msg;
      spline_local.getSplineMsg(spline_msg);
      
      if (spline_local.getNumKnots() > 0) {
          size_t last_knot_idx = spline_local.getNumKnots() - 1;
          Eigen::Vector3d last_pos = spline_local.getKnotPosition(last_knot_idx);
          Eigen::Quaterniond last_rot = spline_local.getKnotOrientation(last_knot_idx);
          ROS_INFO_STREAM("[SF] run: Last knot state before publish: Pos=[" << last_pos.x() << ", " << last_pos.y() << ", " << last_pos.z() << "], Rot(w,x,y,z)=[" << last_rot.w() << ", " << last_rot.x() << ", " << last_rot.y() << ", " << last_rot.z() << "]");
      }

      sfip::Estimate est_msg;
      est_msg.spline = spline_msg;
      est_msg.if_full_window.data = (solver_flag != INITIAL);
      est_msg.runtime.data = average_runtime;
      ROS_INFO_STREAM("[SF] run: Publishing estimate. solver_flag=" << (solver_flag == INITIAL ? "INITIAL" : "FULLSIZE") << ", if_full_window=" << (est_msg.if_full_window.data ? "true" : "false") << ", num_knots=" << spline_local.getNumKnots());
      pub_est.publish(est_msg);
      // displayControlPoints();
      if (solver_flag == FULLSIZE)
        spline_local.removeSingleOldState();
    } else {
        ROS_INFO_STREAM_THROTTLE(1.0, "[SF] run: Waiting for initialization() to return true...");
    }
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  static constexpr double NS_TO_S = 1e-9;

  CalibParam calib_param;

  ros::Subscriber sub_imu;
  ros::Subscriber sub_pose;
  ros::Publisher pub_est;
  ros::Publisher pub_start_time;

  Parameters param;

  Eigen::aligned_deque<ImuData> imu_buff;
  Eigen::aligned_deque<PoseData> pose_buff;

  Eigen::aligned_deque<ImuData> imu_window;
  Eigen::aligned_deque<PoseData> pose_window;

  size_t window_count;
  int window_size;
  bool if_pose_only;


  int64_t dt_ns;
  int64_t bag_start_time;
  int64_t last_imu_t_ns;
  int64_t next_knot_TimeNs;

  enum SolverFlag { INITIAL, FULLSIZE };
  SolverFlag solver_flag;
  SplineState spline_local;

  size_t bias_block_offset;
  size_t gravity_block_offset;
  size_t hess_size;
  bool pose_fixed;
  int max_iter;
  double lambda;
  double lambda_vee;
  double average_runtime;

  void readParameters(ros::NodeHandle &nh) {
    if (CommonUtils::readParam<double>(nh, "imu_sample_coeff") == 0) {
      if_pose_only = true;
    } else {
      if_pose_only = false;
    }
    param.if_opt_g = CommonUtils::readParam<bool>(nh, "if_opt_g", true);
    max_iter = CommonUtils::readParam<int>(nh, "max_iter");
    dt_ns = 1e9 / CommonUtils::readParam<int>(nh, "control_point_fps");
    bag_start_time = 0;
    window_size = CommonUtils::readParam<int>(nh, "window_size");
    std::vector<double> accel_var_inv =
        CommonUtils::readParam<std::vector<double>>(nh, "accel_var_inv");
    param.accel_var_inv << accel_var_inv.at(0), accel_var_inv.at(1),
        accel_var_inv.at(2);
    std::vector<double> bias_accel_var_inv =
        CommonUtils::readParam<std::vector<double>>(nh, "bias_accel_var_inv");
    param.bias_accel_var_inv << bias_accel_var_inv.at(0),
        bias_accel_var_inv.at(1), bias_accel_var_inv.at(2);
    param.w_acc = CommonUtils::readParam<double>(nh, "w_accel");
    param.w_bias_acc = CommonUtils::readParam<double>(nh, "w_bias_accel");
    std::vector<double> gyro_var_inv =
        CommonUtils::readParam<std::vector<double>>(nh, "gyro_var_inv");
    param.gyro_var_inv << gyro_var_inv.at(0), gyro_var_inv.at(1),
        gyro_var_inv.at(2);
    std::vector<double> bias_gyro_var_inv =
        CommonUtils::readParam<std::vector<double>>(nh, "bias_gyro_var_inv");
    param.bias_gyro_var_inv << bias_gyro_var_inv.at(0), bias_gyro_var_inv.at(1),
        bias_gyro_var_inv.at(2);
    param.w_gyro = CommonUtils::readParam<double>(nh, "w_gyro");
    param.w_bias_gyro = CommonUtils::readParam<double>(nh, "w_bias_gyro");
    param.w_pose_pos = CommonUtils::readParam<double>(nh, "w_pose_pos", 1.0);
    param.w_pose_rot = CommonUtils::readParam<double>(nh, "w_pose_rot", 1.0);
    // Optimizer damping/LM parameters
    lambda = CommonUtils::readParam<double>(nh, "initial_lambda", 1e-6);
    lambda_vee = CommonUtils::readParam<double>(nh, "initial_lambda_vee", 2.0);
    std::vector<double> gravity_vec = 
        CommonUtils::readParam<std::vector<double>>(nh, "gravity_initial", {0.0, 0.0, -9.80665});
    if(gravity_vec.size() == 3) {
        param.gravity << gravity_vec[0], gravity_vec[1], gravity_vec[2];
    } else {
        ROS_WARN("Initial gravity vector not specified correctly or missing, using default [0,0,-9.80665]");
        param.gravity << 0.0, 0.0, -9.80665;
    }
    calib_param.gravity = param.gravity;

    if (if_pose_only) {
        // No IMU information => gravity unobservable; disable optimisation
        param.if_opt_g = false;
    }
  }

  void getImuCallback(const sensor_msgs::ImuConstPtr &imu_msg) {
    int64_t t_ns = imu_msg->header.stamp.toNSec();
    
    // Validate IMU timestamp sequence
    if (!imu_buff.empty() && t_ns <= imu_buff.back().timestampNanoseconds) {
        ROS_WARN_STREAM_THROTTLE(1.0, "[SF] IMU timestamp out of sequence: current=" << t_ns << " last=" << imu_buff.back().timestampNanoseconds << ". Skipping.");
        return;
    }
    
    Eigen::Vector3d acc(imu_msg->linear_acceleration.x,
                        imu_msg->linear_acceleration.y,
                        imu_msg->linear_acceleration.z);
    Eigen::Vector3d gyro(imu_msg->angular_velocity.x,
                         imu_msg->angular_velocity.y,
                         imu_msg->angular_velocity.z);
    ImuData imu(t_ns, gyro, acc);
    imu_buff.push_back(imu);
    
    ROS_DEBUG_STREAM_THROTTLE(1.0, "[SF] IMU buffer size: " << imu_buff.size() << ", latest timestamp: " << t_ns);
  }

  void getPoseCallback(const geometry_msgs::PoseStampedConstPtr& pose_msg) {
    int64_t t_ns = pose_msg->header.stamp.toNSec();
    
    // Validate pose timestamp sequence
    if (!pose_buff.empty() && t_ns <= pose_buff.back().timestampNanoseconds) {
        ROS_WARN_STREAM_THROTTLE(1.0, "[SF] Pose timestamp out of sequence: current=" << t_ns << " last=" << pose_buff.back().timestampNanoseconds << ". Skipping.");
        return;
    }
    
    Eigen::Quaterniond q(pose_msg->pose.orientation.w, pose_msg->pose.orientation.x,
                         pose_msg->pose.orientation.y, pose_msg->pose.orientation.z);
    Eigen::Vector3d p(pose_msg->pose.position.x, pose_msg->pose.position.y,
                      pose_msg->pose.position.z);
    PoseData pose(t_ns, q, p);
    pose_buff.push_back(pose);
    
    ROS_DEBUG_STREAM_THROTTLE(1.0, "[SF] Pose buffer size: " << pose_buff.size() << ", latest timestamp: " << t_ns);
  }

  template <typename type_data>
  void updateMeasurements(Eigen::aligned_deque<type_data> &data_window,
                          Eigen::aligned_deque<type_data> &data_buff) {
    int64_t t_window_l = spline_local.minTimeNanoseconds();
    if (!data_window.empty()) {
      while (data_window.front().timestampNanoseconds < t_window_l) {
        data_window.pop_front();
      }
    }
    int64_t t_window_r = spline_local.maxTimeNanoseconds();

    size_t added_count = 0;
    for (size_t i = 0; i < data_buff.size(); i++) {
      auto v = data_buff.at(i);
      if (v.timestampNanoseconds >= t_window_l && v.timestampNanoseconds <= t_window_r) {
        data_window.push_back(v);
        added_count++;
      } else if (v.timestampNanoseconds > t_window_r) {
        break;
      }
    }

    while (data_buff.front().timestampNanoseconds <= t_window_r) {
      data_buff.pop_front();
      if (data_buff.empty())
        break;
    }

    // Ensure the window is strictly time-ordered (can be violated when bag
    // publishes multiple topics with slightly different latencies)
    std::sort(data_window.begin(), data_window.end(),
              [](const type_data& a, const type_data& b){return a.timestampNanoseconds < b.timestampNanoseconds;});
  }

  bool initialization() {
    if(if_pose_only) {
      if (pose_buff.empty())
        return false;
    } else {
      if (imu_buff.empty() && pose_buff.empty())
        return false;
    }
    static bool param_set = false;
    static bool initialize_control_point = false;
    if (initialize_control_point) {

      int64_t min_time_available = 1e18;
      if (!imu_buff.empty()){
          min_time_available = imu_buff.back().timestampNanoseconds;
      }
      if(!pose_buff.empty()){
          min_time_available = std::min(min_time_available, pose_buff.back().timestampNanoseconds);
      }
      if (min_time_available > spline_local.nextMaxTimeNanoseconds()) {
          Eigen::Quaterniond q_ini =
            spline_local.getKnotOrientation(spline_local.getNumKnots() - 1);
        Eigen::Quaterniond q_ini_backup = q_ini;
        Eigen::Vector3d pos_ini =
            spline_local.getKnotPosition(spline_local.getNumKnots() - 1);
        Eigen::Matrix<double, 6, 1> bias_ini =
            spline_local.getKnotBias(spline_local.getNumKnots() - 1);
        if (!if_pose_only) {
          ROS_INFO_STREAM("[SF] initialization: numKnots = " << spline_local.getNumKnots() << ", current last_imu_t_ns = " << last_imu_t_ns << ", imu_window size = " << imu_window.size());
          if (spline_local.getNumKnots() <= 2) {
            last_imu_t_ns = bag_start_time;
            ROS_INFO_STREAM("[SF] initialization: Using bag_start_time for last_imu_t_ns: " << last_imu_t_ns);
          } else {
            if (imu_window.empty()) {
              ROS_WARN_THROTTLE(1.0, "[SF] initialization: IMU window is empty when expecting data for integration. Using latest knot time as fallback for last_imu_t_ns.");
              if (spline_local.getNumKnots() > 0) {
              last_imu_t_ns = spline_local.getKnotTimeNanoseconds(spline_local.getNumKnots() - 1);
              ROS_INFO_STREAM("[SF] initialization: IMU window empty. Fallback last_imu_t_ns to " << last_imu_t_ns);
              } else {
                ROS_ERROR("[SF] initialization: No knots available for fallback timestamp");
                return false;
              }
            } else {
              last_imu_t_ns = imu_window.back().timestampNanoseconds;
              ROS_INFO_STREAM("[SF] initialization: Using imu_window.back().timestampNanoseconds for last_imu_t_ns: " << last_imu_t_ns);
            }
          }
        integration(next_knot_TimeNs, q_ini, pos_ini);
        }
        
        if (q_ini_backup.dot(q_ini) < 0)
          q_ini = Eigen::Quaterniond(-q_ini.w(), -q_ini.x(), -q_ini.y(),
                                     -q_ini.z());
        spline_local.addSingleStateKnot(q_ini, pos_ini, bias_ini);
        next_knot_TimeNs += dt_ns;
        return true;
      } else {
        return false;
      }
    } else {
      if (!param_set) {
        param_set = setParameters();
        std_msgs::Int64 start_time;
        start_time.data = bag_start_time;
        pub_start_time.publish(start_time);
      }
      if (param_set) {
        spline_local.init(dt_ns, 0, bag_start_time);
        
        if (!if_pose_only) {
          Eigen::Vector3d gravity_sum(0, 0, 0);
          size_t n_imu = imu_buff.size();
          for (size_t i = 0; i < n_imu; i++) {
            gravity_sum += imu_buff.at(i).accel;
          }
          gravity_sum /= n_imu;
          if (n_imu > 0 && gravity_sum.norm() > 1.0) {
            Eigen::Vector3d gravity_ave = gravity_sum.normalized() * 9.80665;
            calib_param.gravity = gravity_ave;
            ROS_INFO_STREAM("Initial gravity estimated from IMU and updated in calib_param: " << calib_param.gravity.transpose());
          } else if (n_imu > 0) {
            ROS_WARN("Initial IMU readings for gravity estimation seem small/invalid or n_imu=0. Using configured/default gravity for calib_param.");
          }
        }

        initialize_control_point = true;
        // Mandatory: wait until at least one pose measurement is available so we
        // can anchor the spline.  This avoids a large initial residual.
        if (pose_buff.empty()) {
          ROS_INFO_THROTTLE(1.0, "[SF] initialization: waiting for first pose measurement to anchor spline ...");
          return false;
        }

        const Eigen::Quaterniond q_anchor = pose_buff.front().orientation;
        const Eigen::Vector3d    p_anchor = pose_buff.front().position;
        ROS_INFO_STREAM("[SF] initialization: Anchoring spline to first pose. Pos=" << p_anchor.transpose()
                        << "  Rot(w,x,y,z)=[" << q_anchor.w() << ", " << q_anchor.x() << ", " << q_anchor.y() << ", " << q_anchor.z() << "]");

        const int num_initial_knots = 2;   // anchor + duplicate to keep zero-velocity start
        for (int i = 0; i < num_initial_knots; ++i) {
          Eigen::Matrix<double, 6, 1> bias_ini = Eigen::Matrix<double, 6, 1>::Zero();
          spline_local.addSingleStateKnot(q_anchor, p_anchor, bias_ini);
          next_knot_TimeNs += dt_ns;
        }
      }
      return false;
    }
  }

  bool optimization() {
    if (spline_local.getNumKnots() < 2)
      return false;
    // if (if_pose_only && pose_window.empty()) {
    //     ROS_DEBUG_THROTTLE(1.0, "Optimization skipped: Pose only mode and no Pose data in the current window.");
    //     return false;
    // }
    // if (!if_pose_only && imu_window.empty() && pose_window.empty()) {
    //   ROS_DEBUG_THROTTLE(
    //       1.0,
    //       "Optimization skipped: No IMU or Pose data in the current window.");
    //   return false;
    // }

    if(!if_pose_only) {
      updateMeasurements(imu_window, imu_buff);
      if (imu_window.empty() && pose_window.empty() && !if_pose_only){
          ROS_DEBUG_THROTTLE(1.0, "Optimization skipped: Still no IMU data after update and pose window is empty.");
          return false;
      }
    }
    updateMeasurements(pose_window, pose_buff);
     if (if_pose_only && pose_window.empty()){
          ROS_DEBUG_THROTTLE(1.0, "Optimization skipped: Pose only mode and still no pose data after update.");
          return false;
     }
    
    bool converged = false;
    int opt_iter = 0;
    pose_fixed = false;
    if (solver_flag == INITIAL) {
      ROS_INFO_STREAM("[SF] optimization (outer): solver_flag is INITIAL. pose_fixed=true. num_knots=" << spline_local.getNumKnots());
      pose_fixed = true;
    } else {
      ROS_INFO_STREAM("[SF] optimization (outer): solver_flag is FULLSIZE. pose_fixed=false. num_knots=" << spline_local.getNumKnots());
    }

    updateLinearizerSize();

    while (!converged && opt_iter < max_iter) {
      converged = optimize(opt_iter);
      opt_iter++;
    }
    ROS_INFO_STREAM("[SF] optimization (outer): Finished with " << opt_iter << " iterations. Converged: " << std::boolalpha << converged);
    return converged;
  }

  bool setParameters() {
    if (if_pose_only) {
        if (pose_buff.empty()) return false;
        bag_start_time = pose_buff.front().timestampNanoseconds;
    } else {
        if (imu_buff.empty() && pose_buff.empty()) return false;
        if (!imu_buff.empty()) {
            bag_start_time = imu_buff.front().timestampNanoseconds;
            if (!pose_buff.empty()) {
                bag_start_time = std::min(bag_start_time, pose_buff.front().timestampNanoseconds);
            }
        } else {
            bag_start_time = pose_buff.front().timestampNanoseconds;
        }
    }
    next_knot_TimeNs = bag_start_time;
    ROS_INFO_STREAM("SplineFusion parameters set. Bag start time: " << bag_start_time);
    return true;
  }

  void integrateStep(int64_t prevTime, int64_t dt_, const ImuData &imu,
                     Eigen::Matrix3d &Rs, Eigen::Vector3d &Ps,
                     Eigen::Vector3d &Vs) {
    static bool first_imu = false;
    static Eigen::Vector3d acc_0;
    static Eigen::Vector3d gyr_0;
    Eigen::Vector3d linear_acceleration = imu.accel;
    Eigen::Vector3d angular_velocity = imu.gyro;
    if (!first_imu) {
      first_imu = true;
      acc_0 = linear_acceleration;
      gyr_0 = angular_velocity;
    }
    Eigen::Vector3d g = calib_param.gravity;
    Eigen::Vector3d ba;
    Eigen::Vector3d bg;
    Eigen::Matrix<double, 6, 1> bias = spline_local.interpolateBias(prevTime);
    ba = bias.head<3>();
    bg = bias.tail<3>();
    double dt = dt_ * NS_TO_S;
    Eigen::Vector3d un_acc_0;
    un_acc_0 = Rs * (acc_0 - ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - bg;
    Rs *= Quater::deltaQ(un_gyr * dt).toRotationMatrix();
    Eigen::Vector3d un_acc_1 = Rs * (linear_acceleration - ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    Ps += dt * Vs + 0.5 * dt * dt * un_acc;
    Vs += dt * un_acc;
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
  }

  void integration(const int64_t curTime, Eigen::Quaterniond &qs,
                   Eigen::Vector3d &Ps) {
    std::vector<ImuData> imu_vec;
    ROS_INFO_STREAM("[SF] integration: Attempting to get IMU data for interval [" << last_imu_t_ns << ", " << curTime << "]");
    getIMUInterval(last_imu_t_ns, curTime, imu_vec);
    ROS_INFO_STREAM("[SF] integration: Found " << imu_vec.size() << " IMU messages for the interval.");
    if (!imu_vec.empty()) {
      Eigen::Quaterniond qs0;
      spline_local.interpolateQuaternion(last_imu_t_ns, &qs0);
      Eigen::Matrix3d Rs0(qs0);
      Eigen::Vector3d Ps0 = spline_local.interpolatePosition(last_imu_t_ns);
      Eigen::Vector3d Vs0 = spline_local.interpolatePosition<1>(last_imu_t_ns);

      for (size_t i = 0; i < imu_vec.size(); i++) {
        int64_t dt;
        int64_t t_ns = imu_vec[i].timestampNanoseconds;
        if (i == 0) {
          dt = t_ns - last_imu_t_ns;
        } else {
          dt = t_ns - imu_vec[i - 1].timestampNanoseconds;
        }
        if (dt <= 0) {
            // ROS_WARN_THROTTLE(1.0, "integration: dt_ns_step (duration %ld ns) is zero or negative for IMU @ %ld. Skipping this IMU step.", dt_ns_step, current_imu_timestampNanoseconds);
            continue; 
        }
        integrateStep(last_imu_t_ns, dt, imu_vec[i], Rs0, Ps0, Vs0);
      }
      qs = Eigen::Quaterniond(Rs0);
      Ps = Ps0;
    } else {
      qs = spline_local.extrapolateKnotOrientation(1);
      Ps = spline_local.extrapolateKnotPosition(1);
    }
  }

  bool getIMUInterval(int64_t t0, int64_t t1, std::vector<ImuData> &imu_vec) {
    imu_vec.clear();
    if (imu_buff.empty()) {
      ROS_INFO_THROTTLE(1.0, "getIMUInterval: No IMU data in imu_buff.");
      return false;
    }
    int idx = 0;
    while (imu_buff.at(idx).timestampNanoseconds <= std::min(imu_buff.back().timestampNanoseconds, t1)) {
      imu_vec.push_back(imu_buff.at(idx));
      idx++;
      if (idx >= imu_buff.size())
        break;
    }
    return true;
  }

  bool optimize(const int iter) {
    Linearizer lopt(bias_block_offset, gravity_block_offset, hess_size,
                    &spline_local, &param, pose_fixed);
    
    if (!if_pose_only && !imu_window.empty())
      lopt(imu_window);
    
    if (!pose_window.empty())
        lopt(pose_window);
    
    ROS_INFO_STREAM("[SF] optimize: iter=" << iter << "  initial_cost=" << lopt.error << "  lambda=" << lambda 
                    << "  imu_window_size=" << imu_window.size() << "  pose_window_size=" << pose_window.size());
    
    // Check for numerical issues
    if (!std::isfinite(lopt.error) || lopt.error > 1e12) {
        ROS_ERROR_STREAM("[SF] optimize: Cost is not finite or too large: " << lopt.error << ". Terminating optimization.");
        return true; // terminate to prevent further divergence
    }
    
    if (iter) {
      double gradient_max_norm = lopt.accum.getB().array().abs().maxCoeff();
      ROS_DEBUG_STREAM("[SF] optimize: gradient_max_norm=" << gradient_max_norm);
      if (gradient_max_norm < 1e-8)
        return true;
    }
    lopt.accum.setup_solver();
    Eigen::VectorXd Hdiag = lopt.accum.Hdiagonal();
    
    // Check Hessian diagonal for numerical issues
    double hdiag_max = Hdiag.maxCoeff();
    double hdiag_min = Hdiag.minCoeff();
    if (!std::isfinite(hdiag_max) || !std::isfinite(hdiag_min) || hdiag_max > 1e12 || hdiag_min < 1e-12) {
        ROS_ERROR_STREAM("[SF] optimize: Hessian diagonal has numerical issues. max=" << hdiag_max << " min=" << hdiag_min);
        return true; // terminate
    }
    
    bool stop = false;

    while (!stop) {
      Eigen::VectorXd Hdiag_lambda = Hdiag * lambda;
      for (int i = 0; i < Hdiag_lambda.size(); i++) {
        Hdiag_lambda[i] = std::max(Hdiag_lambda[i], 1e-18);
      }
      Eigen::VectorXd inc_full = -lopt.accum.solve(&Hdiag_lambda);

      // Check increment for numerical issues
      double inc_norm = inc_full.norm();
      if (!std::isfinite(inc_norm) || inc_norm > 100.0) { // 100 is a reasonable upper bound for increments
          ROS_ERROR_STREAM("[SF] optimize: Increment norm too large or not finite: " << inc_norm << ". Rejecting step.");
          lambda = std::min(100.0, lambda_vee * lambda);
          lambda_vee *= 2;
          if (abs(lambda - 100.0) < 1e-3) {
              stop = true;
          }
          continue;
      }

      Eigen::aligned_deque<Eigen::Vector3d> knots_trans_backup;
      Eigen::aligned_deque<Eigen::Quaterniond> knots_rot_backup;
      Eigen::aligned_deque<Eigen::Matrix<double, 6, 1>> knots_bias_backup;
      spline_local.getAllStateKnots(knots_trans_backup, knots_rot_backup,
                                    knots_bias_backup);
      CalibParam calib_param_backup = calib_param;
      applyIncrement(inc_full);
      ComputeErrorSplineOpt eopt(&spline_local, &calib_param, &param);
      
      if (!if_pose_only && !imu_window.empty())
        eopt(imu_window);
      if (!pose_window.empty())
        eopt(pose_window);
      
      // Check new cost for numerical issues
      if (!std::isfinite(eopt.error) || eopt.error > 1e12) {
          ROS_ERROR_STREAM("[SF] optimize: New cost is not finite or too large: " << eopt.error << ". Rejecting step.");
          lambda = std::min(100.0, lambda_vee * lambda);
          lambda_vee *= 2;
          spline_local.setAllKnots(knots_trans_backup, knots_rot_backup,
                                   knots_bias_backup);
          calib_param.setCalibParam(calib_param_backup);
          if (abs(lambda - 100.0) < 1e-3) {
              stop = true;
          }
          continue;
      }
      
      double f_diff = lopt.error - eopt.error;
      double l_diff = 0.5 * inc_full.dot(inc_full * lambda - lopt.accum.getB());
      double step_quality = f_diff / l_diff;
      ROS_INFO_STREAM("[SF] optimize: iter=" << iter << "  step_quality=" << step_quality << "  f_diff=" << f_diff << "  l_diff=" << l_diff << "  new_cost=" << eopt.error << "  lambda(before update)=" << lambda);
      if (step_quality < 0) {
        lambda = std::min(100.0, lambda_vee * lambda);
        ROS_INFO_STREAM("[SF] optimize: step rejected. Increasing lambda to " << lambda);
        if (abs(lambda - 100.0) < 1e-3) {
          stop = true;
        }
        lambda_vee *= 2;
        spline_local.setAllKnots(knots_trans_backup, knots_rot_backup,
                                 knots_bias_backup);
        calib_param.setCalibParam(calib_param_backup);
      } else {
        if (inc_full.norm() / ((double)spline_local.getNumKnots()) < 1e-10 ||
            abs(f_diff) / lopt.error < 1e-6) {
          stop = true;
        }
        lambda = std::max(
            1e-18, lambda * std::max(1.0 / 3,
                                     1 - std::pow(2 * step_quality - 1, 3.0)));
        ROS_INFO_STREAM("[SF] optimize: step accepted. New lambda=" << lambda);
        lambda_vee = 2;
        break;
      }
    }
    return stop;
  }

  void updateLinearizerSize() {
    int num_knots = spline_local.getNumKnots();
    bias_block_offset = Linearizer::POSE_SIZE * num_knots;
    hess_size = bias_block_offset;
    if (!if_pose_only) {
      hess_size += Linearizer::ACCEL_BIAS_SIZE * num_knots;
      hess_size += Linearizer::GYRO_BIAS_SIZE * num_knots;
    }
    gravity_block_offset = hess_size;
    if (param.if_opt_g) {
        hess_size += Linearizer::G_SIZE;
    }

    ROS_INFO_STREAM("[SF] updateLinearizerSize: num_knots="
                    << num_knots << "  pose_block=" << bias_block_offset
                    << "  gravity_offset=" << gravity_block_offset
                    << "  total_hess_size=" << hess_size);
  }

  void applyIncrement(Eigen::VectorXd &inc_full) {
    if (inc_full.size() == 0) {
      ROS_WARN("[SF] applyIncrement: Empty increment vector, skipping");
      return;
    }
    
    size_t num_knots = spline_local.getNumKnots();
    if (num_knots == 0) {
      ROS_WARN("[SF] applyIncrement: No knots in spline, skipping");
      return;
    }
    
    // Check if increment vector has enough elements for pose updates
    size_t required_pose_size = Linearizer::POSE_SIZE * num_knots;
    if (inc_full.size() < required_pose_size) {
      ROS_ERROR_STREAM("[SF] applyIncrement: Increment vector too small. Required: " 
                       << required_pose_size << ", got: " << inc_full.size());
      return;
    }
    
    for (size_t i = 0; i < num_knots; i++) {
      Eigen::Matrix<double, 6, 1> inc =
          inc_full.segment<Linearizer::POSE_SIZE>(Linearizer::POSE_SIZE * i);
      spline_local.applyPoseIncrement(i, inc);
    }
    spline_local.checkQuaternionControlPoints();
    if (!if_pose_only) {
      for (size_t i = 0; i < num_knots; i++) {
        Eigen::Matrix<double, 6, 1> inc =
            inc_full.segment<Linearizer::BIAS_SIZE>(bias_block_offset +
                                                    Linearizer::BIAS_SIZE * i);
        spline_local.applyBiasIncrement(i, inc);
      }
      spline_local.updateBiasIdleFirstWindow();

      if (param.if_opt_g) {
        if ((int)inc_full.size() >= gravity_block_offset + Linearizer::G_SIZE) {
          Eigen::VectorXd dg =
              inc_full.segment<Linearizer::G_SIZE>(gravity_block_offset);
          Eigen::Vector3d g0 = (calib_param.gravity +
                                Sphere::TangentBasis(calib_param.gravity) * dg)
                                   .normalized() *
                               9.80665;
          calib_param.gravity = g0;
        } else {
          ROS_WARN_STREAM_THROTTLE(1.0, "[SF] applyIncrement: inc_full.size() (" << inc_full.size() 
                             << ") is too small for gravity_block_offset (" << gravity_block_offset 
                             << ") + G_SIZE (" << Linearizer::G_SIZE 
                             << "). Skipping gravity update this iteration.");
        }
      }
    }
  }
};

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "sfuise");
  ROS_INFO("\033[1;32m---->\033[0m Starting SplineFusion.");
  ros::NodeHandle nh("~");
  SplineFusion estimator(nh);
  ros::Rate rate(1000);
  while (ros::ok()) {
    ros::spinOnce();
    estimator.run();
    rate.sleep();
  }
  return 0;
}
