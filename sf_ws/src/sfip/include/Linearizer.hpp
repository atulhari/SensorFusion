#pragma once

#include "utils/common_utils.hpp"
#include "utils/math_tools.hpp"
#include "Accumulator.hpp"
#include "SplineState.hpp"
#include "Residuals.hpp"

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

    Parameters() : w_pose_pos(1.0), w_pose_rot(1.0) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct Linearizer
{
    static const int POSE_SIZE = 6;
    static const int POS_SIZE = 3;
    static const int POS_OFFSET = 0;
    static const int ROT_SIZE = 3;
    static const int ROT_OFFSET = 3;
    static const int ACCEL_BIAS_SIZE = 3;
    static const int GYRO_BIAS_SIZE = 3;
    static const int BIAS_SIZE = ACCEL_BIAS_SIZE + GYRO_BIAS_SIZE; // Size of one bias knot (accel + gyro)
    static const int G_SIZE = 2;                                   // For gravity optimization (2 params for spherical coord?)

    static const int ACCEL_BIAS_OFFSET = 0;  // Within a BIAS_SIZE block
    static const int GYRO_BIAS_OFFSET = ACCEL_BIAS_SIZE; // Within a BIAS_SIZE block
    static const int G_OFFSET = 0; // Offset for gravity within its own block

    SparseHashAccumulator accum;
    double error;
    size_t bias_block_offset, gravity_block_offset, opt_size;

    SplineState* spline;
    const Parameters* param;

    const bool pose_fixed;

    Linearizer(size_t _bias_block_offset, size_t _gravity_block_offset, size_t _opt_size, SplineState* spl,
        const Parameters* par, const bool _pose_fixed)
        : bias_block_offset(_bias_block_offset), gravity_block_offset(_gravity_block_offset), opt_size(_opt_size),
          spline(spl), param(par), pose_fixed(_pose_fixed)
    {
        accum.reset(opt_size);
        error = 0;
    }

    ~Linearizer() {}

    void operator()(const Eigen::aligned_deque<ImuData>& r) {
        const size_t N = opt_size;                 // full size of state vector
        auto inside = [N](size_t start, size_t blk){ return start + blk <= N; };

        size_t set_fixed = 1;
        Eigen::Vector3d accel_var_inv = param->accel_var_inv;
        Eigen::Vector3d gyro_var_inv = param->gyro_var_inv;
        const double w_acc = param->w_acc;
        const double w_gyro = param->w_gyro;
        accel_var_inv *= w_acc;
        gyro_var_inv *= w_gyro;
        Eigen::Vector3d bias_accel_var_inv = param->bias_accel_var_inv;
        Eigen::Vector3d bias_gyro_var_inv = param->bias_gyro_var_inv;
        bias_accel_var_inv *= param->w_bias_acc;
        bias_gyro_var_inv *= param->w_bias_gyro;
        double num_imu = r.size();
        accel_var_inv /= num_imu;
        gyro_var_inv /= num_imu;
        bias_accel_var_inv /= (num_imu - 1);
        bias_gyro_var_inv /= (num_imu - 1);
        for (const auto& pm : r) {
            Jacobian36 J_accel;
            Jacobian33 J_gyro;
            Jacobian J_bias;
            Eigen::Matrix3d J_bias_a, J_bias_g;
            Eigen::Matrix<double, 3, 2> J_g;
            int64_t t = pm.timestampNanoseconds;
            Eigen::Matrix<double, 6, 1> residual;
            residual = Residuals::imuResidualJacobian(t, spline, &pm.accel, &pm.gyro, param->gravity,
                                                      &J_accel, &J_gyro, &J_bias, &J_g);
            const Eigen::Vector3d r_a = residual.segment<3>(3);
            error += r_a.transpose() * accel_var_inv.asDiagonal() * r_a;
            size_t start_g = gravity_block_offset;
            size_t num_Ji = J_accel.d_val_d_knot.size();
            for (size_t i = 0; i < num_Ji; i++) {
                size_t start_i = (J_accel.start_idx + i) * POSE_SIZE;
                if (pose_fixed && start_i < set_fixed * POSE_SIZE) {
                    continue;
                }
                for (size_t j = 0; j <= i; j++) {
                    size_t start_j = (J_accel.start_idx + j) * POSE_SIZE;
                    if (pose_fixed && start_j < set_fixed * POSE_SIZE) {
                        continue;
                    }
                    if (!inside(start_i, POSE_SIZE) || !inside(start_j, POSE_SIZE)) {
                        ROS_DEBUG_STREAM_THROTTLE(1.0,"[Linearizer] Skipped Jacobian/B-vector block out of range in IMU processing");
                        continue;
                    }
                    accum.addH<POSE_SIZE, POSE_SIZE>(start_i, start_j,
                    J_accel.d_val_d_knot[i].transpose() * accel_var_inv.asDiagonal() * J_accel.d_val_d_knot[j]);
                }
                for (size_t j = 0; j < num_Ji; j++) {
                    size_t start_bias_a = bias_block_offset + (J_bias.start_idx + j) * BIAS_SIZE + ACCEL_BIAS_OFFSET;
                    if (!inside(start_bias_a, ACCEL_BIAS_SIZE) || !inside(start_i, POSE_SIZE)) {
                        ROS_DEBUG_STREAM_THROTTLE(1.0,"[Linearizer] Skipped Jacobian/B-vector block out of range in IMU processing");
                        continue;
                    }
                    accum.addH<ACCEL_BIAS_SIZE, POSE_SIZE>(start_bias_a, start_i,
                        J_bias.d_val_d_knot[j] * accel_var_inv.asDiagonal() * J_accel.d_val_d_knot[i]);
                }
                if (param->if_opt_g) {
                    if (!inside(start_g, G_SIZE) || !inside(start_i, POSE_SIZE)) {
                        ROS_DEBUG_STREAM_THROTTLE(1.0,"[Linearizer] Skipped Jacobian/B-vector block out of range in IMU processing");
                    } else {
                        accum.addH<G_SIZE, POSE_SIZE>(start_g, start_i,
                            J_g.transpose() * accel_var_inv.asDiagonal() * J_accel.d_val_d_knot[i]);
                    }
                }
                if (!inside(start_i, POSE_SIZE)) {
                    ROS_DEBUG_STREAM_THROTTLE(1.0,"[Linearizer] Skipped Jacobian/B-vector block out of range in IMU processing");
                } else {
                    accum.addB<POSE_SIZE>(start_i, J_accel.d_val_d_knot[i].transpose() * accel_var_inv.asDiagonal() * r_a);
                }
            }
            size_t num_J_bias_knots = J_bias.d_val_d_knot.size();
            for (size_t i = 0; i < num_J_bias_knots; i++) {
                size_t start_bias_ai = bias_block_offset + (J_bias.start_idx + i) * BIAS_SIZE + ACCEL_BIAS_OFFSET;
                for (size_t j = 0; j <= i; j++) {
                    size_t start_bias_aj = bias_block_offset + (J_bias.start_idx + j) * BIAS_SIZE + ACCEL_BIAS_OFFSET;
                    if (!inside(start_bias_ai, ACCEL_BIAS_SIZE) || !inside(start_bias_aj, ACCEL_BIAS_SIZE)) {
                        ROS_DEBUG_STREAM_THROTTLE(1.0,"[Linearizer] Skipped Jacobian/B-vector block out of range in IMU processing");
                        continue;
                    }
                    Eigen::Matrix3d JT_w_J = J_bias.d_val_d_knot[i] * accel_var_inv.asDiagonal() * J_bias.d_val_d_knot[j];
                    accum.addH<ACCEL_BIAS_SIZE, ACCEL_BIAS_SIZE>(start_bias_ai, start_bias_aj, JT_w_J);
                }
                if (!inside(start_bias_ai, ACCEL_BIAS_SIZE)) {
                     ROS_DEBUG_STREAM_THROTTLE(1.0,"[Linearizer] Skipped Jacobian/B-vector block out of range in IMU processing");
                } else {
                    Eigen::Vector3d JT_w_r = J_bias.d_val_d_knot[i] * accel_var_inv.asDiagonal() * r_a;
                    accum.addB<ACCEL_BIAS_SIZE>(start_bias_ai, JT_w_r);
                }
                if (param->if_opt_g) {
                    if (!inside(start_g, G_SIZE) || !inside(start_bias_ai, ACCEL_BIAS_SIZE)) {
                        ROS_DEBUG_STREAM_THROTTLE(1.0,"[Linearizer] Skipped Jacobian/B-vector block out of range in IMU processing");
                    } else {
                         accum.addH<G_SIZE, ACCEL_BIAS_SIZE>(start_g, start_bias_ai, J_g.transpose() *
                                                                 accel_var_inv.asDiagonal() * J_bias.d_val_d_knot[i]);
                    }
                }
            }
            if (param->if_opt_g) {
                if (!inside(start_g, G_SIZE)) {
                    ROS_DEBUG_STREAM_THROTTLE(1.0,"[Linearizer] Skipped Jacobian/B-vector block out of range in IMU processing");
                } else {
                    accum.addH<G_SIZE, G_SIZE>(start_g, start_g, J_g.transpose() * accel_var_inv.asDiagonal() * J_g);
                    accum.addB<G_SIZE>(start_g, J_g.transpose() * accel_var_inv.asDiagonal() * r_a);
                }
            }
            const Eigen::Vector3d r_g = residual.head(3);
            error += r_g.transpose() * gyro_var_inv.asDiagonal() * r_g;
            size_t num_J_gyro_knots = J_gyro.d_val_d_knot.size();
            for (size_t i = 0; i < num_J_gyro_knots; i++) {
                size_t start_i_rot = (J_gyro.start_idx + i) * POSE_SIZE + ROT_OFFSET; 
                if (pose_fixed && ((J_gyro.start_idx + i) * POSE_SIZE) < set_fixed * POSE_SIZE) {
                    continue;
                }
                for (size_t j = 0; j <= i; j++) {
                    size_t start_j_rot = (J_gyro.start_idx + j) * POSE_SIZE + ROT_OFFSET;
                     if (pose_fixed && ((J_gyro.start_idx + j) * POSE_SIZE) < set_fixed * POSE_SIZE) {
                        continue;
                    }
                    if (!inside(start_i_rot, ROT_SIZE) || !inside(start_j_rot, ROT_SIZE)) {
                        ROS_DEBUG_STREAM_THROTTLE(1.0,"[Linearizer] Skipped Jacobian/B-vector block out of range in IMU processing");
                        continue;
                    }
                    accum.addH<ROT_SIZE, ROT_SIZE>(start_i_rot, start_j_rot, J_gyro.d_val_d_knot[i].transpose() *
                                                        gyro_var_inv.asDiagonal() * J_gyro.d_val_d_knot[j]);
                }
                for (size_t j = 0; j < num_J_bias_knots; j++) { 
                    size_t start_bias_g = bias_block_offset + (J_bias.start_idx + j) * BIAS_SIZE + GYRO_BIAS_OFFSET;
                    if (!inside(start_bias_g, GYRO_BIAS_SIZE) || !inside(start_i_rot, ROT_SIZE)) {
                        ROS_DEBUG_STREAM_THROTTLE(1.0,"[Linearizer] Skipped Jacobian/B-vector block out of range in IMU processing");
                        continue;
                    }
                    accum.addH<GYRO_BIAS_SIZE, ROT_SIZE>(start_bias_g, start_i_rot, J_bias.d_val_d_knot[j] *
                        gyro_var_inv.asDiagonal() * J_gyro.d_val_d_knot[i]);
                }
                if (!inside(start_i_rot, ROT_SIZE)) {
                    ROS_DEBUG_STREAM_THROTTLE(1.0,"[Linearizer] Skipped Jacobian/B-vector block out of range in IMU processing");
                } else {
                    accum.addB<ROT_SIZE>(start_i_rot, J_gyro.d_val_d_knot[i].transpose() * gyro_var_inv.asDiagonal() * r_g);
                }
            }
            for (size_t i = 0; i < num_J_bias_knots; i++) {
                size_t start_bias_gi = bias_block_offset + (J_bias.start_idx + i) * BIAS_SIZE + GYRO_BIAS_OFFSET;

                for (size_t j = 0; j <= i; j++) {
                    size_t start_bias_gj = bias_block_offset + (J_bias.start_idx + j) * BIAS_SIZE + GYRO_BIAS_OFFSET;
                    if (!inside(start_bias_gi, GYRO_BIAS_SIZE) || !inside(start_bias_gj, GYRO_BIAS_SIZE)) {
                        ROS_DEBUG_STREAM_THROTTLE(1.0,"[Linearizer] Skipped Jacobian/B-vector block out of range in IMU processing");
                        continue;
                    }
                    Eigen::Matrix3d JT_w_J = J_bias.d_val_d_knot[i] * gyro_var_inv.asDiagonal() * J_bias.d_val_d_knot[j];
                    accum.addH<GYRO_BIAS_SIZE, GYRO_BIAS_SIZE>(start_bias_gi, start_bias_gj, JT_w_J);
                }
                if (!inside(start_bias_gi, GYRO_BIAS_SIZE)) {
                    ROS_DEBUG_STREAM_THROTTLE(1.0,"[Linearizer] Skipped Jacobian/B-vector block out of range in IMU processing");
                } else {
                    Eigen::Vector3d JT_w_r = J_bias.d_val_d_knot[i] * gyro_var_inv.asDiagonal() * r_g;
                    accum.addB<GYRO_BIAS_SIZE>(start_bias_gi, JT_w_r);
                }
            }
        }
        Eigen::aligned_deque<ImuData>::const_iterator it = r.begin();
        Jacobian Jb0;
        Eigen::Matrix<double, 6, 1> b0 = spline->interpolateBias((*it).timestampNanoseconds, &Jb0);
        int64_t t0 = (*it).timestampNanoseconds;
        it++;
        size_t num_J0 = Jb0.d_val_d_knot.size();
        while (it != r.end()) {
            Jacobian Jb1;
            Eigen::Matrix<double, 6, 1> b1 = spline->interpolateBias((*it).timestampNanoseconds, &Jb1);
            int64_t t1 = (*it).timestampNanoseconds;
            size_t num_J1 = Jb1.d_val_d_knot.size();
            Eigen::Vector3d r_ba = b1.head<3>() - b0.head<3>();
            Eigen::Vector3d r_bg = b1.tail<3>() - b0.tail<3>();
            error += r_ba.transpose() * bias_accel_var_inv.asDiagonal() * r_ba;
            error += r_bg.transpose() * bias_gyro_var_inv.asDiagonal() * r_bg;
            size_t delta_idx = Jb1.start_idx - Jb0.start_idx;
            delta_idx = delta_idx > 4 ? 4 : delta_idx;
            size_t max_num_cp = std::max(num_J0, num_J1);
            Eigen::aligned_vector<std::pair<size_t, double>> vJb(max_num_cp + delta_idx);
            for (size_t i = 0; i < max_num_cp; i++) {
                bool set_idx = false;
                if (i < num_J0) {
                    vJb[i].first = Jb0.start_idx + i;
                    set_idx = true;
                    vJb[i].second = - Jb0.d_val_d_knot[i];
                }
                if (i >= delta_idx) {
                    if (!set_idx)
                        vJb[i].first = Jb1.start_idx + i - delta_idx;
                    vJb[i].second += Jb1.d_val_d_knot[i - delta_idx];
                }
            }
            for (size_t i = 0; i < delta_idx; i++) {
                vJb[i + max_num_cp].first = Jb1.start_idx + i + max_num_cp - delta_idx;
                vJb[i + max_num_cp].second = Jb1.d_val_d_knot[max_num_cp - delta_idx + i];
            }
            for (size_t i = 0; i < vJb.size(); i++) {
                size_t start_i_bias_block = bias_block_offset + vJb[i].first * BIAS_SIZE;
                size_t start_bias_ai = start_i_bias_block  + ACCEL_BIAS_OFFSET;
                size_t start_bias_gi = start_i_bias_block  + GYRO_BIAS_OFFSET;
                
                for (size_t j = 0; j <= i; j++) {
                    size_t start_j_bias_block = bias_block_offset + vJb[j].first * BIAS_SIZE;
                    size_t start_bias_aj = start_j_bias_block  + ACCEL_BIAS_OFFSET;
                    size_t start_bias_gj = start_j_bias_block  + GYRO_BIAS_OFFSET;
                    double JT_J = vJb[i].second * vJb[j].second;
                    Eigen::Matrix3d JT_wba_J = JT_J * bias_accel_var_inv.asDiagonal();
                    Eigen::Matrix3d JT_wbg_J = JT_J * bias_gyro_var_inv.asDiagonal();

                    if (!inside(start_bias_ai, ACCEL_BIAS_SIZE) || !inside(start_bias_aj, ACCEL_BIAS_SIZE)) {
                        ROS_DEBUG_STREAM_THROTTLE(1.0,"[Linearizer] Skipped Jacobian/B-vector block out of range in IMU processing (bias-diff loop)");
                    } else {
                       accum.addH<ACCEL_BIAS_SIZE, ACCEL_BIAS_SIZE>(start_bias_ai, start_bias_aj, JT_wba_J);
                    }

                    if (!inside(start_bias_gi, GYRO_BIAS_SIZE) || !inside(start_bias_gj, GYRO_BIAS_SIZE)) {
                        ROS_DEBUG_STREAM_THROTTLE(1.0,"[Linearizer] Skipped Jacobian/B-vector block out of range in IMU processing (bias-diff loop)");
                    } else {
                       accum.addH<GYRO_BIAS_SIZE, GYRO_BIAS_SIZE>(start_bias_gi, start_bias_gj, JT_wbg_J);
                    }
                }
                if (!inside(start_bias_ai, ACCEL_BIAS_SIZE)) {
                    ROS_DEBUG_STREAM_THROTTLE(1.0,"[Linearizer] Skipped Jacobian/B-vector block out of range in IMU processing (bias-diff loop)");
                } else {
                    accum.addB<ACCEL_BIAS_SIZE>(start_bias_ai, vJb[i].second * bias_accel_var_inv.asDiagonal() * r_ba);
                }

                if (!inside(start_bias_gi, GYRO_BIAS_SIZE)) {
                     ROS_DEBUG_STREAM_THROTTLE(1.0,"[Linearizer] Skipped Jacobian/B-vector block out of range in IMU processing (bias-diff loop)");
                } else {
                    accum.addB<GYRO_BIAS_SIZE>(start_bias_gi, vJb[i].second * bias_gyro_var_inv.asDiagonal() * r_bg);
                }
            }
            b0 = b1;
            Jb0 = Jb1;
            t0 = t1;
            num_J0 = num_J1;
            it++;
        }
    }

    void operator()(const Eigen::aligned_deque<PoseData> &poses) {
      if (poses.empty())
        return;

      /* ---------- build 6×6 information matrix (same window scaling as IMU)
       * ---- */
      const double inv_n = 1.0 / static_cast<double>(poses.size());
      const double wp = param->w_pose_pos * std::sqrt(inv_n);
      const double wq = param->w_pose_rot * std::sqrt(inv_n);

      Eigen::Matrix<double, 6, 6> W = Eigen::Matrix<double, 6, 6>::Zero();
      W.diagonal().head<3>().setConstant(wp * wp);
      W.diagonal().tail<3>().setConstant(wq * wq);

      constexpr size_t set_fixed = 1; // identical to IMU branch

      /* ---------- loop over all pose measurements
       * ------------------------------- */
      for (const PoseData &m : poses) {
        /* residual + Jacobian wrt spline knots */
        Jacobian66 Jpose; // filled inside
        Eigen::Matrix<double, 6, 1> r = Residuals::poseResidualJacobian(
            m.timestampNanoseconds, spline, m.position, m.orientation,
            &Jpose); // fills Jpose

        /* accumulate χ² error */
        error += r.transpose() * W * r;

        /* number of active control‑points that affect this timestamp */
        const size_t K = Jpose.d_val_d_knot.size();

        /* build H,b using exactly the same pattern as the IMU code  */
        for (size_t i = 0; i < K; ++i) {
          const size_t row = (Jpose.start_idx + i) * POSE_SIZE;
          if (row + POSE_SIZE > opt_size) // <-- NEW GUARD
            continue;
          if (pose_fixed && row < set_fixed * POSE_SIZE)
            continue;

          /* self‑term -------------------------------------------------------
           */
          const Eigen::Matrix<double, 6, 6> JT_W_Jii =
              Jpose.d_val_d_knot[i].transpose() * W * Jpose.d_val_d_knot[i];
          accum.addH<POSE_SIZE, POSE_SIZE>(row, row, JT_W_Jii);

          const Eigen::Matrix<double, 6, 1> JT_W_r_i =
              Jpose.d_val_d_knot[i].transpose() * W * r;
          accum.addB<POSE_SIZE>(row, JT_W_r_i);

          /* cross‑terms  j<i  ---------------------------------------------- */
          for (size_t j = 0; j < i; ++j) {
            const size_t col = (Jpose.start_idx + j) * POSE_SIZE;
            if (col + POSE_SIZE > opt_size) // <-- NEW GUARD
              continue;
            if (pose_fixed && col < set_fixed * POSE_SIZE)
              continue;

            const Eigen::Matrix<double, 6, 6> JT_W_Jij =
                Jpose.d_val_d_knot[i].transpose() * W * Jpose.d_val_d_knot[j];
            accum.addH<POSE_SIZE, POSE_SIZE>(row, col, JT_W_Jij);
          }
        }
      }
    }

     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ComputeErrorSplineOpt
{
    double error;
    SplineState* spline;
    CalibParam* calib_param_ptr;
    const Parameters* param;

    ComputeErrorSplineOpt(SplineState* spl, CalibParam* cal_param, const Parameters* par)
        : spline(spl), calib_param_ptr(cal_param), param(par)
    {
        error = 0;
    }

    ~ComputeErrorSplineOpt() {}

    void operator()(const Eigen::aligned_deque<ImuData>& r)
    {
        Eigen::Vector3d accel_var_inv = param->accel_var_inv;
        Eigen::Vector3d gyro_var_inv = param->gyro_var_inv;
        const double w_acc = param->w_acc;
        const double w_gyro = param->w_gyro;
        accel_var_inv *= w_acc;
        gyro_var_inv *= w_gyro;
        double num_imu = r.size();
        accel_var_inv /= num_imu;
        gyro_var_inv /= num_imu;
        Eigen::Vector3d bias_accel_var_inv = param->bias_accel_var_inv;
        Eigen::Vector3d bias_gyro_var_inv = param->bias_gyro_var_inv;
        bias_accel_var_inv *= param->w_bias_acc;
        bias_gyro_var_inv *= param->w_bias_gyro;
        bias_accel_var_inv /= (num_imu - 1);
        bias_gyro_var_inv /= (num_imu - 1);
        for (const auto& pm : r) {
            int64_t t = pm.timestampNanoseconds;
            Eigen::Matrix<double, 6, 1> residual = Residuals::imuResidual(t, spline, &pm.accel, &pm.gyro, calib_param_ptr->gravity);
            const Eigen::Vector3d r_a = residual.segment<3>(3);
            error += r_a.transpose() * accel_var_inv.asDiagonal() * r_a;
            const Eigen::Vector3d r_g = residual.head(3);
            error += r_g.transpose() * gyro_var_inv.asDiagonal() * r_g;
        }
        Eigen::aligned_deque<ImuData>::const_iterator it = r.begin();
        Eigen::Matrix<double, 6, 1> b0 = spline->interpolateBias((*it).timestampNanoseconds);
        it ++;
        while (it != r.end()) {
            Eigen::Matrix<double, 6, 1> b1 = spline->interpolateBias((*it).timestampNanoseconds);
            Eigen::Vector3d r_ba = b1.head<3>() - b0.head<3>();
            Eigen::Vector3d r_bg = b1.tail<3>() - b0.tail<3>();
            error += r_ba.transpose() * bias_accel_var_inv.asDiagonal() * r_ba;
            error += r_bg.transpose() * bias_gyro_var_inv.asDiagonal() * r_bg;
            b0 = b1;
            it++;
        }
    }

    void operator()(const Eigen::aligned_deque<PoseData> &poses) {
      if (poses.empty())
        return;

      const double inv_n = 1.0 / static_cast<double>(poses.size());
      const double wp = param->w_pose_pos * std::sqrt(inv_n);
      const double wq = param->w_pose_rot * std::sqrt(inv_n);

      Eigen::Matrix<double, 6, 6> W = Eigen::Matrix<double, 6, 6>::Zero();
      W.diagonal().head<3>().setConstant(wp * wp);
      W.diagonal().tail<3>().setConstant(wq * wq);

      for (const PoseData &m : poses) {
        Eigen::Matrix<double, 6, 1> r = Residuals::poseResidual(
            m.timestampNanoseconds, spline, m.position, m.orientation);

        error += r.transpose() * W * r;
      }
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
