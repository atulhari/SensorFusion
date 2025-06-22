#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>
#include "../include/Residuals.hpp"
#include "../include/SplineState.hpp"
#include "../include/Linearizer.hpp"
#include "../include/utils/math_tools.hpp"

class ResidualTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize a simple spline for testing
        spline.init(1e8, 0, 0);  // 0.1 second intervals, starting at time 0
        
        // Add simple test knots with known states
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        Eigen::Matrix<double, 6, 1> bias = Eigen::Matrix<double, 6, 1>::Zero();
        
        for (int i = 0; i < 5; i++) {
            pos = Eigen::Vector3d(i * 0.1, 0, 0);  // Moving along x-axis
            spline.addSingleStateKnot(q, pos, bias);
        }
        
        gravity << 0, 0, -9.81;
    }
    
    SplineState spline;
    Eigen::Vector3d gravity;
};

// Test 1: Check if IMU residual computation is consistent
TEST_F(ResidualTest, IMUResidualConsistency) {
    int64_t test_time = 1e8;  // 0.1 seconds
    Eigen::Vector3d test_accel(0, 0, -9.81);  // Pure gravity
    Eigen::Vector3d test_gyro(0, 0, 0);       // No rotation
    
    // Compute residual without Jacobian
    Eigen::Matrix<double, 6, 1> residual_no_jac = 
        Residuals::imuResidual(test_time, &spline, &test_accel, &test_gyro, gravity);
    
    // Compute residual with Jacobian
    Jacobian36 J_accel;
    Jacobian33 J_gyro;
    Jacobian J_bias;
    Eigen::Matrix<double, 3, 2> J_gravity;
    
    Eigen::Matrix<double, 6, 1> residual_with_jac = 
        Residuals::imuResidualJacobian(test_time, &spline, &test_accel, &test_gyro, gravity,
                                       &J_accel, &J_gyro, &J_bias, &J_gravity);
    
    std::cout << "Residual without Jacobian: " << residual_no_jac.transpose() << std::endl;
    std::cout << "Residual with Jacobian: " << residual_with_jac.transpose() << std::endl;
    
    // They should be identical
    EXPECT_TRUE(residual_no_jac.isApprox(residual_with_jac, 1e-10))
        << "Residuals differ by: " << (residual_no_jac - residual_with_jac).norm();
}

// Test 2: Check Jacobian sizes and NaN/Inf values
TEST_F(ResidualTest, JacobianSanityCheck) {
    int64_t test_time = 1e8;
    Eigen::Vector3d test_accel(0, 0, -9.81);
    Eigen::Vector3d test_gyro(0, 0, 0);
    
    Jacobian36 J_accel;
    Jacobian33 J_gyro;
    Jacobian J_bias;
    Eigen::Matrix<double, 3, 2> J_gravity;
    
    Residuals::imuResidualJacobian(test_time, &spline, &test_accel, &test_gyro, gravity,
                                   &J_accel, &J_gyro, &J_bias, &J_gravity);
    
    std::cout << "J_accel size: " << J_accel.d_val_d_knot.size() << std::endl;
    std::cout << "J_gyro size: " << J_gyro.d_val_d_knot.size() << std::endl;
    std::cout << "J_bias size: " << J_bias.d_val_d_knot.size() << std::endl;
    
    // Check for NaN/Inf in Jacobians
    for (size_t i = 0; i < J_accel.d_val_d_knot.size(); i++) {
        EXPECT_TRUE(J_accel.d_val_d_knot[i].allFinite()) 
            << "J_accel[" << i << "] contains NaN/Inf: \n" << J_accel.d_val_d_knot[i];
        std::cout << "J_accel[" << i << "]:\n" << J_accel.d_val_d_knot[i] << std::endl;
    }
    
    for (size_t i = 0; i < J_gyro.d_val_d_knot.size(); i++) {
        EXPECT_TRUE(J_gyro.d_val_d_knot[i].allFinite()) 
            << "J_gyro[" << i << "] contains NaN/Inf: \n" << J_gyro.d_val_d_knot[i];
        std::cout << "J_gyro[" << i << "]:\n" << J_gyro.d_val_d_knot[i] << std::endl;
    }
    
    for (size_t i = 0; i < J_bias.d_val_d_knot.size(); i++) {
        EXPECT_TRUE(std::isfinite(J_bias.d_val_d_knot[i])) 
            << "J_bias[" << i << "] contains NaN/Inf: " << J_bias.d_val_d_knot[i];
        std::cout << "J_bias[" << i << "]: " << J_bias.d_val_d_knot[i] << std::endl;
    }
}

// Test 3: Test the problematic acceleration Jacobian computation
TEST_F(ResidualTest, AccelerationJacobianBug) {
    int64_t test_time = 1e8;
    Eigen::Vector3d test_accel(0, 0, -9.81);
    Eigen::Vector3d test_gyro(0, 0, 0);
    
    // Get the spline state at test time
    Eigen::Quaterniond q_interp;
    spline.interpolateQuaternion(test_time, &q_interp);
    Eigen::Vector3d accel_world = spline.interpolatePosition<2>(test_time) + gravity;
    
    std::cout << "Interpolated quaternion: w=" << q_interp.w() 
              << " x=" << q_interp.x() << " y=" << q_interp.y() << " z=" << q_interp.z() << std::endl;
    std::cout << "Acceleration in world frame: " << accel_world.transpose() << std::endl;
    
    Eigen::Matrix3d rot_world_to_body = q_interp.inverse().toRotationMatrix();
    std::cout << "Rotation matrix:\n" << rot_world_to_body << std::endl;
    
    Eigen::Vector3d accel_body = rot_world_to_body * accel_world;
    std::cout << "Expected acceleration in body frame: " << accel_body.transpose() << std::endl;
    std::cout << "Measured acceleration: " << test_accel.transpose() << std::endl;
    std::cout << "Raw acceleration residual: " << (accel_body - test_accel).transpose() << std::endl;
}

// Test 4: Check weight and variance computations in Linearizer
TEST_F(ResidualTest, WeightComputationTest) {
    Parameters param;
    param.accel_var_inv << 1000.0, 1000.0, 1000.0;
    param.w_acc = 1.0;
    
    // Simulate what happens in Linearizer
    Eigen::Vector3d accel_var_inv = param.accel_var_inv;
    accel_var_inv *= param.w_acc;  // Should still be [1000, 1000, 1000]
    accel_var_inv = accel_var_inv.cwiseProduct(accel_var_inv);  // Now [1e6, 1e6, 1e6]
    
    double num_imu = 10.0;
    accel_var_inv /= num_imu;  // Now [1e5, 1e5, 1e5]
    
    std::cout << "Final accel_var_inv: " << accel_var_inv.transpose() << std::endl;
    
    // Test a residual
    Eigen::Vector3d test_residual(0.01, 0.01, 0.01);  // 1cm residual
    double cost = test_residual.transpose() * accel_var_inv.asDiagonal() * test_residual;
    std::cout << "Cost for 1cm residual: " << cost << std::endl;
    
    // This should be reasonable, not 1e89!
    EXPECT_LT(cost, 1e6) << "Cost computation produces unreasonably large values";
}

// Test 5: Matrix dimension consistency
TEST_F(ResidualTest, MatrixDimensionTest) {
    int64_t test_time = 1e8;
    Eigen::Vector3d test_accel(0, 0, -9.81);
    Eigen::Vector3d test_gyro(0, 0, 0);
    
    Jacobian36 J_accel;
    Jacobian J_bias;
    
    Residuals::imuResidualJacobian(test_time, &spline, &test_accel, &test_gyro, gravity,
                                   &J_accel, nullptr, &J_bias, nullptr);
    
    // Check that J_accel matrices are properly sized (3x6)
    for (size_t i = 0; i < J_accel.d_val_d_knot.size(); i++) {
        EXPECT_EQ(J_accel.d_val_d_knot[i].rows(), 3);
        EXPECT_EQ(J_accel.d_val_d_knot[i].cols(), 6);
    }
    
    // Check the problematic line from Residuals.hpp:
    // Jacc->d_val_d_knot[i].template topLeftCorner<3, 3>() =
    //     (rotWorldToBody * Eigen::Matrix3d::Identity()) * JlineAcc.d_val_d_knot[i];
    
    // This is WRONG! JlineAcc.d_val_d_knot[i] is a scalar, not a 3x3 matrix!
    Jacobian JlineAcc;
    spline.interpolatePosition<2>(test_time, &JlineAcc);
    
    for (size_t i = 0; i < JlineAcc.d_val_d_knot.size(); i++) {
        std::cout << "JlineAcc[" << i << "] scalar: " << JlineAcc.d_val_d_knot[i] << std::endl;
    }
}

TEST_F(ResidualTest, JacobianDimensionBug) {
    int64_t test_time = 1e8;
    Eigen::Vector3d test_accel(0, 0, -9.81);
    Eigen::Vector3d test_gyro(0, 0, 0);
    
    Jacobian JlineAcc;
    spline.interpolatePosition<2>(test_time, &JlineAcc);
    
    for (size_t i = 0; i < JlineAcc.d_val_d_knot.size(); i++) {
        std::cout << "JlineAcc[" << i << "] scalar: " << JlineAcc.d_val_d_knot[i] << std::endl;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 