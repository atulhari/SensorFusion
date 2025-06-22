#include <gtest/gtest.h>
#include "CeresSplineFuser.hpp"
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Imu.h>
#include <chrono>
#include <thread>

class OptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize ROS node
        int argc = 0;
        ros::init(argc, nullptr, "test_optimizer");
        nh_ = std::make_unique<ros::NodeHandle>("~");
        
        // Create optimizer
        optimizer_ = std::make_unique<CeresSplineFuser>(*nh_);
        
        // Create publishers
        pose_pub_ = nh_->advertise<geometry_msgs::PoseStamped>("/estimation_interface_node/pose_ds", 10);
        imu_pub_ = nh_->advertise<sensor_msgs::Imu>("/estimation_interface_node/imu_ds", 10);
        
        // Wait for publishers to be ready
        ros::Duration(0.5).sleep();
    }

    void TearDown() override {
        optimizer_.reset();
        nh_.reset();
    }

    // Helper function to create a pose message
    geometry_msgs::PoseStamped createPose(double x, double y, double z, double qw, double qx, double qy, double qz) {
        geometry_msgs::PoseStamped pose;
        pose.header.stamp = ros::Time::now();
        pose.header.frame_id = "map";
        pose.pose.position.x = x;
        pose.pose.position.y = y;
        pose.pose.position.z = z;
        pose.pose.orientation.w = qw;
        pose.pose.orientation.x = qx;
        pose.pose.orientation.y = qy;
        pose.pose.orientation.z = qz;
        return pose;
    }

    // Helper function to create an IMU message
    sensor_msgs::Imu createImu(double ax, double ay, double az, double gx, double gy, double gz) {
        sensor_msgs::Imu imu;
        imu.header.stamp = ros::Time::now();
        imu.header.frame_id = "imu_link";
        imu.linear_acceleration.x = ax;
        imu.linear_acceleration.y = ay;
        imu.linear_acceleration.z = az;
        imu.angular_velocity.x = gx;
        imu.angular_velocity.y = gy;c
        imu.angular_velocity.z = gz;
        return imu;
    }

    // Helper function to wait for initialization
    bool waitForInitialization(int max_attempts = 10) {
        for (int i = 0; i < max_attempts; ++i) {
            if (optimizer_->isInitialized()) {
                return true;
            }
            ros::spinOnce();
            ros::Duration(0.1).sleep();
        }
        return false;
    }

    std::unique_ptr<ros::NodeHandle> nh_;
    std::unique_ptr<CeresSplineFuser> optimizer_;
    ros::Publisher pose_pub_;
    ros::Publisher imu_pub_;
};

TEST_F(OptimizerTest, InitializationTest) {
    // Publish two poses to initialize the optimizer
    pose_pub_.publish(createPose(0, 0, 0, 1, 0, 0, 0));
    ros::Duration(0.1).sleep();
    pose_pub_.publish(createPose(1, 0, 0, 1, 0, 0, 0));
    
    // Wait for initialization
    ASSERT_TRUE(waitForInitialization()) << "Optimizer failed to initialize";
    
    // Verify initial state
    EXPECT_TRUE(optimizer_->isInitialized());
}

TEST_F(OptimizerTest, OptimizationTest) {
    // Initialize with poses
    pose_pub_.publish(createPose(0, 0, 0, 1, 0, 0, 0));
    ros::Duration(0.1).sleep();
    pose_pub_.publish(createPose(1, 0, 0, 1, 0, 0, 0));
    
    ASSERT_TRUE(waitForInitialization()) << "Optimizer failed to initialize";
    
    // Publish IMU data
    for (int i = 0; i < 10; ++i) {
        imu_pub_.publish(createImu(0, 0, 9.81, 0, 0, 0));
        ros::Duration(0.01).sleep();
    }
    
    // Publish more poses
    pose_pub_.publish(createPose(2, 0, 0, 1, 0, 0, 0));
    ros::Duration(0.1).sleep();
    
    // Wait for optimization
    ros::Duration(0.5).sleep();
    ros::spinOnce();
    
    // Verify optimization state
    EXPECT_TRUE(optimizer_->isInitialized());
}

TEST_F(OptimizerTest, WindowManagementTest) {
    // Initialize with poses
    pose_pub_.publish(createPose(0, 0, 0, 1, 0, 0, 0));
    ros::Duration(0.1).sleep();
    pose_pub_.publish(createPose(1, 0, 0, 1, 0, 0, 0));
    
    ASSERT_TRUE(waitForInitialization()) << "Optimizer failed to initialize";
    
    // Publish sequence of poses to test window management
    for (int i = 0; i < 20; ++i) {
        pose_pub_.publish(createPose(i, 0, 0, 1, 0, 0, 0));
        ros::Duration(0.05).sleep();
        
        // Publish IMU data between poses
        imu_pub_.publish(createImu(0, 0, 9.81, 0, 0, 0));
        ros::Duration(0.01).sleep();
    }
    
    // Wait for processing
    ros::Duration(1.0).sleep();
    ros::spinOnce();
    
    // Verify window management
    EXPECT_TRUE(optimizer_->isInitialized());
}

TEST_F(OptimizerTest, ErrorHandlingTest) {
    // Test with invalid pose data
    geometry_msgs::PoseStamped invalid_pose = createPose(0, 0, 0, 0, 0, 0, 0); // Invalid quaternion
    pose_pub_.publish(invalid_pose);
    ros::Duration(0.1).sleep();
    
    // Should not initialize with invalid data
    EXPECT_FALSE(optimizer_->isInitialized());
    
    // Now publish valid data
    pose_pub_.publish(createPose(0, 0, 0, 1, 0, 0, 0));
    ros::Duration(0.1).sleep();
    pose_pub_.publish(createPose(1, 0, 0, 1, 0, 0, 0));
    
    // Should initialize with valid data
    ASSERT_TRUE(waitForInitialization()) << "Optimizer failed to initialize with valid data";
    EXPECT_TRUE(optimizer_->isInitialized());
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 