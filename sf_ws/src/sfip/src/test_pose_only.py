#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
import math
import time


def publish_test_data():
    """
    Publish test data for pose-only mode testing
    """
    rospy.init_node("pose_only_test")

    # Publisher for pose data
    pose_pub = rospy.Publisher("/kdlidar_ros/pose", PoseStamped, queue_size=10)

    # Static TF broadcaster for camera -> os_imu
    tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

    # Publish static transform
    tf_msg = TransformStamped()
    tf_msg.header.stamp = rospy.Time.now()
    tf_msg.header.frame_id = "camera"
    tf_msg.child_frame_id = "os_imu"
    tf_msg.transform.translation.x = 0.0
    tf_msg.transform.translation.y = 0.0
    tf_msg.transform.translation.z = 0.0
    tf_msg.transform.rotation.x = 0.0
    tf_msg.transform.rotation.y = 0.0
    tf_msg.transform.rotation.z = 0.0
    tf_msg.transform.rotation.w = 1.0
    tf_broadcaster.sendTransform(tf_msg)

    rospy.loginfo("Published static TF from camera to os_imu")

    # Wait for subscribers
    rospy.sleep(2.0)

    rate = rospy.Rate(20)  # 20 Hz pose data
    start_time = rospy.Time.now()
    count = 0

    while not rospy.is_shutdown() and count < 200:  # Publish for 10 seconds
        current_time = rospy.Time.now()
        dt = (current_time - start_time).to_sec()

        # Create a simple circular trajectory
        radius = 2.0
        omega = 0.5  # rad/s

        pose_msg = PoseStamped()
        pose_msg.header.stamp = current_time
        pose_msg.header.frame_id = "map"

        # Position in a circle
        pose_msg.pose.position.x = radius * math.cos(omega * dt)
        pose_msg.pose.position.y = radius * math.sin(omega * dt)
        pose_msg.pose.position.z = 1.0

        # Orientation facing the direction of motion
        yaw = omega * dt + math.pi / 2
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = math.sin(yaw / 2)
        pose_msg.pose.orientation.w = math.cos(yaw / 2)

        pose_pub.publish(pose_msg)

        if count % 20 == 0:
            rospy.loginfo(
                f"Published pose {count}: pos=[{pose_msg.pose.position.x:.2f}, {pose_msg.pose.position.y:.2f}, {pose_msg.pose.position.z:.2f}]"
            )

        count += 1
        rate.sleep()

    rospy.loginfo(f"Finished publishing {count} pose messages")


if __name__ == "__main__":
    try:
        publish_test_data()
    except rospy.ROSInterruptException:
        pass
