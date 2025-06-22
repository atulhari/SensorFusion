#!/usr/bin/env python3
# file: fake_pose_pub.py
#
# Publish synthetic PoseStamped messages on /kdlidar_ros/pose
# and drive /clock so that use_sim_time behaves correctly.

import time
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler
from rosgraph_msgs.msg import Clock


def make_pose_msg(sim_t, cfg):
    """
    Build a geometry_msgs/PoseStamped at simulated time sim_t (rospy.Time).
    """
    # seconds since start as float
    t = sim_t.to_sec()
    # -------- trajectory position ------------------------------------
    if cfg["traj_type"] == "parabola":
        x = cfg["vel"] * t
        z = cfg["a"] * x**2 + cfg["z0"]
    else:  # 'line'
        x = cfg["vel"] * t
        z = cfg["z0"]

    y = cfg["y0"]
    # -------- orientation  (yaw-only, facing +X) ---------------------
    q = quaternion_from_euler(0.0, 0.0, 0.0)  # roll, pitch, yaw

    msg = PoseStamped()
    msg.header.stamp = sim_t
    msg.header.frame_id = cfg["frame"]

    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z

    msg.pose.orientation = Quaternion(*q)
    return msg


def main():
    rospy.init_node("fake_pose_publisher")

    # ─── parameters ────────────────────────────────────────────────
    cfg = {
        "hz": rospy.get_param("~rate_hz", 20.0),
        "vel": rospy.get_param("~vel_m_s", 1.0),
        "traj_type": rospy.get_param("~traj_type", "line"),  # 'line' | 'parabola'
        "a": rospy.get_param("~parabola_coeff", 0.02),  # z = a·x²
        "z0": rospy.get_param("~z0", 0.0),
        "y0": rospy.get_param("~y0", 0.0),
        "frame": rospy.get_param("~frame_id", "map"),
        "duration": rospy.get_param(
            "~duration_s", 30.0
        ),  # stop after this (≤0 to ignore)
        "num_poses": rospy.get_param("~max_poses", 0),  # stop after N msgs (0 = ignore)
    }

    assert cfg["traj_type"] in (
        "line",
        "parabola",
    ), "traj_type must be 'line' or 'parabola'"

    pose_pub = rospy.Publisher("/kdlidar_ros/pose", PoseStamped, queue_size=10)
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)

    rate = rospy.Rate(cfg["hz"])
    start_wall = time.time()
    count = 0

    rospy.loginfo(
        "fake_pose_pub: publishing %s trajectory at %.1f Hz",
        cfg["traj_type"],
        cfg["hz"],
    )
    while not rospy.is_shutdown():
        # real elapsed
        wall_elapsed = time.time() - start_wall
        # simulated time since start
        sim_t = rospy.Time.from_sec(wall_elapsed)

        # publish clock for use_sim_time
        clock_pub.publish(Clock(sim_t))

        # build & publish pose stamped at sim_t
        msg = make_pose_msg(sim_t, cfg)
        pose_pub.publish(msg)

        count += 1
        if (cfg["duration"] > 0 and wall_elapsed >= cfg["duration"]) or (
            cfg["num_poses"] > 0 and count >= cfg["num_poses"]
        ):
            rospy.loginfo(
                "fake_pose_pub: finished after %.2f s, %d poses", wall_elapsed, count
            )
            break

        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
