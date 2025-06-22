# SFIP: Spline Fusion for Inertial and Pose Data

This package implements a sensor fusion solution to estimate the pose of a platform at a higher frequency by fusing low-frequency platform pose measurements with high-frequency IMU data.

## System Setup

*   **Platform**: The primary entity whose pose is to be estimated.
*   **IMU Sensor**: An Inertial Measurement Unit is mounted on the platform with a known or calibratable extrinsic transformation (translation and rotation) relative to the platform's reference frame.
*   **External Pose Source**: A system (e.g., motion capture, GPS-RTK, visual SLAM) providing sporadic but accurate pose measurements of the platform.

## Data Flow

*   **Inputs**:
    *   `geometry_msgs/PoseStamped`: Topic carrying low-frequency but accurate pose measurements of the platform in a global frame.
    *   `sensor_msgs/Imu`: Topic carrying high-frequency inertial measurements (angular velocity and linear acceleration) from the IMU in its own frame.
*   **Output**:
    *   `geometry_msgs/PoseStamped`: Topic publishing the high-frequency, fused pose estimate of the platform in the global frame.

## Core Methodology

*   **State Representation**: The SE(3) pose (position and orientation) of the platform over time is represented using B-splines operating on quaternions for orientation and 3D vectors for position.
*   **Estimation Approach**: A moving-window optimization framework is employed. This involves:
    *   Fixing the first knot to the pose, to take its absolute value.
    *   Accumulating IMU measurements and external pose measurements within a time window.
    *   Formulating a non-linear least-squares optimization problem to find the spline control points that best fit the measurements.
    *   Solving the optimization problem iteratively.
    *   Marginalizing or sliding the window as new data arrives to maintain real-time performance.
*   **Calibration**: The extrinsic transformation between the platform (`camera` is its frame id) and IMU is obtained from the frames using the `tf_static` message.
The pose is thhe pose of platform frame `camera` the pose and platform to imu tranform can be used to bring pose to the imu frame, the fusion is done in imu frame and the output pose is returned back to the camera frame. 

*   **Timestamping**: The fusion can work in pose only mode, so basically we dont need imu data. When Imu data is available, add prefiltering to make sure we dont push IMU samples older than last use pose, so Imu buffer is a bigger buffer but we keep removing as we filter out IMU data based on timestamp. We would add a subsampling or prefiltering based on the fps we are getting withoout downsample.

*   **Querrying Estimate**: The output is estimated from the spline at a target frequency (usually 60 Hz) and this needs to be maintained without drops, we don't run the optimization at this frequency this is just getting the fused pose from the optimized spline. The optimization itself will run at a lower frequency more than pose frequency of 20Hz.

