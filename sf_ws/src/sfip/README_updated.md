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
    The pose is the pose of platform frame `camera`; the pose and platform-to-IMU transform can be used to bring the pose to the IMU frame. The fusion is done in the IMU frame and the output pose is returned back to the camera frame.

## Library Architecture

The SFIP package has been refactored to separate the core algorithm from ROS dependencies. The library now has the following structure:

```
SplineFusionCore           // Main class that orchestrates the fusion process
  |
  ├── Types                // Common data types and structures
  |
  ├── WindowManager        // Manages the sliding window of control points
  |
  ├── Optimizer            // Handles optimization of the spline
  |
  ├── DataProcessor        // Prepares data and performs IMU integration
  |
  └── SplineState          // Manages the B-spline representation
```

### ROS Integration

To maintain compatibility with ROS, an adapter layer has been created:

```
SplineFusionAdapter        // Adapts the core library to ROS
  |
  └── SplineFusionCore     // Uses the core library
```

## Building and Running

### Prerequisites

* ROS (Melodic or newer)
* Eigen3
* SuiteSparse

### Building

```bash
# Clone the repository
git clone https://github.com/yourusername/SensorFusion.git
cd SensorFusion

# Build the package
catkin_make
```

### Running

To run the node:

```bash
# Source the workspace
source devel/setup.bash

# Launch the node
roslaunch sfip sfip.launch
```

## Configuration Parameters

The behavior of the fusion algorithm can be customized through the following parameters:

*   **`control_point_fps`**: The frequency of the control points in the spline (default: 20Hz).
*   **`window_size`**: The number of control points in the optimization window (default: 10).
*   **`imu_sample_coeff`**: Coefficient for IMU downsampling. Set to 0 for pose-only mode (default: 1.0).
*   **`pose_sample_coeff`**: Coefficient for pose downsampling (default: 1.0).
*   **`platform_frame_id`**: Frame ID of the platform (default: "camera_link").
*   **`imu_frame`**: Frame ID of the IMU (default: "imu_link").
*   **`output_frame`**: Frame ID for the output poses (default: "map").

## Additional Information

For more detailed information about the implementation and theory, please refer to:

* [REFACTORING.md](REFACTORING.md): Details about the refactoring of the codebase.

## License

This project is licensed under the GPLv3 License - see the LICENSE file for details.
