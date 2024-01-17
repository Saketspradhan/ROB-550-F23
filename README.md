# Robotics Systems Lab (ROB 550)| University of Michigan 

## Armlab

### Overview
This project involved developing a robotic pick-and-place system using a 5-Degree-of-Freedom (DOF) robotic arm, computer vision, and motion planning techniques. The goal was to pick, sort, stack, and arrange colored blocks on a workspace using the robotic arm, guided by computer vision.

### Key components of the project:

- Camera calibration to determine intrinsic parameters and extrinsic transformation between camera and robot frames
- Modeling robotic arm kinematics to map end-effector poses to joint angles
- Computer vision techniques to segment and classify blocks, providing 3D position and orientation data
- Motion planning to generate pick-and-place actions accounting for block positions, orientations, and dimensions

The system demonstrated the ability to carry out complex block sorting, stacking, and arrangement tasks.

### Methodology
The project involved the following key steps:

- Manual and automatic camera calibration using a checkerboard and AprilTags
- Mapping between image pixels and 3D world coordinates
- Modeling forward and inverse kinematics using DH parameters
- Detecting blocks using depth images and thresholding color images
- Offsetting inverse kinematics to compensate for joint sag
- Generating waypoints for pick-and-place motions
- Visual servoing to precisely pick and place blocks

### Results
The system succeeded in all competition tasks, including:

- Sorting blocks by size
- Stacking blocks on target locations
- Arranging blocks in specified patterns
- Constructing an 18-block tower (current Robotics department record holder for the most blocks stacked vertically)

The integrated calibration, perception, planning, and control framework enabled robust performance across diverse manipulation tasks.

### Discussion
Potential improvements include:

- Enhanced calibration for higher accuracy
- Alternative computer vision techniques
- More DOFs for dexterous manipulation
- Smoother trajectories and motion optimization

Overall, the project demonstrates effective robotic pick-and-place guided by computer vision. The versatility makes it a promising platform for automation applications.

## Botlab

### Overview
This project involved developing software for MBot, a two-wheeled differential drive robot, to enable autonomous mapping, localization, path planning, and navigation. 

### Key components:

- Motor control through PID tuning and odometry estimation
- Particle filter and mapping algorithms for SLAM-based localization
- A* path planning for optimal path generation
- Exploration logic to traverse and map unknown environments
- A forklift mechanism to move and stack boxes

The integrated system was evaluated in a competition across 4 events requiring autonomous traversal, mapping, and item manipulation.

### Methodology
The project involved the following key steps:

- Calibrating and tuning PID control for the wheel motors
- Estimating odometry from encoders and IMU
- Implementing particle filter for localization with sensor model and motion model
- Building occupancy grid maps from LIDAR scans
- Path planning with A* search and obstacle avoidance
- Frontier-based exploration to map unknown environments
- Designing a 3D-printed forklift with threaded screws for lifting crates

### Results
The implemented system demonstrated:

- Precise motor control within 0.02 m/s of commands
- Low odometry errors (~0.1 m) over short paths
- Accurate mapping and localization with SLAM
- Fast optimal path planning under 100 ms
- Successful exploration and mapping
- Effective (though manual) lifting and stacking with the forklift

The robot succeeded in all competition events, including exploration, traversal, and object manipulation.

### Discussion
The PID control, path planning, and mapping components were robust. Potential improvements include:

- Smoother trajectories with different controllers or pruning
- More accurate pose estimation by fusing encoders and IMU
- Autonomous control for the forklift mechanism
- Incorporating additional sensors for enhanced perception

Overall, the project demonstrated a capable autonomous mobile robot system using SLAM, path planning, and control techniques. Further work can build on this foundation to enable more complex behaviors.
