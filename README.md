# Robotics Systems Lab | University of Michigan 

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
