"""!
Implements the RXArm class.

The RXArm class contains:

* last feedback from joints
* functions to command the joints
* functions to get feedback from joints
* functions to do FK and IK
* A function to read the RXArm config file

You will upgrade some functions and also implement others according to the comments given in the code.
"""
import numpy as np
from functools import partial
from kinematics import FK_dh, FK_pox, getPoseFromT, IK_geometric, getPossiblePose, getDownPose, clamp
from constants import *
import time
import csv
import sys, os

from builtins import super
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from resource.config_parse import *
from sensor_msgs.msg import JointState
import rclpy

sys.path.append('../../interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/interbotix_xs_modules/xs_robot')
from arm import InterbotixManipulatorXS
from mr_descriptions import ModernRoboticsDescription as mrd


def _ensure_initialized(func):
    """!
    @brief      Decorator to skip the function if the RXArm is not initialized.

    @param      func  The function to wrap

    @return     The wraped function
    """
    def func_out(self, *args, **kwargs):
        if self.initialized:
            return func(self, *args, **kwargs)
        else:
            print('WARNING: Trying to use the RXArm before initialized')

    return func_out


class RXArm(InterbotixManipulatorXS):
    """!
    @brief      This class describes a RXArm wrapper class for the rx200
    """

    GRIPPER_OPEN = True
    GRIPPER_CLOSED = False

    def __init__(self, dhConfigFile=None):
        """!
        @brief      Constructs a new instance.

                    Starts the RXArm run thread but does not initialize the Joints. Call RXArm.initialize to initialize the
                    Joints.

        @param      dhConfigFile  The configuration file that defines the DH parameters for the robot
        """
        super().__init__(robot_model="rx200")
        self.joint_names = self.arm.group_info.joint_names
        self.num_joints = 5
        # Gripper
        self.gripper_state = True
        self.gripper_val = 0
        # State
        self.initialized = False
        self.was45 = False
        # Cmd
        self.position_cmd = None
        self.moving_time = 2.0
        self.accel_time = 0.5
        # Feedback
        self.position_fb = None
        self.velocity_fb = None
        self.effort_fb = None
        # DH Params
        self.dh_params = []
        self.dhConfigFile = dhConfigFile
        if (dhConfigFile is not None):
            self.dh_params = RXArm.parseDhParamFile(dhConfigFile)
        #POX params
        self.M_matrix = []
        self.S_list = []


    def initialize(self):
        """!
        @brief      Initializes the RXArm from given configuration file.

                    Initializes the Joints and serial port

        @return     True is succes False otherwise
        """
        self.initialized = False
        # Wait for other threads to finish with the RXArm instead of locking every single call
        time.sleep(0.25)
        """ Commanded Values """
        self.position = [0.0] * self.num_joints  # radians
        """ Feedback Values """
        self.position_fb = [0.0] * self.num_joints  # radians
        self.velocity_fb = [0.0] * self.num_joints  # 0 to 1 ???
        self.effort_fb = [0.0] * self.num_joints  # -1 to 1

        # Reset estop and initialized
        self.estop = False
        self.enable_torque()
        self.moving_time = 2.0
        self.accel_time = 0.5
        self.arm.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=False)
        self.gripper.release()
        self.initialized = True
        return self.initialized


    def sleep(self):
        self.moving_time = 2.0
        self.accel_time = 1.0
        self.arm.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=True)
        self.arm.go_to_sleep_pose(moving_time=self.moving_time,
                              accel_time=self.accel_time,
                              blocking=False)
        self.initialized = False
    

    def initWait(self):
        """
        Initializes, then waits some amount of time
        """
        self.initialize()
        time.sleep(2)
    

    def sleepWait(self):
        """
        Sleeps, then waits some amount of time
        """
        self.sleep()
        time.sleep(2)

    
    def IK_geometric(self, pose):
        """
        Gets the joint angles given a pose
        
        Params:
            pose    A pose vector
        
        Returns:
            joint_angles    The joint angles returned from the IK
        """
        return IK_geometric(self.dh_params, pose)


    def set_positions(self, joint_positions, moving_time=None, accel_time=None, blocking=False):
        """!
         @brief      Sets the positions.

         @param      joint_angles  The joint angles
         """
        if moving_time == None:
            moving_time = self.moving_time
        if accel_time == None:
            accel_time = self.accel_time
        
        if joint_positions is None:
            print('Tried to go to invalid position')
            return
        
        print('Going to joint positions', np.array2string(joint_positions, precision=2, suppress_small=True))

        self.arm.set_joint_positions(joint_positions,
                                 moving_time=moving_time,
                                 accel_time=accel_time,
                                 blocking=blocking)
    
    
    # Don't use this! Use smooth_set_pose instead
    def set_pose(self, pose, moving_time=None, accel_time=None, blocking=False):
        if moving_time == None:
            moving_time = self.moving_time
        if accel_time == None:
            accel_time = self.accel_time
        
        # Reassign yaw if necessary
        if len(pose) == 6:
            pose[5] = np.arctan2(-pose[0], pose[1])
        
        joint_positions = self.IK_geometric(pose)
        if joint_positions is None:
            print(f'Tried to go to invalid position at {np.array2string(pose, precision=2, suppress_small=True)}')
            return
        print('Going to', np.array2string(pose, precision=1, suppress_small=True))
        self.set_positions(joint_positions, moving_time, accel_time, blocking)
    
    
    def smooth_set_pose(self, pose, time_params=None):
        """
        Adaptively sets the times and puts waypoints in the middle of a trajectory to smoothly set the pose
        
        Params:
            pose            A pose vector
            time_params     Either a number for moving_time, or a tuple for (accel_time, moving_time)
        """
        self.was45 = False
        if pose is None:
            time.sleep(0.5)
            return
        
        pose = pose.copy()

        # TWEAK: Manual correction of IK
        r = np.linalg.norm(np.array([pose[0], pose[1]]))
        theta = np.arctan2(-pose[0], pose[1])
        pose[2] += .04 * r
        r -= 7
        pose[0] = r * -np.sin(theta)
        pose[1] = r * np.cos(theta)
        
        joint_positions = self.IK_geometric(pose)
        
        if joint_positions is None:
            oldPose = pose.copy()
            pose = self.getPossiblePose(pose.copy())
            if pose[3] == 3*np.pi/4:
                pose[2] = oldPose[2]
                self.was45 = True
                pose[4] = 0.0
            joint_positions = self.IK_geometric(pose)
            if joint_positions is None:
                print(f'Tried to go to invalid position at {np.array2string(pose, precision=2, suppress_small=True)}')
                return
        
        print('Going to', np.array2string(pose, precision=1, suppress_small=True))
        
        self.smooth_set_positions(joint_positions, time_params=time_params)
    
    
    def smooth_set_positions(self, joint_positions, time_params=None):
        """
        Adaptively sets the times and puts waypoints in the middle of a trajectory to smoothly set the positions
        
        Params:
            joint_positions An array of joint angles
            time_params     Either a number for moving_time, or a tuple for (accel_time, moving_time)
        """
        
        if time_params is not None:
            if isinstance(time_params, int) or isinstance(time_params, float):
                self.moving_time = time_params
                self.accel_time = 0.5
            else:
                self.moving_time, self.accel_time = time_params

        # Calculate moving time based on traveled distance
        else:
            joint_diff = np.abs(clamp(self.get_positions()[0]) - clamp(joint_positions[0]))
            if joint_diff > 2:
                self.moving_time = 2
                self.accel_time = 0.5
            elif joint_diff >= 0.1:
                self.moving_time = 1
                self.accel_time = 0.4
            else:
                self.moving_time = 0.7
                self.accel_time = 0.2
            print('Joint diff:', joint_diff)

        joint_positions[4] = clamp(joint_positions[4])
        if joint_positions[4] > 0:
            joint_positions[4] -= np.pi

        print('Going to joint positions', np.array2string(joint_positions, precision=2, suppress_small=True))
        print('Move time:', self.moving_time, ', Accel time:', self.accel_time)

        worked = self.arm.set_joint_positions(joint_positions,
                                 moving_time=self.moving_time,
                                 accel_time=self.accel_time,
                                 blocking=False)
        print('Worked:', worked)
        if not worked:
            # self.moving_time = 2.0
            # self.accel_time = 0.5
            # # self.smooth_set_pose(getDownPose(0, 175, 100))
            # self.arm.go_to_home_pose(moving_time=2,
            #                  accel_time=0.5,
            #                  blocking=False)
            # time.sleep(2.1)

            # Try again halfway
            halfway = (self.get_positions() + joint_positions) / 2
            self.smooth_set_positions(halfway, time_params=time_params)
            self.smooth_set_positions(joint_positions, time_params=time_params)
            return
        time.sleep(self.moving_time+0.05)
        
        # Reset acceleration and moving
        self.set_accel_time(0.5)
        self.set_moving_time(2)
    
    
    def interpCartesian(self, startPose, endPose, n=5, time_params=None):
        """
        Moves the arm in a trajectory interpolating in configuration space
        
        Params:
            startPose       The starting pose
            endPose         The ending pose
            n               The number of waypoints to interpolate with
            time_params     Either a number for moving_time, or a tuple for (accel_time, moving_time)
        """
        for t in np.linspace(0, 1, n+1):
            newPose = startPose + t*(endPose-startPose)
            self.smooth_set_pose(newPose, time_params=time_params)
    
    
    def interpJoints(self, startJoints, endJoints, n=5, time_params=None):
        """
        Moves the arm interpolating in joint space
        
        Params:
            startJoints       The starting joint angles
            endJoints         The ending joint angles
            n                 The number of waypoints to interpolate with
            time_params       Either a number for moving_time, or a tuple for (accel_time, moving_time)
        """
        for t in np.linspace(0, 1, n+1):
            newJoints = startJoints + t*(endJoints-startJoints)
            self.smooth_set_positions(newJoints, time_params=time_params)
    

    def getPossiblePose(self, pose):
        """
        A wrapper that tries to convert an unreachable pose to a close reachable one
        
        Params:
            pose       A pose vector
        
        Returns:
            newPose    The possible pose vector
        """
        return getPossiblePose(self.dh_params, pose)
    
    
    def set_gripper(self, gripper_state):
        """
        Sets the gripper state
        
        Params:
            gripper_state   A boolean where GRIPPER_OPEN=True and GRIPER_CLOSED=False
        """
        if gripper_state == self.GRIPPER_OPEN:
            self.gripper.release()
        elif gripper_state == self.GRIPPER_CLOSED:
            self.gripper.grasp()
        else:
            print('Gripper error')
            raise ValueError('Gripper state must be boolean')


    def set_moving_time(self, moving_time):
        self.moving_time = moving_time


    def set_accel_time(self, accel_time):
        self.accel_time = accel_time


    def detectionToPickPos(self, detection):
        """
        Given a block detection, determines where the arm should go to pick it up
        
        Params:
            detection   A block detection
        
        Returns:
            pos         A position to pick the block at
        """
        worldPos = detection.worldPos.copy()

        if detection.size == 'large':
            if detection.isStacked:
                worldPos[2] -= 20
            else:
                worldPos[2] -= 30

        elif detection.size == 'arch':
            worldPos[2] -= 20

        elif detection.size == 'small':
            worldPos[2] -= 17
            
            # TWEAK: Bottom left quadrant being bad
            if worldPos[0] < -150 and worldPos[1] < 250:
                worldPos[1] += 10

        elif detection.size == 'semi':
            worldPos[2] -= 10
        
        worldPos[2] = max(5, worldPos[2])

        return worldPos


    def executeWaypoint(self, waypoint):
        """
        Makes the arm follow a single waypoint
        
        Params:
            waypoint    A waypoint. Can either be a WP (move), GP (gripper), or DL (delay)
        """
        # Pose
        if isinstance(waypoint, WP):
            self.smooth_set_pose(waypoint.pose, time_params=waypoint.time_params)
        
        # Gripper
        elif isinstance(waypoint, GP):
            self.set_gripper(waypoint.gripper_state)
        
        # Delay
        elif isinstance(waypoint, DL):
            time.sleep(waypoint.delay)
        
        else:
            raise ValueError("Waypoints must be either WP, GP, or DL")
    

    def executeWaypoints(self, waypoints):
        """
        Executes all waypoints in a list
        
        Params:
            waypoints   A list of waypoints. Waypoints must be either WP (move), GP (gripper), or DL (delay)
        """
        for waypoint in waypoints: self.executeWaypoint(waypoint)


    def manipulateAt(self, pos, theta, pickOrPlace='pick', doThetaCorrection=True):
        """
        Pick or place automatically. The world pos is at the bottom of the gripper.
        Goes directly to the positions and doesn't reset at the end.
        
        Params:
            pos                 The position to go to
            theta               The angle to put the wrist
            pickOrPlace         A string, either 'pick' or 'place'
            doThetaCorrection   Whether to change the theta for small blocks
        """

        # Helps with picking up small blocks better
        if doThetaCorrection and pos[0] < 0 and pickOrPlace == 'pick':
            theta += np.pi/2
        theta = clamp(theta)

        if pickOrPlace.lower() == 'pick':
            grip = self.GRIPPER_CLOSED
        else:
            grip = self.GRIPPER_OPEN

        aboveOffsetFirst = 60
        aboveZSecond = 110

        # Above first time
        abovePos1 = pos.copy()
        abovePos1[2] += aboveOffsetFirst
        abovePose = getDownPose(*abovePos1, theta)
        
        # Pickup
        manipPose = getDownPose(*pos, theta)

        # Above second time
        abovePos2 = pos.copy()
        abovePos2[2] = max(aboveZSecond, abovePos1[2])
        abovePose2 = getDownPose(*abovePos2, theta)
        
        waypoints = [
            WP(abovePose), # Above pickup
            WP(manipPose), # Pickup
            GP(grip), # Grasp
            WP(abovePose2), # Above pickup again
        ]
        
        self.executeWaypoints(waypoints)

        print()


    def pickDetection(self, detection, doThetaCorrection=True):
        """
        Given a detection, picks it up
        
        Params:
            detection           A block detection
            doThetaCorrection   Whether to change the theta for small blocks
        """
        # Picks up a detected block
        print('Picking', detection)
        pickWorldPos = self.detectionToPickPos(detection)
        self.pickAt(pickWorldPos, detection.theta, doThetaCorrection=doThetaCorrection)

        if detection.size == 'semi':
            time.sleep(1)
            theta = detection.theta
            # Correct bad grabs by rotating pi/2 and grabbing again
            while self.gripper_val < -0.4:
                theta += np.pi/2
                if theta > 2*np.pi:
                    theta -= 2*np.pi
                print('Bad grab')
                self.gripper.release()
                self.pickAt(pickWorldPos, theta, doThetaCorrection=False)
                time.sleep(1)


    def pickAt(self, pickWorldPos, pickWorldTheta, doThetaCorrection=True):
        """
        Pick automatically. The world pos is at the bottom of the gripper.
        Goes directly to the positions and doesn't reset at the end.
        
        Params:
            pickWorldPos        The world position to pick at
            pickWorldTheta      The theta to rotate the gripper while picking
            doThetaCorrection   Whether to change the theta for small blocks
        """
        print('Picking at', np.array2string(pickWorldPos, precision=2, suppress_small=True))
        self.manipulateAt(pickWorldPos, pickWorldTheta, pickOrPlace='pick', doThetaCorrection=doThetaCorrection)
    

    def placeAt(self, placeWorldPos, placeWorldTheta=0):
        """
        Place automatically. The world pos is at the bottom of the gripper.
        Goes directly to the positions and doesn't reset at the end.
        
        Params:
            placeWorldPos       The world position to place at
            placeWorldTheta     The theta to rotate the gripper while picking
        """
        print('Placing at', np.array2string(placeWorldPos, precision=2, suppress_small=True))
        if self.was45:
            self.realignBlock('small')
        self.manipulateAt(placeWorldPos, placeWorldTheta, pickOrPlace='place')
    

    def realignBlock(self, size):
        """
        Realigns the block in the gripper to a known rotation, used for placing
        
        Params:
            size    The block size
        """
        # Realigns a block already in the grippers
        abovePose = getDownPose(150, -100, 50, gripperAngle=np.pi/2)
        dropPose = getDownPose(100, -100, 5, gripperAngle=np.pi/2)
        moveForwardPose = getDownPose(200, -100, 5, gripperAngle=np.pi/2)
        rotateForwardPose = getDownPose(200, -100, 5, gripperAngle=0)

        if size == 'large':
            pushBackPose = getDownPose(120, -100, 5, gripperAngle=0)
            pickPose = getDownPose(70, -90, 5, gripperAngle=np.pi/2)
        elif size == 'arch':
            # give semicircle and arch tower more space from realignment zone
            abovePose = getDownPose(150, -125, 50, gripperAngle=np.pi/2)
            dropPose = getDownPose(100, -125, 5, gripperAngle=np.pi/2)
            moveForwardPose = getDownPose(200, -125, 5, gripperAngle=np.pi/2)
            rotateForwardPose = getDownPose(200, -125, 5, gripperAngle=0)
            
            pushBackPose = getDownPose(150, -125, 5, gripperAngle=0)
            pickPose = getDownPose(74, -125, 0, gripperAngle=np.pi/2)
        elif size == 'semi':
            # give semicircle and arch tower more space from realignment zone
            abovePose = getDownPose(150, -125, 50, gripperAngle=np.pi/2)
            dropPose = getDownPose(100, -125, 5, gripperAngle=np.pi/2)
            moveForwardPose = getDownPose(200, -125, 5, gripperAngle=np.pi/2)
            rotateForwardPose = getDownPose(200, -125, 5, gripperAngle=0)

            pushBackPose = getDownPose(115, -125, 5, gripperAngle=0)
            pickPose = getDownPose(64, -125, 0, gripperAngle=np.pi/2)
        else:   # samll
            pushBackPose = getDownPose(115, -100, 5, gripperAngle=0)
            pickPose = getDownPose(64, -95, 0, gripperAngle=np.pi/2)
        
        self.executeWaypoints([
            WP(abovePose),
            WP(dropPose),
            GP(self.GRIPPER_OPEN),
            WP(moveForwardPose),
            WP(rotateForwardPose),
            WP(pushBackPose),       # block depend
            WP(moveForwardPose),    # might be
            WP(pickPose),           # block depend
            GP(self.GRIPPER_CLOSED),
            WP(abovePose),
        ])
    

    def knockOver(self, worldPos):
        """
        Knocks over a stack of blocks
        
        Params:
            worldPos    The world coordinates of the top of the stack
        """
        # Left of stack
        pos1 = worldPos.copy()
        pos1[0] -= 60
        pos1[2] += 5
        pose1 = getDownPose(*pos1)

        # Right of stack
        pos2 = worldPos.copy()
        pos2[0] += 10
        pos2[2] += 10
        pose2 = getDownPose(*pos2)

        self.executeWaypoints([
            WP(pose1),
            WP(pose2),
        ])


    def disable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.core.robot_set_motor_registers('group', 'all', 'Torque_Enable', 0)


    def enable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.core.robot_set_motor_registers('group', 'all', 'Torque_Enable', 1)


    def get_positions(self):
        """!
        @brief      Gets the positions.

        @return     The positions.
        """
        return self.position_fb


    def get_velocities(self):
        """!
        @brief      Gets the velocities.

        @return     The velocities.
        """
        return self.velocity_fb


    def get_efforts(self):
        """!
        @brief      Gets the loads.

        @return     The loads.
        """
        return self.effort_fb


    def get_ee_pose(self):
        """!
        @brief      Get the EE pose.

        @return     The EE pose as [x, y, z, roll, pitch, yaw].
        """
        ee_T = FK_dh(self.dh_params, self.get_positions(), self.num_joints)
        ee_pose = getPoseFromT(ee_T)
        return ee_pose


    @_ensure_initialized
    def get_wrist_pose(self):
        """!
        @brief      TODO Get the wrist pose.

        @return     The wrist pose as [x, y, z, phi] or as needed.
        """
        return [0, 0, 0, 0]


    def parse_pox_param_file(self):
        """!
        @brief      TODO Parse a PoX config file

        @return     0 if file was parsed, -1 otherwise
        """
        return -1


    @staticmethod
    def parseDhParamFile(dhConfigFile):
        print("Parsing DH config file...")
        dh_params = parseDhParamFile(dhConfigFile)
        print("DH config file parse exit.")
        return dh_params


    def get_dh_parameters(self):
        """!
        @brief      Gets the dh parameters.

        @return     The dh parameters.
        """
        return self.dh_params


# Waypoint
class WP():
    def __init__(self, pose, time_params=None):
        self.pose = pose
        self.time_params = time_params
    
    def __str__(self):
        return f'WP(pose={repr(self.pose)}, time_params={self.time_params})'

# Gripper
class GP():
    def __init__(self, gripper_state):
        self.gripper_state = gripper_state
    
    def __str__(self):
        return f'GP(gripper_state={self.gripper_state})'

# Delay
class DL():
    def __init__(self, delay):
        self.delay = delay
    
    def __str__(self):
        return f'DL(delay={self.delay})'


class RXArmThread(QThread):
    """!
    @brief      This class describes a RXArm thread.
    """
    updateJointReadout = pyqtSignal(list)
    updateEndEffectorReadout = pyqtSignal(list)

    def __init__(self, rxarm, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      RXArm  The RXArm
        @param      parent  The parent
        @details    TODO: set any additional initial parameters (like PID gains) here
        """
        QThread.__init__(self, parent=parent)
        self.rxarm = rxarm
        self.node = rclpy.create_node('rxarm_thread')
        self.subscription = self.node.create_subscription(
            JointState,
            '/rx200/joint_states',
            self.callback,
            10
        )
        self.subscription  # prevent unused variable warning
        rclpy.spin_once(self.node, timeout_sec=0.5)


    def callback(self, data):
        self.rxarm.position_fb = np.asarray(data.position)[0:5]
        self.rxarm.gripper_val = data.position[5]
        self.rxarm.velocity_fb = np.asarray(data.velocity)[0:5]
        self.rxarm.effort_fb = np.asarray(data.effort)[0:5]
        self.updateJointReadout.emit(self.rxarm.position_fb.tolist())
        self.updateEndEffectorReadout.emit(self.rxarm.get_ee_pose().tolist())
        #for name in self.rxarm.joint_names:
        #    print("{0} gains: {1}".format(name, self.rxarm.get_motor_pid_params(name)))
        if (__name__ == '__main__'):
            print(self.rxarm.position_fb)


    def run(self):
        """!
        @brief      Updates the RXArm Joints at a set rate if the RXArm is initialized.
        """
        while True:
            rclpy.spin_once(self.node)
            time.sleep(0.02)


if __name__ == '__main__':
    rclpy.init() # for test
    rxarm = RXArm()
    print(rxarm.joint_names)
    armThread = RXArmThread(rxarm)
    armThread.start()
    try:
        joint_positions = [-1.0, 0.5, 0.5, 0, 1.57]
        rxarm.initialize()

        rxarm.arm.go_to_home_pose()
        rxarm.set_gripper_pressure(0.5)
        rxarm.set_joint_positions(joint_positions,
                                  moving_time=2.0,
                                  accel_time=0.5,
                                  blocking=True)
        rxarm.gripper.grasp()
        rxarm.arm.go_to_home_pose()
        rxarm.gripper.release()
        rxarm.sleep()

    except KeyboardInterrupt:
        print("Shutting down")

    rclpy.shutdown()