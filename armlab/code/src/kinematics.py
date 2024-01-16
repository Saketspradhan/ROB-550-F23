"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials
"""

import numpy as np
from numpy import sin, cos, pi, sqrt
from numpy import arctan2 as atan2
# expm is a matrix exponential function
from scipy.linalg import expm

def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > pi:
        angle -= 2 * pi
    while angle <= -pi:
        angle += 2 * pi
    return angle


def FK_dh(dhParams, jointAngles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dhParams     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      jointAngles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    
    T = np.eye(4)
    
    for i in range(link):
        dh = dhParams[i]
        theta = jointAngles[i]
        
        # Gets transformation to next link, then updates T to the end of it
        dh_T = getTransformFromDh(dh[0], dh[1], dh[2], dh[3]+theta)
        # T @= dh_T # This works in numpy 1.25+
        T = T @ dh_T
    
    return T


def getTransformFromDh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """
    
    return np.array([
        [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
        [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
        [0, sin(alpha), cos(alpha), d],
        [0, 0, 0, 1]
    ])


def getEulerAnglesFromT(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    
    theta = atan2(np.sqrt(1-T[2,2]**2), T[2,2])
    if sin(theta) > 0:
        phi = atan2(T[1,2], T[0,2])
        psi = atan2(T[2,1], -T[2,0])
    else:
        phi = atan2(-T[1,2], -T[0,2])
        psi = atan2(-T[1,2], T[2,0])

    return np.array([theta, phi, psi])


def getRpyAnglesFromT(T):
    """!
    @brief      Gets the roll-pitch-yaw angles from a transformation matrix.

    @param      T     transformation matrix

    @return     The roll-pitch-yaw angles from T.
    """
    # This is bad at singularities, TODO fix
    roll = atan2(T[2,1], T[2,2])
    pitch = atan2(-T[2,0], np.sqrt(1-T[2,0]**2))
    yaw = atan2(T[1,0], T[0,0])
    
    return np.array([roll, pitch, yaw])


def rpy2R(roll, pitch, yaw):
    """!
    @brief      Gets a rotation matrix from roll-pitch-yaw parameters

    @param      roll    the roll
                pitch   the pitch
                yaw     the yaw

    @return     The rotation matrix resulting from the roll-pitch-yaw parmeters
    """
    R = np.array([
        [cos(yaw), -sin(yaw), 0],
        [sin(yaw), cos(yaw), 0],
        [0, 0, 1],
    ]) @ np.array([
        [cos(pitch), 0, sin(pitch)],
        [0, 1, 0],
        [-sin(pitch), 0, cos(pitch)],
    ]) @ np.array([
        [1, 0, 0],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll), cos(roll)],
    ])
    return R


def euler2R(theta, phi, psi):
    """!
    @brief      Gets a rotation matrix from zyz Euler angles

    @param      theta   the first z rotation
                phi     the y rotation
                psi     the second z rotation

    @return     The rotation matrix resulting from the euler angle parmeters
    """
    R = np.array([
        [cos(theta), -sin(theta), 0],
        [sin(theta), cos(theta), 0],
        [0, 0, 1]
    ]) @ np.array([
        [cos(phi), 0, sin(phi)],
        [0, 1, 0],
        [-sin(phi), 0, cos(phi)],
    ]) @ np.array([
        [cos(psi), -sin(psi), 0],
        [sin(psi), cos(psi), 0],
        [0, 0, 1]
    ])
    return R


def getPoseFromT(T):
    """!
    @brief      Gets the pose from T.

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    
    xyz = T[0:3, 3]
    angles = getRpyAnglesFromT(T)
    
    return np.concatenate((xyz, angles))


def getDownPose(x, y, z, gripperAngle=0):
    return np.array([x, y, z, pi, gripperAngle])


def getPossiblePose(dhParams, pose):
    """
    A wrapper that tries to convert an unreachable pose to a close reachable one
    
    Params:
        pose       A pose vector
    
    Returns:
        newPose    The possible pose vector
    """
    jointAngles = IK_geometric(dhParams, pose)
    if jointAngles is not None:
        return pose
    else:
        if pose[2] > 0:
            pose[2] -= 10
            return getPossiblePose(dhParams, pose)
        elif pose[2] < 0:
            pose[2] = 0
            return getPossiblePose(dhParams, pose)
        elif pose[2] == 0:
            pose[3] = 3*pi/4
            pose[4] = pi
            pose[4] = clamp(pose[4])
            pose[2] = 5
            return pose


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    pass


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def IK_geometric(dhParams, pose, returnAll=False):
    """!
    @brief      Get all possible joint configs that produce the pose.

    @param      dhParams    The dh parameters
    @param      pose        The desired pose vector as np.array
    @param      returnAll   Whether to return all 4 IK solutions, or just 1

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    
    # No guarantees on correctness
    
    pose = np.asarray(pose)
    
    # T matrix
    if pose.shape == (4, 4):
        R = pose[0:3,0:3]
        x = pose[0,3]
        y = pose[1,3]
        z = pose[2,3]
    # Pose vector (x, y, z, roll, pitch, yaw)
    elif len(pose) == 6:
        x, y, z, roll, pitch, yaw = pose
        R = rpy2R(roll, pitch, yaw)
    elif len(pose) == 5 or len(pose) == 4:
        # Pose vector (x, y, z, vertical angle)
        # Pose vector (x, y, z, vertical angle, gripper rotation)
        # Vertical angle is measured from straight up, clockwise
        x, y, z, roll = pose[:4]
        roll = -roll
        pitch = 0
        yaw = atan2(-x, y)
        
        R = rpy2R(roll, pitch, yaw)
    
    thetas = np.zeros((4,5))
    
    # Useful dh parameters
    a = dhParams[:,0]
    d = dhParams[:,2]
    
    # Radius and theta1
    r = np.sqrt(x**2 + y**2)
    thetas[:2, 0] = atan2(-x, y)
    thetas[2:, 0] = clamp(thetas[0,0] + pi)
    
    # Wrist center
    oc = pose[0:3] - d[4]*R.dot(np.array([0,0,1]))
    r_p = np.sqrt(oc[0]**2 + oc[1]**2)
    z_p = oc[2] - d[0]
    
    # theta2 and theta3
    acos_this = (r_p**2 + z_p**2 - a[1]**2 - a[2]**2) / (2*a[1]*a[2])
    if acos_this < -1 or acos_this > 1: return # Ensure it is valid acos value
    theta3_prime_ik = np.arccos(acos_this)
    theta3_prime_ik = np.array([theta3_prime_ik, -theta3_prime_ik])
    theta2_prime_ik = atan2(r_p, z_p) - atan2(a[2]*sin(theta3_prime_ik), a[1]+a[2]*cos(theta3_prime_ik))
    
    theta2_prime_ik = np.concatenate((theta2_prime_ik, -theta2_prime_ik))

    beta1 = atan2(50, 200)
    thetas[:4,1] = theta2_prime_ik - beta1
    thetas[:2,2] = theta3_prime_ik - pi/2 + beta1
    thetas[2:,2] = 3*pi/2 - theta3_prime_ik + beta1
    
    # theta4
    thetas[:2, 3] = atan2(oc[2]-z, r-r_p) - thetas[:2, 1] - thetas[:2, 2]
    thetas[2:, 3] = atan2(oc[2]-z, r_p-r) - thetas[2:, 1] - thetas[2:, 2]
    
    # theta5
    if len(pose) == 5 and pose[3] == pi:
        for i in range(4):
            thetas[i,4] = clamp(pose[4] + thetas[i,0])
    else:
        # for i, theta_row in enumerate(thetas):
            # R4 = FK_dh(dhParams, theta_row, 4)[:3,:3]
            # rot_z = R4.T @ R
            # thetas[i,4] = atan2(rot_z[1,0], rot_z[0,0])
        thetas[:,4] = pose[4]
    # Also need to take into account singularities, TODO later

    if returnAll:
        return thetas
    else:
        return thetas[0,:].reshape(5)


def write_dh_csv():
    beta1 = atan2(50, 200)
    dh = np.array([
        # a, alpha, d, theta
        [0, pi/2, 103.91, -pi/2],
        [205.73, 0, 0, beta1+pi/2],
        [200, 0, 0, -beta1+pi/2],
        [0, -pi/2, 0, -pi/2],
        [0, 0, 174.15, 0]
    ])
    # This positions the end at the tip of the gripper
    # We may want to decrease d_5 to make it in the middle
    import csv
    
    with open('../config/rx200_dh.csv', 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['dh_a','dh_alpha','dh_d','dh_theta'])
        csv_writer.writerows(dh.tolist())


def test_fk():
    np.set_printoptions(precision=3, suppress=True)
    from resource.config_parse import parseDhParamFile
    dh = parseDhParamFile('../config/rx200_dh.csv')
    
    # Random values
    thetas = [
        0.8147,
        0.9058,
        0.1270,
        0.9134,
        0.6324,
    ]

    # np.set_printoptions(precision=4, suppress=True)
    print(FK_dh(dh, thetas, 5))
    
    """Correct:
    [[  -0.373    0.625   -0.686 -211.222]
    [   0.352   -0.589   -0.728  199.193]
    [  -0.859   -0.512    0.      16.054]
    [   0.       0.       0.       1.   ]]
    """


def test_ik():
    np.set_printoptions(precision=3, suppress=True)
    from resource.config_parse import parseDhParamFile
    
    thetas = np.array([
        -0.9,0.2,0.5,0,0.7
    ])
    dh = parseDhParamFile('../config/rx200_dh.csv')
    T = FK_dh(dh, thetas, 5)
    pose = getPoseFromT(T)
    
    pose = np.array([0, 180, 525, np.pi/2-0.1, 0])

    ik_thetas = IK_geometric(dh, pose)
    ik_T = FK_dh(dh, ik_thetas, 5)
    ik_pose = getPoseFromT(ik_T)

    print('Real thetas:', thetas)
    print('Forward T:\n', T)
    print('Forward pose:', pose)
    print()
    print('Found thetas:', ik_thetas)
    print('IK forward T:\n', ik_T)
    print('IK forward pose:', ik_pose)
    
    return ik_thetas


def test_matlab():
    dh = np.array([
        0.0,1.5707963267948966,103.91,-1.5707963267948966,
        205.73,0.0,0.0,1.8157749899217608,
        200.0,0.0,0.0,1.3258176636680323,
        0.0,-1.5707963267948966,0.0,-1.5707963267948966,
        0.0,0.0,174.15,0.0
    ]).reshape((5,4))
    pose = np.array([200, 200, 10, pi/2, 0])
    ik_thetas = IK_geometric(dh, pose)
    return ik_thetas.reshape((5,1))


def test_ik_all():
    np.set_printoptions(precision=3, suppress=True)
    from resource.config_parse import parseDhParamFile
    
    dh = parseDhParamFile('../config/rx200_dh.csv')
    pose = np.array([-100, 200, 130, pi/2, 0.1])
    
    ik_thetas = IK_geometric(dh, pose, returnAll=True)
    print(ik_thetas)
    for row_thetas in ik_thetas:
        ik_T = FK_dh(dh, row_thetas, 5)
        pose = getPoseFromT(ik_T)
        
        print('Forward pose:', pose)


if __name__ == '__main__':
    test_ik()
    
    # test_ik_all()