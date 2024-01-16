#!/usr/bin/python
"""!
Main GUI for Arm lab
"""
import os, sys
script_path = os.path.dirname(os.path.realpath(__file__))

import argparse
import cv2
import numpy as np
import rclpy
import time
import datetime
from functools import partial

from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QFileDialog

from resource.ui import Ui_MainWindow
from rxarm import RXArm, RXArmThread
from camera import Camera, VideoThread
from state_machine import StateMachine, StateMachineThread
from constants import *

class Gui(QMainWindow):
    """!
    Main GUI Class

    Contains the main function and interfaces between the GUI and functions.
    """
    def __init__(self, parent=None, dhConfigFile=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        """ Groups of ui commonents """
        self.joint_readouts = [
            self.ui.rdoutBaseJC,
            self.ui.rdoutShoulderJC,
            self.ui.rdoutElbowJC,
            self.ui.rdoutWristAJC,
            self.ui.rdoutWristRJC,
        ]
        self.joint_slider_rdouts = [
            self.ui.rdoutBase,
            self.ui.rdoutShoulder,
            self.ui.rdoutElbow,
            self.ui.rdoutWristA,
            self.ui.rdoutWristR,
        ]
        self.joint_sliders = [
            self.ui.sldrBase,
            self.ui.sldrShoulder,
            self.ui.sldrElbow,
            self.ui.sldrWristA,
            self.ui.sldrWristR,
        ]
        """Objects Using Other Classes"""
        self.camera = Camera()
        print('Creating rx arm...')
        if (dhConfigFile is not None):
            self.rxarm = RXArm(dhConfigFile=dhConfigFile)
        else:
            self.rxarm = RXArm()
        print('Done creating rx arm instance.')
        self.sm = StateMachine(self.rxarm, self.camera)
        """
        Attach Functions to Buttons & Sliders
        """
        # Video
        self.ui.videoDisplay.setMouseTracking(True)
        self.ui.videoDisplay.mouseMoveEvent = self.trackMouse
        self.ui.videoDisplay.mousePressEvent = self.onClick

        # Buttons
        # Handy lambda function falsethat can be used with Partial to only set the new state if the rxarm is initialized
        #nxt_if_arm_init = lambda nextState: self.sm.setNextState(nextState if self.rxarm.initialized else None)
        nxt_if_arm_init = lambda nextState: self.sm.setNextState(nextState)
        self.ui.btn_estop.clicked.connect(self.estop)
        self.ui.btn_init_arm.clicked.connect(self.initRxarm)
        self.ui.btn_torq_off.clicked.connect(
            lambda: self.rxarm.disable_torque())
        self.ui.btn_torq_on.clicked.connect(lambda: self.rxarm.enable_torque())
        self.ui.btn_sleep_arm.clicked.connect(lambda: self.rxarm.sleep())

        # Competition buttons
        self.ui.btn_task1.clicked.connect(partial(nxt_if_arm_init, 'task_1'))
        self.ui.btn_task2.clicked.connect(partial(nxt_if_arm_init, 'task_2'))
        self.ui.btn_task3.clicked.connect(partial(nxt_if_arm_init, 'task_3'))
        self.ui.btn_task4.clicked.connect(partial(nxt_if_arm_init, 'task_4'))
        self.ui.btn_task5.clicked.connect(partial(nxt_if_arm_init, 'task_5'))

        #User Buttons
        self.ui.btnUser1.setText('Calibrate')
        self.ui.btnUser1.clicked.connect(self.calibrateCamera)
        self.ui.btnUser2.setText('Open Gripper')
        self.ui.btnUser2.clicked.connect(self.openGripper)
        self.ui.btnUser3.setText('Close Gripper')
        self.ui.btnUser3.clicked.connect(self.closeGripper)
        self.ui.btnUser4.setText('Execute')
        self.ui.btnUser4.clicked.connect(partial(nxt_if_arm_init, 'execute'))
        self.ui.btnUser5.setText('Start Teaching')
        self.ui.btnUser5.clicked.connect(self.toggleTeach)
        self.ui.btnUser6.setText('Record Waypoint')
        self.ui.btnUser6.clicked.connect(lambda: self.sm.teachPosition())
        self.ui.btnUser7.setText('Replay Taught Waypoints')
        self.ui.btnUser7.clicked.connect(partial(nxt_if_arm_init, 'replay_taught'))
        self.ui.btnUser8.setText('Click and Drop Mode')
        self.ui.btnUser8.clicked.connect(partial(nxt_if_arm_init, 'click_drop_pick'))
        self.ui.btnUser9.setText('Calibrate IK')
        self.ui.btnUser9.clicked.connect(self.startIkCalibrate)
        self.ui.btnUser10.setText('Take Snapshot')
        self.ui.btnUser10.clicked.connect(self.recordSnapshot)
        self.ui.btnUser11.setText('Task Tester')
        self.ui.btnUser11.clicked.connect(partial(nxt_if_arm_init, 'task_tester'))
        self.ui.btnUser12.setText('Calibrate Depth Homography')
        self.ui.btnUser12.clicked.connect(self.startDepthExtraHomographyCalibrate)
        self.ui.btnUser13.setText('Reset Depth Average')
        self.ui.btnUser13.clicked.connect(self.resetDepthAvg)

        # Sliders
        for sldr in self.joint_sliders:
            sldr.valueChanged.connect(self.sliderChange)
        self.ui.sldrMoveTime.valueChanged.connect(self.sliderChange)
        self.ui.sldrAccelTime.valueChanged.connect(self.sliderChange)
        # Direct Control
        self.ui.chk_directcontrol.stateChanged.connect(self.directControlChk)
        # Status
        self.ui.rdoutStatus.setText('Waiting for input')
        """initalize manual control off"""
        self.ui.SliderFrame.setEnabled(False)
        """Setup Threads"""

        # State machine
        self.StateMachineThread = StateMachineThread(self.sm)
        self.StateMachineThread.updateStatusMessage.connect(
            self.updateStatusMessage)
        self.StateMachineThread.start()
        self.VideoThread = VideoThread(self.camera)
        self.VideoThread.updateFrame.connect(self.setImage)
        self.VideoThread.start()
        self.ArmThread = RXArmThread(self.rxarm)
        self.ArmThread.updateJointReadout.connect(self.updateJointReadout)
        self.ArmThread.updateEndEffectorReadout.connect(
            self.updateEndEffectorReadout)
        self.ArmThread.start()

    """ Slots attach callback functions to signals emitted from threads"""

    @pyqtSlot(str)
    def updateStatusMessage(self, msg):
        self.ui.rdoutStatus.setText(msg)

    @pyqtSlot(list)
    def updateJointReadout(self, joints):
        for rdout, joint in zip(self.joint_readouts, joints):
            rdout.setText(str('%+.2f' % (joint * R2D)))

    @pyqtSlot(list)
    def updateEndEffectorReadout(self, pos):
        self.ui.rdoutX.setText(str('%+.2f mm' % (pos[0])))
        self.ui.rdoutY.setText(str('%+.2f mm' % (pos[1])))
        self.ui.rdoutZ.setText(str('%+.2f mm' % (pos[2])))
        self.ui.rdoutPhi.setText(str('%+.2f rad' % (pos[3])))
        self.ui.rdoutTheta.setText(str('%+.2f rad' % (pos[4])))
        self.ui.rdoutPsi.setText(str('%+.2f rad' % (pos[5])))

    @pyqtSlot(QImage, QImage, QImage, QImage)
    def setImage(self, rgb_image, depth_image, tag_image, grid_image):
        """!
        @brief      Display the images from the camera.

        @param      rgb_image    The rgb image
        @param      depth_image  The depth image
        """
        if (self.ui.radioVideo.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(rgb_image))
        if (self.ui.radioDepth.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(depth_image))
        if (self.ui.radioUsr1.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(tag_image))
        if (self.ui.radioUsr2.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(grid_image))

    """ Other callback functions attached to GUI elements"""

    def estop(self):
        self.rxarm.disable_torque()
        self.sm.setNextState('estop')

    def sliderChange(self):
        """!
        @brief Slider changed

        Function to change the slider labels when sliders are moved and to command the arm to the given position
        """
        for rdout, sldr in zip(self.joint_slider_rdouts, self.joint_sliders):
            rdout.setText(str(sldr.value()))

        self.ui.rdoutMoveTime.setText(
            str(self.ui.sldrMoveTime.value() / 10.0) + 's')
        self.ui.rdoutAccelTime.setText(
            str(self.ui.sldrAccelTime.value() / 20.0) + 's')
        self.rxarm.set_moving_time(self.ui.sldrMoveTime.value() / 10.0)
        self.rxarm.set_accel_time(self.ui.sldrAccelTime.value() / 20.0)

        # Do nothing if the rxarm is not initialized
        if self.rxarm.initialized:
            joint_positions = np.array(
                [sldr.value() * D2R for sldr in self.joint_sliders])
            # Only send the joints that the rxarm has
            self.rxarm.set_positions(joint_positions[0:self.rxarm.num_joints])

    def directControlChk(self, state):
        """!
        @brief      Changes to direct control mode

                    Will only work if the rxarm is initialized.

        @param      state  State of the checkbox
        """
        if state == Qt.Checked and self.rxarm.initialized:
            # Go to manual and enable sliders
            self.sm.setNextState('manual')
            self.ui.SliderFrame.setEnabled(True)
        else:
            # Lock sliders and go to idle
            self.sm.setNextState('idle')
            self.ui.SliderFrame.setEnabled(False)
            self.ui.chk_directcontrol.setChecked(False)

    def trackMouse(self, mouse_event):
        """!
        @brief      Show the mouse position in GUI

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """

        pt = mouse_event.pos()
        if self.camera.DepthFrameRaw.any() != 0:
            z = self.camera.getDepthAt(pt.x(), pt.y())
            self.ui.rdoutMousePixels.setText('(%.0f,%.0f,%.0f)' %
                                             (pt.x(), pt.y(), z))
            pt_world = self.camera.uvToWorld(pt.x(), pt.y())
            zdiff = self.camera.DepthFrameCWarped[pt.y()][pt.x()]
            self.ui.rdoutMouseWorld.setText('(%.0f,%.0f,%.0f) (zdiff=%.0f)' %
                                            (pt_world[0], pt_world[1], pt_world[2], zdiff))

    def onClick(self, mouse_event):
        self.calibrateMousePress(mouse_event)
    
    def calibrateMousePress(self, mouse_event):
        """!
        @brief Record mouse click positions for calibration

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """
        """ Get mouse posiiton """
        pt = mouse_event.pos()
        self.camera.lastClick = np.array([pt.x(), pt.y()])
        self.camera.newClick = True

    def initRxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.ui.SliderFrame.setEnabled(False)
        self.ui.chk_directcontrol.setChecked(False)
        self.rxarm.enable_torque()
        self.sm.setNextState('initialize_rxarm')
    
    def toggleTeach(self):
        """
        Toggles teaching mode
        """
        if self.sm.currentState == 'teach':
            self.ui.btnUser5.setText('Start Teaching')
        else:
            self.ui.btnUser5.setText('Stop Teaching')

        self.sm.toggleTeach()
    
    def openGripper(self):
        if self.sm.currentState == 'teach':
            self.sm.teachGripperState(self.rxarm.GRIPPER_OPEN)
        else:
            self.rxarm.gripper.release()

    def closeGripper(self):
        if self.sm.currentState == 'teach':
            self.sm.teachGripperState(self.rxarm.GRIPPER_CLOSED)
        else:
            self.rxarm.gripper.grasp()
        
    def calibrateCamera(self):
        if self.camera.cameraCalibrated:
            self.ui.btnUser1.setText('Calibrate')
            self.camera.cameraCalibrated = False
            self.sm.cameraCalibrationStarted = False
            if self.camera.useDepthSampling: self.camera.resetDepthSampling()
        else:
            self.ui.btnUser1.setText('Uncalibrate')
            self.sm.setNextState('calibrate')
    
    def startIkCalibrate(self):
        self.rxarm.initialize()
        time.sleep(1)
        self.rxarm.gripper.grasp()
        self.sm.goToNextIkCalibratePos()
        self.camera.newClick = False
        self.sm.setNextState('ik_calibrate')
    

    def startDepthExtraHomographyCalibrate(self):
        """
        Starts calibrating to get a new depth homography matrix
        """
        self.camera.newClick = False
        self.camera.depthHomographyPx = []
        self.camera.depthExtraHomography = np.eye(3)
        self.camera.depthHomography = np.array(self.camera.colorHomography)
        self.sm.setNextState('depth_homography')
    

    def resetDepthAvg(self):
        """
        Gets a new average depth image of the empty board and saves the file
        """
        self.camera.useDepthSampling = True
        self.camera.cameraCalibrated = False
        self.camera.saveDepthImage = True
        self.calibrateCamera()
    

    def recordSnapshot(self):
        """
        Records several useful images as a snapshot
        """
        timestamp = str(datetime.datetime.now())
        # Make path
        start_path = os.path.join('../recorded_data/', timestamp)
        os.mkdir(start_path)
        # Raw images
        color_img = cv2.cvtColor(self.camera.VideoFrameRaw, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(start_path, 'color_raw.png'), color_img)
        cv2.imwrite(os.path.join(start_path, 'depth_raw.png'), self.camera.DepthFrameRaw)

        annotated_img = cv2.cvtColor(self.camera.LastAnnotatedFrame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(start_path, 'color_annotated.png'), annotated_img)

        if self.camera.cameraCalibrated:
            # Average depth
            depthAvg = self.camera.DepthFrameAvg.astype(np.uint16)
            cv2.imwrite(os.path.join(start_path, 'depth_avg_raw.png'), depthAvg)
            cv2.imwrite(os.path.join(start_path, 'depth_corrected_raw.png'), self.camera.DepthFrameCorrected)

            # Matrix file
            fileText = ''

            fileText += 'Intrinsic matrix\n'
            fileText += str(self.camera.intrinsicMatrix) + '\n\n'
            
            fileText += 'Extrinsic matrix\n'
            fileText += str(self.camera.extrinsicMatrix) + '\n\n'

            # Warped images
            if self.camera.useHomography:
                depth_avg_warped = cv2.warpPerspective(depthAvg, self.camera.depthHomography,
                                                (depthAvg.shape[1], depthAvg.shape[0]))
                color_warped = cv2.cvtColor(self.camera.VideoFrameWarped, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(start_path, 'color_warped.png'), color_warped)
                cv2.imwrite(os.path.join(start_path, 'depth_warped.png'), self.camera.DepthFrameWarped)
                cv2.imwrite(os.path.join(start_path, 'depth_avg_warped.png'), depth_avg_warped)
                cv2.imwrite(os.path.join(start_path, 'depth_corrected_warped.png'), self.camera.DepthFrameCWarped)
                
                fileText += 'Color homography matrix\n'
                fileText += str(self.camera.colorHomography) + '\n\n'
                fileText += 'Depth homography matrix\n'
                fileText += str(self.camera.depthHomography) + '\n\n'
            
            # Write matrices
            with open(os.path.join(start_path, 'matrices.txt'), 'w') as f:
                f.write(fileText)


def main(args=None):
    """!
    @brief      Starts the GUI
    """
    app = QApplication(sys.argv)
    app_window = Gui(dhConfigFile=args['dhconfig'])
    app_window.show()

    # Set thread priorities
    app_window.VideoThread.setPriority(QThread.HighPriority)
    app_window.ArmThread.setPriority(QThread.NormalPriority)
    app_window.StateMachineThread.setPriority(QThread.LowPriority)

    sys.exit(app.exec_())


# Run main if this file is being run directly
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c",
                    "--dhconfig",
                    required=False,
                    help="path to DH parameters csv file")
    main(args=vars(ap.parse_args()))
