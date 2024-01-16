"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
import cv2
from kinematics import getDownPose, clamp, FK_dh, getPoseFromT
from label_blocks import COLOR_TYPES, markAreaAsBlock, basePoint2Px
from rxarm import RXArm, WP, GP, DL
from constants import *


class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.statusMessage = 'State: Idle'
        self.currentState = 'idle'
        self.nextState = 'idle'
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,          0.0,        0.0],
            [0.75*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.5*-np.pi/2,   -0.5,      -0.3,      np.pi/2,        0.0],
            [0.25*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.0,             0.0,       0.0,          0.0,        0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [0.5*np.pi/2,     0.5,       0.3,     -np.pi/3,        0.0],
            [0.75*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [np.pi/2,         0.5,       0.3,     -np.pi/3,        0.0],
            [0.0,             0.0,       0.0,          0.0,        0.0]]
        
        self.taughtWaypoints = []
        self.recordedAngles = []
        
        ikCalibrateNodes = np.array([
            # x, y, z=30,120, angle=pi/2, gripper_angle=0
            [-250, -75],
            [-250, 225],
            [0, 225],
            [250, 225],
            [250, -75],
            [150, -75],
            [150, 175],
            [0, 175],
            [-150, 175],
            [-150, -75],
        ])
        self.ikCalibrateWaypoints = []
        self.ikCalibrateJoints = []
        self.ikCalibrateIdealJoints = []
        for node in ikCalibrateNodes:
            self.ikCalibrateWaypoints.append([node[0], node[1], 35, np.pi, 0])
            self.ikCalibrateWaypoints.append([node[0], node[1], 120, np.pi, 0])
        self.ikCalibrateWaypoints = np.array(self.ikCalibrateWaypoints)
        self.ikCalibrateStep = 0
        
        self.cameraCalibrationStarted = False
        self.blockDetections = []
        self.semiBlockDetections = []

        self.colorSortKey = lambda d: COLOR_TYPES.index(d.color)


    def setNextState(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.nextState = state


    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.
        """
        if self.nextState == 'initialize_rxarm':
            self.initialize_rxarm()

        if self.nextState == 'idle':
            self.idle()

        if self.nextState == 'estop':
            self.estop()

        if self.nextState == 'execute':
            self.execute()

        if self.nextState == 'calibrate':
            self.calibrate()

        if self.nextState == 'detect':
            self.detect()

        if self.nextState == 'manual':
            self.manual()
        
        if self.nextState == 'teach':
            self.teach()
        
        if self.nextState == 'replay_taught':
            self.replayTaught()
        
        if self.nextState == 'click_drop_pick':
            self.clickDropOpen()
        
        if self.nextState == 'click_drop_place':
            self.clickDropClosed()
        
        if self.nextState == 'ik_calibrate':
            self.ikCalibrate()
        
        if self.nextState == 'task_tester':
            self.taskTester()
        
        if self.nextState == 'depth_homography':
            self.depthExtraHomographyCalibrate()
        
        if self.nextState == 'task_1':
            self.task1PickAndSort()

        if self.nextState == 'task_2':
            self.task2PickAndStack()

        if self.nextState == 'task_3':
            self.task3LineThemUp()

        if self.nextState == 'task_4':
            self.task4StackThemHigh()
        
        if self.nextState == 'task_5':
            self.task5ToTheSky()


    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.statusMessage = 'State: Manual - Use sliders to control arm'
        self.currentState = 'manual'


    def idle(self):
        """!
        @brief      Do nothing
        """
        self.statusMessage = 'State: Idle - Waiting for input'
        self.currentState = 'idle'


    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.statusMessage = 'EMERGENCY STOP - Check rxarm and restart program'
        self.currentState = 'estop'
        self.rxarm.disable_torque()


    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.statusMessage = 'State: Execute - Executing motion plan'
        self.nextState = 'idle'

        for joint_pos in self.waypoints:
            self.rxarm.set_positions(joint_pos)
            time.sleep(2)
    
    
    def teach(self):
        self.currentState = 'teach'
    
    
    def toggleTeach(self):
        if self.currentState != 'teach':
            self.rxarm.disable_torque()
            self.nextState = 'teach'
            self.statusMessage = 'State: Teach - Record Point 1'
            self.taughtWaypoints = []
        else:
            self.nextState = 'idle'
            self.rxarm.enable_torque()
            print(self.taughtWaypoints)


    def teachPosition(self):
        if self.currentState != 'teach':
            return
        
        self.taughtWaypoints.append({'type': 'angles', 'data': self.rxarm.get_positions()})
        self.statusMessage = f'State: Teach - Record Point {len(self.taughtWaypoints)+1}'
    
    
    def teachGripperState(self, gripper_state):
        if self.currentState != 'teach':
            return
        
        self.taughtWaypoints.append({'type': 'gripper', 'data': gripper_state})
        self.statusMessage = f'State: Teach - Record Point {len(self.taughtWaypoints)+1}'
    
    
    def replayTaught(self):
        self.statusMessage = 'State: Replaying Taught Waypoints'
        self.currentState = 'replay_taught'
        self.nextState = 'idle'

        # Go to each waypoint
        if len(self.taughtWaypoints) > 0:
            for waypoint in self.taughtWaypoints:
                if waypoint['type'] == 'angles':
                    self.rxarm.smooth_set_positions(waypoint['data'])

                elif waypoint['type'] == 'gripper':
                    self.rxarm.set_gripper(waypoint['data'])
                    time.sleep(2)
    

    def clickDropOpen(self):
        self.statusMessage = 'State: Click to pick up a block'
        self.currentState = 'click_drop_pick'
        
        if self.camera.newClick:
            self.click_drop_move()
    
    
    def clickDropClosed(self):
        self.statusMessage = 'State: Click to drop a block'
        self.currentState = 'click_drop_place'
        
        if self.camera.newClick:
            self.click_drop_move()
    
    
    def click_drop_move(self):
        world_pos = self.camera.uvToWorld(*self.camera.lastClick)
        self.camera.newClick = False
        
        if self.currentState == 'click_drop_pick':
            world_pos[2] = max(5, world_pos[2]-30)
            self.rxarm.pickAt(world_pos, 0)
            self.nextState = 'click_drop_place'
        else:
            world_pos[2] += 5
            self.rxarm.placeAt(world_pos, 0)
            self.nextState = 'click_drop_pick'


    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.currentState = 'calibrate'
        
        if not self.cameraCalibrationStarted:
            # Make sure we have enough apriltags
            if self.camera.tagDetections is None:
                print('Cannot calibrate: Not enough tags found')
                return
            if len(self.camera.tagDetections.detections) < self.camera.numTags:
                print('Cannot calibrate: Not enough tags found')
                return
            
            tag_centers_px = [None] * self.camera.numTags
            tag_corners_px = [None] * self.camera.numTags
            found_tags = []
            for detection in self.camera.tagDetections.detections:
                tag_id = detection.id
                if tag_id not in self.camera.usedTags:
                    continue
                found_tags.append(tag_id)

                # Get the pixel coordinates of each tag center
                tag_center_px = (detection.centre.x, detection.centre.y)
                # Make sure the pixel coords and world coords are aligned
                tag_centers_px[self.camera.usedTags.index(tag_id)] = tag_center_px

                # Get the pixel coordinates of each tag corner
                corner_points = [(corner.x, corner.y) for corner in detection.corners]
                # print(detection.corners)
                tag_corners_px[self.camera.usedTags.index(tag_id)] = corner_points
            
            # Make sure we have tags 1-4
            found_tags.sort()
            if found_tags != sorted(self.camera.usedTags):
                return

            # Collect every source point for pnp
            img_pts = []
            for tag_id in range(self.camera.numTags):
                img_pts.append(tag_centers_px[tag_id])
                img_pts.extend(tag_corners_px[tag_id])
            
            img_pts = np.array(img_pts)
            
            # Generate every destination point for pnp
            halfwidth = 25/2 # mm
            world_pts = []
            for tag_id in range(self.camera.numTags):
                # Center
                tag_center = self.camera.tagLocations[tag_id]
                world_pts.append(tag_center)
                # Corners
                x, y, z = tag_center
                world_pts.extend([
                    (x-halfwidth, y-halfwidth, z),
                    (x+halfwidth, y-halfwidth, z),
                    (x+halfwidth, y+halfwidth, z),
                    (x-halfwidth, y+halfwidth, z),
                ])
            
            world_pts = np.array(world_pts)
            
            self.camera.getCalibration(world_pts, img_pts)
            self.cameraCalibrationStarted = True
            self.statusMessage = 'Calibration - Completed Calibration'
        
        done = False
        # Get depth samples for average depth
        if self.cameraCalibrationStarted and self.camera.useDepthSampling:
            done = self.camera.getDepthSample()
        # We loaded the depth sampling from a file
        elif not self.camera.useDepthSampling:
            done = True
        
        # Go to next state
        if done:
            self.nextState = 'idle'
            self.camera.cameraCalibrated = True
            time.sleep(2)


    def ikCalibrate(self):
        self.currentState = 'ik_calibrate'
        curr_point = self.ikCalibrateWaypoints[max(0, self.ikCalibrateStep-1)][:3]
        self.statusMessage = f'Calibrating IK - At {curr_point} - Record the position that the arm moves to, then click to advance'
        
        if self.camera.newClick:
            self.goToNextIkCalibratePos()
    
    
    def goToNextIkCalibratePos(self):
        if self.ikCalibrateStep != 0: self.ikCalibrateJoints.append(self.rxarm.get_positions().tolist())
        if self.ikCalibrateStep < len(self.ikCalibrateWaypoints):
            self.ikCalibrateIdealJoints.append(self.rxarm.IK_geometric(self.ikCalibrateWaypoints[self.ikCalibrateStep]).tolist())
            self.rxarm.smooth_set_pose(self.ikCalibrateWaypoints[self.ikCalibrateStep])
            self.camera.newClick = False
            self.ikCalibrateStep += 1
        else:
            print('Ideal Joints')
            print(self.ikCalibrateIdealJoints)
            print()

            print('Real Joints')
            print(self.ikCalibrateJoints)
            print()

            self.nextState = 'idle'
            self.rxarm.gripper.release()
            self.rxarm.sleep()
    

    def depthExtraHomographyCalibrate(self):
        self.currentState = 'depth_homography'
        self.statusMessage = 'Depth Homography Calibration - Click the four corners of the board in the depth frame (start at top left, clockwise)'

        self.camera.calibratingDepth = True

        if self.camera.newClick:
            self.camera.depthHomographyPx.append(self.camera.lastClick)
            self.camera.newClick = False
            if len(self.camera.depthHomographyPx) >= 4:
                self.camera.depthExtraHomographyCalibrate()
                self.nextState = 'idle'


    def relocateBlock(self, detection, safe=False):
        """
        Given a detection, relocates the block to a valid space
        
        Params:
            detection   A block detection
            safe        Whether to reset to a middle waypoint to avoid knocking over stacks
        """
        print('Relocating', detection)

        if detection.size == 'small': # TWEAK: Pick up small blocks on stacks near the top
            detection.worldPos[2] += 4
        
        if detection.size == 'semi':
            self.rxarm.pickDetection(detection, doThetaCorrection=False)
        else:
            self.rxarm.pickDetection(detection)

        if safe:
            midPose = getDownPose(0, 125, 100)
            self.rxarm.smooth_set_pose(midPose)

        placePos = self.camera.getFreePoint()
        placePos = np.array([placePos[0], placePos[1], 5])
        self.rxarm.placeAt(placePos)
    

    def unstackBlocks(self, disallowedSpace=None, safe=False):
        """
        Unstacks all blocks until they are in a single layer
        
        Params:
            disallowedSpace     An image of the extra allowed and disallowed regions. Disallowed=0, Allowed=255
            safe                Whether to reset to a middle waypoint to avoid knocking over stacks
        """
        print('Unstacking blocks\n')
        
        self.detect()
        self.camera.getFreeSpace(disallowedSpace)
        
        # Sees if there is a stack
        foundStack = False
        for detection in self.blockDetections:
            if detection.isStacked:
                foundStack = True
                break

        if foundStack:
            self.rxarm.initWait()

            self.blockDetections.sort(key = lambda d: d.zdiff, reverse=True)

            # Pick up all stacks it sees
            for detection in self.blockDetections:
                if detection.isStacked:
                    self.relocateBlock(detection, safe=safe)
            
            # Check again if we need to unstack underneath them
            self.rxarm.sleepWait()
            self.unstackBlocks(disallowedSpace=disallowedSpace, safe=safe)


    def removeBlocksFromSpace(self, disallowedSpace):
        """
        Relocates each block in a disallowed region to a valid space. Only call this function with already unstacked blocks!
        
        Params:
            disallowedSpace     An image of the extra allowed and disallowed regions. Disallowed=0, Allowed=255
        """
        print('Removing blocks from disallowed space\n')
        
        if self.camera.saveFreeSpace: cv2.imwrite('../disallowed.png', disallowedSpace)

        self.detect()
        self.camera.getFreeSpace(disallowedSpace)

        # Get blocks in bad locations
        badBlocks = []
        for detection in self.blockDetections:
            if disallowedSpace[detection.v][detection.u] == 0:
                badBlocks.append(detection)
        
        for detection in self.semiBlockDetections:
            if disallowedSpace[detection.v][detection.u] == 0:
                badBlocks.append(detection)
        
        # Relocate them if necessary
        if badBlocks:
            self.rxarm.initWait()

            for detection in badBlocks:
                self.relocateBlock(detection)

            self.rxarm.sleepWait()
            self.removeBlocksFromSpace(disallowedSpace)


    def taskTester(self):
        """
        A button that executes some temporary commands, for debugging
        """
        self.nextState = 'idle'

        self.detect()
        self.rxarm.initWait()

        for detection in self.semiBlockDetections:
            print(detection)
            if detection.size == 'semi':
                # pos = detection.worldPos
                # self.rxarm.smooth_set_pose(getDownPose(*pos), detection.theta)
                # self.rxarm.gripper.grasp()
                self.rxarm.pickDetection(detection, doThetaCorrection=False)
                self.rxarm.placeAt(np.array([0, 175, 5]))

        # self.rxarm.executeWaypoints([
        #     WP(np.array([0, 350, 150, np.pi/2, 0])),
        #     WP(np.array([0, 350, 100, np.pi/2, 0])),
        #     WP(np.array([0, 350, 50, np.pi/2, 0])),
        #     WP(np.array([0, 350, 100, np.pi/2, 0])),
        # ])

        self.rxarm.sleepWait()


    def task1PickAndSort(self):
        self.detect()
        self.nextState = 'idle'
        self.statusMessage = 'Task 1 - Pick and Sort'
        self.rxarm.initWait()

        # Generate place positions
        largeOffset = 70
        smallOffset = 50

        placeZ = 10
        
        largePlace = np.array([100, -125, placeZ])
        smallPlace = np.array([-100, -125, placeZ])

        largePlaces = []
        smallPlaces = []
        for i in range(4):
            largePlaces.append(largePlace + np.array([i*largeOffset, 0, 0])) # Farther back
            largePlaces.append(largePlace + np.array([i*largeOffset, largeOffset, 0])) # Closer up

            smallPlaces.append(smallPlace + np.array([-i*smallOffset, 0, 0])) # Farther back
            smallPlaces.append(smallPlace + np.array([-i*smallOffset, smallOffset, 0])) # Closer up

        # Remove blocks already sorted
        self.blockDetections = list(filter(lambda d: d.worldPos[1] > 0, self.blockDetections))

        # Sort blocks
        running = True
        while running:
            for detection in self.blockDetections:
                if detection.size == 'small':
                    placeWorldPos = smallPlaces.pop(0)
                
                elif detection.size == 'large':
                    placeWorldPos = largePlaces.pop(0)
                
                self.rxarm.pickDetection(detection)
                self.rxarm.placeAt(placeWorldPos, np.pi/2)
            
            # Reset again
            self.rxarm.sleepWait()
            self.detect()
            self.blockDetections = list(filter(lambda d: d.worldPos[1] > 0, self.blockDetections))
            if len(self.blockDetections) == 0:
                running = False
            else:
                self.rxarm.initWait()
    

    def task2PickAndStack(self):
        self.nextState = 'idle'
        self.statusMessage = 'Task 2 - Pick and Stack'

        def newInit():
            self.rxarm.moving_time = 2.0
            self.rxarm.accel_time = 0.5
            self.rxarm.smooth_set_pose(getDownPose(0, 175, 100))
        self.rxarm.initialize = newInit

        def newInitWait():
            self.rxarm.initialize()
            time.sleep(1)
        self.rxarm.initWait = newInitWait

        def newSleep():
            self.rxarm.moving_time = 2.0
            self.rxarm.accel_time = 0.5
            self.rxarm.initWait()
            time.sleep(0.5)
            self.rxarm.arm.go_to_sleep_pose(moving_time=self.rxarm.moving_time,
                                accel_time=self.rxarm.accel_time,
                                blocking=False)
            time.sleep(0.5)
        self.rxarm.sleepWait = newSleep

        leftPos = self.camera.tagLocations[0].copy()
        leftPos[2] = 5
        rightPos = self.camera.tagLocations[1].copy()
        rightPos[2] = 5
        topTagPos = self.camera.tagLocations[3].copy()

        nextSide = 'right'

        leftPosPx = basePoint2Px(*leftPos)
        rightPosPx = basePoint2Px(*rightPos)
        topTagPosPx = basePoint2Px(*topTagPos)
        # Make sure there aren't any blocks where we already want to go
        disallowedSpace = self.camera.getEmptyDisallowedSpace()
        disallowedSpace *= 0 # Disallowed by default
        cv2.rectangle(disallowedSpace, topTagPosPx+np.array([10,10]), rightPosPx-np.array([10,10]), 255, cv2.FILLED) # Allowed between apriltags
        disallowedSpace = markAreaAsBlock(*leftPosPx, disallowedSpace)
        disallowedSpace = markAreaAsBlock(*rightPosPx, disallowedSpace)
        self.unstackBlocks(disallowedSpace)

        # Remove blocks already stacked from possible blocks
        def filterDetections():
            i = 0
            while i < len(self.blockDetections):
                detection = self.blockDetections[i]
                if detection.getPlanarDistTo(leftPos[:2]) < 70 or \
                    detection.getPlanarDistTo(rightPos[:2]) < 70:
                    del self.blockDetections[i]
                else:
                    i += 1
            
            smallBlocks = list(filter(lambda d: d.size == 'small', self.blockDetections))
            largeBlocks = list(filter(lambda d: d.size == 'large', self.blockDetections))

            return smallBlocks, largeBlocks

        smallBlocks, largeBlocks = filterDetections()

        self.rxarm.initWait()

        running = True
        while running:
            for detections in [largeBlocks, smallBlocks]:
                for detection in detections:
                    self.rxarm.pickDetection(detection)

                    if nextSide == 'right':
                        placeWorldPos = leftPos.copy()
                        leftPos[2] += detection.height
                        nextSide = 'left'
                    else:
                        placeWorldPos = rightPos.copy()
                        rightPos[2] += detection.height
                        nextSide = 'right'
                    
                    self.rxarm.placeAt(placeWorldPos)
                
            # Reset again
            self.rxarm.sleepWait()
            self.detect()
            smallBlocks, largeBlocks = filterDetections()
            if len(self.blockDetections) == 0:
                running = False
            else:
                self.rxarm.initWait()
    

    def task3LineThemUp(self):
        self.nextState = 'idle'
        self.statusMessage = 'Task 3 - Line Them Up'
        
        # Remove distractors
        self.detect()
        badDetections = list(filter(lambda d: not d.isSquare, self.blockDetections))
        
        midJoints = np.array([0,0,0,-0.3,0])

        while badDetections:
            self.rxarm.initWait()
        
            badPlacePos = np.array([-150, -220, 70])
            for detection in badDetections:
                self.rxarm.pickDetection(detection, doThetaCorrection=False)
                self.rxarm.smooth_set_positions(midJoints)
                self.rxarm.placeAt(badPlacePos)
                self.rxarm.smooth_set_positions(midJoints)
            
            self.rxarm.sleepWait()
            self.detect()
            badDetections = list(filter(lambda d: not d.isSquare, self.blockDetections))


        # Place locations
        largePos = np.array([-150, 250, 5])
        smallPos = np.array([-150, 150, 5])

        largeOffset = 60
        smallOffset = 50

        # Disallowed area
        startWorldPos = largePos.copy()
        endWorldPos = smallPos + np.array([6*smallOffset, 0, 0])

        startBadPx = basePoint2Px(*startWorldPos)
        endBadPx = basePoint2Px(*endWorldPos)
        pxOffset = 70
        pxOffsetRect = np.array([pxOffset,pxOffset]).astype(np.uint32)

        startBadPx -= pxOffsetRect
        endBadPx += pxOffsetRect

        disallowedSpace = self.camera.getEmptyDisallowedSpace()
        cv2.rectangle(disallowedSpace, startBadPx, endBadPx, 0, cv2.FILLED)
        cv2.rectangle(disallowedSpace, (643,541), WIN_SIZE_T, 0, cv2.FILLED)

        self.unstackBlocks(disallowedSpace, safe=True)
        self.removeBlocksFromSpace(disallowedSpace)

        self.detect()

        self.blockDetections.sort(key=self.colorSortKey)

        self.rxarm.initWait()

        for detection in self.blockDetections:
            self.rxarm.pickDetection(detection)
            if detection.size == 'large':
                self.rxarm.placeAt(largePos, np.pi/2)
                largePos[0] += largeOffset

            elif detection.size == 'small':
                self.rxarm.placeAt(smallPos, np.pi/2)
                smallPos[0] += smallOffset
        
        # Push the blocks closer
        waypoints = []
        for startPos, offset, midOffsetScales in zip(
                [largePos, smallPos], # startPos
                [largeOffset, smallOffset], # offset in mm
                [(0,2,3), (0,2,3.25)] # midway moving offset scales
            ):
            startPos[0] += 30
            startPos[1] -= 6
            startPos[2] = 12
            abovePosStart = startPos.copy()
            abovePosStart[2] += 60

            # Start above
            waypoints.append(WP(getDownPose(*abovePosStart)))

            # Waypoints along the line
            for midOffsetScale in midOffsetScales:
                midOffset = offset * midOffsetScale
                midPos = startPos.copy()
                midPos[0] -= midOffset
                waypoints.append(WP(getDownPose(*midPos)))
        
        waypoints.append(WP(getDownPose(0, 175, 60)))
        
        self.rxarm.executeWaypoints(waypoints)
        
        self.rxarm.sleepWait()


    def task4StackThemHigh(self):
        self.nextState = 'idle'
        self.statusMessage = 'Task 4 - Stack them High'

        stack1Pos = np.array([150, -25, 4]) # originally [150, -25, 5]
        stack2Pos = np.array([-150, -25, 5])

        # Get space near the stacks
        disallowedSpace = self.camera.getEmptyDisallowedSpace()
        cv2.rectangle(disallowedSpace, (0,480), WIN_SIZE_T, 0, cv2.FILLED)

        self.unstackBlocks(disallowedSpace)
        self.removeBlocksFromSpace(disallowedSpace)

        # Process small, arches, semicircles
        self.detect()
        self.blockDetections.sort(key=self.colorSortKey)
        self.semiBlockDetections.sort(key=self.colorSortKey)

        arches = list(filter(lambda d: d.size == 'arch', self.blockDetections))
        smallBlocks = list(filter(lambda d: d.size == 'small', self.blockDetections))

        for arch in arches:
            arch.theta = clamp(arch.theta + np.pi/2)

        self.rxarm.initWait()

        for detection in smallBlocks:
            self.rxarm.pickDetection(detection)
            self.rxarm.realignBlock('small') # Realign to be able to make the stack
            self.rxarm.placeAt(stack2Pos)
            stack2Pos[2] += 25

        for semicircle, arch in zip(self.semiBlockDetections, arches):
            self.rxarm.pickDetection(semicircle, doThetaCorrection=False)
            # Realign if necessary
            # time.sleep(1)
            # theta = detection.theta
            # newPickPos = self.rxarm.detectionToPickPos(detection)
            # while self.rxarm.gripper_val < -0.4:
            #     theta += np.pi/2
            #     if theta > 2*np.pi:
            #         theta -= 2*np.pi
            #     print('Bad grab')
            #     self.rxarm.gripper.release()
            #     self.rxarm.pickAt(newPickPos, theta, doThetaCorrection=False)
            self.rxarm.smooth_set_pose(getDownPose(stack1Pos[0], stack1Pos[1], stack1Pos[2] + 150))
            self.rxarm.realignBlock('semi') # Realign to be able to make the stack
            self.rxarm.smooth_set_pose(getDownPose(stack1Pos[0], stack1Pos[1], stack1Pos[2] + 150))
            
            # Place
            self.rxarm.placeAt(stack1Pos)
            stack1Pos[2] += 6

            self.rxarm.pickDetection(arch, doThetaCorrection=False)
            self.rxarm.smooth_set_pose(getDownPose(stack1Pos[0], stack1Pos[1], stack1Pos[2] + 150))
            self.rxarm.realignBlock('arch') # Realign to be able to make the stack
            self.rxarm.smooth_set_pose(getDownPose(stack1Pos[0], stack1Pos[1], stack1Pos[2] + 150))
            self.rxarm.placeAt(stack1Pos)
            stack1Pos[2] += 20
        
        self.rxarm.sleepWait()

    
    def task4StackThemHighLvl2(self):
        self.nextState = 'idle'
        self.statusMessage = 'Task 4 - Stack them High'
        self.rxarm.goSlow = False

        stack1Pos = np.array([150, -25, 5])
        stack2Pos = np.array([-150, -25, 5])

        # Get space near the stacks
        disallowedSpace = self.camera.getEmptyDisallowedSpace()
        cv2.rectangle(disallowedSpace, (0,480), (WIN_W,WIN_H), 0, cv2.FILLED)

        self.unstackBlocks(disallowedSpace, safe=True)
        self.removeBlocksFromSpace(disallowedSpace)

        self.detect()
        self.blockDetections.sort(key = lambda d: COLOR_TYPES.index(d.color))

        smallBlocks = list(filter(lambda d: d.size == 'small', self.blockDetections))
        largeBlocks = list(filter(lambda d: d.size == 'large', self.blockDetections))

        self.rxarm.initWait()

        for detection in smallBlocks:
            self.rxarm.pickDetection(detection)
            self.rxarm.realignBlock('small') # Realign to be able to make the stack

            self.rxarm.placeAt(stack2Pos)
            stack2Pos[2] += 25
        
        large_i = 1
        for detection in largeBlocks:
            self.rxarm.pickDetection(detection)
            if large_i == 6:
                self.rxarm.executeWaypoints([
                    WP(np.array([150, -25, 225, np.pi, 0])),
                    WP(np.array([150, -25, 215, np.pi, 0])),
                    GP(RXArm.GRIPPER_OPEN),
                    WP(np.array([150, 50, 225, np.pi, 0]))
                ])
            else:
                self.rxarm.placeAt(stack1Pos)
            stack1Pos[2] += 40
            large_i += 1
        
        self.rxarm.sleepWait()


    def task5ToTheSky(self):
        self.nextState = 'idle'
        self.statusMessage = 'Task 5 - To the Sky!'

        def newInit():
            self.rxarm.moving_time = 2.0
            self.rxarm.accel_time = 0.5
            self.rxarm.smooth_set_pose(getDownPose(0, 75, 100, np.pi))
        self.rxarm.initialize = newInit

        def newInitWait():
            self.rxarm.initialize()
            time.sleep(0.1)
        self.rxarm.initWait = newInitWait

        def newSleep():
            self.rxarm.moving_time = 2.0
            self.rxarm.accel_time = 0.5
            self.rxarm.initWait()
            time.sleep(2.5)
            self.rxarm.arm.go_to_sleep_pose(moving_time=self.rxarm.moving_time,
                                accel_time=self.rxarm.accel_time,
                                blocking=False)
            time.sleep(0.5)
        self.rxarm.sleepWait = newSleep

        # Vertically place a block on the main stack
        def horizPlace(pos, i):
            placePose = np.array([*pos, np.pi/2, 0])

            abovePose1 = np.array([placePose[0], placePose[1], placePose[2]+50, np.pi/2, 0])
            if i == 13:
                abovePose2 = np.array([placePose[0], placePose[1]-60, placePose[2]+50, np.pi/2-1.2, 0])
                rightPose = np.array([placePose[1]-50, 0, abovePose1[2], np.pi/2, 0])
            else:
                abovePose2 = np.array([placePose[0], placePose[1], placePose[2]+50, np.pi/2, 0])
                rightPose = np.array([placePose[1], 0, abovePose1[2], np.pi/2, 0])

            self.rxarm.executeWaypoints([
                WP(rightPose),
                WP(abovePose1),
                WP(placePose),
                GP(RXArm.GRIPPER_OPEN),
                WP(abovePose2),
                WP(rightPose)
            ])

        # Vertically place first 5 blocks
        def placeTo5():
            for i in range(5):
                self.rxarm.pickAt(pickPos, 0)
                self.rxarm.placeAt(stackPos, 0)
                if i >= 3:
                    self.rxarm.smooth_set_pose(getDownPose(stackPos[0], stackPos[1]-50, stackPos[2]+50), np.pi)
                stackPos[2] += 39
                self.rxarm.initWait()
        
        def placeTo13():
            for i in range(8):
                self.rxarm.pickAt(pickPos, 0)
                horizPlace(stackPos, i+6)
                stackPos[2] += 38
        
        # Build up other stack
        def buildOtherStack():
            for i in range(6):
                self.rxarm.pickAt(pickPos, 0)

                abovePose = moveStackPlacePose.copy()
                abovePose[2] += 50
                resetPose = moveStackPlacePose.copy()
                resetPose[2] += 150

                self.rxarm.executeWaypoints([
                    WP(resetPose),
                    WP(abovePose),
                    WP(moveStackPlacePose),
                    GP(RXArm.GRIPPER_OPEN),
                    WP(abovePose),
                    WP(resetPose),
                ])
                moveStackPlacePose[2] += 39
        
        # Grab base of stack
        def grabOtherStack():
            abovePose1 = np.array([200, 200, moveStackPlacePose[2]+100, np.pi/2, 0])
            abovePose2 = np.array([200, 200, 50, np.pi/2, -0.05])

            self.rxarm.executeWaypoints([
                WP(abovePose1),
                WP(abovePose2),
                WP(moveStackBasePose),
                GP(RXArm.GRIPPER_CLOSED),
            ])
        
        # Moves the other stack to the top of the main stack
        def moveOtherStack():
            slow_times = (1.5, 5)
            gravCompRoll = np.pi/2-0.15
            
            # Slightly up
            slightUpPose = moveStackBasePose.copy()
            slightUpPose[2] += 30
            self.rxarm.smooth_set_pose(slightUpPose, time_params=slow_times)

            # Straight up
            startPose = slightUpPose.copy()
            startPose[3] = gravCompRoll
            straightUpPose = startPose.copy()
            straightUpPose[2] = 450
            self.rxarm.interpCartesian(startPose, straightUpPose, n=5, time_params=slow_times)

            # Get end first for joints
            aboveStackPose = np.array([0, 174, 535, gravCompRoll-0.05, 0])
            aboveStackJoints = self.rxarm.IK_geometric(aboveStackPose)
            aboveStackJoints[3] += 0.06
            
            # Right
            rightJoints = aboveStackJoints.copy()
            rightJoints[0] = -0.5
            rightPose = getPoseFromT(FK_dh(self.rxarm.dh_params, rightJoints, self.rxarm.num_joints))
            rightPose = np.array([rightPose[0], rightPose[1], rightPose[2], gravCompRoll, 0])
            self.rxarm.interpCartesian(straightUpPose, rightPose, n=5, time_params=slow_times)

            self.rxarm.smooth_set_positions(aboveStackJoints, time_params=slow_times)
            
            # Placing
            placeJoints = aboveStackJoints.copy()
            placeJoints[3] += 0.01
            self.rxarm.smooth_set_positions(placeJoints, time_params=slow_times)
            # self.rxarm.interpCartesian(aboveStackPose, placePose, n=2, time_params=slow_times)

            self.rxarm.gripper.release()

        pickPos = np.array([225, 1, 10])
        stackPos = np.array([0, 178, 9])

        self.rxarm.initWait()

        placeTo5()

        # Reset stackPos for horizontal
        stackPos = np.array([0, 182, 40*5 - 3])
        placeTo13()

        # 13th block tester
        # stackPos[2] = stackPos[2] + 39*7
        # self.rxarm.pickAt(pickPos, 0)
        # horizPlace(stackPos, 13)
        ####
        
        moveStackPlacePose = np.array([250, 250, 10, np.pi/2, 0])
        buildOtherStack()

        # Stack build tester
        # moveStackPlacePose[2] += 39*5
        ####

        moveStackBasePose = np.array([255, 255, 13, np.pi/2, 0])
        grabOtherStack()

        moveOtherStack()


    def detect(self):
        """!
        @brief      Detect the blocks
        """
        self.blockDetections = list(self.camera.blockDetections)
        self.semiBlockDetections = list(self.camera.semiBlockDetections)

        # print(self.semiBlockDetections)
    

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.currentState = 'initialize_rxarm'
        self.statusMessage = 'RXArm Initialized!'
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.statusMessage = 'State: Failed to initialize the rxarm!'
            time.sleep(5)
        self.nextState = 'idle'


class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine


    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.statusMessage)
            time.sleep(0.05)