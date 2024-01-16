#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
from extrinsic_matrix import getExtrinsic
from label_blocks import getContours, drawBounds, getBlockDetections, BlockDetection, getFreeSpace, getFreePoint, basePx2Point, getSemiDetections
from constants import *

class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Constructs a new instance.
        """
        WIN_COLOR_SIZE = (WIN_SIZE[0], WIN_SIZE[1], 3)
        
        self.useDepthSampling = False # Whether to recalculate the depth sampling every time we calibrate
        self.saveDepthImage = False # Whether to save the calibrated depth image as an image
        self.saveFreeSpace = False # Whether to save the free space as an image, for debugging
        
        self.VideoFrameRaw = np.zeros(WIN_COLOR_SIZE).astype(np.uint8)
        self.VideoFrame = np.zeros(WIN_COLOR_SIZE).astype(np.uint8)
        self.LastAnnotatedFrame = np.zeros(WIN_COLOR_SIZE).astype(np.uint8)
        self.VideoFrameWarped = np.zeros(WIN_COLOR_SIZE).astype(np.uint8)
        
        self.GridFrame = np.zeros(WIN_COLOR_SIZE).astype(np.uint8)
        self.TagImageFrame = np.zeros(WIN_COLOR_SIZE).astype(np.uint8)
        
        self.DepthFrameRaw = np.zeros(WIN_SIZE).astype(np.uint16)
        self.DepthFrameWarped = np.zeros(WIN_SIZE).astype(np.uint16)
        self.DepthFrameHSV = np.zeros(WIN_COLOR_SIZE).astype(np.uint8)
        self.DepthFrameRGB = np.zeros(WIN_COLOR_SIZE).astype(np.uint8)
        
        self.DepthFrameCorrected = np.zeros(WIN_SIZE).astype(np.uint16)
        self.DepthFrameCWarped = np.zeros(WIN_SIZE).astype(np.uint16)
        self.DepthFrameAvg = np.zeros(WIN_SIZE).astype(np.int16)
        if not self.useDepthSampling: self.DepthFrameAvg = cv2.imread('../config/depth_avg_raw.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.int16)
        self.depthSamples = []
        self.maxDepthSamples = 5

        self.freeSpace = np.zeros(WIN_SIZE)

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.useHomography = True
        self.calibratingDepth = False
        # self.intrinsic_matrix = np.array([
        #     [879.268380, 0.000000, 605.519128],
        #     [0.000000, 872.553845, 423.115912],
        #     [0.000000, 0.000000, 1.000000]
        # ])
        self.intrinsicMatrix = np.array([
            [904.317626953125, 0.0, 644.0140380859375],
            [0.0, 904.8245239257812, 360.77752685546875],
            [0.0, 0.0, 1.0]
        ]) # Factory matrix
        self.extrinsicMatrix = getExtrinsic()
        self.invIntrinsic = np.linalg.inv(self.intrinsicMatrix)
        self.invExtrinsic = np.linalg.inv(self.extrinsicMatrix)
        self.colorHomography = np.eye(3)
        self.invColorHomography = np.eye(3)
        self.depthHomography = np.eye(3)
        self.depthExtraHomography = [[1.026119744552121, -0.020500183657092223, -15.241206644975952], [0.004069138283928153, 1.0256092182362686, -11.719422852454715], [1.144054744946532e-05, -1.8778589584547655e-05, 1.0]]
        
        self.lastClick = np.array([0, 0])
        self.newClick = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.depthHomographyPx = []
        
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.gridPoints = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        
        self.tagDetections = None
        self.usedTags = [7, 2, 3, 4]
        self.numTags = len(self.usedTags)
        self.tagLocations = np.array([[-250, -25, 0], [250, -25, 0], [250, 275, 0], [-250, 275, 0]])
        self.boardCornerLocations = np.array([(-500, 475), (500, 475), (500, -175), (-500, -175)])
        self.boardCornerDstPx = self.getBoardCornerDstPx()
        
        """ block info """
        self.blockContours = []
        self.semiBlockDetections = []


    def processVideoFrame(self, VideoFrame):
        """!
        @brief      Process a video frame and detect blocks
        """
        drawBounds(VideoFrame)

        self.blockContours = getContours(self.DepthFrameCWarped, VideoFrame)
        self.blockDetections = getBlockDetections(self.blockContours, self.VideoFrameWarped, self.DepthFrameCWarped, VideoFrame, self)
        # self.semiBlockDetections = getSemiDetections(self.DepthFrameCWarped, self.VideoFrameWarped, modifiedImage=VideoFrame, camera=self)
        
        # Print detection xy
        print([(d.worldPos[0], d.worldPos[1]) for d in self.blockDetections])
        print()

        # Epilepsy warning: Free space overlay on top of color image
        # freeSpaceOverlay = np.repeat(self.freeSpace[:,:,np.newaxis], 3, axis=2).astype(np.uint8)
        # self.VideoFrame = cv2.addWeighted(VideoFrame, 0.9, freeSpaceOverlay, 0.1, 0)

        self.LastAnnotatedFrame = VideoFrame.copy()


    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        # Depth calibration shows edge detection
        if self.calibratingDepth:
            # From the opencv_examples
            depth = self.DepthFrameWarped
            depth = np.clip(depth, 0, 2000).astype(np.uint8)
            depth = cv2.normalize(depth, depth, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
            edges = cv2.Canny(depth, 60, 150)
            depth = np.bitwise_or(depth, edges[:,:,np.newaxis])
            self.DepthFrameRGB = depth
            return
        
        if self.cameraCalibrated and self.useHomography:
            depth = self.DepthFrameCWarped
        else:
            depth = self.DepthFrameRaw
        
        self.DepthFrameHSV[..., 0] = depth >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)


    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread('data/rgb_image.png', cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)


    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameAvg = cv2.imread('data/depth_image.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)


    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        # try:
        frame = self.VideoFrame.copy()

        if self.cameraCalibrated: self.processVideoFrame(frame)

        img = QImage(frame, frame.shape[1], frame.shape[0],
                        QImage.Format_RGB888)
        return img
        # except:
        #     return None


    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, WIN_SIZE_T)
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            print('Error in convertQtGridFrame')
            return None


    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            frame = self.DepthFrameRGB.copy()

            img = QImage(frame, frame.shape[1],
                         frame.shape[0], QImage.Format_RGB888)
            return img
        except:
            print('Error in convertQtDepthFrame')
            return None


    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, WIN_SIZE_T)
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            print('Error in convertQtTagFrame')
            return None


    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)


    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    
    def applyHomography(self, u, v, H):
        """!
        @brief      Given pixel coordinates and a homography matrix, applies the homography
        
        @param      u   The u/x coordinate
        @param      v   The v/y coordinate
        @param      H   The homography matrix
        
        @return     The transformed pixel coordinates
        """
        original = np.array([u, v, 1]).reshape(3,1)
        transformed = H @ original
        transformed /= transformed[2]
        transformed = transformed[:2].reshape(2)

        return transformed
        
    
    def getDepthAt(self, u, v):
        """!
        @brief      Given u, v pixel coordinates, find the lidar's reported depth at those coordinates
        
        @param      u   The pixel u/x coordinate
        @param      v   The pixel v/y coordinate
        
        @return     The depth at that pixel from the lidar sensor
        """
        # Makes sure that we don't get an IndexError
        v_clip = np.clip(v, 0, self.DepthFrameRaw.shape[0]-1)
        u_clip = np.clip(u, 0, self.DepthFrameRaw.shape[1]-1)
        
        if self.cameraCalibrated and self.useHomography:
            return self.DepthFrameWarped[v_clip][u_clip]
        else:
            return self.DepthFrameRaw[v_clip][u_clip]


    def uvToWorld(self, u, v):
        """!
        @brief      Given a known pixel coordinate, gets the world coordinate
        
        @param      u   The pixel u/x coordinate
        @param      v   The pixel v/y coordinate
        
        @return     The world coordinates at that pixel
        """
        
        if self.DepthFrameRaw.any() != 0:
            z = self.getDepthAt(u, v)
            # Undo homography to find original u, v
            if self.cameraCalibrated and self.useHomography:
                px = self.applyHomography(u, v, self.invColorHomography)
                px = px.astype(np.int32)
                u, v = px
            # Pixel to cam frame
            cam_frame = z * self.invIntrinsic @ np.array([u, v, 1]).reshape(3, 1)
            cam_frame = np.vstack((cam_frame, np.array([1])))
            # Cam frame to world frame
            world_frame = self.invExtrinsic @ cam_frame
            world_frame = world_frame[:3]
            world_z = world_frame[2]
            world_frame = world_frame.reshape(3)
            world_frame[2] = world_z + (v - 140) * (20-10)/(690-140) + 10 # TWEAK: Wacky offset
            return world_frame
    
    
    def worldToUV(self, x, y, z, rounding=True):
        """!
        @brief      Given a world, gets the pixel coordinate
        
        @param      x           The world x coordinate
        @param      y           The world y coordinate
        @param      z           The world z coordinate
        @param      rounding    Whether to round the pixel coordinates
        
        @return     The pixel coordinates at that world point
        """
        
        world_frame = np.array([x, y, z, 1])
        # Find in camea frame
        cam_frame = self.extrinsicMatrix @ world_frame
        cam_frame = cam_frame[:3]
        # Find pixel
        px = self.intrinsicMatrix @ cam_frame
        px /= px[2]
        # Chop to get u, v
        px = px[:2]
        # Apply homography
        if self.cameraCalibrated and self.useHomography:
            px = self.applyHomography(px[0], px[1], self.colorHomography)
        
        # Round for indexing
        if rounding:
            px = px.astype(np.int32)
        
        px = px.reshape(2)

        return px
    
    
    def getCalibration(self, worldPts, imgPts):
        """
        Gets an extrinsic matrix using solvePnP
        
        Params:
            worldPts    The world points
            imgPts      The image points
        """
        worldPts = np.asarray(worldPts, dtype=np.float64)
        imgPts = np.asarray(imgPts, dtype=np.float64)

        # print('\nWorld points:\n', worldPts)
        # print('Image points:\n', imgPts)

        # Get H matrix
        _, rot_vec, trans_vec = cv2.solvePnP(worldPts, imgPts, self.intrinsicMatrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
        rot_mat = cv2.Rodrigues(rot_vec)[0]

        H = np.eye(4)
        H[0:3, 0:3] = rot_mat
        H[0:3, 3] = trans_vec.reshape(1,3)

        print('Found extrinsic matrix:')
        print(H)
        
        self.extrinsicMatrix = H
        self.invExtrinsic = np.linalg.inv(H)
        self.getHomography()


    def getBoardCornerDstPx(self):
        """
        Gets where the board corners are in a warped image
        
        Returns:
            boardCornerDstPx    The destination pixels
        """
        m_y = 0.02 # y offset %

        # x and y pixel offsets
        o_y = WIN_H*m_y
        o_x = 0.5*(WIN_W - 20/13*WIN_H*(1-2*m_y))

        boardCornerDstPx = np.array([
            [o_x, o_y],
            [WIN_W-o_x, o_y],
            [WIN_W-o_x, WIN_H-o_y],
            [o_x, WIN_H-o_y],
        ])
        return boardCornerDstPx
    
    
    def getHomography(self):
        """
        Gets the current color and depth homography after calibration
        """
        # Get corner locations in pixels
        boardCornerPx = np.array([self.worldToUV(pos[0], pos[1], 0) for pos in self.boardCornerLocations])
        
        self.colorHomography = cv2.findHomography(boardCornerPx, self.boardCornerDstPx)[0]
        self.invColorHomography = np.linalg.inv(self.colorHomography)

        self.depthHomography = self.colorHomography @ self.depthExtraHomography
        
        print('\nColor Homography:\n', self.colorHomography)
        print('\nDepth Homography:\n', self.depthHomography)
    

    def depthExtraHomographyCalibrate(self):
        """
        Re-computes the depth homography matrix from clicked positions
        """
        self.depthExtraHomography = cv2.findHomography(np.array(self.depthHomographyPx), self.boardCornerDstPx)[0]
        self.depthHomography = self.colorHomography @ self.depthExtraHomography
        self.calibratingDepth = False
        print(self.depthExtraHomography.tolist())
    

    def resetDepthSampling(self):
        self.depthSamples = []
        self.DepthFrameAvg = np.zeros(WIN_SIZE).astype(np.int16)
    

    def getDepthSample(self):
        """
        Records the current depth image and averages it if we have collected enough samples
        
        Returns:
            finished    Whether we collected enough samples and got an average
        """
        self.depthSamples.append(self.DepthFrameRaw.copy())
        if len(self.depthSamples) >= self.maxDepthSamples:
            averaging = np.array(self.depthSamples)
            self.DepthFrameAvg = np.mean(averaging, axis=0, dtype=np.int16)
            if self.saveDepthImage:
                cv2.imwrite('../config/depth_avg_raw.png', self.DepthFrameAvg.astype(np.uint16))
                self.saveDepthImage = False
            return True

        return False
    

    def getFreeSpace(self, disallowedSpace=None):
        """
        Gets the free space given some disallowed region
        
        Params:
            disallowedSpace     A B/W image of the disallowed region. 0=Disallowed, 255=Allowed
        """
        self.freeSpace = getFreeSpace(self.DepthFrameCWarped)
        if disallowedSpace is not None: self.freeSpace = cv2.bitwise_and(self.freeSpace, disallowedSpace)

        if self.saveFreeSpace: cv2.imwrite('../free_space.png', self.freeSpace)


    def getFreePoint(self):
        """
        Given the current free space, gets an available point
        
        Returns:
            point   An image point (u, v)
        """
        u, v, freeSpace = getFreePoint(self.freeSpace)
        self.freeSpace = freeSpace

        if self.saveFreeSpace: cv2.imwrite('../free_space.png', self.freeSpace)

        return basePx2Point(u, v)


    def getEmptyDisallowedSpace(self):
        """
        Gets an empty disallowed space, where everything is presumed allowed
        
        Returns:
            disallowedSpace     A B/W image of the disallowed space
        """
        return np.ones(WIN_SIZE, np.uint8)*255
    

    def projectGridInRGBImage(self):
        """!
        @brief      Projects a grid onto the image using a pre-loaded matrix or a calibrated one
        """
        modified_image = self.VideoFrameRaw.copy()
        if self.cameraCalibrated and self.useHomography: modified_image = cv2.warpPerspective(modified_image, self.colorHomography, (modified_image.shape[1], modified_image.shape[0]))
        
        grid_color = (0, 255, 0)
        
        # Draw grid
        for r in range(self.gridPoints[0].shape[0]):
            for c in range(self.gridPoints[0].shape[1]):
                pos = np.array([self.gridPoints[0][r][c],
                                self.gridPoints[1][r][c],
                                0])
                px = self.worldToUV(*pos)

                cv2.circle(modified_image, px, 3, grid_color, -1)
        
        self.GridFrame = modified_image
     
     
    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection
        """
        detection_color = (0, 255, 0)
        
        modified_image = self.VideoFrameRaw.copy()
        if self.cameraCalibrated and self.useHomography: modified_image = cv2.warpPerspective(modified_image, self.colorHomography, (modified_image.shape[1], modified_image.shape[0]))

        if msg is None:
            self.TagImageFrame = modified_image
            return
        
        # Draw center and corners for each detected tag
        for detection in msg.detections:
            # Center
            x, y = int(detection.centre.x), int(detection.centre.y)
            if self.useHomography: x, y = self.applyHomography(x, y, self.colorHomography).astype(np.int32)
            cv2.circle(modified_image, (x, y), 3, detection_color, -1)
            
            # Corners (bl, br, tr, tl)
            if self.useHomography:
                corner_points = [self.applyHomography(corner.x, corner.y, self.colorHomography).astype(np.int32) for corner in detection.corners]
            else:
                corner_points = [[corner.x, corner.y] for corner in detection.corners]
                
            corner_points.append(corner_points[0])
            corner_points = np.array(corner_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(modified_image, [corner_points], False, detection_color, 1)
            
            # Label
            tag_id = detection.id
            side_len = np.linalg.norm(corner_points[0] - corner_points[1])
            text_pos = np.array([x+1.5*side_len, y+1.5*side_len], dtype=np.int32)
            cv2.putText(modified_image, f'ID: {tag_id}', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        
        # modified_image
        self.TagImageFrame = modified_image


class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera


    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image
        self.camera.VideoFrameRaw = cv_image.copy()
        self.camera.VideoFrameWarped = cv_image.copy()
        if self.camera.cameraCalibrated and self.camera.useHomography:
            self.camera.VideoFrameWarped = cv2.warpPerspective(self.camera.VideoFrameWarped, self.camera.colorHomography,
                                                        (self.camera.VideoFrameWarped.shape[1], self.camera.VideoFrameWarped.shape[0]))
            self.camera.VideoFrameWarped = cv2.resize(self.camera.VideoFrameWarped, WIN_SIZE_T)

            self.camera.VideoFrame = self.camera.VideoFrameWarped.copy()


class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera


    def callback(self, msg):
        self.camera.tagDetections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera


    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        # print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera


    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        if self.camera.cameraCalibrated:
            self.camera.DepthFrameWarped = cv2.warpPerspective(self.camera.DepthFrameRaw, self.camera.depthHomography,
                                               (self.camera.DepthFrameRaw.shape[1], self.camera.DepthFrameRaw.shape[0]))
            rawSigned = self.camera.DepthFrameRaw.astype(np.int16)
            self.camera.DepthFrameCorrected = np.maximum(self.camera.DepthFrameAvg - rawSigned, 0).astype(np.uint16)
            self.camera.DepthFrameCWarped = cv2.warpPerspective(self.camera.DepthFrameCorrected, self.camera.depthHomography,
                                               (self.camera.DepthFrameCorrected.shape[1], self.camera.DepthFrameCorrected.shape[0]))
        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)


    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            print('KeyboardInterrupt in run')
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        except:
            print('Error in camera main')
        finally:
            videoThread.executor.shutdown()
    except:
        print('Error in camera main2')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()