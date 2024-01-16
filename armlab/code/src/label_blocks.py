import cv2
import numpy as np
from constants import *

nonSquare = 0.75 # side ratio

# At z=0, converting (x,y) to (u,v)
m_y = 0.02
base_mm2px = WIN_H * (1 - 2*m_y) / (13*50)
baseOrigin = np.array((WIN_W/2, WIN_H*m_y + 475*base_mm2px), np.int32)
armMaxReachPx = int(350 * base_mm2px)

# Workspace and arm bounds
workspace_tl = np.array((150, 30))
workspace_br = (WIN_W,WIN_H)-workspace_tl
arm_tl = np.array((540, 380))
arm_br = np.array((741, 720))
apriltag_tl = np.array((352, 235))
apriltag_br = np.array((928, 535))

# Generate mask for outside board and arm
boardMask = np.zeros((WIN_H,WIN_W), np.uint8)
cv2.rectangle(boardMask, workspace_tl, workspace_br, 255, cv2.FILLED)
cv2.rectangle(boardMask, arm_tl, arm_br, 0, cv2.FILLED)

# Thresholding
above_board = 15 # Minimum height at which something is considered above the board
minSize = 300
largeSize = 1500
stackContourHeightOffset = 10 # How much farther down we look for the top of a stack
maxSingleBlockHeight = 42 # Maximum height a single block can have before being considered a stack

font = cv2.FONT_HERSHEY_SIMPLEX
COLOR_TYPES = ('red', 'orange', 'yellow', 'green', 'blue', 'violet')
colors = [ # in rgb
    {'id': 'red', 'color': (163, 14, 3)},
    {'id': 'orange', 'color': (196, 112, 2)},
    {'id': 'yellow', 'color': (194, 171, 19)},
    {'id': 'green', 'color': (18, 130, 48)},
    {'id': 'blue', 'color': (49, 65, 235)},
    {'id': 'violet', 'color': (102, 46, 171)},
]
name2Rgb = {color['id'] : color['color'] for color in colors}
name2Rgb['Unknown hue'] = (0, 0, 0)


class BlockDetection():
    def __init__(self, u, v, theta, color, size, zdiff, sideRatio, camera=None):
        self.u = u
        self.v = v
        self.thetaDeg = theta
        self.theta = self.thetaDeg * D2R
        self.color = color
        self.size = size
        match self.size:
            case 'large':
                self.height = 37
            case 'small':
                self.height = 25
            case 'semi':
                self.height = 16
            case 'arch':
                self.height = 29
        self.zdiff = zdiff
        self.sideRatio = sideRatio
        self.isSquare = sideRatio > nonSquare
        self.isStacked = zdiff > maxSingleBlockHeight
        if camera is not None: self.worldPosFromCamera(camera)
        self.camera = camera
    
    def worldPosFromCamera(self, camera):
        self.worldPos = camera.uvToWorld(self.u, self.v)
    
    def changeWorldPos(self, newPos):
        self.worldPos = newPos
    
    def getDistTo(self, pos):
        return np.linalg.norm(pos - self.worldPos)

    def getPlanarDistTo(self, pos):
        return np.linalg.norm(pos - self.worldPos[:2])
    
    def __str__(self):
        posStr = np.array2string(self.worldPos, precision=2, suppress_small=True)
        return f'{self.size} {self.color} block at {posStr}, rotated {self.thetaDeg:.1f}'

    def __repr__(self):
        return '"' + str(self) + '"'


def getMedianColor(image, contour, multiple=False):
    """
    Gets the median hue of an image within a contour
    
    Params:
        image       An image
        contour     A contour
        multiple    Whether the contour passed in is a list of contours
    
    Returns:
        median      The median hue value
    """
    if not multiple: contour = [contour]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contour, -1, 255, -1)

    inMask = image[:,:,0][mask > 0]
    median = np.median(inMask)
    return median


def hueToColor(average_hue):
    """
    Given a hue, returns which color it is
    
    Params:
        average_hue     A hue value
    
    Returns:
        color           The color name as a string
    """
    if 154 <= average_hue <= 180 or average_hue < 3:
        classification = "red"
    elif  3 <= average_hue < 17:
        classification = "orange"
    elif  17 <= average_hue < 66:
        classification = "yellow"
    elif  66 <= average_hue < 97:
        classification = "green"
    elif  97 <= average_hue < 110:
        classification = "blue"
    elif  110 <= average_hue < 154:
        classification = "violet"
    else:
        classification = "Unknown hue"
    
    return classification


def retrieveAreaColor(image, contour):
    """
    Given an image and a contour, gets the color of the block as a string
    
    Params:
        image       An HSV image
        contour     A contour
    
    Returns:
        color       The color name as a string
        hue_value   The average hue of the area
    """
    hue_value = getMedianColor(image, contour)
    closest = hueToColor(hue_value)

    return closest, hue_value


def getFreeSpace(depthData):
    """
    Given a depth image, returns which areas are free to place new blocks
    
    Params:
        depthData       A depth image
    
    Returns:
        freeSpace       A B/W image of available place locations
    """
    # Depend only on the homography
    maskOffsetPx = np.array([50,50])
    
    # Threshold and apply mask to get block locations
    _, thresh = cv2.threshold(depthData, above_board, 2**16, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    # Dilate so we don't hit the blocks
    kernel = np.ones((101,101))
    contourMask = cv2.dilate(thresh, kernel)
    
    # Generate mask for reachable area
    reachableMask = np.zeros_like(thresh)
    cv2.circle(reachableMask, baseOrigin, armMaxReachPx, 255, cv2.FILLED)
    
    # cv2.rectangle(reachableMask, apriltag_tl, apriltag_br, 255, cv2.FILLED)
    newBoardMask = np.zeros_like(thresh)
    cv2.rectangle(newBoardMask, workspace_tl+maskOffsetPx, workspace_br-maskOffsetPx, 255, cv2.FILLED)
    cv2.rectangle(newBoardMask, arm_tl-maskOffsetPx, arm_br+maskOffsetPx, 0, cv2.FILLED)

    reachableMask = cv2.bitwise_and(reachableMask, newBoardMask)

    # Apply mask
    freeSpace = cv2.bitwise_and(reachableMask, cv2.bitwise_not(contourMask))
    
    return freeSpace


def markAreaAsBlock(u, v, space):
    """
    Given a free space, draws a rectangle at a location to denote a block is there, so the area is no longer free
    
    Params:
        u           The u pixel coordinate
        v           The v pixel coordinate
        space       A free space image
    
    Returns:
        newSpace   The new free space
    """
    center = np.array([u,v])
    
    l = 70 # Size of a block in pixels to occlude
    offset = np.array([l,l])

    # Block out a new region on the free space
    space = cv2.rectangle(space, center-offset, center+offset, 0, cv2.FILLED)
    return space


def getFreePoint(freeSpace):
    """
    Given a free space, picks a random available point, then marks the space as if a block was placed there
    
    Params:
        freeSpace       A free space image
    
    Returns:
        u           An available u pixel coordiante
        v           An available v pixel coordinate
        newSpace    The new free space with a block occluded at (u,v)
    """
    # Pick a random nonzero point to place the next block
    nonzero_i, nonzero_j = np.nonzero(freeSpace)
    if len(nonzero_i) == 0: return -1, -1, freeSpace # Nothing left
    chosen_i = np.random.randint(len(nonzero_i))
    
    v, u = nonzero_i[chosen_i], nonzero_j[chosen_i]
    freeSpace = markAreaAsBlock(u, v, freeSpace)
    
    return u, v, freeSpace


def basePoint2Px(x, y, z=0):
    """
    Converts a coordinate from the world frame to the image frame at z=0
    
    Params:
        x   The x world coordinate
        y   The y world coordinate
    
    Returns:
        u   The u pixel coordinate
        v   The v pixel coordinate
    """
    return (baseOrigin + np.array((x,-y))*base_mm2px).astype(np.uint32)


def basePx2Point(u, v):
    """
    Converts a coordinate from the image frame to the world frame at z=0
    
    Params:
        u   The u pixel coordinate
        v   The v pixel coordinate
    
    Returns:
        x   The x world coordinate
        y   The y world coordinate
    """
    return np.array((u-baseOrigin[0], baseOrigin[1]-v)) / base_mm2px


def drawBounds(image):
    """
    Draws the bounds on an image
    
    Params:
        image   A color image
    """
    cv2.rectangle(image, workspace_tl, workspace_br, (0, 0, 255), 2)
    cv2.rectangle(image, arm_tl, arm_br, (0, 0, 255), 2)


def pruneTinyContours(contours, minSize=minSize):
    """
    Removes tiny contours that are too small to be a block detection
    
    Params:
        contours        A list of contours
        minSize         The minimum size to keep. Contours are kept if area >= minSize
    
    Returns:
        keepContours    The kept contours
    """
    keepContours = []
    for c in contours:
        if cv2.contourArea(c) >= minSize:
            keepContours.append(c)
            # print(cv2.contourArea(c))
    return keepContours


def getContours(depthData, colorImage=None, lowThresh=above_board, highThresh=1000, stackContourHeightOffset=stackContourHeightOffset):
    """
    Gets contours from depth data
    
    Params:
        depthData       A depth image
        colorImage      A color image to draw on
        lowThresh       The low depth threshold
        highThresh      The high depth threshold
        stackContourHeightOffset    How much to offset the height when re-thresholding stacks
    
    Returns:
        contours        A list of contours
    """
    # Threshold and apply mask
    _, thresh = cv2.threshold(depthData, lowThresh, highThresh, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    thresh = cv2.bitwise_and(thresh, boardMask)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = pruneTinyContours(contours)
    
    keepContours = []
    # Re-threshold to only get the top block for a stack
    for contour in contours:
        # Draw single contour
        singleContourMask = np.zeros_like(depthData, dtype=np.uint8)
        cv2.drawContours(singleContourMask, [contour], -1, 255, -1)
        
        # Get max height, determine if it is stacked or not
        maxHeightInContour = np.percentile(depthData[singleContourMask > 0], 90)
        # If stacked, threshold again based on max depth - offset
        if maxHeightInContour > maxSingleBlockHeight:
            _, thresh = cv2.threshold(depthData, maxHeightInContour-stackContourHeightOffset, 65536, cv2.THRESH_BINARY)
            thresh = thresh.astype(np.uint8)
            thresh = cv2.bitwise_and(thresh, singleContourMask)
        
            # Get contour again
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            keepContours.extend(contours)
        # Remove low contours
        elif maxHeightInContour < 20:
            continue
        # Otherwise, keep the contour as is
        else:
            keepContours.append(contour)
        
    contours = pruneTinyContours(keepContours)
    
    if colorImage is not None:
        cv2.drawContours(colorImage, contours, -1, (255, 0, 255), 2)

    return contours


def getBlockDetections(contours, image, depthImage, modifiedImage=None, camera=None, semis=False):
    """
    Finds the blocks in an image
    
    Params:
        contours        The found contours
        image           An RGB image
        depthImage      A depth image
        modifiedImage   An image to draw on
        camera          A camera, used to calculate world coordinates of the detections
        semis           Whether we are detecting semicircles
        
    Returns:
        detections      A list of block detections
    """
    detections = []
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    for contour in contours:
        # Color
        color, hue_value = retrieveAreaColor(image, contour)
        
        # Bounding box
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        
        # Center
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        # Z difference
        zdiff = depthImage[cy][cx]

        # Whether we are square
        sideRatio = min(rect[1]) / max(rect[1]) # Ideally 1 for a square

        # Semicircle
        if semis:
            blockSize = 'semi'
            center_xy = rect[0]
            theta, modifiedImage = compute_semicircle_grasp_angle(box, modifiedImage, center_xy, depthImage)
            theta += np.pi/2
        # Non-square
        elif sideRatio <= nonSquare:
            blockSize = 'arch'
            theta, modifiedImage = compute_distractor_grasp_angle(box, modifiedImage)
            theta = -theta
        # Square
        else:
            blockSize = 'large' if cv2.contourArea(contour) >= largeSize else 'small'
            theta = rect[2] # In degrees now
        
        # Draw on image
        if modifiedImage is not None:
            cv2.drawContours(modifiedImage, [box], 0, name2Rgb[color], 2)

            if sideRatio <= nonSquare:
                cv2.putText(modifiedImage, '(Distractor)', (cx-30, cy+60), font, 0.5, (255,0,0), thickness=2)

            cv2.putText(modifiedImage, blockSize + ' ' + color, (cx-30, cy+40), font, 0.75, name2Rgb[color], thickness=2)
            cv2.putText(modifiedImage, str(int(theta)) + ' deg', (cx, cy), font, 0.5, (255,255,255), thickness=2)
            if zdiff > maxSingleBlockHeight: cv2.putText(modifiedImage, '(stack)', (cx-30, cy+60), font, 0.5, (255,255,255), thickness=2)
        
        detections.append(BlockDetection(cx, cy, theta, color, blockSize, zdiff, sideRatio, camera=camera))
    
    return detections


def compute_distractor_grasp_angle(box,rgb_image):
    """
    input
        box: np.array (4,2), computed as
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
        rgb_iamge: np.array (h,w,3)
    output
        angle_val: float in degrees (0-180deg)
            angle of line wrt horizontal line
        rgb_image: np.array (h,w,3)
            adds visuals
            adds red circles for inner side points
            adds blue circles for inner side points to grab
            adds blue line for orientaiton of gripper
    """
    center_coord = np.zeros((2))
    coords = np.vstack((box,box[0,:]))
    side_lengths = np.zeros((4))
    center_coords = np.zeros((4,2)).astype(int)
    
    # find the centers for each side of the rectangle
    for coord_idx in range(coords.shape[0]-1):
        center_coord = (coords[coord_idx] + coords[coord_idx+1])/2
        side_lengths[coord_idx] = np.linalg.norm(coords[coord_idx] - coords[coord_idx+1])
        center_coords[coord_idx] = (coords[coord_idx] + coords[coord_idx+1])/2
        
        # draw red dots for each point and boxes around each point
        cv2.circle(rgb_image,(int(center_coord[0]),int(center_coord[1])), 3, (0,0,255), -1)
    
    # find the largest lengths
    side_lengths_cor = [side_lengths[0] + side_lengths[2],
                        side_lengths[1] + side_lengths[3]]
    high_idx = np.argmin(side_lengths_cor)

    # draw blue circles and line for grasping orientation
    cv2.circle(rgb_image,(int(center_coords[high_idx,0]),int(center_coords[high_idx,1])), 3, (255,0,0), -1)
    cv2.circle(rgb_image,(int(center_coords[high_idx+2,0]),int(center_coords[high_idx+2,1])), 3, (255,0,0), -1)
    cv2.line(rgb_image, (int(center_coords[high_idx,0]),int(center_coords[high_idx,1])),
                            (int(center_coords[high_idx+2,0]),int(center_coords[high_idx+2,1])), (255,0,0), 3)
    
    # compute line angle wrt horizontal image line
    delta_x = int(center_coords[high_idx,0]) - int(center_coords[high_idx+2,0])
    delta_y = int(center_coords[high_idx,1]) - int(center_coords[high_idx+2,1])
    angle_val = -np.arctan2(delta_y, delta_x) * R2D

    return angle_val, rgb_image


def compute_average_height(contour,depth_data):
    # compute average height
    # create boolean hot mask for extracting just the polygon
    mask = np.zeros((depth_data.shape[0], depth_data.shape[1]))
    cv2.fillConvexPoly(mask, contour.reshape(-1,2), 1)
    mask = mask.astype(bool)

    # create image that shows just the polygon of interest sets everything else to black
    out_depth = np.zeros_like(depth_data)
    out_depth[mask] = depth_data[mask]   # image of just the polygon, everything else is black
    height_avg = int(out_depth[np.nonzero(out_depth)].mean())

    return height_avg


def compute_semicircle_grasp_angle(box,rgb_image,center_xy,depth_data):
    import skimage.measure
    """
    input
        box: np.array (4,2), computed as
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
        rgb_image: np.array (h,w,3)
        center_xy: tuple (float,float), computed as
            rect = cv2.minAreaRect(contour)
            center_xy = rect[0]
        depth_data: np.array (h,w)
    output
        angle_val: float in degrees (0-180deg)
            angle of line wrt horizontal line
        rgb_image: np.array (h,w,3)
            adds visuals
            adds red circles for inner side points
            adds blue circles for inner side points to grab
            adds blue line for orientaiton of gripper
    """
    coords = np.vstack((box,box[0,:]))
    
    # find depth values at each side
    depth_values = np.zeros((4))
    center_coords = np.zeros((4,2)).astype(int)
    offset_coords = np.zeros((4,2)).astype(int)
    for coord_idx in range(coords.shape[0]-1):
        side_c = (coords[coord_idx] + coords[coord_idx+1])/2
        # inner_c = np.array([(3*side_c[0] + center_xy[0]),(3*side_c[1] + center_xy[1])])/4
        # inner_c = np.array([(2*side_c[0] + 2*center_xy[0]),(2*side_c[1] + 2*center_xy[1])])/4
        # inner_c = np.array([(1*side_c[0] + 3*center_xy[0]),(1*side_c[1] + 3*center_xy[1])])/4
        inner_c = side_c
        
        offset_coord = coords[coord_idx] + 0.15*(coords[(coord_idx+2)%4] - coords[coord_idx])

        offset_coord = offset_coord.astype(int)
        inner_c = inner_c.astype(int)
        
        center_coords[coord_idx] = inner_c
        offset_coords[coord_idx] = offset_coord

        if rgb_image is not None:
            # draw blue dots for each point and boxes around each point
            cv2.circle(rgb_image, (inner_c[0], inner_c[1]), 3, (0,0,255), -1)
            # draw green dots for each offset point
            cv2.circle(rgb_image, (offset_coord[0], offset_coord[1]), 3, (0,255,0), -1)
    
    lines = []
    maxes = []
    avgs = []
    for i in range(4):
        plus1 = (i+1) % 4
        lines.append(skimage.measure.profile_line(depth_data, (offset_coords[i,1], offset_coords[i,0]), (offset_coords[plus1,1], offset_coords[plus1,0]), linewidth=1, reduce_func=np.mean))
        lines[i] = lines[i][lines[i] > 3]
        maxes.append(np.max(lines[i]))
        avgs.append(np.mean(lines[i]))
    
    avg1 = (maxes[0] + maxes[2]) / 2
    avg2 = (maxes[1] + maxes[3]) / 2

    # max1 = max(maxes[0], maxes[2])
    # max2 = max(maxes[1], maxes[3])
    
    # print((avg1, avg2))
    # print((max1, max2))
    # print()
    
    if avg1 < avg2:
        if rgb_image is not None: cv2.line(rgb_image, center_coords[1], center_coords[3], (255, 0, 0), 3)
        deltas = center_coords[3] - center_coords[1]
    else:
        if rgb_image is not None: cv2.line(rgb_image, center_coords[0], center_coords[2], (255, 0, 0), 3)
        deltas = center_coords[2] - center_coords[0]

    angle_val = np.arctan2(deltas[1], deltas[0]) * R2D

    return angle_val, rgb_image


def getSemiDetections(depthData, colorImage, modifiedImage=None, camera=None):
    """
    Gets semicircles, which are lower to the ground than squares
    
    Params:
        depthData       A depth image
        colorImage      An RGB image
        modifiedImage   An image to draw on
        camera          A camera, used to calculate world coordinates of the detections
        
    Returns:
        detections      A list of semicircle detections
    """
    # Threshold and apply mask
    lowThresh = 5
    highThresh = 1000
    # Threshold and apply mask
    _, thresh = cv2.threshold(depthData, lowThresh, highThresh, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    thresh = cv2.bitwise_and(thresh, boardMask)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = pruneTinyContours(contours)

    keepContours = []
    # Re-threshold to only get the top block for a stack
    for contour in contours:
        # Draw single contour
        singleContourMask = np.zeros_like(depthData, dtype=np.uint8)
        cv2.drawContours(singleContourMask, [contour], -1, 255, -1)
        
        # Get max height, determine if it is stacked or not
        maxHeightInContour = np.percentile(depthData[singleContourMask > 0], 90)
        # Only keep low contours
        if maxHeightInContour < 20:
            keepContours.append(contour)
    contours = pruneTinyContours(keepContours)

    if modifiedImage is not None:
        cv2.drawContours(modifiedImage, contours, -1, (255, 0, 0), 2)
    
    # Get detections
    detections = getBlockDetections(contours, colorImage, depthData, modifiedImage=modifiedImage, camera=camera, semis=True)

    return detections


if __name__ == '__main__':
    folder = 'semi (3)'
    COLOR_IMG = f'../recorded_data/{folder}/color_warped.png'
    DEPTH_IMG = f'../recorded_data/{folder}/depth_corrected_warped.png'
    depthData = cv2.imread(DEPTH_IMG, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    rgbImage = cv2.imread(COLOR_IMG)
    modifiedImage = cv2.imread(COLOR_IMG)
    
    getSemiDetections(depthData, rgbImage, modifiedImage)
    
    # getContours(depthData, modifiedImage, 4, 1000)

    cv2.imshow("Image window", modifiedImage)
    while True:
        k = cv2.waitKey(0)
        if k == 27:  # quit with ESC
            break
    cv2.destroyAllWindows()