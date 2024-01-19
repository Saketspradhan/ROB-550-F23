import cv2
import time
import numpy as np
import yaml
from gst_cam import camera

# Load camera parameters from yaml
with open("camera_params.yaml", 'r') as f:
    params = yaml.safe_load(f)

camera_matrix = np.array(params['camera_matrix'], dtype=np.float32)
distortion_coefficients = np.array(params['distortion_coefficients'], dtype=np.float32)


w, h = 1280, 720
cap = cv2.VideoCapture(camera(0, w, h))

time.sleep(3)

while True:
    ret, frame = cap.read()
    frame = cv2.undistort(frame, camera_matrix, distortion_coefficients)
    # flip for mirror image
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
