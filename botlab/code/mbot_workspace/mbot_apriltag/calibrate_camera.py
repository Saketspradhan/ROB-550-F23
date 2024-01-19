import cv2
import time
from gst_cam import camera
import numpy as np
import yaml

# Globals
objpoints = []
imgpoints = []

def read_yaml_file(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data

def write_yaml_file(filepath, data):
    with open(filepath, 'w') as file:
        yaml.dump(data, file)

def get_chessboard_points(chessboard_size, square_size):
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    return objp

def calibrate_camera(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return True, mtx, dist

def draw_text_on_image(image, text, position, font_scale=0.7, thickness=2):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

def main():
    print("Main")
    config = read_yaml_file('calibration_config.yaml')
    chessboard_size = tuple(config['chessboard_size'])
    square_size = config['square_size']
    cap_width = config['camera_width']
    cap_height = config['camera_height']

    cap = cv2.VideoCapture(camera(0, cap_width, cap_height))
    if not cap.isOpened():
        print("Error: Couldn't open the camera.")
        return

    calibration_successful = False
    calibrated_frame = None
    captured_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # flip for mirror image
        frame = cv2.flip(frame, 1)

        if not calibration_successful:
            # Show the number of frames captured
            draw_text_on_image(frame, f"Captured Frames: {captured_frames}", (10, 30))
            draw_text_on_image(frame, f"Press [SPACE] to capture a frame", (10, 60))
            if captured_frames > 10:
                draw_text_on_image(frame, f"Press [ENTER] to calibrate", (10, 90))

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == 32 and ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            # Only add to captured frames if checkerboard is detected
            if ret:
                cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
                captured_frames += 1
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners)
                objpoints.append(get_chessboard_points(chessboard_size, square_size))

        elif key == 13:
            calibration_successful, mtx, dist = calibrate_camera(frame)
            if calibration_successful:
                print("Calibration Successful!")
                print(mtx)
                print(dist)
                camera_params = {
                    "camera_matrix": mtx.tolist(),
                    "distortion_coefficients": dist.tolist()
                }
                write_yaml_file("camera_params.yaml", camera_params)

        if calibration_successful and frame is not None:
            # Message that calibration was applied
            frame = cv2.undistort(frame, mtx, dist, None)
            draw_text_on_image(frame, "Calibration Applied", (10, 30))
        
        # Display Frame
        cv2.imshow("Camera Stream", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()