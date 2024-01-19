# mbot_apriltag
Camera calibration and Apriltag pose estimation code. Assumes that we are running on a Jetson (using GStreamer, NVArgus to access the camera). Also assumes that OpenCV (for Python3) is built with GStreamer enabled. If not, see https://github.com/mdegans/nano_build_opencv.

## Installing

Do `sudo ./install.sh`.

## Usage

First, calibrate your camera. You will need a checkerboard with size specified in calibration_config.yaml (where size = the number of points within the board, so 9x7 squares total). 

After that, run view_apriltags.py to estimate pose of apriltags within the frame. The camera's focus may have to be adjusted. Assumes you are using the `tagCustom48h12` family of tags.

You can also run view_camera.py and view_camera_calibrated.py to show the camera feed.

Keep in mind that everything in this repository should be run on a Desktop and will not work in VSCode or a SSH session (since they spawn a window to display the camera feed).
