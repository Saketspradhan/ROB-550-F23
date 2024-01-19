import cv2
import time
from gst_cam import camera

w, h = 1280, 720
cap = cv2.VideoCapture(camera(0, w, h))

time.sleep(3)

while True:
    ret, frame = cap.read()
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
