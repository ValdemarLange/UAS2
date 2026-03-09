import cv2
import numpy as np

input = cv2.VideoCapture("Calibration_video.mov")

i = 0

while True:
    ret, img = input.read()
    if not ret:
        break

    if i % 100 == 0:
        cv2.imwrite("camera-calibration-with-large-chessboards/input/miniProject/image_" + str(i) + ".jpg", img)

    i += 1

input.release()
cv2.destroyAllWindows() 
    
