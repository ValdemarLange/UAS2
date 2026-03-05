from MarkerTracker import MarkerTracker
import cv2
import numpy as np


input = cv2.VideoCapture("04_fiducial_markers/input/video_with_aruco_markers_dict_4x4_250.mov")

tracker = MarkerTracker(order=4, kernel_size=5, scale_factor=0.1)
# tracker.track_marker_with_missing_black_leg = False

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, parameters)


while True:
    ret, img = input.read()
    if not ret:
        break
    # img_small = cv2.resize(img, None, fx=0.5, fy=0.5)

    corners, ids, rejected = detector.detectMarkers(img)
	    
    img_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)

    cv2.imshow("Video", img_markers)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


input.release()
cv2.destroyAllWindows()


