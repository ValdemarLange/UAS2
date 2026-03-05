from MarkerTracker import MarkerTracker
import cv2
import numpy as np

input =cv2.imread("04_fiducial_markers/input/hubsanwithmarker.jpg")

input = cv2.resize(input, None, fx=0.2, fy=0.2)

dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)


marker = cv2.aruco.generateImageMarker(dict, 0, 200)




cv2.imshow("frame", marker)
cv2.waitKey(0)

