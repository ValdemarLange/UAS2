import numpy as np
import cv2

img = cv2.imread("../input/deer.jpg")
assert img is not None, "Failed to load image."

red = img[:, :, 2]
green = img[:, :, 1]
blue = img[:, :, 0]

cv2.imwrite("../output/ex163_red.png", red)
cv2.imwrite("../output/ex163_green.png", green)
cv2.imwrite("../output/ex163_blue.png", blue)