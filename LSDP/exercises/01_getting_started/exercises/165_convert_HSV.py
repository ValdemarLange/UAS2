import numpy as np
import cv2

img = cv2.imread("../input/deer.jpg")
assert img is not None, "Failed to load image."

img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

hue = img_hsv[:, :, 0]
saturation = img_hsv[:, :, 1]
value = img_hsv[:, :, 2]

cv2.imwrite("../output/ex165_hue.png", hue)
cv2.imwrite("../output/ex165_saturation.png", saturation)
cv2.imwrite("../output/ex165_value.png", value)