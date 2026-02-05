import numpy as np
import cv2

img = cv2.imread("../input/deer.jpg")
assert img is not None, "Failed to load image."

cv2.line(img, (20, 30), (140, 120), (0, 0, 255), 3)
cv2.imwrite("../output/ex162.png", img)