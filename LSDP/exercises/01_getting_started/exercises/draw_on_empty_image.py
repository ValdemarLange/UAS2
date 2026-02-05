import numpy as np
import cv2
img = np.zeros((100, 200, 3), np.uint8)
cv2.line(img, (20, 30), (40, 120),
(0, 0, 255), 3)
cv2.imwrite("test.png", img)