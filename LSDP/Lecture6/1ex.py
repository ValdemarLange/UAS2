import cv2
import numpy as np

img = cv2.imread("input/sequence-penguin.jpg")

sift = cv2.SIFT_create()

keypoints = sift.detect(img)

cv2.drawKeypoints(image=img, keypoints=keypoints, outImage=img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

