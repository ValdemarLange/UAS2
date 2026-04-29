import cv2
import numpy as np
import matplotlib.pyplot as plt


filename = "image.png"
img = cv2.imread(filename)
img_copy = img.copy()

filename = "13_1_2_segmentation_inrange_lab.png"
segmented_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# =========== COUNT ORANGE BLOBS USING findContours =================

# Locate contours.
contours, hierarchy = cv2.findContours(segmented_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw a circle above the center of each of the detected contours.
for contour in contours:
    M = cv2.moments(contour)
    if M['m00'] == 0:
        continue

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.circle(img, (cx, cy), 5, (0, 0, 255), 2)

# print("Number of detected pumpkin blobs: %d" % len(contours))

cv2.namedWindow("pumpkins marked", cv2.WINDOW_NORMAL)
cv2.resizeWindow("pumpkins marked", 700, 700)
cv2.imshow("pumpkins marked", img)
cv2.waitKey(0)


# ====================== 13.2.2 ======================================

dst = cv2.GaussianBlur(segmented_img, (5, 5), 0)
_, blurred_binary = cv2.threshold(dst, 130, 255, cv2.THRESH_BINARY)

cv2.namedWindow("before blur", cv2.WINDOW_NORMAL)
cv2.resizeWindow("before blur", 700, 700)
cv2.imshow("before blur", segmented_img)

cv2.namedWindow("after blur", cv2.WINDOW_NORMAL)
cv2.resizeWindow("after blur", 700, 700)
cv2.imshow("after blur", blurred_binary)

cv2.waitKey(0)

kernel = np.ones((15, 15), np.uint8)

closed_image = cv2.morphologyEx(blurred_binary, cv2.MORPH_CLOSE, kernel)


cv2.namedWindow("filtered", cv2.WINDOW_NORMAL)
cv2.resizeWindow("filtered", 700, 700)


cv2.imshow("filtered", closed_image)
cv2.waitKey(0)
# ====================== 13.2.3 ======================================


    # Locate contours.
contours_filt, hierarchy_filt = cv2.findContours(closed_image, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

# ====================== 13.2.4 ======================================

# Draw a circle above the center of each of the detected contours.
for contour in contours_filt:
    M = cv2.moments(contour)
    if M['m00'] == 0:
        continue

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.circle(img_copy, (cx, cy), 5, (0, 0, 255), 2)

print("Number of detected pumpkin blobs: %d" % len(contours))

print("Number of detected pumpkin blobs in filtered image: %d" % len(contours_filt))

cv2.namedWindow("pumpkins marked filtered", cv2.WINDOW_NORMAL)
cv2.resizeWindow("pumpkins marked filtered", 700, 700)
cv2.imshow("pumpkins marked filtered", img_copy)
cv2.waitKey(0)
