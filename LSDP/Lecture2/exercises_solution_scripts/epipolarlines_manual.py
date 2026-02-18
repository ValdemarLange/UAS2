"""
2026-02-11 Created for the LSDP class on Camera in Space.
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def draw_lines(img, lines, thickness=2):
    image = img.copy()
    r, c, ch = image.shape
    for line in lines:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[2]/line[1]])
        x1, y1 = map(int, [c, -(line[2]+line[0]*c)/line[1]])
        image = cv.line(image, (x0, y0), (x1, y1), color, thickness=thickness)
    return image


def draw_points(img, key_points, radius=2, thickness=2, color=(0, 255, 0)):
    image = img.copy()

    for k in key_points:
        c = (int(k[0]), int(k[1]))
        cv.circle(image, c, radius, color, thickness)

    return image


# data import
left = '1.jpeg'
right = '2.jpeg'
img_l = cv.imread(left)
img_r = cv.imread(right)

pts_left = np.loadtxt("left_pixels.txt", delimiter="\t")
pts_right = np.loadtxt("right_pixels.txt", delimiter="\t")

pts1 = np.int32(pts_left).T
pts2 = np.int32(pts_right).T

img_l_points = draw_points(img_l, pts_left.T)
img_r_points = draw_points(img_r, pts_right.T)

F, _ = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

#------------------------------------------------------
# Draw epilines from left to right
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
lines_left = draw_lines(img_l, lines1)

lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
lines_right = draw_lines(img_r, lines2)

plt.figure(figsize=(15, 10))
plt.subplot(121), plt.imshow(cv.cvtColor(img_l_points, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(122), plt.imshow(cv.cvtColor(lines_right, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig('epipolarlines_manual.png', bbox_inches='tight')

plt.figure(figsize=(15, 10))
plt.subplot(121), plt.imshow(cv.cvtColor(lines_left, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(122), plt.imshow(cv.cvtColor(img_r_points, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
