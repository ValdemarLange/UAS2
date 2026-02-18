"""
2026-02-11 Created for the LSDP class on Camera in Space.
Triangulation and reprojection
"""

from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import xml.etree.ElementTree as ET


def plt_approximation(event, x, y, flags, param):

    # # grab references to the global variables
    global ref_pt
    if event == cv.EVENT_LBUTTONDOWN:
        ref_pt.append((x, y))


def draw_points(image, key_points, radius=10, thickness=5, color=(0, 255, 0)):

    for k in key_points:
        c = (int(k[0]), int(k[1]))
        cv.circle(image, c, radius, color, thickness)

    return image


class Image:

    def __init__(self, line, f):
        elements = line.split('\t')

        self.ID = elements[0]+'.JPG'
        self.EP = np.array([[float(elements[1])],
                           [float(elements[2])],
                           [float(elements[3])]])
        self.t = np.array([[0],
                           [0],
                           [0]])
        self.K = np.array([[f, 0, 0],
                           [0, f, 0],
                           [0, 0, 1]]) #cx and cy to be filled after image has been opened
        self.calibrated = True
        self.R = np.array([[float(elements[4]), float(elements[5]), float(elements[6])],
                           [float(elements[7]), float(elements[8]), float(elements[9])],
                           [float(elements[10]), float(elements[11]), float(elements[12])]])#.T


# ---------------------------------------------------------------------
# load images to use for sampling

left = '1.jpeg'
right = '2.jpeg'

img_l = cv.imread(left)
img_r = cv.imread(right)

# show images
plt.subplot(121), plt.imshow(cv.cvtColor(img_l, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(122), plt.imshow(cv.cvtColor(img_r, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# ---------------------------------------------------------------------

# record points for both images
# get points position for left image
ref_pt = []

for i in range(0, 15):
    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.setMouseCallback("image", plt_approximation)
    cv.imshow("image", img_l)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.setMouseCallback("image", plt_approximation)
    cv.imshow("image", img_r)
    cv.waitKey(0)
    cv.destroyAllWindows()

numbers = [i for i in range(0, 30)]
firsts = [n for n in numbers if n % 2 == 0]
seconds = [n for n in numbers if n % 2 != 0]

ref_pt_left = [ref_pt[i] for i in firsts]
img_l_points = draw_points(img_l, ref_pt_left)
left_pix = np.asarray(ref_pt_left).astype(np.float32).T

ref_pt_right = [ref_pt[i] for i in seconds]
img_r_points = draw_points(img_r, ref_pt_right)
right_pix = np.asarray(ref_pt_right).astype(np.float32).T

plt.subplot(121), plt.imshow(cv.cvtColor(img_l_points, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(122), plt.imshow(cv.cvtColor(img_r_points, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

np.savetxt("left_pixels.txt", left_pix, fmt="%.2f", delimiter="\t")
np.savetxt("right_pixels.txt", right_pix, fmt="%.2f", delimiter="\t")



