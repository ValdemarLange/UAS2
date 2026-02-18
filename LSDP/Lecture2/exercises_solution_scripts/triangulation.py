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


def draw_points(image, key_points, radius=10, thickness=5, color=(0, 0, 255)):

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
# load data and images
fid = '../2026_LSDP_Camera_in_Space/EP_opencv.txt'
cal_id = '../2026_LSDP_Camera_in_Space/camera.xml'
image_path = '../2026_LSDP_Camera_in_Space/IMG/'

tree = ET.parse(cal_id)
root = tree.getroot()

# Extract the <f> element
focal = float(root.find("f").text)

with open(fid) as f:
    for noElements, l in enumerate(f):
        pass

image_dict = {}

with open(fid) as f:
    f.readline()

    for i in range(0, noElements):
        p = f.readline()
        image = Image(p, focal)
        image_dict[image.ID] = image


# ---------------------------------------------------------------------
# images to use for calculation
l_id = '284'
r_id = '334'

left_raw = 'DJI_0' + l_id + '.JPG'
right_raw = 'DJI_0' + r_id + '.JPG'

left = image_path + 'DJI_0' + l_id + '.JPG'
right = image_path + 'DJI_0' + r_id + '.JPG'

img_l = cv.imread(left)
img_r = cv.imread(right)

image_dict[left_raw].K[0, 2] = img_l.shape[1]/2
image_dict[left_raw].K[1, 2] = img_l.shape[0]/2

image_dict[right_raw].K[0, 2] = img_r.shape[1]/2
image_dict[right_raw].K[1, 2] = img_r.shape[0]/2


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

cv.namedWindow("image", cv.WINDOW_NORMAL)
cv.setMouseCallback("image", plt_approximation)
cv.imshow("image", img_l)
cv.waitKey(0)
cv.destroyAllWindows()

img_l_points = draw_points(img_l, ref_pt)

left_pix = np.asarray(ref_pt).astype(np.float32).T

# get points position for right image

ref_pt = []

cv.namedWindow("image", cv.WINDOW_NORMAL)
cv.setMouseCallback("image", plt_approximation)
cv.imshow("image", img_r)
cv.waitKey(0)
cv.destroyAllWindows()

img_r_points = draw_points(img_r, ref_pt)

right_pix = np.asarray(ref_pt).astype(np.float32).T

plt.subplot(121), plt.imshow(cv.cvtColor(img_l_points, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(122), plt.imshow(cv.cvtColor(img_r_points, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# ---------------------------------------------------------------------
# triangulate
image_dict[left_raw].t = - image_dict[left_raw].R @ image_dict[left_raw].EP
image_dict[right_raw].t = - image_dict[right_raw].R @ image_dict[right_raw].EP

P_l = image_dict[left_raw].K @ np.hstack((image_dict[left_raw].R, image_dict[left_raw].t))
P_r = image_dict[right_raw].K @ np.hstack((image_dict[right_raw].R, image_dict[right_raw].t))

points_homogenous = cv.triangulatePoints(P_l, P_r, left_pix, right_pix)
points_XYZ = points_homogenous[:3, :] / points_homogenous[3, :]

# ---------------------------------------------------------------------
# reproject

n = points_XYZ.shape[1]

points_homogenous = np.vstack((points_XYZ, np.ones((1, n))))

proj_left = P_l @ points_homogenous
proj_right = P_r @ points_homogenous

# Normalize
proj_left /= proj_left[2, :]
proj_right /= proj_right[2, :]


# Calculate error
dist_left = left_pix - proj_left[0:2, :]
dist_right = right_pix - proj_right[0:2, :]
error_left = (dist_left[0, :]**2 + dist_left[1, :]**2)**0.5
error_right = (dist_right[0, :]**2 + dist_right[1, :]**2)**0.5

# ---------------------------------------------------------------------
# show results

np.set_printoptions(suppress=True)

print('images EP')
print(np.hstack((image_dict[left_raw].EP, image_dict[right_raw].EP)))

print('resultant points')
print(np.vstack((points_XYZ, error_left, error_right)))
