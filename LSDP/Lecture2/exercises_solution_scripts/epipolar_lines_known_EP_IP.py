"""
2026-02-11 Created for the LSDP class on Camera in Space.
"""

from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import xml.etree.ElementTree as ET


def skew(v):
    return np.array([[0, -v[2,0], v[1,0]],
                     [v[2,0], 0, -v[0,0]],
                     [-v[1,0], v[0,0], 0]])


def draw_lines(image, lines, thickness=2):
    img = image.copy()
    h, w = img.shape[:2]

    for line in lines:
        a, b, c = line
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # Handle vertical lines (b ≈ 0)
        if abs(b) < 1e-6:
            x = int(-c / a) if abs(a) > 1e-6 else 0
            cv.line(img, (x, 0), (x, h), color, thickness)
        else:
            y0 = int(-c / b)
            y1 = int(-(c + a * w) / b)
            cv.line(img, (0, y0), (w, y1), color, thickness)

    return img


def plt_approximation(event, x, y, flags, param):

    # # grab references to the global variables
    global ref_pt
    if event == cv.EVENT_LBUTTONDOWN:
        ref_pt.append((x, y))


def draw_points(image, key_points, radius=10, thickness=5, color=(0, 0, 255)):
    img = image.copy()
    for k in key_points:
        c = (int(k[0]), int(k[1]))
        cv.circle(img, c, radius, color, thickness)

    return img


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
fid = 'C:/Users/elpa/02_TEACHING/2024-LSDP/03_MATERIALS_EXERCISES/Class_1/3D_reconstruction/EP_opencv.txt'
cal_id = 'C:/Users/elpa/02_TEACHING/2024-LSDP/03_MATERIALS_EXERCISES/Class_1/3D_reconstruction/camera.xml'
image_path = 'C:/Users/elpa/02_TEACHING/2024-LSDP/03_MATERIALS_EXERCISES/Class_1/3D_reconstruction/IMG/'

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

# calculate fundamental matrix

R = image_dict[right_raw].R @ image_dict[left_raw].R.T  # rotation from left to right
t = image_dict[right_raw].R @ (image_dict[left_raw].EP - image_dict[right_raw].EP)  # camera space translation
Kinv = np.linalg.inv(image_dict[left_raw].K)

F_lr = Kinv.T @ skew(t) @ R @ Kinv

R = image_dict[left_raw].R @ image_dict[right_raw].R.T  # rotation from right to left
t = image_dict[left_raw].R @ (image_dict[right_raw].EP - image_dict[left_raw].EP)  # camera space translation

Kinv = np.linalg.inv(image_dict[left_raw].K)

F_rl = Kinv.T @ skew(t) @ R @ Kinv

# ---------------------------------------------------------------------

# record points for  left image

ref_pt = []

cv.namedWindow("image", cv.WINDOW_NORMAL)
cv.setMouseCallback("image", plt_approximation)
cv.imshow("image", img_l)
cv.waitKey(0)
cv.destroyAllWindows()

img_l_points = draw_points(img_l, ref_pt)

left_pix = np.asarray(ref_pt).astype(np.float32).T

# calculate epipolar lines
left_pix_homogenous = np.vstack((left_pix, np.ones((1, left_pix.shape[1]))))
right_lines =(F_rl @ left_pix_homogenous).T

img_r_lines = draw_lines(img_r, right_lines, thickness=10)

plt.subplot(121), plt.imshow(cv.cvtColor(img_l_points, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(122), plt.imshow(cv.cvtColor(img_r_lines, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# ---------------------------------------------------------------------

# record points for  right image

ref_pt = []

cv.namedWindow("image", cv.WINDOW_NORMAL)
cv.setMouseCallback("image", plt_approximation)
cv.imshow("image", img_r)
cv.waitKey(0)
cv.destroyAllWindows()

img_r_points = draw_points(img_r, ref_pt)

right_pix = np.asarray(ref_pt).astype(np.float32).T

# calculate epipolar lines
right_pix_homogenous = np.vstack((right_pix, np.ones((1, right_pix.shape[1]))))
left_lines = (F_rl @ right_pix_homogenous).T

img_l_lines = draw_lines(img_l, left_lines, thickness=10)

plt.subplot(121), plt.imshow(cv.cvtColor(img_l_lines, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(122), plt.imshow(cv.cvtColor(img_r_points, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()