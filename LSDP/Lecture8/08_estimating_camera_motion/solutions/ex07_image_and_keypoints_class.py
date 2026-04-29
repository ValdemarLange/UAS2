#!/usr/bin/env python
import cv2
import numpy as np
import math
import argparse
import plotly.graph_objects as go
from codetiming import Timer
from icecream import ic


# Code borrowed from the following sources
# https://docs.opencv.org/master/d1/d89/tutorial_py_orb.html
# https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
# https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
# https://www.morethantechnical.com/2016/10/17/structure-from-motion-toy-lib-upgrades-to-opencv-3/


def isRotationMatrix(R):
    # Checks if a matrix is a valid rotation matrix.
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    Imatrix = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(Imatrix - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    """Calculates rotation matrix to euler angles

    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).
    """

    assert isRotationMatrix(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


class ImageAndKeypoints:
    def __init__(self, detector_name):
        if detector_name == "ORB":
            # Use ORB features
            self.detector = cv2.ORB_create()
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            # Use SIFT features
            self.detector = cv2.SIFT_create()
            self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Camera parameters are taken from a manual calibration
        # of the used camera.
        # See the file ../input/camera_calibration_extended.txt
        self.cameraMatrix = np.array(
            [
                [704.48172143, 0.0, 637.4243092],
                [0.0, 704.01349597, 375.7176407],
                [0.0, 0.0, 1.0],
            ]
        )
        self.distCoeffs = np.array(
            [
                [
                    8.93382520e-02,
                    -1.57262105e-01,
                    -9.82974653e-05,
                    5.65668273e-04,
                    7.19784192e-02,
                ]
            ]
        )

        # Factor for rescaling images (usually used to downscale images)
        self.scale_factor = 1
        self.cameraMatrix *= self.scale_factor
        self.cameraMatrix[2, 2] = 1

    def set_image(self, image):
        width = int(image.shape[1] * self.scale_factor)
        height = int(image.shape[0] * self.scale_factor)
        dim = (width, height)

        self.image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def detect_keypoints(self):
        # find keypoints and descriptors with the selected feature detector
        self.keypoints, self.descriptors = self.detector.detectAndCompute(
            self.image, None
        )
        self.kp_colors = []
        for kp in self.keypoints:
            point = kp.pt
            color = self.image[int(point[1]), int(point[0])]
            self.kp_colors.append(color)

    def show_key_point_information(self):
        ic(len(self.keypoints))


class TriangulatePointsFromTwoImages:
    def __init__(self):
        pass

    @Timer(text="load_images {:.4f}")
    def load_images(self, filename_one, filename_two):
        # Load images
        self.img1 = cv2.imread(filename_one)
        self.img2 = cv2.imread(filename_two)

    def main(self, filename_one, filename_two):
        self.load_images(filename_one, filename_two)
        detector = "SIFT"

        image1 = ImageAndKeypoints(detector)
        image1.set_image(self.img1)
        image1.detect_keypoints()
        image1.show_key_point_information()

        image2 = ImageAndKeypoints(detector)
        image2.set_image(self.img2)
        image2.detect_keypoints()
        image2.show_key_point_information()

        return


# Parse command line args
parser = argparse.ArgumentParser()
parser.add_argument("image1", type=str, help="path to the first image")
parser.add_argument("image2", type=str, help="path to the second image")
args = parser.parse_args()
TPFTI = TriangulatePointsFromTwoImages()
TPFTI.main(args.image1, args.image2)
