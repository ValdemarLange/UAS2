import cv2
import numpy as np
from icecream import ic


class RecoverCameraMovement:
    def load_images(self):
        # Load images and convert them to grayscale
        self.img_image1 = cv2.imread("../input/my_photo-1.jpg")
        self.img_image2 = cv2.imread("../input/my_photo-2.jpg")
        self.gray_image1 = cv2.cvtColor(self.img_image1, cv2.COLOR_BGR2GRAY)
        self.gray_image2 = cv2.cvtColor(self.img_image2, cv2.COLOR_BGR2GRAY)

    def set_camera_matrix(self):
        self.cameraMatrix = np.array(
            [
                [704, 0.0, 637],
                [0.0, 704, 376],
                [0.0, 0.0, 1.0],
            ]
        )

    def detect_and_match_sift_features(self):
        # Detect and match sift features
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.kp_image1, des_image1 = sift.detectAndCompute(self.gray_image1, None)
        self.kp_image2, des_image2 = sift.detectAndCompute(self.gray_image2, None)
        self.matches = bf.match(des_image1, des_image2)
        self.kp_colors = []
        for kp in self.kp_image1:
            point = kp.pt
            color = self.img_image1[int(point[1]), int(point[0])]
            self.kp_colors.append(color)

    def extract_point_pairs_from_matches(self):
        # Convert data structures
        points1_temp = []
        points2_temp = []
        match_indices_temp = []
        for idx, m in enumerate(self.matches):
            points1_temp.append(self.kp_image1[m.queryIdx].pt)
            points2_temp.append(self.kp_image2[m.trainIdx].pt)
            match_indices_temp.append(idx)

        # Convert points1 and point2 to floats.
        self.points1 = np.float32(points1_temp)
        self.points2 = np.float32(points2_temp)

    def determine_essential_matrix_and_filter_points(self):
        # Determine the essential matrix, handle outliers
        # using the ransac algorithm.
        ransacReprojecThreshold = 1
        confidence = 0.99
        self.essentialMatrix, self.essential_mask = cv2.findEssentialMat(
            self.points1,
            self.points2,
            self.cameraMatrix,
            method=cv2.FM_RANSAC,
            prob=confidence,
            threshold=ransacReprojecThreshold
        )
        self.points1filtered = self.points1[self.essential_mask.ravel() == 1]
        self.points2filtered = self.points2[self.essential_mask.ravel() == 1]

    def decompose_essential_matrix(self):
        # Decompose the essential matrix to determine the rotation and
        # translation of the second camera relative to the first camera.
        retval, self.R, self.t, mask = cv2.recoverPose(
            self.essentialMatrix, self.points1, self.points2, self.cameraMatrix
        )

    def output_camera_movement(self):
        # Output the determined values.
        ic(self.R)
        ic(self.t)

        # Verify that the determined rotation and translation
        # can be used to construct the essential matrix.
        fudge_factor = -1 / np.sqrt(2)
        tx = np.array(
            [
                [0, -self.t[2, 0], self.t[1, 0]],
                [self.t[2, 0], 0, -self.t[0, 0]],
                [-self.t[1, 0], self.t[0, 0], 0],
            ]
        )
        estimated_essential_matrix = tx @ self.R * fudge_factor

        # Output the essential matrix and the reconstruction.
        ic(estimated_essential_matrix)
        ic(self.essentialMatrix)

    def main(self):
        self.load_images()
        self.set_camera_matrix()
        self.detect_and_match_sift_features()
        self.extract_point_pairs_from_matches()
        self.determine_essential_matrix_and_filter_points()
        self.decompose_essential_matrix()
        self.output_camera_movement()


tp = RecoverCameraMovement()
tp.main()
