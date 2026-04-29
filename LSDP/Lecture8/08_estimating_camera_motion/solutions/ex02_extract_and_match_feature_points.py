import cv2
import numpy as np
from icecream import ic


class DrawRawMatches:
    def load_images(self):
        # Load images and convert them to grayscale
        self.img_image1 = cv2.imread("../input/my_photo-1.jpg")
        self.img_image2 = cv2.imread("../input/my_photo-2.jpg")
        self.gray_image1 = cv2.cvtColor(self.img_image1, cv2.COLOR_BGR2GRAY)
        self.gray_image2 = cv2.cvtColor(self.img_image2, cv2.COLOR_BGR2GRAY)

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

        self.match_indices = np.int32(match_indices_temp)

    def determine_fundamental_matrix(self):
        # Determine the essential matrix, handle outliers
        # using the ransac algorithm.
        ransacReprojecThreshold = 1
        confidence = 0.99
        self.fundamental_matrix, self.fundamental_mask = cv2.findFundamentalMat(
            self.points1,
            self.points2,
            cv2.FM_RANSAC,
            confidence=confidence,
            ransacReprojThreshold=ransacReprojecThreshold,
        )

    def draw_matches(self):
        # Draw the filtered matches.
        img3 = cv2.drawMatches(
            self.img_image1,
            self.kp_image1,
            self.img_image2,
            self.kp_image2,
            self.matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        cv2.imwrite(
            "../output/ex02_matching_features.png", img3
        )


    def main(self):
        self.load_images()
        self.detect_and_match_sift_features()
        self.extract_point_pairs_from_matches()
        self.determine_fundamental_matrix()
        self.draw_matches()
        ic(self.fundamental_matrix)


tp = DrawRawMatches()
tp.main()

