import cv2
import numpy as np
from icecream import ic
import plotly.graph_objects as go


class TriangulatePoints:
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
            cv2.FM_RANSAC,
            confidence,
            ransacReprojecThreshold,
        )
        self.points1filtered = self.points1[self.essential_mask.ravel() == 1]
        self.points2filtered = self.points2[self.essential_mask.ravel() == 1]

    def decompose_essential_matrix(self):
        # Decompose the essential matrix to determine the rotation and
        # translation of the second camera relative to the first camera.
        retval, self.R, self.t, mask = cv2.recoverPose(
            self.essentialMatrix, self.points1, self.points2, self.cameraMatrix
        )
        ic(self.R)
        ic(self.t)

        # Create projection matrices from the estimated camera
        # positions
        self.null_projection_matrix = self.cameraMatrix @ np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        )
        self.projection_matrix = self.cameraMatrix @ np.hstack((self.R.T, self.t))

    def triangulate_points(self):
        self.points3d_reconstr = cv2.triangulatePoints(
            self.projection_matrix,
            self.null_projection_matrix,
            self.points1filtered.T,
            self.points2filtered.T,
        )
        # Convert back to unit value in the homogeneous part.
        self.points3d_reconstr /= self.points3d_reconstr[3, :]

    def create_interactive_3d_plot(self):
        transform = self.generate_rotation_transform()
        point3dtemp = transform @ self.points3d_reconstr

        xs = point3dtemp[0]
        ys = point3dtemp[1]
        zs = point3dtemp[2]

        formatted_colors = [f"rgb({R}, {G}, {B})" for R, G, B in self.kp_colors]
        plotlyfig = go.Figure(
            data=[
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=formatted_colors,  # set color to an array/list of desired values
                        colorscale="Viridis",  # choose a colorscale
                        opacity=0.8,
                    ),
                )
            ],
            layout={"title": "3D points from triangulation"},
        )

        # Show estimated camera positions
        t_rotated = transform @ np.vstack((self.t, 1))
        camera2x = t_rotated[0, 0]
        camera2y = t_rotated[1, 0]
        camera2z = t_rotated[2, 0]
        plotlyfig.add_trace(
            go.Scatter3d(
                x=[0, camera2x],
                y=[0, camera2y],
                z=[0, camera2z],
                mode="markers",
                marker=dict(size=10, color=["red", "green"]),
            )
        )

        xptp = np.ptp(np.hstack((xs, 0, camera2x)))
        yptp = np.ptp(np.hstack((ys, 0, camera2y)))
        zptp = np.ptp(np.hstack((zs, 0, camera2z)))
        plotlyfig.update_layout(
            scene=dict(
                aspectmode="data",
                aspectratio=go.layout.scene.Aspectratio(
                    x=xptp, y=xptp / yptp, z=zptp / xptp
                ),
            )
        )

        plotlyfig.show()

    def generate_rotation_transform(self):
        pitch = -45 / 180 * np.pi
        transform_1 = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(pitch), np.sin(pitch), 0],
                [0, -np.sin(pitch), np.cos(pitch), 0],
                [0, 0, 0, 1],
            ]
        )
        roll = 0 / 180 * np.pi
        transform_2 = np.array(
            [
                [np.cos(roll), 0, np.sin(roll), 0],
                [0, 1, 0, 0],
                [-np.sin(roll), 0, np.cos(roll), 0],
                [0, 0, 0, 1],
            ]
        )
        yaw = 0 / 180 * np.pi
        transform_3 = np.array(
            [
                [np.cos(yaw), np.sin(yaw), 0, 0],
                [-np.sin(yaw), np.cos(yaw), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        transform = transform_1 @ transform_2 @ transform_3
        return transform

    def main(self):
        self.load_images()
        self.set_camera_matrix()
        self.detect_and_match_sift_features()
        self.extract_point_pairs_from_matches()
        self.determine_essential_matrix_and_filter_points()
        self.decompose_essential_matrix()
        self.triangulate_points()
        self.create_interactive_3d_plot()


tp = TriangulatePoints()
tp.main()
