import cv2
import numpy as np
import collections

# import pangolin
import OpenGL.GL as gl
import ThreeDimViewer

# If you encounter an error like this
# libGL error: failed to load driver: iris
# libGL error: failed to load driver: swrast
#
# This stackoverflow post provides a solution
# https://stackoverflow.com/questions/71010343/cannot-load-swrast-and-iris-drivers-in-fedora-35/72200748#72200748
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6


Feature = collections.namedtuple("Feature", ["keypoint", "descriptor", "feature_id"])

Match = collections.namedtuple(
    "Match",
    [
        "featureid1",
        "featureid2",
        "keypoint1",
        "keypoint2",
        "descriptor1",
        "descriptor2",
        "distance",
        "color",
    ],
)

Match3D = collections.namedtuple(
    "Match3D",
    [
        "featureid1",
        "featureid2",
        "keypoint1",
        "keypoint2",
        "descriptor1",
        "descriptor2",
        "distance",
        "color",
        "point",
    ],
)

MatchWithMap = collections.namedtuple(
    "MatchWithMap",
    [
        "featureid1",
        "featureid2",
        "imagecoord",
        "mapcoord",
        "descriptor1",
        "descriptor2",
        "distance",
    ],
)


class FrameGenerator:
    def __init__(self, detector):
        self.next_image_counter = 0
        self.detector = detector

    def make_frame(self, image):
        """
        Create a frame by extracting features from the provided image.

        This method should only be called once for each image.
        Each of the extracted features will be assigned a unique
        id, whic will help with tracking of individual features
        later in the pipeline.
        """
        # Create a frame and assign it a unique id.
        frame = Frame(image)
        frame.id = self.next_image_counter
        self.next_image_counter += 1

        # Extract features
        frame.keypoints, frame.descriptors = self.detector.detectAndCompute(
            frame.image, None
        )
        enumerated_features = enumerate(zip(frame.keypoints, frame.descriptors))

        # Save features in a list with the following elements
        # keypoint, descriptor, feature_id
        # where the feature_id refers to the image id and the feature
        # number.
        frame.features = [
            Feature(keypoint, descriptor, (frame.id, idx))
            for (idx, (keypoint, descriptor)) in enumerated_features
        ]

        return frame


class Frame:
    """
    Class / structure for saving information about a single frame.
    """

    def __init__(self, image=None):
        self.image = image
        self.id = None
        self.keypoints = None
        self.descriptors = None
        self.features = None

    def __repr__(self):
        return repr("Frame %d" % (self.id))


class ImagePair:
    """
    Class for working with image pairs.
    """

    def __init__(self, frame1, frame2, matcher, camera_matrix):
        self.frame1 = frame1
        self.frame2 = frame2
        self.matcher = matcher
        self.camera_matrix = camera_matrix

    def match_features(self):
        temp = self.matcher.match(self.frame1.descriptors, self.frame2.descriptors)
        # Make a list with the following values
        # - feature 1 id
        # - feature 2 id
        # - image coordinate 1
        # - image coordinate 2
        # - match distance
        self.raw_matches = [
            Match(
                self.frame1.features[match.queryIdx].feature_id,
                self.frame2.features[match.trainIdx].feature_id,
                self.frame1.features[match.queryIdx].keypoint.pt,
                self.frame2.features[match.trainIdx].keypoint.pt,
                self.frame1.features[match.queryIdx].descriptor,
                self.frame2.features[match.trainIdx].descriptor,
                match.distance,
                np.random.random((3)),
            )
            for idx, match in enumerate(temp)
        ]

        # Perform a very crude filtering of the matches
        self.filtered_matches = [
            match for match in self.raw_matches if match.distance < 50
        ]

    def visualize_matches(self, matches):
        # Get width of image
        w = self.frame1.image.shape[1]
        # Place the images next to each other.
        vis = np.concatenate((self.frame1.image, self.frame2.image), axis=1)

        # Draw the matches
        for match in matches:
            start_coord = (int(match[2][0]), int(match[2][1]))
            end_coord = (int(match[3][0] + w), int(match[3][1]))
            thickness = 1
            color = list(match.color * 256)
            vis = cv2.line(vis, start_coord, end_coord, color, thickness)

        return vis

    def determine_essential_matrix(self, matches):
        points_in_frame_1, points_in_frame_2 = self.get_image_points(matches)

        confidence = 0.99
        ransacReprojecThreshold = 1
        self.essential_matrix, mask = cv2.findEssentialMat(
            points_in_frame_1,
            points_in_frame_2,
            self.camera_matrix,
            cv2.FM_RANSAC,
            confidence,
            ransacReprojecThreshold,
        )

        inlier_matches = [
            match for match, inlier in zip(matches, mask.ravel() == 1) if inlier
        ]

        return inlier_matches

    def get_image_points(self, matches):
        points_in_frame_1 = np.array(
            [match.keypoint1 for match in matches], dtype=np.float64
        )
        points_in_frame_2 = np.array(
            [match.keypoint2 for match in matches], dtype=np.float64
        )
        return points_in_frame_1, points_in_frame_2

    def estimate_camera_movement(self, matches):
        points_in_frame_1, points_in_frame_2 = self.get_image_points(matches)

        retval, self.R, self.t, mask = cv2.recoverPose(
            self.essential_matrix,
            points_in_frame_1,
            points_in_frame_2,
            self.camera_matrix,
        )

    def reconstruct_3d_points(
        self, matches, first_projection_matrix=None, second_projection_matrix=None
    ):
        identify_transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        estimated_transform = np.hstack((self.R.T, -self.R.T @ self.t))

        self.null_projection_matrix = self.camera_matrix @ identify_transform
        self.projection_matrix = self.camera_matrix @ estimated_transform

        if first_projection_matrix is not None:
            self.null_projection_matrix = first_projection_matrix
        if second_projection_matrix is not None:
            self.projection_matrix = second_projection_matrix

        points_in_frame_1, points_in_frame_2 = self.get_image_points(matches)

        self.points3d_reconstr = cv2.triangulatePoints(
            self.projection_matrix,
            self.null_projection_matrix,
            points_in_frame_1.T,
            points_in_frame_2.T,
        )

        # Convert back to unit value in the homogeneous part.
        self.points3d_reconstr /= self.points3d_reconstr[3, :]

        self.matches_with_3d_information = [
            Match3D(
                match.featureid1,
                match.featureid2,
                match.keypoint1,
                match.keypoint2,
                match.descriptor1,
                match.descriptor2,
                match.distance,
                match.color,
                (
                    self.points3d_reconstr[0, idx],
                    self.points3d_reconstr[1, idx],
                    self.points3d_reconstr[2, idx],
                ),
            )
            for idx, match in enumerate(matches)
        ]

    def set_points_to_draw(self, points, cameras):
        np_points = np.array([point.point for point in points])
        np_colors = np.array([point.color for point in points])
        np_poses = np.array([camera.pose() for camera in cameras])
        self.q.put((np_points, np_colors, np_poses))


class ThreeImages:
    def __init__(self):
        self.detector = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.frame_generator = FrameGenerator(self.detector)

        self.camera_matrix = np.array(
            [
                [704.48172143, 0.0, 637.4243092],
                [0.0, 704.01349597, 375.7176407],
                [0.0, 0.0, 1.0],
            ]
        )

    def load_images(self, filename_one, filename_two, filename_three):
        img1 = cv2.imread(filename_one)
        img2 = cv2.imread(filename_two)
        img3 = cv2.imread(filename_three)

        self.frame1 = self.frame_generator.make_frame(img1)
        self.frame2 = self.frame_generator.make_frame(img2)
        self.frame3 = self.frame_generator.make_frame(img3)

    def main(self):
        ip = ImagePair(self.frame1, self.frame2, self.bf, self.camera_matrix)

        ip.match_features()
        image_to_show = ip.visualize_matches(ip.filtered_matches)
        essential_matches = ip.determine_essential_matrix(ip.filtered_matches)
        ip.estimate_camera_movement(essential_matches)
        ip.reconstruct_3d_points(essential_matches)
        # print(ip.matches_with_3d_information)

        descriptors_from_current_frame = [
            feature.descriptor for feature in self.frame3.features
        ]

        descriptors_in_map = [
            point.descriptor1 for point in ip.matches_with_3d_information
        ]

        temp = self.bf.match(
            np.array(descriptors_from_current_frame), np.array(descriptors_in_map)
        )

        print("Matches with map")
        print(len(temp))
        matches_with_map = []
        for match in temp:
            image_feature = self.frame3.features[match.queryIdx]
            map_feature = ip.matches_with_3d_information[match.trainIdx]
            t = MatchWithMap(
                image_feature.feature_id,
                map_feature.featureid2,
                image_feature.keypoint.pt,
                map_feature.point,
                image_feature.descriptor,
                map_feature.descriptor2,
                match.distance,
            )
            if match.distance < 60:
                matches_with_map.append(t)
                # print(t)

        print("Filtered matches with map")
        print(len(matches_with_map))
        image_coords = [match.imagecoord for match in matches_with_map]

        map_coords = [match.mapcoord for match in matches_with_map]

        try:
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.array(map_coords),
                np.array(image_coords),
                self.camera_matrix,
                np.zeros(4),
            )

            if retval:
                R, _ = cv2.Rodrigues(rvec.T[0])
                print("Estimated camera location")
                print(R)
                print(tvec.T[0])
        except:
            pass

        viewport = ThreeDimViewer.ThreeDimViewer()
        viewport.vertices = [point for point in map_coords]
        viewport.colors = [(255, 255, 255) for point in map_coords]

        viewport.cameras = []
        viewport.main()

        cv2.imshow("matches", image_to_show)
        cv2.waitKey(1000000)


ti = ThreeImages()
ti.load_images(
    "../../08_estimating_camera_motion/input/my_photo-1.jpg",
    "../../08_estimating_camera_motion/input/my_photo-2.jpg",
    "../../08_estimating_camera_motion/input/my_photo-3.jpg",
)
ti.main()
