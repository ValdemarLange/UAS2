import cv2
import numpy as np
import glob
import OpenGL.GL as gl
import collections


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


def quarternion_to_rotation_matrix(q):
    """
    The formula for converting from a quarternion to a rotation
    matrix is taken from here:
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    """
    qw = q.w()
    qx = q.x()
    qy = q.y()
    qz = q.z()
    R11 = 1 - 2 * qy**2 - 2 * qz**2
    R12 = 2 * qx * qy - 2 * qz * qw
    R13 = 2 * qx * qz + 2 * qy * qw
    R21 = 2 * qx * qy + 2 * qz * qw
    R22 = 1 - 2 * qx**2 - 2 * qz**2
    R23 = 2 * qy * qz - 2 * qx * qw
    R31 = 2 * qx * qz - 2 * qy * qw
    R32 = 2 * qy * qz + 2 * qx * qw
    R33 = 1 - 2 * qx**2 - 2 * qy**2
    R = np.array([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])
    return R


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
            match for match in self.raw_matches if match.distance < 30
        ]

    def visualize_matches(self, matches):
        h, w, _ = self.frame1.image.shape
        # Place the images next to each other.
        vis = np.concatenate((self.frame1.image, self.frame2.image), axis=1)

        # Draw the matches
        for match in matches:
            start_coord = (int(match.keypoint1[0]), int(match.keypoint1[1]))
            end_coord = (int(match.keypoint2[0] + w), int(match.keypoint2[1]))
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
            self.null_projection_matrix = self.camera_matrix @ first_projection_matrix
        if second_projection_matrix is not None:
            self.projection_matrix = self.camera_matrix @ second_projection_matrix

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


def main():
    if True:
        print("Using the SIFT feature detector and descriptor")
        detector = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:
        print("Using the ORB feature detector and descriptor")
        detector = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    fg = FrameGenerator(detector)
    image1 = cv2.imread("../../08_estimating_camera_motion/input/my_photo-1.jpg")
    image2 = cv2.imread("../../08_estimating_camera_motion/input/my_photo-2.jpg")
    frame1 = fg.make_frame(image1)
    frame2 = fg.make_frame(image2)

    camera_matrix = np.array(
        [
            [704.48, 0.0, 637.42],
            [0.000000000000e00, 704.01349, 375.7176],
            [0.000000000000e00, 0.000000000000e00, 1.000000000000e00],
        ]
    )
    ip = ImagePair(frame1, frame2, bf, camera_matrix)

    ip.match_features()
    essential_matches = ip.determine_essential_matrix(ip.filtered_matches)
    ip.estimate_camera_movement(essential_matches)
    ip.reconstruct_3d_points(essential_matches)
    image_to_show = ip.visualize_matches(essential_matches)
    print(ip.points3d_reconstr.transpose())
    cv2.imshow("matches", image_to_show)
    cv2.waitKey(-1)


main()
