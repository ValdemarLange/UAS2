import cv2
import numpy as np
from icecream import ic


class VideoStabilizer:
    def __init__(self):
        self.has_seen_first_frame = False
        self.method = "orb"

        # Initiate STAR detector
        self.orb = cv2.ORB_create()

        # Initiate FAST object with default values
        self.fast = cv2.FastFeatureDetector_create(nonmaxSuppression=True, threshold=90)

        # Initiate AKAZE object with default values
        self.akaze = cv2.AKAZE_create(max_points=1000)

        # create BFMatcher object
        self.matcher = cv2.DescriptorMatcher_create(
            cv2.DescriptorMatcher_BRUTEFORCE_HAMMING
        )

    def determine_homography_to_reference_frame(self, frame):
        pass

    def stabilize_frame(self, frame):
        if not self.has_seen_first_frame:
            self.deal_with_first_frame(frame)

        self.detect_features(frame)

        # Match descriptors and apply the Lowe filter
        matches = list(self.matcher.knnMatch(self.des, self.des_first_frame, 2))

        # It is possible to apply the Lowe filter here, try to see if it makes a difference.
        # matches = self.lowe_filtering(matches)

        self.determine_source_and_destination_coordinates(matches)

        # There are two methods for locating the homography, RANSAC and LMEDS. 
        # Try both and see if it makes a difference.
        # M, mask = cv2.findHomography(self.src_pts, self.dst_pts, cv2.RANSAC, 5.0)
        M, mask = cv2.findHomography(self.src_pts, self.dst_pts, cv2.LMEDS, 1.0)

        frame = cv2.warpPerspective(frame, M, dsize=(frame.shape[1], frame.shape[0]))

        return frame

    def determine_source_and_destination_coordinates(self, matches):
        self.src_pts = []
        self.dst_pts = []
        for m, n in matches:
                self.src_pts.append(self.kp[m.queryIdx].pt)
                self.dst_pts.append(self.kp_first_frame[m.trainIdx].pt)
        self.src_pts = np.array(self.src_pts)
        self.dst_pts = np.array(self.dst_pts)

    def lowe_filtering(self, matches):
        filtered_matches = []
        nn_match_ratio = 0.8  # Nearest neighbor matching ratio
        for m, n in matches:
            if m.distance < nn_match_ratio * n.distance:
                filtered_matches.append((m, n))
        return filtered_matches

    def detect_features(self, frame):
        if self.method == "orb":
            self.kp = self.fast.detect(frame, None)
            self.kp, self.des = self.orb.compute(frame, self.kp)
        else:
            self.kp, self.des = self.akaze.detectAndCompute(frame, None)

    def deal_with_first_frame(self, frame):
        cv2.imwrite("../output/ex14_referenceframe.png", frame)
        self.first_frame = frame
        if self.method == "orb":
            self.kp_first_frame = self.fast.detect(frame, None)
            self.kp_first_frame, self.des_first_frame = self.orb.compute(
                    frame, self.kp_first_frame
                )
        else:
            self.kp_first_frame, self.des_first_frame = self.akaze.detectAndCompute(
                    frame, None
                )
        self.has_seen_first_frame = True


class FrameIterator:
    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)
        self.vs = VideoStabilizer()

    def frame_generator(self):
        # Define a generator that yields frames from the video
        while 1:
            ret, frame = self.cap.read()
            if ret is not True:
                break
            yield frame
        self.cap.release()

    def main(self):
        for frame in self.frame_generator():
            stabilized_frame = self.vs.stabilize_frame(frame)

            # Mark a point in the frame (corner of a large building) 
            # to see how well the stabilization works
            cv2.circle(frame, (1828, 49), 20, (255, 255, 255), 2)
            cv2.circle(stabilized_frame, (1828, 49), 20, (255, 255, 255), 2)
            cv2.imshow("frame", frame)
            cv2.imshow("frame_stabilized", stabilized_frame)

            # Deal with key presses
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("p"):
                cv2.waitKey(100000)
            elif k == ord("s"):
                cv2.imwrite("../output/ex14_stillimage.png", frame)


fi = FrameIterator(
    "../input/2016-06-24 Krydset Sdr Boulevard Kløvermosevej.mkv"
)
fi.main()
