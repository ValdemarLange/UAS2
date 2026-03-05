from MarkerLocator.MarkerTracker import MarkerTracker
import cv2
import numpy as np


def main():
    filename = "../input/video_with_n_fold_markers.mov"
    cap = cv2.VideoCapture(filename)

    # Open and move windows
    cv2.namedWindow("frame")
    cv2.namedWindow("output/ex02_magnitude_image.jpg")
    cv2.moveWindow("output/ex02_magnitude_image.jpg", 900, 0)

    # Initialize tracker
    tracker = MarkerTracker(order=5, kernel_size=25, scale_factor=0.1)
    tracker.track_marker_with_missing_black_leg = False

    # Analyse each frame individually
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break

        # Scale down the image and convert it to grayscale
        frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
        grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Locate the marker
        pose = tracker.locate_marker(grayscale_image)

        # Extract the marker response from the tracker object
        magnitude = np.sqrt(tracker.frame_sum_squared)

        # Visualise the location of the located marker and indicate the quality
        # of the detected marker by altering the line color.
        # Red: bad marker quality
        # Green: high marker quality
        color = (0, int(255 * pose.quality), 255 - int(255 * pose.quality))
        cv2.line(frame, (0, 0), (int(pose.x), int(pose.y)), color, 2)

        # Show annotated input image and the magnitude response image.
        cv2.imshow("output/ex02_magnitude_image.jpg", magnitude / 315)
        cv2.imshow("frame", frame)

        # Make it possible to stop the program by pressing 'q'.
        k = cv2.waitKey(30)
        if k == ord("q"):
            break


main()
