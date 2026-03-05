from MarkerLocator.MarkerTracker import MarkerTracker
import cv2
import numpy as np


def main():
    # Load image
    filename = "MarkerLocator/documentation/pythonpic/input/hubsanwithmarker.jpg"
    img = cv2.imread(filename)

    # Scale image down
    img_small = cv2.resize(img, None, fx=0.4, fy=0.4)

    # Write input image to the output folder
    cv2.imwrite("../output/ex02_input_image.jpg", img_small)

    # Instantiate the marker tracker
    tracker = MarkerTracker(order=4, kernel_size=45, scale_factor=0.25)
    tracker.track_marker_with_missing_black_leg = False

    # Convert image to grayscale and apply the marker tracker to the image
    grayscale_image = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    pose = tracker.locate_marker(grayscale_image)
    print("x: ", pose.x)
    print("y: ", pose.y)
    print("theta: ", pose.theta)
    print("quality: ", pose.quality)

    # Extract the marker response
    magnitude = np.sqrt(tracker.frame_sum_squared)

    # Save the marker response in the output diretory
    cv2.imwrite("../output/ex02_magnitude_image.jpg", magnitude / 5)


main()
