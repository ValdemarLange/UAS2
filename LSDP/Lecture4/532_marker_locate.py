from MarkerTracker import MarkerTracker
import cv2
import numpy as np

input = cv2.imread("04_fiducial_markers/input/hubsanwithmarker.jpg")

img_small = cv2.resize(input, None, fx=0.1, fy=0.1)


tracker = MarkerTracker(order=4, kernel_size=91, scale_factor=0.01)
# tracker.track_marker_with_missing_black_leg = False


grayscale_image = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

pose = tracker.locate_marker(grayscale_image)

print("x: ",pose.x,"y: ",pose.y,"theta: ",pose.theta,"quality: ",pose.quality, "order: ", pose.order)

 # Extract the marker response
magnitude = np.sqrt(tracker.frame_sum_squared)
max_magnitude = np.max(magnitude)

# Save the marker response in the output diretory
output = magnitude / max_magnitude * 255

output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
output = cv2.circle(output, (int(pose.x), int(pose.y)), 3, (0, 0, 255))


cv2.imwrite("532_output.jpg", output)
