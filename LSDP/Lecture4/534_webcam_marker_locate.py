from MarkerTracker import MarkerTracker
import cv2
import numpy as np

input = cv2.VideoCapture(0)

# input = cv2.imread("04_fiducial_markers/input/hubsanwithmarker.jpg")

tracker = MarkerTracker(order=4, kernel_size=25, scale_factor=0.1)
# tracker.track_marker_with_missing_black_leg = False

while True:
    ret, img = input.read()
    if not ret:
        break
    img_small = cv2.resize(img, None, fx=0.5, fy=0.5)

    grayscale_image = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

    pose = tracker.locate_marker(grayscale_image)

    if pose is not None:
        img_small = cv2.circle(img_small, (int(pose.x), int(pose.y)), 3, (0, 0, 255))

    cv2.imshow("Video", img_small)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


input.release()
cv2.destroyAllWindows()

# print("x: ",pose.x,"y: ",pose.y,"theta: ",pose.theta,"quality: ",pose.quality, "order: ", pose.order)

 # Extract the marker response
# magnitude = np.sqrt(tracker.frame_sum_squared)
# max_magnitude = np.max(magnitude)

# # Save the marker response in the output diretory
# output = magnitude / max_magnitude * 255

# output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
# output = cv2.circle(output, (int(pose.x), int(pose.y)), 3, (0, 0, 255))


# cv2.imwrite("532_output.jpg", output)
