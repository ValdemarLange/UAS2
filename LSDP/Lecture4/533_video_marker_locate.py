from MarkerTracker import MarkerTracker
import cv2
import numpy as np

input = cv2.VideoCapture("04_fiducial_markers/input/video_with_n_fold_markers.mov")

# input = cv2.imread("04_fiducial_markers/input/hubsanwithmarker.jpg")

tracker = MarkerTracker(order=4, kernel_size=16, scale_factor=0.1)
# tracker.track_marker_with_missing_black_leg = False

while True:
    ret, img = input.read()
    if not ret:
        break
    img_small = cv2.resize(img, None, fx=0.3, fy=0.3)

    grayscale_image = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

    pose = tracker.locate_marker(grayscale_image)

    if pose is not None:
        img_small = cv2.circle(img_small, (int(pose.x), int(pose.y)), 3, (0, 0, 255), 30)

    cv2.imshow("Video", img_small)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


input.release()
cv2.destroyAllWindows()

