import cv2
import numpy as np


def main():
    filename = "../input/video_with_aruco_markers_dict_4x4_250.mov"
    cap = cv2.VideoCapture(filename)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Intrinsic camera matrix and distortion coefficients
    # Obtained by camera calibration
    cameraMatrix = np.array([[1913, 0, 965], [0, 1914, 550], [0, 0, 1]], dtype=float)
    distCoeffs = np.array([[0.30057, -2.08331, 0.00094, 0.002270, 4.3221]], dtype=float)

    # Analyse each frame individually
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break

        # Detect aruco markers
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)
        frame_markers = cv2.aruco.drawDetectedMarkers(
            frame.copy(), markerCorners, markerIds
        )

        rvecs, tvecs, _objPoints = my_estimatePoseSingleMarkers(
            markerCorners, 0.05, cameraMatrix, distCoeffs
        )

        for rvec, tvec in zip(rvecs, tvecs):
            frame_markers = cv2.drawFrameAxes(
                frame_markers, cameraMatrix, distCoeffs, rvec, tvec, 0.05
            )

        # Show annotated input image and the magnitude response image.
        cv2.imshow("frame", frame_markers)

        # Make it possible to stop the program by pressing 'q'.
        k = cv2.waitKey(30)
        if k == ord("q"):
            break
        if k == ord("p"):
            cv2.waitKey(100000)


# Apparently the method estimatePoseSingleMarkers have been removed from OpenCV
# here is it replaced with the solvePnP method instead.
# https://stackoverflow.com/a/76802895
def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    """
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    """
    marker_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(
            marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE
        )
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash


main()
