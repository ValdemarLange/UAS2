import cv2
import numpy as np
from icecream import ic


def main():
    filename = "../input/video_with_aruco_markers_dict_4x4_250.mov"
    filename = "../input/2021-03-15 21.06.47.mov"
    cap = cv2.VideoCapture(filename)

    # Load dictionary of aruco markers
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    markerLength = 60
    # Instantiate an aruco board that matches the one in the obtained video.
    board = cv2.aruco.GridBoard(
        size=(6, 2),
        markerLength=markerLength,
        markerSeparation=20,
        dictionary=dictionary,
    )

    # Intrinsic camera matrix and distortion coefficients
    # Obtained by camera calibration
    cameraMatrix = np.array([[1913, 0, 965], [0, 1914, 550], [0, 0, 1]], dtype=float)
    distCoeffs = np.array([0.30057, -2.08331, 0.00094, 0.002270], dtype=float)

    cameraMatrix = np.array(
        [
            [1.91047117e03, 0.00000000e00, 9.41577607e02],
            [0.00000000e00, 1.91430187e03, 5.52445587e02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
        dtype=float,
    )
    distCoeffs = np.array(
        [1.94579481e-01, -1.31485089e00, 7.31609141e-04, 5.65056352e-04, 2.34190615e00],
        dtype=float,
    )

    # Analyse each frame individually
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break

        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)
        detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = (
            detector.refineDetectedMarkers(
                frame,
                board,
                markerCorners,
                markerIds,
                rejectedCandidates,
                cameraMatrix,
                distCoeffs,
            )
        )

        frame_markers = cv2.aruco.drawDetectedMarkers(
            frame.copy(), detectedCorners, detectedIds
        )

        rvecs, tvecs, _objPoints = my_estimatePoseSingleMarkers(
            detectedCorners, markerLength, cameraMatrix, distCoeffs
        )
        ic(rvecs)

        if len(rvecs) > 0:
            for rvec, tvec in zip(rvecs, tvecs):
                frame_markers = cv2.drawFrameAxes(
                    frame_markers, cameraMatrix, distCoeffs, rvec, tvec, 30
                )

            # Estimate board pose
            obj_points, img_points = board.matchImagePoints(
                detectedCorners, detectedIds
            )

            if obj_points is not None:
                # Hack for making the z-axis of the entire aruco board appear 
                # to go out of the board / screen.
                obj_points = obj_points @ np.diag([1, -1, 1])

                ret, rvec2, tvec2 = cv2.solvePnP(
                    obj_points, img_points, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_IPPE, rvec=rvec, tvec = tvec
                )

                frame_markers = cv2.drawFrameAxes(
                    frame_markers,
                    cameraMatrix=cameraMatrix,
                    distCoeffs=distCoeffs,
                    rvec=rvec2,
                    tvec=tvec2,
                    length=2 * markerLength,
                    thickness=5,
                )

        # Show annotated input image and the magnitude response image.
        cv2.imshow("frame", frame_markers)

        # Make it possible to stop the program by pressing 'q'
        # and pausing it by pressing 'p'.
        k = cv2.waitKey(100)
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
        #ic(marker_points, c)
        #ic(R, t)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash


main()
