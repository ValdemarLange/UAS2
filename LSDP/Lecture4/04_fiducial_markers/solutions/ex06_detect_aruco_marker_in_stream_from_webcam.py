import cv2


def main():
    filename = 2
    cap = cv2.VideoCapture(filename)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Analyse each frame individually
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break

        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)
        print(markerCorners)

        frame_markers = cv2.aruco.drawDetectedMarkers(
            frame.copy(), markerCorners, markerIds
        )

        # Show annotated input image and the magnitude response image.
        cv2.imshow("frame", frame_markers)

        # Make it possible to stop the program by pressing 'q'
        # and pausing it by pressing 'p'.
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
        if k == ord("p"):
            cv2.waitKey(100000)


main()
