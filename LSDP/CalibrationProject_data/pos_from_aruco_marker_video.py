import cv2
import numpy as np
import csv

def main():
    input = cv2.VideoCapture("dict_4x4_250_01.mov")

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    cameraMatrix = np.array([[1634.91342176, 0, 949.66136209], [0, 1633.08633252, 539.77849754], [0, 0, 1]], dtype=float)
    distCoeffs = np.array([[1.55546583e-01, -5.09095214e-01,  2.87703175e-03,  3.36845996e-05, 5.95306801e-01]], dtype=float)

    i = 0

    translation = []

    while True:
        ret, img = input.read()
        if not ret:
            break

        corners, ids, rejected = detector.detectMarkers(img)
            
        frame_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)

        rvecs, tvecs, _objPoints = my_estimatePoseSingleMarkers(
            corners, 0.1, cameraMatrix, distCoeffs
        )
        try:
            for rvec, tvec, id in zip(rvecs, tvecs, ids):
                if id == 1:
                    frame_markers = cv2.drawFrameAxes(
                        frame_markers, cameraMatrix, distCoeffs, rvec, tvec, 0.1
                    )

                    R, _ = cv2.Rodrigues(rvec)
                    
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = tvec.flatten()
                    
                    T_inv = np.linalg.inv(T)
                    trans_vec = T_inv[0:3, 3]
                    translation.append(trans_vec.flatten())


        except Exception as e:
            print(e)
            
        # rot_mat = cv2.Rodrigues(rvecs[0])


        # if i == 10:
            # print("rvec", rvecs)
            # print(rot_mat)
            # print("rvec", rvecs)
            # print("tvec", tvecs)
            # print(_objPoints)

        i += 1

        cv2.imshow("Video", frame_markers)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    input.release()
    cv2.destroyAllWindows()

    with open('translation_after_rot.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tx", "ty", "tz"])
        writer.writerows(translation)

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
