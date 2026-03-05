import cv2


def main():
    # Load one of the built in aruco dictionaries
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    # Generate the marker
    marker_id = 7
    marker_size = 200
    marker_image = cv2.aruco.generateImageMarker(
        aruco_dict, marker_id, marker_size, None, 1
    )

    # Write the image to the output directory
    cv2.imwrite("../output/ex05_arucomarker_DICT_4x4_7.png", marker_image)


main()
