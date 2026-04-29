import cv2
import numpy as np


def main():
    filename = "../input/blackboard_red.jpg"
    # The location of the blackboard corners was determined in gimp
    # Upper left, upper right, lower right, lower left
    blackboard_corners = np.array([[66, 287], [1386, 71], [1377, 861], [56, 632]])
    output_width = 2000
    output_height = 500
    corrected_image_corners = np.array(
        [[0, 0], [output_width, 0], [output_width, output_height], [0, output_height]]
    )

    img = cv2.imread(filename)
    img_for_annotation = img.copy()

    # Check that the points are placed correctly in the
    # image.
    for idx in range(4):
        cv2.circle(
            img_for_annotation, tuple(blackboard_corners[idx, :]), 10, (255, 0, 0), 2
        )
    cv2.imshow("img", img_for_annotation)
    cv2.waitKey(-1)

    # Determine perspective transform
    ret, mask = cv2.findHomography(blackboard_corners, corrected_image_corners)
    print(ret)
    warped_image = cv2.warpPerspective(img, ret, (output_width, output_height))

    cv2.imwrite("../output/ex12_corrected_perspective.jpg", warped_image)


main()
