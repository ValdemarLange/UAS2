import cv2
import numpy as np


def locate_paper_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Segment using otsu
    _, segmented = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(
        segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find contour, with the largest area, that can be approximated by a four sided polygon.
    max_area = 0
    paper_corners = None
    allowed_error = 20
    closed = True
    for contour in contours:
        area = cv2.contourArea(contour)
        corners = cv2.approxPolyDP(contour, allowed_error, closed)
        if corners.shape == (4, 1, 2) and area > max_area:
            max_area = area
            paper_corners = corners

    return paper_corners


def correct_perspective(img):
    paper_corners = locate_paper_corners(img)

    if paper_corners is not None:
        print(paper_corners)

        # Approximate the ratio of the A4 paper, which is 1:sqrt(2)
        width = 400
        height = int(width * np.sqrt(2))
        corrected_image_corners = np.array(
            [[width, 0], [0, 0], [0, height], [width, height]]
        )

        # Determine perspective transform
        ret, _ = cv2.findHomography(paper_corners, corrected_image_corners)
        warped_image = cv2.warpPerspective(img, ret, (width, height))
        return warped_image

    return img


def main():
    filename = "../input/a4paper.jpg"
    img = cv2.imread(filename)
    warped_image = correct_perspective(img)
    cv2.imwrite("../output/ex13_corrected_perspective.jpg", warped_image)


def correct_a4paper_in_webcam_stream():
    cap = cv2.VideoCapture(2)

    cv2.namedWindow("frame")
    cv2.namedWindow("corrected")
    cv2.moveWindow("frame", 0, 0)
    cv2.moveWindow("corrected", 600, 0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        try:
            corrected = correct_perspective(frame)
        except Exception as e:
            print(e)
            corrected = frame

        # Display the resulting frame
        cv2.imshow("frame", frame)
        cv2.imshow("corrected", corrected)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()


# correct_a4paper_in_webcam_stream()
main()
