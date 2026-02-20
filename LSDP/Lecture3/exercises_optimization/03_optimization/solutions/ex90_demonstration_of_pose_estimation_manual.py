import numpy as np
import cv2

def main():
    cap = cv2.VideoCapture(1)

    cv2.namedWindow("frame")
    cv2.moveWindow("frame", 0, 0)
    cv2.namedWindow("reference")
    cv2.moveWindow("reference", 700, 0)
    cv2.namedWindow("ghost")
    cv2.moveWindow("ghost", 700, 400)
    reference_img = None
    while True:
        ret, frame = cap.read()
        if reference_img is None:
            reference_img = frame

        cv2.imshow("frame", frame)
        cv2.imshow("reference", reference_img)
        cv2.imshow("ghost", cv2.addWeighted(frame, 0.5, reference_img, 0.5, gamma = 0))
        k = cv2.waitKey(10)
        if k == ord("r"):
            reference_img = frame
        if k == ord("q"):
            break

main()