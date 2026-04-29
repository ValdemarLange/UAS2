import cv2
import argparse
from pathlib import Path
from icecream import ic
import os


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video")
    parser.add_argument(
        "input", type=lambda p: Path(p).absolute(), help="input video file"
    )
    parser.add_argument(
        "output", type=lambda p: Path(p).absolute(), help="output directory"
    )
    parser.add_argument("step", type=int, help="step between frames")

    args = parser.parse_args()

    print(args.input)
    print(args.output)
    print(args.step)

    cap = cv2.VideoCapture(str(args.input))
    frame_counter = 0

    try:
        os.makedirs(args.output)
    except FileExistsError:
        pass
    while cap.isOpened():
        ret_val, frame = cap.read()
        if not ret_val:
            fake_frame_skip = 1000
            while not ret_val and fake_frame_skip > 0:
                fake_frame_skip -= 1
                ret_val, frame = cap.read()
            if not ret_val:
                break

        frame_counter += 1

        if frame_counter % args.step == 0:
            ic(frame_counter)
            filename = "%s/frame%05d.jpg" % (args.output, frame_counter)
            cv2.imwrite(filename, frame)

        if ret_val is False:
            break

        cv2.imshow("frame", frame)

        k = cv2.waitKey(3)
        if k == ord("q"):
            break


main()
