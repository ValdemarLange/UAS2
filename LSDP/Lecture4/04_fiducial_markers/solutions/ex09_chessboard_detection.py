from pathlib import Path
import cv2
from corner_detector import ChessBoardCornerDetector
from icecream import ic


filename = "../input_camera_calibration/image_0520.jpg"
img = cv2.imread(filename)

cbcd = ChessBoardCornerDetector()
cbcd.distance_scale = 200
calibration_points, coverage, stats = cbcd.detect_chess_board_corners(
    img,
    debug=True,
    path_to_image=Path(filename),
    path_to_output_folder=Path("../output"),
)
corners, image_points = cbcd.extract_corners_and_image_points(calibration_points)

ic(corners.shape)
ic(image_points.shape)
ic(corners[0:5])
ic(image_points[0:5])
