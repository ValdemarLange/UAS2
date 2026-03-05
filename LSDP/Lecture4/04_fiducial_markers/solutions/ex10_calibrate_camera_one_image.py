from corner_detector import ChessBoardCornerDetector
import cv2
from pathlib import Path
from icecream import ic
import numpy as np
from PinholeCameraModel import PinholeCameraModel
from LevenbergMarquardt import LevenbergMarquardt
from ProjectionVisualizer import ProjectionVisualizer


def single_image_main(initial_focal_length=500):
    cbcd = ChessBoardCornerDetector()
    cbcd.distance_scale = 200
    filename = "../input_camera_calibration/image_0520.jpg"
    img = cv2.imread(filename)
    calibration_points, coverage, stats = cbcd.detect_chess_board_corners(
        img,
        debug=True,
        path_to_image=Path(filename),
        path_to_output_folder=Path("../output"),
    )
    # ic(calibration_points)
    # ic(coverage)
    # ic(stats)

    corners, image_points = cbcd.extract_corners_and_image_points(calibration_points)

    # ic(corners.shape)
    # ic(image_points.shape)

    def K_from_cam_param(fx, fy, cx, cy):
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    camera_model = PinholeCameraModel()

    pose = np.array([initial_focal_length, 800, 400, 0, 0, 0, 0, 0, 20])
    camera_model.K = K_from_cam_param(pose[0], pose[0], pose[1], pose[2])
    projections = camera_model.fProject(pose[3:9], corners.transpose())
    ic(projections.shape)

    def projection_function(pose):
        measurement_uncertainty = 1.5
        measurement_uncertainty = 0.39
        camera_model.K = K_from_cam_param(pose[0], pose[0], pose[1], pose[2])
        projections = camera_model.fProject(pose[3:9], corners.transpose())
        observations = image_points.transpose().reshape((1, -1))
        return (observations - projections) / measurement_uncertainty

    # pv = ProjectionVisualizer(img, filename)
    lm = LevenbergMarquardt(projection_function, pose)
    for k in range(55):
        lm.iterate()
        pose = lm.param
        camera_model.K = K_from_cam_param(pose[0], pose[0], pose[1], pose[2])
        # pv.visualise_pose_estimate(camera_model.fProject(pose[3:9], corners.transpose()),
        #                         image_points)
    ic(lm.residual_error)
    ic(lm.param)
    ic(lm.param[0:3])
    lm.estimate_uncertainties()
    ic(lm.param_uncert[0:3])
    ic(lm.combined_uncertainties[0:3, 0:3])
    ic(lm.combined_uncertainties_vector[0:3])


# From a single image, the determined focal length
# depends a lot on the initial guess, but the squared error
# is the same in all three cases.
# This indicates that there are multiple local minima or a
# valley in the loss function.
print("single_image_main(10)")
single_image_main(10)
print("single_image_main(1000)")
single_image_main(1000)
print("single_image_main(100000)")
single_image_main(100000)
