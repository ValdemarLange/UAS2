from corner_detector import ChessBoardCornerDetector
import cv2
from pathlib import Path
from icecream import ic
import numpy as np
from PinholeCameraModel import PinholeCameraModel
from LevenbergMarquardt import LevenbergMarquardt
from ProjectionVisualizer import ProjectionVisualizer


def double_image_main(initial_focal_length = 500):
    cbcd = ChessBoardCornerDetector()
    cbcd.distance_scale = 200
    filename1 = "../input_camera_calibration/image_0520.jpg"
    filename2 = "../input_camera_calibration/image_0660.jpg"
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    calibration_points1, coverage, stats = cbcd.detect_chess_board_corners(img1, 
                                    debug = True, 
                                    path_to_image=Path(filename1), 
                                    path_to_output_folder=Path("../output"))
    calibration_points2, coverage, stats = cbcd.detect_chess_board_corners(img2, 
                                    debug = True, 
                                    path_to_image=Path(filename2), 
                                    path_to_output_folder=Path("../output"))

    corners1, image_points1 = cbcd.extract_corners_and_image_points(calibration_points1)
    corners2, image_points2 = cbcd.extract_corners_and_image_points(calibration_points2)

    ic(corners1.shape)
    ic(image_points1.shape)
    ic(corners2.shape)
    ic(image_points2.shape)

    def K_from_cam_param(fx, fy, cx, cy): 
        return np.array([
                [fx, 0, cx], 
                [0, fy, cy], 
                [0, 0, 1]
                ])

    camera_model = PinholeCameraModel()

    pose = np.array([500, 800, 400, 0, 0, 0, 11, 12, 13, 0, 0, 0, 11, 12, 13])
    pose = np.array([initial_focal_length, 800, 400, 
                     0.03, 3.14, -1.57, 
                     3, -6, 5, 
                     0.08, 3.46, -3.27, 
                     5, -4, 60])

    camera_model.K = K_from_cam_param(pose[0], pose[0], pose[1], pose[2])

    def projection_function(pose):
        # The uncertainty have been adjusted to give a quality of fit around 0.5
        measurement_uncertainty = 0.39
        camera_model.K = K_from_cam_param(pose[0], pose[0], pose[1], pose[2])

        projections1 = camera_model.fProject(pose[3:9], corners1.transpose())
        observations1 = image_points1.transpose().reshape((1, -1))
        normalized_errors1 = (observations1 - projections1) / measurement_uncertainty

        projections2 = camera_model.fProject(pose[9:15], corners2.transpose())
        observations2 = image_points2.transpose().reshape((1, -1))
        normalized_errors2 = (observations2 - projections2) / measurement_uncertainty
        
        normalized_errors = np.hstack((normalized_errors1, normalized_errors2))
        return normalized_errors
    
    lm = LevenbergMarquardt(projection_function, pose)
    # ic(lm.residual_error)
    for k in range(175):
        lm.iterate()
        pose = lm.param
        camera_model.K = K_from_cam_param(pose[0], pose[0], pose[1], pose[2])

    ic(lm.residual_error)
    # ic(lm.param)
    # pv1 = ProjectionVisualizer(img1, filename1)
    # pv2 = ProjectionVisualizer(img2, filename2)
    # pv1.visualise_pose_estimate(camera_model.fProject(pose[3:9], corners1.transpose()), 
    #                         image_points1)
    # pv2.visualise_pose_estimate(camera_model.fProject(pose[9:15], corners2.transpose()), 
    #                         image_points2)

    # cv2.waitKey(0)
    lm.estimate_uncertainties()
    ic(lm.param[0:3])
    #ic(lm.scale_one_dim)
    #ic(lm.scale_multi_dim)
    # ic(lm.squared_residual_error)
    # ic(lm.goodness_of_fit)
    ic(lm.param_uncert[0:3])
    ic(lm.combined_uncertainties[0:3, 0:3])
    ic(lm.combined_uncertainties_vector[0:3])


# From two images, there is only one minima.
print("double_image_main(100)")
double_image_main(100)
print("double_image_main(1000)")
double_image_main(1000)
print("double_image_main(100000)")
double_image_main(100000)