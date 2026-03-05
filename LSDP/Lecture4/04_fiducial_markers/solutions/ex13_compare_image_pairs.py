from corner_detector import ChessBoardCornerDetector
import cv2
from pathlib import Path
from icecream import ic
import numpy as np
from PinholeCameraModel import PinholeCameraModel
from LevenbergMarquardt import LevenbergMarquardt
from ProjectionVisualizer import ProjectionVisualizer


def double_image_main(filename1, filename2, initial_focal_length = 500):
    cbcd = ChessBoardCornerDetector()
    cbcd.distance_scale = 200
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

    lm.estimate_uncertainties()
    # ic(lm.param[0:3])
    # ic(lm.param_uncert[0:3])
    # ic(lm.combined_uncertainties[0:3, 0:3])
    # ic(lm.combined_uncertainties_vector[0:3])
    cam_1_rotation_matrix = PinholeCameraModel.rotationMatrix(None, *pose[3:6])
    cam_2_rotation_matrix = PinholeCameraModel.rotationMatrix(None, *pose[9:12])
    rel_rotation = cam_1_rotation_matrix @ cam_2_rotation_matrix.transpose()
    rot_vector, _ = cv2.Rodrigues(rel_rotation)
    rotation = np.linalg.norm(rot_vector)

    return lm.param[0], lm.combined_uncertainties_vector[0], lm.param[1], lm.combined_uncertainties_vector[1], rotation


# From two images, there is only one minima.
# print("img1 img2")
# double_image_main("../input_camera_calibration/phone/img1.jpg", 
#                   "../input_camera_calibration/phone/img2.jpg")

images = [
    "../input_camera_calibration/phone/img1.jpg",    
    "../input_camera_calibration/phone/img2.jpg",    
    "../input_camera_calibration/phone/img3.jpg",    
    "../input_camera_calibration/phone/img4.jpg",    
    "../input_camera_calibration/phone/img5.jpg",    
    "../input_camera_calibration/phone/img6.jpg",    
    "../input_camera_calibration/phone/img7.jpg",    
    "../input_camera_calibration/phone/img8.jpg",
]

for img1 in images:
    for img2 in images:
        if img2 > img1: 
            f, f_uncert, cx, cxuncert, rotation = double_image_main(img1, img2)
            print(img1[-8:-4], img2[-8:-4], f, f_uncert, cx, cxuncert, rotation)
