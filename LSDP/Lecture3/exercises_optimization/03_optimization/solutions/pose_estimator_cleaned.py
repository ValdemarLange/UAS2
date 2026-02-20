from math import cos, sin
import numpy as np
from icecream import ic
from typing import Callable, Tuple
import cv2
from scipy.stats import chi2

"""

    idx cx cy x y z
    0,1355,950, 8, 0, 4
    1,1216,920, 7, 0, 4
    2,1359,879, 8, 1, 4
    3,1386,872, 8, 0, 3
    4,1219,854, 7, 1, 4
    5,1388,806, 8, 1, 3
    6,511,772, 1, 0, 4
    7,417,753, 0, 0, 4
    8,1435,742, 8, 0, 1
    9,1316,724, 7, 0, 1
    10,501,708, 1, 1, 4
    11,1199,705, 6, 0, 1
    12,407,691, 0, 1, 4
    13,1090,686, 5, 0, 1
    14,1439,678, 8, 1, 1
    15,642,670, 1, 0, 2
    16,985,669, 4, 0, 1
    17,1317,661, 7, 1, 1
    18,572,657, 1, 1, 3
    19,887,652, 3, 0, 1
    20,1199,643, 6, 1, 1
    21,492,643, 1, 2, 4
    22,792,637, 2, 0, 1
    23,396,627, 0, 2, 4
    24,1089,625, 5, 1, 1
    25,701,622, 1, 0, 1
    26,1444,612, 8, 2, 1
    27,984,610, 4, 1, 1
    28,635,610, 1, 1, 2
    29,1320,595, 7, 2, 1
    30,884,593, 3, 1, 1
    31,563,593, 1, 2, 3
    32,788,580, 2, 1, 1
    33,481,576, 1, 3, 4
    34,696,565, 1, 1, 1
    35,384,560, 0, 3, 4
    36,629,551, 1, 2, 2
    37,1448,543, 8, 3, 1
    38,556,529, 1, 3, 3
    39,1322,527, 7, 3, 1
    40,690,509, 1, 2, 1
    41,471,507, 1, 4, 4
    42,372,491, 0, 4, 4
    43,621,487, 1, 3, 2
    44,1453,472, 8, 4, 1
    45,547,464, 1, 4, 3
    46,1326,458, 7, 4, 1
    47,685,448, 1, 3, 1
    48,1203,445, 6, 4, 1
    49,1085,434, 5, 4, 1
    50,461,433, 1, 5, 4
    51,616,425, 1, 4, 2
    52,977,421, 4, 4, 1
    53,360,420, 0, 5, 4
    54,873,411, 3, 4, 1
    55,774,400, 2, 4, 1
    56,1459,399, 8, 5, 1
    57,538,394, 1, 5, 3
    58,676,389, 1, 4, 1
    59,1329,386, 7, 5, 1
    60,1204,376, 6, 5, 1
    61,1087,364, 5, 5, 1
    62,449,360, 1, 6, 4
    63,607,359, 1, 5, 2 
    64,975,354, 4, 5, 1
    65,347,346, 0, 6, 4
    66,869,345, 3, 5, 1
    67,769,336, 2, 5, 1
    68,669,326, 1, 5, 1
    69,528,323, 1, 6, 3
    70,1465,321, 8, 6, 1
    71,1333,311, 7, 6, 1
    72,1206,303, 6, 6, 1
    73,1086,294, 5, 6, 1
    74,599,291, 1, 6, 2
    75,973,286, 4, 6, 1
    76,438,281, 1, 7, 4
    77,865,277, 3, 6, 1
    78,763,269, 2, 6, 1
    79,335,269, 0, 7, 4
    80,663,261, 1, 6, 1
    81,520,247, 1, 7, 3
    82,1470,242, 8, 7, 1
    83,1336,233, 7, 7, 1
    84,1207,227, 6, 7, 1
    85,1086,222, 5, 7, 1
    86,592,219, 1, 7, 2
    87,970,214, 4, 7, 1
    88,860,206, 3, 7, 1
    89,428,203, 1, 8, 4
    90,757,199, 2, 7, 1
    91,656,194, 1, 7, 1
    92,322,191, 0, 8, 4
    93,511,176, 1, 8, 3
    94,1476,163, 8, 8, 1
    95,1340,158, 7, 8, 1
    96,1086,148, 5, 8, 1
    97,856,136, 3, 8, 1
    98,650,125, 1, 8, 1
"""


class WireframeModel():
    def __init__(self):
        corners_raw = np.array([
            [8,0,4,1],
            [7,0,4,1],
            [8,1,4,1],
            [8,0,3,1],
            [7,1,4,1],
            [8,1,3,1],
            [1,0,4,1],
            [0,0,4,1],
            [8,0,1,1],
            [7,0,1,1],
            [1,1,4,1],
            [6,0,1,1],
            [0,1,4,1],
            [5,0,1,1],
            [8,1,1,1],
            [1,0,2,1],
            [4,0,1,1],
            [7,1,1,1],
            [1,1,3,1],
            [3,0,1,1],
            [6,1,1,1],
            [1,2,4,1],
            [2,0,1,1],
            [0,2,4,1],
            [5,1,1,1],
            [1,0,1,1],
            [8,2,1,1],
            [4,1,1,1],
            [1,1,2,1],
            [7,2,1,1],
            [3,1,1,1],
            [1,2,3,1],
            [2,1,1,1],
            [1,3,4,1],
            [1,1,1,1],
            [0,3,4,1],
            [1,2,2,1],
            [8,3,1,1],
            [1,3,3,1],
            [7,3,1,1],
            [1,2,1,1],
            [1,4,4,1],
            [0,4,4,1],
            [1,3,2,1],
            [8,4,1,1],
            [1,4,3,1],
            [7,4,1,1],
            [1,3,1,1],
            [6,4,1,1],
            [5,4,1,1],
            [1,5,4,1],
            [1,4,2,1],
            [4,4,1,1],
            [0,5,4,1],
            [3,4,1,1],
            [2,4,1,1],
            [8,5,1,1],
            [1,5,3,1],
            [1,4,1,1],
            [7,5,1,1],
            [6,5,1,1],
            [5,5,1,1],
            [1,6,4,1],
            [1,5,2,1],
            [4,5,1,1],
            [0,6,4,1],
            [3,5,1,1],
            [2,5,1,1],
            [1,5,1,1],
            [1,6,3,1],
            [8,6,1,1],
            [7,6,1,1],
            [6,6,1,1],
            [5,6,1,1],
            [1,6,2,1],
            [4,6,1,1],
            [1,7,4,1],
            [3,6,1,1],
            [2,6,1,1],
            [0,7,4,1],
            [1,6,1,1],
            [1,7,3,1],
            [8,7,1,1],
            [7,7,1,1],
            [6,7,1,1],
            [5,7,1,1],
            [1,7,2,1],
            [4,7,1,1],
            [3,7,1,1],
            [1,8,4,1],
            [2,7,1,1],
            [1,7,1,1],
            [0,8,4,1],
            [1,8,3,1],
            [8,8,1,1],
            [7,8,1,1],
            [5,8,1,1],
            [3,8,1,1],
            [1,8,1,1],
            ])
        corners = (corners_raw @ np.diag([3.15, 1.9, 3.15, 1])).transpose()

        self.points = corners
        self.connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)]


class ObservedPoints():
    def __init__(self, image_points):
        self.image_points = image_points


class PoseEstimatorOrig():
    def __init__(self):
        self.parameters_to_optimize = [True, True, True, True, True, True]
        self.residual_error = 100000
        self.damping = 10
        self.wireframe_model = None
        self.img_orig = None


    def set_parameters_to_optimize(self, input_param_to_optimize):
        self.parameters_to_optimize = input_param_to_optimize


    def rotationMatrix(self, ax, ay, az):
        Rx = np.array([[1, 0, 0], [0, cos(ax), -sin(ax)], [0, sin(ax), cos(ax)]])
        Ry = np.array([[cos(ay), 0, sin(ay)], [0, 1, 0], [-sin(ay), 0, cos(ay)]])
        Rz = np.array([[cos(az), -sin(az), 0], [sin(az), cos(az), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        return R


    def externalMatrix(self, x):
        ax = x[0]
        ay = x[1]
        az = x[2]
        tx = x[3]
        ty = x[4]
        tz = x[5]

        R = self.rotationMatrix(ax, ay, az)
        Mext = np.concatenate((R, np.array([[tx], [ty], [tz]])), 1)
        return Mext


    def fProject(self, x, P_M):
        # Function structure from https://www.youtube.com/watch?v=kq3c6QpcAGc
        Mext = self.externalMatrix(x)
            
        ph = self.K @ Mext @ P_M
        ph[0, :] = np.true_divide(ph[0, :], ph[2, :])
        ph[1, :] = np.true_divide(ph[1, :], ph[2, :])
        ph = ph[0:2, :]

        ph = ph.reshape((1, -1))

        return ph
                    

    def visualize_image_points(self):
        prev_position = None
        for point in self.image_points:
            position = (int(point[0]), int(point[1]))
            if prev_position is not None:
                pass # cv2.line(self.img, position, prev_position, (255, 0, 0), 1)
            prev_position = position


    def visualise_projected_points(self, projected_points):
        prev_position = None
        for point in projected_points.reshape((2, -1)).transpose():
            position = (int(point[0]), int(point[1]))
            cv2.circle(self.img, position, 3, (0, 0, 255), -1)
            if prev_position is not None:
                pass
                # cv2.line(self.img, position, prev_position, (255, 255, 0), 1)
            prev_position = position


    def visualise_wireframe_model(self):
        projected_points = self.fProject(self.pose, self.wireframe_model.points)
        reshaped_projected_points = projected_points.reshape((2, -1)).transpose()
        for (idx_from, idx_to) in self.wireframe_model.connections:
            try:
                point_from = reshaped_projected_points[idx_from]
                point_to = reshaped_projected_points[idx_to]
                position_from = (int(point_from[0]), int(point_from[1]))
                position_to = (int(point_to[0]), int(point_to[1]))
                # cv2.line(self.img, position_from, position_to, (0, 255, 255), 1)
            except Exception as e:
                print(e)


    def visualise_projection_errors(self, projected_points):
        temp = projected_points.reshape((2, -1)).transpose()
        for idx in range(temp.shape[0]):
            p1 = (int(temp[idx, 0]), int(temp[idx, 1]))
            p2 = (int(self.image_points[idx, 0]), int(self.image_points[idx, 1]))
            cv2.line(self.img, p1, p2, (0, 0, 255), 1)


    def visualise_pose_estimate(self):
        print("visualise_pose_estimate")
        self.img = self.img_orig.copy()
        projected_points = self.fProject(self.pose, self.wireframe_model.points)
        #ic(self.pose)
        #ic(self.corners)
        #ic(projected_points)
        self.visualise_projected_points(projected_points)
        self.visualize_image_points()
        self.visualise_projection_errors(projected_points)
        self.visualise_wireframe_model()
        cv2.imshow("test", self.img)


    def prepare_optimization(self):
        def projection_function(pose):
            measurement_uncertainty = 1.5
            projections = self.fProject(pose, self.wireframe_model.points)
            ic(self.wireframe_model.points.shape)
            ic(projections.shape)
            observations = self.image_points.transpose().reshape((1, -1))
            return (observations - projections) / measurement_uncertainty
        
        self.lm = LevenbergMarquard(projection_function, 
                               self.pose)


    def K_from_cam_param(self, pose_and_camera_settings): 
        return np.array([
                [pose_and_camera_settings[6], 0, pose_and_camera_settings[7]], 
                [0, pose_and_camera_settings[6], pose_and_camera_settings[8]], 
                [0, 0, 1]
                ])


    def prepare_optimization_with_focal_length(self, cam_param):
        def projection_function(pose_and_camera_settings):
            measurement_uncertainty = 1.5
            pose = pose_and_camera_settings[0:6]
            self.K = self.K_from_cam_param(pose_and_camera_settings)
            projections = self.fProject(pose, self.wireframe_model.points)
            observations = self.image_points.transpose().reshape((1, -1))
            return (observations - projections) / measurement_uncertainty
        
        self.lm = LevenbergMarquard(projection_function, 
                               np.concatenate((self.pose, cam_param)))
        self.lm.parameters_to_optimize = [True, True, True, True, True, True, True, True, True]
        ic(self.lm.param)




    def update_pose_estimate(self, iteration = 1):
        self.lm.iterate()
        self.pose = self.lm.param


    def update_pose_estimate_with_focal_length(self, iteration = 1):
        self.lm.iterate()
        self.pose = self.lm.param[0:6]
        ic(self.lm.param)
        self.K = self.K_from_cam_param(self.lm.param)
        ic(self.K)



    def estimate_pose_and_visualize(self, K, image_points, corners, pose, image):
        self.image_points = image_points
        self.corners = corners
        self.K = K
        self.img_orig = image
        self.pose = pose

        for iteration in range(2000):
            self.update_pose_estimate(iteration)
            print("Residual error: ", self.residual_error)
            print(self.pose)
            self.visualise_pose_estimate()
            # cv2.imshow("test", self.img)
            #key = cv2.waitKey(-1)
            #if key is ord('q'):
            #    break
        return 


    def estimate_pose(self, K, image_points, corners, pose, iterations):
        self.image_points = image_points
        self.corners = corners
        self.K = K
        self.pose = pose

        for iteration in range(iterations):
            self.update_pose_estimate(iteration)
        return 



class LevenbergMarquard():
    def __init__(self, 
                 function: Callable[[np.ndarray], np.ndarray], 
                 initial_param: np.ndarray, 
                 damping: float = 100
                 ) -> None:
        self.parameters_to_optimize = initial_param == initial_param
        self.damping = damping
        self.func = function
        self.param = initial_param
        self.residual_error = np.linalg.norm(self.func(self.param))


    def jacobian(self, 
                 param: np.ndarray, 
                 func: Callable[[np.ndarray], np.ndarray]
                 ) -> Tuple[np.ndarray, np.ndarray]: 
        e = 0.00001
        delta = np.zeros(param.shape)

        # Calculate the function values at the given pose
        projected_points = func(param)

        # Calculate jacobian by perturbing the pose prior 
        # to calculating the function values.
        j = np.zeros((projected_points.shape[1], param.shape[0]))
        for k in range(param.shape[0]):
            delta_k = delta.copy()
            delta_k[k] = e
            param_temp = param + delta_k.transpose()
            func_value = func(param_temp)
            j[:, k] = (func_value - projected_points) / e

        # Limit the jacobian to the parameters that should be optimized.
        j = j[:, self.parameters_to_optimize]

        return (projected_points, j)


    def iterate(self) -> None:
        # Get projection errors and the associated jacobian
        self.projection_errors, j = self.jacobian(self.param, self.func)

        # Levenberg Marquard update rule
        self.coefficient_covariance_matrix = j.transpose() @ j
        t2 = np.diag(np.diag(self.coefficient_covariance_matrix)) * self.damping
        t3 = self.coefficient_covariance_matrix + t2
        param_update = np.linalg.inv(t3) @ j.transpose() @ self.projection_errors.transpose()

        # Unpack to full solution
        dx = np.zeros((self.param.shape[0], 1))
        dx[self.parameters_to_optimize] = param_update
        updated_x = self.param - dx.reshape((-1))
        updated_residual_error = np.linalg.norm(self.func(updated_x))

        if self.residual_error < updated_residual_error:
            # Squared error was increased, reject update and increase damping
            self.damping = self.damping * 10
        else:
            # Squared error was reduced, accept update and decrease damping
            self.param = updated_x
            self.damping = self.damping / 3
            self.residual_error = updated_residual_error

        return
    
    def estimate_uncertainties(self, p = 0.99):
        self.squared_residual_error = self.residual_error**2

        # Determine how many standard deviations we should go out
        # to cover a given probability (p).
        # TODO: I am unsure if it should be split into these two cases (one_dim vs multi_dim)
        self.scale_one_dim = chi2.ppf(p, 1)
        self.scale_multi_dim = chi2.ppf(p, self.param.size)

        # Equation 15.4.15 from Numerical Recipes in C 2002
        self.param_uncert = self.scale_one_dim * 1 / np.sqrt(np.diag(self.coefficient_covariance_matrix))

        # Equation on page 660 in Numerical Recipes in C 2002
        self.goodness_of_fit = 1 - chi2.cdf(self.residual_error**2, self.projection_errors.size)
    
        # Build matrix with uncertainties for independent parameters
        # Equation 15.4.15 from Numerical Recipes in C 2002
        delta = np.zeros(self.param.shape)
        self.independent_uncertainties = np.zeros((self.param.size, self.param.size))
        for k in range(self.param.size):
            delta_k = delta.copy()
            delta_k[k] = 1
            vector = self.param_uncert * delta_k
            self.independent_uncertainties[k, :] = vector
        #ic(self.independent_uncertainties)

        # Build matrix with uncertainties for combined parameters
        # Based on equation 15.4.18 in Numericap Recipes in C 2002
        u, s, vh = np.linalg.svd(np.linalg.inv(self.coefficient_covariance_matrix))
        self.combination_uncert = self.scale_multi_dim * np.sqrt(s)
        self.combined_uncertainties = np.zeros((self.param.size, self.param.size))
        for k in range(self.param.size):
            vector = self.scale_multi_dim * vh[k] * np.sqrt(s[k])
            self.combined_uncertainties[k, :] = vector
        #ic(self.combined_uncertainties)

                   


def main():
        # Example with all front key points
        # Calibration matrix

        image_points = np.array([
            [159, 86], 
            [180, 209], 
            [211, 227], 
            [254, 208], 
            [289, 226], 
            [329, 207], 
            [323, 87], 
            [245, 60]])
        

        image_points = np.array([
            [1355,950],
            [1216,920],
            [1359,879],
            [1386,872],
            [1219,854],
            [1388,806],
            [511,772],
            [417,753],
            [1435,742],
            [1316,724],
            [501,708],
            [1199,705],
            [407,691],
            [1090,686],
            [1439,678],
            [642,670],
            [985,669],
            [1317,661],
            [572,657],
            [887,652],
            [1199,643],
            [492,643],
            [792,637],
            [396,627],
            [1089,625],
            [701,622],
            [1444,612],
            [984,610],
            [635,610],
            [1320,595],
            [884,593],
            [563,593],
            [788,580],
            [481,576],
            [696,565],
            [384,560],
            [629,551],
            [1448,543],
            [556,529],
            [1322,527],
            [690,509],
            [471,507],
            [372,491],
            [621,487],
            [1453,472],
            [547,464],
            [1326,458],
            [685,448],
            [1203,445],
            [1085,434],
            [461,433],
            [616,425],
            [977,421],
            [360,420],
            [873,411],
            [774,400],
            [1459,399],
            [538,394],
            [676,389],
            [1329,386],
            [1204,376],
            [1087,364],
            [449,360],
            [607,359],
            [975,354],
            [347,346],
            [869,345],
            [769,336],
            [669,326],
            [528,323],
            [1465,321],
            [1333,311],
            [1206,303],
            [1086,294],
            [599,291],
            [973,286],
            [438,281],
            [865,277],
            [763,269],
            [335,269],
            [663,261],
            [520,247],
            [1470,242],
            [1336,233],
            [1207,227],
            [1086,222],
            [592,219],
            [970,214],
            [860,206],
            [428,203],
            [757,199],
            [656,194],
            [322,191],
            [511,176],
            [1476,163],
            [1340,158],
            [1086,148],
            [856,136],
            [650,125],
        ])        

        img = cv2.imread("85_estimate_camera_pose/input/image_0200.jpg")
        # Calibration matrix
        # K = np.array([[890, 0, 395], [0, 890, 315], [0, 0, 1]])
        K = np.array([[1807, 0, 966], [0, 1807, 503], [0, 0, 1]])

        PoseE = PoseEstimatorOrig()
        PoseE.img_orig = img
        PoseE.K = K
        PoseE.wireframe_model = WireframeModel()
        PoseE.image_points = image_points

        PoseE.pose = np.array([0.2, 2, 0, 12, 23, -6])

        PoseE.pose = np.array([1.7323787 , -0.16753503, -0.18408168,  
                               8.69359019, -5.21568803, 26.97788855])
        PoseE.prepare_optimization()

        for _ in range(20):
            PoseE.update_pose_estimate(iteration=1)
            PoseE.visualise_pose_estimate()
            k = cv2.waitKey(10000)
            ic(PoseE.pose)
            ic(PoseE.lm.residual_error)
            if k == ord('q'):
                break
        ic(np.argmax(np.abs(PoseE.lm.projection_errors)))
        ic(np.argmax(np.abs(PoseE.lm.projection_errors)) - image_points.shape[0])
        PoseE.lm.estimate_uncertainties(p = 0.99)
        ic(PoseE.lm.squared_residual_error)
        ic(PoseE.lm.goodness_of_fit)
        ic(PoseE.lm.param)
        ic(PoseE.lm.param_uncert)


        ic("Can now also update focal length and optical center")
        #PoseE.pose = np.array([1.7323787 , -0.16753503, -0.18408168,  
        #                       8.69359019, -5.21568803, 26.97788855])
        cam_param = np.array([500, 100, 100])

        PoseE.prepare_optimization_with_focal_length(cam_param=cam_param)
        for _ in range(20):
            PoseE.update_pose_estimate_with_focal_length(iteration=1)
            PoseE.visualise_pose_estimate()
            k = cv2.waitKey(10000)
            ic(PoseE.pose)
            ic(PoseE.lm.residual_error)
            if k == ord('q'):
                break
        PoseE.lm.estimate_uncertainties(p = 0.99)
        ic(PoseE.lm.squared_residual_error)
        ic(PoseE.lm.goodness_of_fit)
        ic(PoseE.lm.param)
        ic(PoseE.lm.param_uncert)


"""
Calibration matrix from opencv camera calibration tool.
Calibration matrix: 
[1802.78328318    0.          980.71844996]
[   0.         1800.60466427  542.82721191]
[0. 0. 1.]
Obtained results
[1807.83     0.00   966.59]
[   0.00  1807.83   503.03]
[   0.00     0.00     1.00]
"""


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:7.4f}".format(x)})
    #np.set_printoptions(formatter={'float': lambda x: "{0:5.2f}".format(x)})
    main()