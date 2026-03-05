import cv2

class ProjectionVisualizer():
    def __init__(self, img, img_name):
        self.img_orig = img
        self.window_name = img_name
                    

    def visualize_image_points(self, image_points):
        prev_position = None
        for point in image_points:
            position = (int(point[0]), int(point[1]))
            if prev_position is not None:
                pass # cv2.line(self.img, position, prev_position, (255, 0, 0), 1)
            prev_position = position


    def visualise_projected_points(self, projected_points):
        for point in projected_points.reshape((2, -1)).transpose():
            position = (int(point[0]), int(point[1]))
            cv2.circle(self.img, position, 3, (0, 0, 255), -1)


    def visualise_projection_errors(self, projected_points, image_points):
        temp = projected_points.reshape((2, -1)).transpose()
        for idx in range(temp.shape[0]):
            p1 = (int(temp[idx, 0]), int(temp[idx, 1]))
            p2 = (int(image_points[idx, 0]), int(image_points[idx, 1]))
            cv2.line(self.img, p1, p2, (0, 0, 255), 1)


    def visualise_pose_estimate(self, projected_points, image_points):
        self.img = self.img_orig.copy()
        self.visualise_projected_points(projected_points)
        #self.visualize_image_points(image_points)
        self.visualise_projection_errors(projected_points, image_points)
        cv2.imshow(self.window_name, self.img)
