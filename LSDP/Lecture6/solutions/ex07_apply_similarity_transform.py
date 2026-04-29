import numpy as np

# Definition of the transformation matrix
theta = np.pi / 6
transformation_matrix = np.array(
    [[np.cos(theta), np.sin(theta), -50], [-np.sin(theta), np.cos(theta), 65]]
)
print("Transformation matrix")
print(transformation_matrix)
print()

# Input the point to transform
input_point = np.array([[123], [78]])

# Express the input point in homogeneous coordinates
input_point_hc = np.vstack((input_point, 1))
print("input point in homogeneous coordinates")
print(input_point_hc)
print()

# Transform the point
output_point = transformation_matrix @ input_point_hc
print("output point")
print(output_point)
print()
