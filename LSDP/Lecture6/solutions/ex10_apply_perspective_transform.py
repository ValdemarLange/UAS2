import numpy as np

# Definition of the transformation matrix
transformation_matrix = np.array(
    [[0.588, -0.607, 172], [0.0532, 0.0772, 122], [0.000166, -0.00169, 1]]
)

# Input the point to transform
input_point = np.array([[345], [234]])

# Express the input point in homogeneous coordinates
input_point_hc = np.vstack((input_point, 1))
print("Input point in homogeneous coordinates")
print(input_point_hc)
print()

# Transform the point
output_point_hc = transformation_matrix @ input_point_hc
print("Output point in homogeneous coordinates")
print(output_point_hc)
print()

# Convert the output point to normal coordinates
output_point = output_point_hc[0:2] / output_point_hc[2]
print("Output point in normal coordinates")
print(output_point)
print()
