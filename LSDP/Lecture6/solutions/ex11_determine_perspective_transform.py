import numpy as np

# List the matched point in the following format
# x1, y1, x2, y2
point_pairs = np.array(
    [
        [230, 1781, 100, 1600],
        [2967, 1297, 2900, 1600],
        [2941, 607, 2900, 100],
        [203, 59, 100, 100],
    ]
)

# Set up a linear system of equations for the perspective transformation
# x2 * (g*x1 + h*y1 + 1) = a*x1 + b*y1 + c
# y2 * (g*x1 + h*y1 + 1) = d*x1 + e*y1 + f
# Structure the equations so that all the unknowns appear on the right.
# a*x1 + b*y1 + c - g*x1*x2 - h*y1*x2 = x2
# d*x1 + e*y1 + f - g*x1*y2 - h*y1*y2 = y2
# The unknowns are the parameters of the transform [a, b, c, d, e, f, g, h].
# The equations can be written in the form A*x = b, where
# A = [[x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2],
#      [0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2]]
# x = [a, b, c, d, e, f, g, h]^T
# b = [x2, y2]^T


def get_equations_from_point_pair(x1, y1, x2, y2):
    """
    Return equations written in extended matrix form
    for the point pair (x1, y1) and (x2, y2).
    """
    res = np.array(
        [
            [x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2, x2],
            [0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2, y2],
        ]
    )
    return res


# For all point pairs generate two equations and
# collect them to a system of equations that
# specifies the similarity transform.
all_equations = np.array([[]])
for k in range(point_pairs.shape[0]):
    point_pair = point_pairs[k, :]
    # The *variable syntax is similar to doing the following
    # variable[0], variable[1], variable[2], variable[3].
    eqns = get_equations_from_point_pair(*point_pair)

    # Handle the first iteration where the all_equations
    # array is empty.
    try:
        all_equations = np.concatenate((all_equations, eqns))
    except ValueError:
        all_equations = eqns

# Solve the system of equations
coefficient_matrix = all_equations[:, :-1]
other_parts = all_equations[:, -1]
solution = np.linalg.solve(coefficient_matrix, other_parts)
print("Solution to the linear system of equations")
print(solution)
print()

# Format the solution as a transformation matrix
transformation_matrix = np.array(
    [
        [solution[0], solution[1], solution[2]],
        [solution[3], solution[4], solution[5]],
        [solution[6], solution[7], 1],
    ]
)
print("Solution formatted as a transformation matrix")
with np.printoptions(precision=6, floatmode="fixed", suppress=True):
    print(transformation_matrix)
    print()


print("Transformation matrix applied to each of the four given points")
for k in range(point_pairs.shape[0]):
    point = np.hstack((point_pairs[k, 0:2], 1))
    output_point_hc = transformation_matrix @ point
    output_point = output_point_hc[0:2] / output_point_hc[2]
    print(point, " -> ", output_point)
print()
