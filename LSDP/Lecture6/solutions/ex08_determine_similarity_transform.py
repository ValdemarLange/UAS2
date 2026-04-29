import numpy as np

# List the matched point in the following format
# x1, y1, x2, y2
point_pairs = np.array([[230, 1781, 100, 1600], [2967, 1297, 2900, 1600]])

# Set up a linear system of equations for the similarity transformation
# x2 = a*x1 + b*y1 + c
# y2 = -b*x1 + a*y1 + d
# The unknowns are the parameters of the transform [a, b, c, d].
# The equations can be written in the form A*x = b, where
# A = [[x1, y1, 1, 0],
#      [y1, -x1, 0, 1]]
# x = [a, b, c, d]^T
# b = [x2, y2]^T


def get_equations_from_point_pair(x1, y1, x2, y2):
    """
    Return equations written in extended matrix form
    for the point pair (x1, y1) and (x2, y2).
    """
    res = np.array([[x1, y1, 1, 0, x2], [y1, -x1, 0, 1, y2]])
    return res


# For all point pairs generate two equations and
# collect them to a system of equations that
# specifies the similarity transform.
all_equations = np.array([[]])
for k in range(point_pairs.shape[0]):
    point_pair = point_pairs[k, :]
    # The *variable syntaks is similar to doing the following
    # variable[0], variable[1], variable[2], variable[3].
    eqns = get_equations_from_point_pair(*point_pair)

    # Hand the first iteration where the all_equations
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
    [[solution[0], solution[1], solution[2]], [-solution[1], solution[0], solution[3]]]
)
print("Solution formatted as a transformation matrix")
with np.printoptions(precision=3, floatmode="fixed", suppress=True):
    print(transformation_matrix)
    print()

# Test the solution by applying it to the two known input points
print("Apply the transformation to the first point")
point1 = np.array([[230], [1781], [1]])
print(transformation_matrix @ point1)
print()

print("Apply the transformation to the second point")
point2 = np.array([[2967], [1297], [1]])
print(transformation_matrix @ point2)
print()
