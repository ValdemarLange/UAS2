import numpy as np
from icecream import ic


def decompose_essential_matrix(E):
    # Algorithm from wikipedia
    # https://en.wikipedia.org/wiki/Essential_matrix#Extracting_rotation_and_translation
    D = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    u, s, vh = np.linalg.svd(essential_matrix, full_matrices=True)

    RE1 = u @ D @ vh
    RE2 = u @ D.T @ vh

    tE1 = u[:, 2].reshape((3, 1))
    tE2 = -u[:, 2].reshape((3, 1))

    return (RE1, RE2, tE1, tE2)


essential_matrix = np.array(
    [
        [-0.00300216, -0.43213014, -0.14442965],
        [0.33859683, 0.01502989, -0.60288981],
        [0.11929845, 0.54699833, -0.0246066],
    ]
)

print("Input essential matrix")
ic(essential_matrix)

RE1, RE2, tE1, tE2 = decompose_essential_matrix(essential_matrix)

print("Rotation solution 1")
ic(RE1)
print("Rotation solution 2")
ic(RE2)

print("Translation solution 1")
ic(tE1)
print("Translation solution 2")
ic(tE2)

# Reference value obtained from opencv recover pose method.
# To select the proper combination of rotation and translation
# additional information was used in form of image coordinates of
# common points in both images.
R = np.array(
    [
        [0.98594991, 0.04674898, -0.16036614],
        [-0.04766509, 0.99886163, -0.00186845],
        [0.16009624, 0.00948606, 0.98705583],
    ]
)

t = np.array([[-0.76471512], [0.20808734], [-0.60984461]])

print("Target rotation matrix")
ic(R)

print("Target displacement")
ic(t)

print("Reconstruct essential matrix from estimated rotation and displacement")
t = tE2
R = RE1
tcross = np.array(
    [[0, -t[2][0], t[1][0]], [t[2][0], 0, -t[0][0]], [-t[1][0], t[0][0], 0]]
)
ic(tcross)

reconstructed_e_matrix = tcross @ R
# Scale the reconstructed essential matrix, to match the
# essential matrix provided in the exercise.
reconstructed_e_matrix = (
    reconstructed_e_matrix / reconstructed_e_matrix[0][0] * essential_matrix[0][0]
)
ic(reconstructed_e_matrix)
ic(essential_matrix)
ic(reconstructed_e_matrix - essential_matrix)
