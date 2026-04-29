import numpy as np

def decompose_essential_matrix(E):
    svd = np.linalg.svd(E)
    D = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    R1 = svd.U @ D @ svd.Vh # Vh is already transposed
    R2 = svd.U @ D.transpose() @ svd.Vh
    t1 = svd.U @ np.array([[0], [0], [1]])
    t2 = svd.U @ np.array([[0], [0], [-1]])

    return (R1, R2, t1, t2)


essential_matrix = np.array(
    [
        [-0.00300216, -0.43213014, -0.14442965],
        [0.33859683, 0.01502989, -0.60288981],
        [0.11929845, 0.54699833, -0.0246066],
    ]
)

(R1, R2, t1, t2) = decompose_essential_matrix(essential_matrix)

print("Rotation solution 1")
print(R1)
print("Rotation solution 2")
print(R2)

print("Translation solution 1")
print(t1)
print("Translation solution 2")
print(t2)

